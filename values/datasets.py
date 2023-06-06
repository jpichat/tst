import copy
import pathlib
import typing
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter, iconcat
from itertools import groupby, zip_longest
import functools

from values.base import Identifier


class Dataset:
    def __init__(
        self,
        id: Identifier,
        path: typing.Union[str, pathlib.Path],
        include=None,
        exclude=None,
        targets: typing.List[str] = None,
        na_values: dict = None,
        types: dict = None,
        datetime_options: typing.Optional[DatetimeOptions] = None,
        formulas: typing.Optional[typing.Mapping[str, str]] = None,
        validated: typing.Optional[Validated] = None,
        inclusive_modeling: bool = False,
        project_id=None,
    ):
        """
        Create a new dataset entity.

        Args:
            id: An :py:class:`~Identifier`.
            path: Path to the data on disk.
            include: A list of field names to include, ``None`` means include all fields.
            exclude: A list of field names to exclude, ``None`` means no fields are excluded.
            targets: A list of target fields, must be a subset of ``include``.
            na_values: A dictionary mapping a field name to a list of missing values.
            types: A dictionary mapping a field name to a type.
            datetime_options: A :py:class:`~mlcore.timeseries.values.options.DatetimeOptions` object or None.
            formulas: A dictionary mapping a new field name to a formula expression string, or None.
            validated: A bool, indicating if the dataset has been validated or not.
            inclusive_modeling: A bool, indicating whether targets are also used as exog during modeling.
            project_id: Unused.
        """
        self.id = Identifier(id)
        self.project_id = project_id
        self.path = pathlib.Path(path)
        self.include = include or None
        self.exclude = exclude or None
        self.targets = targets
        self.na_values = na_values
        self.types = types
        self.datetime_options = datetime_options
        self.formulas = formulas
        self.validated = validated or Validated()
        self.inclusive_modeling = inclusive_modeling
        assert not set(self.include or []) & set(self.exclude or [])
        assert not set(self.targets or []) & set(self.exclude or [])
        if self.targets and self.include:
            assert set(self.targets).issubset(set(self.include))
        if self.formulas and self.include:
            assert set(self.formulas).isdisjoint(set(self.include))
        if self.formulas and self.exclude:
            assert set(self.formulas).isdisjoint(set(self.exclude))
        self._len = None

    def link(self, project):
        self.project_id = project.id
        return self

    def updated(self, **kwargs) -> "Dataset":
        """
        Modify one or more attributes of a dataset.
        Only pass keyword arguments for attributes that should be changed.

        Args:
            id: Provide new id to create a new Dataset, otherwise the change is "in-place"
            path: Provide a new path to create a new derived dataset, make sure to provide a new id if you provide a new path!
            include: Optional keyword argument
            exclude: Optional keyword argument
            targets: Optional keyword argument
            na_values: Optional keyword argument
            types: Optional keyword argument
            datetime_options: Optional keyword argument
            formulas: Optional keyword argument
            inclusive_modeling: Optional keyword argument

        Returns:
            An updated Dataset object.
        """
        if "validated" in kwargs:
            raise KeyError(
                "Dataset.validated is updated automatically. For manual updates, mutate Validated in-place."
            )

        new_include = kwargs.get("include", self.include)
        new_exclude = kwargs.get("exclude", self.exclude)
        if new_include != self.include and new_exclude is not None:
            new_exclude = sorted(set(new_exclude) - set(new_include or []))
        elif new_exclude != self.exclude and new_include is not None:
            new_include = sorted(set(new_include) - set(new_exclude or []))

        new_targets = kwargs.get("targets", self.targets)
        if new_targets is not None and new_include is not None:
            new_include = sorted(set(new_include) | set(new_targets))

        new_formulas = kwargs.get("formulas", self.formulas)
        included_formulas = set(new_formulas or []) & set(new_include or [])
        if included_formulas:
            new_include = sorted(set(new_include) - included_formulas)
        excluded_formulas = set(new_formulas or []) & set(new_exclude or [])
        if excluded_formulas:
            new_exclude = sorted(set(new_exclude) - excluded_formulas)
            new_formulas = {
                field: formula
                for field, formula in new_formulas.items()
                if field not in excluded_formulas
            }

        # date field
        new_datetime_options = kwargs.get("datetime_options", self.datetime_options)
        if new_datetime_options != self.datetime_options:
            self.validated.date_field = False
        # target fields
        targets_diff = set(new_targets or []) - set(self.targets or [])
        if targets_diff:
            self.validated.target_fields = False
        # formulas
        exclude_diff = set(new_exclude or []) - set(self.exclude or [])
        if targets_diff or exclude_diff or dictdiff(self.formulas, new_formulas):
            self.validated.formulas = False

        updated = Dataset(
            id=kwargs.get("id", self.id),
            path=kwargs.get("path", self.path),
            include=new_include,
            exclude=new_exclude,
            targets=new_targets,
            na_values=kwargs.get("na_values", self.na_values),
            types=kwargs.get("types", self.types),
            datetime_options=new_datetime_options,
            formulas=new_formulas,
            validated=self.validated,
            inclusive_modeling=kwargs.get("inclusive_modeling", self.inclusive_modeling),
            project_id=self.project_id,
        )

        return updated

    def datefield_excluded(self) -> "Dataset":
        """Return a new dataset with the datetime field excluded"""
        if self.datetime_options is None or self.datetime_options.field is None:
            return self
        new_exclude = self.exclude or []
        if self.datetime_options.field not in new_exclude:
            new_exclude = new_exclude + [self.datetime_options.field]
        return self.updated(exclude=new_exclude)

    def datefield_included(self) -> "Dataset":
        """Return a new dataset with the datetime field included"""
        if self.datetime_options is None or self.datetime_options.field is None:
            return self
        if self.datetime_options.field in (self.exclude or []):
            new_exclude = [field for field in self.exclude if field != self.datetime_options.field]
        else:
            new_exclude = self.exclude
        if self.include is not None and self.datetime_options.field not in self.include:
            new_include = self.include + [self.datetime_options.field]
        else:
            new_include = self.include
        return self.updated(include=new_include, exclude=new_exclude)

    @property
    def target(self):
        if not self.targets or len(self.targets) != 1:
            raise ValueError("Expected exactly 1 target, not {}".format(len(self.targets or [])))
        return self.targets[0]

    @property
    def exogenous(self) -> typing.List[str]:
        """
        Return a list of the exogenous column names.

        If there are no exogenous columns then an empty list is returned.

        .. note:: Formula fields are not considered exogenous here!
           (formulas either derive from an endogenous field, or derived from another exogenous field (which then cannot be ignored))
        """
        all_fields = self.datefield_excluded().to_mlcore().inputs
        return sorted(set(all_fields) - set(self.targets or []))

    @property
    def numeric_fields(self):
        _exclude = set(self.exclude or [])
        return [
            field
            for field in (self.types or {})
            if self.types[field] == FieldTypes.numeric
            and (not self.datetime_options or self.datetime_options.field != field)
            and field not in _exclude
        ]

    @property
    def category_fields(self):
        _exclude = set(self.exclude or [])
        return [
            field
            for field in (self.types or {})
            if self.types[field] == FieldTypes.category
            and (not self.datetime_options or self.datetime_options.field != field)
            and field not in _exclude
        ]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(id={self.id},"
            f" path={self.path},"
            f" include={self.include},"
            f" exclude={self.exclude},"
            f" targets={self.targets},"
            f" na_values={self.na_values},"
            f" types={self.types},"
            f" datetime_options={self.datetime_options},"
            f" formulas={self.formulas},"
            f" validated={self.validated},"
            f" inclusive_modeling={self.inclusive_modeling},"
            f" project_id={self.project_id})"
        )

    def __len__(self):
        if self._len is None:
            self._len = len(self.to_mlcore())
        return self._len

    def to_pandas(self):
        raise NotImplementedError


class DatasetType:
    train = "train"
    validation = "validation"
    test = "test"


class Mask:
    def __init__(
        self,
        id: Identifier,
        fields: typing.List[str],
        rc_ind: typing.List[typing.List[int]],
        shape: typing.List[int],
    ):
        """Sparse mask

        Args
        ----
        id : Identifier
            mask id

        rc_ind : tuple
            (row_ind, col_ind)

            A=[[0, 0, 1, 0, 0],
               [1, 0, 0, 0, 0],
               [1, 0, 0, 1, 0],
               [0, 0, 0, 0, 1]]
            [el.tolist() for el in csr_matrix(A).nonzero()]  # by row
                > [[0, 1, 2, 2, 3], [2, 0, 0, 3, 4]]
            [el.tolist() for el in csr_matrix(A).transpose().nonzero()[::-1]]  # by column
                > [[1, 2, 0, 2, 3], [0, 0, 2, 3, 4]]

        shape : list
            [nrows, ncols] of original data, A (prior to masking)

        fields : list
            list of sorted fields (according to col_ind in rc_ind)
        """
        assert isinstance(fields, list), type(fields)
        self.fields = fields
        self.rc_ind = rc_ind
        self.shape = shape
        self.id = id

    def rind_from_field(self, field: typing.Union[str, int]) -> list:
        """returns row_ind given a field (column name/index)"""
        if isinstance(field, str):
            col_ind = self.fields.index(field)
        else:
            assert np.min(self.rc_ind[1]) <= field <= np.max(self.rc_ind[1])
            col_ind = field
        return [t[0] for t in zip(*self.rc_ind) if t[1] == col_ind]

    def link(self, dataset: Dataset):
        self.id = dataset.id
        return self

    def to_frame(self, dtype=int) -> pd.DataFrame:
        """converts sparse mask to dense dataframe mask"""
        assert len(self.rc_ind[0]) == len(self.rc_ind[1])
        data = np.ones(len(self.rc_ind[0]))
        return pd.DataFrame.sparse.from_spmatrix(
            csr_matrix((data, self.rc_ind), shape=self.shape), columns=self.fields
        ).astype(dtype)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_dict(self, dates: pd.Series = None, date_format: str = None) -> dict:
        """"""
        if dates is None:
            dates = np.arange(self.shape[0])
        else:
            assert len(dates) == self.shape[0], "dimension mismatch!"
            dates = dates.values

        rc_z = list(zip(*self.rc_ind))
        rc_z.sort(key=itemgetter(1))
        mv_dates = [[dates[g[0]] for g in groups] for _, groups in groupby(rc_z, key=itemgetter(1))]

        # putting back empty lists for fields with no missing values (columns)
        for i in set(np.arange(self.shape[1])) - set(self.rc_ind[1]):
            mv_dates.insert(i, [])

        return {
            "fields": self.fields,
            "missingValuesDates": mv_dates,
            "dateFormat": date_format,
        }


class Fill:
    def __init__(
        self,
        id: Identifier,
        fields: typing.List[str],
        fill_with: typing.List[typing.List[typing.Union[str, float]]],
        at_index: typing.List[typing.List[int]],
    ):
        """fill fields with values at given row indices

        Args
        ----
        id : Identifier
            fill id

        fields : List[str]
            list of fields with missing values

        fill_with : List[List[Union[str, float]]]
            values to fill each field with

        at_index : List[List[int]]
            (row) indices of fields' missing values to be filled
        """
        assert fields is not None, "missing fields"
        assert all(isinstance(a, list) for a in (fields, fill_with, at_index))
        assert len(fields) == len(fill_with) == len(at_index)
        self.fields = fields
        self.fill_with = fill_with
        self.at_index = at_index
        self.id = id

    def appended(
        self,
        field: str,
        values: list,
        locs: list,
    ) -> "Fill":
        """append new field with new filling values at new locations"""
        assert isinstance(field, str)
        assert all(isinstance(v, list) for v in (values, locs))
        assert len(values) == len(locs), "dimension mismatch"

        return Fill(
            self.id,
            self.fields + [field],
            self.fill_with + [values],
            self.at_index + [locs],
        )

    def updated(
        self,
        field: typing.Union[str, int] = None,
        new_values: typing.List[typing.Union[str, float]] = None,
    ) -> "Fill":
        """update a field with a new set of filling values, at given locations"""
        assert field is not None, "missing field"
        if new_values is None:
            return Fill(
                self.id,
                self.fields,
                self.fill_with,
                self.at_index,
            )

        if isinstance(field, str):
            i = self.fields.index(field)
        else:
            assert 0 <= field < len(self.fields)
            i = field

        # just in case
        if isinstance(new_values, pd.DataFrame):
            new_values = new_values[field]

        new_fill_with = [el for k, el in enumerate(self.fill_with) if k != i]
        new_fill_with.insert(i, list(new_values))

        return Fill(
            self.id,
            self.fields,
            new_fill_with,
            self.at_index,
        )

    @staticmethod
    def derived_from(
        mask: Mask,
        imputed_data: pd.DataFrame,
        fields: list = None,
    ) -> "Fill":
        """derive Fill object from and imputed data

        Args
        ----
        mask : Mask
            mask of original data with missing values

        imputed_data : pandas Dataframe or array
            imputed data that contain ()/refer to all or part of masked fields and fields (if given)

        fields : List[str]
            if one interested in deriving a `Fill` entity for a subset of fields
        """
        assert isinstance(mask, Mask), type(mask)
        assert isinstance(imputed_data, pd.DataFrame), type(imputed_data)

        if fields is not None:
            assert isinstance(fields, list), type(fields)
            assert all(f in mask.fields for f in fields), "some fields were not found in mask!"
        else:
            assert set(mask.fields) == set(
                imputed_data.columns
            ), "some fields were not found in mask!"

        fields_ = fields or mask.fields
        fill_with = []
        at_index = []

        for f in fields_:
            rind = mask.rind_from_field(f)
            at_index.append(rind)
            fill_with.append(np.array(imputed_data[f])[rind].tolist())

        return Fill(
            mask.id,
            fields_,
            fill_with,
            at_index,
        )

    def filter_by_field(self, field: typing.Union[str, int]) -> "Fill":
        """only get information of a single field"""
        if isinstance(field, str):
            assert field in self.fields, f"{field} cannot be found among {self.fields}"
            i = self.fields.index(field)
        else:
            assert 0 <= field < len(self.fields)
            i = field

        return Fill(
            self.id,
            [field],
            [self.fill_with[i]],
            [self.at_index[i]],
        )

    def link(self, id: Identifier):
        """different from other entities' `link` method to avoid loading the dataset in
        `save_imputation` task (since we only need its id and it's given in the request already)
        """
        self.id = id  # could be the id of a mask (iself linked to a dataset), or a Dataset.id
        return self

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def to_dict(self, dates: pd.Series = None, date_format: str = None) -> dict:
        """
        Returns
        -------
        {
            "fields": ["x", "z"],
            "data":[
                   [["1990-01", 0, 42], ["1990-02", 1, 42], ["1990-12", 11, 42]],
                   [["1990-11", 10, 42]],
                ],
            "dateFormat":"%Y-%m"
        }
        or
        {
            "fields": ["x", "z"],
            "data":[
                   [[None, 0, 42], [None, 1, 42], [None, 11, 42]],
                   [[None, 10, 42]],
                ],
            "dateFormat": None
        }
        """
        no_dates = dates is None
        if no_dates:
            dates_ = [[None]] * len(self.at_index)
        else:
            assert len(dates) > max(functools.reduce(iconcat, self.at_index, []))
            dates = dates.to_numpy()
            dates_ = [dates[ind].tolist() for ind in self.at_index]

        data = [
            [list(el) for el in zip_longest(date, ind, val)]
            for date, ind, val in zip(dates_, self.at_index, self.fill_with)
        ]

        return {
            "fields": self.fields,
            "data": data,
            "dateFormat": date_format,
        }
