import time
import math
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter
from typing import Optional, Union
import logging

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process.kernels import (
    Matern,
    Product,
    ExpSineSquared,
    WhiteKernel,
    Sum,
    RationalQuadratic,
    RBF,
    DotProduct,
    ConstantKernel,
)
from scipy.stats.mstats import theilslopes
from scipy.linalg import solve_triangular, cho_solve, cho_factor
from scipy.optimize import fmin_l_bfgs_b

from values.types import FieldTypes
from tst.transform.decompose import SSA
from tst.transform.standardise import Normalize
from tst.transform.ignore import IgnoreField
from tst.utils import max_consecutive_nans, embed, embed_np, diag_avg, windowed_name


class SimpleImputer(TransformerMixin, BaseEstimator):
    """in-place simple missing value imputation of a 1d series"""

    class Strategies:
        MEAN = "mean"  # numeric only
        MEDIAN = "median"  # numeric only
        MOST_FREQUENT = "most_frequent"  # numeric/category

    NA_SUFF = "_NA"

    def __init__(
        self,
        field: str,
        field_type: Optional[FieldTypes] = None,
        mask: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, list]] = None,
        fill_with: Optional[Union[float, str]] = None,
        strategy: Optional[Strategies] = None,
        n_skip: Optional[int] = None,
        return_mask: Optional[bool] = False,
    ):
        """
        Args
        ----
        field : str
            name of field to impute

        field_type : FieldTypes
            numeric/categorical

        mask : pd.Series/pd.DataFrame/np.ndarray/lists
            precomputed mask

        fill_with : float/str
            value to fill missing observations with

        strategy : Strategies
            imputation strategy

        n_skip : int
            number of leading nan rows to skip

        return_mask : bool
            whether mask is added to the output frame
        """

        # sanity checks
        assert isinstance(field, str)
        assert field_type is not None, "missing field type"
        assert FieldTypes.is_valid(field_type), f"unknown field type: {field_type}"

        if fill_with is None:
            assert strategy is not None, "missing strategy"
        if strategy is not None:
            assert (
                fill_with is None
            ), "only one of `strategy` or `fill_with` should be passed"
        if strategy in (self.Strategies.MEDIAN, self.Strategies.MEAN):
            assert (
                field_type == FieldTypes.numeric
            ), "choose `most_frequent` for categorical fields"

        self.field = field
        self.field_type = field_type
        self.n_skip = n_skip
        self.strategy = strategy
        self.fill_with = fill_with
        self.mask = (
            mask if mask is None else self._validate_precomputed_mask(mask, field)
        )
        self.return_mask = return_mask
        self.na_field = self.field + self.NA_SUFF if self.return_mask else []

    @staticmethod
    def _validate_precomputed_mask(mask, field) -> pd.Series:
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, name=field).astype(bool)
            return mask
        elif isinstance(mask, pd.Series):
            assert mask.name == field
            return mask.astype(bool)
        elif isinstance(mask, pd.DataFrame):
            assert field in mask.columns
            return mask[field].astype(bool)
        else:
            raise TypeError("unsupported type for given mask")

    def fit(self, chunk: pd.DataFrame):
        if self.mask is None:
            self.mask = chunk[self.field].isnull()

        if self.fill_with is None:
            if self.strategy == self.Strategies.MEAN:
                self.fill_with = np.nanmean(chunk[self.field])
            elif self.strategy == self.Strategies.MEDIAN:
                self.fill_with = np.nanmedian(chunk[self.field])
            elif self.strategy == self.Strategies.MOST_FREQUENT:
                mc = Counter(chunk[self.field]).most_common(2)
                if not isinstance(mc[0][0], str) and math.isnan(mc[0][0]):
                    self.fill_with = mc[1][0]
                else:
                    self.fill_with = mc[0][0]
            else:
                raise ValueError("unknown strategy")

        return self

    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        try:
            chunk_ = chunk.copy()
            mask_name = self.mask.name

            if self.n_skip is not None:
                self.mask.iloc[: self.n_skip] = False
                chunk_[self.field].where(~self.mask, other=self.fill_with, inplace=True)
            else:
                chunk_[self.field].where(~self.mask, other=self.fill_with, inplace=True)

            if self.return_mask:
                return pd.concat(
                    [chunk_, self.mask.rename(mask_name + self.NA_SUFF)],
                    axis="columns",
                )
            else:
                return chunk_

        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    def serialize(self):
        return dict(
            field=self.field,
            field_type=self.field_type,
            n_skip=self.n_skip,
            fill_with=self.fill_with,
            strategy=self.strategy,
            mask=self.mask,
            return_mask=self.return_mask,
        )

    @classmethod
    def deserialize(cls, data):
        return cls(
            data["field"],
            field_type=data["field_type"],
            n_skip=data["n_skip"],
            fill_with=data["fill_with"],
            strategy=data["strategy"],
            mask=data["mask"],
            return_mask=data["return_mask"],
        )


class RollingWindowImputer(TransformerMixin, BaseEstimator):
    """fwd/bwd imputation of a single field using k previous observations"""

    class Strategies:
        MEAN = "mean"  # numeric only
        MEDIAN = "median"  # numeric only
        TREND = "trend"  # numeric only
        MOST_FREQUENT = "most_frequent"  # numeric/category

    NA_SUFF = "_NA"

    def __init__(
        self,
        field: str,
        field_type: Optional[FieldTypes] = None,
        mask: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, list]] = None,
        window_size: Optional[Union[int]] = None,
        strategy: Optional[Strategies] = None,
        n_skip: Optional[int] = None,
        backward: Optional[bool] = False,
        return_mask: Optional[bool] = False,
    ):
        """
        Args
        ----
        field : str
            name of field to impute

        field_type : FieldTypes
            numeric/categorical

        mask : pd.Series/pd.DataFrame/np.ndarray/lists
            precomputed mask

        window_size : int
            number of previous values to consider for the chosen strategy

        strategy : Strategies
            imputation strategy

        n_skip : int
            number of leading nan rows to skip and leave unchanged

        backward : bool
            whether filling is done backwards

        return_mask : bool
            whether mask is added to the output frame
        """
        # sanity checks
        assert isinstance(field, str)
        assert field_type is not None, "missing field type"
        assert FieldTypes.is_valid(field_type), f"unknown field type: {field_type}"
        if strategy in (
            self.Strategies.MEDIAN,
            self.Strategies.MEAN,
            self.Strategies.TREND,
        ):
            assert (
                field_type == FieldTypes.numeric
            ), "choose `most_frequent` for categories"

        self.field = field
        self.field_type = field_type
        self.n_skip = n_skip
        self.strategy = strategy
        self.window_size = (
            window_size or 1
        )  # default-fills with directly preceding value
        self.backward = backward
        self.mask = (
            mask if mask is None else self._validate_precomputed_mask(mask, field)
        )
        self.return_mask = return_mask
        self.na_field = self.field + self.NA_SUFF if self.return_mask else []

    @staticmethod
    def _validate_precomputed_mask(mask, field) -> pd.Series:
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, name=field).astype(bool)
            return mask
        elif isinstance(mask, pd.Series):
            assert mask.name == field
            return mask.astype(bool)
        elif isinstance(mask, pd.DataFrame):
            assert field in mask.columns
            return mask[field].astype(bool)
        else:
            raise TypeError("unsupported type for given mask")

    def fit(self, chunk: pd.DataFrame):
        chunk_ = chunk.copy()
        if self.backward:
            chunk_ = reverse(chunk_, self.field)

        if self.mask is None or self.backward:
            # recompute mask if backward (in case it was given fwd)
            self.mask = chunk_[self.field].isnull()

        # precompute locations of portions to fill
        self.nans_loc = np.argwhere(self.mask.values).flatten()
        if self.n_skip is not None:
            if self.backward:
                self.nans_loc = self.nans_loc[self.nans_loc < len(chunk_) - self.n_skip]
            else:
                self.nans_loc = self.nans_loc[self.nans_loc > self.n_skip]

        return self

    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        try:
            chunk_ = chunk.copy()
            if self.backward:
                chunk_ = reverse(chunk_, self.field)

            filled_ = self._fill(
                chunk_[self.field].values,
                window_size=self.window_size,
                nans_loc=self.nans_loc,
                strategy=self.strategy,
                n_skip=0 if self.backward else self.n_skip,
                is_category=self.field_type == FieldTypes.category,
            )
            chunk_[self.field] = filled_

            # unreverse if bwd
            chunk_ = (
                reverse(chunk_[chunk.columns], self.field)
                if self.backward
                else chunk_[chunk.columns]
            )

            # return mask
            if self.return_mask:
                return pd.concat(
                    [
                        chunk_,
                        self.mask.rename(
                            columns={c: c + self.NA_SUFF for c in self.mask.columns}
                        ),
                    ],
                    axis="columns",
                )
            else:
                return chunk_

        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    @classmethod
    def _fill(
        cls,
        x: np.ndarray,
        window_size: int = None,
        nans_loc: np.ndarray = None,
        strategy: Strategies = None,
        n_skip: int = None,
        is_category: bool = False,
    ) -> np.ndarray:
        def __fill_with(window: np.ndarray):
            if strategy == cls.Strategies.MEAN:
                return np.mean(window)
            elif strategy == cls.Strategies.MEDIAN:
                return np.median(window)
            elif strategy == cls.Strategies.TREND:
                a, b = theilslopes(window)[:2]
                return a * len(window) + b
            elif strategy == cls.Strategies.MOST_FREQUENT:
                return Counter(window).most_common(1)[0][0]
            else:
                raise ValueError("unknown extrapolation")

        assert window_size is not None, "missing window size"
        assert nans_loc is not None, "missing nan indices"
        assert strategy is not None, "missing strategy"

        n_skip = n_skip or 0
        # use first non-null category (resp. 0) for leading mv of categorical (resp. numeric) series
        first_valid = x[(~pd.isnull(x)).argmax(0)] if is_category else 0

        for r in nans_loc:
            if r - (n_skip + window_size) < 0:
                win = x[n_skip:r]
                win = win[~pd.isnull(win)]
                if len(win) == 0:
                    x[r] = first_valid
                elif len(win) == 1:
                    x[r] = x[r - 1]
                else:
                    x[r] = __fill_with(win)
            else:
                win = x[r - window_size : r]
                x[r] = __fill_with(win)

        return x

    def serialize(self):
        return dict(
            field=self.field,
            field_type=self.field_type,
            window_size=self.window_size,
            n_skip=self.n_skip,
            strategy=self.strategy,
            backward=self.backward,
            mask=self.mask,
            return_mask=self.return_mask,
        )

    @classmethod
    def deserialize(cls, data):
        return cls(
            data["fields"],
            field_type=data["field_type"],
            window_size=data["window_size"],
            n_skip=data["n_skip"],
            strategy=data["strategy"],
            backward=data["backward"],
            mask=data["mask"],
            return_mask=data["return_mask"],
        )


class SSAImputer(TransformerMixin, BaseEstimator):
    """in place SSA-based imputation of a single windowed time series (in Hankel form)

    NB: assumes that dimensions have been kept during windowing

    References
    -----------
    .. [1] (Kondrashov and Ghil, 2006) Spatio-temporal filling of missing points in geophysical data sets
    https://hal.archives-ouvertes.fr/file/index/docid/331089/filename/npg-13-151-2006.pdf
    """

    def __init__(
        self,
        field: str,
        embedding_dimension: int = None,
        n_components: int = 3,
        n_ev: int = None,
        var_threshold=None,
        grouping_method=None,
        use_k_components: int = None,
        mask: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, list]] = None,
    ):
        """
        Args
        ----
        field : str
            field to impute

        embedding_dimension : int
            window size for SSA

        n_components : int (optional, default: 3)
            number of (grouped if n_components<n_ev) components to decompose input series into.

        n_ev : int (optional, default: None)
            number of leading eigenvectors to use for projections (must be >=n_components if given)

        var_threshold : None, "auto", or float (percentage) (optional, default: None)
            coarseness of decomposition (to be set when one has little information
            about linear subspace dimension i.e. with n_ev = None, var_threshold=None,
            it is set to 99% by default)

        grouping_method : str (optional, default: "wcorr")
            method used to group components (if n_components<n_ev): "wcorr"/"eigen/None"

        use_k_components : int (optional, default: 3)
            number of reconstructed components (grouped if n_components<n_ev) to use
            for imputation (must be <=n_components if given)

        mask : pd.Series/pd.DataFrame/np.ndarray/lists
            precomputed mask
        """
        # sanity checks
        assert isinstance(field, str)
        assert embedding_dimension is not None, "missing embedding dimension"
        assert use_k_components <= n_components, (
            f"the number of components to use for imputation ({use_k_components}) should be less "
            f"than that from ssa decomposition ({n_components})"
        )

        self.field = field
        self.embedding_dimension = embedding_dimension
        self.n_components = n_components
        self.n_ev = n_ev
        self.var_thresh = var_threshold
        if use_k_components is None:
            use_k_components = n_components
        self.grouping_method = grouping_method
        self.use_k_components = use_k_components
        self.mask = (
            mask if mask is None else self._validate_precomputed_mask(mask, field)
        )
        self.mask_field = None
        self._transforms = []
        self._result = None

    @staticmethod
    def _validate_precomputed_mask(mask, field) -> pd.Series:
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, name=field).astype(bool)
            return mask
        elif isinstance(mask, pd.Series):
            assert mask.name == field
            return mask.astype(bool)
        elif isinstance(mask, pd.DataFrame):
            assert field in mask.columns
            return mask[field].astype(bool)
        else:
            raise TypeError("unsupported type for given mask")

    def fit(self, df: pd.DataFrame, max_iter: int = 1):
        df_ = df.copy()
        self.max_iter = max_iter
        return_mask = True

        logging.info(
            f">>> [SSA imputer] requested to use {self.use_k_components}/{self.n_components} SSA components"
        )

        start = time.time()

        for k in range(self.use_k_components):
            i = 0

            while i < self.max_iter:
                # normalize
                norm_i = Normalize(self.field, denormalize=True)
                norm_i.fit(df_)

                # decompose
                ssa_i = SSA(
                    self.field,
                    embedding_dimension=self.embedding_dimension,
                    n_components=self.n_components,
                    n_eigenvectors=self.n_ev,
                    var_threshold=self.var_thresh,
                    grouping_method=self.grouping_method,
                    force_isolate_trend=True,
                )
                ssa_i.fit(df_)

                # impute
                if k > 0 or i > 0:
                    self.mask_field = imputer_i.mask_field
                    return_mask = False

                imputer_i = ImputeIteration(
                    self.field,
                    ssa_i.component_fields(k=list(range(1, k + 2))),
                    mask=self.mask,
                    mask_field=self.mask_field,
                    return_mask=return_mask,
                )
                imputer_i.fit(df_)

                # ignore ssa component fields
                ignore_i = IgnoreField(ssa_i.component_fields())

                # save
                self._transforms.extend(
                    [norm_i, ssa_i, imputer_i, ignore_i, norm_i.reverse]
                )

                # apply transforms to continue fit on imputed series
                df_ = norm_i.transform(df_)
                df_ = ssa_i.transform(df_)
                df_ = imputer_i.transform(df_)
                df_ = ignore_i.transform(df_)
                df_ = norm_i.reverse.transform(df_)

                # overwrite frame until last iteration
                self._result = df_

                i += 1

            components_index = (
                1 if k == 0 else "+".join([str(c) for c in range(1, k + 2)])
            )
            logging.info(
                f"filled using component{'s' if k > 0 else ''} {components_index}"
            )
            del components_index

        logging.info(
            f"done... (total {i * (k + 1)} iter. in {time.time() - start:.4}s)"
        )

        return self

    def transform(self, df) -> pd.DataFrame:
        """
        ..note:: the last series of ``fit`` is the imputation result, so we just return it and no need to go through
            self._transforms
        """
        if self._result is not None:
            return self._result
        else:
            raise NotFittedError(">>> [SSA imputer] transform should be fitted first")

    def serialize(self):
        return dict(
            field=self.field,
            embedding_dimension=self.embedding_dimension,
            n_components=self.n_components,
            n_ev=self.n_ev,
            var_threshold=self.var_thresh,
            grouping_method=self.grouping_method,
            use_k_components=self.use_k_components,
            mask=self.mask,
            max_iter=self.max_iter,
            mask_field=self.mask_field,
        )

    @classmethod
    def deserialize(cls, data):
        v = cls(
            data["field"],
            embedding_dimension=data["embedding_dimension"],
            n_components=data["n_components"],
            n_ev=data["n_ev"],
            var_threshold=data["var_threshold"],
            grouping_method=data["grouping_method"],
            use_k_components=data["use_k_components"],
            mask=data["mask"],
        )
        v.max_iter = data["max_iter"]
        v.mask_field = data["mask_field"]
        return v


class ImputeIteration(TransformerMixin, BaseEstimator):

    SEPARATOR = "_"
    M_SUFFIX = SEPARATOR + "MASK"

    def __init__(
        self,
        field: str,
        imputing_fields: list,
        mask=None,
        mask_field: str = None,
        return_mask: bool = True,
    ):
        """
        Args
        ----
        field : str
            fields to be imputed

        imputing_fields : list
            field(s) used for imputation (e.g. could be one SSA component's fields, or 2, 7, etc)

        mask_field : str (optional)
            mask of missing values locations. if None, then a mask is constructed with 1s where
            NaNs are in the target series (from fields)

        return_mask : Boolean
            whether locations of missing values are returned
        """
        self.field = field
        self.imputing_fields = imputing_fields
        self.no_mask_field = mask_field is None
        self.mask = mask
        self.mask_field = (
            self.field + self.M_SUFFIX if mask_field is None else mask_field
        )
        self.return_mask = return_mask
        inputs = [self.field] + self.imputing_fields
        self.output_fields = inputs + [self.mask_field] if self.return_mask else inputs
        del inputs

    def fit(self, df: pd.DataFrame):
        if self.mask is None:
            if self.no_mask_field:
                self.mask = df[self.field].isnull()
                self.mask.columns = self.mask_field
            else:
                self.mask = df[self.mask_field]

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ = df.copy()
        df_values = df_[self.field].values
        new_values = np.add.reduce(
            [df_[field].values for field in self.imputing_fields]
        )
        bool_mask = self.mask.values.astype(dtype=bool)
        df_values[bool_mask] = new_values[bool_mask]
        del bool_mask
        df_.loc[:, self.field] = df_values

        if self.return_mask:
            df_ = pd.concat(
                [df_, self.mask.to_frame(name=self.mask_field)], axis="columns"
            )

        return df_

    def serialize(self):
        return dict(
            field=self.field,
            imputing_fields=self.imputing_fields,
            output_fields=self.output_fields,
        )

    @classmethod
    def deserialize(cls, data):
        # FIXME: this does not properly serialize masking_field and return_mask!
        return cls(
            data["field"],
            imputing_fields=data["imputing_fields"],
            # output_fields=data["output_fields"],
        )


class GPImputer(TransformerMixin, BaseEstimator):
    """in-place GP imputation of a single 1d series using single or composite kernels

    NOTE: part of the code is borrowed from sklearn GaussianProcessRegressor and the rest is
    about trying out stuffs.
    -> sparse approximation of the covariance matrix to alleviate the fact that this version
    of GPImputer is not processing batches of training data
    . FITC (fully independent training conditional)
    . VFE (variational free energy)
    . both reduce the expressiveness of the GP (reduced number of eigenvectors used to fit the
    data); might help for speedup when little data is missing?
    -> recursive (sparse) GP (with Kalman filter-like updates?)

    TODO:
    . recursive GP
    . allow for control over kernels (composition)

    References:
    -----------
    .. [0] Algorithm 2.1 in (Rasmussen and Williams, 2006)
    .. [1] https://github.com/scikit-learn/scikit-learn/blob/7813f7efb/sklearn/gaussian_process/gpr.py#L22
    .. [2] (Bauer, 2017) Understanding Probabilistic Sparse Gaussian Process Approximations
    https://papers.nips.cc/paper/6477-understanding-probabilistic-sparse-gaussian-process-approximations.pdf

    """

    class Kernels:
        MATERN = "matern"
        EXP_SIN_SQUARED = "exp_sin_squared"
        RATIONAL_QUADRATIC = "rational_quadratic"
        RBF = "rbf"  # good in general
        DOT_PRODUCT = "dot_product"

    STDEV_SUFFIX = "_STDEV"
    POST_SUFFIX = "_POSTERIOR"

    def __init__(
        self,
        field: str,
        kernel: Optional[Kernels] = Kernels.MATERN,
        n_skip: Optional[int] = None,
        mask: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, list]] = None,
        return_std: Optional[bool] = False,
        n_samples_posterior: Optional[int] = None,
        noise_level: Optional[float] = None,
        normalize: Optional[bool] = False,
        backward: Optional[bool] = False,
        verbose: Optional[bool] = False,
    ):
        """
        Args:
        -----
        field : list
            single time series field to fill

        kernel : Kernels
            type of kernel to use

        n_skip : int
            number of leading nan rows (first chunk) to ignore, ie. those are not counted as
            missing values to fill and added back after imputation is completed

        return_std : bool
            whether to return predicted std

        n_sample_posterior : additional n samples from GP posterior are drawn at missing
            values locations

        noise_level : float
            level of noise when using a WhiteKernel
        """
        assert isinstance(field, str)

        self.field = field
        self.mask = (
            mask if mask is None else self._validate_precomputed_mask(mask, field)
        )
        self.seed = 42
        self.noise_level = noise_level or 0
        self.kernel = self._init_kernel(kernel)
        self.n_skip = n_skip or 0
        self.return_std = return_std
        self.n_samples_posterior = n_samples_posterior
        self.normalize = normalize
        self.backward = backward
        self.verbose = verbose
        self.y_train = None
        self.x_train = None
        self.log_marginal_likelihood_value = None
        self.L = None
        self.K_aa_inv = None
        self.alpha = None

    @staticmethod
    def _validate_precomputed_mask(mask, field) -> pd.Series:
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, name=field).astype(bool)
            return mask
        elif isinstance(mask, pd.Series):
            assert mask.name == field
            return mask.astype(bool)
        elif isinstance(mask, pd.DataFrame):
            assert field in mask.columns
            return mask[field].astype(bool)
        else:
            raise TypeError("unsupported type for given mask")

    def _init_kernel(self, kernel: Kernels):
        if kernel == self.Kernels.MATERN:
            return Product(
                ConstantKernel(),
                Sum(Matern(), WhiteKernel(noise_level=self.noise_level)),
            )
        elif kernel == self.Kernels.EXP_SIN_SQUARED:
            return Product(
                ConstantKernel(),
                Sum(ExpSineSquared(), WhiteKernel(noise_level=self.noise_level)),
            )
        elif kernel == self.Kernels.RATIONAL_QUADRATIC:
            return Product(
                ConstantKernel(),
                Sum(RationalQuadratic(), WhiteKernel(noise_level=self.noise_level)),
            )
        elif kernel == self.Kernels.RBF:
            return Product(
                ConstantKernel(), Sum(RBF(), WhiteKernel(noise_level=self.noise_level))
            )
        elif kernel == self.Kernels.DOT_PRODUCT:
            return Product(
                ConstantKernel(),
                Sum(DotProduct() ** 2, WhiteKernel(noise_level=self.noise_level)),
            )
        else:
            raise TypeError("unsupported type of kernel")

    def fit(self, df: pd.DataFrame, max_evals: int = 1):
        def obj_func(theta):
            lml, grad = self.log_marginal_likelihood(theta, eval_gradient=True)
            return -1 * lml, -1 * grad

        np.random.seed(self.seed)

        chunk_ = df.copy()
        if self.backward:
            chunk_ = reverse(chunk_, [self.field])

        # normalize
        if self.normalize:
            chunk_ = Normalize(self.field).fit_transform(chunk_)

        y_ = chunk_[self.field].values
        # get values locations
        if self.mask is None or self.backward:
            # recompute mask in case it was given fwd
            nm = ~np.isnan(y_)
        else:
            nm = ~self.mask.to_numpy()

        self.y_train = y_[nm].reshape(-1, 1)
        self.x_train = np.where(nm)[0].reshape(-1, 1)
        initial_params = self.kernel.theta

        if self.verbose:
            logging.info(
                f">>> minimisation of negative lml\n **max_evals: {max_evals}\n "
                f"**initial hyper-parameters: {initial_params}"
            )

        start = time.time()

        # optimise using initial theta and bounds given by kernel
        all_evals = [(self._optimise(obj_func, initial_params, self.kernel.bounds))]

        # max_evals additional runs using log-uniform randomly chosen initial theta
        if max_evals > 0:
            bounds = self.kernel.bounds
            for _i in range(max_evals):
                theta_initial = np.random.uniform(bounds[:, 0], bounds[:, 1])
                all_evals.append(self._optimise(obj_func, theta_initial, bounds))

        # select set of params with minimal (negative) log-marginal likelihood
        lml_values = np.hstack(list(map(itemgetter(1), all_evals)))
        self.kernel.theta = all_evals[np.argmin(lml_values)][0]
        self.log_marginal_likelihood_value = -1 * np.min(lml_values)

        if self.verbose:
            logging.info(
                f">>> optimisation... done ({np.round(time.time() - start, 3)}s). "
                f"Final lml: {np.round(self.log_marginal_likelihood_value, 3)}"
            )
            logging.info(f">>> best hyperparams:{self.kernel.theta}")

        K_aa = self.kernel(self.x_train)

        # pre-compute L, alpha and inverse prior covariance using best hyperparams
        if not is_PD(K_aa):
            if self.verbose:
                logging.warning(">>> K_aa was not PD...")
            K_aa = make_PD(K_aa)

        self.L = cho_factor(K_aa, lower=True)
        self.K_aa_inv = cho_solve(self.L, np.eye(K_aa.shape[0]))
        self.alpha = cho_solve(self.L, self.y_train)  # (3) in Alg. 2.1

        return self

    @staticmethod
    def _optimise(obj_func, initial_theta, bounds):
        theta_opt, func_min, _ = fmin_l_bfgs_b(
            obj_func, initial_theta, bounds=bounds, iprint=0
        )
        return theta_opt, func_min

    def log_marginal_likelihood(self, theta=None, eval_gradient=False):
        """log-marginal likelihood (eq.2.30)"""
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "cannot compute partial derivatives wrt. params if no params is given"
                )
            return self.log_marginal_likelihood_value
        else:
            kernel = self.kernel.clone_with_theta(theta)
            if eval_gradient:
                K_aa, K_aa_grad = kernel(self.x_train, eval_gradient=True)
            else:
                K_aa = kernel(self.x_train)

            if not is_PD(K_aa):
                if self.verbose:
                    logging.warning(">>> K_aa was not PD...")
                K_aa = make_PD(K_aa)

            L = cho_factor(K_aa, lower=True)
            alpha = cho_solve(L, self.y_train)  # step 3) in Alg. 2.1
            # log marginal likelihood terms (eq.2.30)
            lml_ = -0.5 * np.dot(self.y_train.T, alpha)  # controls fit to data
            lml_ -= np.log(np.diag(L[0])).sum()  # controls complexity of the model
            lml_ -= K_aa.shape[0] / 2 * np.log(2 * np.pi)  # normalisation constant
            lml = lml_.sum(-1)

        if eval_gradient:
            K_aa_inv = cho_solve(L, np.eye(K_aa.shape[0]))  # may take long..
            # partial derivatives of marginal likelihood wrt hyperparams (eq.5.9)
            lml_grad = (
                0.5
                * np.einsum(
                    "ijl,ijk->kl",
                    np.einsum("ik,jk->ijk", alpha, alpha) - K_aa_inv[:, :, np.newaxis],
                    K_aa_grad,
                ).sum(-1)
            )
            return lml, lml_grad
        else:
            return lml

    def transform(self, chunk: pd.DataFrame):
        """predicting at missing values

        chunk : dataframe
            test points are locations of predictions -nans-, eq. to "x*"
            in (Rasmussen and Williams, 2006) and "_b" here
        """
        try:
            chunk_ = chunk.copy()
            if self.backward:
                chunk_ = reverse(chunk_, self.field)

            if self.normalize:
                NT = Normalize(self.field, denormalize=True)
                NT.fit(chunk_)
                chunk_ = NT.transform(chunk_)

            m = np.isnan(chunk_[self.field].values[self.n_skip :])
            missing_x = np.where(m)[0].reshape(-1, 1)

            if len(missing_x) == 0:
                chunk_values = chunk_[self.field].values
                if self.return_std:
                    chunk_[self.field + self.STDEV_SUFFIX] = np.zeros(len(chunk_))
                if self.n_samples_posterior is not None:
                    chunk_[self.posterior_samples_fields] = pd.DataFrame(
                        np.tile(chunk_values, (self.n_samples_posterior, 1)).T
                    )

            else:
                K_ab = self.kernel(self.x_train, missing_x)
                # evaluate kernel at test points using fixed parameters
                K_bb = self.kernel(missing_x)
                # predictive mean (eq.2.25)
                mean = np.einsum("ij,jk->i", K_ab.T, self.alpha)
                # predictive variance (eq.2.26)
                if self.return_std:
                    v = solve_triangular(self.L[0], K_ab, lower=True)
                    var = np.diag(K_bb) - np.einsum("ij,ij->j", v, v)
                    std = np.zeros(len(chunk_))
                    # pylint: disable=unsupported-assignment-operation
                    std.flat[missing_x] = np.sqrt(var)

                # fill
                chunk_[self.field].fillna(
                    value=dict(zip(missing_x.ravel(), mean)), inplace=True
                )

                if self.return_std:
                    chunk_[self.field + self.STDEV_SUFFIX] = std

                if self.n_samples_posterior is not None:
                    # (eq.2.24) posterior covariance
                    cov = K_bb - np.linalg.multi_dot([K_ab.T, self.K_aa_inv, K_ab])
                    # draw samples from posterior at missing values locations
                    p = np.random.multivariate_normal(
                        mean, cov, self.n_samples_posterior
                    )
                    temp = np.tile(
                        chunk_[self.field].values, (self.n_samples_posterior, 1)
                    )
                    temp[:, missing_x.ravel()] = p
                    chunk_[self.posterior_samples_fields] = pd.DataFrame(temp.T)

            chunk_ = (
                NT.reverse.apply_chunk(chunk_) if self.normalize else chunk_
            )  # denormalize

            return (
                reverse(chunk_[chunk.columns], self.field)
                if self.backward
                else chunk_[chunk.columns]
            )

        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    @property
    def output_fields(self):
        return (
            [self.field, [self.field + self.STDEV_SUFFIX]]
            if self.return_std
            else [self.field]
        )

    @property
    def posterior_samples_fields(self):
        if self.n_samples_posterior is not None:
            return [
                self.field + self.POST_SUFFIX + f"{i + 1}"
                for i in range(self.n_samples_posterior)
            ]
        else:
            return []

    def serialize(self):
        return dict(
            field=self.field,
            n_skip=self.n_skip,
            return_std=self.return_std,
            n_samples_posterior=self.n_samples_posterior,
            noise_level=self.noise_level,
            normalize=self.normalize,
            backward=self.backward,
            mask=self.mask,
            verbose=self.verbose,
        )

    @classmethod
    def deserialize(cls, data):
        return cls(
            data["field"],
            n_skip=data["n_skip"],
            return_std=data["return_std"],
            n_samples_posterior=data["n_samples_posterior"],
            verbose=data["verbose"],
            noise_level=data["noise_level"],
            backward=data["backward"],
            mask=data["mask"],
            normalize=data["normalize"],
        )


class SVTImputer(TransformerMixin, BaseEstimator):
    class Solver:
        SVT = "svt"
        SVT_FAST = "svt_fast"
        PMF = "pmf"  # noqa
        PMF_BIASED = "pmf_biased"  # noqa
        MCMT = "mcmt"  # noqa

    def __init__(
        self,
        field: str,
        mask: Optional[Union[pd.Series, pd.DataFrame, np.ndarray, list]] = None,
        embedding_dimension: Optional[int] = None,
        solver: Optional[str] = Solver.SVT,
        max_iterations: Optional[int] = 400,
        n_skip: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        """inplace imputation of missing values in a 1d series using singular value thresholding

        Args
        ----
        fields : list
            time series values field

        embedding_dimension : int (optional, default: None)
            window size to apply to 1d series (inferred if None)

        solver : str (optional, default: svt)
            matrix completion solver
            . svt/svt_fast: singular value thresholding [1,2]
            TODO:
            . pmf/pmf_biased: solves probabilistic matrix factorization via alternating LS [3,4]
            . mcmt: matrix completion under multiple linear transformations [5]

        mask : pd.Series/pd.DataFrame/np.ndarray/list
            precomputed mask

        NOTE:
        . may be slow since we compute svd on a large matrix... to be replaced with fast SVT.
        . since spectral analysis is made on chunks individually, those should be as large as
        possible, in order to include as much information as possible

        References
        -----------
        .. [1] (Cai, 2008) A Singular Value Thresholding Algorithm for Matrix Completion
        https://arxiv.org/pdf/0810.3286.pdf
        .. [2] (Cai, 2013) Fast Singular Value Thresholding without Singular Value Decomposition
        https://www.math.ust.hk/~jfcai/paper/fastSVT.pdf
        .. [2b] (Xu, 2015) The minimal measurement number for low-rank matrices recovery
        https://arxiv.org/pdf/1505.07204.pdf
        .. [3] (Salakhutdinov and Mnih, 2007) Probabilistic Matrix Factorization
        https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
        .. [4] (Paterek, 2007) Improving regularized singular value decomposition for collaborative filtering
        https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
        .. [5] (Li, 2019) Matrix completion under multiple linear transformations
        http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Guaranteed_Matrix_Completion_Under_Multiple_Linear_Transformations_CVPR_2019_paper.pdf
        """
        assert isinstance(field, str)

        self.field = field
        self.solver = solver
        self.embedding_dimension = embedding_dimension
        self.n_skip = n_skip
        self.max_iterations = max_iterations
        self.mask = (
            mask if mask is None else self._validate_precomputed_mask(mask, field)
        )
        self.verbose = verbose

    @staticmethod
    def _validate_precomputed_mask(mask, field) -> pd.Series:
        if isinstance(mask, (np.ndarray, list)):
            mask = pd.Series(mask, name=field).astype(bool)
            return mask
        elif isinstance(mask, pd.Series):
            assert mask.name == field
            return mask.astype(bool)
        elif isinstance(mask, pd.DataFrame):
            assert field in mask.columns
            return mask[field].astype(bool)
        else:
            raise TypeError("unsupported type for given mask")

    @staticmethod
    def _validate_window(df, field, nskip: int = 0, window=None, verbose: bool = True):
        # checking clusters of nans
        m = max_consecutive_nans(df[field].values.reshape(1, -1), nskip=nskip)
        lb = m[0] + 1  # inclusive lower bound
        ub = len(df[field]) - m[0] + 1

        if window is None:
            return int((lb + ub) / 2)
        else:
            if not lb <= window <= ub:
                w = int((lb + ub) / 2)
                if verbose:
                    logging.warning(
                        f"window ({window}) lies outside [{lb},{ub}] -> forcing w={w}"
                    )
                return w
            else:
                return window

    def fit(self, chunk):
        chunk_ = chunk.copy()[[self.field]]

        # validate window
        w = self._validate_window(
            chunk_, self.field, nskip=self.n_skip, window=self.embedding_dimension
        )

        # normalize input
        NT = Normalize(self.field, denormalize=True)
        NT.fit_begin()
        NT.fit_chunk(chunk)
        NT.fit_end()
        chunk_ = NT.apply_chunk(chunk_)

        # window
        dfw = embed(chunk_.loc[self.n_skip :, :], [self.field], w, keep_dims=False)
        windowed_fields = [windowed_name(self.field, t) for t in range(-w + 1, 1)]
        if self.mask is None:
            mask_inv = ~dfw[windowed_fields].isnull().to_numpy().astype(int)
        else:
            mask_inv = embed_np(~self.mask.to_numpy(), w).astype(int)

        # setting nans to 0
        Xz = np.nan_to_num(dfw[windowed_fields])

        # solve
        if self.solver == self.Solver.SVT:
            self.X = svt(Xz, mask=mask_inv, max_iterations=self.max_iterations)
        elif self.solver == self.Solver.SVT_FAST:
            raise NotImplementedError()
        else:
            raise ValueError("unknown solver")

        # update fill
        self.X[mask_inv] = Xz[mask_inv]

        return self

    def transform(self, chunk: pd.DataFrame) -> pd.DataFrame:
        try:
            chunk_ = chunk.copy()

            # normalize input
            NT = Normalize(self.field, denormalize=True)
            NT.fit_begin()
            NT.fit_chunk(chunk)
            NT.fit_end()
            chunk_ = NT.apply_chunk(chunk_)

            # put back skipped nans
            if self.n_skip > 0:
                chunk_[self.field] = np.r_[[np.nan] * self.n_skip, diag_avg(self.X)]
            else:
                chunk_[self.field] = diag_avg(self.X)

            # denormalize
            chunk_ = NT.reverse.apply_chunk(chunk_)

            return chunk_[chunk.columns]

        except AttributeError:
            raise NotFittedError("transform should be fitted first")

    def serialize(self):
        return dict(
            field=self.field,
            embedding_dimension=self.embedding_dimension,
            solver=self.solver,
            max_iterations=self.max_iterations,
            n_skip=self.n_skip,
            mask=self.mask,
            verbose=self.verbose,
        )

    @classmethod
    def deserialize(cls, data):
        return cls(
            data["field"],
            embedding_dimension=data["embedding_dimension"],
            solver=data["solver"],
            max_iterations=data["max_iterations"],
            mask=data["mask"],
            n_skip=data["n_skip"],
            verbose=data["verbose"],
        )


# Helper functions


def reverse(
    df: pd.DataFrame, field_or_fields: Optional[Union[list, str]] = None
) -> pd.DataFrame:
    """reverse a field's values"""
    df_ = df.copy()
    if field_or_fields is None:
        df_ = df_[::-1].reset_index(drop=True)  # reverse whole frame
    else:
        df_[field_or_fields] = df_[field_or_fields][::-1].reset_index(drop=True)
    return df_


def svt(A, mask=None, tau=None, delta=None, epsilon=1e-2, max_iterations=1000):
    """low-rank matrix completion using iterative singular value thresholding
    (see Section 5.1.5 in [1])

    Args
    ----
    A : np.ndarray
            (m x n) matrix to impute

    mask : np.ndarray (optional, default: None)
            0s at missing values, 1s otherwise

    tau : float (optional, default: 5*(m+n)/2)
            singular value threshold

    delta : float
            step size per iteration; default to 1.2 times the under-sampling ratio

    epsilon : float (optional, default: 1e-2)
                convergence condition on the relative reconstruction error

    Returns
    -------
    filled matrix (quasi-Hankel)

    References
    ----------
    .. [1] (Cai, 2008) A Singular Value Thresholding Algorithm for Matrix Completion
    https://arxiv.org/pdf/0810.3286.pdf
    """
    if mask is None:
        mask = 1 * ~np.isnan(A)

    Y = np.zeros_like(A)
    if not tau:
        tau = 5 * np.sum(A.shape) / 2
    if not delta:
        delta = 1.2 * np.prod(A.shape) / np.sum(mask)

    for _k in range(max_iterations):
        U, S, V = np.linalg.svd(Y, full_matrices=False)
        S = np.maximum(S - tau, 0)  # singular value shrinkage operator
        X = np.linalg.multi_dot([U, np.diag(S), V])
        Y += delta * mask * (A - X)
        # normalised errors
        err = np.linalg.norm(mask * (X - A)) / np.linalg.norm(mask * A)
        if err < epsilon:
            break

    return X


def make_PD(A):
    """Find the nearest (symmetric) positive-definite matrix to A

    Reference
    ---------
    .. [1] (Higham, 1988) Computing a nearest symmetric positive semi-definite matrix
    """
    # ensure A is square
    assert A.shape[0] == A.shape[1], "A must be square"

    B = (A + A.T) / 2  # symmetrize A into B
    _, s, V = np.linalg.svd(B)  # symmetric polar factor of B
    H = np.dot(V.T, np.dot(np.diag(s), V))
    A_hat = (B + H) / 2
    A_psd = (A_hat + A_hat.T) / 2  # make symmetric

    # Cholesky test on A_psd
    if is_PD(A_psd):
        return A_psd
    else:
        # add small noise to diagonal
        s = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not is_PD(A_psd):
            min_ev = np.min(np.real(np.linalg.eigvals(A_psd)))
            A_psd += I * (-min_ev * k ** 2 + s)
            k += 1
        return A_psd


def is_PD(B):
    """Cholesky test for positive-definiteness of a square matrix B"""
    assert B.shape[0] == B.shape[1], "matrix is not square"
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


# def svt_fast(A, mask, delta=None, epsilon=1e-2):
#     """matrix completion using fast svt [2]

#     References
#     ----------
#     .. [2] (Cai, 2013) Fast Singular Value Thresholding without Singular Value Decomposition
#         https://www.math.ust.hk/~jfcai/paper/fastSVT.pdf
#     """
#     Y = np.zeros_like(A)
#     m,n=Y.shape

#     if not tau:
#         tau=5*np.sum(A.shape)/2
#     if not delta:
#         delta=1.2*np.prod(A.shape)/np.sum(mask)


#     W,Z=np.polar(Y)
#     lambdas,V=np.eigh(Z)
#     idx=lambdas.argsort()[::-1]
#     lambdas=lambdas[idx]
#     V=V[:,idx]

#     Z=Z-
#     P = np.zeros_like(A)
#     for in range(max_iterations):
#         B = 2*P - Z - tau*np.eye((m,n))
#         C = cho_factor(B, lower=True) # B is symmetric
#         B_inv = cho_solve(C, np.eye((m,n)))
#         P_new = 0.5*P + 0.25*Z + 0.75*np.eye((m,n)) - np.dot(B_inv, (P - 0.25*np.linalg.matrix_power(Z,2) - 0.75*np.eye((m,n)))

#         err=np.linalg.norm(P_new-P)/np.linalg.norm(Z)

#         if err<epsilon:
#             break
