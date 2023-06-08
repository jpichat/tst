import warnings
import typing as t
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class Normalize(TransformerMixin, BaseEstimator):
    """
    Normalizes a numeric column to mean=0 and standard deviation=1

    Ignores any NaN in the input.
    """

    def __init__(self, field: str, mean=None, stdev=None, denormalize=False):
        self.field = field
        self.mean = mean
        self.stdev = stdev
        self.denormalize = denormalize

    def fit(self, df: pd.DataFrame):
        if self.field not in df.columns:
            # Assume field is a target, gracefully ignore it.
            return df

        x = self._get_values(df, self.field)
        self.mean = np.nanmean(x)
        self.stdev = np.nanstd(x)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is not None and self.stdev is not None:
            x = self._get_values(df, self.field)
            y = (x - self.mean) / (self.stdev + np.finfo(float).eps)
            if abs(self.stdev) / 10.0 <= np.finfo(float).eps:
                y = x * 0.0
                warnings.warn(
                    ">>> [Normalize] processing a field with stddev below numerical precision. "
                    "There might be a problem in preceding transforms",
                )
            df[[self.field]] = y

            return df
        else:
            raise NotFittedError(">>> [Normalize] tranform should be fitted first")

    @staticmethod
    def _get_values(df: pd.DataFrame, field: str):
        if df[field].dtype == "object":
            return pd.to_numeric(df[field].values, errors="coerce")
        else:
            return df[field].values

    @property
    def reverse(self):
        if self.denormalize:
            return Denormalize(self.field, mean=self.mean, stdev=self.stdev)

    def serialize(self):
        return dict(
            field=self.field,
            mean=self.mean,
            stdev=self.stdev,
            denormalize=self.denormalize,
        )

    @classmethod
    def deserialize(cls, data):
        return cls(
            data["field"],
            mean=data["mean"],
            stdev=data["stdev"],
            denormalize=data["denormalize"],
        )


class Denormalize(Normalize):
    """
    Denormalizes predictions.
    """

    @property
    def reverse(self):
        return Normalize(self.field, mean=self.mean, stdev=self.stdev, denormalize=True)

    def transform(self, df: pd.DataFrame):
        y = df[self.field].values
        x = y * self.stdev + self.mean
        df[self.field] = x
        return df


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Scaling each field in-place to a given range"""

    def __init__(self, field: str, rescaling_range: t.Tuple = (0, 1)):
        """
        Args
        ----
        field : str
            field to rescale

        rescaling_range : tuple
            min/max of rescaled output fields

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Feature_scaling
        .. [2] https://en.wikipedia.org/wiki/Normalization_(image_processing)
        """
        self.field = field
        self._min_out = rescaling_range[0]
        self._max_out = rescaling_range[1]
        self._min_in = None
        self._max_in = None

    def fit(self, df: pd.DataFrame):
        self._min_in = np.inf
        self._max_in = -np.inf

        with warnings.catch_warnings():

            # suppress "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)

            min_ = np.nanmin(df[self.field].values, axis=0)
            max_ = np.nanmax(df[self.field].values, axis=0)
            # update global min/max
            self._min_in = np.nan_to_num(
                np.clip(min_, -np.inf, self._min_in), nan=np.inf
            )
            self._max_in = np.nan_to_num(
                np.clip(max_, self._max_in, np.inf), nan=-np.inf
            )

    def transform(self, df: pd.DataFrame):
        is_fitted = self._max_in is not None and self._min_in is not None
        assert is_fitted, "Transform should be fitted first"
        chunk_keep = df[list(set(df.columns) - set([self.field]))]
        denominator = (self._max_in - self._min_in).astype(float)
        denominator[denominator == 0.0] = np.inf  # for safe divisions by 0
        std = (df[self.field].values - self._min_in) / denominator
        chunk_scaled = pd.DataFrame(
            std * (self._max_out - self._min_out) + self._min_out, columns=[self.field]
        )
        return pd.concat([chunk_keep, chunk_scaled], axis="columns")

    @property
    def fields_range(self) -> dict:
        return {self.field: self._max_in - self._min_in}

    @property
    def fields_min(self) -> dict:
        return {self.field: self._min_in}

    @property
    def fields_max(self) -> dict:
        return {self.field: self._max_in}

    @property
    def reverse(self):
        scaler = MinMaxScaler(self.field, rescaling_range=(self._min_in, self._max_in))
        scaler._min_in = np.array([self._min_out])
        scaler._max_in = np.array([self._max_out])
        return scaler

    def serialize(self):
        return dict(
            fields=self.field,
            range=(self._min_out, self._max_out),
            min_in=self._min_in,
            max_in=self._max_in,
        )

    @classmethod
    def deserialize(cls, data):
        v = cls(data["fields"], rescaling_range=data["range"])
        v._min_in = data["min_in"]
        v._max_in = data["max_in"]
        return v
