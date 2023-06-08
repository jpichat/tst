import typing as t
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class IgnoreField(TransformerMixin, BaseEstimator):
    """
    This transforms removes fields in the input from the output
    """

    def __init__(self, fields: t.List[str]):
        assert isinstance(fields, list)
        assert len(fields)
        self.fields = fields

    def fit(self, df: pd.DataFrame):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=[self.fields], errors="ignore")
        return df

    def export(self):
        return dict(fields=self.fields)

    def serialize(self):
        return dict(fields=self.fields)

    @classmethod
    def deserialize(cls, data):
        return IgnoreField(data["fields"])
