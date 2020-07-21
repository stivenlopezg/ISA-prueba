import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnsSelector(BaseEstimator, TransformerMixin):
    """
    Select the columns of interest of a DataFrame
    """

    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self.columns]
        return X

