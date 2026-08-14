import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AgeBinTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        source_col="Age",
        output_col="Age_bin",
        bins=None,
        labels=None,
        right=False,
    ):
        self.source_col = source_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels
        self.right = right

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        bins = (
            self.bins
            if self.bins is not None
            else [0, 14, 35, 60, 100]
        )

        labels = (
            self.labels
            if self.labels is not None
            else ["0", "2", "3", "1"]
        )

        X[self.output_col] = pd.cut(
            X[self.source_col],
            bins=bins,
            labels=labels,
            right=self.right,
        )

        return X