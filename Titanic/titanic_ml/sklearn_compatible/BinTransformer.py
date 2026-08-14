import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class BinTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        source_col,
        output_col,
        bins,
        labels=None,
        right=False,
    ):
        self.source_col = source_col
        self.output_col = output_col
        self.bins = bins
        self.labels = labels
        self.right = right

    def fit(self, X, y=None):
        if self.source_col not in X.columns:
            raise ValueError(
                f"Source column '{self.source_col}' not found."
            )

        if (
            self.labels is not None
            and len(self.labels) != len(self.bins) - 1
        ):
            raise ValueError(
                "Number of labels must equal number of bins - 1."
            )

        return self

    def transform(self, X):
        X = X.copy()

        X[self.output_col] = pd.cut(
            X[self.source_col],
            bins=self.bins,
            labels=self.labels,
            right=self.right,
        )

        return X