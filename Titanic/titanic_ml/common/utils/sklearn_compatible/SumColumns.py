from sklearn.base import BaseEstimator, TransformerMixin


class SumColumnsTransformer(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(
        self,
        source_cols,
        output_col,
        constant=0,
    ):
        self.source_cols = source_cols
        self.output_col = output_col
        self.constant = constant

    def fit(self, X, y=None):
        missing_cols = [
            col
            for col in self.source_cols
            if col not in X.columns
        ]

        if missing_cols:
            raise ValueError(
                f"Source columns {missing_cols} not found."
            )

        return self

    def transform(self, X):
        X = X.copy()

        X[self.output_col] = (
            X[self.source_cols].sum(axis=1)
            + self.constant
        )

        return X