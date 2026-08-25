from sklearn.base import BaseEstimator, TransformerMixin

class ValueCountTransformer(
    BaseEstimator,
    TransformerMixin,
):
    def __init__(
        self,
        source_col,
        output_col,
    ):
        self.source_col = source_col
        self.output_col = output_col

    def fit(self, X, y=None):
        if self.source_col not in X.columns:
            raise ValueError(
                f"Source column '{self.source_col}' not found."
            )

        return self

    def transform(self, X):
        X = X.copy()

        X[self.output_col] = (
            X.groupby(
                self.source_col
            )[self.source_col]
            .transform("size")
        )

        return X