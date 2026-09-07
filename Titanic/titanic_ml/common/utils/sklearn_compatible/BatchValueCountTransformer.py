from sklearn.base import BaseEstimator, TransformerMixin

class BatchValueCountTransformer(
    BaseEstimator,
    TransformerMixin,
):

    """
    A transformer that counts the occurrences of each unique value in a specified source column and creates a new column with these counts.
    This transformer is compatible with scikit-learn's fit/transform interface.
    
    The output depends on the batch currently being transformed.
    
    """
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