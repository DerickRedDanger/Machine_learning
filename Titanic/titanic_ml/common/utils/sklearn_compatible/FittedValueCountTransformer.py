from sklearn.base import BaseEstimator, TransformerMixin

class FittedValueCountTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    A transformer that counts the occurrences of each unique value in a specified source column and creates a new column with these counts.
    This transformer is compatible with scikit-learn's fit/transform interface.
    
    Counts are learned during fit() and reused during transform()
    """

    def __init__(
        self,
        source_col,
        output_col,
    ):
        self.source_col = source_col
        self.output_col = output_col

    def fit(self, X, y=None):
        self.counts_ = X[self.source_col].value_counts()
        return self

    def transform(self, X):
        X = X.copy()
        X[self.output_col] = X[self.source_col].map(
            self.counts_
        )
        return X
