from sklearn.base import BaseEstimator, TransformerMixin

class JuxtapositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, source_cols, output_col):
        self.source_cols = source_cols
        self.output_col = output_col

    def fit(self, X, y=None):
        if not all(col in X.columns for col in self.source_cols):
            missing_cols = [col for col in self.source_cols if col not in X.columns]
            raise ValueError(f"Source columns {missing_cols} not found.")
        return self

    def transform(self, X):
        X = X.copy()
        X[self.output_col] = X[self.source_cols].astype(str).agg(' '.join, axis=1)
        return X