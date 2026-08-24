from sklearn.base import BaseEstimator, TransformerMixin

class FeatureDividedByFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerator_col, denominator_col, output_col):
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col
        self.output_col = output_col

    def fit(self, X, y=None):
        if self.numerator_col not in X.columns:
            raise ValueError(
                f"Numerator column '{self.numerator_col}' not found."
            )
        if self.denominator_col not in X.columns:
            raise ValueError(
                f"Denominator column '{self.denominator_col}' not found."
            )
        return self

    def transform(self, X):
        X = X.copy()
        X[self.output_col] = X[self.numerator_col] / X[self.denominator_col]
        return X