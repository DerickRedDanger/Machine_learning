from sklearn.base import BaseEstimator, TransformerMixin


class TitleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.source_col = "Name"
        self.output_col = "Title"

    def fit(self, X, y=None):
        if self.source_col not in X.columns:
            raise ValueError(
                f"Source column '{self.source_col}' not found."
            )

        return self

    def transform(self, X):
        X = X.copy()

        title_name = (
            X[self.source_col]
            .str.split(",")
            .str[1]
        )

        title = (
            title_name
            .str.split(".")
            .str[0]
        )

        X[self.output_col] = title.str.strip()

        X[self.output_col] = (
            X[self.output_col]
            .replace({
                "Mlle": "Miss",
                "Ms": "Miss",
                "Mme": "Mrs",
            })
        )

        rare_titles = [
            "Dr",
            "Rev",
            "Col",
            "Major",
            "Don",
            "Lady",
            "Sir",
            "Capt",
            "the Countess",
            "Jonkheer",
        ]

        X[self.output_col] = (
            X[self.output_col]
            .replace(rare_titles, "Rare")
        )

        return X


TITLE_TRANSFORM = {
    "id": "title",
    "transformer": TitleTransformer,
    "requires": ["Name"],
    "produces": ["Title"],
    "owns": ["Title"],
}