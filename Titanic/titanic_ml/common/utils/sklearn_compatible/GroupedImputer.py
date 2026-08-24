import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupedImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        target_col,
        group_cols,
        agg_func="median",
        fallback_to_global=True,
    ):
        self.target_col = target_col
        self.group_cols = group_cols
        self.agg_func = agg_func
        self.fallback_to_global = fallback_to_global

    def fit(self, X, y=None):
        X = X.copy()

        if self.target_col not in X.columns:
            raise ValueError(
                f"Target column '{self.target_col}' not found."
            )

        missing_group_cols = [
            col
            for col in self.group_cols
            if col not in X.columns
        ]

        if missing_group_cols:
            raise ValueError(
                f"Group columns not found: {missing_group_cols}"
            )

        if self.agg_func == "median":
            self.group_values_ = (
                X.groupby(self.group_cols)[self.target_col]
                .median()
            )

            self.global_value_ = (
                X[self.target_col].median()
            )

        elif self.agg_func == "mean":
            self.group_values_ = (
                X.groupby(self.group_cols)[self.target_col]
                .mean()
            )

            self.global_value_ = (
                X[self.target_col].mean()
            )

        elif self.agg_func == "mode":
            self.group_values_ = (
                X.groupby(self.group_cols)[self.target_col]
                .agg(
                    lambda s:
                    s.mode().iloc[0]
                    if not s.mode().empty
                    else pd.NA
                )
            )

            global_mode = X[self.target_col].mode()

            self.global_value_ = (
                global_mode.iloc[0]
                if not global_mode.empty
                else pd.NA
            )

        else:
            raise ValueError(
                f"Unsupported agg_func: '{self.agg_func}'"
            )

        return self

    def transform(self, X):
        X = X.copy()

        missing_mask = X[self.target_col].isna()

        if not missing_mask.any():
            return X

        group_values = (
            X.loc[missing_mask, self.group_cols]
        )

        if len(self.group_cols) == 1:
            mapped_values = (
                group_values[self.group_cols[0]]
                .map(self.group_values_)
            )

        else:
            group_index = pd.MultiIndex.from_frame(
                group_values
            )

            mapped_values = pd.Series(
                self.group_values_
                .reindex(group_index)
                .to_numpy(),
                index=group_values.index,
            )

        X.loc[
            missing_mask,
            self.target_col
        ] = mapped_values

        if self.fallback_to_global:
            X[self.target_col] = (
                X[self.target_col]
                .fillna(self.global_value_)
            )

        return X