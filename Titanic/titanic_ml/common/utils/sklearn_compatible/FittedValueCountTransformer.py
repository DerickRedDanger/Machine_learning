from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class FittedValueCountTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Count occurrences of values in a column during fit() and use those
    learned counts to create a new column during transform().

    Parameters
    ----------
    source_col : str
        Column whose values will be counted.

    output_col : str
        Name of the column containing the learned counts.

    unseen_value : scalar, default=np.nan
        Value assigned during transform() when a non-missing value was
        not present during fit().

    missing_value : scalar, default=np.nan
        Value assigned to missing values (Nan) during transform() when
        count_missing=False.

        Ignored when count_missing=True, because missing values are counted
        during fit() and therefore have a learned count.

    count_missing : bool, default=False 
        Passed as dropna=not self.count_missing to pandas.Series.value_counts(),
        So counts_missing=False means dropna=True.

        If False:
            Missing values are not treated as a learned category and receive
            `missing_value` during transform().

        If True:
            missing values are counted like any other observed value and receive
            their learned count during transform().
    """

    def __init__(
        self,
        source_col,
        output_col,
        unseen_value=np.nan,
        missing_value=np.nan,
        count_missing=False,
    ):
        self.source_col = source_col
        self.output_col = output_col
        self.unseen_value = unseen_value
        self.missing_value = missing_value
        self.count_missing = count_missing

    def fit(self, X, y=None):
        source = X[self.source_col]

        self.counts_ = source.value_counts(
            dropna=not self.count_missing
        )

        return self

    def transform(self, X):
        X = X.copy()

        source = X[self.source_col]
        counts = source.map(self.counts_)

        missing_mask = source.isna()
        unseen_mask = counts.isna() & ~missing_mask

        # Handle values that were not observed during fit().
        counts.loc[unseen_mask] = self.unseen_value

        # Missing values need an explicit fallback only when they
        # were excluded from the fitted counts.
        if not self.count_missing:
            counts.loc[missing_mask] = self.missing_value

        X[self.output_col] = counts

        return X