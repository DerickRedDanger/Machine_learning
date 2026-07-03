import pandas as pd

def add_age_bin(df):
    df = df.copy()

    # Age groups are encoded as an ordinal feature.
    #
    # Initial hypothesis:
    #   Children (<14)  -> more dependent on accompanying adults.
    #   Young adults    -> capable of acting independently, but potentially less experienced.
    #   Adults          -> expected to react more independently and take initiative.
    #   Elderly (60+)   -> potentially reduced mobility during evacuation.
    #
    # The ordinal encoding follows this intended ordering:
    #   Child   -> 0
    #   Elderly -> 1
    #   Young   -> 2
    #   Adult   -> 3
    #
    # The age boundaries and final ordinal encoding were refined through
    # experimentation and selected because they consistently produced the
    # best cross-validation performance.

    bins = [0, 14, 35, 60, 100]
    labels = ["0", "2", "3", "1"]

    df["Age_bin"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        right=False,
    )

    return df