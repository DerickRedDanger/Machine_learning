import pandas as pd

from titanic_ml.paths import TRAIN_PATH, TEST_PATH


def load_train_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def load_test_data() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def split_features_target(
    df: pd.DataFrame,
    target_col: str = "Survived",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=target_col)
    y = df[target_col]

    return X, y