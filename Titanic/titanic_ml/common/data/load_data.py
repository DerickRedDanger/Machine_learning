import pandas as pd

from titanic_ml.paths import TRAIN_PATH, TEST_PATH


def load_train_data():
    return pd.read_csv(TRAIN_PATH)


def load_test_data():
    return pd.read_csv(TEST_PATH)
