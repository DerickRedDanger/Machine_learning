from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Data Directory
DATA_DIR = ROOT_DIR/ 'data'
PROCESSED_DATA_DIR = DATA_DIR/'processed'
RAW_DATA_DIR = DATA_DIR/'raw'
SUBMISSIONS_DATA_DIR = DATA_DIR/'submissions'

TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
SUBMISSION_EXAMPLE_PATH = RAW_DATA_DIR / "gender_submission.csv"

# Scripts Directory
SCRIPTS_DIR = ROOT_DIR/'scripts'

# Source Directory:
SRC_DIR=ROOT_DIR/'src'
# Continue

# print(SUBMISSIONS_DATA_DIR)

import os
print(os.environ["KAGGLE_USERNAME"])