from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

ZIP_PATH = RAW_DATA_DIR / "titanic.zip"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
GENDER_SUBMISSION_PATH = RAW_DATA_DIR / "gender_submission.csv"

# DEBUG
# print(PROJECT_ROOT)
# print(DATA_DIR)
# print(SUBMISSIONS_DIR)
# print(GENDER_SUBMISSION_PATH)
# print(TRAIN_PATH)
# print(TEST_PATH)
