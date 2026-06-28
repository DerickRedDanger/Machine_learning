from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# print(f"Project root: {PROJECT_ROOT}")

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENT_RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"
EXPERIMENT_CONFIGS_FILE = RESULTS_DIR / "experiments_used_config.json"
EXPERIMENT_FEATURE_EFFECT = RESULTS_DIR / "experiments_feature_effect.json"

# print(f"Results directory: {RESULTS_DIR}")
# print(f"Experiment results file: {EXPERIMENT_RESULTS_FILE}")
# print(f"Experiment configs file: {EXPERIMENT_CONFIGS_FILE}")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SUBMISSIONS_DIR = DATA_DIR / "submissions"

ZIP_PATH = RAW_DATA_DIR / "titanic.zip"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
GENDER_SUBMISSION_PATH = RAW_DATA_DIR / "gender_submission.csv"

TITANIC_ML_ROOT = PROJECT_ROOT / "titanic_ml"

# DEBUG
# print(PROJECT_ROOT)
# print(DATA_DIR)
# print(SUBMISSIONS_DIR)
# print(GENDER_SUBMISSION_PATH)
# print(TRAIN_PATH)
# print(TEST_PATH)
