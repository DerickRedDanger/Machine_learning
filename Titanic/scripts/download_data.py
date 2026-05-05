from pathlib import Path
import subprocess
import zipfile
from titanic_ml.paths import RAW_DATA_DIR, ZIP_PATH, TRAIN_PATH, TEST_PATH, GENDER_SUBMISSION_PATH


# ROOT_DIR = Path(__file__).resolve().parents[1]
# RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
# ZIP_PATH = RAW_DATA_DIR / "titanic.zip"

EXPECTED_FILES = [
    TRAIN_PATH,
    TEST_PATH,
    GENDER_SUBMISSION_PATH,
]


def titanic_data_exists() -> bool:
    return all(path.exists() for path in EXPECTED_FILES)


def download_titanic_data() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if titanic_data_exists():
        print("Titanic raw data already exists in data/raw/.")
        return

    try:
        print("Downloading Titanic dataset from Kaggle...")

        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                "titanic",
                "-p",
                str(RAW_DATA_DIR),
            ],
            check=True,
        )

        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)

        ZIP_PATH.unlink()

        print("Titanic data was successfully downloaded and extracted to data/raw/.")

    except FileNotFoundError:
        print(
            "\nKaggle CLI was not found.\n"
            "Install it with:\n\n"
            "    pip install kaggle\n\n"
            "Then follow the Kaggle API setup instructions in README.md.\n"
        )

    except subprocess.CalledProcessError:
        print(
            "\nCould not download the Titanic dataset from Kaggle.\n"
            "This usually means your Kaggle API token is missing or invalid.\n\n"
            "Please follow the Kaggle API setup instructions in README.md.\n"
        )

    except zipfile.BadZipFile:
        print(
            "\nThe downloaded file was not a valid zip file.\n"
            "Delete data/raw/titanic.zip and try again.\n"
        )


if __name__ == "__main__":
    download_titanic_data()