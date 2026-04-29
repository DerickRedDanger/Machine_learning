import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

# Data Directory
RAW_DIR = ROOT_DIR/ 'data'/ 'raw'

RAW_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = RAW_DIR / "titanic.zip"


def download_data():
    print("Downloading Titanic dataset...")

    os.system(f"kaggle competitions download -c titanic -p {RAW_DIR}")

    print("Download complete.")


def unzip_data():
    print("Unzipping dataset...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(RAW_DIR)

    ZIP_PATH.unlink()  # delete zip file

    print("Unzip complete.")


if __name__ == "__main__":
    download_data()
    unzip_data()