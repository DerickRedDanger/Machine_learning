import os
import zipfile
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATA_DIR / "titanic.zip"


def download_data():
    print("Downloading Titanic dataset...")

    os.system(f"kaggle competitions download -c titanic -p {DATA_DIR}")

    print("Download complete.")


def unzip_data():
    print("Unzipping dataset...")

    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)

    ZIP_PATH.unlink()  # delete zip file

    print("Unzip complete.")


if __name__ == "__main__":
    download_data()
    unzip_data()