from pathlib import Path
import shutil


def check_package_import():
    try:
        import titanic_ml
        print("Package import: OK")
    except ImportError:
        print("Package import: FAILED")
        print("Run: pip install -e .")


def check_data():
    expected_files = [
        Path("data/raw/train.csv"),
        Path("data/raw/test.csv"),
        Path("data/raw/gender_submission.csv"),
    ]

    missing = [path for path in expected_files if not path.exists()]

    if not missing:
        print("Data files: OK")
    else:
        print("Data files: MISSING")
        for path in missing:
            print(f"  - {path}")
        print("Run: python scripts/download_data.py")


def check_kaggle_cli():
    if shutil.which("kaggle"):
        print("Kaggle CLI: OK")
    else:
        print("Kaggle CLI: NOT FOUND")
        print("Run: pip install kaggle")


if __name__ == "__main__":
    check_package_import()
    check_data()
    check_kaggle_cli()