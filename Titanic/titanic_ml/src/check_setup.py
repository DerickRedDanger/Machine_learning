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
    import titanic_ml.paths as paths
    expected_files = [
        paths.TRAIN_PATH,
        paths.TEST_PATH,
        paths.GENDER_SUBMISSION_PATH,
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

# DEBUG
