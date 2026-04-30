## Dataset setup

This project uses the Titanic dataset from Kaggle's Titanic competition.

The dataset files are **not included in this repository**.  
To run the project, you need to download them from Kaggle using your own Kaggle account and API credentials.

Expected files:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/gender_submission.csv

### Dataset Download steps

1 - Install the Kaggle CLI by running in the terminal:
pip install kaggle

2 - Create a Kaggle API token:
Go to Kaggle.
Open your account settings.
Go to the API section.
Create an API token.

3 - Kaggle credentials:
Kaggle supports multiple authentication methods. The simplest method for this project is the legacy kaggle.json file or using a environment variable.

Option A - environment variable:
Execute the following in the terminal, replacing Your_token_here with the one given by Kaggle:

export KAGGLE_API_TOKEN=your_token_here

Option B - legacy kaggle.json:
Place your kaggle.json file in the Kaggle config folder.

For Linux/macOS/Codespaces, execute the following in the terminal:

mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

For Windows PowerShell:
mkdir $env:USERPROFILE\.kaggle

Then move kaggle.json into:
C:\Users\YOUR_USERNAME\.kaggle\kaggle.json

python scripts/download_data.py

4 - Download the Titanic dataset:
run:
python scripts/download_data.py

Or manually go to scripts and execute it.

The script will:

Check whether the Titanic files already exist in data/raw/.
Download the dataset from Kaggle if the files are missing.
Extract the CSV files.
Print setup instructions if Kaggle authentication is missing.