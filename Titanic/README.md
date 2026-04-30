## Dataset Setup

This project uses the Titanic dataset from Kaggle's Titanic competition.

The dataset files are **not included in this repository**.  
To run the project, you must download them using your own Kaggle account.

### Expected files

```text
data/raw/train.csv
data/raw/test.csv
data/raw/gender_submission.csv
````

---

## Step 1 — Install Kaggle CLI

```bash
pip install kaggle
```

---

## Step 2 — Create a Kaggle API token

1. Go to Kaggle
2. Open **Account Settings**
3. Scroll to the **API** section
4. Click **Create New Token**

This will download a `kaggle.json` file.

---

## Step 3 — Configure credentials

You can choose one of the following methods:

### Option A — `kaggle.json` (recommended for beginners)

#### Linux/macOS/Codespaces

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

#### Windows (PowerShell)

```powershell
mkdir $env:USERPROFILE\.kaggle
```

Move `kaggle.json` to:

```text
C:\Users\YOUR_USERNAME\.kaggle\kaggle.json
```

---

### Option B — Environment variable

```bash
export KAGGLE_API_TOKEN=your_token_here
```

PowerShell:

```powershell
$env:KAGGLE_API_TOKEN="your_token_here"
```

---

## Step 4 — Download the dataset

Run:

```bash
python scripts/download_data.py
```

---

## What the script does

* Checks if the dataset already exists
* Downloads it from Kaggle if needed
* Extracts the files into `data/raw/`
* Shows helpful messages if setup is missing

---

## Security note

Never commit your Kaggle credentials.

Make sure your `.gitignore` includes:

```gitignore
data/
kaggle.json
.env
```