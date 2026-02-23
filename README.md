# MLPR Dashboard

This project is a Streamlit dashboard for:
- Defect-proneness classification (`defect_prone`)
- Next-release effort prediction (`effort_next_lines_added`)

The app uses `final_dataset_6000.csv` and models defined in `model_service.py`.

## Project structure

- `app.py`: Streamlit UI and prediction workflow
- `model_service.py`: model training/loading and prediction logic
- `requirements.txt`: Python dependencies
- `final_dataset_6000.csv`: main dataset used by the app
- `artifacts/`: serialized trained models and metadata
- `214250J.ipynb`: exploratory analysis and model development notebook
- `metadata.pdf`: data description for each field in the dataset

## Prerequisites

- Python 3.10+ (3.11 recommended)
- `pip`

## Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux (bash/zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in terminal (usually `http://localhost:8501`).

## Notes

- If `artifacts/models.joblib` and `artifacts/metadata.json` do not exist or are incompatible, the app retrains models automatically.
- Batch prediction is available via CSV upload in the app UI.
