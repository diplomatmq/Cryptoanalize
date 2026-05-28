# Wallet Type Classification Demo (BABD-13)

Solovej Egor

This project demonstrates wallet type classification using the BABD-13 feature set and a Random Forest model. It includes a Streamlit UI, data prep scripts, training, and report figure generation.

## Requirements
- Python 3.9+
- pip
- Optional: `python-docx` for the report builder script
- Optional: access to mempool.space for BTC feature collection

## Setup
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
# Optional (report builder)
pip install python-docx
```

## Quick Start (Streamlit)
1. Ensure the dataset exists at `data/babd13_wallet4.csv`.
2. Train the model if `models/babd13_wallet_type_model.joblib` is missing:
```bash
python scripts/train_wallet_classifier.py --dataset data/babd13_wallet4.csv --model models/babd13_wallet_type_model.joblib
```
3. Run the demo:
```bash
streamlit run streamlit_wallet_demo.py
```

## Optional: Wallet Lookup Tab
The Streamlit app includes a wallet lookup tab that relies on an `app/` package from the full project and API keys loaded from `.env`.
- Ensure the `app/` package is available in the repo root or the parent folder.
- Provide `.env` with the required API keys for the explorer client (see the full project settings).
- BTC lookup also needs `models/babd13_btc_wallet_model.joblib` (see below).

## Data Preparation
Prepare the 4-class subset from BABD-13 (requires the original `BABD-13.csv` file):
```bash
python scripts/prepare_babd13_dataset.py --input BABD-13.csv --output data/babd13_wallet4.csv
```

## Train a Model
```bash
python scripts/train_wallet_classifier.py --dataset data/babd13_wallet4.csv --model models/babd13_wallet_type_model.joblib
```

## Generate Report Figures
```bash
python scripts/generate_report_figures.py --dataset data/babd13_wallet4.csv --model models/babd13_wallet_type_model.joblib --outdir outputs/figures
```

## Optional: BTC Dataset + Model
This step uses the mempool.space public API and can take time.
```bash
python scripts/prepare_babd13_btc_dataset.py --input BABD-13.csv --out data/babd13_btc_features.csv
python scripts/train_wallet_classifier.py --dataset data/babd13_btc_features.csv --model models/babd13_btc_wallet_model.joblib
```

## Scripts Overview
- `streamlit_wallet_demo.py`: Streamlit UI for metrics, predictions, graphs, and wallet lookup.
- `scripts/prepare_babd13_dataset.py`: Build a 4-class subset from BABD-13.
- `scripts/prepare_babd13_btc_dataset.py`: Collect BTC features from mempool.space.
- `scripts/train_wallet_classifier.py`: Train a Random Forest classifier.
- `scripts/generate_report_figures.py`: Export figures into `outputs/figures`.
- `scripts/build_course_note_egor_gost.py`: Build a DOCX report (requires `python-docx`).
- `scripts/btc_wallet_features.py`: BTC feature extraction helpers.

## Notes
- `models/` and `outputs/` are generated artifacts and are ignored by git. Recreate them locally using the scripts above.
