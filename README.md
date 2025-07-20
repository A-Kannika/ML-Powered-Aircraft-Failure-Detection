# ML-Powered Aircraft Failure Prediction (Project is in progress)

A machine learning project that predicts Remaining Useful Life (RUL) of aircraft engines using NASA’s CMAPSS dataset. I compare traditional models (Random Forest, XGBoost) with LSTM deep learning and provide a real-time demo using Streamlit.

---

## Project Overview

1. **Preprocessing** (`load_data.py`)
   - Reads NASA CMAPSS FD\* datasets.
   - Computes Remaining Useful Life (RUL) and labels.
   - Normalizes sensors per operating condition.
   - Saves cleaned CSVs to `CMAPSSData`.

2. **Modeling** (`model.py`)
   - Generates overlapping sequences for each engine.
   - Trains and evaluates:
     - Random Forest
     - XGBoost
     - LSTM (with early stopping and sequence input)
   - Visualizes predicted vs actual RUL for LSTM.

3. **Real-Time Demo** (`streamlit_app.py`, optional)
   - User inputs sensor values or uploads data.
   - Model predicts RUL live.
   - Built with Streamlit for easy visualization.

---

## Getting Started

### Dataset
NASA Commercial Modular Aero-Propulsion System Simulation (CMAPSS) datasets:

Turbofan Engine Degradation Simulation
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

### Prerequisites

- Python 3.10+
- `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`, `keras`, `matplotlib`
- (Optional) `streamlit`, `joblib`

```bash
pip install pandas numpy scikit-learn xgboost tensorflow keras matplotlib streamlit joblib
```

### Preprocess Data

```bash
python load_data.py
```

### Train & Evaluate Models

```bash
python model.py
```

- Trains Random Forest, XGBoost, and LSTM models.
- Prints MAE, RMSE, and R² for each.
- Shows chart for LSTM’s prediction vs. actual.

---

### Citation / Resources
- Dataset: NASA CMAPSS datasets
- Core paper: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run‑to‑Failure Simulation", PHM08.
- Inspired by projects like archd3sai/Predictive‑Maintenance‑of‑Aircraft‑Engine




