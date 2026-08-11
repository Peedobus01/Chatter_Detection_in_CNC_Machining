# Chatter Detection in CNC Machining

A data science project focused on detecting and predicting chatter vibrations in CNC machining processes using machine learning techniques.

## Project Structure

```
Chatter Detection in CNC Machining/
│
├── data/
│   ├── raw/              # Original, unmodified sensor/experimental data
│   └── processed/        # Cleaned and feature-engineered datasets
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb       # Data loading, cleaning & EDA
│   ├── 02_xgboost_optimization.ipynb     # XGBoost model training & tuning
│   ├── 03_1d_cnn_pytorch.ipynb           # 1D CNN for time-series classification
│   └── 04_stability_lobe_diagrams.ipynb  # Stability lobe diagram analysis
│
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## Notebooks Overview

| Notebook | Description |
|---|---|
| `01_data_preprocessing` | Signal loading, noise filtering, feature extraction from vibration/acoustic data |
| `02_xgboost_optimization` | Gradient boosting classifier with hyperparameter optimization (Optuna/GridSearch) |
| `03_1d_cnn_pytorch` | Deep learning approach using 1D Convolutional Neural Networks on raw time-series |
| `04_stability_lobe_diagrams` | Analytical generation and visualization of stability lobe diagrams (SLD) |

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Place raw data** in `data/raw/`

3. **Run notebooks in order**, starting with `01_data_preprocessing.ipynb`

## Key Techniques

- Vibration signal processing (FFT, STFT, wavelets)
- Feature engineering from time & frequency domains
- XGBoost classification with hyperparameter tuning
- 1D CNN for raw time-series chatter detection
- Stability Lobe Diagram (SLD) generation
