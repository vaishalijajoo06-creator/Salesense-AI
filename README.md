# SalesSense AI — Retail Sales Forecasting & Demand Planning

An end-to-end AI/ML internship project that predicts daily retail sales quantities
from five years of historical data, compares multiple forecasting models,
evaluates them on a held-out validation period, and presents results in an
interactive Streamlit dashboard.

---

## Problem Statement

Given five years of daily sales records across **10 stores** and **50 items** (500
store-item time series), predict daily sales quantities for the following 90 days.
The goal is to build a complete, production-quality forecasting pipeline with
correct feature engineering, leakage-free validation, and interactive reporting.

---

## Dataset

| File | Rows | Columns | Description |
|---|---|---|---|
| `data/train.csv` | 913,000 | `date, store, item, sales` | 2013-01-01 to 2017-12-31 |
| `data/test.csv` | 45,000 | `id, date, store, item` | 2018-01-01 to 2018-03-31 (no labels) |
| `data/sample_submission.csv` | 45,000 | `id, sales` | Submission format |

**Dataset statistics (computed from actual data):**
- Total training rows: 913,000
- Unique stores: 10 | Unique items: 50 | Unique series: 500
- Training date range: 2013-01-01 → 2017-12-31 (1,826 days)
- Total units sold (training): 47,704,512
- Mean daily sales per series: 52.25 units
- Sales range: 0 – 231 | Missing values: 0 | Duplicate rows: 0

---

## Project Structure

```
salesense-ai/
├── data/                         # Raw CSV files (never modified)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── src/
│   ├── data_loader.py            # CSV loading, validation, EDA statistics
│   ├── eda.py                    # Plotly chart functions
│   ├── features.py               # Feature engineering (lags, rolling, calendar)
│   ├── splitter.py               # Chronological train/val split
│   ├── evaluate.py               # MAE, RMSE, WAPE, SMAPE metrics
│   ├── forecast.py               # Recursive multi-step forecaster
│   ├── visualizations.py         # Reusable Plotly helpers (model outputs)
│   └── models/
│       ├── baseline.py           # NaiveSeasonal, SeasonalMean
│       ├── linear.py             # RidgeForecaster
│       ├── gradient_boost.py     # LGBMForecaster
│       └── sarima.py             # SARIMAForecaster (interface only, not trained)
├── tests/
│   ├── test_data_loader.py
│   ├── test_evaluate.py
│   ├── test_features.py
│   ├── test_forecast.py
│   └── test_models.py
├── outputs/                      # Generated artefacts (git-ignored contents)
│   ├── validation_metrics.csv    # Actual model evaluation results
│   └── val_predictions.csv       # Per-row predictions on validation set
├── app.py                        # Streamlit dashboard entry point
├── train_and_evaluate.py         # One-time training + evaluation script
├── requirements.txt
└── README.md
```

---

## Quick Start (Windows PowerShell)

### 1. Create and activate a virtual environment

```powershell
py -m venv .venv
.venv\Scripts\activate
```

> **Note:** Use `py` (Python Launcher), not `python` — the bare `python` command
> may be intercepted by the Windows Store on some machines. After activating the
> venv, `python` also works.

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

This installs all required packages (~400 MB, takes 1–3 minutes on first run).

### 3. Train the models and generate validation metrics

This step is required before the Model Comparison page works:

```powershell
py train_and_evaluate.py
```

Expected output (approx. 2 minutes total):
```
Stage 4 - Model Training & Evaluation
[1/5] Loading training data...   Loaded 913,000 rows
[2/5] Building features...       Feature matrix: (913000, 17)
[3/5] Splitting train/val...     Train: 867,000 | Val: 46,000
[4/5] Training models...         NaiveSeasonal, SeasonalMean, Ridge, LightGBM
[5/5] Computing metrics...
Best model (lowest WAPE): LightGBM  WAPE=10.87%
```

### 4. Launch the Streamlit dashboard

```powershell
streamlit run app.py
```

Opens `http://localhost:8501` in your browser automatically.

### 5. Run the test suite

```powershell
pytest tests/ -v
```

Expected: **148 passed, 0 failed, 0 skipped**.

---

## Feature Engineering

All features are computed in [`src/features.py`](src/features.py) using the
`build_features()` function, which is the single entry point for both training
and inference.

### Calendar features
| Feature | Description |
|---|---|
| `dayofweek` | 0 = Monday … 6 = Sunday |
| `dayofmonth` | Day of month (1–31) |
| `month` | Calendar month (1–12) |
| `year` | Calendar year |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |

### Lag features (per store-item series)
| Feature | Description |
|---|---|
| `lag_1` | Sales 1 day ago |
| `lag_7` | Sales 7 days ago (same weekday last week) |
| `lag_14` | Sales 14 days ago |
| `lag_28` | Sales 28 days ago (same weekday ~4 weeks ago) |

### Rolling features (per store-item series)
| Feature | Description |
|---|---|
| `rolling_mean_7` | 7-day rolling mean of sales (excludes today) |
| `rolling_mean_28` | 28-day rolling mean of sales (excludes today) |
| `rolling_std_7` | 7-day rolling std (NaN if < 2 observations) |
| `rolling_std_28` | 28-day rolling std (NaN if < 2 observations) |

**Leakage prevention:** Rolling statistics use `shift(1)` before `rolling(w)`,
so the window for day `t` covers only days `t-1` through `t-w`. No current-day
or future sales are included. Lag features use `groupby.shift(k)`, which looks
strictly backward. All computations are per `(store, item)` group — one series
never contaminates another.

**Missing history:** Early rows (first 28 days of each series) receive `NaN`
for lag/rolling features. These rows are dropped during model training
(`dropna()`) but are not silently filled with misleading values.

---

## Models Compared

All models are trained on the same training period and evaluated on the same
46,000-row validation set. The training script is [`train_and_evaluate.py`](train_and_evaluate.py).

| Model | Module | Description |
|---|---|---|
| **NaiveSeasonal** | `src/models/baseline.py` | Predicts `lag_7` (same weekday last week) |
| **SeasonalMean** | `src/models/baseline.py` | Per-(store, item, dayofweek) historical mean |
| **Ridge** | `src/models/linear.py` | Global Ridge regression with all 13 features + StandardScaler |
| **LightGBM** | `src/models/gradient_boost.py` | Global LightGBM regressor (500 trees, lr=0.05, leaves=31) |
| ~~SARIMA~~ | `src/models/sarima.py` | *Not trained* — fitting 500 per-series ARIMA models is impractical (~hours); see module docstring |

---

## Validation Methodology

### Data split (chronological — never shuffled)

| Split | Date Range | Rows |
|---|---|---|
| **Training** | 2013-01-01 → 2017-09-30 | 867,000 |
| **Validation** | 2017-10-01 → 2017-12-31 | 46,000 |
| **Test horizon** | 2018-01-01 → 2018-03-31 | 45,000 (no labels) |

The 3-month validation window mirrors the 3-month test horizon, making
validation performance a reliable proxy for generalisation.

### Evaluation metrics

| Metric | Formula | Notes |
|---|---|---|
| **WAPE** (primary) | `sum(|y-ŷ|) / sum(|y|) × 100` | Robust to single zero-sales rows; lower is better |
| **MAE** | `mean(|y-ŷ|)` | Interpretable in units/day |
| **RMSE** | `sqrt(mean((y-ŷ)²))` | Penalises large errors more |
| **SMAPE** | `mean(|y-ŷ| / ((|y|+|ŷ|)/2)) × 100` | Competition metric; kept for reference |

### Verified validation results

The metrics below are computed from the saved `outputs/val_predictions.csv`
and independently recalculated from scratch during Stage 6 verification.
No values are fabricated or hardcoded.

| Rank | Model | MAE | RMSE | WAPE (%) | SMAPE (%) |
|---|---|---|---|---|---|
| 🥇 1 | **LightGBM** | **5.9413** | **7.6889** | **10.87** | **12.51** |
| 2 | Ridge | 7.0361 | 9.2449 | 12.87 | 14.89 |
| 3 | SeasonalMean | 8.9675 | 11.5746 | 16.40 | 17.64 |
| 4 | NaiveSeasonal | 9.0763 | 12.0507 | 16.60 | 18.56 |

**Selected model: LightGBM** — lowest WAPE and MAE across all metrics.
LightGBM delivers a **5.7 percentage-point WAPE improvement** over the
NaiveSeasonal baseline.

---

## Forecasting Workflow

Future forecasts are produced by [`src/forecast.py`](src/forecast.py) using
`forecast_series()`. The algorithm:

1. Extract the last 38 days of known history for the selected (store, item).
2. For each future step `t`:
   a. Append a placeholder row with `sales = NaN`.
   b. Compute calendar, lag, and rolling features from the combined tail.
   c. Predict `sales[t]` using LightGBM.
   d. Clip to 0 (sales cannot be negative).
   e. Append the **prediction** (not any actual value) to the tail.
3. Repeat for the full horizon (7, 30, or 90 days).

**Leakage guarantee:** Actual future sales values are never read. The tail
that accumulates predictions is entirely derived from historical actuals
and prior predictions.

**Known limitation:** Recursive forecasting accumulates error over longer
horizons. The 90-day forecast is less accurate than the 7-day forecast.
Direct multi-output models (e.g. XGBoost with horizon-specific targets)
would improve long-horizon accuracy.

---

## Dashboard Pages

| Page | Description |
|---|---|
| **Home** | Project overview, stage progress, dataset file status |
| **Data Explorer** | Filter by store/item, view raw time series, dataset summary metrics |
| **EDA** | 9 interactive Plotly charts: trend, seasonality, distribution, store/item comparisons, heatmap |
| **Model Comparison** | Metrics table, bar chart, actual vs predicted overlay, error distributions, model rationale |
| **Forecast Viewer** | 7/30/90-day recursive forecast, historical context chart, summary metrics, CSV download |

---

## Test Suite

```
pytest tests/ -v
```

| Test file | Tests | Covers |
|---|---|---|
| `test_data_loader.py` | 19 | CSV schema, nulls, date range, dataset_summary, compute_eda_stats |
| `test_evaluate.py` | 17 | MAE, RMSE, WAPE, SMAPE formulas; compare_models alignment |
| `test_features.py` | 47 | Calendar, lag, rolling features; leakage prevention; output shape |
| `test_forecast.py` | 27 | Horizon correctness, date ranges, leakage, non-negative, CSV output |
| `test_models.py` | 38 | Splitter, baselines, Ridge, LightGBM fit/predict contracts |
| **Total** | **148** | **0 failures, 0 skipped** |

---

## Technology Stack

| Library | Version | Role |
|---|---|---|
| Python | 3.14.5 | Runtime |
| pandas | 3.0.6 | Data wrangling |
| numpy | 2.5.4 | Numerical operations |
| scikit-learn | 1.9.1 | Ridge regression, StandardScaler |
| lightgbm | 4.7.0 | Gradient boosting model |
| statsmodels | 0.15.0 | SARIMA (interface only) |
| plotly | 7.1.0 | Interactive charts |
| streamlit | 1.65.0 | Dashboard UI |
| pytest | 9.1.1 | Unit testing |

---

## Known Limitations and Future Improvements

| Limitation | Suggested Improvement |
|---|---|
| Recursive error accumulation over long horizons | Use direct multi-output forecasting or quantile regression |
| No confidence / prediction intervals | Add bootstrap or conformal prediction intervals |
| LightGBM not tuned | Add Optuna hyperparameter search |
| SARIMA not implemented | Use per-series SARIMA on a subset of series for comparison |
| First Forecast Viewer load takes ~30s | Pre-train and cache the model to disk |
| No cross-validation | Implement walk-forward (time-series) cross-validation |
| 913K-row feature build takes ~5s | Profile and parallelise with Numba or Polars |
