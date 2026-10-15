"""
train_and_evaluate.py
---------------------
Runs Stage 4: builds features, splits chronologically, trains models,
evaluates on the validation set, and saves results to outputs/.

Results written:
  outputs/validation_metrics.csv   -- comparison table
  outputs/val_predictions.csv      -- all model predictions on val set

Run with:
  .venv\\Scripts\\python.exe train_and_evaluate.py
"""

import pathlib, sys, json, time
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import pandas as pd

from src.data_loader import load_train
from src.features   import build_features
from src.splitter   import split_train_val
from src.evaluate   import compare_models, mae, rmse, wape

OUTPUT_DIR = pathlib.Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("Stage 4 — Model Training & Evaluation")
print("=" * 60)

# ── 1. Load and build features ───────────────────────────────────────────────
print("\n[1/5] Loading training data...")
t0 = time.time()
raw = load_train()
print(f"  Loaded {len(raw):,} rows in {time.time()-t0:.1f}s")

print("\n[2/5] Building features (lags, rolling, calendar)...")
t0 = time.time()
feat = build_features(raw)
print(f"  Feature matrix: {feat.shape}  ({time.time()-t0:.1f}s)")

# ── 2. Chronological split ───────────────────────────────────────────────────
print("\n[3/5] Splitting train / validation...")
train_df, val_df = split_train_val(feat)
print(f"  Train : {len(train_df):,} rows  ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
print(f"  Val   : {len(val_df):,} rows   ({val_df['date'].min().date()} to {val_df['date'].max().date()})")

y_val = val_df["sales"].values.astype(float)

# ── 3. Train and predict — Seasonal Naive ────────────────────────────────────
print("\n[4/5] Training models...")

from src.models.baseline import NaiveSeasonal, SeasonalMean
from src.models.linear   import RidgeForecaster
from src.models.gradient_boost import LGBMForecaster

# — Seasonal Naive —
t0 = time.time()
naive = NaiveSeasonal()
naive.fit(train_df)
pred_naive = naive.predict(val_df)
print(f"  NaiveSeasonal   done  ({time.time()-t0:.1f}s)")

# — Seasonal Mean —
t0 = time.time()
smean = SeasonalMean()
smean.fit(train_df)
pred_smean = smean.predict(val_df)
print(f"  SeasonalMean    done  ({time.time()-t0:.1f}s)")

# — Ridge —
t0 = time.time()
ridge = RidgeForecaster(alpha=1.0)
ridge.fit(train_df)
pred_ridge = ridge.predict(val_df)
print(f"  Ridge           done  ({time.time()-t0:.1f}s)")

# — LightGBM —
t0 = time.time()
lgbm = LGBMForecaster(n_estimators=500, learning_rate=0.05, num_leaves=31)
lgbm.fit(train_df)
pred_lgbm = lgbm.predict(val_df)
print(f"  LightGBM        done  ({time.time()-t0:.1f}s)")

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
print("\n[5/5] Computing validation metrics...")

results = {
    "NaiveSeasonal": {"y_true": y_val, "y_pred": pred_naive},
    "SeasonalMean":  {"y_true": y_val, "y_pred": pred_smean},
    "Ridge":         {"y_true": y_val, "y_pred": pred_ridge},
    "LightGBM":      {"y_true": y_val, "y_pred": pred_lgbm},
}

metrics_df = compare_models(results)
print("\n" + metrics_df.to_string(index=False))

# ── 5. Save results ───────────────────────────────────────────────────────────
metrics_df.to_csv(OUTPUT_DIR / "validation_metrics.csv", index=False)
print(f"\nMetrics saved: {OUTPUT_DIR / 'validation_metrics.csv'}")

val_preds = val_df[["date", "store", "item", "sales"]].copy()
val_preds["NaiveSeasonal"] = pred_naive
val_preds["SeasonalMean"]  = pred_smean
val_preds["Ridge"]         = pred_ridge
val_preds["LightGBM"]      = pred_lgbm
val_preds.to_csv(OUTPUT_DIR / "val_predictions.csv", index=False)
print(f"Predictions saved: {OUTPUT_DIR / 'val_predictions.csv'}")

# ── 6. Best model summary ─────────────────────────────────────────────────────
best = metrics_df.iloc[0]
print(f"\n{'='*60}")
print(f"Best model (lowest WAPE): {best['model']}")
print(f"  MAE={best['MAE']:.4f}  RMSE={best['RMSE']:.4f}  WAPE={best['WAPE']:.4f}%  SMAPE={best['SMAPE']:.4f}")
print("=" * 60)
