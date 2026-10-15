"""
evaluate.py
-----------
Forecasting evaluation metrics and model-comparison utilities.

Metrics
-------
MAE   : Mean Absolute Error
RMSE  : Root Mean Squared Error
WAPE  : Weighted Absolute Percentage Error
         = sum(|y-ŷ|) / sum(|y|)  × 100
         Denominator is sum of actuals, so zero-sales rows do not
         individually blow up the metric; the sum is never zero on
         a full validation window with positive sales.
SMAPE : Symmetric MAPE (kept for reference / competition compatibility)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (0–100+ scale, lower is better).

    WAPE = sum(|y - ŷ|) / sum(|y|) × 100

    Zero-denominator handling: if sum(|y|) == 0 (all actuals are zero)
    the function returns 0.0 if predictions are also all zero, else 100.0.
    This is conservative and avoids a silent division-by-zero.

    Parameters
    ----------
    y_true : array-like of actual values
    y_pred : array-like of predicted values

    Returns
    -------
    float  WAPE (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0.0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0.0 else 100.0
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0–200 scale)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    safe_denom = np.where(mask, denominator, 1.0)
    scores = np.where(mask, np.abs(y_true - y_pred) / safe_denom, 0.0)
    return float(np.mean(scores) * 100)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a dict of model result dicts.

    Parameters
    ----------
    results : dict
        Keys are model names.  Values are dicts with keys:
          ``y_true`` – array-like of actual values
          ``y_pred`` – array-like of predicted values

    Returns
    -------
    pd.DataFrame  columns: model, MAE, RMSE, WAPE, SMAPE
                  sorted ascending by WAPE (primary ranking metric).
    """
    rows = []
    for model_name, data in results.items():
        yt = data["y_true"]
        yp = data["y_pred"]
        rows.append({
            "model": model_name,
            "MAE":   round(mae(yt, yp),   4),
            "RMSE":  round(rmse(yt, yp),  4),
            "WAPE":  round(wape(yt, yp),  4),
            "SMAPE": round(smape(yt, yp), 4),
        })
    return pd.DataFrame(rows).sort_values("WAPE").reset_index(drop=True)
