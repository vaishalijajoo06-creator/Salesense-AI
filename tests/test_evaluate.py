"""
test_evaluate.py
----------------
Unit tests for src/evaluate.py metric functions.
These are pure numerical functions with no file dependencies.
"""

from __future__ import annotations

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# smape
# ---------------------------------------------------------------------------

def test_smape_perfect_prediction():
    """SMAPE must be 0 when predictions exactly equal actuals."""
    from src.evaluate import smape
    y = np.array([10.0, 20.0, 30.0])
    assert smape(y, y) == pytest.approx(0.0)


def test_smape_symmetric():
    """SMAPE(a, b) must equal SMAPE(b, a)."""
    from src.evaluate import smape
    a = np.array([10.0, 15.0, 25.0])
    b = np.array([12.0, 13.0, 30.0])
    assert smape(a, b) == pytest.approx(smape(b, a))


def test_smape_both_zero():
    """SMAPE must be 0 when both actual and predicted are 0."""
    from src.evaluate import smape
    assert smape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == pytest.approx(0.0)


def test_smape_non_negative():
    """SMAPE must always be >= 0."""
    from src.evaluate import smape
    rng = np.random.default_rng(42)
    y_true = rng.integers(1, 100, 50).astype(float)
    y_pred = rng.integers(1, 100, 50).astype(float)
    assert smape(y_true, y_pred) >= 0.0


def test_smape_scale():
    """SMAPE is on the 0–200 scale."""
    from src.evaluate import smape
    # Predict 0 for actual=1 → each row: |1-0|/((1+0)/2)*100 = 200
    assert smape(np.array([1.0]), np.array([0.0])) == pytest.approx(200.0)


def test_smape_known_value():
    """Manual calculation: actual=[10], pred=[20] → SMAPE = 200*10/30 ≈ 66.667."""
    from src.evaluate import smape
    result = smape(np.array([10.0]), np.array([20.0]))
    assert result == pytest.approx(200 * 10 / 30, rel=1e-5)


# ---------------------------------------------------------------------------
# rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect_prediction():
    from src.evaluate import rmse
    y = np.array([5.0, 10.0, 15.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_non_negative():
    from src.evaluate import rmse
    rng = np.random.default_rng(0)
    y_t = rng.integers(1, 50, 30).astype(float)
    y_p = rng.integers(1, 50, 30).astype(float)
    assert rmse(y_t, y_p) >= 0.0


def test_rmse_known_value():
    """RMSE([0,0],[3,4]) = sqrt((9+16)/2) = sqrt(12.5)."""
    from src.evaluate import rmse
    result = rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0]))
    assert result == pytest.approx(math.sqrt(12.5))


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

def test_mae_perfect_prediction():
    from src.evaluate import mae
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == pytest.approx(0.0)


def test_mae_positive():
    """MAE must always be >= 0."""
    from src.evaluate import mae
    rng = np.random.default_rng(7)
    y_t = rng.integers(1, 100, 40).astype(float)
    y_p = rng.integers(1, 100, 40).astype(float)
    assert mae(y_t, y_p) >= 0.0


def test_mae_known_value():
    """MAE([1,2,3],[4,5,6]) = mean([3,3,3]) = 3."""
    from src.evaluate import mae
    result = mae(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
    assert result == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

def test_compare_models_returns_dataframe():
    """compare_models must return a DataFrame with a 'model' column."""
    import pandas as pd
    from src.evaluate import compare_models
    results = {
        "ModelA": {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([11.0, 19.0])},
        "ModelB": {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([5.0, 25.0])},
    }
    df = compare_models(results)
    assert isinstance(df, pd.DataFrame)
    assert "model" in df.columns


def test_compare_models_sorted_by_wape():
    """compare_models result must be sorted ascending by WAPE (primary metric)."""
    from src.evaluate import compare_models
    results = {
        "Good": {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([10.5, 19.5])},
        "Bad":  {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([1.0,  100.0])},
    }
    df = compare_models(results)
    wape_vals = df["WAPE"].tolist()
    assert wape_vals == sorted(wape_vals)


def test_compare_models_columns():
    from src.evaluate import compare_models
    results = {
        "M": {"y_true": np.array([5.0]), "y_pred": np.array([5.0])},
    }
    df = compare_models(results)
    for col in ("model", "MAE", "RMSE", "WAPE", "SMAPE"):
        assert col in df.columns


def test_compare_models_model_column_aligned_with_metrics():
    """Regression: model name in each row must match the metrics for that row.

    Guards against a column-shift bug where model names and numeric values
    become misaligned after sorting (e.g. caused by .style usage breaking
    DataFrame column order in the display layer).
    """
    from src.evaluate import compare_models, wape as wape_fn
    results = {
        "Bad":  {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([1.0, 40.0])},
        "Good": {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([10.5, 19.5])},
    }
    df = compare_models(results)
    # After WAPE-sort, first row must be "Good" with a lower WAPE
    assert df.iloc[0]["model"] == "Good", (
        f"Row 0 model should be 'Good' (lowest WAPE) but got '{df.iloc[0]['model']}'"
    )
    assert df.iloc[1]["model"] == "Bad", (
        f"Row 1 model should be 'Bad' but got '{df.iloc[1]['model']}'"
    )
    # The WAPE value in row 0 must actually equal wape(Good's actuals, Good's preds)
    expected_wape_good = wape_fn(np.array([10.0, 20.0]), np.array([10.5, 19.5]))
    assert df.iloc[0]["WAPE"] == pytest.approx(expected_wape_good, rel=1e-4), (
        f"Row 0 WAPE {df.iloc[0]['WAPE']} != expected {expected_wape_good:.4f}. "
        "Model name and metric values are misaligned."
    )
    # The WAPE value in row 1 must equal wape(Bad's actuals, Bad's preds)
    expected_wape_bad = wape_fn(np.array([10.0, 20.0]), np.array([1.0, 40.0]))
    assert df.iloc[1]["WAPE"] == pytest.approx(expected_wape_bad, rel=1e-4), (
        f"Row 1 WAPE {df.iloc[1]['WAPE']} != expected {expected_wape_bad:.4f}. "
        "Model name and metric values are misaligned."
    )


def test_compare_models_column_order():
    """Column order must be: model, MAE, RMSE, WAPE, SMAPE (no unexpected inserts)."""
    from src.evaluate import compare_models
    df = compare_models({"M": {"y_true": np.array([5.0]), "y_pred": np.array([5.0])}})
    assert list(df.columns) == ["model", "MAE", "RMSE", "WAPE", "SMAPE"]

