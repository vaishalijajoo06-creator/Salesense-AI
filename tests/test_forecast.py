"""
test_forecast.py
----------------
Tests for src/forecast.py — Stage 5.

Categories
----------
A. forecast_series  – output shape, date range, non-negative values,
                      correct horizon, leakage prevention, error handling
B. _build_one_row_features – lag correctness, no future sales used
C. generate_submission – output schema, row count, non-negative, CSV write
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal stub model that returns a fixed constant
# ---------------------------------------------------------------------------

class _ConstModel:
    """Returns a constant prediction for every row."""
    def __init__(self, value: float = 10.0):
        self._value = value
        # Mimic the feature-col list that real models need
        from src.features import FEATURE_COLS
        self._feature_cols = ["store", "item"] + FEATURE_COLS

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._value)


class _SeriesModel:
    """Returns predictions equal to the lag_7 feature (seasonal naive)."""
    def __init__(self):
        from src.features import FEATURE_COLS
        self._feature_cols = ["store", "item"] + FEATURE_COLS

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if "lag_7" in df.columns:
            vals = df["lag_7"].fillna(10.0).values.astype(float)
        else:
            vals = np.full(len(df), 10.0)
        return np.clip(vals, 0, None)


# ---------------------------------------------------------------------------
# Shared history fixture
# ---------------------------------------------------------------------------

def _make_history(n: int = 60, store: int = 1, item: int = 1,
                  sales_start: int = 10) -> pd.DataFrame:
    """n days of history for a single (store, item) pair."""
    dates = pd.date_range("2017-10-01", periods=n, freq="D")
    return pd.DataFrame({
        "date":  dates,
        "store": pd.array([store] * n, dtype="int16"),
        "item":  pd.array([item]  * n, dtype="int16"),
        "sales": pd.array(range(sales_start, sales_start + n), dtype="int32"),
    })


# ===========================================================================
# A. forecast_series
# ===========================================================================

class TestForecastSeries:

    def test_output_row_count_equals_horizon(self):
        from src.forecast import forecast_series
        hist = _make_history(60)
        for h in [7, 30, 90]:
            result = forecast_series(_ConstModel(), hist, 1, 1, h)
            assert len(result) == h, f"Expected {h} rows, got {len(result)}"

    def test_output_columns(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(60), 1, 1, 7)
        for col in ("date", "store", "item", "predicted_sales"):
            assert col in result.columns

    def test_dates_are_consecutive(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(60), 1, 1, 30)
        dates = result["date"].sort_values().reset_index(drop=True)
        deltas = (dates.diff().dropna()).dt.days
        assert (deltas == 1).all(), "Forecast dates are not consecutive daily"

    def test_forecast_starts_day_after_origin(self):
        from src.forecast import forecast_series
        hist = _make_history(60)
        origin = hist["date"].max()
        result = forecast_series(_ConstModel(), hist, 1, 1, 7)
        assert result["date"].min() == origin + pd.Timedelta(days=1)

    def test_forecast_ends_at_correct_date(self):
        from src.forecast import forecast_series
        hist = _make_history(60)
        origin = hist["date"].max()
        result = forecast_series(_ConstModel(), hist, 1, 1, 30)
        assert result["date"].max() == origin + pd.Timedelta(days=30)

    def test_predictions_are_non_negative(self):
        """Even if the model returns negative values, clipping must apply."""
        from src.forecast import forecast_series

        class _NegModel:
            _feature_cols = ["store", "item"]
            def predict(self, df):
                return np.full(len(df), -5.0)

        result = forecast_series(_NegModel(), _make_history(60), 1, 1, 14)
        assert (result["predicted_sales"] >= 0).all()

    def test_constant_model_output(self):
        """A model that always returns 42 must yield 42.0 for all steps."""
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(42.0), _make_history(60), 1, 1, 7)
        for v in result["predicted_sales"]:
            assert v == pytest.approx(42.0)

    def test_store_item_columns_correct(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(60, store=3, item=7), 3, 7, 10)
        assert (result["store"] == 3).all()
        assert (result["item"] == 7).all()

    def test_horizon_7(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(60), 1, 1, 7)
        assert len(result) == 7

    def test_horizon_30(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(60), 1, 1, 30)
        assert len(result) == 30

    def test_horizon_90(self):
        from src.forecast import forecast_series
        result = forecast_series(_ConstModel(), _make_history(90), 1, 1, 90)
        assert len(result) == 90

    def test_invalid_horizon_zero_raises(self):
        from src.forecast import forecast_series
        with pytest.raises(ValueError, match="horizon_days"):
            forecast_series(_ConstModel(), _make_history(60), 1, 1, 0)

    def test_invalid_horizon_too_large_raises(self):
        from src.forecast import forecast_series
        with pytest.raises(ValueError, match="horizon_days"):
            forecast_series(_ConstModel(), _make_history(60), 1, 1, 366)

    def test_missing_store_item_raises(self):
        from src.forecast import forecast_series
        hist = _make_history(60, store=1, item=1)
        with pytest.raises(ValueError, match="No history"):
            forecast_series(_ConstModel(), hist, 99, 99, 7)

    def test_custom_origin_date(self):
        """Forecasting from a past origin must start the day after that origin."""
        from src.forecast import forecast_series
        hist = _make_history(60)
        mid_date = hist["date"].iloc[30]
        result = forecast_series(_ConstModel(), hist, 1, 1, 7, origin_date=mid_date)
        assert result["date"].min() == mid_date + pd.Timedelta(days=1)

    def test_origin_before_all_history_raises(self):
        from src.forecast import forecast_series
        hist = _make_history(60)
        too_early = hist["date"].min() - pd.Timedelta(days=1)
        with pytest.raises(ValueError, match="No history"):
            forecast_series(_ConstModel(), hist, 1, 1, 7, origin_date=too_early)


# ===========================================================================
# B. Leakage prevention — _build_one_row_features
# ===========================================================================

class TestBuildOneRowFeatures:

    def test_future_sales_not_in_lag1(self):
        """lag_1 for the future row must come from the last tail row, not NaN."""
        from src.forecast import _build_one_row_features
        tail = _make_history(30)
        last_sales = float(tail["sales"].iloc[-1])
        last_date  = tail["date"].iloc[-1]
        next_date  = last_date + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        # lag_1 of the future row = sales of the last tail row
        assert float(row["lag_1"].values[0]) == pytest.approx(last_sales)

    def test_future_sales_col_is_nan(self):
        """The future row's 'sales' value must remain NaN (never filled)."""
        from src.forecast import _build_one_row_features
        tail = _make_history(30)
        next_date = tail["date"].iloc[-1] + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        assert pd.isna(row["sales"].values[0])

    def test_lag_does_not_use_future_value(self):
        """lag_1 must not equal the (unknown) future sales value."""
        from src.forecast import _build_one_row_features
        tail = _make_history(30)
        future_sales = 999.0   # hypothetical future — must never appear in lags
        next_date = tail["date"].iloc[-1] + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        # lag_1 is the previous actual, not 999
        assert float(row["lag_1"].values[0]) != pytest.approx(future_sales)

    def test_rolling_mean_uses_only_past(self):
        """rolling_mean_7 must equal the mean of the last 7 known sales values."""
        from src.forecast import _build_one_row_features
        tail = _make_history(30, sales_start=10)  # sales = 10,11,12,...,39
        last_date  = tail["date"].iloc[-1]
        next_date  = last_date + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        # rolling_mean_7 should be mean of sales[23..29] (last 7 of 30)
        expected = float(tail["sales"].iloc[-7:].mean())
        assert float(row["rolling_mean_7"].values[0]) == pytest.approx(expected, rel=1e-4)

    def test_output_has_one_row(self):
        from src.forecast import _build_one_row_features
        tail = _make_history(30)
        next_date = tail["date"].iloc[-1] + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        assert len(row) == 1

    def test_output_has_calendar_features(self):
        from src.forecast import _build_one_row_features
        from src.features import CALENDAR_COLS
        tail = _make_history(30)
        next_date = tail["date"].iloc[-1] + pd.Timedelta(days=1)
        row = _build_one_row_features(tail, next_date, 1, 1)
        for col in CALENDAR_COLS:
            assert col in row.columns


# ===========================================================================
# C. generate_submission
# ===========================================================================

class TestGenerateSubmission:

    def _make_test_df(self) -> pd.DataFrame:
        """Tiny test set: 1 store, 1 item, 7 days."""
        dates = pd.date_range("2017-12-02", periods=7, freq="D")
        return pd.DataFrame({
            "id":    range(7),
            "date":  dates,
            "store": pd.array([1] * 7, dtype="int16"),
            "item":  pd.array([1] * 7, dtype="int16"),
        })

    def test_output_columns(self, tmp_path):
        from src.forecast import generate_submission
        hist = _make_history(60)
        test = self._make_test_df()
        sub = generate_submission(
            _ConstModel(5.0), test, hist,
            output_path=tmp_path / "sub.csv",
        )
        assert "id" in sub.columns
        assert "sales" in sub.columns

    def test_output_row_count(self, tmp_path):
        from src.forecast import generate_submission
        hist = _make_history(60)
        test = self._make_test_df()
        sub = generate_submission(
            _ConstModel(5.0), test, hist,
            output_path=tmp_path / "sub.csv",
        )
        assert len(sub) == len(test)

    def test_non_negative_predictions(self, tmp_path):
        from src.forecast import generate_submission
        hist = _make_history(60)
        test = self._make_test_df()
        sub = generate_submission(
            _ConstModel(0.0), test, hist,
            output_path=tmp_path / "sub.csv",
        )
        assert (sub["sales"] >= 0).all()

    def test_csv_written(self, tmp_path):
        from src.forecast import generate_submission
        hist = _make_history(60)
        test = self._make_test_df()
        out = tmp_path / "submission.csv"
        generate_submission(_ConstModel(), test, hist, output_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert set(loaded.columns) == {"id", "sales"}

    def test_sorted_by_id(self, tmp_path):
        from src.forecast import generate_submission
        hist = _make_history(60)
        test = self._make_test_df()
        sub = generate_submission(
            _ConstModel(), test, hist,
            output_path=tmp_path / "sub.csv",
        )
        assert sub["id"].tolist() == sorted(sub["id"].tolist())
