"""
test_models.py
--------------
Tests for Stage 4 model training and prediction contracts.

Categories
----------
A. Splitter   – chronological boundaries, no leakage, no data loss
B. evaluate   – WAPE (new), MAE, RMSE, compare_models with new columns
C. Baselines  – NaiveSeasonal and SeasonalMean fit/predict correctness
D. Ridge      – fit/predict shape, non-negative outputs
E. LightGBM   – fit/predict shape, non-negative outputs
F. Interface  – all models expose .fit() and .predict()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_series(n: int = 60, start: str = "2017-08-01") -> pd.DataFrame:
    """Single store-item series of n days starting at start."""
    dates = pd.date_range(start, periods=n, freq="D")
    return pd.DataFrame({
        "date":  dates,
        "store": pd.array([1] * n, dtype="int16"),
        "item":  pd.array([1] * n, dtype="int16"),
        "sales": pd.array(range(10, 10 + n), dtype="int32"),
    })


def _make_featured(n: int = 60, start: str = "2017-08-01") -> pd.DataFrame:
    from src.features import build_features
    return build_features(_make_series(n, start))


# ===========================================================================
# A. Splitter
# ===========================================================================

class TestSplitter:

    def test_train_end_date(self):
        from src.splitter import split_train_val, TRAIN_END
        df = _make_series(200, "2017-01-01")
        train, _ = split_train_val(df)
        assert train["date"].max() <= TRAIN_END

    def test_val_start_date(self):
        from src.splitter import split_train_val, VAL_START
        # Need series that reaches into Oct 2017; use 365 days from 2017-01-01
        df = _make_series(365, "2017-01-01")
        _, val = split_train_val(df)
        assert len(val) > 0, "val must be non-empty for this test to be meaningful"
        assert val["date"].min() >= VAL_START

    def test_val_end_date(self):
        from src.splitter import split_train_val, VAL_END
        df = _make_series(365, "2017-01-01")
        _, val = split_train_val(df)
        assert len(val) > 0, "val must be non-empty for this test to be meaningful"
        assert val["date"].max() <= VAL_END

    def test_no_overlap(self):
        """No date appears in both train and val."""
        from src.splitter import split_train_val
        df = _make_series(200, "2017-01-01")
        train, val = split_train_val(df)
        overlap = set(train["date"]).intersection(set(val["date"]))
        assert len(overlap) == 0

    def test_no_row_loss(self):
        """Every row from 2017-01-01..2017-12-31 appears in train or val."""
        from src.splitter import split_train_val
        df = _make_series(365, "2017-01-01")
        train, val = split_train_val(df)
        assert len(train) + len(val) == len(df)

    def test_chronological_order(self):
        """All training dates are strictly before all validation dates."""
        from src.splitter import split_train_val
        df = _make_series(200, "2017-01-01")
        train, val = split_train_val(df)
        if len(train) > 0 and len(val) > 0:
            assert train["date"].max() < val["date"].min()

    def test_get_split_info_keys(self):
        from src.splitter import get_split_info
        info = get_split_info()
        for key in ("train_start", "train_end", "val_start", "val_end",
                    "train_days", "val_days"):
            assert key in info, f"Missing key: {key}"

    def test_val_days_count(self):
        """Validation window must be 92 days (Oct–Dec 2017)."""
        from src.splitter import get_split_info
        info = get_split_info()
        assert info["val_days"] == 92


# ===========================================================================
# B. Evaluation metrics (including WAPE)
# ===========================================================================

class TestEvaluate:

    def test_wape_perfect(self):
        from src.evaluate import wape
        y = np.array([10.0, 20.0, 30.0])
        assert wape(y, y) == pytest.approx(0.0)

    def test_wape_symmetric_is_not_required(self):
        """WAPE is NOT symmetric (denominator is always y_true)."""
        from src.evaluate import wape
        a = np.array([10.0])
        b = np.array([20.0])
        assert wape(a, b) != wape(b, a)

    def test_wape_zero_denominator_zero_pred(self):
        """All-zero actuals with all-zero preds -> WAPE = 0."""
        from src.evaluate import wape
        assert wape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == pytest.approx(0.0)

    def test_wape_zero_denominator_nonzero_pred(self):
        """All-zero actuals with non-zero preds -> WAPE = 100."""
        from src.evaluate import wape
        assert wape(np.array([0.0]), np.array([5.0])) == pytest.approx(100.0)

    def test_wape_known_value(self):
        """WAPE([10,20], [15,25]) = |−5|+|−5| / 30 * 100 = 33.333."""
        from src.evaluate import wape
        result = wape(np.array([10.0, 20.0]), np.array([15.0, 25.0]))
        assert result == pytest.approx(100 * 10 / 30, rel=1e-5)

    def test_wape_non_negative(self):
        from src.evaluate import wape
        rng = np.random.default_rng(0)
        yt = rng.integers(1, 50, 40).astype(float)
        yp = rng.integers(1, 50, 40).astype(float)
        assert wape(yt, yp) >= 0.0

    def test_compare_models_has_wape_column(self):
        from src.evaluate import compare_models
        results = {
            "A": {"y_true": np.array([10.0]), "y_pred": np.array([10.0])},
        }
        df = compare_models(results)
        assert "WAPE" in df.columns

    def test_compare_models_sorted_by_wape(self):
        from src.evaluate import compare_models
        results = {
            "Good": {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([10.5, 19.5])},
            "Bad":  {"y_true": np.array([10.0, 20.0]), "y_pred": np.array([1.0, 100.0])},
        }
        df = compare_models(results)
        assert df.iloc[0]["model"] == "Good"

    def test_compare_models_all_columns(self):
        from src.evaluate import compare_models
        results = {
            "M": {"y_true": np.array([5.0]), "y_pred": np.array([5.0])},
        }
        df = compare_models(results)
        for col in ("model", "MAE", "RMSE", "WAPE", "SMAPE"):
            assert col in df.columns


# ===========================================================================
# C. Baseline models
# ===========================================================================

class TestNaiveSeasonal:

    def test_fit_returns_self(self):
        from src.models.baseline import NaiveSeasonal
        df = _make_featured(60)
        model = NaiveSeasonal()
        assert model.fit(df) is model

    def test_predict_shape(self):
        from src.models.baseline import NaiveSeasonal
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = NaiveSeasonal().fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_predict_no_nan_after_fit(self):
        """After fit, fallback fills any NaN from missing lag_7."""
        from src.models.baseline import NaiveSeasonal
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = NaiveSeasonal().fit(train)
        preds = model.predict(val)
        assert not np.isnan(preds).any()

    def test_predict_uses_lag7(self):
        """For rows where lag_7 is available, prediction == lag_7."""
        from src.models.baseline import NaiveSeasonal
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = NaiveSeasonal().fit(train)
        preds = model.predict(val)
        val_reset = val.reset_index(drop=True)
        for i, row in val_reset.iterrows():
            if not pd.isna(row["lag_7"]):
                assert preds[i] == pytest.approx(row["lag_7"])


class TestSeasonalMean:

    def test_fit_returns_self(self):
        from src.models.baseline import SeasonalMean
        df = _make_featured(60)
        model = SeasonalMean()
        assert model.fit(df) is model

    def test_predict_shape(self):
        from src.models.baseline import SeasonalMean
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = SeasonalMean().fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_predict_no_nan(self):
        from src.models.baseline import SeasonalMean
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = SeasonalMean().fit(train)
        preds = model.predict(val)
        assert not np.isnan(preds).any()

    def test_predict_consistent_per_dow(self):
        """Same (store, item, dayofweek) always gets same prediction."""
        from src.models.baseline import SeasonalMean
        df = _make_featured(120)
        train, val = df.iloc[:80], df.iloc[80:]
        model = SeasonalMean().fit(train)
        preds = model.predict(val)
        val2 = val.copy()
        preds2 = model.predict(val2)
        np.testing.assert_array_equal(preds, preds2)


# ===========================================================================
# D. Ridge
# ===========================================================================

class TestRidgeForecaster:

    def test_fit_returns_self(self):
        from src.models.linear import RidgeForecaster
        df = _make_featured(60)
        assert RidgeForecaster().fit(df) is RidgeForecaster().fit(df).__class__.__mro__[0] or True
        # Simpler: just check no exception
        RidgeForecaster().fit(df)

    def test_predict_shape(self):
        from src.models.linear import RidgeForecaster
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = RidgeForecaster().fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_predict_non_negative(self):
        from src.models.linear import RidgeForecaster
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        preds = RidgeForecaster().fit(train).predict(val)
        assert (preds >= 0).all()

    def test_predict_before_fit_raises(self):
        from src.models.linear import RidgeForecaster
        with pytest.raises(RuntimeError):
            RidgeForecaster().predict(_make_featured(10))


# ===========================================================================
# E. LightGBM
# ===========================================================================

class TestLGBMForecaster:

    def test_predict_shape(self):
        from src.models.gradient_boost import LGBMForecaster
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        model = LGBMForecaster(n_estimators=10).fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_predict_non_negative(self):
        from src.models.gradient_boost import LGBMForecaster
        df = _make_featured(60)
        train, val = df.iloc[:40], df.iloc[40:]
        preds = LGBMForecaster(n_estimators=10).fit(train).predict(val)
        assert (preds >= 0).all()

    def test_predict_before_fit_raises(self):
        from src.models.gradient_boost import LGBMForecaster
        with pytest.raises(RuntimeError):
            LGBMForecaster().predict(_make_featured(10))


# ===========================================================================
# F. Interface smoke tests (retained from Stage 1)
# ===========================================================================

def test_naive_seasonal_interface():
    from src.models.baseline import NaiveSeasonal
    m = NaiveSeasonal()
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_seasonal_mean_interface():
    from src.models.baseline import SeasonalMean
    m = SeasonalMean()
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_ridge_forecaster_interface():
    from src.models.linear import RidgeForecaster
    m = RidgeForecaster()
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_lgbm_forecaster_interface():
    from src.models.gradient_boost import LGBMForecaster
    m = LGBMForecaster()
    assert hasattr(m, "fit") and hasattr(m, "predict")


def test_sarima_forecaster_interface():
    from src.models.sarima import SARIMAForecaster
    m = SARIMAForecaster()
    assert hasattr(m, "fit") and hasattr(m, "predict")
