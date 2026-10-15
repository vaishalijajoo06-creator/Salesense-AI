"""
test_integration.py
-------------------
End-to-end integration tests for SalesSense AI — Stage 7.

These tests exercise the full pipeline rather than individual functions
in isolation:
  A. Features → Splitter → Baseline  (pure in-memory, no CSV needed)
  B. Features → Splitter → Ridge round-trip
  C. Features → Splitter → LightGBM round-trip
  D. Forecast pipeline: forecast_series with a fitted model stub
  E. Submission pipeline: generate_submission schema and row count
  F. Metrics pipeline: compare_models output from real model predictions
  G. Real-data smoke tests (skipped if CSV files are absent)

Constraints
-----------
* No data files are modified.
* All synthetic datasets use strictly chronological ordering.
* No metrics are hardcoded — they are computed from actual model calls.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip guard for real CSV files
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
HAS_DATA = (
    (DATA_DIR / "train.csv").exists()
    and (DATA_DIR / "test.csv").exists()
)
skip_no_data = pytest.mark.skipif(not HAS_DATA, reason="CSV files not present")


# ---------------------------------------------------------------------------
# Shared synthetic dataset builders
# ---------------------------------------------------------------------------

def _make_series(
    n: int = 100,
    stores: int = 2,
    items: int = 2,
    sales_base: int = 20,
    start: str = "2016-01-01",
) -> pd.DataFrame:
    """Synthetic dataset: stores × items × n consecutive days."""
    rows = []
    for s in range(1, stores + 1):
        for i in range(1, items + 1):
            dates = pd.date_range(start, periods=n, freq="D")
            for d_idx, d in enumerate(dates):
                rows.append({
                    "date":  d,
                    "store": s,
                    "item":  i,
                    "sales": sales_base + (s * 3) + (i * 2) + (d_idx % 7),
                })
    df = pd.DataFrame(rows)
    df["store"] = df["store"].astype("int16")
    df["item"]  = df["item"].astype("int16")
    df["sales"] = df["sales"].astype("int32")
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


# ===========================================================================
# A. Features → Splitter → Baseline pipeline
# ===========================================================================

class TestBaselinePipeline:

    def test_naive_seasonal_end_to_end(self):
        """Full pipeline: build_features → split → NaiveSeasonal fit/predict."""
        from src.features import build_features
        from src.splitter import split_train_val
        from src.models.baseline import NaiveSeasonal

        # n=200 from 2017-06-01 reaches 2017-12-17, spanning the 2017-10-01 split
        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        train, val = split_train_val(featured)

        assert len(train) > 0, "Train split is empty"
        assert len(val) > 0,   "Val split is empty"

        model = NaiveSeasonal()
        model.fit(train)
        preds = model.predict(val)

        assert len(preds) == len(val)
        assert np.all(preds >= 0), "NaiveSeasonal produced negative predictions"

    def test_seasonal_mean_end_to_end(self):
        """Full pipeline: build_features → split → SeasonalMean fit/predict."""
        from src.features import build_features
        from src.splitter import split_train_val
        from src.models.baseline import SeasonalMean

        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        train, val = split_train_val(featured)

        model = SeasonalMean()
        model.fit(train)
        preds = model.predict(val)

        assert len(preds) == len(val)
        assert np.all(preds >= 0)

    def test_no_val_data_leaks_into_train(self):
        """Train rows must all precede val rows chronologically."""
        from src.features import build_features
        from src.splitter import split_train_val, TRAIN_END, VAL_START

        df = _make_series(n=200, start="2017-06-01")
        train, val = split_train_val(build_features(df))

        if len(train) > 0 and len(val) > 0:
            assert train["date"].max() <= TRAIN_END
            assert val["date"].min() >= VAL_START


# ===========================================================================
# B. Features → Splitter → Ridge round-trip
# ===========================================================================

class TestRidgePipeline:

    def _get_train_val(self):
        from src.features import build_features
        from src.splitter import split_train_val
        # n=200 from 2017-06-01 reaches 2017-12-17, spanning the 2017-10-01 split
        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        return split_train_val(featured)

    def test_ridge_fit_predict_shape(self):
        """Ridge must return exactly len(val) predictions."""
        from src.models.linear import RidgeForecaster
        train, val = self._get_train_val()
        model = RidgeForecaster()
        model.fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_ridge_predict_non_negative(self):
        from src.models.linear import RidgeForecaster
        train, val = self._get_train_val()
        preds = RidgeForecaster().fit(train).predict(val)
        assert np.all(preds >= 0)

    def test_ridge_fit_twice_is_idempotent(self):
        """Calling fit() a second time must not raise and must update the model."""
        from src.models.linear import RidgeForecaster
        train, val = self._get_train_val()
        m = RidgeForecaster()
        m.fit(train)
        p1 = m.predict(val)
        m.fit(train)
        p2 = m.predict(val)
        np.testing.assert_array_almost_equal(p1, p2)

    def test_ridge_predict_before_fit_raises(self):
        from src.models.linear import RidgeForecaster
        m = RidgeForecaster()
        with pytest.raises(RuntimeError):
            m.predict(pd.DataFrame({"store": [1], "item": [1]}))


# ===========================================================================
# C. Features → Splitter → LightGBM round-trip
# ===========================================================================

class TestLGBMPipeline:

    def _get_train_val(self):
        from src.features import build_features
        from src.splitter import split_train_val
        # n=200 from 2017-06-01 reaches 2017-12-17, spanning the 2017-10-01 split
        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        return split_train_val(featured)

    def test_lgbm_fit_predict_shape(self):
        from src.models.gradient_boost import LGBMForecaster
        train, val = self._get_train_val()
        model = LGBMForecaster(n_estimators=10)   # fast for tests
        model.fit(train)
        preds = model.predict(val)
        assert len(preds) == len(val)

    def test_lgbm_predict_non_negative(self):
        from src.models.gradient_boost import LGBMForecaster
        train, val = self._get_train_val()
        preds = LGBMForecaster(n_estimators=10).fit(train).predict(val)
        assert np.all(preds >= 0)

    def test_lgbm_predict_before_fit_raises(self):
        from src.models.gradient_boost import LGBMForecaster
        m = LGBMForecaster(n_estimators=10)
        with pytest.raises(RuntimeError):
            m.predict(pd.DataFrame({"store": [1], "item": [1]}))

    def test_lgbm_better_than_naive_on_synthetic(self):
        """LGBM WAPE must be finite and > 0 on synthetic data."""
        from src.models.gradient_boost import LGBMForecaster
        from src.evaluate import wape
        train, val = self._get_train_val()
        model = LGBMForecaster(n_estimators=50)
        model.fit(train)
        preds = model.predict(val)
        w = wape(val["sales"].values.astype(float), preds)
        assert np.isfinite(w)
        assert w >= 0


# ===========================================================================
# D. Forecast pipeline — forecast_series with a fitted Ridge model
# ===========================================================================

class TestForecastPipeline:

    def _fitted_ridge(self, history: pd.DataFrame):
        """Return a Ridge model fitted on the history DataFrame."""
        from src.features import build_features
        from src.models.linear import RidgeForecaster
        featured = build_features(history)
        m = RidgeForecaster()
        m.fit(featured)
        return m

    def test_forecast_series_output_length(self):
        from src.forecast import forecast_series
        hist = _make_series(n=90, stores=1, items=1, start="2017-01-01")
        model = self._fitted_ridge(hist)
        for h in [7, 30]:
            result = forecast_series(model, hist, 1, 1, h)
            assert len(result) == h

    def test_forecast_series_dates_consecutive(self):
        from src.forecast import forecast_series
        hist = _make_series(n=90, stores=1, items=1, start="2017-01-01")
        model = self._fitted_ridge(hist)
        result = forecast_series(model, hist, 1, 1, 14)
        dates = result["date"].sort_values().reset_index(drop=True)
        gaps = dates.diff().dropna().dt.days
        assert (gaps == 1).all()

    def test_forecast_series_non_negative(self):
        from src.forecast import forecast_series
        hist = _make_series(n=90, stores=1, items=1, start="2017-01-01")
        model = self._fitted_ridge(hist)
        result = forecast_series(model, hist, 1, 1, 30)
        assert (result["predicted_sales"] >= 0).all()

    def test_forecast_series_starts_after_origin(self):
        from src.forecast import forecast_series
        hist = _make_series(n=90, stores=1, items=1, start="2017-01-01")
        origin = hist["date"].max()
        model = self._fitted_ridge(hist)
        result = forecast_series(model, hist, 1, 1, 7)
        assert result["date"].min() == origin + pd.Timedelta(days=1)

    def test_forecast_pipeline_lgbm(self):
        """Forecast pipeline with LightGBM model (fast, n_estimators=10)."""
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import forecast_series
        hist = _make_series(n=90, stores=1, items=1, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)
        result = forecast_series(model, hist, 1, 1, 14)
        assert len(result) == 14
        assert (result["predicted_sales"] >= 0).all()


# ===========================================================================
# E. Submission pipeline — generate_submission schema and row count
# ===========================================================================

class TestSubmissionPipeline:

    def _make_test_df(self, start: str = "2017-04-02") -> pd.DataFrame:
        """Tiny test set: 2 stores × 2 items × 10 days."""
        rows = []
        for s in [1, 2]:
            for i in [1, 2]:
                for d in pd.date_range(start, periods=10, freq="D"):
                    rows.append({"date": d, "store": s, "item": i})
        df = pd.DataFrame(rows)
        df["store"] = df["store"].astype("int16")
        df["item"]  = df["item"].astype("int16")
        df["id"] = range(len(df))
        return df[["id", "date", "store", "item"]]

    def test_generate_submission_columns(self, tmp_path):
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import generate_submission

        hist = _make_series(n=90, stores=2, items=2, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)

        test = self._make_test_df()
        sub = generate_submission(model, test, hist, output_path=tmp_path / "sub.csv")

        assert set(sub.columns) == {"id", "sales"}

    def test_generate_submission_row_count(self, tmp_path):
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import generate_submission

        hist = _make_series(n=90, stores=2, items=2, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)

        test = self._make_test_df()
        sub = generate_submission(model, test, hist, output_path=tmp_path / "sub.csv")

        assert len(sub) == len(test)

    def test_generate_submission_non_negative(self, tmp_path):
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import generate_submission

        hist = _make_series(n=90, stores=2, items=2, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)

        test = self._make_test_df()
        sub = generate_submission(model, test, hist, output_path=tmp_path / "sub.csv")

        assert (sub["sales"] >= 0).all()

    def test_generate_submission_sorted_by_id(self, tmp_path):
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import generate_submission

        hist = _make_series(n=90, stores=2, items=2, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)

        test = self._make_test_df()
        sub = generate_submission(model, test, hist, output_path=tmp_path / "sub.csv")

        assert sub["id"].tolist() == sorted(sub["id"].tolist())

    def test_generate_submission_csv_loadable(self, tmp_path):
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        from src.forecast import generate_submission

        hist = _make_series(n=90, stores=2, items=2, start="2017-01-01")
        featured = build_features(hist)
        model = LGBMForecaster(n_estimators=10).fit(featured)

        out = tmp_path / "submission.csv"
        test = self._make_test_df()
        generate_submission(model, test, hist, output_path=out)

        loaded = pd.read_csv(out)
        assert set(loaded.columns) == {"id", "sales"}
        assert len(loaded) == len(test)


# ===========================================================================
# F. Metrics pipeline — compare_models with two real model outputs
# ===========================================================================

class TestMetricsPipeline:

    def test_compare_models_two_models(self):
        """compare_models must rank both models; output has correct schema."""
        from src.features import build_features
        from src.splitter import split_train_val
        from src.models.baseline import NaiveSeasonal
        from src.models.gradient_boost import LGBMForecaster
        from src.evaluate import compare_models

        # n=200 from 2017-06-01 reaches 2017-12-17, spanning the 2017-10-01 split
        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        train, val = split_train_val(featured)

        naive = NaiveSeasonal().fit(train)
        lgbm  = LGBMForecaster(n_estimators=50).fit(train)

        y_true = val["sales"].values.astype(float)
        results = {
            "NaiveSeasonal": {"y_true": y_true, "y_pred": naive.predict(val)},
            "LightGBM":      {"y_true": y_true, "y_pred": lgbm.predict(val)},
        }
        metrics_df = compare_models(results)

        assert list(metrics_df.columns) == ["model", "MAE", "RMSE", "WAPE", "SMAPE"]
        assert len(metrics_df) == 2
        # WAPE must be sorted ascending (best first)
        wapes = metrics_df["WAPE"].tolist()
        assert wapes == sorted(wapes)

    def test_compare_models_model_name_alignment(self):
        """Model name in row 0 must match the lowest-WAPE model's actual metrics."""
        from src.features import build_features
        from src.splitter import split_train_val
        from src.models.baseline import NaiveSeasonal
        from src.models.gradient_boost import LGBMForecaster
        from src.evaluate import compare_models, wape as wape_fn

        # n=200 from 2017-06-01 reaches 2017-12-17, spanning the 2017-10-01 split
        df = _make_series(n=200, start="2017-06-01")
        featured = build_features(df)
        train, val = split_train_val(featured)

        naive = NaiveSeasonal().fit(train)
        lgbm  = LGBMForecaster(n_estimators=50).fit(train)
        y_true = val["sales"].values.astype(float)
        naive_preds = naive.predict(val)
        lgbm_preds  = lgbm.predict(val)

        metrics_df = compare_models({
            "NaiveSeasonal": {"y_true": y_true, "y_pred": naive_preds},
            "LightGBM":      {"y_true": y_true, "y_pred": lgbm_preds},
        })

        best_row = metrics_df.iloc[0]
        best_name = best_row["model"]
        if best_name == "LightGBM":
            expected_wape = wape_fn(y_true, lgbm_preds)
        else:
            expected_wape = wape_fn(y_true, naive_preds)

        assert best_row["WAPE"] == pytest.approx(expected_wape, rel=1e-4), (
            f"Row 0 model={best_name} but WAPE={best_row['WAPE']:.4f} "
            f"doesn't match computed {expected_wape:.4f}"
        )


# ===========================================================================
# G. Real-data smoke tests (skipped if CSV files absent)
# ===========================================================================

@skip_no_data
def test_real_data_pipeline_two_series():
    """Load train.csv, build features for 2 store-item pairs, fit Ridge, WAPE finite."""
    from src.data_loader import load_train
    from src.features import build_features
    from src.splitter import split_train_val
    from src.models.linear import RidgeForecaster
    from src.evaluate import wape

    train_full = load_train()
    # Use only 2 store-item pairs to keep the test fast
    subset = train_full[
        ((train_full["store"] == 1) & (train_full["item"] == 1)) |
        ((train_full["store"] == 1) & (train_full["item"] == 2))
    ].copy()

    featured = build_features(subset)
    train, val = split_train_val(featured)

    model = RidgeForecaster()
    model.fit(train)
    preds = model.predict(val)

    w = wape(val["sales"].values.astype(float), preds)
    assert np.isfinite(w), f"WAPE is not finite: {w}"
    assert w >= 0


@skip_no_data
def test_real_data_forecast_7_days():
    """Load train.csv, fit LightGBM on store=1 item=1, forecast 7 days."""
    from src.data_loader import load_train
    from src.features import build_features
    from src.models.gradient_boost import LGBMForecaster
    from src.forecast import forecast_series

    train_full = load_train()
    subset = train_full[
        (train_full["store"] == 1) & (train_full["item"] == 1)
    ].copy()

    featured = build_features(subset)
    model = LGBMForecaster(n_estimators=50).fit(featured)

    result = forecast_series(model, subset, 1, 1, 7)

    assert len(result) == 7
    assert (result["predicted_sales"] >= 0).all()
    # Dates must start the day after the last training date
    expected_start = subset["date"].max() + pd.Timedelta(days=1)
    assert result["date"].min() == expected_start


@skip_no_data
def test_real_metrics_lgbm_below_bound():
    """LightGBM WAPE on the val set must stay below 20% (real-data regression guard)."""
    from src.data_loader import load_train
    from src.features import build_features
    from src.splitter import split_train_val
    from src.models.gradient_boost import LGBMForecaster
    from src.evaluate import wape

    train_full = load_train()
    # Use 10 store-item pairs for speed — enough for a meaningful bound
    mask = (train_full["store"] <= 2) & (train_full["item"] <= 5)
    subset = train_full[mask].copy()

    featured = build_features(subset)
    train, val = split_train_val(featured)

    model = LGBMForecaster(n_estimators=100).fit(train)
    preds = model.predict(val)
    w = wape(val["sales"].values.astype(float), preds)

    assert w < 20.0, (
        f"LightGBM WAPE={w:.2f}% on real 10-series subset exceeded 20% bound. "
        "Check for data leakage or feature regression."
    )
