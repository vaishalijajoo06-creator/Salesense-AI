"""
test_features.py
----------------
Comprehensive tests for src/features.py — Stage 3.

Test categories
---------------
A. Calendar features          – columns, types, value ranges
B. Lag features               – correct values, per-group isolation,
                                missing history → NaN (not filled),
                                no leakage (lag never references today)
C. Rolling features           – correct mean/std, shift(1) leakage guard,
                                min-period behaviour
D. build_features pipeline    – output shape, column presence, no row loss,
                                sales column unchanged, inference path
E. FEATURE_COLS constant      – completeness and uniqueness
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _make_single_series(n: int = 40, sales_start: int = 10) -> pd.DataFrame:
    """One store-item pair, n consecutive days, sales = sales_start + i."""
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date":  dates,
        "store": pd.array([1] * n, dtype="int16"),
        "item":  pd.array([1] * n, dtype="int16"),
        "sales": pd.array(range(sales_start, sales_start + n), dtype="int32"),
    })


def _make_two_series(n: int = 40) -> pd.DataFrame:
    """Two store-item pairs with *different* sales patterns on the same dates."""
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    s1 = pd.DataFrame({
        "date":  dates,
        "store": pd.array([1] * n, dtype="int16"),
        "item":  pd.array([1] * n, dtype="int16"),
        "sales": pd.array(range(100, 100 + n), dtype="int32"),
    })
    s2 = pd.DataFrame({
        "date":  dates,
        "store": pd.array([2] * n, dtype="int16"),
        "item":  pd.array([2] * n, dtype="int16"),
        "sales": pd.array(range(200, 200 + n), dtype="int32"),
    })
    return pd.concat([s1, s2], ignore_index=True)


def _make_short_series(n: int = 5) -> pd.DataFrame:
    """Series shorter than the largest lag window (28 days)."""
    dates = pd.date_range("2013-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date":  dates,
        "store": pd.array([1] * n, dtype="int16"),
        "item":  pd.array([1] * n, dtype="int16"),
        "sales": pd.array(range(1, n + 1), dtype="int32"),
    })


# ===========================================================================
# A. Calendar features
# ===========================================================================

class TestCalendarFeatures:

    def test_required_columns_present(self):
        from src.features import add_calendar_features, CALENDAR_COLS
        result = add_calendar_features(_make_single_series(10))
        for col in CALENDAR_COLS:
            assert col in result.columns, f"Missing calendar column: {col}"

    def test_dayofweek_range(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(14))
        assert result["dayofweek"].between(0, 6).all()

    def test_dayofmonth_range(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(365))
        assert result["dayofmonth"].between(1, 31).all()

    def test_month_range(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(365))
        assert result["month"].between(1, 12).all()

    def test_year_values(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(10))
        assert (result["year"] == 2013).all()

    def test_is_weekend_dtype(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(14))
        assert result["is_weekend"].dtype == np.dtype("int8")

    def test_is_weekend_values_binary(self):
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(14))
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_is_weekend_matches_dayofweek(self):
        """is_weekend must be 1 iff dayofweek >= 5."""
        from src.features import add_calendar_features
        result = add_calendar_features(_make_single_series(14))
        expected = (result["dayofweek"] >= 5).astype("int8")
        pd.testing.assert_series_equal(result["is_weekend"], expected,
                                       check_names=False)

    def test_no_row_loss(self):
        from src.features import add_calendar_features
        df = _make_single_series(20)
        assert len(add_calendar_features(df)) == len(df)

    def test_original_columns_unchanged(self):
        from src.features import add_calendar_features
        df = _make_single_series(10)
        result = add_calendar_features(df)
        for col in ("date", "store", "item", "sales"):
            pd.testing.assert_series_equal(result[col].reset_index(drop=True),
                                           df[col].reset_index(drop=True))

    def test_no_sales_column_required(self):
        """add_calendar_features must work on a DataFrame without 'sales'."""
        from src.features import add_calendar_features
        df = _make_single_series(5).drop(columns=["sales"])
        result = add_calendar_features(df)
        assert "dayofweek" in result.columns


# ===========================================================================
# B. Lag features
# ===========================================================================

class TestLagFeatures:

    def test_lag_columns_present(self):
        from src.features import add_lag_features, LAG_COLS
        result = add_lag_features(_make_single_series(40))
        for col in LAG_COLS:
            assert col in result.columns, f"Missing lag column: {col}"

    def test_lag1_correct_value(self):
        """lag_1 at row t must equal sales at row t-1 (same series)."""
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        # sales[i] = 10 + i, so lag_1[i] = 10 + (i-1) for i >= 1
        for i in range(1, len(result)):
            assert result.loc[i, "lag_1"] == result.loc[i - 1, "sales"]

    def test_lag7_correct_value(self):
        """lag_7 at row t must equal sales exactly 7 rows earlier."""
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        for i in range(7, len(result)):
            assert result.loc[i, "lag_7"] == result.loc[i - 7, "sales"]

    def test_lag14_correct_value(self):
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        for i in range(14, len(result)):
            assert result.loc[i, "lag_14"] == result.loc[i - 14, "sales"]

    def test_lag28_correct_value(self):
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        for i in range(28, len(result)):
            assert result.loc[i, "lag_28"] == result.loc[i - 28, "sales"]

    def test_early_rows_are_nan_not_filled(self):
        """Rows with insufficient history must have NaN, not a filled value."""
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        # lag_1 row 0 has no prior observation → NaN
        assert pd.isna(result.loc[0, "lag_1"])
        # lag_7: first 7 rows must be NaN
        assert result.loc[:6, "lag_7"].isna().all()
        # lag_28: first 28 rows must be NaN
        assert result.loc[:27, "lag_28"].isna().all()

    def test_no_leakage_lag_never_references_today(self):
        """lag_1 for row 0 must not equal sales[0] (that would be today)."""
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        # lag_1[1] should be sales[0]; lag_1[0] must be NaN not sales[0]
        assert pd.isna(result.loc[0, "lag_1"])
        assert result.loc[1, "lag_1"] == result.loc[0, "sales"]

    def test_lag_isolated_per_store_item(self):
        """lag of series 1 must not bleed into series 2."""
        from src.features import add_lag_features
        df = _make_two_series(40)
        result = add_lag_features(df)

        s1 = result[(result["store"] == 1) & (result["item"] == 1)].sort_values("date").reset_index(drop=True)
        s2 = result[(result["store"] == 2) & (result["item"] == 2)].sort_values("date").reset_index(drop=True)

        # Series 2's lag_1[1] must be its own sales[0] (200), not series 1's (100)
        assert s2.loc[1, "lag_1"] == pytest.approx(200.0)
        assert s1.loc[1, "lag_1"] == pytest.approx(100.0)

    def test_lag_first_row_of_second_series_is_nan(self):
        """The first row of each series must have NaN for all lags."""
        from src.features import add_lag_features
        df = _make_two_series(40)
        result = add_lag_features(df)
        for store, item in [(1, 1), (2, 2)]:
            first_row = (
                result[(result["store"] == store) & (result["item"] == item)]
                .sort_values("date")
                .iloc[0]
            )
            for lag_col in ("lag_1", "lag_7", "lag_14", "lag_28"):
                assert pd.isna(first_row[lag_col]), (
                    f"Expected NaN for {lag_col} at first row of "
                    f"store={store}, item={item}"
                )

    def test_short_series_all_lag28_nan(self):
        """Series shorter than 28 days → lag_28 must be all NaN."""
        from src.features import add_lag_features
        df = _make_short_series(5)
        result = add_lag_features(df)
        assert result["lag_28"].isna().all()

    def test_no_row_loss(self):
        from src.features import add_lag_features
        df = _make_single_series(40)
        assert len(add_lag_features(df)) == len(df)

    def test_sales_column_unchanged(self):
        """add_lag_features must not modify the original sales values."""
        from src.features import add_lag_features
        df = _make_single_series(40)
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        original = df.sort_values("date").reset_index(drop=True)
        pd.testing.assert_series_equal(result["sales"], original["sales"])

    def test_date_ordering_respected(self):
        """Even if input is shuffled, lag must be computed in date order."""
        from src.features import add_lag_features
        df = _make_single_series(40).sample(frac=1, random_state=42)  # shuffle
        result = add_lag_features(df).sort_values("date").reset_index(drop=True)
        for i in range(1, len(result)):
            assert result.loc[i, "lag_1"] == result.loc[i - 1, "sales"]


# ===========================================================================
# C. Rolling features
# ===========================================================================

class TestRollingFeatures:

    def test_rolling_columns_present(self):
        from src.features import add_rolling_features, ROLLING_COLS
        result = add_rolling_features(_make_single_series(40))
        for col in ROLLING_COLS:
            assert col in result.columns, f"Missing rolling column: {col}"

    def test_rolling_mean_7_no_leakage(self):
        """rolling_mean_7 at row t must not use sales[t]."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        # At row 1: only sales[0] is available → mean = sales[0]
        assert result.loc[1, "rolling_mean_7"] == pytest.approx(
            float(df.sort_values("date").iloc[0]["sales"])
        )

    def test_rolling_mean_7_correct_value(self):
        """rolling_mean_7 at row 8 (index 8) must equal mean of sales[1..7]."""
        from src.features import add_rolling_features
        df = _make_single_series(40)  # sales = 10, 11, 12, ...
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        # For row index 8 (the 9th day), the window is sales[2..8] after shift(1):
        # shifted sales at index 8 = sales[7] = 17
        # window looks back 7 steps on shifted: values = sales[1..7] = [11,12,13,14,15,16,17]
        expected = float(np.mean([11, 12, 13, 14, 15, 16, 17]))
        assert result.loc[8, "rolling_mean_7"] == pytest.approx(expected, rel=1e-5)

    def test_rolling_mean_28_correct_value(self):
        """rolling_mean_28 at row 30 must equal mean of sales[2..29]."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        # At row 30, shifted = sales[29] = 10+29=39; window = sales[2..29]
        expected = float(np.mean(range(12, 40)))  # sales[2]=12 … sales[29]=39
        assert result.loc[30, "rolling_mean_28"] == pytest.approx(expected, rel=1e-5)

    def test_rolling_std_nan_for_first_row(self):
        """rolling_std must be NaN for the very first row (undefined variance)."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        assert pd.isna(result.loc[0, "rolling_std_7"])

    def test_rolling_std_defined_from_row2(self):
        """rolling_std_7 must be non-NaN once ≥ 2 observations are available."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        # row 1 has shift(1) = sales[0], prior rows = 1 → std still NaN (min_periods=2)
        # row 2 has sales[0], sales[1] available → std is defined
        assert not pd.isna(result.loc[2, "rolling_std_7"])

    def test_rolling_isolated_per_store_item(self):
        """Rolling stats must not cross series boundaries."""
        from src.features import add_rolling_features
        df = _make_two_series(40)
        result = add_rolling_features(df)

        s1 = result[(result["store"] == 1) & (result["item"] == 1)].sort_values("date").reset_index(drop=True)
        s2 = result[(result["store"] == 2) & (result["item"] == 2)].sort_values("date").reset_index(drop=True)

        # At row 8, series 1 mean_7 uses sales 100–106; series 2 uses 200–206
        expected_s1 = float(np.mean([101, 102, 103, 104, 105, 106, 107]))
        expected_s2 = float(np.mean([201, 202, 203, 204, 205, 206, 207]))
        assert s1.loc[8, "rolling_mean_7"] == pytest.approx(expected_s1, rel=1e-5)
        assert s2.loc[8, "rolling_mean_7"] == pytest.approx(expected_s2, rel=1e-5)

    def test_rolling_first_row_uses_no_history(self):
        """row 0: shift(1) is NaN → rolling_mean should also be NaN at row 0."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        result = add_rolling_features(df).sort_values("date").reset_index(drop=True)
        # After shift(1), row 0 is NaN; rolling(min_periods=1) on a single NaN → NaN
        assert pd.isna(result.loc[0, "rolling_mean_7"])

    def test_no_row_loss(self):
        from src.features import add_rolling_features
        df = _make_single_series(40)
        assert len(add_rolling_features(df)) == len(df)

    def test_date_ordering_respected(self):
        """Shuffled input must produce the same rolling values as sorted input."""
        from src.features import add_rolling_features
        df = _make_single_series(40)
        shuffled = df.sample(frac=1, random_state=7)
        r_sorted  = add_rolling_features(df).sort_values(["store","item","date"]).reset_index(drop=True)
        r_shuffled = add_rolling_features(shuffled).sort_values(["store","item","date"]).reset_index(drop=True)
        pd.testing.assert_series_equal(
            r_sorted["rolling_mean_7"],
            r_shuffled["rolling_mean_7"],
            check_names=False,
        )


# ===========================================================================
# D. build_features pipeline
# ===========================================================================

class TestBuildFeatures:

    def test_all_feature_cols_present(self):
        """build_features must produce every column in FEATURE_COLS."""
        from src.features import build_features, FEATURE_COLS
        df = _make_single_series(40)
        result = build_features(df)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"

    def test_original_columns_preserved(self):
        from src.features import build_features
        df = _make_single_series(40)
        result = build_features(df)
        for col in ("date", "store", "item", "sales"):
            assert col in result.columns

    def test_sales_column_not_modified(self):
        from src.features import build_features
        df = _make_single_series(40)
        result = build_features(df).sort_values(["store","item","date"]).reset_index(drop=True)
        original = df.sort_values(["store","item","date"]).reset_index(drop=True)
        pd.testing.assert_series_equal(result["sales"], original["sales"])

    def test_no_row_loss(self):
        from src.features import build_features
        df = _make_single_series(40)
        assert len(build_features(df)) == len(df)

    def test_output_shape_two_series(self):
        from src.features import build_features
        df = _make_two_series(40)
        result = build_features(df)
        assert len(result) == len(df)

    def test_inference_path_no_sales_column(self):
        """build_features must not crash when 'sales' column is absent."""
        from src.features import build_features, CALENDAR_COLS
        df = _make_single_series(10).drop(columns=["sales"])
        result = build_features(df)
        for col in CALENDAR_COLS:
            assert col in result.columns
        # Lag and rolling cols exist but are all NaN
        for col in ("lag_1", "lag_7", "rolling_mean_7"):
            assert col in result.columns
            assert result[col].isna().all()

    def test_short_series_nan_for_missing_lags(self):
        """Short series: lag_28 must be entirely NaN (not 0)."""
        from src.features import build_features
        df = _make_short_series(5)
        result = build_features(df)
        assert result["lag_28"].isna().all()
        # Also ensure NaN is genuine NaN, not 0
        assert not (result["lag_28"] == 0).any()

    def test_sorted_output(self):
        """Output must be sorted by (store, item, date)."""
        from src.features import build_features
        df = _make_two_series(20).sample(frac=1, random_state=99)
        result = build_features(df)
        expected_order = result.sort_values(["store", "item", "date"]).index.tolist()
        assert result.index.tolist() == expected_order

    def test_feature_cols_helper(self):
        from src.features import build_features, get_feature_cols, FEATURE_COLS
        df = _make_single_series(40)
        result = build_features(df)
        found = get_feature_cols(result)
        assert set(found) == set(FEATURE_COLS)


# ===========================================================================
# E. FEATURE_COLS constant
# ===========================================================================

class TestFeatureCols:

    def test_no_duplicates(self):
        from src.features import FEATURE_COLS
        assert len(FEATURE_COLS) == len(set(FEATURE_COLS))

    def test_calendar_cols_included(self):
        from src.features import FEATURE_COLS, CALENDAR_COLS
        for col in CALENDAR_COLS:
            assert col in FEATURE_COLS

    def test_lag_cols_included(self):
        from src.features import FEATURE_COLS, LAG_COLS
        for col in LAG_COLS:
            assert col in FEATURE_COLS

    def test_rolling_cols_included(self):
        from src.features import FEATURE_COLS, ROLLING_COLS
        for col in ROLLING_COLS:
            assert col in FEATURE_COLS

    def test_sales_not_in_feature_cols(self):
        """sales is the target, not a feature."""
        from src.features import FEATURE_COLS
        assert "sales" not in FEATURE_COLS
