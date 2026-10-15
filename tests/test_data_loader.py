"""
test_data_loader.py
-------------------
Unit tests for src/data_loader.py.

Tests that do not require the CSV files (schema validation, error
messages, dataset_summary, compute_eda_stats) are fully implemented.
Tests that require the actual CSV files are marked with
@pytest.mark.skipif so CI passes without data present.
"""

from __future__ import annotations

import pathlib
import pandas as pd
import numpy as np
import pytest

# Determine if the real data files are available
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
HAS_DATA = (
    (DATA_DIR / "train.csv").exists()
    and (DATA_DIR / "test.csv").exists()
    and (DATA_DIR / "sample_submission.csv").exists()
)
skip_no_data = pytest.mark.skipif(not HAS_DATA, reason="CSV files not present")


# ---------------------------------------------------------------------------
# _validate_train / _validate_test — tested via public API with bad DataFrames
# ---------------------------------------------------------------------------

def test_validate_train_missing_column():
    """load_train validation must raise ValueError on missing columns."""
    from src.data_loader import _validate_train
    bad = pd.DataFrame({"date": [], "store": [], "item": []})  # 'sales' missing
    with pytest.raises(ValueError, match="missing expected columns"):
        _validate_train(bad)


def test_validate_train_null_sales():
    """load_train validation must raise ValueError if sales has nulls."""
    from src.data_loader import _validate_train
    bad = pd.DataFrame({
        "date": pd.to_datetime(["2013-01-01"]),
        "store": pd.array([1], dtype="int16"),
        "item": pd.array([1], dtype="int16"),
        "sales": pd.array([None], dtype="object"),
    })
    with pytest.raises(ValueError, match="null value"):
        _validate_train(bad)


def test_validate_test_missing_column():
    """load_test validation must raise ValueError on missing columns."""
    from src.data_loader import _validate_test
    bad = pd.DataFrame({"id": [], "date": [], "store": []})  # 'item' missing
    with pytest.raises(ValueError, match="missing expected columns"):
        _validate_test(bad)


# ---------------------------------------------------------------------------
# dataset_summary — tested with synthetic DataFrames
# ---------------------------------------------------------------------------

def _make_train_df():
    dates = pd.date_range("2013-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates,
        "store": pd.array([1] * 10, dtype="int16"),
        "item": pd.array([1] * 10, dtype="int16"),
        "sales": pd.array(list(range(10)), dtype="int32"),
    })


def test_dataset_summary_keys():
    """dataset_summary must return all expected keys for a train-like DataFrame."""
    from src.data_loader import dataset_summary
    df = _make_train_df()
    s = dataset_summary(df)
    for key in ("rows", "columns", "nulls_total", "duplicate_rows",
                "date_min", "date_max", "unique_dates",
                "unique_stores", "unique_items",
                "sales_min", "sales_max", "sales_mean", "sales_total"):
        assert key in s, f"Missing key: {key}"


def test_dataset_summary_row_count():
    from src.data_loader import dataset_summary
    df = _make_train_df()
    assert dataset_summary(df)["rows"] == 10


def test_dataset_summary_no_nulls():
    from src.data_loader import dataset_summary
    df = _make_train_df()
    assert dataset_summary(df)["nulls_total"] == 0


def test_dataset_summary_sales_total():
    from src.data_loader import dataset_summary
    df = _make_train_df()
    # sales = 0..9, sum = 45
    assert dataset_summary(df)["sales_total"] == 45


def test_dataset_summary_zero_sales():
    from src.data_loader import dataset_summary
    df = _make_train_df()
    # sales starts from 0, so one zero-sales row
    assert dataset_summary(df)["zero_sales_rows"] == 1


# ---------------------------------------------------------------------------
# compute_eda_stats — tested with synthetic DataFrame
# ---------------------------------------------------------------------------

def _make_multi_df():
    """Two stores × two items × 14 days = 56 rows."""
    rows = []
    for store in [1, 2]:
        for item in [1, 2]:
            dates = pd.date_range("2013-01-01", periods=14, freq="D")
            for i, d in enumerate(dates):
                rows.append({"date": d, "store": store, "item": item, "sales": (store + item + i) % 20 + 1})
    df = pd.DataFrame(rows)
    df["store"] = df["store"].astype("int16")
    df["item"] = df["item"].astype("int16")
    df["sales"] = df["sales"].astype("int32")
    return df


def test_compute_eda_stats_keys():
    from src.data_loader import compute_eda_stats
    stats = compute_eda_stats(_make_multi_df())
    for key in ("daily", "monthly", "by_weekday", "by_store_total",
                "by_item_total", "top10_items", "top10_stores", "missing_dates"):
        assert key in stats, f"Missing key: {key}"


def test_compute_eda_stats_daily_length():
    """daily Series should have one entry per unique date in the input."""
    from src.data_loader import compute_eda_stats
    df = _make_multi_df()
    stats = compute_eda_stats(df)
    assert len(stats["daily"]) == df["date"].nunique()


def test_compute_eda_stats_no_missing_dates():
    """A contiguous 14-day range should have zero missing dates."""
    from src.data_loader import compute_eda_stats
    stats = compute_eda_stats(_make_multi_df())
    assert stats["missing_dates"] == []


def test_compute_eda_stats_top10_columns():
    from src.data_loader import compute_eda_stats
    stats = compute_eda_stats(_make_multi_df())
    assert list(stats["top10_items"].columns) == ["item", "total_sales"]
    assert list(stats["top10_stores"].columns) == ["store", "total_sales"]


# ---------------------------------------------------------------------------
# Real CSV tests (skip if files not present)
# ---------------------------------------------------------------------------

@skip_no_data
def test_load_train_columns():
    """train.csv must have exactly date, store, item, sales."""
    from src.data_loader import load_train
    df = load_train()
    assert set(df.columns) == {"date", "store", "item", "sales"}


@skip_no_data
def test_load_train_row_count():
    from src.data_loader import load_train
    df = load_train()
    assert len(df) == 913_000


@skip_no_data
def test_load_train_no_nulls():
    from src.data_loader import load_train
    df = load_train()
    assert df.isnull().sum().sum() == 0


@skip_no_data
def test_load_train_date_range():
    from src.data_loader import load_train
    df = load_train()
    assert str(df["date"].min().date()) == "2013-01-01"
    assert str(df["date"].max().date()) == "2017-12-31"


@skip_no_data
def test_load_train_unique_stores_and_items():
    from src.data_loader import load_train
    df = load_train()
    assert df["store"].nunique() == 10
    assert df["item"].nunique() == 50


@skip_no_data
def test_load_test_columns():
    from src.data_loader import load_test
    df = load_test()
    assert set(df.columns) == {"id", "date", "store", "item"}


@skip_no_data
def test_load_test_row_count():
    from src.data_loader import load_test
    df = load_test()
    assert len(df) == 45_000
