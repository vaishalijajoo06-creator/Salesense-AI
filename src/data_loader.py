"""
data_loader.py
--------------
Responsible for reading the raw CSV files from data/ and returning
clean, correctly-typed DataFrames.  No transformation or feature
engineering happens here — just loading and validation.
"""

from __future__ import annotations

import pathlib
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    """Load training data with correct dtypes.

    Returns
    -------
    pd.DataFrame
        Columns: date (datetime64[ns]), store (int16), item (int16), sales (int32)

    Raises
    ------
    FileNotFoundError  if data/train.csv does not exist.
    ValueError         if required columns are missing or sales contains nulls.
    """
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training file not found: {TRAIN_PATH}. "
            "Place train.csv in the data/ folder."
        )
    df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["date"],
        dtype={"store": "int16", "item": "int16", "sales": "int32"},
    )
    _validate_train(df)
    return df


def load_test() -> pd.DataFrame:
    """Load test data with correct dtypes.

    Returns
    -------
    pd.DataFrame
        Columns: id (int32), date (datetime64[ns]), store (int16), item (int16)

    Raises
    ------
    FileNotFoundError  if data/test.csv does not exist.
    ValueError         if required columns are missing.
    """
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test file not found: {TEST_PATH}. "
            "Place test.csv in the data/ folder."
        )
    df = pd.read_csv(
        TEST_PATH,
        parse_dates=["date"],
        dtype={"id": "int32", "store": "int16", "item": "int16"},
    )
    _validate_test(df)
    return df


def load_sample_submission() -> pd.DataFrame:
    """Load the sample submission file.

    Returns
    -------
    pd.DataFrame
        Columns: id (int32), sales (int32)

    Raises
    ------
    FileNotFoundError  if data/sample_submission.csv does not exist.
    """
    if not SUBMISSION_PATH.exists():
        raise FileNotFoundError(
            f"Submission file not found: {SUBMISSION_PATH}. "
            "Place sample_submission.csv in the data/ folder."
        )
    return pd.read_csv(
        SUBMISSION_PATH,
        dtype={"id": "int32", "sales": "int32"},
    )


# ---------------------------------------------------------------------------
# Validation helpers (raise on unexpected schema)
# ---------------------------------------------------------------------------

def _validate_train(df: pd.DataFrame) -> None:
    expected = {"date", "store", "item", "sales"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"train.csv is missing expected columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )
    null_sales = int(df["sales"].isnull().sum())
    if null_sales:
        raise ValueError(
            f"train.csv contains {null_sales} null value(s) in the 'sales' column."
        )


def _validate_test(df: pd.DataFrame) -> None:
    expected = {"id", "date", "store", "item"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"test.csv is missing expected columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a plain-dict summary of a loaded DataFrame for display.

    Works on both train and test DataFrames.  All values are JSON-safe
    primitives (str, int, float, list, None).
    """
    summary: dict = {
        "rows": len(df),
        "columns": list(df.columns),
        "nulls_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if "date" in df.columns:
        summary["date_min"] = str(df["date"].min().date())
        summary["date_max"] = str(df["date"].max().date())
        summary["unique_dates"] = int(df["date"].nunique())

    if "store" in df.columns:
        summary["unique_stores"] = int(df["store"].nunique())
        summary["store_ids"] = sorted(df["store"].unique().tolist())

    if "item" in df.columns:
        summary["unique_items"] = int(df["item"].nunique())

    if "sales" in df.columns:
        summary["sales_min"] = int(df["sales"].min())
        summary["sales_max"] = int(df["sales"].max())
        summary["sales_mean"] = round(float(df["sales"].mean()), 4)
        summary["sales_total"] = int(df["sales"].sum())
        summary["zero_sales_rows"] = int((df["sales"] == 0).sum())

    return summary


def compute_eda_stats(df: pd.DataFrame) -> dict:
    """Compute all EDA aggregate statistics from the training DataFrame.

    Returns a dict of pre-computed Pandas objects so that the Streamlit
    app and tests can consume them without re-running expensive groupbys.

    Parameters
    ----------
    df : pd.DataFrame  as returned by load_train()

    Returns
    -------
    dict with keys:
        daily          – daily total sales Series (index=date)
        monthly        – avg sales per calendar month (1–12)
        by_weekday     – avg sales per day-of-week (0=Mon … 6=Sun)
        by_store_total – total sales per store
        by_item_total  – total sales per item
        top10_items    – top 10 items by total sales
        top10_stores   – top 10 stores by total sales
        missing_dates  – list of calendar dates absent from the dataset
    """
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.dayofweek  # 0=Mon

    daily = df.groupby("date")["sales"].sum()

    monthly = df.groupby("month")["sales"].mean().round(4)

    by_weekday = df.groupby("weekday")["sales"].mean().round(4)

    by_store_total = df.groupby("store")["sales"].sum().sort_index()

    by_item_total = df.groupby("item")["sales"].sum().sort_index()

    top10_items = by_item_total.nlargest(10).reset_index()
    top10_items.columns = ["item", "total_sales"]

    top10_stores = by_store_total.nlargest(10).reset_index()
    top10_stores.columns = ["store", "total_sales"]

    # Missing calendar dates in training range
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    actual_dates = pd.DatetimeIndex(df["date"].unique())
    missing_dates = full_range.difference(actual_dates).tolist()

    return {
        "daily": daily,
        "monthly": monthly,
        "by_weekday": by_weekday,
        "by_store_total": by_store_total,
        "by_item_total": by_item_total,
        "top10_items": top10_items,
        "top10_stores": top10_stores,
        "missing_dates": missing_dates,
    }
