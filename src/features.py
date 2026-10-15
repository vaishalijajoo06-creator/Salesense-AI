"""
features.py
-----------
Feature engineering pipeline for SalesSense AI.

Design principles
-----------------
1. **No target leakage.**  Lag and rolling features are derived from
   *past* sales only.  Specifically:
   - Lag-k means the sale value k days *before* the current row.
   - Rolling statistics are computed on a window that ends at lag-1
     (the day before the current row), never on the current day.
   - When this module is called at inference time, `sales` values for
     future rows will simply be NaN; the lag/rolling logic reads only
     existing (historical) rows, so no leakage occurs.

2. **Per store-item computation.**  Lag and rolling windows are applied
   inside each (store, item) group so that one series' history never
   bleeds into another.

3. **NaN for early records.**  Rows that lack enough history to compute
   a lag or rolling value receive NaN — they are never silently filled
   with a misleading constant.  Callers (model trainers) can then
   decide whether to drop those rows or impute them explicitly.

4. **Reusability.**  `build_features()` is the single entry point for
   both training and inference.  Pass a DataFrame with a `sales` column
   for training; pass one where `sales` is absent (or NaN for future
   dates) for inference — the calendar features still work and the lag
   features propagate from the historical portion appended by the caller.

Public API
----------
add_calendar_features(df)  → df with calendar columns added
add_lag_features(df)       → df with lag_1/7/14/28 columns added
add_rolling_features(df)   → df with rolling mean/std columns added
build_features(df)         → df with all feature columns added
FEATURE_COLS               → list of all feature column names produced
LAG_DAYS                   → lag offsets used
ROLLING_WINDOWS            → rolling window sizes used
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants — single source of truth for column names
# ---------------------------------------------------------------------------

#: Lag offsets (in days) applied per store-item series.
LAG_DAYS: list[int] = [1, 7, 14, 28]

#: Rolling window sizes (in days) applied to the lag-1 shifted series.
ROLLING_WINDOWS: list[int] = [7, 28]

#: Calendar feature column names produced by add_calendar_features().
CALENDAR_COLS: list[str] = [
    "dayofweek",
    "dayofmonth",
    "month",
    "year",
    "is_weekend",
]

#: Lag feature column names produced by add_lag_features().
LAG_COLS: list[str] = [f"lag_{k}" for k in LAG_DAYS]

#: Rolling feature column names produced by add_rolling_features().
ROLLING_COLS: list[str] = [
    f"rolling_mean_{w}" for w in ROLLING_WINDOWS
] + [
    f"rolling_std_{w}" for w in ROLLING_WINDOWS
]

#: Complete list of all engineered feature columns (excludes raw columns).
FEATURE_COLS: list[str] = CALENDAR_COLS + LAG_COLS + ROLLING_COLS


# ---------------------------------------------------------------------------
# 1. Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based calendar features derived from the ``date`` column.

    New columns
    -----------
    dayofweek  : int  0 = Monday … 6 = Sunday
    dayofmonth : int  1–31
    month      : int  1–12
    year       : int  e.g. 2013
    is_weekend : int8 1 if Saturday or Sunday, else 0

    Parameters
    ----------
    df : pd.DataFrame  must contain a ``date`` column of dtype datetime64.

    Returns
    -------
    pd.DataFrame  copy of *df* with five new columns appended.

    Notes
    -----
    Does **not** require a ``sales`` column — safe to call on test data.
    """
    df = df.copy()
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofmonth"] = df["date"].dt.day
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    return df


# ---------------------------------------------------------------------------
# 2. Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(store, item) lag features for each offset in ``LAG_DAYS``.

    For each lag k a column ``lag_k`` is added containing the value of
    ``sales`` exactly k days before the current row *within the same
    store-item series*.

    Rows that do not have k prior observations within their series
    receive **NaN** (not filled with zero or a mean).  This is
    intentional: the caller can decide how to handle them.

    Parameters
    ----------
    df : pd.DataFrame  must contain columns ``date``, ``store``,
                       ``item``, and ``sales``.
                       **Must be sorted** by (store, item, date) before
                       calling, or pass ``sort=True``.

    Returns
    -------
    pd.DataFrame  copy of *df* with ``lag_1``, ``lag_7``, ``lag_14``,
                  ``lag_28`` columns appended.

    Leakage note
    ------------
    ``lag_k`` uses ``shift(k)`` inside each group.  Because the group
    is sorted by date (ascending), ``shift(k)`` moves the window
    *backwards* — i.e. the value at row t becomes the value at row t+k,
    so row t+k sees only data up to t.  No future information is used.
    """
    df = df.copy().sort_values(["store", "item", "date"])

    grouped_sales = df.groupby(["store", "item"], sort=False)["sales"]

    for k in LAG_DAYS:
        df[f"lag_{k}"] = grouped_sales.shift(k)

    return df


# ---------------------------------------------------------------------------
# 3. Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-(store, item) rolling mean and std features.

    Rolling statistics are computed on a window ending at **lag-1**
    (the day before the current row).  This is equivalent to:
      - shift(1) to move sales one step forward (so the window never
        includes today's sale), then
      - rolling(w, min_periods=1) to compute mean/std.

    This guarantees no target leakage: the rolling window for row t
    includes only sales at t-1, t-2, …, t-w.

    New columns
    -----------
    rolling_mean_7  : 7-day rolling mean of lagged sales
    rolling_mean_28 : 28-day rolling mean of lagged sales
    rolling_std_7   : 7-day rolling std  of lagged sales (NaN if < 2 obs)
    rolling_std_28  : 28-day rolling std  of lagged sales (NaN if < 2 obs)

    Parameters
    ----------
    df : pd.DataFrame  must contain ``date``, ``store``, ``item``,
                       ``sales``.

    Returns
    -------
    pd.DataFrame  copy of *df* with four rolling columns appended.

    Leakage note
    ------------
    ``shift(1)`` ensures the window for each row starts at the
    *previous* day.  ``min_periods=1`` means a partial window is used
    for the earliest observations rather than returning NaN for entire
    windows — the trade-off is documented here and callers should be
    aware that early rows have high-variance estimates.  ``rolling_std``
    still returns NaN for the very first row of each series (only one
    observation, variance undefined).
    """
    df = df.copy().sort_values(["store", "item", "date"])

    def _rolling_per_group(group: pd.Series, window: int, func: str) -> pd.Series:
        # shift(1) moves today → tomorrow's history, so the window
        # sees only yesterday-and-earlier for every row.
        shifted = group.shift(1)
        if func == "mean":
            return shifted.rolling(window, min_periods=1).mean()
        else:  # "std"
            return shifted.rolling(window, min_periods=2).std()

    grouped = df.groupby(["store", "item"], sort=False)["sales"]

    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = grouped.transform(
            lambda g, w=w: _rolling_per_group(g, w, "mean")
        )
        df[f"rolling_std_{w}"] = grouped.transform(
            lambda g, w=w: _rolling_per_group(g, w, "std")
        )

    return df


# ---------------------------------------------------------------------------
# 4. Composite pipeline
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps and return the enriched DataFrame.

    This is the **single entry point** for both training and inference.
    The steps are applied in order:
      1. Calendar features  (no sales column required)
      2. Lag features       (requires sales; NaN where history is missing)
      3. Rolling features   (requires sales; NaN where history is missing)

    The original ``sales`` column is **never modified**.

    Parameters
    ----------
    df : pd.DataFrame  must contain ``date``, ``store``, ``item``.
                       ``sales`` is required for lags/rolling; if absent
                       those columns will be all-NaN (valid at inference).

    Returns
    -------
    pd.DataFrame  with all FEATURE_COLS added, sorted by
                  (store, item, date).

    Missing-value policy
    --------------------
    Early rows without sufficient history for lags/rolling will have NaN
    in the corresponding feature columns.  These rows must be handled by
    the caller before passing to a model (e.g. ``dropna()`` during
    training, or keeping them for inference where a model can handle NaN
    or where the lag history is prepended from the training tail).
    """
    if "sales" not in df.columns:
        # Inference path: add a NaN sales column so lag/rolling code
        # runs without crashing; columns will be NaN for future rows.
        df = df.copy()
        df["sales"] = np.nan

    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


# ---------------------------------------------------------------------------
# 5. Helper: list feature columns present in a DataFrame
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the subset of FEATURE_COLS that actually exist in *df*."""
    return [c for c in FEATURE_COLS if c in df.columns]
