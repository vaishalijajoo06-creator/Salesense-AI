"""
forecast.py
-----------
Future sales forecasting engine for SalesSense AI.

Public API
----------
forecast_series(model, history_df, store, item, horizon_days)
    Recursive multi-step forecast for a single (store, item) pair.
    Returns a DataFrame with columns: date, store, item, predicted_sales.

generate_submission(model, test_df, output_path)
    Batch forecast for the competition test set (all store-item pairs).
    Returns a DataFrame with columns: id, sales.

Leakage prevention
------------------
The recursive forecaster never reads actual future sales.  At each step:
  1. Build a full feature row for the *next* date using only:
     - Calendar features (derived from the date itself)
     - Lag features from the known history tail
     - Rolling features from the known history tail
  2. Predict the next day's sales.
  3. Append that *prediction* (not any actual future value) to the history
     so that subsequent lag/rolling windows use it.

This means forecast error accumulates over long horizons — a known
limitation of recursive forecasting that is documented in the docstring.
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

from src.features import (
    LAG_DAYS,
    ROLLING_WINDOWS,
    add_calendar_features,
    FEATURE_COLS,
)

OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "outputs"

# How many historical tail rows to keep for feature computation.
# We need at least max(LAG_DAYS) days of history for lags, plus
# max(ROLLING_WINDOWS) for rolling means.  Keep a comfortable buffer.
_TAIL_ROWS = max(max(LAG_DAYS), max(ROLLING_WINDOWS)) + 10   # = 38


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_one_row_features(
    tail: pd.DataFrame,
    next_date: pd.Timestamp,
    store: int,
    item: int,
) -> pd.DataFrame:
    """Compute feature values for a single future date using the history tail.

    Parameters
    ----------
    tail      : DataFrame with columns date, store, item, sales
                (last _TAIL_ROWS rows of the known series, sorted by date).
    next_date : The future date for which to build features.
    store, item : identifiers (carried through as integer columns).

    Returns
    -------
    pd.DataFrame with exactly one row containing all FEATURE_COLS plus
    store and item.  'sales' is set to NaN for the future row.

    Leakage note
    ------------
    The future row has sales=NaN.  When we call shift/rolling on the
    combined frame [tail + future_row], the NaN sales is never included
    in any lag/rolling window because shift(k) looks backward k steps —
    the NaN row is the *last* row, so it is shifted *out* of the window
    for every previous row, and its own lag values come from the tail.
    """
    future_row = pd.DataFrame({
        "date":  [next_date],
        "store": pd.array([store], dtype="int16"),
        "item":  pd.array([item],  dtype="int16"),
        "sales": [np.nan],
    })

    combined = pd.concat([tail, future_row], ignore_index=True)
    combined = combined.sort_values(["store", "item", "date"]).reset_index(drop=True)

    # ── Calendar features ────────────────────────────────────────────────
    combined = add_calendar_features(combined)

    # ── Lag features ────────────────────────────────────────────────────
    # groupby.shift(k) on a single-series frame is safe: the NaN future
    # sales row contributes NaN to nothing (it is at position -1).
    grouped_sales = combined.groupby(["store", "item"], sort=False)["sales"]
    for k in LAG_DAYS:
        combined[f"lag_{k}"] = grouped_sales.shift(k)

    # ── Rolling features ────────────────────────────────────────────────
    def _roll(group: pd.Series, window: int, func: str) -> pd.Series:
        shifted = group.shift(1)          # shift(1) = look at yesterday
        if func == "mean":
            return shifted.rolling(window, min_periods=1).mean()
        return shifted.rolling(window, min_periods=2).std()

    grouped = combined.groupby(["store", "item"], sort=False)["sales"]
    for w in ROLLING_WINDOWS:
        combined[f"rolling_mean_{w}"] = grouped.transform(
            lambda g, w=w: _roll(g, w, "mean")
        )
        combined[f"rolling_std_{w}"] = grouped.transform(
            lambda g, w=w: _roll(g, w, "std")
        )

    # Return only the last row (the future date)
    return combined.tail(1).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API — single-series recursive forecaster
# ---------------------------------------------------------------------------

def forecast_series(
    model,
    history_df: pd.DataFrame,
    store: int,
    item: int,
    horizon_days: int,
    origin_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Produce a *horizon_days*-step recursive forecast for one (store, item).

    Parameters
    ----------
    model        : fitted model with a .predict(df) method that accepts a
                   DataFrame containing store, item, and FEATURE_COLS.
    history_df   : full training DataFrame (date, store, item, sales).
                   Only rows matching (store, item) are used; the rest are
                   ignored.
    store, item  : identifiers to forecast.
    horizon_days : number of future days to forecast (1–365).
    origin_date  : last known date.  Defaults to the last date in history_df
                   for the given (store, item).

    Returns
    -------
    pd.DataFrame
        Columns: date, store, item, predicted_sales
        Rows: horizon_days rows, one per future day.

    Raises
    ------
    ValueError  if the (store, item) series has no history.
    ValueError  if horizon_days < 1 or horizon_days > 365.

    Leakage guarantee
    -----------------
    At every step t, only sales values at t-1, t-2, … (previously recorded
    actuals OR previously generated predictions) are used.  The actual
    sales values for future dates are never read from history_df — they
    simply do not exist there for future dates.
    """
    if horizon_days < 1 or horizon_days > 365:
        raise ValueError(f"horizon_days must be in [1, 365], got {horizon_days}.")

    # Filter to the requested series
    series = (
        history_df[(history_df["store"] == store) & (history_df["item"] == item)]
        .sort_values("date")
        .reset_index(drop=True)
    )
    if len(series) == 0:
        raise ValueError(
            f"No history found for store={store}, item={item}. "
            "Check that these IDs exist in the training data."
        )

    if origin_date is None:
        origin_date = series["date"].max()
    else:
        # Clip history to the requested origin
        series = series[series["date"] <= origin_date].reset_index(drop=True)
        if len(series) == 0:
            raise ValueError(
                f"No history for store={store}, item={item} "
                f"before origin_date={origin_date.date()}."
            )

    # Keep a rolling tail of the last _TAIL_ROWS rows
    tail = series[["date", "store", "item", "sales"]].tail(_TAIL_ROWS).copy()

    predictions: list[dict] = []

    for step in range(horizon_days):
        next_date = origin_date + pd.Timedelta(days=step + 1)

        # Build a single-row feature frame
        feature_row = _build_one_row_features(tail, next_date, store, item)

        # Predict (model handles NaN-filled features internally)
        pred_val = float(model.predict(feature_row)[0])
        pred_val = max(0.0, pred_val)          # guarantee non-negative

        predictions.append({
            "date":            next_date,
            "store":           store,
            "item":            item,
            "predicted_sales": round(pred_val, 2),
        })

        # Append the prediction to the tail so future steps can use it as lag
        new_row = pd.DataFrame({
            "date":  [next_date],
            "store": pd.array([store], dtype="int16"),
            "item":  pd.array([item],  dtype="int16"),
            "sales": [pred_val],
        })
        tail = pd.concat([tail, new_row], ignore_index=True).tail(_TAIL_ROWS)

    return pd.DataFrame(predictions)


# ---------------------------------------------------------------------------
# Public API — batch submission generator (competition test set)
# ---------------------------------------------------------------------------

def generate_submission(
    model,
    test_df: pd.DataFrame,
    history_df: pd.DataFrame,
    output_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Generate submission predictions for the competition test set.

    Iterates over every (store, item) pair in test_df, determines the
    horizon for each, runs forecast_series, and assembles the submission.

    Parameters
    ----------
    model       : fitted model (same interface as forecast_series).
    test_df     : DataFrame from load_test() — columns id, date, store, item.
    history_df  : full training DataFrame — columns date, store, item, sales.
    output_path : where to save the CSV.  Defaults to outputs/predictions.csv.

    Returns
    -------
    pd.DataFrame  with columns id (int), sales (float), sorted by id.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "predictions.csv"

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Origin = last training date for each (store, item)
    last_train_date = (
        history_df.groupby(["store", "item"])["date"].max()
    )

    # Sort test rows so we can iterate group by group
    test_sorted = test_df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    groups = test_sorted.groupby(["store", "item"])

    results: list[pd.DataFrame] = []

    for (store, item), grp in groups:
        origin = last_train_date.get((store, item))
        if origin is None:
            # Unseen (store, item): fill with 0 — logged as a warning
            grp = grp.copy()
            grp["sales"] = 0.0
            results.append(grp[["id", "sales"]])
            continue

        horizon = len(grp)
        preds_df = forecast_series(
            model, history_df, int(store), int(item),
            horizon_days=horizon, origin_date=origin,
        )

        # Match predictions back to test row ids (both are date-sorted)
        grp = grp.copy().reset_index(drop=True)
        grp["sales"] = preds_df["predicted_sales"].values
        results.append(grp[["id", "sales"]])

    submission = (
        pd.concat(results, ignore_index=True)
        .sort_values("id")
        .reset_index(drop=True)
    )
    submission["sales"] = submission["sales"].round(2)
    submission.to_csv(output_path, index=False)
    return submission


def load_submission(path: pathlib.Path | None = None) -> pd.DataFrame:
    """Load a previously generated submission CSV."""
    if path is None:
        path = OUTPUT_DIR / "predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Submission file not found: {path}")
    return pd.read_csv(path)
