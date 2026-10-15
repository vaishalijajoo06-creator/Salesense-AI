"""
splitter.py
-----------
Chronological train / validation split for SalesSense AI.

Split boundaries
----------------
Training period   : 2013-01-01 → 2017-09-30  (4 yr 9 mo, ~1735 days)
Validation period : 2017-10-01 → 2017-12-31  (3 months = 92 days)
Test horizon      : 2018-01-01 → 2018-03-31  (held-out, labels absent)

Rationale
---------
* The competition test set covers Jan–Mar 2018.  A 3-month validation
  window mirrors that horizon exactly, so validation performance is a
  realistic proxy for generalisation.
* The split is strictly chronological: validation rows are always
  *after* all training rows for every store-item pair.  Random shuffling
  is never used.
* Features (lags, rolling) for validation rows are computed from
  training-period observations only — validation target values are never
  seen when computing lag features for training.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Fixed date boundaries (single source of truth)
# ---------------------------------------------------------------------------
TRAIN_END   = pd.Timestamp("2017-09-30")
VAL_START   = pd.Timestamp("2017-10-01")
VAL_END     = pd.Timestamp("2017-12-31")


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split *df* into training and validation sets by date.

    Parameters
    ----------
    df : pd.DataFrame  must contain a ``date`` column (datetime64).

    Returns
    -------
    (train_df, val_df) tuple of DataFrames
        train_df : rows where date <= 2017-09-30
        val_df   : rows where date is in [2017-10-01, 2017-12-31]

    Notes
    -----
    * No rows are dropped — every row lands in exactly one split.
    * The caller is responsible for calling ``build_features()`` on the
      combined frame *before* splitting so that validation-row lag
      features draw from training-period history without leakage
      (lag/rolling windows look backward, so no val target is exposed).
    """
    train = df[df["date"] <= TRAIN_END].copy()
    val   = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    return train, val


def get_split_info() -> dict:
    """Return the split date boundaries as a plain dict (for display)."""
    return {
        "train_start": "2013-01-01",
        "train_end":   str(TRAIN_END.date()),
        "val_start":   str(VAL_START.date()),
        "val_end":     str(VAL_END.date()),
        "train_days":  (TRAIN_END - pd.Timestamp("2013-01-01")).days + 1,
        "val_days":    (VAL_END - VAL_START).days + 1,
    }
