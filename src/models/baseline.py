"""
baseline.py
-----------
Two simple baseline forecasters:

NaiveSeasonal
    Predicts the same value as the same weekday one week ago (lag-7).
    This is the standard seasonal-naive model for weekly-seasonal retail
    data and is a strong baseline for this dataset.

SeasonalMean
    Predicts the historical mean sales for each (store, item, dayofweek)
    combination.  Fits on training data only; requires no features
    beyond the raw DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class NaiveSeasonal:
    """Predict the sales value from exactly 7 days ago (same weekday).

    This is the simplest meaningful baseline for weekly-seasonal data:
    "next Tuesday will look like last Tuesday."

    At predict time the DataFrame must include a ``lag_7`` column
    (produced by ``build_features``).  Rows where ``lag_7`` is NaN
    (first 7 days of each series) receive the series mean as a fallback.
    """

    def fit(self, df: pd.DataFrame) -> "NaiveSeasonal":
        """Compute per-(store, item) mean sales as a NaN fallback."""
        self._fallback: pd.Series = (
            df.groupby(["store", "item"])["sales"].mean()
        )
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return lag_7 where available, else the training mean."""
        if "lag_7" not in df.columns:
            raise ValueError(
                "NaiveSeasonal.predict requires a 'lag_7' column. "
                "Run build_features() before predicting."
            )
        preds = df["lag_7"].copy().values.astype(float)
        fallback_means = (
            df[["store", "item"]]
            .merge(
                self._fallback.reset_index().rename(columns={"sales": "_fb"}),
                on=["store", "item"],
                how="left",
            )["_fb"]
            .values
        )
        nan_mask = np.isnan(preds)
        preds[nan_mask] = fallback_means[nan_mask]
        return preds


class SeasonalMean:
    """Predict the historical mean sales per (store, item, dayofweek).

    This captures the systematic within-week pattern without requiring
    any lag features, making it a robust baseline even for early rows
    that lack lag history.
    """

    def fit(self, df: pd.DataFrame) -> "SeasonalMean":
        """Compute group means from training data.

        Parameters
        ----------
        df : DataFrame with columns date, store, item, sales.
        """
        df = df.copy()
        df["_dow"] = df["date"].dt.dayofweek
        self._means: pd.Series = (
            df.groupby(["store", "item", "_dow"])["sales"].mean()
        )
        # Also store overall per-series mean as ultimate fallback
        self._series_mean: pd.Series = df.groupby(["store", "item"])["sales"].mean()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Look up the fitted group mean for each row.

        Parameters
        ----------
        df : DataFrame with columns date, store, item.
        """
        df = df.copy()
        df["_dow"] = df["date"].dt.dayofweek
        merged = df[["store", "item", "_dow"]].merge(
            self._means.reset_index().rename(columns={"sales": "_pred"}),
            on=["store", "item", "_dow"],
            how="left",
        )
        preds = merged["_pred"].values.astype(float)
        # Fallback for unseen (store, item, dow) combos
        if np.isnan(preds).any():
            fallback = df[["store", "item"]].merge(
                self._series_mean.reset_index().rename(columns={"sales": "_fb"}),
                on=["store", "item"],
                how="left",
            )["_fb"].values
            nan_mask = np.isnan(preds)
            preds[nan_mask] = fallback[nan_mask]
        return preds
