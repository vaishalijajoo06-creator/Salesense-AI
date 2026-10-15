"""
linear.py
---------
Ridge regression model for sales forecasting.

Uses all engineered features from features.py (calendar + lag + rolling).
Fits a single global Ridge model across all store-item series, with
store and item as integer features so the model can learn per-series
offsets without a separate model per series.

This is efficient (one sklearn fit on ~885 k rows), interpretable, and
provides a strong regularised linear baseline above the naive models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS, get_feature_cols


class RidgeForecaster:
    """Global Ridge regression across all store-item series.

    Parameters
    ----------
    alpha : float  L2 regularisation strength (default 1.0).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._scaler = StandardScaler()
        self._model: Ridge | None = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _prepare_X(self, df: pd.DataFrame, fit_scaler: bool = False) -> np.ndarray:
        """Extract and scale the feature matrix."""
        # Core engineered features that exist in df
        base_cols = get_feature_cols(df)
        # Add store and item as integer features (global cross-series offsets)
        id_cols = [c for c in ("store", "item") if c in df.columns]
        cols = id_cols + base_cols

        # Drop rows where any lag/rolling feature is NaN (early rows)
        X = df[cols].copy()
        if fit_scaler:
            self._feature_cols = cols
            X_scaled = self._scaler.fit_transform(X.fillna(0))
        else:
            X_scaled = self._scaler.transform(X[self._feature_cols].fillna(0))
        return X_scaled

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "RidgeForecaster":
        """Fit on rows that have complete feature history.

        Parameters
        ----------
        df : DataFrame produced by build_features() — training split only.
        y  : ignored if 'sales' is in df; provided for API symmetry.
        """
        if y is None:
            y = df["sales"]

        # Drop rows with any NaN in features (early history rows)
        feature_mask = df[get_feature_cols(df)].notna().all(axis=1)
        df_clean = df[feature_mask]
        y_clean  = y[feature_mask]

        X = self._prepare_X(df_clean, fit_scaler=True)
        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, y_clean.values)
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict for all rows in df (NaN features filled with 0)."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X = self._prepare_X(df, fit_scaler=False)
        preds = self._model.predict(X)
        return np.clip(preds, 0, None)   # sales cannot be negative
