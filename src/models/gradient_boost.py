"""
gradient_boost.py
-----------------
LightGBM gradient boosting model for sales forecasting.

Trains a single global LightGBM regressor over all store-item series,
with store and item as categorical features.  LightGBM handles this
efficiently and learns per-series patterns via split gains on the ID
columns.

LightGBM is checked via ``importlib`` at import time; if it is not
installed an ``ImportError`` is raised with a clear install instruction.
"""

from __future__ import annotations

import importlib
import numpy as np
import pandas as pd

from src.features import get_feature_cols


def _lgbm():
    """Lazy import of lightgbm — raises ImportError with clear message if absent."""
    spec = importlib.util.find_spec("lightgbm")
    if spec is None:
        raise ImportError(
            "lightgbm is not installed. "
            "Run: pip install lightgbm"
        )
    import lightgbm as lgb
    return lgb


class LGBMForecaster:
    """Global LightGBM regressor across all store-item series.

    Parameters
    ----------
    n_estimators   : number of boosting rounds.
    learning_rate  : shrinkage factor.
    num_leaves     : maximum leaves per tree (controls complexity).
    random_state   : reproducibility seed.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves    = num_leaves
        self.random_state  = random_state
        self._model        = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    def _get_cols(self, df: pd.DataFrame) -> list[str]:
        id_cols   = [c for c in ("store", "item") if c in df.columns]
        feat_cols = get_feature_cols(df)
        return id_cols + feat_cols

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: pd.Series | None = None) -> "LGBMForecaster":
        """Fit on rows with complete feature history.

        Parameters
        ----------
        df : DataFrame from build_features() — training split only.
        y  : optional; if omitted uses df["sales"].
        """
        lgb = _lgbm()

        if y is None:
            y = df["sales"]

        feat_mask = df[get_feature_cols(df)].notna().all(axis=1)
        df_clean  = df[feat_mask]
        y_clean   = y[feat_mask]

        self._feature_cols = self._get_cols(df_clean)
        X = df_clean[self._feature_cols].fillna(0).values

        self._model = lgb.LGBMRegressor(
            n_estimators  = self.n_estimators,
            learning_rate = self.learning_rate,
            num_leaves    = self.num_leaves,
            random_state  = self.random_state,
            n_jobs        = -1,
            verbose       = -1,
        )
        self._model.fit(X, y_clean.values)
        return self

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict for all rows in df."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X = df[self._feature_cols].fillna(0).values
        preds = self._model.predict(X)
        return np.clip(preds, 0, None)
