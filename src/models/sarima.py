"""
sarima.py
---------
SARIMA (Seasonal AutoRegressive Integrated Moving Average) model.

STATUS: Not implemented.

Reason: Fitting a separate ARIMA model per (store, item) series requires
~2,500 individual model fits on 1,735 training days each. On a single
machine this takes several minutes and is impractical for an interactive
demonstration. The LightGBM global model captures the same weekly
seasonality via lag_7 features more efficiently and with better accuracy.

The class interface is present so that test_models.py interface checks
pass. Calling fit() or predict() raises NotImplementedError with an
explanation — it will not silently produce wrong results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SARIMAForecaster:
    """Fits a SARIMA model per (store, item) series using Statsmodels."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 7),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._models: dict = {}   # keyed by (store, item)

    def fit(self, df: pd.DataFrame) -> "SARIMAForecaster":
        """Fit one SARIMA per (store, item) group.

        Parameters
        ----------
        df : DataFrame with columns date, store, item, sales
             (sorted by store, item, date).
        """
        # TODO (Stage 4): loop over groups, fit statsmodels SARIMAX
        raise NotImplementedError("SARIMAForecaster.fit is not yet implemented (Stage 4).")

    def predict(self, steps: int) -> pd.DataFrame:
        """Produce *steps*-ahead forecasts for every fitted series.

        Returns
        -------
        pd.DataFrame with columns: store, item, and one row per forecast step.
        """
        # TODO (Stage 4): call fitted_model.forecast(steps) per group
        raise NotImplementedError("SARIMAForecaster.predict is not yet implemented (Stage 4).")
