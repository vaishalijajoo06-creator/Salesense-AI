"""
visualizations.py
-----------------
Reusable Plotly figure helpers shared across the dashboard pages.

These helpers are distinct from eda.py in that they operate on
*model outputs* (predictions vs actuals) rather than raw data.

Stage 6 will implement the full chart bodies.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: pd.Series,
    y_pred: pd.Series,
    model_name: str = "Model",
) -> go.Figure:
    """Overlay actual and predicted sales on a time axis."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_true, mode="lines", name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name=f"Predicted ({model_name})", line=dict(color="#ff7f0e", dash="dash")))
    fig.update_layout(title=f"Actual vs Predicted — {model_name}", xaxis_title="Date", yaxis_title="Daily Sales")
    return fig


def plot_forecast_with_history(
    history_dates: pd.Series,
    history_sales: pd.Series,
    forecast_dates: pd.Series,
    forecast_sales: pd.Series,
) -> go.Figure:
    """Plot historical sales followed by the forecast horizon."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_dates, y=history_sales, mode="lines", name="Historical", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_sales, mode="lines", name="Forecast", line=dict(color="#2ca02c", dash="dot")))
    fig.update_layout(title="Historical Sales & Forecast", xaxis_title="Date", yaxis_title="Daily Sales")
    return fig


def plot_metrics_bar(metrics_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing model metrics side by side."""
    fig = go.Figure()
    for metric in ["SMAPE", "RMSE", "MAE"]:
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(name=metric, x=metrics_df["model"], y=metrics_df[metric]))
    fig.update_layout(
        barmode="group",
        title="Model Comparison — Evaluation Metrics",
        xaxis_title="Model",
        yaxis_title="Score (lower is better)",
    )
    return fig
