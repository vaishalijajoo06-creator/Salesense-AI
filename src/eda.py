"""
eda.py
------
Exploratory Data Analysis helpers for SalesSense AI.

Every public function accepts pre-computed aggregates (produced by
data_loader.compute_eda_stats) and/or the raw training DataFrame, and
returns a Plotly Figure.  This keeps the chart logic pure and testable:
no file I/O, no Streamlit calls.

Chart inventory
---------------
plot_daily_sales_trend      – line chart: total daily sales over 5 years
plot_monthly_seasonality    – bar chart: avg sales per calendar month
plot_weekday_pattern        – bar chart: avg sales per day-of-week
plot_sales_distribution     – histogram: distribution of daily per-series sales
plot_store_comparison       – bar chart: total sales per store
plot_item_comparison        – bar chart: total sales per item (all items)
plot_top10_items            – horizontal bar: top 10 items by total sales
plot_store_item_heatmap     – heatmap: avg sales for each store × item pair
plot_yearly_overlay         – multi-line: daily sales overlaid per year
plot_single_series          – line chart for one (store, item) pair
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Day-of-week label map (pandas dayofweek: 0 = Monday)
_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May",
                 "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# 1. Daily sales trend
# ---------------------------------------------------------------------------

def plot_daily_sales_trend(daily: pd.Series) -> go.Figure:
    """Line chart of total daily sales across all stores and items.

    Parameters
    ----------
    daily : pd.Series  index=date, values=total daily sales
                       (from compute_eda_stats()["daily"])
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index,
        y=daily.values,
        mode="lines",
        name="Daily Total Sales",
        line=dict(color="#1f77b4", width=1),
    ))
    # 30-day rolling average overlay
    rolling = daily.rolling(30, center=True).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index,
        y=rolling.values,
        mode="lines",
        name="30-day Rolling Avg",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.update_layout(
        title="Daily Total Sales — All Stores & Items (2013–2017)",
        xaxis_title="Date",
        yaxis_title="Total Units Sold",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Monthly seasonality
# ---------------------------------------------------------------------------

def plot_monthly_seasonality(monthly: pd.Series) -> go.Figure:
    """Bar chart of average daily sales per calendar month.

    Parameters
    ----------
    monthly : pd.Series  index=month (1–12), values=avg sales
                         (from compute_eda_stats()["monthly"])
    """
    labels = [_MONTH_LABELS[m - 1] for m in monthly.index]
    fig = go.Figure(go.Bar(
        x=labels,
        y=monthly.values,
        marker_color="#2ca02c",
        text=[f"{v:.1f}" for v in monthly.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Daily Sales by Month (Seasonal Pattern)",
        xaxis_title="Month",
        yaxis_title="Avg Units Sold per Store-Item",
        yaxis=dict(range=[0, monthly.max() * 1.15]),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Weekday pattern
# ---------------------------------------------------------------------------

def plot_weekday_pattern(by_weekday: pd.Series) -> go.Figure:
    """Bar chart of average daily sales per day-of-week.

    Parameters
    ----------
    by_weekday : pd.Series  index=dayofweek (0=Mon), values=avg sales
                            (from compute_eda_stats()["by_weekday"])
    """
    labels = [_DOW_LABELS[d] for d in by_weekday.index]
    colors = ["#d62728" if d >= 5 else "#1f77b4" for d in by_weekday.index]
    fig = go.Figure(go.Bar(
        x=labels,
        y=by_weekday.values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in by_weekday.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Average Daily Sales by Day of Week",
        xaxis_title="Day of Week  (red = weekend)",
        yaxis_title="Avg Units Sold",
        yaxis=dict(range=[0, by_weekday.max() * 1.15]),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Sales distribution
# ---------------------------------------------------------------------------

def plot_sales_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of per-row (store-item-day) sales values.

    Parameters
    ----------
    df : raw training DataFrame with a 'sales' column
    """
    fig = go.Figure(go.Histogram(
        x=df["sales"],
        nbinsx=60,
        marker_color="#9467bd",
        opacity=0.8,
    ))
    mean_val = float(df["sales"].mean())
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#d62728",
        annotation_text=f"Mean = {mean_val:.1f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Distribution of Daily Sales per Store-Item",
        xaxis_title="Daily Sales (units)",
        yaxis_title="Frequency",
        bargap=0.05,
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Store comparison
# ---------------------------------------------------------------------------

def plot_store_comparison(by_store_total: pd.Series) -> go.Figure:
    """Bar chart of total sales per store.

    Parameters
    ----------
    by_store_total : pd.Series  index=store, values=total sales
                                (from compute_eda_stats()["by_store_total"])
    """
    fig = go.Figure(go.Bar(
        x=[f"Store {s}" for s in by_store_total.index],
        y=by_store_total.values,
        marker_color="#17becf",
        text=[f"{v:,}" for v in by_store_total.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Total Sales by Store (2013–2017)",
        xaxis_title="Store",
        yaxis_title="Total Units Sold",
        yaxis=dict(range=[0, by_store_total.max() * 1.12]),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Item comparison (all items)
# ---------------------------------------------------------------------------

def plot_item_comparison(by_item_total: pd.Series) -> go.Figure:
    """Bar chart of total sales per item.

    Parameters
    ----------
    by_item_total : pd.Series  index=item, values=total sales
                               (from compute_eda_stats()["by_item_total"])
    """
    fig = go.Figure(go.Bar(
        x=[f"Item {i}" for i in by_item_total.index],
        y=by_item_total.values,
        marker_color="#bcbd22",
    ))
    fig.update_layout(
        title="Total Sales by Item (2013–2017)",
        xaxis_title="Item",
        yaxis_title="Total Units Sold",
        xaxis=dict(tickangle=-45),
    )
    return fig


# ---------------------------------------------------------------------------
# 7. Top-10 items
# ---------------------------------------------------------------------------

def plot_top10_items(top10_items: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of the top 10 items by total sales.

    Parameters
    ----------
    top10_items : pd.DataFrame  columns: item, total_sales
                                (from compute_eda_stats()["top10_items"])
    """
    df = top10_items.sort_values("total_sales")
    fig = go.Figure(go.Bar(
        y=[f"Item {i}" for i in df["item"]],
        x=df["total_sales"],
        orientation="h",
        marker_color="#e377c2",
        text=[f"{v:,}" for v in df["total_sales"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 10 Items by Total Sales",
        xaxis_title="Total Units Sold",
        yaxis_title="Item",
        xaxis=dict(range=[0, df["total_sales"].max() * 1.12]),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Store × Item heatmap
# ---------------------------------------------------------------------------

def plot_store_item_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of average daily sales for each store × item combination.

    Parameters
    ----------
    df : raw training DataFrame
    """
    pivot = df.groupby(["store", "item"])["sales"].mean().unstack(level="item")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"Item {c}" for c in pivot.columns],
        y=[f"Store {r}" for r in pivot.index],
        colorscale="YlOrRd",
        colorbar=dict(title="Avg Daily Sales"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Avg Daily Sales Heatmap — Store × Item",
        xaxis_title="Item",
        yaxis_title="Store",
        xaxis=dict(tickangle=-45),
    )
    return fig


# ---------------------------------------------------------------------------
# 9. Yearly overlay (growth over years)
# ---------------------------------------------------------------------------

def plot_yearly_overlay(df: pd.DataFrame) -> go.Figure:
    """Overlay total daily sales for each year on a single Jan–Dec axis.

    This reveals year-on-year growth and confirms seasonality repeats.

    Parameters
    ----------
    df : raw training DataFrame
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear

    daily_year = (
        df.groupby(["year", "dayofyear"])["sales"]
        .sum()
        .reset_index()
    )

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, year in enumerate(sorted(daily_year["year"].unique())):
        subset = daily_year[daily_year["year"] == year]
        fig.add_trace(go.Scatter(
            x=subset["dayofyear"],
            y=subset["sales"],
            mode="lines",
            name=str(year),
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))
    fig.update_layout(
        title="Year-on-Year Daily Sales Overlay",
        xaxis_title="Day of Year",
        yaxis_title="Total Units Sold",
        legend_title="Year",
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Single store-item series
# ---------------------------------------------------------------------------

def plot_single_series(df: pd.DataFrame, store: int, item: int) -> go.Figure:
    """Line chart for a single (store, item) time series.

    Parameters
    ----------
    df    : raw training DataFrame
    store : store id
    item  : item id
    """
    series = (
        df[(df["store"] == store) & (df["item"] == item)]
        .sort_values("date")
        .set_index("date")["sales"]
    )
    if series.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No data for Store {store}, Item {item}")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode="lines",
        name=f"Store {store} · Item {item}",
        line=dict(color="#1f77b4", width=1),
    ))
    rolling = series.rolling(14, center=True).mean()
    fig.add_trace(go.Scatter(
        x=rolling.index,
        y=rolling.values,
        mode="lines",
        name="14-day Rolling Avg",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=f"Daily Sales — Store {store}, Item {item}",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
    )
    return fig
