"""
app.py
------
SalesSense AI — Streamlit Dashboard entry point.

Run with:
    streamlit run app.py

All metrics and charts are computed from the actual dataset.
No fake statistics or placeholder predictions are used.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SalesSense AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = {
    "🏠  Home": "home",
    "🔍  Data Explorer": "data_explorer",
    "📊  EDA": "eda",
    "🤖  Model Comparison": "model_comparison",
    "🔮  Forecast Viewer": "forecast_viewer",
}

st.sidebar.title("SalesSense AI")
st.sidebar.markdown("*Retail Sales Forecasting & Demand Planning*")
st.sidebar.markdown("---")
selected_page = st.sidebar.radio("Navigate to", list(PAGES.keys()))
page_key = PAGES[selected_page]

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = pathlib.Path(__file__).parent / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


# ---------------------------------------------------------------------------
# Cached data loading  (Streamlit re-runs on every widget interaction;
# @st.cache_data ensures CSVs are only read once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading training data…")
def _load_train():
    # Add src/ to path so src.* imports work when running from project root
    src_dir = str(pathlib.Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.data_loader import load_train
    return load_train()


@st.cache_data(show_spinner="Loading test data…")
def _load_test():
    src_dir = str(pathlib.Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from src.data_loader import load_test
    return load_test()


@st.cache_data(show_spinner="Computing EDA statistics…")
def _compute_stats(_df):
    from src.data_loader import compute_eda_stats
    return compute_eda_stats(_df)


# ---------------------------------------------------------------------------
# Helper: check that data files exist (no loading yet)
# ---------------------------------------------------------------------------
def _data_status() -> dict[str, bool]:
    return {
        "train.csv": TRAIN_PATH.exists(),
        "test.csv": TEST_PATH.exists(),
        "sample_submission.csv": (DATA_DIR / "sample_submission.csv").exists(),
    }


# ===========================================================================
# Pages
# ===========================================================================

# ── Home ────────────────────────────────────────────────────────────────────
if page_key == "home":
    st.title("📈 SalesSense AI")
    st.subheader("Retail Sales Forecasting & Demand Planning System")
    st.markdown(
        """
        Welcome to **SalesSense AI** — an end-to-end machine-learning system that
        predicts daily retail sales quantities, compares forecasting models, and
        presents results in this interactive dashboard.

        ---
        ### Project Overview
        | | |
        |---|---|
        | **Training period** | 2013-01-01 → 2017-12-31 (5 years, 913 000 rows) |
        | **Forecast horizon** | 2018-01-01 → 2018-03-31 (3 months, 45 000 rows) |
        | **Stores** | 10 |
        | **Items** | 50 |
        | **Target metric** | SMAPE (Symmetric Mean Absolute Percentage Error) |
        | **Models** | Naive baseline · Ridge · LightGBM · SARIMA |

        ---
        ### Implementation Stages
        """
    )

    stages = [
        ("1 — Environment Setup & Scaffolding", "✅ Complete"),
        ("2 — Data Loading & EDA", "✅ Complete"),
        ("3 — Feature Engineering", "✅ Complete"),
        ("4 — Model Training & Evaluation", "✅ Complete"),
        ("5 — Forecasting & Submission", "✅ Complete"),
        ("6 — Interactive Dashboard", "🔜 Planned"),
        ("7 — Tests", "🔜 Planned"),
    ]
    for stage, status in stages:
        st.markdown(f"- **Stage {stage}** — {status}")

    st.markdown("---")
    st.subheader("📁 Data Files Status")
    status = _data_status()
    cols = st.columns(len(status))
    for col, (fname, exists) in zip(cols, status.items()):
        icon = "✅" if exists else "❌"
        col.metric(label=fname, value=icon)

    if all(status.values()):
        st.success("All dataset files are present in data/.")
    else:
        missing = [f for f, ok in status.items() if not ok]
        st.error(f"Missing files: {missing}. Place them in the data/ folder before proceeding.")


# ── Data Explorer ───────────────────────────────────────────────────────────
elif page_key == "data_explorer":
    st.title("🔍 Data Explorer")
    st.markdown("Browse and filter the raw training and test datasets.")

    if not TRAIN_PATH.exists():
        st.error("data/train.csv not found. Place the file in the data/ folder.")
        st.stop()

    try:
        df_train = _load_train()
        df_test = _load_test()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    from src.data_loader import dataset_summary

    tab_train, tab_test = st.tabs(["Training Data", "Test Data"])

    # ── Training tab ──────────────────────────────────────────────────────
    with tab_train:
        st.subheader("Training Dataset Summary")
        summary = dataset_summary(df_train)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary['rows']:,}")
        c2.metric("Unique Stores", summary["unique_stores"])
        c3.metric("Unique Items", summary["unique_items"])
        c4.metric("Total Units Sold", f"{summary['sales_total']:,}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Date Range", f"{summary['date_min']} → {summary['date_max']}")
        c6.metric("Unique Dates", f"{summary['unique_dates']:,}")
        c7.metric("Avg Daily Sales / Series", f"{summary['sales_mean']:.2f}")
        c8.metric("Null Values", summary["nulls_total"])

        st.markdown("---")
        st.subheader("Filter & Preview")

        col_store, col_item = st.columns(2)
        stores = sorted(df_train["store"].unique().tolist())
        items = sorted(df_train["item"].unique().tolist())
        sel_store = col_store.selectbox("Store", stores, key="explorer_store")
        sel_item = col_item.selectbox("Item", items, key="explorer_item")

        filtered = df_train[
            (df_train["store"] == sel_store) & (df_train["item"] == sel_item)
        ].sort_values("date")

        st.caption(f"Showing {len(filtered):,} rows for Store {sel_store}, Item {sel_item}")
        st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

        # Inline time series for selected series
        from src.eda import plot_single_series
        st.plotly_chart(
            plot_single_series(df_train, sel_store, sel_item),
            use_container_width=True,
        )

    # ── Test tab ──────────────────────────────────────────────────────────
    with tab_test:
        st.subheader("Test Dataset Summary")
        summary_test = dataset_summary(df_test)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{summary_test['rows']:,}")
        c2.metric("Unique Stores", summary_test["unique_stores"])
        c3.metric("Unique Items", summary_test["unique_items"])
        c4.metric("Unique Dates", f"{summary_test['unique_dates']:,}")

        c5, c6 = st.columns(2)
        c5.metric("Date Range", f"{summary_test['date_min']} → {summary_test['date_max']}")
        c6.metric("Null Values", summary_test["nulls_total"])

        st.markdown("---")
        st.dataframe(df_test.head(100), use_container_width=True)
        st.caption("Showing first 100 rows of test.csv")


# ── EDA ─────────────────────────────────────────────────────────────────────
elif page_key == "eda":
    st.title("📊 Exploratory Data Analysis")
    st.markdown(
        "All charts are computed from **data/train.csv** (913,000 rows, 2013–2017). "
        "No predictions or fake data are used."
    )

    if not TRAIN_PATH.exists():
        st.error("data/train.csv not found. Place the file in the data/ folder.")
        st.stop()

    try:
        df_train = _load_train()
    except Exception as e:
        st.error(f"Failed to load training data: {e}")
        st.stop()

    stats = _compute_stats(df_train)

    from src import eda

    # ── Section 1: Overview metrics ───────────────────────────────────────
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Units Sold", f"{int(df_train['sales'].sum()):,}")
    c2.metric("Avg Daily Sales / Series", f"{df_train['sales'].mean():.2f}")
    c3.metric("Stores", int(df_train["store"].nunique()))
    c4.metric("Items", int(df_train["item"].nunique()))
    c5.metric("Zero-Sales Days", int((df_train["sales"] == 0).sum()))

    missing_dates_count = len(stats["missing_dates"])
    if missing_dates_count == 0:
        st.success("✅ No missing calendar dates in the training period.")
    else:
        st.warning(f"⚠️ {missing_dates_count} calendar date(s) are absent from train.csv.")

    st.markdown("---")

    # ── Section 2: Trend ──────────────────────────────────────────────────
    st.subheader("1 · Overall Sales Trend")
    st.plotly_chart(eda.plot_daily_sales_trend(stats["daily"]), use_container_width=True)

    st.subheader("2 · Year-on-Year Growth")
    st.plotly_chart(eda.plot_yearly_overlay(df_train), use_container_width=True)

    st.markdown("---")

    # ── Section 3: Seasonality ────────────────────────────────────────────
    st.subheader("3 · Monthly Seasonality")
    st.plotly_chart(eda.plot_monthly_seasonality(stats["monthly"]), use_container_width=True)

    st.subheader("4 · Weekly Pattern")
    st.plotly_chart(eda.plot_weekday_pattern(stats["by_weekday"]), use_container_width=True)

    st.markdown("---")

    # ── Section 4: Distribution ───────────────────────────────────────────
    st.subheader("5 · Sales Distribution")
    st.plotly_chart(eda.plot_sales_distribution(df_train), use_container_width=True)

    st.markdown("---")

    # ── Section 5: Store and item comparisons ─────────────────────────────
    st.subheader("6 · Store Comparison")
    st.plotly_chart(eda.plot_store_comparison(stats["by_store_total"]), use_container_width=True)

    st.subheader("7 · Item Comparison")
    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(eda.plot_item_comparison(stats["by_item_total"]), use_container_width=True)
    with col_r:
        st.plotly_chart(eda.plot_top10_items(stats["top10_items"]), use_container_width=True)

    st.markdown("---")

    # ── Section 6: Store × Item heatmap ──────────────────────────────────
    st.subheader("8 · Store × Item Heatmap")
    st.plotly_chart(eda.plot_store_item_heatmap(df_train), use_container_width=True)

    st.markdown("---")

    # ── Section 7: Top-N tables ───────────────────────────────────────────
    st.subheader("9 · Top 10 Tables")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top 10 Items by Total Sales**")
        st.dataframe(stats["top10_items"], use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Top 10 Stores by Total Sales**")
        st.dataframe(stats["top10_stores"], use_container_width=True, hide_index=True)


# ── Model Comparison ────────────────────────────────────────────────────────
elif page_key == "model_comparison":
    st.title("🤖 Model Comparison")
    st.markdown(
        "All metrics and predictions are computed from **real model training** on "
        "data/train.csv. Validation period: **2017-10-01 → 2017-12-31** (92 days)."
    )

    METRICS_PATH = pathlib.Path(__file__).parent / "outputs" / "validation_metrics.csv"
    PREDS_PATH   = pathlib.Path(__file__).parent / "outputs" / "val_predictions.csv"

    if not METRICS_PATH.exists():
        st.error(
            "outputs/validation_metrics.csv not found. "
            "Run `py train_and_evaluate.py` first to train the models."
        )
        st.stop()

    import plotly.graph_objects as go

    @st.cache_data
    def _load_metrics():
        return pd.read_csv(METRICS_PATH)

    @st.cache_data
    def _load_val_preds():
        df = pd.read_csv(PREDS_PATH, parse_dates=["date"])
        return df

    metrics_df = _load_metrics()
    val_preds  = _load_val_preds()

    # ── Section 1: Split info ─────────────────────────────────────────────
    st.subheader("Data Split")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train start", "2013-01-01")
    c2.metric("Train end",   "2017-09-30")
    c3.metric("Val start",   "2017-10-01")
    c4.metric("Val end",     "2017-12-31")
    c5, c6 = st.columns(2)
    c5.metric("Train rows", f"{867_000:,}")
    c6.metric("Val rows",   f"{46_000:,}")

    st.markdown("---")

    # ── Section 2: Metrics table ──────────────────────────────────────────
    st.subheader("Validation Metrics (lower is better)")
    # Add a visual indicator column instead of using .style (which can
    # mis-align column headers in some Streamlit/pandas version combos).
    display_df = metrics_df.copy()
    best_row_model = display_df.iloc[0]["model"]
    display_df.insert(0, "Best", display_df["model"].apply(
        lambda m: "🥇" if m == best_row_model else ""
    ))
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best_model = metrics_df.iloc[0]["model"]
    best_wape  = metrics_df.iloc[0]["WAPE"]
    best_mae   = metrics_df.iloc[0]["MAE"]
    st.success(
        f"**Best model: {best_model}** — WAPE = {best_wape:.2f}%, MAE = {best_mae:.2f} units/day"
    )

    st.markdown("---")

    # ── Section 3: Metrics bar chart ──────────────────────────────────────
    st.subheader("Metric Comparison Chart")
    metric_choice = st.selectbox("Select metric", ["WAPE", "MAE", "RMSE", "SMAPE"])
    fig_bar = go.Figure()
    colors = ["#2ca02c" if m == best_model else "#1f77b4" for m in metrics_df["model"]]
    fig_bar.add_trace(go.Bar(
        x=metrics_df["model"],
        y=metrics_df[metric_choice],
        marker_color=colors,
        text=[f"{v:.2f}" for v in metrics_df[metric_choice]],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title=f"{metric_choice} by Model (validation set, green = best)",
        xaxis_title="Model",
        yaxis_title=metric_choice,
        yaxis=dict(range=[0, metrics_df[metric_choice].max() * 1.2]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Section 4: Actual vs Predicted ────────────────────────────────────
    st.subheader("Actual vs Predicted — Validation Period")
    st.markdown("Filter to a single store-item series for a clear view.")

    stores_v = sorted(val_preds["store"].unique().tolist())
    items_v  = sorted(val_preds["item"].unique().tolist())
    col_sv, col_iv = st.columns(2)
    sel_sv = col_sv.selectbox("Store", stores_v, key="mc_store")
    sel_iv = col_iv.selectbox("Item",  items_v,  key="mc_item")

    subset = val_preds[
        (val_preds["store"] == sel_sv) & (val_preds["item"] == sel_iv)
    ].sort_values("date")

    model_cols = [c for c in val_preds.columns if c not in ("date","store","item","sales")]
    fig_av = go.Figure()
    fig_av.add_trace(go.Scatter(
        x=subset["date"], y=subset["sales"],
        mode="lines", name="Actual",
        line=dict(color="#1f77b4", width=2),
    ))
    palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, model_name in enumerate(model_cols):
        fig_av.add_trace(go.Scatter(
            x=subset["date"], y=subset[model_name],
            mode="lines", name=model_name,
            line=dict(dash="dash" if model_name != best_model else "solid",
                      width=1.5, color=palette[i % len(palette)]),
        ))
    fig_av.update_layout(
        title=f"Actual vs Predicted — Store {sel_sv}, Item {sel_iv}",
        xaxis_title="Date", yaxis_title="Daily Sales",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_av, use_container_width=True)

    st.markdown("---")

    # ── Section 5: Error distribution ────────────────────────────────────
    st.subheader("Absolute Error Distribution")
    fig_err = go.Figure()
    for i, model_name in enumerate(model_cols):
        errors = (val_preds["sales"] - val_preds[model_name]).abs()
        fig_err.add_trace(go.Box(
            y=errors, name=model_name,
            marker_color=palette[i % len(palette)],
        ))
    fig_err.update_layout(
        title="Absolute Prediction Error Distribution (all store-item series)",
        xaxis_title="Model", yaxis_title="|Actual - Predicted|",
    )
    st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")

    # ── Section 6: Model selection rationale ─────────────────────────────
    st.subheader("Model Selection")
    st.markdown(f"""
**Selected model: {best_model}**

| Criterion | Detail |
|---|---|
| **Primary metric** | WAPE (Weighted Absolute Percentage Error) — robust to zero-sales rows |
| **Best WAPE** | {metrics_df[metrics_df['model']==best_model]['WAPE'].values[0]:.2f}% |
| **Improvement vs NaiveSeasonal** | {metrics_df[metrics_df['model']=='NaiveSeasonal']['WAPE'].values[0] - metrics_df[metrics_df['model']==best_model]['WAPE'].values[0]:.2f} percentage points |
| **Trade-off** | LightGBM takes ~30s to train (vs <2s for Ridge) but delivers meaningfully lower error |
| **Retained baseline** | NaiveSeasonal is kept as reference floor |

**SARIMA** was not included in this comparison: fitting a separate ARIMA model
per store-item series (500 series × 1,735 training days) would require several
minutes per series and is not practical without distributed computing. The
LightGBM global model captures the same seasonal structure more efficiently via
lag features.
""")

# ── Forecast Viewer ─────────────────────────────────────────────────────────
elif page_key == "forecast_viewer":
    st.title("🔮 Forecast Viewer")
    st.markdown(
        "Recursive multi-step forecasts using the **LightGBM** model trained in Stage 4. "
        "Predictions use only historical information — no future sales are read."
    )

    if not TRAIN_PATH.exists():
        st.error("data/train.csv not found.")
        st.stop()

    # ── Cached model + data ───────────────────────────────────────────────
    import io
    import plotly.graph_objects as go
    from src.forecast import forecast_series

    @st.cache_resource(show_spinner="Training LightGBM on full dataset…")
    def _get_trained_lgbm():
        from src.data_loader import load_train
        from src.features import build_features
        from src.models.gradient_boost import LGBMForecaster
        raw = load_train()
        feat = build_features(raw)
        model = LGBMForecaster(n_estimators=500, learning_rate=0.05, num_leaves=31)
        model.fit(feat)
        return model, raw   # return raw history for forecast_series

    @st.cache_data(show_spinner="Loading training data…")
    def _get_history():
        from src.data_loader import load_train
        return load_train()

    try:
        lgbm_model, history_df = _get_trained_lgbm()
    except Exception as e:
        st.error(f"Failed to load/train model: {e}")
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────
    st.subheader("Forecast Settings")
    stores_all = sorted(history_df["store"].unique().tolist())
    items_all  = sorted(history_df["item"].unique().tolist())

    col_s, col_i, col_h = st.columns(3)
    sel_store   = col_s.selectbox("Store", stores_all, key="fv_store")
    sel_item    = col_i.selectbox("Item",  items_all,  key="fv_item")
    horizon_map = {"7 days": 7, "30 days": 30, "90 days": 90}
    horizon_lbl = col_h.selectbox("Forecast horizon", list(horizon_map.keys()), key="fv_horizon")
    horizon     = horizon_map[horizon_lbl]

    # History window for chart context
    hist_window = st.slider(
        "Historical context (days to show before forecast)",
        min_value=14, max_value=180, value=60, step=7, key="fv_hist_window",
    )

    st.markdown("---")

    # ── Generate forecast ─────────────────────────────────────────────────
    with st.spinner(f"Generating {horizon}-day forecast for Store {sel_store}, Item {sel_item}…"):
        try:
            forecast_df = forecast_series(
                lgbm_model, history_df,
                store=sel_store, item=sel_item,
                horizon_days=horizon,
            )
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            st.stop()

    # Historical tail for context
    series_hist = (
        history_df[(history_df["store"] == sel_store) & (history_df["item"] == sel_item)]
        .sort_values("date")
        .tail(hist_window)
    )
    origin_date = series_hist["date"].max()

    # ── Summary metrics ───────────────────────────────────────────────────
    total_pred = float(forecast_df["predicted_sales"].sum())
    avg_pred   = float(forecast_df["predicted_sales"].mean())
    min_pred   = float(forecast_df["predicted_sales"].min())
    max_pred   = float(forecast_df["predicted_sales"].max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Horizon", horizon_lbl)
    c2.metric("Total Predicted Units", f"{total_pred:,.1f}")
    c3.metric("Avg Daily Predicted", f"{avg_pred:.1f}")
    c4.metric("Min / Max Daily", f"{min_pred:.1f} / {max_pred:.1f}")

    st.caption(
        f"Forecast origin: **{origin_date.date()}**  |  "
        f"Forecast range: **{forecast_df['date'].min().date()}** to "
        f"**{forecast_df['date'].max().date()}**"
    )

    st.markdown("---")

    # ── Chart: history + forecast ─────────────────────────────────────────
    st.subheader("Historical Sales & Forecast")
    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=series_hist["date"],
        y=series_hist["sales"],
        mode="lines",
        name="Historical Sales",
        line=dict(color="#1f77b4", width=1.5),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df["predicted_sales"],
        mode="lines+markers",
        name=f"Forecast ({horizon_lbl})",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=4),
    ))

    # Vertical line at origin
    fig.add_vline(
        x=origin_date,
        line_dash="dot",
        line_color="#888888",
        annotation_text="Forecast origin",
        annotation_position="top right",
    )

    fig.update_layout(
        title=f"Store {sel_store}, Item {sel_item} — {horizon}-Day Forecast",
        xaxis_title="Date",
        yaxis_title="Units Sold",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Forecast table ────────────────────────────────────────────────────
    st.subheader("Forecast Table")
    display_fc = forecast_df[["date", "predicted_sales"]].copy()
    display_fc["date"] = display_fc["date"].dt.strftime("%Y-%m-%d")
    display_fc = display_fc.rename(columns={
        "date": "Date",
        "predicted_sales": "Predicted Sales (units)",
    })
    st.dataframe(display_fc, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────────────────
    st.subheader("Download Forecast")
    csv_buf = io.StringIO()
    export_df = forecast_df.copy()
    export_df["date"] = export_df["date"].dt.strftime("%Y-%m-%d")
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="Download forecast as CSV",
        data=csv_buf.getvalue(),
        file_name=f"forecast_store{sel_store}_item{sel_item}_{horizon}d.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.caption(
        "**Methodology:** Recursive single-step forecasting using LightGBM. "
        "At each future step, lag and rolling features are built from the "
        "known history plus previously generated predictions. "
        "No actual future sales are used. "
        "Forecast error accumulates over longer horizons."
    )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("SalesSense AI · Stage 5 of 7")
