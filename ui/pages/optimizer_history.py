# ui/pages/optimizer_history.py — Phase 6.9 (Future-Proof)
"""
Optimizer History Dashboard 3.1 (Enhanced)
------------------------------------------
Streamlit dashboard querying the backend /logs/query endpoint
with filtering, pagination, CSV export, and visualization.

Enhancements vs 3.0
-------------------
✓ Fixes Plotly scatter 'size' negative value error
✓ Uses abs(sharpe) for bubble size with color-coded actual Sharpe
✓ Adds basic data validation + chart scaling
✓ Clean rerun compatibility for all Streamlit versions
✓ Ready for Phase 7 analytics (multi-chart extension)
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import plotly.express as px

BACKEND_URL = "http://127.0.0.1:8000"


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def fetch_logs(
    endpoint: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 20,
    offset: int = 0,
    debug: bool = False,
):
    """Query the backend logs endpoint with optional filters."""
    params = {"limit": limit, "offset": offset}
    if endpoint:
        params["endpoint"] = endpoint
    if start:
        params["start"] = start
    if end:
        params["end"] = end

    try:
        r = requests.get(f"{BACKEND_URL}/logs/query", params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if debug:
            st.write("Debug response:", data)
        return data
    except Exception as e:
        st.error(f"⚠️ Failed to load logs: {e}")
        return {"results": [], "count": 0, "total": 0, "last_updated": None}


def format_log_dataframe(records: list[dict]) -> pd.DataFrame:
    """Normalize and flatten nested JSON records for display."""
    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records, sep="_")

    # Expected key columns
    base_cols = [
        "created_at", "endpoint", "tickers",
        "result_sharpe", "result_ann_return",
        "result_ann_vol", "result_summary",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None

    df = df[base_cols + [c for c in df.columns if c not in base_cols]]

    # Format timestamp
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def trigger_rerun():
    """Safe rerun across Streamlit versions."""
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Optimizer History 3.1", layout="wide")

st.title("📜 Optimizer History Dashboard 3.1")
st.caption("View and analyze optimization logs stored in Supabase via AlphaInsights Backend")

# Sidebar filters
st.sidebar.header("🔍 Filters")
endpoint_filter = st.sidebar.selectbox("Endpoint", ["All", "sharpe", "cvar"], index=0)

date_col1, date_col2 = st.sidebar.columns(2)
with date_col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
with date_col2:
    end_date = st.date_input("End Date", datetime.now())

limit = st.sidebar.slider("Rows per page", 5, 100, 20)
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60 seconds", value=False)

if st.sidebar.button("🔄 Manual Refresh"):
    trigger_rerun()

# Query parameters
endpoint = None if endpoint_filter == "All" else endpoint_filter
start_str = start_date.isoformat()
end_str = end_date.isoformat()

# Fetch data
with st.spinner("Fetching logs from backend…"):
    data = fetch_logs(endpoint=endpoint, start=start_str, end=end_str, limit=limit)

# Metrics
meta_col1, meta_col2, meta_col3 = st.columns(3)
meta_col1.metric("Rows Fetched", data.get("count", 0))
meta_col2.metric("Total Logs", data.get("total", 0))
meta_col3.metric("Last Updated", data.get("last_updated", "—"))

# DataFrame
df = format_log_dataframe(data.get("results", []))

if df.empty:
    st.warning("No logs found for selected filters.")
else:
    # CSV Export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name=f"optimizer_logs_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Main Table
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------- #
    # Visualization: Sharpe vs Volatility
    # ------------------------------------------------------------------- #
    if all(col in df.columns for col in ["result_sharpe", "result_ann_vol", "result_ann_return"]):
        df_plot = df.dropna(subset=["result_sharpe", "result_ann_vol", "result_ann_return"])
        if not df_plot.empty:
            # Fix for negative or zero sizes
            df_plot["abs_sharpe"] = df_plot["result_sharpe"].abs().clip(lower=0.05)

            st.subheader("📊 Sharpe Ratio vs Annualized Volatility")
            fig = px.scatter(
                df_plot,
                x="result_ann_vol",
                y="result_ann_return",
                size="abs_sharpe",
                color="result_sharpe",
                color_continuous_scale="RdYlGn",
                hover_data=["endpoint", "created_at", "tickers", "result_summary"],
                labels={
                    "result_ann_vol": "Annualized Volatility (σₐ)",
                    "result_ann_return": "Annualized Return (μₐ)",
                    "result_sharpe": "Sharpe Ratio (SR)"
                },
                title="Portfolio Performance Scatter (Sharpe vs Volatility)",
            )
            fig.update_layout(
                legend_title_text="Sharpe Ratio (color)",
                margin=dict(l=20, r=20, t=60, b=20),
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid Sharpe or volatility data available for visualization.")

# Auto-refresh meta-refresh
if auto_refresh:
    st.markdown(
        "<meta http-equiv='refresh' content='60'>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Backend Source: /logs/query | Phase 6.9 — Stable version with chart fixes, CSV export, and future analytics hooks."
)
