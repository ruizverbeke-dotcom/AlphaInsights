import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
"""
📜 Optimizer History Dashboard — AlphaInsights
----------------------------------------------

View and analyze optimization logs stored in Supabase via AlphaInsights Backend.

Backed by:
- /logs/query  (filtered, paginated)
- /logs/recent (simple tail view, optional)

Uses unified backend fetch layer:
- core.ui_helpers.fetch_backend
- core.ui_config.BACKEND_URL
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Optimizer History", layout="wide")

st.title("📜 Optimizer History Dashboard 3.3")
st.caption("View and analyze optimization logs stored in Supabase via AlphaInsights Backend")

# ---------------------------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    endpoint = st.selectbox(
        "Endpoint",
        options=["(any)", "sharpe", "cvar"],
        index=0,
        help="Filter logs by optimizer endpoint.",
    )
    endpoint_param: Optional[str] = None if endpoint == "(any)" else endpoint

    col_from, col_to = st.columns(2)
    with col_from:
        start_date = st.date_input("Start Date", value=None, help="Filter created_at >= this date")
    with col_to:
        end_date = st.date_input("End Date", value=None, help="Filter created_at <= this date")

    limit = st.slider("Rows per page", 10, 200, 50, step=10)
    offset = st.number_input("Offset (pagination)", min_value=0, value=0, step=limit)

# ---------------------------------------------------------------------------
# Build query parameters for backend
# ---------------------------------------------------------------------------
def _build_params() -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
    }
    if endpoint_param:
        params["endpoint"] = endpoint_param
    if start_date:
        params["start"] = start_date.isoformat()
    if end_date:
        params["end"] = end_date.isoformat()
    return params

# ---------------------------------------------------------------------------
# Load logs from backend via unified fetch
# ---------------------------------------------------------------------------
def load_logs() -> Dict[str, Any]:
    """
    Calls backend /logs/query through unified fetch_backend().
    Returns a normalized dictionary.
    """
    params = _build_params()
    data = fetch_backend("logs/query", params=params)

    # Defensive normalization
    if not isinstance(data, dict):
        return {"count": 0, "total": 0, "last_updated": None, "results": []}

    data.setdefault("results", [])
    data.setdefault("count", len(data["results"]))
    data.setdefault("total", data.get("count", 0))
    return data

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.subheader("Results")

try:
    payload = load_logs()
    results: List[Dict[str, Any]] = payload.get("results", [])
    count = int(payload.get("count", len(results)))
    total = int(payload.get("total", count))
    last_updated = payload.get("last_updated")

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Rows Fetched", count)
    with kpi2:
        st.metric("Total Matching Logs", total)
    with kpi3:
        st.metric("Last Updated", last_updated or "—")

    if not results:
        st.info("No logs found for selected filters.")
    else:
        df = pd.json_normalize(results)

        preferred_cols = [
            "created_at",
            "endpoint",
            "tickers",
            "start",
            "end",
            "result.sharpe",
            "result.es",
            "result.var",
            "result.ann_vol",
            "result.success",
        ]
        cols = [c for c in preferred_cols if c in df.columns] + [
            c for c in df.columns if c not in preferred_cols
        ]
        df = df[cols]
        st.dataframe(df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"⚠️ Failed to load logs: {e}")

# ---------------------------------------------------------------------------
# Insights summary
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🧩 Optimizer Insights Summary")

try:
    summary = fetch_backend("logs/insights")
    if isinstance(summary, dict):
        st.json(summary)
    else:
        st.write(summary)
except Exception as e:
    st.error(f"Failed to load optimizer insights: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 AlphaInsights — Quantitative Intelligence Platform")
