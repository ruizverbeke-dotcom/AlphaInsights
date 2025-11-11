"""
Optimizer History Dashboard

View and analyze optimization logs stored in Supabase via AlphaInsights Backend.
Backed by:
- /logs/query  (filtered, paginated)
- /logs/recent (simple tail view, optional)

This page uses the unified backend fetch layer:
- core.ui_helpers.fetch_backend
- core.ui_config.BACKEND_URL
"""

import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when running via `streamlit run ui/overview.py`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

st.set_page_config(page_title="Optimizer History", layout="wide")

st.title("📜 Optimizer History Dashboard 3.1")
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
# Helper: build params
# ---------------------------------------------------------------------------
def _build_params() -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "limit": limit,
        "offset": offset,
    }
    if endpoint_param:
        params["endpoint"] = endpoint_param
    if start_date:
        # Streamlit may give None; only include if set
        params["start"] = start_date.isoformat()
    if end_date:
        params["end"] = end_date.isoformat()
    return params


# ---------------------------------------------------------------------------
# Fetch logs via backend
# ---------------------------------------------------------------------------
def load_logs() -> Dict[str, Any]:
    """
    Call backend /logs/query through unified fetch_backend.

    Returns a dict:
      {
        "count": int,
        "total": int,
        "last_updated": str | None,
        "results": [ ... ],
      }
    """
    params = _build_params()
    # fetch_backend builds the full URL from BACKEND_URL + path
    data = fetch_backend("logs/query", params=params)

    # Be defensive about shape
    if not isinstance(data, dict):
        return {
            "count": 0,
            "total": 0,
            "last_updated": None,
            "results": [],
        }
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

    # Top-level KPIs
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
        # Normalize into DataFrame for display
        df = pd.json_normalize(results)

        # Nice column ordering if present
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

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

except Exception as e:
    st.error(f"⚠️ Failed to load logs: {e}")
