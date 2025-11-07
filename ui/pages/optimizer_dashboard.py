"""
AlphaInsights Optimizer Dashboard
---------------------------------
Streamlit interface for the CVaR (Expected Shortfall) optimizer.

Now operates as a thin client:

UI (this page) → Backend API (FastAPI) → Analytics Engine (optimize_cvar).

Responsibilities:
- Parse and resolve user inputs (tickers, names like 'Microsoft', 'CAC 40', 'Gold').
- Call the backend /optimize/cvar endpoint with canonical tickers & parameters.
- Display returned weights and risk metrics (ES, VaR, annualized vol).
- Provide real-time backend health diagnostics in the sidebar.

All heavy lifting (data fetch, returns, optimization) happens in the backend.
"""

import sys
import os
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from streamlit_autorefresh import st_autorefresh

# --- Auto refresh must be at top level ---
st_autorefresh(interval=60 * 1000, key="page_autorefresh")

# --- import project root ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ui.components.data_resolver import resolve_tickers  # intelligent name resolver

# Backend base URL (can be overridden via environment for deployment)
BACKEND_URL = os.getenv("ALPHAINSIGHTS_API_URL", "http://127.0.0.1:8000")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _call_cvar_api(
    tickers: List[str],
    start_date: str,
    end_date: str,
    alpha: float,
) -> dict:
    """Call the AlphaInsights Backend CVaR endpoint."""
    payload = {
        "tickers": tickers,
        "start": start_date,
        "end": end_date,
        "alpha": float(alpha),
    }
    try:
        resp = requests.post(f"{BACKEND_URL}/optimize/cvar", json=payload, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to reach backend API at {BACKEND_URL}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Backend error {resp.status_code}: {resp.text}")

    data = resp.json()
    required_keys = {"weights", "es", "var", "ann_vol", "solver", "success", "summary"}
    if not required_keys.issubset(data.keys()):
        raise RuntimeError(f"Backend response missing keys. Got: {sorted(data.keys())}")
    return data


def _ping_backend_health() -> dict:
    """Query the backend /health endpoint to verify system status."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        raise RuntimeError(f"Unexpected status {resp.status_code}")
    except Exception as e:
        raise RuntimeError(f"Backend unreachable: {e}") from e


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="CVaR Optimizer", layout="wide")
st.title("📊 CVaR (Expected Shortfall) Optimizer")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    placeholder="AAPL, MSFT, SPY, CAC 40, Gold",
)

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date")
with col2:
    end_date = st.date_input("End Date")

alpha = st.sidebar.slider("Confidence Level α", 0.90, 0.995, 0.95, 0.005)

use_profile = False
if "active_profile_id" in st.session_state:
    use_profile = st.sidebar.checkbox(
        f"Use Active Profile (ID {st.session_state['active_profile_id']})",
        value=False,
        help="(Planned) Use constraints from the active profile.",
    )

st.sidebar.markdown("**Optional (UI-only for now)**")
max_weight = st.sidebar.number_input(
    "Global Max Weight (not yet enforced by backend)",
    0.0,
    1.0,
    1.0,
    0.05,
    help="Displayed for future integration; currently not wired into API.",
)
excl_text = st.sidebar.text_input(
    "Exclude Tickers (not yet enforced by backend)",
    placeholder="e.g. TSLA, NVDA",
    help="Planned: pass to backend constraints.",
)

run_button = st.sidebar.button("🚀 Run Optimization", type="primary")


# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
if run_button:
    # --- Step 1: Validate basic inputs ---
    if not tickers_input.strip():
        st.error("Please enter at least one ticker or name.")
        st.stop()
    if start_date >= end_date:
        st.error("Start Date must be strictly before End Date.")
        st.stop()

    # --- Step 2: Intelligent ticker resolution ---
    with st.spinner("Resolving tickers..."):
        try:
            resolved, dropped = resolve_tickers(
                tickers_input,
                start=str(start_date),
                end=str(end_date),
            )
        except Exception as e:
            st.error(f"Ticker resolution failed: {e}")
            st.stop()

    if not resolved:
        st.error("No valid tickers could be resolved. Please adjust your input.")
        if dropped:
            st.info(f"Tried and dropped: {', '.join(dropped)}")
        st.stop()

    st.success(f"Resolved tickers: {', '.join(resolved)}")
    if dropped:
        st.warning(f"Dropped tickers with no usable data: {', '.join(dropped)}")

    # --- Step 3: Call backend CVaR optimizer ---
    with st.spinner("Calling backend CVaR optimizer..."):
        try:
            result = _call_cvar_api(
                tickers=resolved,
                start_date=str(start_date),
                end_date=str(end_date),
                alpha=float(alpha),
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

    # --- Step 4: Display results ---
    st.subheader("Optimization Summary")
    st.write(result["summary"])

    weights_dict = result.get("weights", {})
    weights = pd.DataFrame.from_dict(weights_dict, orient="index", columns=["Weight"])
    weights["Weight"] = weights["Weight"].astype(float)
    st.dataframe(weights.style.format("{:.4f}"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Shortfall (ES)", f"{result['es']:.4f}")
    c2.metric("Value-at-Risk (VaR)", f"{result['var']:.4f}")
    c3.metric("Annualized Volatility", f"{result['ann_vol']:.4f}")
    c4.metric("Solver", result["solver"] or "N/A")

    # --- Plot Weights ---
    weights_plot = weights.reset_index().rename(columns={"index": "Asset"})
    weights_plot["Weight"] = weights_plot["Weight"].clip(lower=0.0)
    fig = px.bar(
        weights_plot,
        x="Asset",
        y="Weight",
        title=f"Optimized Portfolio Weights via Backend (α={alpha:.3f})",
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "Enter tickers, select a date range and α, then click **Run Optimization**.\n\n"
        "This dashboard now calls the AlphaInsights Backend API, "
        "making it deployment-ready and agent-friendly."
    )


# --------------------------------------------------------------------------- #
# Backend Health Monitor (Phase 5.6 Final — True Auto-Refresh)
# --------------------------------------------------------------------------- #
st.sidebar.markdown("---")
st.sidebar.subheader("🩺 System Health (auto-refreshes every 60 s)")

if "last_health_data" not in st.session_state:
    st.session_state.last_health_data = None

def _refresh_health():
    try:
        st.session_state.last_health_data = _ping_backend_health()
    except Exception as e:
        st.session_state.last_health_data = {"status": "error", "error": str(e)}

_refresh_health()

data = st.session_state.last_health_data
if data:
    status = data.get("status", "N/A").upper()
    phase = data.get("phase", "N/A")
    timestamp = data.get("timestamp", "N/A")
    st.sidebar.success(f"✅ Backend {status}")
    st.sidebar.write(f"**Phase:** {phase}")
    st.sidebar.write(f"**Time:** {timestamp}")
else:
    st.sidebar.warning("No health data yet. Click Refresh or wait 60 s.")

# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
