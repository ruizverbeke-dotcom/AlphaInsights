"""
CVaR Optimizer Dashboard
------------------------
Performs Conditional Value-at-Risk optimization.

This page:
- Lets the user input tickers, date range, and confidence level α
- Calls the backend /optimize/cvar endpoint
- Displays resulting weights, expected CVaR, and portfolio stats
- Uses unified backend fetch layer (core.ui_helpers.fetch_backend)
"""

import os
import sys
from datetime import datetime
import streamlit as st
import pandas as pd

# ---------------------------------------------------------------------------
# 0. Ensure imports resolve correctly when run from `ui/overview.py`
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

# ---------------------------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CVaR Optimizer", layout="wide")
st.title("⚙️ CVaR Optimizer Dashboard")
st.caption("Perform portfolio optimization based on Conditional Value-at-Risk (CVaR).")

# ---------------------------------------------------------------------------
# 2. Input section
# ---------------------------------------------------------------------------
with st.form("cvar_inputs"):
    st.subheader("📊 Input Parameters")

    tickers = st.text_input("Tickers (comma-separated)", value="AAPL, MSFT, SPY, CAC 40, Gold")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    conf_level = st.select_slider("Confidence Level α", options=[0.90, 0.95, 0.99], value=0.95)

    st.markdown("Optional (UI-only for now)")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Global Max Weight (not yet enforced by backend)", value="1.0")
    with col2:
        st.text_input("Exclude Tickers (not yet enforced by backend)", placeholder="e.g. TSLA, NVDA")

    submitted = st.form_submit_button("Run CVaR Optimization")

# ---------------------------------------------------------------------------
# 3. Backend health indicator
# ---------------------------------------------------------------------------
with st.expander("🩺 System Health (auto-refreshes every 60 s)", expanded=True):
    try:
        health = fetch_backend("health")
        if isinstance(health, dict):
            if health.get("status") == "ok":
                st.success("✅ Backend reachable")
                st.json(health)
            else:
                st.warning(f"⚠️ Backend responded with issues:")
                st.json(health)
        else:
            st.error(f"❌ Unexpected response type: {type(health)}")
    except Exception as e:
        st.error(f"❌ Backend unreachable: {e}")

# ---------------------------------------------------------------------------
# 4. Run optimization
# ---------------------------------------------------------------------------
if submitted:
    st.subheader("🚀 Optimization Results")

    payload = {
        "tickers": [t.strip() for t in tickers.split(",") if t.strip()],
        "start": str(start_date),
        "end": str(end_date),
        "confidence": conf_level,
    }

    try:
        result = fetch_backend("optimize/cvar", params=payload)
        if n
