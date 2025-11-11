import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
# Ensure the app root is in the import path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

"""
AlphaInsights — Overview Dashboard
----------------------------------
Mission control center for system and analytics health.

- Displays backend and Supabase status
- Confirms backend connectivity via /health and /status/summary
- Provides navigation to all dashboards (Streamlit-native navigation)
"""

import streamlit as st
from core.ui_helpers import fetch_backend


# ----------------------------------------------------------------------------
# 1. Page Setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AlphaInsights Overview", layout="wide")
st.title("📈 AlphaInsights Overview")
st.caption("System mission control and entry point for analytics, optimizers, and insights.")


# ----------------------------------------------------------------------------
# 2. Backend Connectivity
# ----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔌 Backend Connection")
    try:
        health = fetch_backend("/health")
        st.success("Backend reachable ✅")
        st.json(health)
    except Exception as e:
        st.error(f"Backend not reachable ❌: {e}")

with col2:
    st.subheader("🧩 System Summary")
    try:
        summary = fetch_backend("/status/summary")
        st.info("Status summary from backend:")
        st.json(summary)
    except Exception as e:
        st.error(f"Failed to fetch summary: {e}")


# ----------------------------------------------------------------------------
# 3. Navigation Guide (Streamlit-native)
# ----------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚀 Explore Dashboards")

pages = {
    "Sharpe Optimizer": "optimizer_dashboard",
    "CVaR Optimizer": "optimization_dashboard",
    "Optimizer History": "optimizer_history",
    "Optimizer Insights": "insights_dashboard",
    "Comparison & Stress Testing": "comparison_dashboard",
    "Valuation Dashboard": "valuation_dashboard",
    "Stress Test Dashboard": "stress_test_dashboard",
}

for name, page in pages.items():
    if st.button(f"➡️ {name}"):
        # ✅ Fixed: assign query param instead of calling
        st.query_params["page"] = page
        st.rerun()


# ----------------------------------------------------------------------------
# 4. Handle Query-based Navigation
# ----------------------------------------------------------------------------
query_params = st.query_params
if "page" in query_params:
    target = query_params["page"]
    if isinstance(target, list):
        target = target[0]
    try:
        st.switch_page(f"pages/{target}.py")
    except Exception:
        st.warning(f"⚠️ Could not navigate to page: {target}")


# ----------------------------------------------------------------------------
# 5. Footer
# ----------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 AlphaInsights — Quantitative Intelligence Platform")
