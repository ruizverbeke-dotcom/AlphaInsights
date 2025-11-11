"""
AlphaInsights — Overview Dashboard
----------------------------------

This page acts as the mission control center:
- Displays backend health and Supabase status.
- Links to all major dashboards.
- Confirms backend connectivity via /health and /status/summary.
"""

import streamlit as st
import requests
import os

st.set_page_config(page_title="AlphaInsights Overview", layout="wide")

# ---------------------------------------------------------------------------
# 1. Header
# ---------------------------------------------------------------------------
st.title("📈 AlphaInsights Overview")
st.caption("System mission control and entry point for analytics, optimizers, and insights.")

# ---------------------------------------------------------------------------
# 2. Backend connectivity
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔌 Backend Connection")
    try:
        health = fetch_backend(f"{BACKEND_URL}/health", timeout=5).json()
        st.success("Backend reachable ✅")
        st.json(health)
    except Exception as e:
        st.error(f"Backend not reachable ❌: {e}")

with col2:
    st.subheader("🧩 System Summary")
    try:
        summary = fetch_backend(f"{BACKEND_URL}/status/summary", timeout=5).json()
        st.info("Status summary from backend:")
        st.json(summary)
    except Exception as e:
        st.error(f"Failed to fetch summary: {e}")

# ---------------------------------------------------------------------------
# 3. Navigation guide
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚀 Explore Dashboards")

st.markdown(
    """
**Optimization Dashboards**
- [Sharpe Optimizer](optimizer_dashboard)
- [CVaR Optimizer](optimization_dashboard)
- [Optimizer History](optimizer_history)

**Analytics & Insights**
- [Optimizer Insights](insights_dashboard)
- [Comparison & Stress Testing](comparison_dashboard)

**Valuation & Risk**
- [Valuation Dashboard](valuation_dashboard)
- [Stress Test Dashboard](stress_test_dashboard)

Use the left sidebar for quick navigation between pages.
"""
)

# ---------------------------------------------------------------------------
# 4. Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 AlphaInsights — Quantitative Intelligence Platform")
