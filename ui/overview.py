"""
AlphaInsights — Streamlit Launcher
----------------------------------
Main entrypoint for the AlphaInsights multipage app.
This file ensures Streamlit loads all dashboards under ui/pages/.
"""

import streamlit as st

st.set_page_config(page_title="AlphaInsights", layout="wide")

st.title("🚀 Welcome to AlphaInsights")
st.caption("Unified Quant Intelligence Platform")

st.markdown("""
Use the sidebar to navigate between dashboards:
- **Overview Dashboard**
- **Optimizer Dashboards** (Sharpe, CVaR, History)
- **Analytics Dashboards** (Insights, Comparison)
- **Valuation & Risk Dashboards**
""")

st.markdown("---")
st.info("Start exploring via the left sidebar navigation.")
st.caption("© 2025 AlphaInsights — Quantitative Intelligence Platform")
