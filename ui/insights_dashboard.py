"""
ui/insights_dashboard.py
------------------------

Streamlit dashboard for optimizer insights.
Pulls aggregated statistics and recent logs
from the FastAPI backend (/logs/insights endpoint).
"""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Optimizer Insights — AlphaInsights",
    layout="wide",
    page_icon="📊",
)

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

st.title("📊 Optimizer Insights — AlphaInsights")
st.caption("Live summary analytics from your optimizer logs (via Supabase)")

# -------------------------------------------------------------------
# Fetch data
# -------------------------------------------------------------------

@st.cache_data(ttl=120)
def load_insights():
    try:
        resp = requests.get(f"{API_URL}/logs/insights?limit=500")
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to load insights from backend: {e}")
        return None


data = load_insights()
if not data:
    st.stop()

src = data.get("source", {})
insights = data.get("insights", {})

# -------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------

c1, c2, c3 = st.columns(3)
c1.metric("Total Runs", src.get("count", 0))
c2.metric("Endpoints", ", ".join(insights.get("runs_by_endpoint", {}).keys()))
c3.metric("Sharpe Records", insights.get("sharpe_stats", {}).get("count", 0))

st.divider()

# -------------------------------------------------------------------
# Detailed statistics
# -------------------------------------------------------------------

st.subheader("Sharpe Ratio Statistics")
sharpe = insights.get("sharpe_stats", {})
if sharpe:
    st.write(pd.DataFrame([sharpe]))
else:
    st.info("No Sharpe data found in optimizer logs.")

st.subheader("CVaR Statistics")
cvar = insights.get("cvar_stats", {})
if cvar:
    st.write(pd.DataFrame([cvar]))
else:
    st.info("No CVaR data found in optimizer logs.")

st.divider()

# -------------------------------------------------------------------
# Recent runs preview
# -------------------------------------------------------------------

st.subheader("Recent Optimizer Runs")

try:
    resp = requests.get(f"{API_URL}/logs/recent?limit=20")
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("results") if isinstance(data, dict) else data
    if rows:
        df = pd.json_normalize(rows)
        st.dataframe(df)
        # If Sharpe values exist, visualize
        if "result.sharpe" in df.columns:
            fig = px.histogram(df, x="result.sharpe", nbins=20,
                               title="Sharpe Ratio Distribution")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No recent logs available.")
except Exception as e:
    st.error(f"Failed to load recent logs: {e}")

st.caption("Updated from backend: /logs/insights + /logs/recent endpoints.")
