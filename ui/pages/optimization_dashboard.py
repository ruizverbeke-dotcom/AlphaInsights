import os
import sys
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

# ---------------------------------------------------------------------------
# Ensure imports resolve correctly when run via Streamlit
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

# ---------------------------------------------------------------------------
# CVaR Optimizer Dashboard
# ---------------------------------------------------------------------------
"""
CVaR Optimizer Dashboard
------------------------
Performs Conditional Value-at-Risk optimization.

Features:
- User inputs: tickers, date range, and confidence level α
- Calls backend /optimize/cvar endpoint (POST)
- Displays resulting weights, expected CVaR, and portfolio stats
- Uses backend health indicator for transparency
"""

# ---------------------------------------------------------------------------
# 1. Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="CVaR Optimizer", layout="wide")
st.title("⚙️ CVaR Optimizer Dashboard")
st.caption("Perform portfolio optimization based on Conditional Value-at-Risk (CVaR).")

# ---------------------------------------------------------------------------
# 2. Input Section
# ---------------------------------------------------------------------------
with st.form("cvar_inputs"):
    st.subheader("📊 Input Parameters")

    tickers = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, SPY, CAC 40, Gold",
        help="Enter one or more tickers separated by commas (e.g., AAPL, MSFT, SPY).",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2021, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.today())

    conf_level = st.select_slider(
        "Confidence Level α",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        help="The confidence level for CVaR (Conditional Value-at-Risk) computation.",
    )

    st.markdown("Optional Parameters (UI-only for now)")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Global Max Weight", value="1.0")
    with col2:
        st.text_input("Exclude Tickers", placeholder="e.g. TSLA, NVDA")

    submitted = st.form_submit_button("🚀 Run CVaR Optimization")

# ---------------------------------------------------------------------------
# 3. Backend Health Indicator
# ---------------------------------------------------------------------------
with st.expander("🩺 System Health", expanded=True):
    try:
        health = fetch_backend("health")
        if isinstance(health, dict) and health.get("status") == "ok":
            st.success("✅ Backend reachable")
            st.json(health)
        else:
            st.warning("⚠️ Backend reachable but returned an unexpected response:")
            st.write(health)
    except Exception as e:
        st.error(f"❌ Backend unreachable: {e}")

# ---------------------------------------------------------------------------
# 4. Optimization Logic
# ---------------------------------------------------------------------------
if submitted:
    st.subheader("🚀 Optimization Results")

    payload = {
        "tickers": [t.strip() for t in tickers.split(",") if t.strip()],
        "start": str(start_date),
        "end": str(end_date),
        "confidence": conf_level,
    }

    if not payload["tickers"]:
        st.error("❌ Please enter at least one valid ticker.")
        st.stop()

    # -----------------------------------------------------------------------
    # Try POST via requests (since fetch_backend only supports GET)
    # -----------------------------------------------------------------------
    try:
        url = BACKEND_URL.rstrip("/") + "/optimize/cvar"
        st.info(f"Sending request to: `{url}`")

        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            st.success("✅ Optimization complete")
            st.json(result)

            # ---------------------------
            # Display portfolio weights
            # ---------------------------
            if "weights" in result and isinstance(result["weights"], dict):
                st.subheader("💼 Portfolio Weights")
                df = pd.DataFrame.from_dict(result["weights"], orient="index", columns=["Weight"])
                df.index.name = "Asset"
                df["Weight (%)"] = df["Weight"] * 100
                st.dataframe(df.style.format({"Weight": "{:.4f}", "Weight (%)": "{:.2f}"}), use_container_width=True)

            # ---------------------------
            # Display optimization summary
            # ---------------------------
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence Level α", f"{conf_level:.2f}")
            col2.metric("Expected CVaR", f"{result.get('cvar', '—')}")
            col3.metric("Expected Return", f"{result.get('expected_return', '—')}")

            if "summary" in result:
                st.info(result["summary"])

        else:
            st.error(f"❌ Optimization failed (status {response.status_code})")
            try:
                st.write(response.json())
            except Exception:
                st.text(response.text)

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network/connection error: {e}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")

# ---------------------------------------------------------------------------
# 5. Footer
# ---------------------------------------------------------------------------
st.caption("AlphaInsights — powered by FastAPI backend and Streamlit frontend.")
