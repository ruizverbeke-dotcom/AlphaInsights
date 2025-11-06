# ui/pages/optimizer_dashboard.py
"""
AlphaInsights Optimizer Dashboard
---------------------------------
Streamlit interface for the CVaR (Expected Shortfall) optimizer.

This page lets users:
- Enter tickers and a date range.
- Choose tail confidence level α (0.90–0.995).
- Toggle "Use Active Profile" if one exists in session state.
- Optionally specify max weight caps and exclusions.
- Run optimization and view results (weights, ES, VaR, Vol).

Agent-ready design: produces deterministic metrics for the Optimizer Agent.
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px

# --- import project root ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from analytics.optimization import optimize_cvar


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close data using yfinance."""
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    data = data.squeeze("columns") if isinstance(data, pd.DataFrame) else data
    df = pd.DataFrame(data)
    df = df.dropna(how="any")
    return df


def _compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from price data."""
    prices = prices.squeeze("columns") if isinstance(prices, pd.DataFrame) else prices
    returns = np.log(prices / prices.shift(1)).dropna()
    returns = returns.squeeze("columns") if isinstance(returns, pd.DataFrame) else returns
    return pd.DataFrame(returns)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="CVaR Optimizer", layout="wide")
st.title("📊 CVaR (Expected Shortfall) Optimizer")

# --- Sidebar Inputs ---
st.sidebar.header("Input Parameters")

tickers_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    placeholder="AAPL, MSFT, SPY",
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
        f"Use Active Profile (ID {st.session_state['active_profile_id']})", value=False
    )

st.sidebar.markdown("**Optional Constraints**")
max_weight = st.sidebar.number_input("Global Max Weight", 0.0, 1.0, 1.0, 0.05)
excl_text = st.sidebar.text_input("Exclude Tickers", placeholder="e.g. TSLA, NVDA")
run_button = st.sidebar.button("🚀 Run Optimization", type="primary")

# --- Main Execution ---
if run_button:
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
        st.stop()

    with st.spinner("Fetching price data..."):
        try:
            prices = _load_price_data(tickers, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    returns = _compute_returns(prices)

    constraints = {
        "max_weight": float(max_weight),
        "excludes": [t.strip().upper() for t in excl_text.split(",") if t.strip()],
        "as_dict": False,
    }

    with st.spinner("Running CVaR Optimization..."):
        result = optimize_cvar(returns, alpha=alpha, constraints=constraints)

    st.subheader("Optimization Summary")
    st.write(result["summary"])

    weights = (
        pd.DataFrame(result["weights"], index=["Weight"]).T
        if isinstance(result["weights"], dict)
        else result["weights"].to_frame("Weight")
    )

    st.dataframe(weights.style.format("{:.4f}"))

    st.metric("Expected Shortfall (ES)", f"{result['es']:.4f}")
    st.metric("Value-at-Risk (VaR)", f"{result['var']:.4f}")
    st.metric("Annualized Volatility", f"{result['ann_vol']:.4f}")
    st.metric("Solver", result["solver"])

    # --- Plot Weights ---
    weights_reset = weights.reset_index().rename(columns={"index": "Asset"})
    weights_reset["Weight"] = weights_reset["Weight"].astype(float)

    weights_reset["Weight"] = weights_reset["Weight"].apply(
        lambda x: float(x) if not np.isnan(x) else 0.0
    )

    weights_reset["Weight"] = weights_reset["Weight"].clip(lower=0)
    fig = px.bar(
        weights_reset,
        x="Asset",
        y="Weight",
        title=f"Optimized Portfolio Weights (α={alpha:.3f})",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Enter tickers, date range, and parameters, then click **Run Optimization**.")

# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
