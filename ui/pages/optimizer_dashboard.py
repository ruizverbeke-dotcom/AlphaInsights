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
from ui.components.data_resolver import resolve_tickers  # ← intelligent name resolver

# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _load_price_data(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted close data using yfinance.

    The function assumes that tickers are already validated/resolved.
    """
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    data = data.squeeze("columns") if isinstance(data, pd.DataFrame) else data
    df = pd.DataFrame(data).dropna(how="any")
    if df.empty:
        raise ValueError("No valid price data found for provided tickers.")
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
        f"Use Active Profile (ID {st.session_state['active_profile_id']})", value=False
    )

st.sidebar.markdown("**Optional Constraints**")
max_weight = st.sidebar.number_input("Global Max Weight", 0.0, 1.0, 1.0, 0.05)
excl_text = st.sidebar.text_input("Exclude Tickers", placeholder="e.g. TSLA, NVDA")
run_button = st.sidebar.button("🚀 Run Optimization", type="primary")

# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
if run_button:
    # --- Step 1: Parse and intelligently resolve tickers ---
    raw_inputs = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if not raw_inputs:
        st.error("Please enter at least one ticker.")
        st.stop()

    with st.spinner("Resolving tickers..."):
        try:
            tickers = resolve_tickers(raw_inputs)
        except Exception as e:
            st.error(f"Ticker resolution failed: {e}")
            st.stop()

    if not tickers:
        st.error("No valid tickers could be resolved from your input.")
        st.stop()

    st.success(f"Resolved tickers: {', '.join(tickers)}")

    # --- Step 2: Fetch data ---
    with st.spinner("Fetching price data..."):
        try:
            prices = _load_price_data(tickers, str(start_date), str(end_date))
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    # --- Step 3: Compute returns ---
    returns = _compute_returns(prices)

    # --- Step 4: Prepare constraints ---
    constraints = {
        "max_weight": float(max_weight),
        "excludes": [t.strip().upper() for t in excl_text.split(",") if t.strip()],
        "as_dict": False,
    }

    # --- Step 5: Run optimization ---
    with st.spinner("Running CVaR Optimization..."):
        result = optimize_cvar(returns, alpha=alpha, constraints=constraints)

    # --- Step 6: Display Results ---
    st.subheader("Optimization Summary")
    st.write(result["summary"])

    weights = (
        pd.DataFrame(result["weights"], index=["Weight"]).T
        if isinstance(result["weights"], dict)
        else result["weights"].to_frame("Weight")
    )

    st.dataframe(weights.style.format("{:.4f}"))

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Shortfall (ES)", f"{result['es']:.4f}")
    col2.metric("Value-at-Risk (VaR)", f"{result['var']:.4f}")
    col3.metric("Annualized Volatility", f"{result['ann_vol']:.4f}")
    col4.metric("Solver", result["solver"])

    # --- Plot Weights ---
    weights_reset = weights.reset_index().rename(columns={"index": "Asset"})
    weights_reset["Weight"] = weights_reset["Weight"].astype(float).clip(lower=0)
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
