import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL
# ===============================================================
# Portfolio Comparison Dashboard — AlphaInsights UI
# ===============================================================
# Author: Ruïz Verbeke
# Created: 2025-10-29
# Description:
#     Streamlit dashboard to compare multiple portfolios using
#     the Sharpe Ratio analytics engine.
# ===============================================================

import streamlit as st
import pandas as pd
import sys
import os

# --- Ensure the project root is in Python's import path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from analytics.sharpe_ratio import compare_portfolios

# ===============================================================
# Page Configuration
# ===============================================================
st.set_page_config(
    page_title="Portfolio Comparison — AlphaInsights",
    page_icon="📊",
    layout="wide"
)

# ===============================================================
# Header
# ===============================================================
st.title("📈 Portfolio Comparison Dashboard")
st.write("""
Compare multiple portfolios side-by-side using Sharpe Ratios and key performance metrics.
Each portfolio can include different tickers and weights.
""")

# ===============================================================
# Sidebar — Portfolio Definitions
# ===============================================================
st.sidebar.header("⚙️ Portfolio Setup")

st.sidebar.markdown("Define up to **10 portfolios** for comparison:")

num_portfolios = st.sidebar.slider("Number of Portfolios", 1, 10, 2)

portfolios = {}
for i in range(num_portfolios):
    with st.sidebar.expander(f"Portfolio {i+1} Settings", expanded=(i < 3)):
        name = st.text_input(f"Name of Portfolio {i+1}", value=f"Portfolio {i+1}")
        tickers = st.text_input(
            f"Tickers for {name} (comma-separated):",
            value="AAPL, MSFT, NVDA" if i == 0 else "KO, PG, JNJ"
        )
        tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

        weights_text = st.text_input(
            f"Weights for {name} (comma-separated, must sum to 1):",
            value=""
        )
        weights = None
        if weights_text:
            try:
                weights = [float(w.strip()) for w in weights_text.split(",")]
            except ValueError:
                st.error(f"Weights for {name} must be numbers separated by commas.")

        portfolios[name] = {"tickers": tickers, "weights": weights}

# ===============================================================
# Sidebar — Global Settings
# ===============================================================
st.sidebar.header("📆 Analysis Period")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

risk_free_rate = st.sidebar.number_input(
    "Annual Risk-Free Rate (as decimal)",
    min_value=0.0,
    max_value=0.10,
    value=0.02,
    step=0.005
)

run_analysis = st.sidebar.button("🚀 Run Comparison")

# ===============================================================
# Main Output Area
# ===============================================================
if run_analysis:
    try:
        result = compare_portfolios(
            portfolios=portfolios,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_rate_annual=risk_free_rate,
        )

        st.success("✅ Comparison complete!")

        # --- Summary Table (key metrics) ---
        st.subheader("📋 Summary: Return, Volatility & Sharpe")
        summary_cols = ["label", "mean_daily_return", "daily_volatility", "sharpe_annualized"]
        summary = result[summary_cols].copy()
        summary.columns = ["Portfolio", "Mean Daily Return", "Daily Volatility", "Sharpe (Annualized)"]
        st.dataframe(summary.style.format({
            "Mean Daily Return": "{:.4f}",
            "Daily Volatility": "{:.4f}",
            "Sharpe (Annualized)": "{:.2f}"
        }), use_container_width=True)

        # --- Full Results Table ---
        st.subheader("📊 Detailed Portfolio Results")
        st.dataframe(result, use_container_width=True)

        # --- Sharpe Ratio Bar Chart ---
        import plotly.express as px
        st.subheader("📈 Annualized Sharpe Ratios")
        fig = px.bar(
            result,
            x="label",
            y="sharpe_annualized",
            text_auto=".2f",
            title="Annualized Sharpe Ratio by Portfolio",
            color="label"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Risk vs Return Scatterplot (Enhanced) ---
        st.subheader("📉 Risk vs Return Scatterplot (with Efficient Frontier Trendline)")

        import numpy as np

        # Sort portfolios by volatility for smoother frontier
        sorted_result = result.sort_values("daily_volatility")

        scatter_fig = px.scatter(
            result,
            x="daily_volatility",
            y="mean_daily_return",
            text="label",
            size="sharpe_annualized",
            color="label",
            title="Risk (Volatility) vs Return (Mean Daily Return)",
            hover_data={
                "label": True,
                "mean_daily_return": ":.4f",
                "daily_volatility": ":.4f",
                "sharpe_annualized": ":.2f"
            },
            labels={
                "daily_volatility": "Daily Volatility (Risk)",
                "mean_daily_return": "Mean Daily Return",
                "label": "Portfolio"
            }
        )

        # Add efficient frontier trendline
        scatter_fig.add_scatter(
            x=sorted_result["daily_volatility"],
            y=sorted_result["mean_daily_return"],
            mode="lines+markers",
            name="Efficient Frontier",
            line=dict(color="black", width=2, dash="dot"),
            marker=dict(size=6, color="black")
        )

        scatter_fig.update_traces(textposition="top center")
        scatter_fig.update_layout(
            xaxis_title="Daily Volatility (Risk)",
            yaxis_title="Mean Daily Return",
            legend_title="Portfolio",
            template="plotly_white"
        )

        st.plotly_chart(scatter_fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👈 Configure your portfolios on the left and click **Run Comparison**.")
