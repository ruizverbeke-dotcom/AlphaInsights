# ===============================================================
# Sharpe Ratio Dashboard — AlphaInsights UI (Error-Free)
# ===============================================================
# Author: Ruïz Verbeke
# Created: 2025-10-29
# Description:
#     Final stable Streamlit dashboard to analyze Sharpe Ratios.
#     Fixed all 1D/2D data shape errors and mixed Plotly+Matplotlib.
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import sys
import os
import yfinance as yf

# --- Ensure the project root is in Python's import path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from analytics.sharpe_ratio import calculate_sharpe_ratio

# ===============================================================
# Page Configuration
# ===============================================================
st.set_page_config(
    page_title="Sharpe Ratio Analysis — AlphaInsights",
    page_icon="📈",
    layout="wide"
)

# ===============================================================
# Header
# ===============================================================
st.title("📊 Sharpe Ratio Analysis Dashboard")
st.write("""
Analyze performance metrics for single assets or custom portfolios.
Select tickers, weights, and a date range to calculate Sharpe Ratios,
returns, and visualize portfolio performance versus a benchmark.
""")

# ===============================================================
# Sidebar
# ===============================================================
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    tickers = st.text_input("Enter tickers (comma-separated):", value="AAPL, MSFT, NVDA")
    tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    risk_free_rate = st.number_input(
        "Annual Risk-Free Rate (as decimal)",
        min_value=0.0, max_value=0.10, value=0.02, step=0.005
    )

    st.subheader("Benchmark Settings")
    benchmark_ticker = st.text_input("Enter benchmark ticker (e.g., SPY for S&P 500):", value="SPY")

    st.subheader("Portfolio Weights (optional)")
    weights_text = st.text_input(f"Enter {len(tickers)} weights separated by commas (optional):", value="")
    weights = None
    if weights_text:
        try:
            weights = [float(w.strip()) for w in weights_text.split(",")]
        except ValueError:
            st.error("Weights must be numbers separated by commas.")

    run_analysis = st.button("🚀 Run Analysis")

# ===============================================================
# Main Output
# ===============================================================
if run_analysis:
    try:
        # -----------------------------------------------------------
        # Calculate Sharpe Ratio and metrics
        # -----------------------------------------------------------
        result = calculate_sharpe_ratio(
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_rate_annual=risk_free_rate,
            weights=weights,
        )

        st.success("✅ Analysis Complete!")

        st.subheader("📋 Sharpe Ratio Results")
        st.dataframe(result, use_container_width=True)

        sharpe = result["sharpe_annualized"].iloc[0]
        mean_return = result["mean_daily_return"].iloc[0] * 252
        volatility = result["daily_volatility"].iloc[0] * np.sqrt(252)

        # -----------------------------------------------------------
        # Summary Cards
        # -----------------------------------------------------------
        st.markdown("### 🧾 Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Annualized Return", f"{mean_return:.2%}")
        col2.metric("Annualized Volatility", f"{volatility:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        # -----------------------------------------------------------
        # Download portfolio data
        # -----------------------------------------------------------
        data = yf.download(
            tickers=tickers,
            start=str(start_date),
            end=str(end_date),
            progress=False,
            group_by="ticker",
            auto_adjust=True
        )

        if len(tickers) == 1:
            prices = pd.DataFrame(data["Close"]) if "Close" in data else data
            prices.columns = [tickers[0]]
        else:
            try:
                prices = pd.concat({t: data[t]["Close"] for t in tickers}, axis=1)
            except Exception:
                prices = pd.concat({t: data[t] for t in tickers}, axis=1)

        prices.dropna(how="all", inplace=True)
        daily_returns = prices.pct_change().dropna()

        if weights is not None:
            weights = np.array(weights)
            portfolio_returns = (daily_returns * weights).sum(axis=1)
        else:
            portfolio_returns = daily_returns.mean(axis=1)

        # --- Ensure clean 1D Series ---
        if isinstance(portfolio_returns, pd.DataFrame):
            portfolio_returns = portfolio_returns.squeeze("columns")
        if portfolio_returns.ndim > 1:
            portfolio_returns = pd.Series(np.ravel(portfolio_returns), index=portfolio_returns.index)

        cumulative = (1 + portfolio_returns).cumprod()

        # -----------------------------------------------------------
        # Benchmark
        # -----------------------------------------------------------
        benchmark_data = yf.download(
            tickers=benchmark_ticker,
            start=str(start_date),
            end=str(end_date),
            progress=False,
            auto_adjust=True
        )
        benchmark_returns = benchmark_data["Close"].pct_change().dropna()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()

        # --- Flatten benchmark if 2D ---
        if isinstance(benchmark_cumulative, pd.DataFrame):
            benchmark_cumulative = benchmark_cumulative.squeeze("columns")
        if benchmark_cumulative.ndim > 1:
            benchmark_cumulative = pd.Series(np.ravel(benchmark_cumulative), index=benchmark_cumulative.index)

        combined = pd.DataFrame({
            "Portfolio": cumulative,
            benchmark_ticker.upper(): benchmark_cumulative
        })

        # --- Ensure all y-values are flat ---
        for col in combined.columns:
            if combined[col].ndim > 1:
                combined[col] = np.ravel(combined[col])

        # -----------------------------------------------------------
        # Plotly: Cumulative Chart
        # -----------------------------------------------------------
        st.subheader("📈 Cumulative Portfolio Growth vs Benchmark")
        fig_cum = px.line(
            combined.reset_index(),
            x="Date" if "Date" in combined.columns else combined.index,
            y=combined.columns,
            title=f"Cumulative Growth: Portfolio vs {benchmark_ticker.upper()}",
            labels={"value": "Growth (1 = Start)", "variable": "Series"},
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # -----------------------------------------------------------
        # Plotly: Return Histogram
        # -----------------------------------------------------------
        st.subheader("📉 Daily Return Distribution")
        fig_hist = px.histogram(
            x=np.ravel(portfolio_returns.values),
            nbins=50,
            title="Distribution of Daily Returns",
            labels={"x": "Daily Return"},
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # -----------------------------------------------------------
        # Plotly: Rolling Sharpe Ratio (Interactive, shape-safe)
        # -----------------------------------------------------------
        st.subheader("📊 Rolling Sharpe Ratio (60-day window)")

        window = 60
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.dropna()

        # --- Flatten to 1D safely ---
        rolling_sharpe = pd.Series(np.ravel(rolling_sharpe), index=rolling_sharpe.index[:len(np.ravel(rolling_sharpe))])

        # --- Build clean DataFrame for Plotly ---
        rolling_df = rolling_sharpe.reset_index()
        rolling_df.columns = ["Date", "Sharpe"]

        # --- Interactive Plotly chart ---
        fig_roll = px.line(
            rolling_df,
            x="Date",
            y="Sharpe",
            title=f"Rolling Sharpe Ratio ({window}-day window)",
            labels={"Sharpe": "Sharpe Ratio"},
        )
        fig_roll.update_traces(line=dict(color="orange", width=2))
        fig_roll.update_layout(
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
        )
        st.plotly_chart(fig_roll, use_container_width=True)


    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👈 Configure your settings and click **Run Analysis** to begin.")




