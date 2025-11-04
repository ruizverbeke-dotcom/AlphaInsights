# ===============================================================
# Sharpe Ratio Dashboard — AlphaInsights UI (Error-Free + Validation v2.1)
# ===============================================================
# Author: Ruïz Verbeke
# Created: 2025-10-29
# Updated: 2025-10-31
# Description:
#     Final stable Streamlit dashboard to analyze Sharpe Ratios.
#     Fixes 1D/2D data shape issues, adds interactive Plotly charts.
#     Adds:
#         • Active Profile Banner with persistence
#         • Universal ticker resolution (Yahoo finance search)
#         • Robust ticker validation + user-friendly warnings
#         • Weight sanity checks + benchmark fallback
#         • FIX: cumulative chart uses x="Date" (no more 'x=index' errors)
# ===============================================================

from __future__ import annotations

import os
import sys
import math
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt  # (kept if you want a Matplotlib fallback later)
import yfinance as yf

# --- Ensure the project root is in Python's import path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from analytics.sharpe_ratio import calculate_sharpe_ratio
from database.queries import get_profile, get_profiles

# ===============================================================
# Utility Helpers
# ===============================================================

# Common name → index/ticker shortcuts
_INDEX_MAP = {
    "CAC 40": "^FCHI",
    "CAC40": "^FCHI",
    "SP500": "^GSPC",
    "S&P 500": "^GSPC",
    "S&P500": "^GSPC",
    "NASDAQ": "^NDX",
    "NASDAQ 100": "^NDX",
    "NDX": "^NDX",
    "DAX": "^GDAXI",
    "BEL20": "^BFX",
    "BEL 20": "^BFX",
    "STOXX600": "^STOXX",
    "EURO STOXX 50": "^STOXX50E",
    "FTSE 100": "^FTSE",
}


def normalize_ticker(name: str) -> str:
    """
    First-pass normalization for well-known index names (case-insensitive).
    If not matched, return the uppercased raw string.
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    key = raw.upper()
    return _INDEX_MAP.get(key, key)


@lru_cache(maxsize=256)
def yahoo_search_symbol(query: str) -> str | None:
    """
    Resolve a free-form query (company/index name) to a Yahoo Finance symbol.
    Returns the best match symbol or None if no hits.

    Examples:
        "ABI SA" -> "ABI.BR"
        "Apple" -> "AAPL"
        "S&P 500" -> "^GSPC"
    """
    try:
        q = query.strip()
        if not q:
            return None
        # If the user already entered a plausible ticker (short alnum + punctuation),
        # try to fast-path return it (validation step will confirm).
        if len(q) <= 6 and all(ch.isalnum() or ch in ".^-" for ch in q):
            return q.upper()

        url = "https://query1.finance.yahoo.com/v1/finance/search"
        resp = requests.get(url, params={"q": q}, timeout=6)
        if resp.status_code != 200:
            return None
        data = resp.json()
        quotes = data.get("quotes") or []
        if not quotes:
            return None

        # Heuristic: prefer symbols that have an exchange + score
        best = None
        best_score = -1.0
        for item in quotes:
            sym = item.get("symbol")
            score = 0.0
            if item.get("exchange"):
                score += 1.0
            if item.get("longname") or item.get("shortname"):
                score += 0.5
            if sym and sym.upper() == q.upper():
                score += 1.0
            if score > best_score and sym:
                best = sym
                best_score = score

        return best.upper() if best else None
    except Exception:
        return None


def resolve_tickers(user_inputs: List[str]) -> Tuple[List[str], List[str]]:
    """
    For each user input, try:
      1) index mapping (normalize_ticker)
      2) Yahoo symbol search (yahoo_search_symbol)

    Returns (resolved_symbols, messages)
      - resolved_symbols: valid symbols (not yet validated with yfinance)
      - messages: info strings describing any interpretation performed
    Deduplicates while preserving order.
    """
    seen = set()
    resolved: List[str] = []
    messages: List[str] = []

    for raw in user_inputs:
        raw = (raw or "").strip()
        if not raw:
            continue

        # Step 1: normalize common names/indices
        norm = normalize_ticker(raw)
        candidate = norm

        # Step 2: if unchanged by normalize, try Yahoo search
        if norm == raw.upper():
            suggestion = yahoo_search_symbol(raw)
            if suggestion and suggestion != raw.upper():
                messages.append(f"ℹ️ '{raw}' interpreted as '{suggestion}'.")
                candidate = suggestion
            elif suggestion:
                candidate = suggestion  # likely raw was already a symbol

        sym = candidate.upper()
        if sym not in seen:
            seen.add(sym)
            resolved.append(sym)

    return resolved, messages


def validate_tickers(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Use yfinance to check if a symbol looks valid.
    We consider it valid if info contains 'regularMarketPrice' or
    we can fetch non-empty historical data for the range later.
    Returns (valid_symbols, warnings)
    """
    valid: List[str] = []
    warnings: List[str] = []
    for t in symbols:
        try:
            # info can be slow/inconsistent; still useful as a quick check
            info = yf.Ticker(t).info
            if info and "regularMarketPrice" in info:
                valid.append(t)
            else:
                # Defer full validation to the download step; keep it for now
                valid.append(t)
        except Exception:
            # If info errors, still allow it to proceed to download validation
            valid.append(t)
    return valid, warnings


def ensure_1d(series_like: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Ensure a pandas Series is 1D (shape-safe) for Plotly/math.
    """
    s = series_like.squeeze("columns") if isinstance(series_like, pd.DataFrame) else series_like
    if getattr(s, "ndim", 1) > 1:
        s = pd.Series(np.ravel(s), index=getattr(s, "index", None))
    return s


def safe_weights_or_none(weights_text: str, n_assets: int) -> np.ndarray | None:
    """
    Parse and validate weights. If invalid length/sum or negatives, return None and let UI fall back to equal-weight.
    """
    if not weights_text.strip():
        return None
    try:
        w = np.array([float(x.strip()) for x in weights_text.split(",")], dtype=float)
    except Exception:
        st.error("❌ Weights must be numbers separated by commas.")
        return None

    if len(w) != n_assets:
        st.warning(f"⚠️ Provided {len(w)} weights for {n_assets} tickers. Falling back to equal-weight.")
        return None
    if np.any(np.isnan(w)) or np.any(w < 0):
        st.warning("⚠️ Weights contain NaN/negative values. Falling back to equal-weight.")
        return None
    s = w.sum()
    if s <= 0:
        st.warning("⚠️ Weights sum to 0. Falling back to equal-weight.")
        return None
    # Normalize to 1
    return w / s


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
# Active Profile Banner (with persistence)
# ===============================================================
st.markdown("### 🧭 Active Profile")

active_id = st.session_state.get("active_profile_id")
if not active_id:
    try:
        profiles = get_profiles()
        if profiles:
            latest_profile = sorted(profiles, key=lambda p: p.updated_at or p.created_at)[-1]
            active_id = latest_profile.id
            st.session_state["active_profile_id"] = active_id
    except Exception:
        # If DB not ready, don't block the page
        active_id = None

if active_id:
    profile = get_profile(active_id)
    if profile:
        region = profile.constraints.get("preferred_region", "—") if profile.constraints else "—"
        st.success(
            f"**{profile.name or 'Unnamed Profile'}**  |  "
            f"Risk Score: {profile.risk_score}/10  |  "
            f"Region: {region}"
        )
    else:
        st.warning("⚠️ Active profile not found. Please create one in the Profile Manager.")
else:
    st.info("ℹ️ No profile found. Go to the Profile Manager page to create one.")

st.divider()

# ===============================================================
# Sidebar
# ===============================================================
with st.sidebar:
    st.header("⚙️ Analysis Settings")

    tickers_input = st.text_input("Enter tickers (comma-separated):", value="AAPL, MSFT, NVDA")
    user_inputs = [t for t in (x.strip() for x in tickers_input.split(",")) if t]

    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    risk_free_rate = st.number_input(
        "Annual Risk-Free Rate (as decimal)",
        min_value=0.0, max_value=0.10, value=0.02, step=0.005
    )

    st.subheader("Benchmark Settings")
    benchmark_input = st.text_input("Enter benchmark (ticker or name, e.g., SPY / S&P 500):", value="SPY").strip()

    st.subheader("Portfolio Weights (optional)")
    weights_text = st.text_input("Enter weights (comma-separated, must match number of tickers):", value="")

    run_analysis = st.button("🚀 Run Analysis")

# ===============================================================
# Main Output
# ===============================================================
if run_analysis:
    try:
        # -----------------------------------------------------------
        # Resolve & validate tickers
        # -----------------------------------------------------------
        resolved, info_msgs = resolve_tickers(user_inputs)
        for m in info_msgs:
            st.info(m)

        if not resolved:
            st.error("❌ Please enter at least one valid ticker or name.")
            st.stop()

        valid_tickers, val_warnings = validate_tickers(resolved)
        for w in val_warnings:
            st.warning(w)

        if not valid_tickers:
            st.error("❌ No valid tickers found. Try symbols like 'AAPL', 'ABI.BR', or index codes like '^FCHI'.")
            st.stop()

        # Parse / validate weights
        w_vec = safe_weights_or_none(weights_text, len(valid_tickers))

        # -----------------------------------------------------------
        # Calculate Sharpe Ratio and metrics (table at top)
        # -----------------------------------------------------------
        result = calculate_sharpe_ratio(
            tickers=valid_tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_rate_annual=risk_free_rate,
            weights=None if w_vec is None else list(map(float, w_vec)),
        )

        if result.empty or result.isna().all().all():
            st.error("❌ No analytics computed (empty result). Check tickers/date range.")
            st.stop()

        st.success("✅ Analysis Complete!")

        st.subheader("📋 Sharpe Ratio Results")
        st.dataframe(result, use_container_width=True)

        sharpe = float(result.get("sharpe_annualized", pd.Series([np.nan])).iloc[0])
        mean_return = float(result.get("mean_daily_return", pd.Series([np.nan])).iloc[0]) * 252.0
        volatility = float(result.get("daily_volatility", pd.Series([np.nan])).iloc[0]) * math.sqrt(252.0)

        # -----------------------------------------------------------
        # Summary Cards
        # -----------------------------------------------------------
        st.markdown("### 🧾 Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Annualized Return", f"{mean_return:.2%}" if not np.isnan(mean_return) else "—")
        col2.metric("Annualized Volatility", f"{volatility:.2%}" if not np.isnan(volatility) else "—")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "—")

        # -----------------------------------------------------------
        # Download price data
        # -----------------------------------------------------------
        data = yf.download(
            tickers=valid_tickers,
            start=str(start_date),
            end=str(end_date),
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )

        if len(valid_tickers) == 1:
            prices = pd.DataFrame(data["Close"]) if "Close" in data else data
            prices.columns = [valid_tickers[0]]
        else:
            # Try to pull close for each; if some keys missing, fallback
            cols = {}
            for t in valid_tickers:
                try:
                    cols[t] = data[t]["Close"]
                except Exception:
                    try:
                        cols[t] = data[t]
                    except Exception:
                        st.warning(f"⚠️ No price column found for '{t}'. Skipping in price matrix.")
            if not cols:
                st.error("❌ No valid price data retrieved for any ticker.")
                st.stop()
            prices = pd.concat(cols, axis=1)

        prices.dropna(how="all", inplace=True)
        if prices.empty:
            st.error("❌ No valid price data retrieved. Please verify tickers or try another range.")
            st.stop()

        daily_returns = prices.pct_change().dropna(how="all")
        if daily_returns.empty:
            st.error("❌ Could not compute daily returns (empty after pct_change).")
            st.stop()

        # Apply weights (equal-weight fallback)
        if w_vec is None:
            portfolio_returns = daily_returns.mean(axis=1)
        else:
            # Align columns in case some tickers dropped due to NA
            aligned = daily_returns.reindex(columns=valid_tickers).fillna(0.0)
            portfolio_returns = (aligned.values @ w_vec).astype(float)
            portfolio_returns = pd.Series(portfolio_returns, index=aligned.index, name="Portfolio")

        # --- Ensure clean 1D Series ---
        portfolio_returns = ensure_1d(portfolio_returns)
        if portfolio_returns.empty or portfolio_returns.isna().all():
            st.error("❌ No valid return data computed.")
            st.stop()

        cumulative = ensure_1d((1.0 + portfolio_returns).cumprod())

        # -----------------------------------------------------------
        # Benchmark (resolve/validate too)
        # -----------------------------------------------------------
        benchmark_sym = normalize_ticker(benchmark_input)
        # Try search if not a mapped index and not short symbol
        if benchmark_sym == benchmark_input.upper():
            maybe = yahoo_search_symbol(benchmark_input)
            if maybe and maybe != benchmark_sym:
                st.info(f"ℹ️ Benchmark '{benchmark_input}' interpreted as '{maybe}'.")
                benchmark_sym = maybe

        benchmark_cumulative = None
        try:
            bench = yf.download(
                tickers=benchmark_sym,
                start=str(start_date),
                end=str(end_date),
                progress=False,
                auto_adjust=True,
            )
            if not bench.empty:
                bret = bench["Close"].pct_change().dropna()
                benchmark_cumulative = ensure_1d((1.0 + bret).cumprod())
            else:
                st.warning(f"⚠️ Benchmark '{benchmark_sym}' returned no data; plotting portfolio only.")
        except Exception:
            st.warning(f"⚠️ Could not download benchmark '{benchmark_sym}'. Plotting portfolio only.")

        # -----------------------------------------------------------
        # Plotly: Cumulative Chart  (FIXED: x uses 'Date' column)
        # -----------------------------------------------------------
        st.subheader("📈 Cumulative Portfolio Growth vs Benchmark")
        if benchmark_cumulative is not None and not benchmark_cumulative.empty:
            combined = pd.DataFrame({"Portfolio": cumulative}).join(
                benchmark_cumulative.rename(benchmark_sym), how="outer"
            )
            combined_reset = combined.reset_index().rename(columns={"index": "Date"})
            y_cols = [c for c in combined_reset.columns if c != "Date"]
            fig_cum = px.line(
                combined_reset,
                x="Date",
                y=y_cols,
                title=f"Cumulative Growth: Portfolio vs {benchmark_sym.upper()}",
                labels={"Date": "Date", "value": "Growth (1 = Start)", "variable": "Series"},
            )
        else:
            combined = pd.DataFrame({"Portfolio": cumulative})
            combined_reset = combined.reset_index().rename(columns={"index": "Date"})
            fig_cum = px.line(
                combined_reset,
                x="Date",
                y="Portfolio",
                title="Cumulative Growth: Portfolio",
                labels={"Date": "Date", "Portfolio": "Growth (1 = Start)"},
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
        # Plotly: Rolling Sharpe Ratio (Interactive)
        # -----------------------------------------------------------
        st.subheader("📊 Rolling Sharpe Ratio (60-day window)")
        window = 60
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        rolling_sharpe = ensure_1d((rolling_mean / rolling_std) * math.sqrt(252.0)).dropna()

        rolling_df = rolling_sharpe.reset_index().rename(columns={"index": "Date", 0: "Sharpe"})
        if "Sharpe" not in rolling_df.columns:
            # if the series had a name, keep it; otherwise, force 'Sharpe'
            value_col = [c for c in rolling_df.columns if c != "Date"][0]
            rolling_df = rolling_df.rename(columns={value_col: "Sharpe"})

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
        st.error(f"❌ Unexpected error: {e}")
else:
    st.info("👈 Configure your settings and click **Run Analysis** to begin.")
