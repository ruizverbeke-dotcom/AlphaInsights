# ===============================================================
# Sharpe Ratio Dashboard — AlphaInsights UI
# ===============================================================
# Phase 6.0 — Backend-Integrated Sharpe Optimizer (Preserved UX)
#
# Author: Ruïz Verbeke
#
# What this page does
# -------------------
# - Preserves the mature Sharpe analysis flow:
#     • Active Profile banner with persistence
#     • Robust ticker resolution helpers
#     • Optional custom weights
#     • Benchmark comparison
#     • Cumulative chart, histogram, rolling Sharpe
# - Integrates the FastAPI /optimize/sharpe endpoint:
#     • If no custom weights are provided:
#           → use backend optimizer weights (Sharpe_SLSQP)
#     • If custom weights are provided:
#           → use them directly (no backend override)
# - Adds robust data validation:
#     • Drops tickers with insufficient data
#     • Realigns weights to surviving tickers
#     • Prevents "Analysis Complete" with NaN metrics
#
# Design Principles
# -----------------
# - No heavy optimization logic here (delegated to backend & analytics).
# - Enforces 1-D safety for all series.
# - Fails gracefully with clear messages, no raw tracebacks for users.
# - Ready for agents: inputs/outputs are clear and JSON-friendly.
# ===============================================================

from __future__ import annotations

import os
import sys
import math
from functools import lru_cache
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt  # kept for potential future use
import yfinance as yf

# --- Ensure the project root is in Python's import path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from analytics.sharpe_ratio import calculate_sharpe_ratio
from database.queries import get_profile, get_profiles
from core.symbol_resolver import resolve_symbol

# ===============================================================
# Global Config
# ===============================================================

DEFAULT_BACKEND_URL = os.getenv("ALPHAINSIGHTS_BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Sharpe Ratio Analysis — AlphaInsights",
    page_icon="📈",
    layout="wide",
)

# ===============================================================
# 1-D Safety Helper
# ===============================================================

def ensure_1d(series_like: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Ensure a pandas Series is strictly 1D for stable math/plotting.

    Enforces the global 1-D safety rule:
      - squeeze() to collapse (N,1)
      - np.ravel() to guarantee flat array
    """
    s = series_like.squeeze("columns") if isinstance(series_like, pd.DataFrame) else series_like
    s = pd.Series(
        np.ravel(s),
        index=getattr(s, "index", None)[: len(np.ravel(s))]
        if getattr(s, "index", None) is not None
        else None,
    )
    return s


# ===============================================================
# Ticker Resolution Helpers (Preserved Logic)
# ===============================================================

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
    Map common index names to canonical tickers; otherwise uppercase.

    This is a deterministic, lightweight mapping layer.
    A richer semantic resolver can plug in later without changing callers.
    """
    raw = (name or "").strip()
    if not raw:
        return ""
    key = raw.upper()
    for k, v in _INDEX_MAP.items():
        if key == k.upper():
            return v
    return key


@lru_cache(maxsize=256)
def yahoo_search_symbol(query: str) -> str | None:
    """
    Resolve a free-form query (company/index name) to a Yahoo Finance symbol.

    This is a pragmatic helper, not a magic AI resolver.
    It:
      - fast-paths plausible tickers,
      - queries Yahoo search,
      - picks the most reasonable candidate.
    """
    try:
        q = (query or "").strip()
        if not q:
            return None

        # If it already looks like a ticker, let data validation handle it later.
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

        best = None
        best_score = -1.0
        for item in quotes:
            sym = item.get("symbol")
            if not sym:
                continue
            score = 0.0
            if item.get("exchange"):
                score += 1.0
            if item.get("longname") or item.get("shortname"):
                score += 0.5
            if sym.upper() == q.upper():
                score += 1.0
            if score > best_score:
                best = sym
                best_score = score

        return best.upper() if best else None
    except Exception:
        return None


def resolve_tickers(user_inputs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Resolve user-provided labels into candidate tickers.

    Steps:
      1) Normalize known index names.
      2) Try Yahoo Finance search for others.
      3) Deduplicate while preserving order.

    Returns:
        resolved_symbols, messages (for transparency)
    """
    seen = set()
    resolved: List[str] = []
    messages: List[str] = []

    for raw in user_inputs:
        raw = (raw or "").strip()
        if not raw:
            continue

        norm = normalize_ticker(raw)
        candidate = norm

        if norm == raw.upper():
            # Phase 6.2: enhanced resolver (static + fuzzy)
            suggestion = resolve_symbol(raw) or yahoo_search_symbol(raw)

            if suggestion and suggestion != raw.upper():
                messages.append(f"ℹ️ '{raw}' interpreted as '{suggestion}'.")
                candidate = suggestion
            elif suggestion:
                candidate = suggestion

        sym = candidate.upper()
        if sym and sym not in seen:
            seen.add(sym)
            resolved.append(sym)

    return resolved, messages


def validate_tickers(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Light validation using yfinance metadata.

    We keep this permissive; hard filtering is applied after downloading prices.
    Returns:
        valid_symbols, warnings
    """
    valid: List[str] = []
    warnings: List[str] = []
    for t in symbols:
        try:
            info = yf.Ticker(t).info
            # Even if thin, we keep it; real check is price history.
            if info and "regularMarketPrice" in info:
                valid.append(t)
            else:
                valid.append(t)
        except Exception:
            valid.append(t)
    return valid, warnings


# ===============================================================
# Weight Parsing Helper
# ===============================================================

def safe_weights_or_none(weights_text: str, n_assets: int) -> np.ndarray | None:
    """
    Parse and validate user-provided weights.

    Returns:
        - np.ndarray normalized to sum 1, if valid
        - None if invalid/incompatible → caller decides fallback.
    """
    if not weights_text.strip():
        return None

    try:
        w = np.array(
            [float(x.strip()) for x in weights_text.split(",")],
            dtype=float,
        )
    except Exception:
        st.error("❌ Weights must be numbers separated by commas.")
        return None

    if len(w) != n_assets:
        st.warning(f"⚠️ Provided {len(w)} weights for {n_assets} tickers. Falling back to backend/equal-weight.")
        return None

    if np.any(np.isnan(w)) or np.any(w < 0):
        st.warning("⚠️ Weights contain NaN/negative values. Falling back to backend/equal-weight.")
        return None

    s = w.sum()
    if s <= 0:
        st.warning("⚠️ Weights sum to 0. Falling back to backend/equal-weight.")
        return None

    return w / s


# ===============================================================
# Backend Integration Helpers
# ===============================================================

def render_backend_health(backend_url: str) -> None:
    """
    Display backend health in the sidebar using /health.

    Never blocks the UI; purely informational.
    """
    url = backend_url.rstrip("/") + "/health"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            st.sidebar.success("Backend: Online")
        else:
            st.sidebar.warning(f"Backend issue (status {resp.status_code})")
    except Exception:
        st.sidebar.error("Backend: Unreachable")


def call_backend_sharpe(
    backend_url: str,
    tickers: List[str],
    start: str,
    end: str,
    rf: float,
) -> Dict[str, Any]:
    """
    Call /optimize/sharpe on the backend to obtain optimized weights & metrics.

    If it fails, returns {} and the caller gracefully falls back.
    """
    url = backend_url.rstrip("/") + "/optimize/sharpe"
    payload = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "rf": rf,
        "alpha": 0.95,  # required by shared OptimizeRequest; ignored by Sharpe logic
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
    except requests.RequestException as e:
        st.warning(f"Backend Sharpe optimizer unreachable: {e}")
        return {}

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.warning(f"Backend Sharpe error ({resp.status_code}): {detail}")
        return {}

    try:
        data = resp.json()
    except ValueError:
        st.warning("Backend returned non-JSON response for Sharpe optimization.")
        return {}

    return data or {}


# ===============================================================
# Header & Active Profile Banner
# ===============================================================

st.title("📊 Sharpe Ratio Analysis & Optimization Dashboard")
st.write(
    "Analyze Sharpe Ratios for assets or custom portfolios, compare against a benchmark, "
    "and (optionally) use the backend optimizer to propose Sharpe-maximizing weights."
)

st.markdown("### 🧭 Active Profile")

active_id = st.session_state.get("active_profile_id")
if not active_id:
    try:
        profiles = get_profiles()
        if profiles:
            latest_profile = sorted(
                profiles,
                key=lambda p: p.updated_at or p.created_at,
            )[-1]
            active_id = latest_profile.id
            st.session_state["active_profile_id"] = active_id
    except Exception:
        active_id = None  # DB not ready → soft fail

if active_id:
    profile = get_profile(active_id)
    if profile:
        constraints = getattr(profile, "constraints", {}) or {}
        region = constraints.get("preferred_region", "—")
        st.success(
            f"**{profile.name or 'Unnamed Profile'}**  |  "
            f"Risk Score: {getattr(profile, 'risk_score', '—')}/10  |  "
            f"Region: {region}"
        )
    else:
        st.warning("⚠️ Active profile not found. Please create one in the Profile Manager.")
else:
    st.info("ℹ️ No profile found. Go to the Profile Manager page to create one.")

st.divider()

# ===============================================================
# Sidebar Controls
# ===============================================================

with st.sidebar:
    st.header("⚙️ Analysis & Optimization Settings")

    backend_url = st.text_input(
        "Backend URL",
        value=DEFAULT_BACKEND_URL,
        help="FastAPI backend base URL (default: http://127.0.0.1:8000)",
    )

    render_backend_health(backend_url)

    tickers_input = st.text_input(
        "Enter tickers (comma-separated):",
        value="AAPL, MSFT, NVDA",
        help="Names or symbols allowed; resolver will interpret.",
    )

    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    risk_free_rate = st.number_input(
        "Annual Risk-Free Rate (as decimal)",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.005,
    )

    st.subheader("Benchmark")
    benchmark_input = st.text_input(
        "Benchmark (ticker or name, e.g., SPY / S&P 500):",
        value="SPY",
    ).strip()

    st.subheader("Portfolio Weights (optional)")
    weights_text = st.text_input(
        "Weights (comma-separated, match #tickers). Leave blank to use Sharpe optimizer:",
        value="",
    )

    run_analysis = st.button("🚀 Run Analysis", type="primary")

# ===============================================================
# Main Logic
# ===============================================================

if run_analysis:
    try:
        # -----------------------------
        # 1. Resolve & validate tickers
        # -----------------------------
        user_inputs = [t for t in (x.strip() for x in tickers_input.split(",")) if t]
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
            st.error(
                "❌ No valid tickers passed validation. "
                "Try symbols like 'AAPL', 'ABI.BR', or '^FCHI'."
            )
            st.stop()

        # -----------------------------
        # 2. Decide weights source
        # -----------------------------
        backend_used = False
        backend_data: Dict[str, Any] = {}

        # Try to parse user-provided weights (if any)
        w_vec = safe_weights_or_none(weights_text, len(valid_tickers))

        if w_vec is None:
            # No valid manual weights → try backend Sharpe optimizer
            start_str = str(start_date)
            end_str = str(end_date)

            with st.spinner("Calling backend Sharpe optimizer for weights..."):
                backend_data = call_backend_sharpe(
                    backend_url=backend_url,
                    tickers=valid_tickers,
                    start=start_str,
                    end=end_str,
                    rf=float(risk_free_rate),
                )

            if backend_data.get("weights"):
                backend_used = True
                weights_from_backend = backend_data["weights"]
                # Align to valid_tickers order
                w_vec = np.array(
                    [float(weights_from_backend.get(t, 0.0)) for t in valid_tickers],
                    dtype=float,
                )
                s = float(w_vec.sum())
                if s > 0:
                    w_vec = w_vec / s
                else:
                    w_vec = np.ones(len(valid_tickers)) / len(valid_tickers)
            else:
                # Backend not available/failed → equal-weight fallback
                st.warning(
                    "Backend Sharpe optimizer unavailable or returned no weights. "
                    "Falling back to equal-weight portfolio."
                )
                w_vec = np.ones(len(valid_tickers)) / len(valid_tickers)
        else:
            # Valid manual weights provided → respect them (no backend override)
            start_str = str(start_date)
            end_str = str(end_date)

        # -----------------------------
        # 3. Download & clean price data
        # -----------------------------
        data = yf.download(
            tickers=valid_tickers,
            start=str(start_date),
            end=str(end_date),
            progress=False,
            group_by="ticker",
            auto_adjust=True,
        )

        # Build prices matrix
        if len(valid_tickers) == 1:
            if "Close" in data:
                prices = pd.DataFrame(data["Close"])
            else:
                prices = data.copy()
            prices.columns = [valid_tickers[0]]
        else:
            cols = {}
            for t in valid_tickers:
                try:
                    if t in data and "Close" in data[t]:
                        cols[t] = data[t]["Close"]
                    elif t in data:
                        cols[t] = data[t]
                except Exception:
                    st.warning(f"⚠️ No price column found for '{t}'. Skipping.")
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

        # --- Data Validation & Auto-Clean Step (Long-Term Stable Pattern) ---
        # Keep only columns with sufficient non-NaN points
        valid_price_cols = [c for c in prices.columns if prices[c].notna().sum() > 5]
        if not valid_price_cols:
            st.error("❌ All tickers returned insufficient data. Please check symbols or date range.")
            st.stop()

        dropped = [c for c in prices.columns if c not in valid_price_cols]
        if dropped:
            st.warning(
                f"⚠️ Dropped invalid/missing tickers due to insufficient data: {', '.join(dropped)}"
            )

        prices = prices[valid_price_cols]
        daily_returns = daily_returns[valid_price_cols].dropna(how="all")

        effective_tickers = list(valid_price_cols)

        # Realign weights to surviving tickers
        if w_vec is not None:
            if len(w_vec) != len(effective_tickers):
                st.warning(
                    "⚠️ Weight vector length mismatch after cleaning — "
                    "re-normalizing across remaining tickers."
                )
                w_vec = np.ones(len(effective_tickers), dtype=float) / float(len(effective_tickers))
        else:
            w_vec = np.ones(len(effective_tickers), dtype=float) / float(len(effective_tickers))

        # -----------------------------
        # 4. Compute Sharpe metrics (using cleaned universe)
        # -----------------------------
        result = calculate_sharpe_ratio(
            tickers=effective_tickers,
            start_date=str(start_date),
            end_date=str(end_date),
            risk_free_rate_annual=risk_free_rate,
            weights=list(map(float, w_vec)),
        )

        if result.empty or result.isna().all().all():
            st.error("❌ No analytics computed (empty/NaN result). Check tickers/date range.")
            st.stop()

        st.success("✅ Analysis Complete!")

        st.subheader("📋 Sharpe Ratio Results")
        st.dataframe(result, use_container_width=True)

        # Prefer backend-reported metrics if backend weights were used and no tickers were dropped
        if backend_used and backend_data and not dropped:
            sharpe_val = float(backend_data.get("sharpe", 0.0))
            ann_ret = float(backend_data.get("ann_return", 0.0))
            ann_vol = float(backend_data.get("ann_vol", 0.0))
        else:
            sharpe_series = result.get("sharpe_annualized", pd.Series([np.nan]))
            sharpe_val = float(sharpe_series.iloc[0])
            mean_ret = float(result.get("mean_daily_return", pd.Series([np.nan])).iloc[0])
            vol_daily = float(result.get("daily_volatility", pd.Series([np.nan])).iloc[0])
            ann_ret = mean_ret * 252.0 if not math.isnan(mean_ret) else float("nan")
            ann_vol = vol_daily * math.sqrt(252.0) if not math.isnan(vol_daily) else float("nan")

        # -----------------------------
        # 5. Summary Metrics
        # -----------------------------
        st.markdown("### 🧾 Key Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Annualized Return",
            f"{ann_ret:.2%}" if not math.isnan(ann_ret) else "—",
        )
        col2.metric(
            "Annualized Volatility",
            f"{ann_vol:.2%}" if not math.isnan(ann_vol) else "—",
        )
        col3.metric(
            "Sharpe Ratio",
            f"{sharpe_val:.2f}" if not math.isnan(sharpe_val) else "—",
        )

        if backend_used and backend_data.get("summary"):
            st.info(f"Backend optimizer: {backend_data['summary']}")

        # -----------------------------
        # 6. Portfolio Returns (using cleaned data & weights)
        # -----------------------------
        aligned = daily_returns.reindex(columns=effective_tickers).fillna(0.0)
        portfolio_returns = aligned.values @ w_vec
        portfolio_returns = pd.Series(portfolio_returns, index=aligned.index, name="Portfolio")
        portfolio_returns = ensure_1d(portfolio_returns)

        if portfolio_returns.empty or portfolio_returns.isna().all():
            st.error("❌ No valid portfolio return data computed after cleaning.")
            st.stop()

        cumulative = ensure_1d((1.0 + portfolio_returns).cumprod())

        # -----------------------------
        # 7. Benchmark
        # -----------------------------
        benchmark_sym = normalize_ticker(benchmark_input) if benchmark_input else ""
        if benchmark_sym == benchmark_input.upper() and benchmark_input:
            maybe = yahoo_search_symbol(benchmark_input)
            if maybe and maybe != benchmark_sym:
                st.info(f"ℹ️ Benchmark '{benchmark_input}' interpreted as '{maybe}'.")
                benchmark_sym = maybe

        benchmark_cumulative = None
        if benchmark_sym:
            try:
                bench = yf.download(
                    tickers=benchmark_sym,
                    start=str(start_date),
                    end=str(end_date),
                    progress=False,
                    auto_adjust=True,
                )
                if not bench.empty:
                    b_close = bench["Close"] if "Close" in bench else bench.iloc[:, 0]
                    bret = b_close.pct_change().dropna()
                    benchmark_cumulative = ensure_1d((1.0 + bret).cumprod())
                else:
                    st.warning(
                        f"⚠️ Benchmark '{benchmark_sym}' returned no data; plotting portfolio only."
                    )
            except Exception:
                st.warning(
                    f"⚠️ Could not download benchmark '{benchmark_sym}'. Plotting portfolio only."
                )

        # -----------------------------
        # 8. Cumulative Chart
        # -----------------------------
        st.subheader("📈 Cumulative Portfolio Growth vs Benchmark")

        if benchmark_cumulative is not None and not benchmark_cumulative.empty:
            combined = pd.DataFrame({"Portfolio": cumulative}).join(
                benchmark_cumulative.rename(benchmark_sym),
                how="outer",
            )
            combined_reset = combined.reset_index().rename(columns={"index": "Date"})
            y_cols = [c for c in combined_reset.columns if c != "Date"]
            fig_cum = px.line(
                combined_reset,
                x="Date",
                y=y_cols,
                title=f"Cumulative Growth: Portfolio vs {benchmark_sym.upper()}",
                labels={
                    "Date": "Date",
                    "value": "Growth (1 = Start)",
                    "variable": "Series",
                },
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

        # -----------------------------
        # 9. Return Distribution
        # -----------------------------
        st.subheader("📉 Daily Return Distribution")
        fig_hist = px.histogram(
            x=np.ravel(portfolio_returns.values),
            nbins=50,
            title="Distribution of Daily Returns",
            labels={"x": "Daily Return"},
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # -----------------------------
        # 10. Rolling Sharpe Ratio
        # -----------------------------
        st.subheader("📊 Rolling Sharpe Ratio (60-day window)")
        window = 60
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        rolling_sharpe = ensure_1d(
            (rolling_mean / (rolling_std + 1e-12)) * math.sqrt(252.0)
        ).dropna()

        if not rolling_sharpe.empty:
            rolling_df = rolling_sharpe.reset_index().rename(columns={"index": "Date"})
            value_col = [c for c in rolling_df.columns if c != "Date"][0]
            rolling_df = rolling_df.rename(columns={value_col: "Sharpe"})
            fig_roll = px.line(
                rolling_df,
                x="Date",
                y="Sharpe",
                title=f"Rolling Sharpe Ratio ({window}-day window)",
                labels={"Sharpe": "Sharpe Ratio"},
            )
            fig_roll.update_layout(
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Sharpe Ratio",
                template="plotly_white",
            )
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Not enough data to compute rolling Sharpe.")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
else:
    st.info("👈 Configure your settings and click **Run Analysis** to begin.")

# ===============================================================
# Compliance Disclaimer
# ===============================================================
st.caption(
    "AlphaInsights is an educational analytics prototype and does not provide investment advice."
)
