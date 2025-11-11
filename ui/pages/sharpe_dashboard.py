from __future__ import annotations

import os
import sys
import math
from functools import lru_cache
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt  # kept for potential future use
import yfinance as yf

# =============================================================================
# Path & Imports Bootstrapping
# =============================================================================
# Ensure proper import resolution when running:
#   streamlit run ui/overview.py
# or:
#   streamlit run ui/pages/sharpe_dashboard.py
# -----------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from analytics.sharpe_ratio import calculate_sharpe_ratio
from database.queries import get_profile, get_profiles
from core.symbol_resolver import resolve_symbol
from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL as CONFIG_BACKEND_URL

# =============================================================================
# Global Config
# =============================================================================
# Single source of truth for backend base URL.
# If you ever centralize this differently, update ui_config and we stay in sync.
DEFAULT_BACKEND_URL: str = CONFIG_BACKEND_URL or os.getenv(
    "ALPHAINSIGHTS_BACKEND_URL",
    "http://127.0.0.1:8000",
)

st.set_page_config(
    page_title="Sharpe Ratio Analysis — AlphaInsights",
    page_icon="📈",
    layout="wide",
)

# =============================================================================
# 1-D Safety Helper
# =============================================================================

def ensure_1d(series_like: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Ensure a pandas Series is strictly 1D for stable math/plotting.

    Enforces the global 1-D safety rule:
      - squeeze() to collapse (N,1)
      - np.ravel() to guarantee flat array
    """
    if isinstance(series_like, pd.DataFrame):
        s = series_like.squeeze("columns")
    else:
        s = series_like

    arr = np.ravel(s)
    idx = getattr(s, "index", None)
    if idx is not None:
        idx = idx[: len(arr)]
    return pd.Series(arr, index=idx)


# =============================================================================
# Ticker Resolution Helpers (Preserved + Hardened)
# =============================================================================

_INDEX_MAP: Dict[str, str] = {
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

    Deterministic, dependency-free mapping.
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
def yahoo_search_symbol(query: str) -> Optional[str]:
    """
    Resolve a free-form query (company/index name) to a Yahoo Finance symbol.

    Design:
      - Short ticker-like strings: returned as-is (uppercased).
      - Uses Yahoo search endpoint directly (NOT the backend).
      - Picks the most plausible candidate using a tiny scoring heuristic.
    """
    q = (query or "").strip()
    if not q:
        return None

    # Already looks like a ticker → let later validation handle it.
    if len(q) <= 6 and all(ch.isalnum() or ch in ".^-" for ch in q):
        return q.upper()

    try:
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

    Pipeline:
      1) Normalize known index names.
      2) Use symbol_resolver + Yahoo Finance search for others.
      3) Deduplicate while preserving order.

    Returns:
        resolved_symbols, messages (for transparency in UI)
    """
    seen: set[str] = set()
    resolved: List[str] = []
    messages: List[str] = []

    for raw in user_inputs:
        raw = (raw or "").strip()
        if not raw:
            continue

        norm = normalize_ticker(raw)
        candidate = norm

        if norm == raw.upper():
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

    Intentionally permissive:
      - We keep borderline tickers; final filter is price history.
    """
    valid: List[str] = []
    warnings: List[str] = []

    for t in symbols:
        try:
            _ = yf.Ticker(t).info  # Probe; failure is soft.
            valid.append(t)
        except Exception:
            # Keep anyway; history check will decide.
            valid.append(t)

    return valid, warnings


# =============================================================================
# Weight Parsing Helper
# =============================================================================

def safe_weights_or_none(weights_text: str, n_assets: int) -> Optional[np.ndarray]:
    """
    Parse and validate user-provided weights.

    Returns:
        - np.ndarray normalized to sum 1, if valid
        - None if invalid/incompatible → caller decides fallback.
    """
    if not weights_text.strip():
        return None

    try:
        w = np.array([float(x.strip()) for x in weights_text.split(",")], dtype=float)
    except Exception:
        st.error("❌ Weights must be numbers separated by commas.")
        return None

    if len(w) != n_assets:
        st.warning(
            f"⚠️ Provided {len(w)} weights for {n_assets} tickers. "
            f"Falling back to backend/equal-weight."
        )
        return None

    if np.any(np.isnan(w)) or np.any(w < 0):
        st.warning("⚠️ Weights contain NaN/negative values. Falling back to backend/equal-weight.")
        return None

    s = float(w.sum())
    if s <= 0:
        st.warning("⚠️ Weights sum to 0. Falling back to backend/equal-weight.")
        return None

    return w / s


# =============================================================================
# Backend Integration Helpers
# =============================================================================

def render_backend_health(user_backend_url: str) -> None:
    """
    Display backend health in the sidebar.

    Behavior:
      - If user overrides the URL in the sidebar → call that directly.
      - Otherwise → use unified fetch_backend("health") which is wired to BACKEND_URL.
    This keeps us forward compatible with the central fetch logic.
    """
    # Normalize override
    override = (user_backend_url or "").strip()
    use_override = override and override != DEFAULT_BACKEND_URL

    try:
        if use_override:
            # Direct call against custom URL
            url = override.rstrip("/") + "/health"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {}
            else:
                data = {}
        else:
            # Unified layer (preferred)
            data = fetch_backend("health")

        if isinstance(data, dict) and data.get("status") == "ok":
            st.sidebar.success("Backend: Online")
        elif isinstance(data, dict) and data:
            st.sidebar.warning("Backend reachable, but health check not OK.")
            st.sidebar.json(data)
        else:
            st.sidebar.error("Backend: Unreachable or invalid health response.")
    except Exception as e:
        st.sidebar.error(f"Backend: Unreachable ({e})")


def call_backend_sharpe(
    backend_url: str,
    tickers: List[str],
    start: str,
    end: str,
    rf: float,
) -> Dict[str, Any]:
    """
    Call /optimize/sharpe on the backend to obtain optimized weights & metrics.

    We intentionally keep this as a direct POST (not via fetch_backend)
    to avoid coupling to helper signatures and to preserve existing API expectations.
    """
    base = (backend_url or DEFAULT_BACKEND_URL).rstrip("/")
    url = f"{base}/optimize/sharpe"

    payload = {
        "tickers": tickers,
        "start": start,
        "end": end,
        "rf": rf,
        # Shared OptimizeRequest may require alpha; ignored by Sharpe logic.
        "alpha": 0.95,
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


# =============================================================================
# Header & Active Profile Banner
# =============================================================================

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
        constraints = getattr(profile, "constraints", {}) or {}  # type: ignore[attr-defined]
        region = constraints.get("preferred_region", "—")
        st.success(
            f"**{getattr(profile, 'name', 'Unnamed Profile')}**  |  "
            f"Risk Score: {getattr(profile, 'risk_score', '—')}/10  |  "
            f"Region: {region}"
        )
    else:
        st.warning("⚠️ Active profile not found. Please create one in the Profile Manager.")
else:
    st.info("ℹ️ No profile found. Go to the Profile Manager page to create one.")

st.divider()


# =============================================================================
# Sidebar Controls
# =============================================================================

with st.sidebar:
    st.header("⚙️ Analysis & Optimization Settings")

    backend_url_input = st.text_input(
        "Backend URL (override, optional)",
        value=DEFAULT_BACKEND_URL,
        help=(
            "FastAPI backend base URL. "
            "If left as default, calls go through the standard AlphaInsights config."
        ),
    )

    # Health summary (uses override if changed)
    render_backend_health(backend_url_input)

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


# =============================================================================
# Main Logic
# =============================================================================

if run_analysis:
    try:
        # ---------------------------------------------------------------------
        # 1. Resolve & validate tickers
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 2. Decide weights source (manual vs backend vs equal-weight)
        # ---------------------------------------------------------------------
        backend_used = False
        backend_data: Dict[str, Any] = {}

        w_vec = safe_weights_or_none(weights_text, len(valid_tickers))

        if w_vec is None:
            # No valid manual weights → try backend Sharpe optimizer
            start_str = str(start_date)
            end_str = str(end_date)

            with st.spinner("Calling backend Sharpe optimizer for weights..."):
                backend_data = call_backend_sharpe(
                    backend_url=backend_url_input,
                    tickers=valid_tickers,
                    start=start_str,
                    end=end_str,
                    rf=float(risk_free_rate),
                )

            if backend_data.get("weights"):
                backend_used = True
                weights_from_backend = backend_data["weights"]
                w_raw = np.array(
                    [float(weights_from_backend.get(t, 0.0)) for t in valid_tickers],
                    dtype=float,
                )
                s = float(w_raw.sum())
                if s > 0:
                    w_vec = w_raw / s
                else:
                    w_vec = np.ones(len(valid_tickers), dtype=float) / float(len(valid_tickers))
            else:
                st.warning(
                    "Backend Sharpe optimizer unavailable or returned no weights. "
                    "Falling back to equal-weight portfolio."
                )
                w_vec = np.ones(len(valid_tickers), dtype=float) / float(len(valid_tickers))
        # else: valid manual weights already chosen and respected.

        # ---------------------------------------------------------------------
        # 3. Download & clean price data
        # ---------------------------------------------------------------------
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
            cols: Dict[str, pd.Series] = {}
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

        # Keep columns with sufficient observations
        valid_price_cols = [c for c in prices.columns if prices[c].notna().sum() > 5]
        if not valid_price_cols:
            st.error("❌ All tickers returned insufficient data. Please check symbols or date range.")
            st.stop()

        dropped = [c for c in prices.columns if c not in valid_price_cols]
        if dropped:
            st.warning(
                f"⚠️ Dropped invalid/missing tickers due to insufficient data: "
                f"{', '.join(map(str, dropped))}"
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

        # ---------------------------------------------------------------------
        # 4. Compute Sharpe metrics via shared analytics module
        # ---------------------------------------------------------------------
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

        # Prefer backend metrics if backend weights used and no assets were dropped
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

        # ---------------------------------------------------------------------
        # 5. Summary Metrics
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 6. Portfolio Returns (cleaned universe & final weights)
        # ---------------------------------------------------------------------
        aligned = daily_returns.reindex(columns=effective_tickers).fillna(0.0)
        portfolio_returns = aligned.values @ w_vec
        portfolio_returns = pd.Series(portfolio_returns, index=aligned.index, name="Portfolio")
        portfolio_returns = ensure_1d(portfolio_returns)

        if portfolio_returns.empty or portfolio_returns.isna().all():
            st.error("❌ No valid portfolio return data computed after cleaning.")
            st.stop()

        cumulative = ensure_1d((1.0 + portfolio_returns).cumprod())

        # ---------------------------------------------------------------------
        # 7. Benchmark Handling
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # 8. Cumulative Growth Chart
        # ---------------------------------------------------------------------
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
                labels={
                    "Date": "Date",
                    "Portfolio": "Growth (1 = Start)",
                },
            )

        st.plotly_chart(fig_cum, use_container_width=True)

        # ---------------------------------------------------------------------
        # 9. Return Distribution
        # ---------------------------------------------------------------------
        st.subheader("📉 Daily Return Distribution")
        fig_hist = px.histogram(
            x=np.ravel(portfolio_returns.values),
            nbins=50,
            title="Distribution of Daily Returns",
            labels={"x": "Daily Return"},
            opacity=0.7,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ---------------------------------------------------------------------
        # 10. Rolling Sharpe Ratio
        # ---------------------------------------------------------------------
        st.subheader("📊 Rolling Sharpe Ratio (60-day window)")

        window = 60
        rolling_mean = portfolio_returns.rolling(window).mean()
        rolling_std = portfolio_returns.rolling(window).std()
        rolling_sharpe = ensure_1d(
            (rolling_mean / (rolling_std + 1e-12)) * math.sqrt(252.0)
        ).dropna()

        if not rolling_sharpe.empty:
            rolling_df = rolling_sharpe.reset_index().rename(columns={"index": "Date"})
            value_col_candidates = [c for c in rolling_df.columns if c != "Date"]
            if value_col_candidates:
                value_col = value_col_candidates[0]
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


# =============================================================================
# Compliance Disclaimer
# =============================================================================
st.caption(
    "AlphaInsights is an educational analytics prototype and does not provide investment advice."
)
