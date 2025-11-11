from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Ensure imports resolve correctly when run via Streamlit
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Valuation Dashboard — AlphaInsights",
    layout="wide",
)

st.title("💹 Valuation Intelligence Dashboard")
st.caption(
    "Compare valuation multiples, quality signals, and payout metrics across assets. "
    "Backend-aware, Yahoo Finance–backed. Educational analytics only — not investment advice."
)

# ---------------------------------------------------------------------------
# Helper: safe backend wrapper (compatible with evolving fetch_backend)
# ---------------------------------------------------------------------------

def call_backend(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Try to call the FastAPI backend via core.ui_helpers.fetch_backend.
    Falls back to direct HTTP GET if the helper signature is older.

    Returns:
        dict payload on success, or None on failure.
    """
    # Normalize path
    normalized_path = path.lstrip("/")

    # 1) Try modern-style helper (path relative to BACKEND_URL)
    try:
        data = fetch_backend(normalized_path, params=params)
        if isinstance(data, dict):
            return data
    except TypeError:
        # Helper may expect a full URL; fall through to requests.
        pass
    except Exception:
        # Any other runtime backend issue → fall back gracefully.
        return None

    # 2) Fallback: direct HTTP GET
    try:
        base = BACKEND_URL.rstrip("/") if BACKEND_URL else "http://127.0.0.1:8000"
        url = f"{base}/{normalized_path}"
        resp = requests.get(url, params=params, timeout=6)
        if resp.ok:
            payload = resp.json()
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None

    return None


def render_backend_health() -> None:
    """Lightweight backend health visualization."""
    with st.expander("🩺 Backend Health", expanded=True):
        payload = call_backend("health")
        if isinstance(payload, dict) and payload.get("status") == "ok":
            st.success("✅ Backend reachable")
            st.json(payload)
        elif payload:
            st.warning("⚠️ Backend responded, but health is not 'ok':")
            st.json(payload)
        else:
            st.info("Backend not reachable or /health not implemented. Falling back to direct market data.")


# ---------------------------------------------------------------------------
# Input Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Valuation Settings")

    tickers_raw = st.text_input(
        "Tickers to analyze (comma-separated)",
        value="AAPL, MSFT, NVDA, SPY",
        help="Use listed tickers (e.g. AAPL, MSFT, NVDA, SPY)."
    )

    benchmark = st.text_input(
        "Optional benchmark (e.g., SPY, ^GSPC)",
        value="SPY",
        help="Shown for context only; not required."
    ).strip().upper()

    region_hint = st.selectbox(
        "Region / Market (hint only)",
        options=["(auto)", "US", "EU", "Global"],
        index=0,
        help="For future backend extensions; does not filter yet."
    )

    show_quality = st.checkbox(
        "Show quality & payout signals (ROE, margins, dividend yield)",
        value=True,
    )

    show_risk_flags = st.checkbox(
        "Highlight potential valuation risk flags",
        value=True,
    )

    run_analysis = st.button("🚀 Run Valuation Scan", type="primary")

render_backend_health()

st.markdown("### 📊 Valuation Overview")
st.write(
    "This dashboard unifies **multiples**, **profitability**, and **payouts** "
    "to help you quickly see which names look stretched or attractive "
    "relative to peers in your selection."
)

# ---------------------------------------------------------------------------
# Yahoo Finance Fallbacks
# ---------------------------------------------------------------------------

VALUATION_FIELDS = [
    "symbol",
    "shortName",
    "currency",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "enterpriseToEbitda",
    "enterpriseToRevenue",
    "trailingEps",
    "pegRatio",
    "dividendYield",
    "payoutRatio",
    "profitMargins",
    "returnOnEquity",
]


def fetch_yf_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Pull a compact valuation snapshot for one ticker using yfinance.
    Designed to be robust against missing keys and API quirks.
    """
    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
    except Exception:
        info = {}

    row = {"symbol": ticker}

    for f in VALUATION_FIELDS:
        if f == "symbol":
            continue
        row[f] = info.get(f, None)

    # Derive a couple of helpful fallbacks
    if row.get("dividendYield") is not None and row["dividendYield"] > 0 and row["dividendYield"] < 1:
        # yfinance often returns dividendYield as fraction
        row["dividendYieldPct"] = float(row["dividendYield"]) * 100.0
    elif row.get("dividendYield") is not None and row["dividendYield"] >= 1:
        # already a percentage-like value
        row["dividendYieldPct"] = float(row["dividendYield"])
    else:
        row["dividendYieldPct"] = None

    return row


def load_backend_or_yf(tickers: List[str]) -> pd.DataFrame:
    """
    Try backend valuation summary first; fall back to Yahoo Finance snapshots.

    Expected backend shape (flexible, best-effort):
      {
        "valuations": [
          { "symbol": "...", "trailingPE": ..., "priceToBook": ..., ... },
          ...
        ]
      }
    """
    # 1) Try backend
    backend_payload = call_backend("valuation/summary", params={"tickers": ",".join(tickers)})

    rows: List[Dict[str, Any]] = []

    if isinstance(backend_payload, dict) and backend_payload.get("valuations"):
        for item in backend_payload["valuations"]:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or item.get("ticker") or "").upper()
            if not sym:
                continue
            row = {"source": "backend", **item}
            row["symbol"] = sym
            rows.append(row)

    # Track which tickers still need data
    have = {r["symbol"] for r in rows if "symbol" in r}
    missing = [t for t in tickers if t not in have]

    # 2) Fallback for missing via yfinance
    for t in missing:
        snap = fetch_yf_snapshot(t)
        snap["source"] = "yfinance"
        rows.append(snap)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize nice columns; don't crash if some are missing
    if "shortName" in df.columns:
        df["name"] = df["shortName"].fillna(df["symbol"])
    else:
        df["name"] = df["symbol"]

    # Sort by symbol for consistency
    df = df.sort_values("symbol").reset_index(drop=True)
    return df


def compute_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple peer-relative valuation markers:
      - valuation_score: combines P/E, P/B, EV/EBITDA (lower is cheaper)
      - quality_score: combines ROE & margins (higher is better)
    Scores are normalized 0–100 within the selected universe.
    """
    df = df.copy()

    # Helper: safe inverse for valuation multiples
    def inv(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        return 1.0 / s.replace(0, np.nan)

    cheap_signal = (
        0.5 * inv("trailingPE") +
        0.3 * inv("priceToBook") +
        0.2 * inv("enterpriseToEbitda")
    )

    qual_signal = (
        0.6 * pd.to_numeric(df.get("returnOnEquity"), errors="coerce") +
        0.4 * pd.to_numeric(df.get("profitMargins"), errors="coerce")
    )

    def to_0_100(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() < 2:
            return pd.Series(np.nan, index=s.index)
        lo, hi = float(s.min()), float(s.max())
        if hi <= lo:
            return pd.Series(50.0, index=s.index)
        return (s - lo) / (hi - lo) * 100.0

    df["valuation_score"] = to_0_100(cheap_signal)      # higher = cheaper vs peers
    df["quality_score"] = to_0_100(qual_signal)         # higher = better quality

    return df


def flag_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight risk/attention flags:
      - Very high P/E
      - Negative margins
      - Extremely high P/B with low ROE
    """
    df = df.copy()

    pe = pd.to_numeric(df.get("trailingPE"), errors="coerce")
    pb = pd.to_numeric(df.get("priceToBook"), errors="coerce")
    pm = pd.to_numeric(df.get("profitMargins"), errors="coerce")
    roe = pd.to_numeric(df.get("returnOnEquity"), errors="coerce")

    flags: List[str] = []
    for i in range(len(df)):
        msgs = []
        if not np.isnan(pe.iloc[i]) and pe.iloc[i] > 40:
            msgs.append("P/E > 40")
        if not np.isnan(pm.iloc[i]) and pm.iloc[i] < 0:
            msgs.append("Negative margins")
        if (
            not np.isnan(pb.iloc[i])
            and pb.iloc[i] > 6
            and (np.isnan(roe.iloc[i]) or roe.iloc[i] < 0.08)
        ):
            msgs.append("High P/B with weak ROE")
        flags.append(" | ".join(msgs) if msgs else "")

    df["risk_flags"] = flags
    return df


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------
if run_analysis:
    # Parse tickers
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.error("❌ Please enter at least one ticker.")
        st.stop()

    if benchmark and benchmark not in tickers:
        tickers.append(benchmark)

    # Load valuations (backend + yfinance)
    with st.spinner("Fetching valuation data..."):
        df_val = load_backend_or_yf(tickers)

    if df_val.empty:
        st.error(
            "❌ Could not retrieve valuation data for the selected tickers. "
            "Check symbols or try again later."
        )
        st.stop()

    # Compute scores
    df_val = compute_relative_scores(df_val)

    if st.session_state.get("valuation_debug"):
        st.write("DEBUG snapshot:")
        st.write(df_val)

    # Optionally apply risk flags
    if st.session_state.get("always_flag_risks", True):
        df_val = flag_risks(df_val)
    elif "valuation_risk_flags" not in st.session_state:
        # still compute but hide from display toggles if needed later
        df_val = flag_risks(df_val)

    # Keep visible subset of columns (safe if missing)
    visible_cols = [
        "symbol",
        "name",
        "currency",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "enterpriseToEbitda",
        "pegRatio",
        "dividendYieldPct",
    ]
    if show_quality:
        visible_cols += ["profitMargins", "returnOnEquity"]
    visible_cols += ["valuation_score", "quality_score"]
    if show_risk_flags:
        visible_cols.append("risk_flags")

    visible_cols = [c for c in visible_cols if c in df_val.columns]

    st.subheader("📋 Valuation Snapshot")
    st.dataframe(
        df_val[visible_cols].sort_values("symbol"),
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------------------------
    # Visual 1: Peer-relative valuation score
    # -----------------------------------------------------------------------
    st.subheader("📈 Peer-relative Valuation Score (0–100)")
    df_score = df_val[["symbol", "valuation_score"]].dropna()
    if not df_score.empty:
        fig_vs = px.bar(
            df_score.sort_values("valuation_score", ascending=False),
            x="symbol",
            y="valuation_score",
            title="Higher = Cheaper vs Selected Peers (Blended Multiple View)",
            labels={"valuation_score": "Valuation Score (rel. cheapness)"},
            text=df_score.sort_values("valuation_score", ascending=False)["valuation_score"].map(
                lambda x: f"{x:.0f}"
            ),
        )
        fig_vs.update_traces(textposition="outside")
        fig_vs.update_layout(yaxis=dict(range=[0, 105]))
        st.plotly_chart(fig_vs, use_container_width=True)
    else:
        st.info("Not enough data to compute relative valuation scores.")

    # -----------------------------------------------------------------------
    # Visual 2: Quality vs Valuation Map
    # -----------------------------------------------------------------------
    st.subheader("🧭 Quality vs Valuation Map")
    if "valuation_score" in df_val.columns and "quality_score" in df_val.columns:
        df_qv = df_val[["symbol", "valuation_score", "quality_score"]].dropna()
        if not df_qv.empty:
            fig_qv = px.scatter(
                df_qv,
                x="valuation_score",
                y="quality_score",
                text="symbol",
                title="Quality vs Valuation — top-right = high quality & cheap",
                labels={
                    "valuation_score": "Cheaper ⟶ (Valuation Score)",
                    "quality_score": "Higher Quality ⟶ (Quality Score)",
                },
            )
            fig_qv.update_traces(textposition="top center")
            st.plotly_chart(fig_qv, use_container_width=True)
        else:
            st.info("Not enough data to render the quality vs valuation map.")
    else:
        st.info("Scores not available — check input symbols or data coverage.")

    # -----------------------------------------------------------------------
    # Risk Flag Summary
    # -----------------------------------------------------------------------
    if show_risk_flags and "risk_flags" in df_val.columns:
        flagged = df_val[df_val["risk_flags"].astype(str).str.len() > 0]
        if not flagged.empty:
            st.subheader("🚩 Attention & Outlier Flags")
            for _, row in flagged.iterrows():
                st.write(f"**{row['symbol']}** — {row['risk_flags']}")
        else:
            st.subheader("🚩 Attention & Outlier Flags")
            st.write("No major heuristic flags based on current data.")

    st.caption(
        "Valuation metrics sourced from backend where available, "
        "falling back to Yahoo Finance. All signals are heuristic and for research only."
    )

else:
    st.info("👈 Configure your tickers and hit **Run Valuation Scan** to generate a peer set analysis.")
