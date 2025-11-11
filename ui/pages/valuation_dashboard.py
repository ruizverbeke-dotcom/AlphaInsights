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
# Symbol normalization helpers (user-friendly inputs: gold, cac 40, alphabet)
# ---------------------------------------------------------------------------

_CANONICAL_MAP = {
    "GOLD": "GC=F",           # generic "gold" → futures proxy
    "XAUUSD": "XAUUSD=X",
    "CAC40": "^FCHI",
    "CAC 40": "^FCHI",
    "S&P500": "^GSPC",
    "S&P 500": "^GSPC",
    "SP500": "^GSPC",
    "EURO STOXX 50": "^STOXX50E",
    "STOXX600": "^STOXX",
    "FTSE100": "^FTSE",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "BEL20": "^BFX",
    "BEL 20": "^BFX",
    "NASDAQ": "^NDX",
    "NASDAQ 100": "^NDX",
    "NDX": "^NDX",
    "ALPHABET": "GOOGL",      # heuristic
}


def normalize_symbol(label: str) -> str:
    """
    Normalize a free-form label into a best-effort ticker.

    This is intentionally simple & deterministic:
    - Trims whitespace
    - Uppercases
    - Applies a small static mapping for common indices / aliases.
    """
    raw = (label or "").strip()
    if not raw:
        return ""
    up = raw.upper()
    return _CANONICAL_MAP.get(up, up)


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
    normalized_path = path.lstrip("/")

    # 1) Preferred: helper with relative path
    try:
        data = fetch_backend(normalized_path, params=params)
        if isinstance(data, dict):
            # If backend explicitly signals error/404 semantics, treat as no data
            if any(k in data for k in ("error", "detail")) and not data.get("valuations"):
                return None
            return data
    except TypeError:
        # Older helper signature → fall through to manual requests
        pass
    except Exception:
        # Any runtime/backend issue → just return None (UI will fall back)
        return None

    # 2) Fallback: direct HTTP GET to BACKEND_URL
    try:
        base = BACKEND_URL.rstrip("/") if BACKEND_URL else "http://127.0.0.1:8000"
        url = f"{base}/{normalized_path}"
        resp = requests.get(url, params=params, timeout=6)
        if resp.ok:
            payload = resp.json()
            if isinstance(payload, dict):
                if any(k in payload for k in ("error", "detail")) and not payload.get("valuations"):
                    return None
                return payload
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
        elif isinstance(payload, dict):
            st.warning("⚠️ Backend responded, but health is not 'ok':")
            st.json(payload)
        else:
            st.info(
                "Backend not reachable or /health not implemented. "
                "Valuation data will fall back to Yahoo Finance where needed."
            )


# ---------------------------------------------------------------------------
# Input Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Valuation Settings")

    tickers_raw = st.text_input(
        "Tickers to analyze (comma-separated)",
        value="AAPL, MSFT, NVDA, SPY, gold, cac 40, alphabet",
        help="Use tickers or well-known names; some aliases are normalized automatically.",
    )

    benchmark = st.text_input(
        "Optional benchmark (e.g., SPY, ^GSPC)",
        value="SPY",
        help="Shown for context only; not required.",
    ).strip()

    region_hint = st.selectbox(
        "Region / Market (hint only)",
        options=["(auto)", "US", "EU", "Global"],
        index=0,
        help="Reserved for future backend extensions; does not filter yet.",
    )

    show_quality = st.checkbox(
        "Show quality & payout signals (ROE, margins, dividend yield)",
        value=True,
    )

    show_risk_flags = st.checkbox(
        "Highlight potential valuation risk flags",
        value=True,
    )

    show_source = st.checkbox(
        "Show data source (backend vs Yahoo Finance)",
        value=False,
    )

    run_analysis = st.button("🚀 Run Valuation Scan", type="primary")

# Render health after controls (consistent across dashboards)
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
    Robust against missing keys and API quirks.
    """
    row: Dict[str, Any] = {"symbol": ticker}

    try:
        t = yf.Ticker(ticker)
        info = getattr(t, "info", {}) or {}
    except Exception:
        info = {}

    for f in VALUATION_FIELDS:
        if f == "symbol":
            continue
        row[f] = info.get(f, None)

    # Normalize dividend yield to percentage when possible
    dy = row.get("dividendYield")
    if dy is not None:
        try:
            dy = float(dy)
            if 0 < dy < 1:
                row["dividendYieldPct"] = dy * 100.0
            elif dy >= 1:
                row["dividendYieldPct"] = dy
            else:
                row["dividendYieldPct"] = None
        except Exception:
            row["dividendYieldPct"] = None
    else:
        row["dividendYieldPct"] = None

    return row


def load_backend_or_yf(tickers: List[str]) -> pd.DataFrame:
    """
    Try backend valuation summary first; fall back to Yahoo Finance snapshots.

    Expected backend shape (best-effort):
      {
        "valuations": [
          { "symbol": "...", "trailingPE": ..., "priceToBook": ..., ... },
          ...
        ]
      }
    """
    rows: List[Dict[str, Any]] = []

    # 1) Backend attempt
    backend_payload = call_backend("valuation/summary", params={"tickers": ",".join(tickers)})

    if isinstance(backend_payload, Dict) and backend_payload.get("valuations"):
        for item in backend_payload["valuations"]:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or item.get("ticker") or "").upper()
            if not sym:
                continue
            row = {"source": "backend", **item}
            row["symbol"] = sym
            rows.append(row)

    have = {r["symbol"] for r in rows if "symbol" in r}
    missing = [t for t in tickers if t not in have]

    # 2) Fallback: Yahoo Finance for missing tickers
    for t in missing:
        snap = fetch_yf_snapshot(t)
        snap["source"] = "yfinance"
        rows.append(snap)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize a friendly name column
    if "shortName" in df.columns:
        df["name"] = df["shortName"].fillna(df["symbol"])
    else:
        df["name"] = df["symbol"]

    # Stable ordering
    df = df.sort_values("symbol").reset_index(drop=True)
    return df


def compute_relative_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build peer-relative signals:

      - valuation_score:
            higher = cheaper vs peers
            blend of inverse P/E, inverse P/B, inverse EV/EBITDA
      - quality_score:
            higher = better quality (ROE + margins)

    Both normalized to 0–100 across the selected universe.
    """
    df = df.copy()

    def inv(col: str) -> pd.Series:
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace(0, np.nan)
        return 1.0 / s

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
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return pd.Series(50.0, index=s.index)
        return (s - lo) / (hi - lo) * 100.0

    df["valuation_score"] = to_0_100(cheap_signal)
    df["quality_score"] = to_0_100(qual_signal)

    return df


def flag_risks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight heuristic flags:
      - P/E > 40
      - Negative profit margins
      - P/B > 6 with weak ROE (< 8%)
    """
    df = df.copy()

    pe = pd.to_numeric(df.get("trailingPE"), errors="coerce")
    pb = pd.to_numeric(df.get("priceToBook"), errors="coerce")
    pm = pd.to_numeric(df.get("profitMargins"), errors="coerce")
    roe = pd.to_numeric(df.get("returnOnEquity"), errors="coerce")

    flags: List[str] = []
    for i in range(len(df)):
        msgs: List[str] = []

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
    # 1) Parse and normalize tickers
    raw_labels = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    tickers = [normalize_symbol(t) for t in raw_labels if normalize_symbol(t)]

    if not tickers:
        st.error("❌ Please enter at least one valid ticker or recognizable name.")
        st.stop()

    bench_sym = normalize_symbol(benchmark) if benchmark else ""
    if bench_sym and bench_sym not in tickers:
        tickers.append(bench_sym)

    # 2) Load valuations (backend + yfinance)
    with st.spinner("Fetching valuation data from backend / market data providers..."):
        df_val = load_backend_or_yf(tickers)

    if df_val.empty:
        st.error(
            "❌ Could not retrieve valuation data for the selected tickers. "
            "Check symbols, or try a simpler set like: AAPL, MSFT, NVDA, SPY."
        )
        st.stop()

    # 3) Compute relative scores (keeps existing logic)
    df_val = compute_relative_scores(df_val)

    # 4) Always compute risk flags (we can choose to hide/show them)
    df_val = flag_risks(df_val)

    # 5) Visible columns
    visible_cols: List[str] = [
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

    if show_source and "source" in df_val.columns:
        visible_cols.insert(1, "source")

    visible_cols = [c for c in visible_cols if c in df_val.columns]

    # 6) Snapshot table
    st.subheader("📋 Valuation Snapshot")
    st.dataframe(
        df_val[visible_cols].sort_values("symbol"),
        use_container_width=True,
        hide_index=True,
    )

    # 7) Peer-relative valuation score
    st.subheader("📈 Peer-relative Valuation Score (0–100)")
    if "valuation_score" in df_val.columns:
        df_score = df_val[["symbol", "valuation_score"]].dropna()
        if not df_score.empty:
            ordered = df_score.sort_values("valuation_score", ascending=False)
            fig_vs = px.bar(
                ordered,
                x="symbol",
                y="valuation_score",
                title="Higher = Cheaper vs Selected Peers (Blended Multiple View)",
                labels={"valuation_score": "Valuation Score (relative cheapness)"},
                text=ordered["valuation_score"].map(lambda x: f"{x:.0f}"),
            )
            fig_vs.update_traces(textposition="outside")
            fig_vs.update_layout(yaxis=dict(range=[0, 105]))
            st.plotly_chart(fig_vs, use_container_width=True)
        else:
            st.info("Not enough data to compute relative valuation scores.")
    else:
        st.info("Valuation scores not available for this selection.")

    # 8) Quality vs Valuation Map
    st.subheader("🧭 Quality vs Valuation Map")
    if {"valuation_score", "quality_score"}.issubset(df_val.columns):
        df_qv = df_val[["symbol", "valuation_score", "quality_score"]].dropna()
        if not df_qv.empty:
            fig_qv = px.scatter(
                df_qv,
                x="valuation_score",
                y="quality_score",
                text="symbol",
                title="Quality vs Valuation — top-right = higher quality & cheaper",
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
        st.info("Quality/valuation scores missing — check data coverage for these tickers.")

    # 9) Risk Flag Summary
    if show_risk_flags and "risk_flags" in df_val.columns:
        st.subheader("🚩 Attention & Outlier Flags")
        flagged = df_val[df_val["risk_flags"].astype(str).str.len() > 0]
        if not flagged.empty:
            for _, row in flagged.iterrows():
                st.write(f"**{row['symbol']}** — {row['risk_flags']}")
        else:
            st.write("No major heuristic flags based on current inputs.")

    st.caption(
        "Valuation metrics sourced from backend where available, "
        "falling back to Yahoo Finance. All signals are heuristic and for research only."
    )

else:
    st.info("👈 Configure your tickers and hit **Run Valuation Scan** to generate a peer set analysis.")
