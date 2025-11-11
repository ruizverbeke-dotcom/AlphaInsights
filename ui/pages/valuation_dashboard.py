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

    - Trims whitespace
    - Uppercases
    - Applies a static mapping for common aliases
    """
    raw = (label or "").strip()
    if not raw:
        return ""
    up = raw.upper()
    return _CANONICAL_MAP.get(up, up)


# ---------------------------------------------------------------------------
# Backend call helper (tolerant to fetch_backend signature & URL styles)
# ---------------------------------------------------------------------------

def call_backend(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Call the FastAPI backend in a future-proof way.

    Strategy:
    1) Try core.ui_helpers.fetch_backend with relative path.
    2) Fall back to direct HTTP GET using BACKEND_URL.
    3) Return dict on success, or None on any failure.

    This helper is shared by valuation/health, valuation/signals, valuation/summary, etc.
    """
    normalized_path = path.lstrip("/")

    # 1) Try via fetch_backend
    try:
        data = fetch_backend(normalized_path, params=params)
        if isinstance(data, dict):
            # If it's a typical FastAPI error payload and no data, treat as None.
            if any(k in data for k in ("error", "detail")) and not (
                data.get("valuations") or data.get("signals")
            ):
                return None
            return data
    except TypeError:
        # Older fetch_backend signatures fall through
        pass
    except Exception:
        # Any runtime error => treat as unavailable; UI will fall back
        return None

    # 2) Fallback: direct GET
    try:
        base = (BACKEND_URL or "http://127.0.0.1:8000").rstrip("/")
        url = f"{base}/{normalized_path}"
        resp = requests.get(url, params=params, timeout=6)
        if resp.ok:
            payload = resp.json()
            if isinstance(payload, dict):
                if any(k in payload for k in ("error", "detail")) and not (
                    payload.get("valuations") or payload.get("signals")
                ):
                    return None
                return payload
    except Exception:
        return None

    return None


# ---------------------------------------------------------------------------
# Backend Health Rendering
# ---------------------------------------------------------------------------

def render_backend_health() -> None:
    """
    Show valuation-aware backend health.

    Priority:
    - /valuation/health
    - /health
    - fallback text if unreachable
    """
    with st.expander("🩺 Backend Health", expanded=True):
        val_health = call_backend("valuation/health")
        if isinstance(val_health, dict) and val_health.get("status") == "ok":
            st.success("✅ Valuation module reachable")
            st.json(val_health)
            return

        core_health = call_backend("health")
        if isinstance(core_health, dict) and core_health.get("status") == "ok":
            st.warning(
                "⚠️ /valuation/health not available, but core backend is online. "
                "Valuation dashboard will use available endpoints or Yahoo Finance."
            )
            st.json(core_health)
        else:
            st.info(
                "Backend health endpoint not reachable. "
                "Valuation data will be fetched directly from Yahoo Finance where needed."
            )


# ---------------------------------------------------------------------------
# Sidebar Controls
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
        help="Optional; included for context.",
    ).strip()

    region_hint = st.selectbox(
        "Region / Market (hint only)",
        options=["(auto)", "US", "EU", "Global"],
        index=0,
        help="Reserved for future routing; no filter implemented yet.",
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

# Render health after controls
render_backend_health()

st.markdown("### 📊 Valuation Overview")
st.write(
    "This dashboard unifies **multiples**, **profitability**, and **payouts** "
    "to help you quickly see which names look stretched or attractive "
    "relative to peers in your selection."
)

# ---------------------------------------------------------------------------
# Shared valuation fields
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


# ---------------------------------------------------------------------------
# Yahoo Finance Snapshot Fallback
# ---------------------------------------------------------------------------

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

    # Normalize dividend yield into a percentage-style field
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

    # Tag local source explicitly
    row.setdefault("source", "yfinance")
    return row


# ---------------------------------------------------------------------------
# Scoring & Flags
# ---------------------------------------------------------------------------

def _to_0_100(s: pd.Series) -> pd.Series:
    """Normalize a numeric Series into 0–100 range; robust to edge cases."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(np.nan, index=s.index)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(50.0, index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def compute_relative_scores(df: pd.DataFrame, override_existing: bool = False) -> pd.DataFrame:
    """
    Build/refresh peer-relative valuation & quality scores.

    - valuation_score: higher = cheaper vs peers
      (blend of 1/PE, 1/PB, 1/EV/EBITDA)
    - quality_score: higher = better ROE & margins

    If override_existing is False:
    - only fill scores when missing or all-NaN.
    """
    df = df.copy()

    def inv(col: str, weight: float) -> pd.Series:
        if col not in df.columns:
            return pd.Series(0.0, index=df.index)
        s = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
        return weight * (1.0 / s)

    cheap_signal = (
        inv("trailingPE", 0.5) +
        inv("priceToBook", 0.3) +
        inv("enterpriseToEbitda", 0.2)
    )

    qual_signal = (
        0.6 * pd.to_numeric(df.get("returnOnEquity"), errors="coerce") +
        0.4 * pd.to_numeric(df.get("profitMargins"), errors="coerce")
    )

    need_val = override_existing or ("valuation_score" not in df.columns) or df["valuation_score"].isna().all()
    need_qual = override_existing or ("quality_score" not in df.columns) or df["quality_score"].isna().all()

    if need_val:
        df["valuation_score"] = _to_0_100(cheap_signal)

    if need_qual:
        df["quality_score"] = _to_0_100(qual_signal)

    return df


def compute_payout_score(df: pd.DataFrame, override_existing: bool = False) -> pd.DataFrame:
    """
    Derive payout_score:
    - Based on dividendYieldPct and payoutRatio sanity.
    - Penalize extreme or unsustainable payout ratios.

    If override_existing is False:
    - Only fill if payout_score missing or empty.
    """
    df = df.copy()

    if not override_existing and "payout_score" in df.columns and df["payout_score"].notna().any():
        return df

    dy = pd.to_numeric(df.get("dividendYieldPct"), errors="coerce")
    pr = pd.to_numeric(df.get("payoutRatio"), errors="coerce")

    # Base signal = dividend yield
    signal = dy.copy()

    # Penalize likely unsustainable or weird payouts
    if not signal.empty:
        bad = (pr > 90) | (pr < 0)  # using percentage-style payout ratios if present
        signal[bad] = signal[bad] * 0.3

    df["payout_score"] = _to_0_100(signal)
    return df


def flag_risks(
    df: pd.DataFrame,
    preserve_existing: bool = True,
) -> pd.DataFrame:
    """
    Lightweight heuristic flags:

    - Very high P/E
    - Negative margins
    - High P/B with weak ROE

    If preserve_existing is True:
    - Append to existing non-empty risk_flags instead of overwriting.
    """
    df = df.copy()

    existing = df.get("risk_flags")
    existing = existing.astype(str) if existing is not None else pd.Series([""] * len(df))

    pe = pd.to_numeric(df.get("trailingPE"), errors="coerce")
    pb = pd.to_numeric(df.get("priceToBook"), errors="coerce")
    pm = pd.to_numeric(df.get("profitMargins"), errors="coerce")
    roe = pd.to_numeric(df.get("returnOnEquity"), errors="coerce")

    new_flags: List[str] = []
    for i in range(len(df)):
        msgs: List[str] = []

        if not np.isnan(pe.iloc[i]) and pe.iloc[i] > 40:
            msgs.append("High PE")

        if not np.isnan(pm.iloc[i]) and pm.iloc[i] < 0:
            msgs.append("Negative margins")

        if (
            not np.isnan(pb.iloc[i])
            and pb.iloc[i] > 6
            and (np.isnan(roe.iloc[i]) or roe.iloc[i] < 0.08)
        ):
            msgs.append("High P/B with weak ROE")

        base_flag = existing.iloc[i].strip() if preserve_existing else ""
        combined = " | ".join([f for f in [base_flag, " | ".join(msgs)] if f])
        new_flags.append(combined)

    df["risk_flags"] = new_flags
    return df


# ---------------------------------------------------------------------------
# Valuation data loader (signals → summary → yfinance)
# ---------------------------------------------------------------------------

def load_valuation_data(tickers: List[str]) -> pd.DataFrame:
    """
    Unified loader with multi-level fallback:

    1. Try /valuation/signals for all tickers (Phase 7.3+).
    2. If missing/incomplete, try /valuation/summary for remaining.
    3. For anything still missing, use direct yfinance snapshots.

    Returns a DataFrame with:
    - symbol, name, currency, core valuation fields
    - possibly valuation_score, quality_score, payout_score from backend
    - source column per row
    """
    rows: List[Dict[str, Any]] = []

    joined = ",".join(tickers)

    # 1) Preferred: /valuation/signals
    signals_payload = call_backend("valuation/signals", params={"tickers": joined})
    if isinstance(signals_payload, dict) and isinstance(signals_payload.get("signals"), list):
        for item in signals_payload["signals"]:
            if not isinstance(item, dict):
                continue
            sym = str(item.get("symbol") or item.get("ticker") or "").upper()
            if not sym:
                continue
            row = dict(item)
            row["symbol"] = sym
            row.setdefault("source", "backend_signals")
            rows.append(row)

    have_symbols = {r["symbol"] for r in rows if "symbol" in r}
    missing_after_signals = [t for t in tickers if t not in have_symbols]

    # 2) Secondary: /valuation/summary for missing tickers
    if missing_after_signals:
        summary_payload = call_backend("valuation/summary", params={"tickers": ",".join(missing_after_signals)})
        vals = summary_payload.get("valuations") if isinstance(summary_payload, dict) else None
        if isinstance(vals, list):
            for item in vals:
                if not isinstance(item, dict):
                    continue
                sym = str(item.get("symbol") or item.get("ticker") or "").upper()
                if not sym or sym in have_symbols:
                    continue
                row = dict(item)
                row["symbol"] = sym
                row.setdefault("source", "backend_summary")
                rows.append(row)
                have_symbols.add(sym)

    # 3) Fallback: Yahoo Finance snapshots for any remaining
    missing_final = [t for t in tickers if t not in have_symbols]
    for t in missing_final:
        snap = fetch_yf_snapshot(t)
        if snap:
            rows.append(snap)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize display name safely
    if "shortName" in df.columns:
        existing_name = df["name"] if "name" in df.columns else None
        if existing_name is not None:
            df["name"] = df["shortName"].fillna(existing_name).fillna(df["symbol"])
        else:
            df["name"] = df["shortName"].fillna(df["symbol"])
    else:
        if "name" in df.columns:
            df["name"] = df["name"].fillna(df["symbol"])
        else:
            df["name"] = df["symbol"]

    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "unknown"

    # Stable order
    df = df.sort_values("symbol").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------------------------
if run_analysis:
    # 1) Parse & normalize user inputs
    raw_labels = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    tickers = [normalize_symbol(t) for t in raw_labels if normalize_symbol(t)]

    if not tickers:
        st.error("❌ Please enter at least one valid ticker or recognizable name.")
        st.stop()

    bench_sym = normalize_symbol(benchmark) if benchmark else ""
    if bench_sym and bench_sym not in tickers:
        tickers.append(bench_sym)

    # 2) Load valuation data with multi-level backend + yfinance fallback
    with st.spinner("Fetching valuation data from backend /valuation endpoints and market data providers..."):
        df_val = load_valuation_data(tickers)

    if df_val.empty:
        st.error(
            "❌ Could not retrieve valuation data for the selected tickers. "
            "Try a simpler set like: AAPL, MSFT, NVDA, SPY."
        )
        st.stop()

    # 3) Ensure scores exist:
    #    - Respect backend-provided scores if present.
    #    - Compute missing ones locally.
    df_val = compute_relative_scores(df_val, override_existing=False)
    df_val = compute_payout_score(df_val, override_existing=False)

    # 4) Ensure risk flags exist / merged
    df_val = flag_risks(df_val, preserve_existing=True)

    # 5) Build visible columns configuration
    visible_cols: List[str] = [
        "symbol",
        "name",
    ]

    if show_source and "source" in df_val.columns:
        visible_cols.append("source")

    visible_cols += [
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

    # Always include scores if present
    for col in ["valuation_score", "quality_score", "payout_score"]:
        if col in df_val.columns:
            visible_cols.append(col)

    if show_risk_flags and "risk_flags" in df_val.columns:
        visible_cols.append("risk_flags")

    # Deduplicate & keep only existing
    seen = set()
    final_cols: List[str] = []
    for c in visible_cols:
        if c in df_val.columns and c not in seen:
            seen.add(c)
            final_cols.append(c)

    # 6) Snapshot table
    st.subheader("📋 Valuation Snapshot")
    st.dataframe(
        df_val[final_cols].sort_values("symbol"),
        use_container_width=True,
        hide_index=True,
    )

    # 7) Peer-relative Valuation Score
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
        st.info("Valuation scores not available for this selection (no suitable multiples).")

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
        flagged = df_val[df_val["risk_flags"].astype(str).str.strip() != ""]
        if not flagged.empty:
            for _, row in flagged.iterrows():
                st.write(f"**{row['symbol']}** — {row['risk_flags']}")
        else:
            st.write("No major heuristic flags based on current inputs.")

    st.caption(
        "Valuation metrics sourced from backend /valuation/signals when available, "
        "falling back to /valuation/summary and Yahoo Finance snapshots. "
        "Scores and flags are heuristic and for research only."
    )

else:
    st.info("👈 Configure your tickers and hit **Run Valuation Scan** to generate a peer set analysis.")
