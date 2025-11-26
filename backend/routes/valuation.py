"""
Valuation Router — AlphaInsights
================================

Provides valuation metrics via yfinance and (future) backend data fusion.

Endpoints:
----------
- GET /valuation/health     → module liveness and metadata
- GET /valuation/summary    → compact valuation metrics for given tickers
- GET /valuation/signals    → computed valuation, quality & payout scores (0–100)

Design:
-------
• Modular FastAPI router (Phase 7.2–7.3)
• Safe, no-exception yfinance integration
• Supports flexible symbol resolution ("gold" → "GC=F", etc.)
• JSON-clean output (no NaNs)
• Adds derived valuation & quality scoring (Phase 7.3)
• Phase 7.4: Supabase-backed caching for /valuation/signals
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Query, HTTPException
import yfinance as yf
import numpy as np
import pandas as pd

# Optional resolver import
try:
    from core.symbol_resolver import resolve_tickers
except Exception:
    resolve_tickers = None

# Supabase cache helpers (Phase 7.4)
try:
    from supabase_client.cache import (
        get_cached_valuation,
        put_cached_valuation,
        is_cache_fresh,
    )
except Exception:
    # If cache module or Supabase are not available, caching is simply disabled.
    get_cached_valuation = None  # type: ignore
    put_cached_valuation = None  # type: ignore
    is_cache_fresh = None  # type: ignore

# --------------------------------------------------------------------------- #
# Router Init
# --------------------------------------------------------------------------- #

router = APIRouter(tags=["valuation"])

# --------------------------------------------------------------------------- #
# Constants & Metadata
# --------------------------------------------------------------------------- #

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

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_cache_key_from_tickers(tickers_list: List[str]) -> str:
    """
    Build a deterministic cache key from the given tickers.

    Phase 7.4 scope:
    - Only tickers are included.
    - Symbols are uppercased and sorted.
    - Future extensions (benchmark, region, versioning) can be added here
      when they actually affect backend behavior.
    """
    normalized = [t.strip().upper() for t in tickers_list if t.strip()]
    unique_sorted = sorted(set(normalized))
    tickers_str = ",".join(unique_sorted)
    return f"valuation:v1:tickers={tickers_str}"


def safe_fetch(ticker: str) -> Dict[str, Any]:
    """
    Fetch valuation metrics safely from yfinance.
    Returns a dict with cleaned numeric and textual data.
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
        row[f] = info.get(f)

    # Normalize dividend yield %
    dy = row.get("dividendYield")
    if isinstance(dy, (int, float)):
        if 0 < dy < 1:
            row["dividendYieldPct"] = dy * 100
        elif dy >= 1:
            row["dividendYieldPct"] = dy
        else:
            row["dividendYieldPct"] = None
    else:
        row["dividendYieldPct"] = None

    return row


# --------------------------------------------------------------------------- #
# Health Endpoint
# --------------------------------------------------------------------------- #


@router.get("/health")
async def valuation_health() -> Dict[str, Any]:
    """
    Lightweight health/status check for the valuation module.
    """
    return {
        "status": "ok",
        "module": "valuation",
        "message": "Valuation router active and ready.",
        "source": "yfinance",
        "version": "1.2",
    }


# --------------------------------------------------------------------------- #
# /valuation/summary
# --------------------------------------------------------------------------- #


@router.get("/summary")
async def valuation_summary(
    tickers: str = Query(
        ...,
        description="Comma-separated ticker list (e.g., AAPL,MSFT,NVDA,SPY)",
    )
) -> Dict[str, Any]:
    """
    Return a unified valuation summary for the requested tickers.

    Steps:
    -------
    1. Optionally resolve human inputs → valid tickers.
    2. Fetch valuation data via yfinance.
    3. Return normalized, JSON-clean dataframe with consistent keys.
    """
    tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    resolved = tickers_list
    resolver_used = False

    # Optional resolver
    if resolve_tickers:
        try:
            resolved, messages = resolve_tickers(tickers_list)
            resolver_used = True
            for msg in messages:
                print(f"[valuation] resolver: {msg}")
        except Exception as e:
            print(f"[valuation] Resolver failed: {e}")
            resolved = tickers_list

    if not resolved:
        raise HTTPException(status_code=400, detail="No valid tickers after resolution.")

    results: List[Dict[str, Any]] = [safe_fetch(t) for t in resolved]
    if not results:
        raise HTTPException(status_code=404, detail="No valuation data retrieved.")

    df = pd.DataFrame(results)
    df = df.fillna(np.nan).replace({np.nan: None})

    payload = {
        "valuations": df.to_dict(orient="records"),
        "meta": {
            "count": len(df),
            "source": "yfinance",
            "resolver_used": resolver_used,
        },
    }
    return payload


# --------------------------------------------------------------------------- #
# /valuation/signals — Phase 7.3 + Phase 7.4 caching
# --------------------------------------------------------------------------- #


@router.get("/signals")
async def valuation_signals(
    tickers: str = Query(
        ...,
        description="Comma-separated ticker list for signal computation",
    )
) -> Dict[str, Any]:
    """
    Return peer-relative valuation, quality, and payout scores (0–100 normalized).

    Adds:
    -----
    - valuation_score  → relative cheapness (PE, PB, EV/EBITDA)
    - quality_score    → profitability strength (ROE, margins)
    - payout_score     → dividend yield vs peers
    - risk_flags       → heuristic attention markers

    Phase 7.4:
    ----------
    - Supabase-backed caching layer around the full response payload.
    - Cache is best-effort and never allowed to break endpoint behavior.
    """
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    # --------------------------------------------------------------------- #
    # 1) Try cache (if cache helpers are available)
    # --------------------------------------------------------------------- #
    cache_payload: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None

    if get_cached_valuation is not None and is_cache_fresh is not None:
        try:
            cache_key = _build_cache_key_from_tickers(tickers_list)
            cache_row = get_cached_valuation(cache_key)
            if cache_row and is_cache_fresh(cache_row):
                maybe_payload = cache_row.get("payload")
                if isinstance(maybe_payload, dict):
                    cache_payload = maybe_payload
        except Exception as e:
            # Any cache issue is logged and ignored.
            print(f"[valuation] Cache lookup failed: {e}")
            cache_payload = None

    if cache_payload is not None:
        # Ensure meta exists and set cache_hit flag.
        meta = cache_payload.setdefault("meta", {})
        meta["cache_hit"] = True
        return cache_payload

    # --------------------------------------------------------------------- #
    # 2) Cache miss or cache disabled → compute live (Phase 7.3 logic)
    # --------------------------------------------------------------------- #
    results = [safe_fetch(t) for t in tickers_list]
    df = pd.DataFrame(results)

    if df.empty:
        raise HTTPException(status_code=404, detail="No valuation data retrieved.")

    # Convert numeric fields
    numeric_cols = [
        "trailingPE",
        "priceToBook",
        "enterpriseToEbitda",
        "profitMargins",
        "returnOnEquity",
        "dividendYieldPct",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")

    # --- Derived Signals ---
    def inv(col: str) -> pd.Series:
        return 1.0 / df[col].replace(0, np.nan)

    cheap_signal = (
        0.5 * inv("trailingPE")
        + 0.3 * inv("priceToBook")
        + 0.2 * inv("enterpriseToEbitda")
    )
    qual_signal = (
        0.6 * df["returnOnEquity"].fillna(0)
        + 0.4 * df["profitMargins"].fillna(0)
    )
    payout_signal = df["dividendYieldPct"].fillna(0)

    def to_0_100(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        if s.notna().sum() < 2:
            return pd.Series(np.nan, index=s.index)
        lo, hi = s.min(), s.max()
        if hi <= lo:
            return pd.Series(50.0, index=s.index)
        return ((s - lo) / (hi - lo)) * 100

    df["valuation_score"] = to_0_100(cheap_signal)
    df["quality_score"] = to_0_100(qual_signal)
    df["payout_score"] = to_0_100(payout_signal)

    # --- Risk Flags ---
    flags: List[str] = []
    for _, r in df.iterrows():
        f = []
        if r.get("trailingPE", 0) > 40:
            f.append("High PE")
        if r.get("profitMargins", 0) < 0:
            f.append("Negative margins")
        if r.get("priceToBook", 0) > 6 and (r.get("returnOnEquity", 0) < 0.08):
            f.append("High PB + weak ROE")
        flags.append(" | ".join(f) if f else "")
    df["risk_flags"] = flags

    df = df.fillna(np.nan).replace({np.nan: None})

    response_payload: Dict[str, Any] = {
        "signals": df.to_dict(orient="records"),
        "meta": {
            "count": len(df),
            "computed_at": pd.Timestamp.utcnow().isoformat(),
            "note": "Peer-relative valuation and quality signals (Phase 7.3).",
            "cache_hit": False,
        },
    }

    # --------------------------------------------------------------------- #
    # 3) Store in cache (best-effort, never fatal)
    # --------------------------------------------------------------------- #
    if put_cached_valuation is not None and cache_key is not None:
        try:
            put_cached_valuation(cache_key, response_payload)
        except Exception as e:
            print(f"[valuation] Error while caching payload for key {cache_key}: {e}")

    return response_payload

# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
