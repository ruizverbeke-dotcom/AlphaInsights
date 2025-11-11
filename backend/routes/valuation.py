"""
Valuation Router — AlphaInsights
================================

Provides valuation metrics via yfinance and (future) backend data fusion.

Endpoints:
----------
- GET /valuation/health     → module liveness and metadata
- GET /valuation/summary    → compact valuation metrics for given tickers

Design:
-------
• Modular FastAPI router (Phase 7.2+)
• Safe, no-exception yfinance integration
• Supports flexible symbol resolution ("gold" → "GC=F", etc.)
• JSON-clean output (no NaNs)
• Future-ready for Supabase / analytics fusion
"""

from __future__ import annotations

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any
import yfinance as yf
import numpy as np
import pandas as pd

# Optional resolver (safe import)
try:
    from core.symbol_resolver import resolve_tickers
except Exception:
    resolve_tickers = None

# --------------------------------------------------------------------------- #
# Router Init
# --------------------------------------------------------------------------- #

router = APIRouter(tags=["valuation"])

# --------------------------------------------------------------------------- #
# Valuation Metadata
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
        "version": "1.1",
    }

# --------------------------------------------------------------------------- #
# Core Data Logic
# --------------------------------------------------------------------------- #

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


@router.get("/summary")
async def valuation_summary(
    tickers: str = Query(..., description="Comma-separated ticker list (e.g., AAPL,MSFT,NVDA,SPY)")
) -> Dict[str, Any]:
    """
    Return a unified valuation summary for the requested tickers.

    Workflow:
    ---------
    1. Try to resolve human-friendly inputs to valid tickers if resolver available.
    2. Fetch compact valuation data via yfinance.
    3. Return normalized, JSON-clean dataframe with consistent keys.

    Returns
    -------
    {
        "valuations": [
            {"symbol": "AAPL", "trailingPE": ..., "priceToBook": ..., ...},
            ...
        ],
        "meta": {"count": 4, "source": "yfinance", "resolver_used": True/False}
    }
    """
    # 1. Parse tickers
    tickers_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers_list:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    resolved = tickers_list
    resolver_used = False

    # 2. Optional resolution
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

    # 3. Fetch data
    results: List[Dict[str, Any]] = []
    for t in resolved:
        snap = safe_fetch(t)
        if snap:
            results.append(snap)

    if not results:
        raise HTTPException(status_code=404, detail="No valuation data retrieved.")

    # 4. Format to DataFrame
    df = pd.DataFrame(results)
    df = df.fillna(np.nan).replace({np.nan: None})

    # 5. Build response
    payload = {
        "valuations": df.to_dict(orient="records"),
        "meta": {
            "count": len(df),
            "source": "yfinance",
            "resolver_used": resolver_used,
        },
    }

    return payload
