"""
AlphaInsights Backend API
=========================

FastAPI backend service exposing optimization, analytics, logging, and health endpoints.

Phase History & Design Intent
-----------------------------
• Phase 6.0–6.2
    - Core CVaR + Sharpe optimizers.
    - Shared symbol resolver for human-friendly inputs:
        e.g. "apple" -> "AAPL", "gold" -> "GC=F", "cac 40" -> "^FCHI".
    - Robust market data pipeline:
        * Prefer 'Adj Close', fallback 'Close'.
        * Cleans and aligns data across tickers.
        * Drops illiquid / broken series with too little data.
    - Sharpe optimization delegated to `analytics.optimization.optimize_sharpe`.
    - Deterministic math, typed models, stable JSON schemas (agent-ready).

• Phase 6.3–6.5
    - Optional Supabase logging for optimizer calls (non-fatal, best-effort).
    - Centralized helpers in `supabase_client`.
    - RLS-compatible insert patterns.

• Phase 6.6–6.8
    - `/logs/recent` and `/logs/query` endpoints:
        * Filter by endpoint, date range, pagination.
        * Designed for UI dashboards (e.g. Optimizer History).

• Phase 6.9+
    - Modular router hooks (`backend.routes.*`) for log insights & analytics.
    - Extensible status/telemetry endpoints.
    - Safe fallbacks if optional components are missing.

This file is intentionally verbose and future-proof. Prefer extending it over
patching ad-hoc logic elsewhere.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# Path setup: ensure project root on sys.path
# --------------------------------------------------------------------------- #

# Allows imports like `core.*`, `analytics.*`, `supabase_client.*` when running via uvicorn
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.health import system_health
from core.symbol_resolver import resolve_tickers
from analytics.optimization import optimize_sharpe

# --------------------------------------------------------------------------- #
# Optional Supabase integration (Phase 6.3+)
# --------------------------------------------------------------------------- #

SUPABASE_ENABLED = False

try:
    # helpers.py is our single front-door to Supabase usage
    from supabase_client.helpers import (
        insert_record,
        fetch_recent,
        get_supabase_client,
        test_connection,
    )

    SUPABASE_ENABLED = True
except Exception as e:  # noqa: BLE001 - we want to log ANY import issue
    insert_record = None          # type: ignore[assignment]
    fetch_recent = None           # type: ignore[assignment]
    get_supabase_client = None    # type: ignore[assignment]
    test_connection = None        # type: ignore[assignment]
    print(f"[Supabase] ⚠️ helpers unavailable — Supabase features disabled. Reason: {e}")

# Read env (for diagnostics only; helpers will also validate)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if SUPABASE_ENABLED and (not SUPABASE_URL or not SUPABASE_ANON_KEY):
    print("[Supabase] ❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY; disabling Supabase logging.")
    SUPABASE_ENABLED = False

# Connection sanity check (non-fatal)
if SUPABASE_ENABLED and test_connection is not None:
    try:
        test_connection(debug=True)
    except Exception as e:  # noqa: BLE001
        print(f"[Supabase] ⚠️ Connection test failed; disabling Supabase logging. Reason: {e}")
        SUPABASE_ENABLED = False

# --------------------------------------------------------------------------- #
# Optional modular routers (Phase 6.9+)
# --------------------------------------------------------------------------- #

ROUTERS_AVAILABLE = False
try:
    # Example: advanced analytics/insights router (if present)
    from backend.routes.log_insights import router as log_insights_router  # type: ignore

    ROUTERS_AVAILABLE = True
except Exception as e:  # noqa: BLE001
    log_insights_router = None  # type: ignore[assignment]
    print(f"[Backend] ℹ️ log_insights router not loaded (ok for now): {e}")

# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="AlphaInsights Backend API",
    version="1.5",
    description=(
        "Backend for portfolio optimization, analytics, logging, and health.\n"
        "- Shared symbol resolver + robust data pipeline.\n"
        "- Sharpe & CVaR optimizers.\n"
        "- Optional Supabase logging & query endpoints.\n"
        "- Modular routers for future analytics."
    ),
)

if ROUTERS_AVAILABLE and log_insights_router is not None:
    app.include_router(log_insights_router, prefix="/logs/insights")
    print("[Backend] ✅ Registered /logs/insights router.")

# --------------------------------------------------------------------------- #
# Pydantic Models (Agent & UI Friendly)
# --------------------------------------------------------------------------- #

class OptimizeRequest(BaseModel):
    """
    Common request model for optimizer endpoints.
    """
    tickers: List[str]
    start: str              # YYYY-MM-DD
    end: str                # YYYY-MM-DD
    alpha: Optional[float] = 0.95   # Used by CVaR
    rf: Optional[float] = 0.0       # Used by Sharpe


class CVaRResponse(BaseModel):
    """
    Response schema for CVaR optimizer (stable contract).
    """
    weights: Dict[str, float]
    es: float               # Expected Shortfall (CVaR)
    var: float              # Value-at-Risk
    ann_vol: float          # Annualized volatility
    solver: str
    success: bool
    summary: str


class SharpeResponse(BaseModel):
    """
    Response schema for Sharpe optimizer (stable contract).
    """
    weights: Dict[str, float]
    sharpe: float
    ann_vol: float
    ann_return: float
    solver: str
    success: bool
    summary: str


# --------------------------------------------------------------------------- #
# Data Loading Helper (Shared for All Optimizers)
# --------------------------------------------------------------------------- #

def _fetch_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Resolve tickers, download prices via yfinance, and compute daily log returns.

    Behavior:
    --------
    1. Resolve user inputs via core.symbol_resolver.
    2. Download OHLCV with multi-asset support.
    3. Prefer 'Adj Close', fallback 'Close'.
    4. Drop unusable/empty series.
    5. Ensure minimum history per asset.
    6. Align calendar; ffill/bfill small gaps.
    7. Compute log returns and drop invalid rows.

    Raises HTTPException(400) for user-facing errors.
    """
    # 1) Symbol resolution
    resolved, messages = resolve_tickers(tickers)
    for msg in messages:
        print(f"[resolver] {msg}")

    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=f"No valid symbols could be resolved from inputs: {tickers}",
        )

    # 2) Historical data download
    try:
        data = yf.download(
            resolved,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
            threads=True,
            timeout=60,
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Data download failed from yfinance: {e}",
        )

    if data is None or data.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data returned for tickers: {resolved}",
        )

    # 3) Price extraction
    if "Adj Close" in data:
        prices = data["Adj Close"]
    elif "Close" in data:
        prices = data["Close"]
    else:
        raise HTTPException(
            status_code=400,
            detail="No 'Adj Close' or 'Close' columns available from yfinance.",
        )

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Normalize MultiIndex columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [str(c[-1]) for c in prices.columns]

    # 4) Drop fully empty columns
    prices = prices.dropna(how="all", axis=1)
    if prices.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No valid price data after initial cleanup for: {resolved}",
        )

    # 5) Liquidity / history filter
    valid_cols = [c for c in prices.columns if prices[c].notna().sum() >= 30]
    if not valid_cols:
        raise HTTPException(
            status_code=400,
            detail=f"No sufficiently liquid/supported tickers after cleaning: {list(prices.columns)}",
        )
    prices = prices[valid_cols]

    # 6) Calendar alignment
    prices = prices.sort_index()
    prices = prices.ffill().bfill()
    prices = prices.dropna(how="all", axis=0)
    if prices.empty or prices.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="Price alignment resulted in an empty price matrix.",
        )

    # 7) Log returns
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all", axis=0)
    if returns.empty or returns.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid returns could be computed after cleaning.",
        )

    return returns


# --------------------------------------------------------------------------- #
# CVaR Optimizer (Local SLSQP Implementation)
# --------------------------------------------------------------------------- #

def _compute_cvar(returns_array: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Compute portfolio CVaR (Expected Shortfall) for given weights.
    """
    portfolio_returns = returns_array @ weights
    cutoff = np.quantile(portfolio_returns, 1 - alpha)
    cvar = portfolio_returns[portfolio_returns <= cutoff].mean()
    return -float(cvar)


def _objective(weights: np.ndarray, returns_array: np.ndarray, alpha: float) -> float:
    """
    Objective function for SLSQP: minimize CVaR.
    """
    return _compute_cvar(returns_array, weights, alpha)


def _optimize_cvar(
    returns: pd.DataFrame,
    alpha: float,
) -> Tuple[Dict[str, float], float, float, float, str, bool]:
    """
    Simple CVaR optimizer using SLSQP (long-only, fully invested).

    Returns
    -------
    weights_dict, es, var, ann_vol, solver_name, success
    """
    R = returns.values
    n = R.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    try:
        res = minimize(
            _objective,
            x0,
            args=(R, alpha),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"disp": False, "maxiter": 500},
        )

        if not res.success:
            raise RuntimeError(res.message)

        weights = np.asarray(res.x, dtype=float)
        weights = weights / weights.sum()

        portfolio = R @ weights
        es = _compute_cvar(R, weights, alpha)
        var = -float(np.quantile(portfolio, 1 - alpha))
        ann_vol = float(np.std(portfolio) * np.sqrt(252.0))

        return (
            {t: float(w) for t, w in zip(returns.columns, weights)},
            float(es),
            float(var),
            ann_vol,
            "CVaR_SLSQP",
            True,
        )

    except Exception:  # noqa: BLE001
        # Fallback: equal-weight across available assets
        weights = np.ones(n, dtype=float) / float(n)
        portfolio = R @ weights
        es = _compute_cvar(R, weights, alpha)
        var = -float(np.quantile(portfolio, 1 - alpha))
        ann_vol = float(np.std(portfolio) * np.sqrt(252.0))

        return (
            {t: float(1.0 / n) for t in returns.columns},
            float(es),
            float(var),
            ann_vol,
            "Fallback_EqualWeight",
            False,
        )


# --------------------------------------------------------------------------- #
# Internal: Supabase logging helper (centralized)
# --------------------------------------------------------------------------- #

def _log_to_supabase(payload: Dict[str, Any]) -> None:
    """
    Centralized, best-effort Supabase logging.

    - No exceptions bubble up to the API layer.
    - Only runs if SUPABASE_ENABLED and insert_record is available.
    - Safe to call from any endpoint.
    """
    if not SUPABASE_ENABLED or insert_record is None:
        return

    try:
        table = "optimizer_logs"
        print(f"[Supabase] → Inserting into '{table}' …")
        print(f"[Supabase] Payload keys: {list(payload.keys())}")
        res = insert_record(table, payload, debug=True)  # type: ignore[arg-type]
        if res:
            print(f"[Supabase] ✅ Insert success → {res}")
        else:
            print("[Supabase] ⚠️ Insert returned no data (check RLS / schema).")
    except Exception as e:  # noqa: BLE001
        print(f"[Supabase] ⚠️ Insert failed: {e}")


# --------------------------------------------------------------------------- #
# Core Routes
# --------------------------------------------------------------------------- #

@app.get("/")
async def root():
    """
    Basic liveness probe.
    """
    return {
        "status": "ok",
        "message": "AlphaInsights Backend is live.",
        "version": app.version,
        "supabase_enabled": SUPABASE_ENABLED,
    }


@app.get("/health")
async def health():
    """
    System health endpoint.

    Delegates to core.health.system_health which:
    - Checks Python/runtime info
    - Optionally checks Supabase connectivity (via supabase_client)
    - Returns a stable, machine-readable payload
    """
    return system_health()


@app.post("/optimize/cvar", response_model=CVaRResponse)
async def optimize_cvar_endpoint(request: OptimizeRequest):
    """
    CVaR optimizer endpoint (Phase 6.x).

    - Uses shared data loader.
    - Runs SLSQP-based CVaR optimizer.
    - Best-effort logging to Supabase (if enabled).
    """
    try:
        alpha = float(request.alpha if request.alpha is not None else 0.95)
        returns = _fetch_data(request.tickers, request.start, request.end)

        weights, es, var, ann_vol, solver, success = _optimize_cvar(returns, alpha)

        summary = (
            f"CVaR optimization (α={alpha:.3f}) "
            f"{'succeeded' if success else 'fell back'} using {solver}; "
            f"ES={es:.4f}, VaR={var:.4f}, σₐ={ann_vol:.4f}."
        )

        result: Dict[str, Any] = {
            "weights": weights,
            "es": es,
            "var": var,
            "ann_vol": ann_vol,
            "solver": solver,
            "success": success,
            "summary": summary,
        }

        _log_to_supabase(
            {
                "endpoint": "cvar",
                "tickers": request.tickers,
                "start": request.start,
                "end": request.end,
                "alpha": alpha,
                "rf": None,
                "result": result,
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/optimize/sharpe", response_model=SharpeResponse)
async def optimize_sharpe_endpoint(request: OptimizeRequest):
    """
    Sharpe optimizer endpoint (Phase 6.x).

    - Uses shared data loader.
    - Delegates to analytics.optimization.optimize_sharpe.
    - Best-effort Supabase logging.
    """
    try:
        returns = _fetch_data(request.tickers, request.start, request.end)
        if returns.empty:
            raise ValueError("No valid return data for requested tickers/date range.")

        rf = float(request.rf if request.rf is not None else 0.0)

        result = optimize_sharpe(returns, risk_free_rate=rf)

        _log_to_supabase(
            {
                "endpoint": "sharpe",
                "tickers": request.tickers,
                "start": request.start,
                "end": request.end,
                "alpha": None,
                "rf": rf,
                "result": result,
            }
        )

        return result

    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Sharpe optimization failed: {e}",
        )


# --------------------------------------------------------------------------- #
# Supabase Log Retrieval Endpoints (for UI & Agents)
# --------------------------------------------------------------------------- #

@app.get("/logs/recent")
async def get_recent_logs(limit: int = 10):
    """
    Return the most recent optimizer logs.

    - Backed by Supabase `optimizer_logs` table.
    - Used by simple history views / debugging tools.
    """
    if not SUPABASE_ENABLED or fetch_recent is None:
        raise HTTPException(status_code=503, detail="Supabase logging is disabled.")

    try:
        logs = fetch_recent("optimizer_logs", limit=limit, debug=True)  # type: ignore[arg-type]
        return logs
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {e}")


@app.get("/logs/query")
async def query_logs(
    endpoint: Optional[str] = Query(
        None, description="Filter by endpoint name, e.g. 'sharpe' or 'cvar'"
    ),
    start: Optional[str] = Query(
        None, description="Filter created_at >= this ISO timestamp/date"
    ),
    end: Optional[str] = Query(
        None, description="Filter created_at <= this ISO timestamp/date"
    ),
    limit: int = Query(
        20, ge=1, le=200, description="Number of rows to return (1–200)"
    ),
    offset: int = Query(
        0, ge=0, description="Offset for pagination (0 = first page)"
    ),
):
    """
    Advanced log query endpoint (Phase 6.6+).

    - Supports endpoint filter, date range, pagination.
    - Returns:
        * count       : rows in this page
        * total       : total matching rows (exact count)
        * last_updated: most recent created_at in this page
        * results     : list of log records
    """
    if not SUPABASE_ENABLED or get_supabase_client is None:
        raise HTTPException(status_code=503, detail="Supabase logging is disabled.")

    try:
        sb = get_supabase_client()
        query = sb.table("optimizer_logs").select("*")

        if endpoint:
            query = query.eq("endpoint", endpoint)
        if start:
            query = query.gte("created_at", start)
        if end:
            query = query.lte("created_at", end)

        # Range is inclusive, so use offset..offset+limit-1
        query = query.order("created_at", desc=True).range(
            offset, offset + limit - 1
        )
        res = query.execute()
        data = res.data or []

        # Exact count for pagination
        total_res = (
            sb.table("optimizer_logs")
            .select("id", count="exact")
            .execute()
        )
        total_count = getattr(total_res, "count", None) or len(data)

        last_updated = data[0]["created_at"] if data else None

        return {
            "count": len(data),
            "total": int(total_count),
            "last_updated": last_updated,
            "results": data,
        }

    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to query logs: {e}")


# --------------------------------------------------------------------------- #
# Status / Future Analytics Placeholder (Phase 7.0+)
# --------------------------------------------------------------------------- #

@app.get("/status/summary")
async def status_summary():
    """
    High-level status summary for dashboards & agents.

    Future extension:
    - Aggregate optimizer usage stats.
    - Expose rolling success rates for optimizers.
    - Surface Supabase latency / error rates.
    """
    return {
        "backend_version": app.version,
        "supabase_enabled": bool(SUPABASE_ENABLED),
        "supabase_url_configured": bool(SUPABASE_URL),
        "routers": {
            "log_insights_registered": bool(ROUTERS_AVAILABLE),
        },
    }


# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
