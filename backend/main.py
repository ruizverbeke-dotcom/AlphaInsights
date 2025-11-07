"""
AlphaInsights Backend API
-------------------------
FastAPI backend service exposing optimization and health endpoints.

Phase 6.0–6.1 Summary
---------------------
- Provides CVaR and Sharpe optimization endpoints.
- Uses a shared symbol resolver to normalize human-friendly inputs:
    e.g. "apple" -> "AAPL", "gold" -> "GC=F", "cac 40" -> "^FCHI".
- Centralizes data loading with robust fallbacks:
    • Prefer 'Adj Close', fallback to 'Close'.
    • Cleans and aligns data across assets (no empty returns matrices).
- Delegates Sharpe optimization core logic to analytics.optimization.optimize_sharpe.
"""

from __future__ import annotations

import os
import sys
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import minimize

# --------------------------------------------------------------------------- #
# Setup: import project modules
# --------------------------------------------------------------------------- #

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from core.health import system_health
from core.symbol_resolver import resolve_tickers
from analytics.optimization import optimize_sharpe  # Sharpe optimizer core (Phase 6.0)

# --------------------------------------------------------------------------- #
# FastAPI App
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="AlphaInsights Backend API",
    version="1.1",
    description=(
        "FastAPI synchronization layer for CVaR, Sharpe, and future analytics endpoints. "
        "Uses shared symbol resolution and robust data loading."
    ),
)

# --------------------------------------------------------------------------- #
# Pydantic Models (Agent-Ready)
# --------------------------------------------------------------------------- #

class OptimizeRequest(BaseModel):
    """
    Common request model for optimizer endpoints.

    Attributes
    ----------
    tickers : List[str]
        Human or symbol inputs (e.g. 'apple', 'AAPL', 'gold', 'cac 40').
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    alpha : float, optional
        Tail/confidence parameter (used by CVaR; ignored by Sharpe).
    rf : float, optional
        Annualized risk-free rate (used by Sharpe).
    """
    tickers: List[str]
    start: str
    end: str
    alpha: Optional[float] = 0.95
    rf: Optional[float] = 0.0


class CVaRResponse(BaseModel):
    """
    Response schema for CVaR optimizer.

    Matches the stable agent-ready contract.
    """
    weights: Dict[str, float]
    es: float
    var: float
    ann_vol: float
    solver: str
    success: bool
    summary: str


class SharpeResponse(BaseModel):
    """
    Response schema for Sharpe optimizer.

    Matches the stable agent-ready contract from analytics.optimize_sharpe.
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
    Resolve tickers, download prices, and compute daily log returns.

    Robust behavior (Phase 6.1.B)
    -----------------------------
    - Uses core.symbol_resolver.resolve_tickers for human inputs.
    - Prefers 'Adj Close', falls back to 'Close'.
    - Drops assets with too little data.
    - Forward/backward fills small gaps to align calendars.
    - Guarantees a non-empty 2-D returns matrix or raises a clean 400 error.

    Returns
    -------
    pd.DataFrame
        DataFrame of log returns (rows = dates, cols = tickers).

    Raises
    ------
    HTTPException(400)
        If resolution or data loading fails in a user-relevant way.
    """
    # 1) Resolve free-form labels to concrete symbols
    resolved, messages = resolve_tickers(tickers)
    for msg in messages:
        # Logged to backend console for traceability
        print(f"[resolver] {msg}")

    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=f"No valid symbols could be resolved from inputs: {tickers}",
        )

    # 2) Download OHLCV data for resolved symbols
    try:
        data = yf.download(
            resolved,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Data download failed from yfinance: {e}",
        )

    if data is None or data.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No data returned for tickers: {resolved}",
        )

    # 3) Extract price matrix (Adj Close preferred)
    if "Adj Close" in data:
        prices = data["Adj Close"]
    elif "Close" in data:
        prices = data["Close"]
    else:
        raise HTTPException(
            status_code=400,
            detail="No 'Adj Close' or 'Close' data available from yfinance.",
        )

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    # Normalize column labels (in case of MultiIndex from yf)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [str(c[-1]) for c in prices.columns]

    # 4) Drop totally empty columns
    prices = prices.dropna(how="all", axis=1)
    if prices.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No valid price data for tickers after initial cleanup: {resolved}",
        )

    # 5) Keep only assets with sufficient observations (e.g. >= 30)
    valid_cols = [c for c in prices.columns if prices[c].notna().sum() >= 30]
    if not valid_cols:
        raise HTTPException(
            status_code=400,
            detail=f"No sufficiently liquid/supported tickers after cleaning: {list(prices.columns)}",
        )
    prices = prices[valid_cols]

    # 6) Align calendars: fill internal gaps, keep overall structure
    prices = prices.sort_index()
    prices = prices.ffill().bfill()

    # Ensure something remains
    prices = prices.dropna(how="all", axis=0)
    if prices.empty or prices.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="Price alignment resulted in empty series.",
        )

    # 7) Compute log returns, drop only rows where all returns are NaN
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.dropna(how="all", axis=0)

    if returns.empty or returns.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid returns could be computed after alignment and cleaning.",
        )

    return returns


# --------------------------------------------------------------------------- #
# Legacy CVaR Optimizer (Local SLSQP Implementation)
# --------------------------------------------------------------------------- #

def _compute_cvar(returns_array: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """
    Compute portfolio CVaR (Expected Shortfall) given returns matrix and weights.
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
) -> tuple[Dict[str, float], float, float, float, str, bool]:
    """
    Simple CVaR optimizer using SLSQP (long-only, fully invested).

    Returns
    -------
    (weights_dict, es, var, ann_vol, solver_name, success)
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

    except Exception:
        # Fallback: equal-weight portfolio over available assets
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
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
async def root():
    """
    Basic liveness probe for backend.
    """
    return {
        "status": "ok",
        "message": "AlphaInsights Backend API is running.",
        "version": app.version,
    }


@app.get("/health")
async def health():
    """
    Expose system health diagnostics from core.health.system_health.
    """
    return system_health()


@app.post("/optimize/cvar", response_model=CVaRResponse)
async def optimize_cvar_endpoint(request: OptimizeRequest):
    """
    CVaR optimizer endpoint.

    - Resolves tickers via shared resolver.
    - Loads and cleans return series.
    - Runs local SLSQP-based CVaR optimizer.
    """
    try:
        alpha = request.alpha if request.alpha is not None else 0.95
        returns = _fetch_data(request.tickers, request.start, request.end)

        weights, es, var, ann_vol, solver, success = _optimize_cvar(returns, alpha)

        summary = (
            f"CVaR optimization (α={alpha:.3f}) "
            f"{'succeeded' if success else 'fell back'} using {solver}; "
            f"ES={es:.4f}, VaR={var:.4f}, σₐ={ann_vol:.4f}."
        )

        return {
            "weights": weights,
            "es": es,
            "var": var,
            "ann_vol": ann_vol,
            "solver": solver,
            "success": success,
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/optimize/sharpe", response_model=SharpeResponse)
async def optimize_sharpe_endpoint(request: OptimizeRequest):
    """
    Sharpe optimizer endpoint (Phase 6.0 / 6.1).

    - Resolves tickers via shared resolver.
    - Loads and cleans log returns.
    - Delegates to analytics.optimization.optimize_sharpe.
    - Returns stable, agent-friendly schema.
    """
    try:
        returns = _fetch_data(request.tickers, request.start, request.end)
        if returns.empty:
            raise ValueError("No valid return data for requested tickers/date range.")

        rf = float(request.rf) if request.rf is not None else 0.0

        # analytics.optimize_sharpe is backend-compatible:
        # it accepts risk_free_rate and `risk_free` via kwargs.
        result = optimize_sharpe(returns, risk_free=rf)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Sharpe optimization failed: {e}",
        )


# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
