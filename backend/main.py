"""
AlphaInsights Backend API
-------------------------
FastAPI backend service exposing optimization and health endpoints.

Phase 5.7 — Solver Integration:
- Implements real CVaR optimization using scipy.optimize.minimize.
- Returns solver name in API responses.
"""

import sys
import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import minimize
import yfinance as yf

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from core.health import system_health

app = FastAPI(
    title="AlphaInsights Backend API",
    version="1.0",
    description="FastAPI synchronization layer for CVaR and future analytics endpoints.",
)

# --------------------------------------------------------------------------- #
# Data Models
# --------------------------------------------------------------------------- #
class CVaRRequest(BaseModel):
    tickers: List[str]
    start: str
    end: str
    alpha: float

class CVaRResponse(BaseModel):
    weights: dict
    es: float
    var: float
    ann_vol: float
    solver: str
    success: bool
    summary: str

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def _load_data(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted close prices and compute daily returns."""
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="any")
    if data.empty:
        raise ValueError("No valid data available for given tickers.")
    return np.log(data / data.shift(1)).dropna()

def _compute_cvar(returns: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Compute Conditional Value-at-Risk (CVaR) for portfolio returns."""
    portfolio_returns = returns @ weights
    cutoff = np.quantile(portfolio_returns, 1 - alpha)
    cvar = portfolio_returns[portfolio_returns <= cutoff].mean()
    return -cvar  # loss perspective

def _objective(weights: np.ndarray, returns: np.ndarray, alpha: float) -> float:
    """Objective function to minimize (CVaR)."""
    return _compute_cvar(returns, weights, alpha)

def _optimize_cvar(returns: np.ndarray, alpha: float) -> tuple[dict, float, float, float, str, bool]:
    """Run real CVaR optimization with SLSQP solver."""
    n = returns.shape[1]
    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    try:
        res = minimize(
            _objective,
            x0,
            args=(returns, alpha),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"disp": False, "maxiter": 500},
        )

        if not res.success:
            raise RuntimeError(res.message)

        weights = res.x
        portfolio = returns @ weights
        es = _compute_cvar(returns, weights, alpha)
        var = -np.quantile(portfolio, 1 - alpha)
        ann_vol = np.std(portfolio) * np.sqrt(252)

        return (
            {t: float(w) for t, w in zip(returns.columns, weights)},
            float(es),
            float(var),
            float(ann_vol),
            "CVaR_SLSQP",
            True,
        )
    except Exception as e:
        # Fallback — equal weight
        n = returns.shape[1]
        weights = np.ones(n) / n
        portfolio = returns @ weights
        es = _compute_cvar(returns, weights, alpha)
        var = -np.quantile(portfolio, 1 - alpha)
        ann_vol = np.std(portfolio) * np.sqrt(252)
        return (
            {t: float(1 / n) for t in returns.columns},
            float(es),
            float(var),
            float(ann_vol),
            "Fallback_EqualWeight",
            False,
        )

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "AlphaInsights Backend API is running.",
        "version": "1.0",
    }

@app.get("/health")
async def health():
    """Expose system health diagnostics."""
    return system_health()

@app.post("/optimize/cvar", response_model=CVaRResponse)
async def optimize_cvar(request: CVaRRequest):
    """Perform real CVaR optimization via SLSQP solver."""
    try:
        returns = _load_data(request.tickers, request.start, request.end)
        weights, es, var, ann_vol, solver, success = _optimize_cvar(returns, request.alpha)

        summary = (
            f"CVaR optimization (α={request.alpha:.3f}) "
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

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
