# backend/main.py
"""
AlphaInsights Backend API
-------------------------
FastAPI layer exposing analytics (CVaR optimizer) via HTTP.

Design:
- Directly callable by UI or external clients.
- Intelligent: auto-fetches data if only tickers/start/end provided.
- Returns JSON-safe output for easy integration with agents or web services.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analytics.optimization import optimize_cvar

app = FastAPI(
    title="AlphaInsights Backend API",
    description="FastAPI synchronization layer for CVaR and future analytics endpoints.",
    version="1.0",
)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CVaRRequest(BaseModel):
    tickers: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    alpha: float
    returns: Optional[List[List[float]]] = None
    columns: Optional[List[str]] = None


class CVaRResponse(BaseModel):
    weights: Dict[str, float]
    es: float
    var: float
    ann_vol: float
    solver: str
    success: bool
    summary: str


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def _fetch_returns(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download prices and compute log returns (clean + aligned)."""
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="any")
    if data.empty:
        raise ValueError("No valid price data available for the requested tickers and dates.")

    returns = np.log(data / data.shift(1)).dropna()
    return returns


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "AlphaInsights Backend API is running.",
        "version": "1.0",
    }


@app.post("/optimize/cvar", response_model=CVaRResponse)
def optimize_cvar_endpoint(payload: CVaRRequest):
    """
    Run CVaR optimization.
    Accepts either:
    - tickers + start + end  → auto-fetch data
    - returns + columns      → direct data mode
    """
    try:
        # 1️⃣ Determine input mode
        if payload.returns and payload.columns:
            df = pd.DataFrame(payload.returns, columns=payload.columns)
        elif payload.tickers and payload.start and payload.end:
            df = _fetch_returns(payload.tickers, payload.start, payload.end)
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either (tickers, start, end) or (returns, columns).",
            )

        # 2️⃣ Run optimization
        result = optimize_cvar(df, alpha=payload.alpha, constraints={"as_dict": True})

        return CVaRResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
