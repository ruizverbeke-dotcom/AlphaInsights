"""
AlphaInsights Optimization Module
---------------------------------
Implements portfolio optimization utilities for AlphaInsights.

Core components:
- CVaR (Expected Shortfall) optimizer using Rockafellar–Uryasev formulation.
- Sharpe Ratio optimizer (Phase 6.0) using SLSQP.
- Minimum-variance baseline for comparisons.

Design Principles
-----------------
- No network calls (pure math only).
- Deterministic, testable, agent-ready.
- Enforces 1-D safety on inputs to avoid (N, 1) vs (N,) bugs.
- Returns JSON-serializable structures suitable for backend/API/agents.

Stable Output Schemas
---------------------
CVaR optimizer:
{
  "weights": pd.Series | dict,   # dict if constraints["as_dict"] = True
  "es": float,                   # Expected Shortfall (on returns, negative tail)
  "var": float,                  # VaR (on returns, negative tail)
  "ann_vol": float,              # Annualized volatility
  "solver": str,                 # e.g. "ECOS" or "SCS"
  "success": bool,
  "summary": str,                # Human-readable one-liner
}

Sharpe optimizer (Phase 6.0):
{
  "weights": { "TICKER": float, ... },
  "sharpe": float,               # Annualized Sharpe ratio
  "ann_vol": float,              # Annualized volatility
  "ann_return": float,           # Annualized expected return
  "solver": "Sharpe_SLSQP",
  "success": bool,
  "summary": str,                # Human-readable one-liner
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.optimize import minimize

try:
    import cvxpy as cp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "cvxpy is required for optimization. Please add it to your environment "
        "(see requirements.txt)."
    ) from e


__all__ = [
    "optimize_cvar",
    "empirical_var_es",
    "sample_min_variance_weights",
    "optimize_sharpe",
]


# --------------------------------------------------------------------------- #
# 1-D / 2-D Safety Helpers
# --------------------------------------------------------------------------- #
def _ensure_2d_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a clean 2-D DataFrame with 1-D numeric columns.

    Enforces the global 1-D safety rule column-wise:
    - squeeze("columns") for (N,1) shapes
    - np.ravel(...) to guarantee contiguous 1-D arrays
    - drop rows with any NaNs

    This keeps downstream math and optimizers stable and predictable.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame with assets as columns.")

    cols: List[pd.Series] = []
    for col in returns.columns:
        s = returns[col]
        s = s.squeeze("columns") if isinstance(s, pd.DataFrame) else s
        flat = np.ravel(s)
        idx = s.index[: len(flat)]
        cols.append(pd.Series(flat, index=idx, name=col))

    df = pd.concat(cols, axis=1).dropna(how="any")
    if df.empty:
        raise ValueError("returns becomes empty after cleanup/dropna.")
    return df


# --------------------------------------------------------------------------- #
# Metrics Helpers
# --------------------------------------------------------------------------- #
def empirical_var_es(loss: pd.Series, alpha: float) -> Tuple[float, float]:
    """
    Empirical VaR and ES (CVaR) at level `alpha` for a *loss* series (L_t = -r_t).

    Parameters
    ----------
    loss : pd.Series
        Loss series L_t. For portfolio returns r_t, define L_t = -r_t.
    alpha : float
        Confidence level in (0, 1). Typical range: 0.90–0.99.

    Returns
    -------
    (var_alpha, es_alpha) : Tuple[float, float]
        VaR_alpha and ES_alpha computed empirically on the loss distribution.
    """
    if not (0.80 <= alpha < 1.0):
        raise ValueError("alpha should be in [0.80, 1.0).")

    s = loss.squeeze("columns") if isinstance(loss, pd.DataFrame) else loss
    s = pd.Series(np.ravel(s), index=loss.index[: len(np.ravel(s))])

    var_alpha = float(np.quantile(s, alpha))
    tail = s[s >= var_alpha]
    es_alpha = float(tail.mean())
    return var_alpha, es_alpha


def _portfolio_stats(
    returns: pd.DataFrame,
    weights: np.ndarray,
    periods_per_year: int = 252,
    alpha: float = 0.95,
) -> Dict[str, float]:
    """
    Compute portfolio tail metrics and annualized volatility for given weights.

    Notes
    -----
    - ES/VaR are reported on returns (negative values indicate left tail),
      obtained by computing on losses and then negating the sign.
    """
    w = np.asarray(weights).reshape(-1, 1)
    port = np.ravel(returns.values @ w)

    vol = float(np.std(port, ddof=1)) if len(port) > 1 else 0.0
    ann_vol = float(np.sqrt(periods_per_year) * vol)

    loss = pd.Series(-port, index=returns.index[: len(port)])
    var_l, es_l = empirical_var_es(loss, alpha)
    return {
        "var": -float(var_l),
        "es": -float(es_l),
        "ann_vol": ann_vol,
    }


# --------------------------------------------------------------------------- #
# Constraints for CVaR Optimizer
# --------------------------------------------------------------------------- #
@dataclass
class CVaRConstraints:
    """
    Constraint container for the CVaR optimizer.

    Attributes
    ----------
    long_only : bool
        If True, enforce w >= 0.
    min_weight : Optional[float]
        Global lower bound for each weight.
    max_weight : Optional[float]
        Global upper bound for each weight.
    caps : Optional[Dict[str, float]]
        Per-asset upper caps, e.g., {"AAPL": 0.30}.
    excludes : Optional[Iterable[str]]
        Assets to force to zero weight.
    budget : float
        Sum of weights (usually 1.0).
    """
    long_only: bool = True
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    caps: Optional[Dict[str, float]] = None
    excludes: Optional[Iterable[str]] = None
    budget: float = 1.0


# --------------------------------------------------------------------------- #
# CVaR Optimizer (Rockafellar–Uryasev)
# --------------------------------------------------------------------------- #
def optimize_cvar(
    returns: pd.DataFrame,
    alpha: float = 0.95,
    constraints: Optional[Dict] = None,
    periods_per_year: int = 252,
) -> Dict[str, Union[pd.Series, dict, float, str, bool]]:
    """
    Minimize Expected Shortfall (CVaR) at confidence level `alpha`.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows=time, cols=assets).
    alpha : float
        Tail confidence level (0 < alpha < 1).
    constraints : dict, optional
        {long_only, min_weight, max_weight, caps, excludes, budget, as_dict}
        - If "as_dict" is True, returns weights as a plain dict (JSON-safe).
    periods_per_year : int
        For annualizing volatility.

    Returns
    -------
    dict
        {
          "weights": pd.Series | dict,
          "es": float,
          "var": float,
          "ann_vol": float,
          "solver": str,
          "success": bool,
          "summary": str,
        }
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1).")

    R = _ensure_2d_frame(returns)
    T, N = R.shape
    names = list(R.columns)

    # Parse constraints
    c = CVaRConstraints()
    as_dict = False
    if constraints:
        for k, v in constraints.items():
            if k == "as_dict":
                as_dict = bool(v)
            elif hasattr(c, k):
                setattr(c, k, v)

    # Decision variables
    w = cp.Variable(N)
    z = cp.Variable()          # VaR level on losses
    u = cp.Variable(T)         # tail excess variables

    cons: List = [u >= 0, u >= -R.values @ w - z, cp.sum(w) == c.budget]

    # Long-only / bounds
    if c.long_only:
        cons += [w >= 0]
    if c.min_weight is not None:
        cons += [w >= float(c.min_weight)]
    if c.max_weight is not None:
        cons += [w <= float(c.max_weight)]

    # Per-asset caps
    if c.caps:
        for i, nm in enumerate(names):
            cap = c.caps.get(nm)
            if cap is not None:
                cons.append(w[i] <= float(cap))

    # Exclusions
    excl_idx = set()
    if c.excludes:
        excl = set(map(str.upper, map(str, c.excludes)))
        for i, nm in enumerate(names):
            if str(nm).upper() in excl:
                cons.append(w[i] == 0.0)
                excl_idx.add(i)

    # Objective: minimize z + (1/((1-alpha)T)) * sum(u)
    tau = 1.0 / ((1.0 - alpha) * T)
    problem = cp.Problem(cp.Minimize(z + tau * cp.sum(u)), cons)

    # Solve with fallback solvers
    solver_used, success = None, False
    for solver in (cp.ECOS, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
            solver_used = solver.name()
        except Exception:
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            success = True
            break

    # Fallback: equal-weight across non-excluded assets
    def _fallback_result() -> Dict[str, Union[pd.Series, dict, float, str, bool]]:
        mask = np.ones(N, dtype=bool)
        for i in excl_idx:
            mask[i] = False
        k = int(mask.sum())
        w_fb = np.zeros(N, dtype=float)
        if k:
            w_fb[mask] = c.budget / k

        stats = _portfolio_stats(R, w_fb, periods_per_year, alpha)
        summary = (
            f"CVaR optimization (α={alpha:.3f}) fell back to equal-weight; "
            f"ES={stats['es']:.4f}, VaR={stats['var']:.4f}, σₐ={stats['ann_vol']:.4f}."
        )
        weights_series = pd.Series(w_fb, index=names, name="weight")
        weights_out: Union[pd.Series, dict] = (
            weights_series.to_dict() if as_dict else weights_series
        )

        return {
            "weights": weights_out,
            "es": stats["es"],
            "var": stats["var"],
            "ann_vol": stats["ann_vol"],
            "solver": solver_used or "N/A",
            "success": False,
            "summary": summary,
        }

    if (w.value is None) or (not success):
        return _fallback_result()

    # Optimal solution
    w_opt = np.asarray(w.value).ravel()

    # Normalize to budget (numerical guard)
    s = w_opt.sum()
    if s:
        w_opt = w_opt / s * c.budget

    stats = _portfolio_stats(R, w_opt, periods_per_year, alpha)
    summary = (
        f"CVaR optimization (α={alpha:.3f}) solved via {solver_used or 'unknown'}; "
        f"ES={stats['es']:.4f}, VaR={stats['var']:.4f}, σₐ={stats['ann_vol']:.4f}."
    )
    weights_series = pd.Series(w_opt, index=names, name="weight")
    weights_out: Union[pd.Series, dict] = (
        weights_series.to_dict() if as_dict else weights_series
    )

    return {
        "weights": weights_out,
        "es": stats["es"],
        "var": stats["var"],
        "ann_vol": stats["ann_vol"],
        "solver": solver_used or "N/A",
        "success": True,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Sharpe Ratio Optimizer (Phase 6.0)
# --------------------------------------------------------------------------- #
def optimize_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> Dict[str, Union[dict, float, str, bool]]:
    """
    Maximize the portfolio's annualized Sharpe Ratio using SLSQP.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (rows=time, cols=assets).
    risk_free_rate : float
        Annualized risk-free rate (e.g. 0.02 for 2%). Assumed constant.
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly, etc.).

    Returns
    -------
    dict
        {
          "weights": { "TICKER": float, ... },
          "sharpe": float,
          "ann_vol": float,
          "ann_return": float,
          "solver": "Sharpe_SLSQP",
          "success": bool,
          "summary": str,
        }

    Notes
    -----
    - Long-only, fully-invested (weights in [0,1], sum to 1).
    - Pure math: no network calls, deterministic for given inputs.
    - Output is JSON-serializable and agent-friendly.
    """
    R = _ensure_2d_frame(returns)
    names = list(R.columns)
    n = len(names)
    if n == 0:
        raise ValueError("No assets provided for Sharpe optimization.")

    # Per-period mean returns & covariance
    mu = R.mean().values          # shape (n,)
    cov = R.cov().values          # shape (n, n)

    if np.any(~np.isfinite(mu)) or np.any(~np.isfinite(cov)):
        raise ValueError("Non-finite values detected in returns; cannot optimize Sharpe.")

    # Initial guess: equal-weight portfolio
    w0 = np.ones(n) / n

    def portfolio_ann_stats(w: np.ndarray) -> Tuple[float, float]:
        """
        Compute annualized (return, volatility) for weights w.
        """
        w = np.asarray(w)
        ret_period = float(np.dot(w, mu))
        vol_period = float(np.sqrt(np.dot(w.T, np.dot(cov, w)))) if n > 0 else 0.0

        ann_return = ret_period * periods_per_year
        ann_vol = vol_period * np.sqrt(periods_per_year) if vol_period > 0 else 0.0
        return ann_return, ann_vol

    def neg_sharpe(w: np.ndarray) -> float:
        """
        Objective for SLSQP: negative annualized Sharpe ratio.
        """
        ann_ret, ann_vol = portfolio_ann_stats(w)
        if ann_vol <= 0:
            # Penalize degenerate solutions to steer optimizer away.
            return 1e6
        return -((ann_ret - risk_free_rate) / ann_vol)

    # Constraints: sum(weights) = 1
    cons = ({
        "type": "eq",
        "fun": lambda w: float(np.sum(w) - 1.0),
    },)

    # Bounds: long-only, each weight in [0, 1]
    bounds = tuple((0.0, 1.0) for _ in range(n))

    try:
        res = minimize(
            neg_sharpe,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"disp": False, "maxiter": 500},
        )
        success = bool(res.success)
        w_opt = np.asarray(res.x).ravel() if res.x is not None else None
    except Exception:
        success = False
        w_opt = None

    solver_name = "Sharpe_SLSQP"

    # Fallback: equal-weight portfolio if optimization fails
    if (w_opt is None) or (not success) or (not np.all(np.isfinite(w_opt))):
        w_fb = w0
        ann_ret, ann_vol = portfolio_ann_stats(w_fb)
        sharpe = (ann_ret - risk_free_rate) / (ann_vol + 1e-12) if ann_vol > 0 else 0.0

        weights = {
            names[i]: float(np.round(w_fb[i], 6))
            for i in range(n)
        }
        summary = (
            "Sharpe optimization failed; using equal-weight portfolio "
            f"(SR={sharpe:.2f}, μₐ={ann_ret:.4f}, σₐ={ann_vol:.4f})."
        )

        return {
            "weights": weights,
            "sharpe": float(np.round(sharpe, 4)),
            "ann_vol": float(np.round(ann_vol, 4)),
            "ann_return": float(np.round(ann_ret, 4)),
            "solver": solver_name,
            "success": False,
            "summary": summary,
        }

    # Normalize and sanitize optimized weights
    w_opt = np.clip(w_opt, 0.0, 1.0)
    s = float(w_opt.sum())
    if s <= 0:
        w_opt = w0
    else:
        w_opt = w_opt / s

    ann_ret, ann_vol = portfolio_ann_stats(w_opt)
    sharpe = (ann_ret - risk_free_rate) / (ann_vol + 1e-12) if ann_vol > 0 else 0.0

    weights = {
        names[i]: float(np.round(w_opt[i], 6))
        for i in range(n)
    }

    summary = (
        f"Sharpe optimization succeeded using SLSQP (SR={sharpe:.2f}, "
        f"μₐ={ann_ret:.4f}, σₐ={ann_vol:.4f})."
    )

    return {
        "weights": weights,
        "sharpe": float(np.round(sharpe, 4)),
        "ann_vol": float(np.round(ann_vol, 4)),
        "ann_return": float(np.round(ann_ret, 4)),
        "solver": solver_name,
        "success": True,
        "summary": summary,
    }


# --------------------------------------------------------------------------- #
# Minimum-Variance Baseline
# --------------------------------------------------------------------------- #
def sample_min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Compute minimum-variance weights (long-only, fully invested) as a baseline.

    This is useful for comparison in dashboards and tests to benchmark
    more advanced optimizers like CVaR and Sharpe.
    """
    R = _ensure_2d_frame(returns)
    cov = np.cov(R.values, rowvar=False)
    cov = np.asarray(cov)
    N = cov.shape[0]

    w = cp.Variable(N)
    cons = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), cons)
    prob.solve(solver=cp.ECOS, verbose=False)

    if w.value is None:
        raise RuntimeError("Min-variance optimization failed.")

    arr = np.asarray(w.value).ravel()
    arr = arr / arr.sum()
    return pd.Series(arr, index=R.columns, name="min_var_weight")


# --------------------------------------------------------------------------- #
# Inline Quick-Check (Local Only)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Simple smoke tests to validate shapes and outputs.
    np.random.seed(42)
    sample_returns = pd.DataFrame(
        np.random.randn(252, 3) / 100,
        columns=["AAPL", "MSFT", "GOOG"],
    )

    print("=== Sharpe Optimizer Smoke Test ===")
    sharpe_res = optimize_sharpe(sample_returns, risk_free_rate=0.02)
    print(sharpe_res)

    print("\n=== CVaR Optimizer Smoke Test ===")
    cvar_res = optimize_cvar(sample_returns, alpha=0.95, constraints={"as_dict": True})
    print(cvar_res)
