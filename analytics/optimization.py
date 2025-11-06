# analytics/optimization.py
"""
AlphaInsights Optimization Module
---------------------------------
Implements the CVaR (Expected Shortfall) optimizer using the
Rockafellar–Uryasev formulation.

Design Principles
- No network calls (pure math).
- Typed, deterministic, 1-D safety enforced on inputs.
- Agent-ready: clean I/O schema + human-readable summary string.
- JSON-safe option for weights via constraints["as_dict"].

Returns schema (stable for agents / API)
----------------------------------------
{
  "weights": pd.Series | dict,   # dict if constraints["as_dict"] = True
  "es": float,                   # Expected Shortfall on returns (negative tail)
  "var": float,                  # VaR on returns (negative tail)
  "ann_vol": float,              # Annualized volatility
  "solver": str,                 # "ECOS" or "SCS" (when available)
  "success": bool,               # True if solver status optimal/near-optimal
  "summary": str,                # Human-readable one-liner for Explainability Agent
}

How it integrates
-----------------
- Called by Optimizer Agent / UI (ui/pages/optimizer_dashboard.py) with a prepared
  returns DataFrame (rows=time, cols=assets).
- Upstream: Data Agent / UI loads prices (e.g., yfinance in UI), computes returns.
- Downstream: Explainability Agent uses the "summary" and metrics to narrate results.
- Automation: This function is deterministic and stateless → safe for schedulers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
]


# --------------------------------------------------------------------------- #
# 1-D Safety Helper
# --------------------------------------------------------------------------- #
def _ensure_2d_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure proper 2-D DataFrame with 1-D numeric columns.

    Enforces the global 1-D safety rule column-wise:
      - squeeze("columns") for (N,1) shapes
      - np.ravel to guarantee contiguous 1-D arrays
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
        Confidence level in (0, 1). Typical 0.90–0.99.

    Returns
    -------
    (var_alpha, es_alpha) : Tuple[float, float]
        VaR_alpha and ES_alpha computed empirically.
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
    Compute per-period portfolio stats and tail metrics on returns.

    Notes
    -----
    - ES/VaR are reported on *returns* (negative values indicate left tail),
      obtained by computing on losses and negating the sign.
    """
    w = np.asarray(weights).reshape(-1, 1)
    port = np.ravel(returns.values @ w)
    vol = float(np.std(port, ddof=1)) if len(port) > 1 else 0.0
    ann_vol = float(np.sqrt(periods_per_year) * vol)

    loss = pd.Series(-port, index=returns.index[: len(port)])
    var_l, es_l = empirical_var_es(loss, alpha)
    return {"var": -float(var_l), "es": -float(es_l), "ann_vol": ann_vol}


# --------------------------------------------------------------------------- #
# Constraints
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
        Tail confidence level (0<alpha<1).
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

    # Fallback: equal weight across non-excluded assets
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
        weights_out: Union[pd.Series, dict] = weights_series.to_dict() if as_dict else weights_series
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
    weights_out: Union[pd.Series, dict] = weights_series.to_dict() if as_dict else weights_series

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
# Optional: simple min-variance baseline (used in comparison charts/tests)
# --------------------------------------------------------------------------- #
def sample_min_variance_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Compute minimum-variance weights (long-only, fully invested) as a simple baseline.
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
