"""
analytics/log_insights.py
-------------------------

Pure helper for summarizing optimizer logs from Supabase.

This module must remain network-agnostic and pure:
- Input: list[dict] (rows from optimizer_logs)
- Output: dict[str, Any] (JSON-serializable insights)

Used by backend.main:/logs/insights.
"""

from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

def compute_optimizer_insights(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics from optimizer_logs.

    Parameters
    ----------
    logs : list of dict
        Rows as returned by supabase_client.helpers.fetch_recent() or /logs/query.

    Returns
    -------
    dict : JSON-serializable summary with keys:
        - total_runs
        - runs_by_endpoint
        - sharpe_stats
        - cvar_stats
    """
    if not logs:
        return {
            "total_runs": 0,
            "runs_by_endpoint": {},
            "sharpe_stats": {},
            "cvar_stats": {},
        }

    total_runs = len(logs)

    # Count by endpoint
    runs_by_endpoint: Dict[str, int] = {}
    sharpe_values = []
    cvar_values = []

    for row in logs:
        endpoint = str(row.get("endpoint", "unknown")).lower()
        runs_by_endpoint[endpoint] = runs_by_endpoint.get(endpoint, 0) + 1

        result = row.get("result") or {}
        if endpoint == "sharpe":
            val = result.get("sharpe")
            if isinstance(val, (int, float)):
                sharpe_values.append(val)
        elif endpoint == "cvar":
            val = result.get("es")
            if isinstance(val, (int, float)):
                cvar_values.append(val)

    def _summary(vals: list[float]) -> Dict[str, float]:
        if not vals:
            return {}
        arr = np.array(vals, dtype=float)
        return {
            "count": len(arr),
            "mean": float(np.nanmean(arr)),
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
        }

    insights = {
        "total_runs": total_runs,
        "runs_by_endpoint": runs_by_endpoint,
        "sharpe_stats": _summary(sharpe_values),
        "cvar_stats": _summary(cvar_values),
    }

    return insights
