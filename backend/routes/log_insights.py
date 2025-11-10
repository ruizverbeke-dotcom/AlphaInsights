"""
Phase 6.9 — Log Insights Endpoint
---------------------------------
Provides summary statistics and trend analytics from optimizer_logs table in Supabase.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from supabase_client.helpers import fetch_recent
from statistics import mean

router = APIRouter(prefix="/logs", tags=["logs"])

@router.get("/insights")
async def get_log_insights(window: str = "30d"):
    """
    Aggregate insights from Supabase optimizer_logs.

    Parameters
    ----------
    window : str
        Time window for aggregation, e.g. "7d", "30d", "90d", or "all".
    """
    try:
        now = datetime.utcnow()
        days = {"7d": 7, "30d": 30, "90d": 90}.get(window, None)

        data = fetch_recent("optimizer_logs", limit=200)
        if not data:
            return {"status": "ok", "count": 0, "message": "No logs found."}

        if days:
            cutoff = now - timedelta(days=days)
            data = [row for row in data if datetime.fromisoformat(row["created_at"].replace("Z", "")) >= cutoff]

        sharpe_values, vol_values, ret_values = [], [], []
        success_count = 0

        for row in data:
            result = row.get("result") or {}
            if result.get("success"):
                success_count += 1
            if "sharpe" in result:
                sharpe_values.append(result["sharpe"])
            if "ann_vol" in result:
                vol_values.append(result["ann_vol"])
            if "ann_return" in result:
                ret_values.append(result["ann_return"])

        insights = {
            "window": window,
            "count": len(data),
            "success_rate": round(success_count / len(data), 3) if data else 0,
            "avg_sharpe": round(mean(sharpe_values), 4) if sharpe_values else None,
            "avg_ann_vol": round(mean(vol_values), 4) if vol_values else None,
            "avg_ann_return": round(mean(ret_values), 4) if ret_values else None,
        }

        return {"status": "ok", "insights": insights}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
