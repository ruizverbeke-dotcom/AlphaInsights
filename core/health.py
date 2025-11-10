"""
core/health.py — Phase 6.5+
-----------------------------------
System health diagnostics for the AlphaInsights backend.

Purpose
-------
- Used by FastAPI `/health` endpoint and Streamlit dashboards.
- Validates Supabase connectivity.
- Reports backend uptime, version, CPU/memory usage, and latency placeholders.
- Returns JSON-safe dict ready for serialization.

Future extensions (Phase 7+)
----------------------------
✅ Add live Supabase latency probes.
✅ Add API uptime metrics (via Prometheus or psutil).
✅ Add GPU metrics if using analytics acceleration.
"""

from __future__ import annotations

import os
import time
import platform
import psutil
from typing import Dict, Any

from supabase_client.config import get_supabase_client


# Cache the process start time for uptime calculation
START_TIME = time.time()


def system_health() -> Dict[str, Any]:
    """
    Return structured backend health diagnostics.

    Returns
    -------
    dict
        JSON-safe health report compatible with frontend HealthSchema.
    """
    status = "ok"
    message = "Backend operational."
    supabase_connected = False
    supabase_url = None

    # --- Supabase connectivity test ---
    try:
        sb = get_supabase_client()
        supabase_url = sb.supabase_url
        # Ping: list a small number of entries to confirm access
        sb.table("optimizer_logs").select("id").limit(1).execute()
        supabase_connected = True
    except Exception as e:
        status = "degraded"
        message = f"Supabase check failed: {e.__class__.__name__}"
        supabase_connected = False

    # --- System metrics ---
    try:
        cpu_load = psutil.cpu_percent(interval=0.2)
        memory_usage = round(psutil.virtual_memory().used / (1024 * 1024), 2)
    except Exception:
        cpu_load = None
        memory_usage = None

    # --- Construct report ---
    uptime_sec = round(time.time() - START_TIME, 2)
    version = os.getenv("BACKEND_VERSION", "1.2")

    health_report = {
        "status": status,
        "message": message,
        "version": version,
        "supabase_connected": supabase_connected,
        "supabase_url": supabase_url,
        "cpu_load": cpu_load,
        "memory_usage": memory_usage,
        "uptime_sec": uptime_sec,
        "system": platform.system(),
        "release": platform.release(),
    }

    return health_report
