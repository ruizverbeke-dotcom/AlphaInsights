"""
AlphaInsights Core Health Monitor
---------------------------------
Phase 5.2 — integrated with Safe Cloud Activation Layer (core.safe_connect).

This script checks:
- Core metadata availability
- Backend presence
- Supabase configuration and safe connection
- Overall system status (ready, degraded, or offline)
"""

from __future__ import annotations
import importlib.util
import json
import os
import sys
from datetime import datetime, UTC

# --- Ensure root path for imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.metadata import get_metadata
from supabase.config import get_supabase_config
from core.safe_connect import test_connection as safe_connect_status


def check_backend_available() -> bool:
    """Check if backend module is importable."""
    return importlib.util.find_spec("backend.main") is not None


def system_health() -> dict:
    """Aggregate all component health signals."""
    supabase_cfg = get_supabase_config()
    supabase_safe = safe_connect_status()

    status = {
        "timestamp": datetime.now(UTC).isoformat(),
        "core_metadata": bool(get_metadata()),
        "backend_available": check_backend_available(),
        "supabase_configured": supabase_cfg.get("configured", False),
        "supabase_connected": supabase_safe.get("connected", False),
        "phase": "5.2 (core-cloud integrated)",
        "status": "ok",
    }

    # Adjust overall status dynamically
    if not status["backend_available"]:
        status["status"] = "degraded"
    if not status["core_metadata"]:
        status["status"] = "critical"
    if not status["supabase_configured"]:
        status["status"] = "limited"

    # Merge diagnostic info
    status["cloud_detail"] = {
        "detected": supabase_safe.get("supabase_detected"),
        "credentials_present": supabase_safe.get("credentials_present"),
        "error": supabase_safe.get("error"),
    }

    return status


if __name__ == "__main__":
    print(json.dumps(system_health(), indent=2))
