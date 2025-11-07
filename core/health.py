"""
core/health.py
--------------
System-wide health diagnostics for AlphaInsights.

Checks:
- Core metadata integrity
- Backend availability (FastAPI app import check)
- Supabase configuration (Phase 5)
"""

from datetime import datetime, timezone
from typing import Dict
import importlib.util
import os
import sys
import json


# --- Ensure project root is importable ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def check_core_metadata() -> bool:
    """Verify that core metadata exists and contains required fields."""
    try:
        from core.metadata import get_metadata
        meta = get_metadata()
        required = {"project", "version", "phase", "maintainer"}
        return all(k in meta for k in required)
    except Exception:
        return False


def check_backend_available() -> bool:
    """Check whether backend.main FastAPI app is importable."""
    try:
        spec = importlib.util.find_spec("backend.main")
        return spec is not None
    except Exception:
        return False


def check_supabase_configured() -> bool:
    """Check if Supabase configuration exists and has valid attributes."""
    try:
        from supabase.config import check_supabase_health
        health = check_supabase_health()
        return bool(health.get("configured"))
    except Exception:
        return False


def system_health() -> Dict[str, object]:
    """Return overall system health snapshot."""
    core_ok = check_core_metadata()
    backend_ok = check_backend_available()
    supabase_ok = check_supabase_configured()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "core_metadata": core_ok,
        "backend_available": backend_ok,
        "supabase_configured": supabase_ok,
        "status": "ok" if core_ok and backend_ok else "warning",
    }


if __name__ == "__main__":
    print(json.dumps(system_health(), indent=2))
