"""
core/ui_config.py
-----------------
Central configuration hub for all Streamlit UI pages.

- Reads backend & Supabase URLs from environment variables.
- Provides global constants for API access.
- Includes lightweight health check.
"""

from __future__ import annotations
import os
import requests

# ---------------------------------------------------------------------------
# Backend configuration
# ---------------------------------------------------------------------------

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# Optional: expose Supabase info for UI references
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY: str | None = os.getenv("SUPABASE_ANON_KEY")

# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

def check_backend_health() -> dict:
    """Ping the backend /health endpoint and return its JSON."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=4)
        return resp.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}
