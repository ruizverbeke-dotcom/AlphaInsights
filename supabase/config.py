"""
Supabase Config — Phase 5.0+ (Bootstrap & Accessor Layer)
---------------------------------------------------------
Handles safe configuration of Supabase URL and API key
with optional cloud connection checks.
"""

from __future__ import annotations
import os
import json


def get_supabase_config() -> dict:
    """Return basic Supabase environment config."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    configured = bool(url and key)

    return {
        "url": url or None,
        "key": "****" if key else None,
        "configured": configured,
        "url_set": bool(url),
        "key_set": bool(key),
        "connected": False,  # to be handled by safe_connect
        "error": None,
        "phase": "5.0 (bootstrap)"
    }


if __name__ == "__main__":
    cfg = get_supabase_config()
    print(json.dumps(cfg, indent=2))
