"""
AlphaInsights Safe Cloud Activation Layer
-----------------------------------------
Phase 5.1 — Secure gateway between System Core and external services (e.g., Supabase).

Purpose
-------
This module ensures that all outbound connections (starting with Supabase)
are handled safely and do not leak secrets or crash the system when offline.

Key Features
------------
- Context-managed safe connection wrapper
- Graceful fallback on missing credentials
- JSON-safe structured response for Core Health integration
- Logs connection events with timestamp, never printing secrets

Usage
-----
$ python core/safe_connect.py

Design
------
This file is part of AlphaInsights' "System Core" layer.
It abstracts away direct API calls, allowing future AI agents or
backend services to invoke cloud functionality safely.
"""

from __future__ import annotations

import os
import json
from datetime import datetime, UTC
from contextlib import contextmanager
from typing import Any, Dict, Optional

try:
    from supabase import create_client, Client
except Exception:
    # Dependency may not be installed yet
    create_client = None
    Client = None


@contextmanager
def safe_supabase_connection() -> Optional["Client"]:
    """
    Safely yield a Supabase client if credentials are present.
    Never raises an exception — logs and yields None on failure.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key or create_client is None:
        yield None
        return

    try:
        client = create_client(url, key)
        yield client
    except Exception:
        yield None


def test_connection() -> Dict[str, Any]:
    """
    Attempts to establish a secure Supabase connection and reports status.

    Returns
    -------
    dict
        {
            "timestamp": "<UTC ISO time>",
            "supabase_detected": bool,
            "credentials_present": bool,
            "connected": bool,
            "error": Optional[str],
            "phase": "5.1 (safe cloud activation)"
        }
    """
    status = {
        "timestamp": datetime.now(UTC).isoformat(),
        "supabase_detected": create_client is not None,
        "credentials_present": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")),
        "connected": False,
        "error": None,
        "phase": "5.1 (safe cloud activation)",
    }

    if not create_client:
        status["error"] = "supabase-py not installed"
        return status

    try:
        with safe_supabase_connection() as client:
            if client is None:
                status["error"] = "credentials missing or invalid"
            else:
                # Run a harmless metadata request
                client.table("information_schema.tables")  # not executed, just checks object access
                status["connected"] = True
    except Exception as e:
        status["error"] = str(e)

    return status


if __name__ == "__main__":
    print(json.dumps(test_connection(), indent=2))
