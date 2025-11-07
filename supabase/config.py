"""
supabase/config.py
------------------
Secure Supabase configuration and connectivity layer (Phase 5.0).

• Reads credentials from environment variables (never hard-coded).
• Verifies connection health non-destructively.
• Used by backend & analytics for profile persistence.

Environment variables expected:
    SUPABASE_URL      → Supabase project URL
    SUPABASE_KEY      → Supabase service role or anon key
"""

import os
from typing import Dict, Any

try:
    from supabase import create_client, Client  # type: ignore
except ImportError:
    create_client = None
    Client = None


def get_supabase_client() -> "Client | None":
    """Create and return a Supabase client if credentials exist."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        return None

    if create_client is None:
        raise ImportError("The 'supabase' Python SDK is not installed. Run: pip install supabase")

    return create_client(url, key)


def check_supabase_health() -> Dict[str, Any]:
    """Perform a lightweight Supabase health check."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    health = {
        "configured": bool(url and key),
        "url_set": bool(url),
        "key_set": bool(key),
        "connected": False,
        "error": None,
        "phase": "5.0 (bootstrap)",
    }

    if not url or not key or create_client is None:
        return health

    try:
        client = create_client(url, key)
        # Non-destructive query to validate connection
        client.table("pg_catalog.pg_tables").select("*").limit(1).execute()
        health["connected"] = True
    except Exception as e:
        health["error"] = str(e)

    return health


if __name__ == "__main__":
    import json
    print(json.dumps(check_supabase_health(), indent=2))
