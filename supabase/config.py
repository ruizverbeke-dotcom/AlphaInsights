"""
supabase/config.py
------------------
Supabase integration scaffold for AlphaInsights (Phase 4.4)
------------------------------------------------------------
This module defines environment-safe configuration and lazy client initialization.
It does not connect yet; full integration arrives in Phase 5 (Cloud Persistence).

Environment variables expected:
- SUPABASE_URL
- SUPABASE_KEY
"""

import os
from dataclasses import dataclass
from typing import Optional

# Optional import guard — only needed when actual client is installed in Phase 5
try:
    from supabase import create_client, Client
except ImportError:  # Safe placeholder
    Client = None
    create_client = None


@dataclass(frozen=True)
class SupabaseConfig:
    """Holds Supabase credentials loaded from environment variables."""
    url: Optional[str] = os.getenv("SUPABASE_URL")
    key: Optional[str] = os.getenv("SUPABASE_KEY")

    @property
    def is_configured(self) -> bool:
        """True if both URL and KEY are present."""
        return bool(self.url and self.key)


def get_supabase_client() -> Optional["Client"]:
    """
    Initialize a Supabase client if credentials exist.
    Returns None if environment variables are missing or client is unavailable.
    """
    cfg = SupabaseConfig()
    if not cfg.is_configured:
        print("⚠️ Supabase credentials not configured. Skipping client init.")
        return None

    if create_client is None:
        print("⚙️ Supabase Python SDK not installed yet (Phase 5). Returning None.")
        return None

    try:
        client = create_client(cfg.url, cfg.key)
        print("✅ Supabase client initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ Supabase client initialization failed: {e}")
        return None


def get_status() -> dict:
    """Lightweight health probe."""
    cfg = SupabaseConfig()
    return {
        "configured": cfg.is_configured,
        "url_set": bool(cfg.url),
        "key_set": bool(cfg.key),
        "phase": "4.4 (scaffold)",
    }


if __name__ == "__main__":
    # Quick self-test when running `python supabase/config.py`
    print("Supabase Config Health →", get_status())
