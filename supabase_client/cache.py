"""
Phase 7.4 — Valuation Cache Helpers
-----------------------------------

This module provides a safe, non-fatal Supabase-backed caching layer
for the /valuation/signals endpoint.

Rules:
- All Supabase failures must be silent (logged but never raised).
- The cache must never break endpoint functionality.
- Cache entries are keyed by a deterministic cache_key string.
- TTL-based staleness checks determine whether a cached entry is usable.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from .helpers import get_supabase_client


# Default TTL (in minutes) if environment variable not provided.
DEFAULT_TTL_MINUTES = 60


def get_cached_valuation(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached valuation payload from Supabase.
    Returns:
        dict representing row with fields:
            - cache_key
            - payload
            - created_at
            - ttl_minutes
            - hit_count
        OR None if not found, expired, or Supabase is unavailable.
    """
    try:
        supabase = get_supabase_client()
        if supabase is None:
            return None

        response = (
            supabase
            .table("valuation_cache")
            .select("*")
            .eq("cache_key", cache_key)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        row = response.data[0]
        return row

    except Exception as e:
        print(f"[Cache] Error in get_cached_valuation({cache_key}): {e}")
        return None


def put_cached_valuation(
    cache_key: str,
    payload: Dict[str, Any],
    ttl_minutes: int = DEFAULT_TTL_MINUTES
) -> None:
    """
    Store or update a cached valuation payload in Supabase.
    On any Supabase error, the function logs and silently returns.
    """
    try:
        supabase = get_supabase_client()
        if supabase is None:
            return

        # Upsert ensures one row per cache_key.
        entry = {
            "cache_key": cache_key,
            "payload": payload,
            "ttl_minutes": ttl_minutes,
            "hit_count": 0,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        supabase.table("valuation_cache").upsert(entry).execute()

    except Exception as e:
        print(f"[Cache] Error in put_cached_valuation({cache_key}): {e}")
        return


def is_cache_fresh(row: Dict[str, Any]) -> bool:
    """
    Determine whether the cached row is still fresh.
    Based on created_at timestamp and ttl_minutes.
    """
    try:
        created_at_raw = row.get("created_at")
        ttl = row.get("ttl_minutes", DEFAULT_TTL_MINUTES)

        if not created_at_raw:
            return False

        # Handle formats like ISO8601 from Supabase.
        created_at = datetime.fromisoformat(created_at_raw.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - created_at

        return age <= timedelta(minutes=ttl)

    except Exception as e:
        print(f"[Cache] Error in is_cache_fresh(): {e}")
        return False
