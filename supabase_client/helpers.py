# supabase_client/helpers.py — Phase 6.3D (enhanced debug + safety)
"""
Utility layer for interacting with Supabase.

Features
--------
- Safe, reusable wrappers for inserting and fetching records.
- Graceful handling of transient errors (e.g., connection or schema issues).
- Automatic timestamp fallback (for tables without default `created_at`).
- Optional verbose debug logging for diagnostics.
- Structured return values for easier backend integration.

Used by backend analytics & optimizer endpoints (Phase 6.3+).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Optional
from supabase_client.config import get_supabase_client


def insert_record(
    table: str,
    data: Dict[str, Any],
    debug: bool = True
) -> Dict[str, Any]:
    """
    Insert a record into a Supabase table safely with debug support.

    Parameters
    ----------
    table : str
        Target table name in Supabase.
    data : dict
        Dictionary of column names and values.
    debug : bool
        If True, prints detailed output.

    Returns
    -------
    dict
        Inserted record data or empty dict on failure.
    """
    try:
        supabase = get_supabase_client()

        # Add fallback timestamp if missing
        if "created_at" not in data:
            data["created_at"] = dt.datetime.utcnow().isoformat()

        if debug:
            print(f"[Supabase] → Inserting into '{table}' …")
            print(f"[Supabase] Payload keys: {list(data.keys())}")

        res = supabase.table(table).insert(data).execute()

        if hasattr(res, "status_code") and res.status_code >= 400:
            if debug:
                print(f"[Supabase] ❌ HTTP {res.status_code}: {res.error_message if hasattr(res, 'error_message') else res}")
            return {}

        if debug:
            print(f"[Supabase] ✅ Insert success → {res.data}")

        return res.data or {}

    except Exception as e:
        if debug:
            print(f"[Supabase] ⚠️ Insert failed: {type(e).__name__}: {e}")
        return {}


def fetch_recent(
    table: str,
    limit: int = 10,
    debug: bool = True
) -> List[Dict[str, Any]]:
    """
    Fetch recent records from a Supabase table.

    Parameters
    ----------
    table : str
        Table name.
    limit : int
        Max rows to fetch.
    debug : bool
        If True, print diagnostic info.

    Returns
    -------
    list[dict]
        Records from Supabase or [] if empty/error.
    """
    try:
        supabase = get_supabase_client()
        if debug:
            print(f"[Supabase] → Fetching latest {limit} from '{table}'")

        res = (
            supabase.table(table)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        records = res.data or []
        if debug:
            print(f"[Supabase] ← Got {len(records)} records")

        return records

    except Exception as e:
        if debug:
            print(f"[Supabase] ⚠️ Fetch failed: {e}")
        return []


def test_connection(debug: bool = True) -> Optional[str]:
    """
    Verify Supabase client connectivity.

    Returns
    -------
    Optional[str]
        Supabase project URL if success, None if failure.
    """
    try:
        sb = get_supabase_client()
        if debug:
            print(f"[Supabase] ✅ Connection OK → {sb.supabase_url}")
        return sb.supabase_url
    except Exception as e:
        if debug:
            print(f"[Supabase] ❌ Connection failed: {e}")
        return None
