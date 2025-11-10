# supabase_client/helpers.py — Phase 6.3C
"""
Utility layer for interacting with Supabase.

Features
--------
- Safe, reusable wrappers for inserting and fetching records.
- Graceful handling of transient errors (e.g., connection or schema issues).
- Automatic timestamp fallback (for tables without default `created_at`).
- Optional debug logging for local testing.

Intended for backend analytics & optimizer logging (Phase 6.3+).
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, Any, List, Optional

from supabase_client.config import get_supabase_client


def insert_record(
    table: str,
    data: Dict[str, Any],
    debug: bool = False
) -> Dict[str, Any]:
    """
    Insert a record into a Supabase table safely.

    Parameters
    ----------
    table : str
        Target table name in Supabase.
    data : dict
        Dictionary of column names and values.
    debug : bool
        If True, prints status messages for local development.

    Returns
    -------
    dict
        Inserted record or empty dict if failed.
    """
    try:
        supabase = get_supabase_client()

        # Add fallback timestamp if table expects it
        if "created_at" not in data:
            data["created_at"] = dt.datetime.utcnow().isoformat()

        res = supabase.table(table).insert(data).execute()
        if debug:
            print(f"[Supabase] ✅ Inserted into '{table}' → {res.data}")
        return res.data or {}

    except Exception as e:
        if debug:
            print(f"[Supabase] ⚠️ Insert failed for '{table}': {e}")
        return {}


def fetch_recent(
    table: str,
    limit: int = 10,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Fetch the most recent records from a Supabase table.

    Parameters
    ----------
    table : str
        Table name.
    limit : int
        Maximum number of rows to return.
    debug : bool
        If True, prints query progress.

    Returns
    -------
    list[dict]
        List of records (may be empty).
    """
    try:
        supabase = get_supabase_client()
        res = (
            supabase.table(table)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        if debug:
            print(f"[Supabase] 🔍 Fetched {len(res.data or [])} records from '{table}'.")
        return res.data or []

    except Exception as e:
        if debug:
            print(f"[Supabase] ⚠️ Fetch failed for '{table}': {e}")
        return []


def test_connection(debug: bool = True) -> Optional[str]:
    """
    Quick sanity check to verify Supabase connection.

    Returns
    -------
    Optional[str]
        Project URL if successful, None otherwise.
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
