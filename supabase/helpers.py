# supabase/helpers.py — Phase 6.3B
from typing import Dict, Any
from supabase.config import get_supabase_client

def insert_record(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a record into a Supabase table."""
    supabase = get_supabase_client()
    res = supabase.table(table).insert(data).execute()
    return res.data or {}

def fetch_recent(table: str, limit: int = 10) -> list[Dict[str, Any]]:
    """Fetch the most recent records."""
    supabase = get_supabase_client()
    res = supabase.table(table).select("*").order("created_at", desc=True).limit(limit).execute()
    return res.data or []
