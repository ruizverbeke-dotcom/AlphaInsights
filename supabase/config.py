# supabase/config.py — Phase 6.3 scaffold
import os
from supabase import create_client, Client

def get_supabase_client() -> Client:
    """Return an authenticated Supabase client if credentials are set."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Supabase credentials not set in environment variables.")
    return create_client(url, key)
