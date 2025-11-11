"""
core/ui_helpers.py
------------------
Shared backend request helpers for all Streamlit dashboards.
Ensures consistent error handling and caching.
"""

from __future__ import annotations
import requests
import streamlit as st
from core.ui_config import BACKEND_URL

@st.cache_data(ttl=60)
def fetch_backend(endpoint: str, params: dict | None = None) -> dict | list:
    """
    Unified safe fetch for GET endpoints.
    Automatically prefixes BACKEND_URL and handles JSON decoding.
    """
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Backend request failed: {e}")
        return {}

def post_backend(endpoint: str, payload: dict) -> dict:
    """Unified POST helper."""
    url = f"{BACKEND_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Backend POST failed: {e}")
        return {}
