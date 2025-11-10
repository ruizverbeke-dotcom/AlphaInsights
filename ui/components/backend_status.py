# ui/components/backend_status.py — Phase 6.5+
"""
Centralized backend health indicator for Streamlit dashboards.

Features
--------
✅ Environment-aware: Reads BACKEND_URL from environment or config.
✅ Type-safe: Uses Pydantic to validate /health schema.
✅ Caching: Efficient via st.cache_data with configurable TTL.
✅ Graceful fallback: Never crashes UI even if backend is offline.
✅ Extensible: Supports advanced diagnostics & dynamic color indicators.

Intended for use across all Streamlit dashboards.
"""

from __future__ import annotations

import os
import requests
import streamlit as st
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# --------------------------------------------------------------------------- #
# Configuration (future-safe, centralized)
# --------------------------------------------------------------------------- #

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_URL = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL)
CACHE_TTL = int(os.getenv("BACKEND_STATUS_TTL", "60"))  # seconds


# --------------------------------------------------------------------------- #
# Typed Health Schema (robust for backend evolution)
# --------------------------------------------------------------------------- #

class HealthSchema(BaseModel):
    """Structured schema for the /health endpoint response."""
    status: str = Field(default="unknown", description="Overall backend status")
    message: Optional[str] = Field(default=None, description="Optional status message")
    version: Optional[str] = Field(default=None, description="Backend version string")
    supabase_connected: Optional[bool] = Field(default=None, description="Supabase connectivity flag")
    supabase_url: Optional[str] = Field(default=None, description="Supabase project URL")
    cpu_load: Optional[float] = Field(default=None, description="Backend CPU load (optional)")
    memory_usage: Optional[float] = Field(default=None, description="Backend memory usage in MB")
    latency_ms: Optional[float] = Field(default=None, description="Approximate round-trip latency in ms")

    def color(self) -> str:
        """Return a color code for this status."""
        return {
            "ok": "green",
            "healthy": "green",
            "degraded": "orange",
            "error": "red",
            "offline": "red",
        }.get(self.status.lower(), "gray")


# --------------------------------------------------------------------------- #
# Health Fetcher (cached + resilient)
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=CACHE_TTL)
def get_backend_status() -> Dict[str, Any]:
    """
    Fetch the backend /health endpoint with structured fallback.

    Returns
    -------
    dict
        Parsed JSON response or a structured error dict.
    """
    url = f"{BACKEND_URL.rstrip('/')}/health"
    try:
        resp = requests.get(url, timeout=5)
        latency_ms = round(resp.elapsed.total_seconds() * 1000, 2)

        if resp.status_code == 200:
            data = resp.json()
            data["latency_ms"] = latency_ms
            health = HealthSchema(**data)
            return health.model_dump()
        else:
            return {
                "status": "error",
                "message": f"HTTP {resp.status_code}: {resp.text[:100]}",
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "offline",
            "message": f"Backend unreachable at {BACKEND_URL} ({e.__class__.__name__})",
        }
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}


# --------------------------------------------------------------------------- #
# UI Renderer
# --------------------------------------------------------------------------- #

def render_status_bar(expanded: bool = False):
    """
    Render a compact backend health summary in the sidebar.

    Parameters
    ----------
    expanded : bool
        If True, show detailed diagnostics; else compact mode.
    """
    st.sidebar.markdown("---")
    st.sidebar.caption("### 🔍 Backend Status")

    health = get_backend_status()
    status = health.get("status", "unknown")
    color = {
        "ok": "green",
        "healthy": "green",
        "degraded": "orange",
        "error": "red",
        "offline": "red",
    }.get(status.lower(), "gray")

    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:600;'>● {status.upper()}</span>",
        unsafe_allow_html=True,
    )

    msg = health.get("message")
    if msg:
        st.sidebar.caption(f"💬 {msg}")

    if health.get("supabase_connected"):
        st.sidebar.caption("☁️ Supabase: connected")
    else:
        st.sidebar.caption("☁️ Supabase: unavailable")

    # Optional diagnostics
    if expanded:
        with st.sidebar.expander("Advanced diagnostics", expanded=False):
            latency = health.get("latency_ms")
            if latency:
                st.write(f"⏱ Latency: {latency} ms")
            if health.get("cpu_load") is not None:
                st.write(f"🧠 CPU load: {health['cpu_load']}%")
            if health.get("memory_usage") is not None:
                st.write(f"💾 Memory: {health['memory_usage']} MB")
            if health.get("version"):
                st.write(f"🧩 Version: {health['version']}")
            st.json(health)


# --------------------------------------------------------------------------- #
# Helper (for use in dashboards)
# --------------------------------------------------------------------------- #

def get_status_color(status: str) -> str:
    """Public helper for coloring elements dynamically by status."""
    return {
        "ok": "green",
        "healthy": "green",
        "degraded": "orange",
        "error": "red",
        "offline": "red",
    }.get(status.lower(), "gray")
