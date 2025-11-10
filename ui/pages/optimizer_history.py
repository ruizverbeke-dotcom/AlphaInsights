# ui/pages/optimizer_history.py — Phase 6.4
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Optimization History", layout="wide")

st.title("📜 Optimization History")
st.caption("Fetched from Supabase via backend `/logs/recent` endpoint")

BACKEND_URL = "http://127.0.0.1:8000"

@st.cache_data(ttl=30)
def fetch_logs(limit: int = 20):
    """Fetch recent optimizer logs from backend."""
    try:
        res = requests.get(f"{BACKEND_URL}/logs/recent", params={"limit": limit}, timeout=30)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Failed to fetch logs: {e}")
        return []

logs = fetch_logs()

if not logs:
    st.info("No logs found yet. Run an optimization to populate history.")
else:
    df = pd.DataFrame(logs)

    # Flatten JSON columns for display
    if "result" in df.columns:
        result_df = pd.json_normalize(df["result"])
        df = pd.concat([df.drop(columns=["result"]), result_df], axis=1)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Raw JSON View"):
        st.json(logs)
