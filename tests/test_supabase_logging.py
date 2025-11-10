# tests/test_supabase_logging.py — Phase 6.3E verification
import json
import requests

def test_supabase_logging_sharpe():
    """Trigger /optimize/sharpe to verify Supabase log insertion."""
    payload = {
        "tickers": ["apple", "microsoft", "gold"],
        "start": "2021-01-01",
        "end": "2024-12-31",
        "rf": 0.02
    }
    res = requests.post("http://127.0.0.1:8000/optimize/sharpe", json=payload, timeout=60)
    print("Status:", res.status_code)
    try:
        data = res.json()
        print(json.dumps(data, indent=2))
    except Exception:
        print("Non-JSON response:", res.text)

if __name__ == "__main__":
    test_supabase_logging_sharpe()
