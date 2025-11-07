"""
AlphaInsights — Core Symbol Resolver
------------------------------------
Phase 6.2.A — Case-Insensitive, Robust Symbol Resolution

Maps human-readable asset names to Yahoo Finance symbols
with fuzzy matching, index awareness, and casing resilience.
"""

from __future__ import annotations
import re
import requests
from functools import lru_cache

# --- Common index / ETF / commodity aliases ---
STATIC_MAP = {
    # companies
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOG",
    "yahoo": "YHOO",

    # indices
    "s&p500": "^GSPC",
    "sp500": "^GSPC",
    "s&p 500": "^GSPC",
    "cac40": "^FCHI",
    "cac 40": "^FCHI",
    "dax": "^GDAXI",
    "bel20": "^BFX",
    "bel 20": "^BFX",
    "ftse100": "^FTSE",
    "ftse 100": "^FTSE",
    "euro stoxx 50": "^STOXX50E",
    "stoxx600": "^STOXX",
    "nasdaq": "^NDX",
    "nasdaq 100": "^NDX",
    "ndx": "^NDX",
    # commodities
    "gold": "GC=F",
    "silver": "SI=F",
    "oil": "CL=F",
    "brent": "BZ=F",
    # popular ETFs
    "spy": "SPY",
    "qqq": "QQQ",
    "voo": "VOO",
    "iwm": "IWM",
    "efa": "EFA",
    "eem": "EEM",
}


def _clean(text: str) -> str:
    """Normalize and lowercase text for stable lookups."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


@lru_cache(maxsize=512)
def resolve_symbol(query: str) -> str | None:
    """
    Try to interpret a user-supplied term and return a valid Yahoo symbol.
    Steps:
      1) Case-insensitive STATIC_MAP match
      2) Yahoo Finance fuzzy search fallback
    """
    if not query:
        return None

    q_clean = _clean(query)
    q_upper = query.strip().upper()

    # --- Step 1: Case-insensitive static map lookup ---
    for key, val in STATIC_MAP.items():
        if q_clean == _clean(key):
            return val

    # --- Step 2: Directly valid ticker-like strings ---
    if len(q_upper) <= 6 and all(ch.isalnum() or ch in ".=^-" for ch in q_upper):
        return q_upper  # Likely already a ticker

    # --- Step 3: Yahoo Finance fuzzy search fallback ---
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        resp = requests.get(url, params={"q": query, "lang": "en-US"}, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        quotes = data.get("quotes") or []
        if not quotes:
            return None

        # Prefer equities, indexes, or ETFs with an exchange field
        for item in quotes:
            sym = item.get("symbol")
            if sym and item.get("exchange") and item.get("quoteType") in {"EQUITY", "INDEX", "ETF"}:
                return sym.upper()

        # Fallback to first symbol if no ideal match found
        best = quotes[0].get("symbol") if quotes else None
        return best.upper() if best else None

    except Exception:
        return None


def resolve_tickers(user_inputs: list[str]) -> tuple[list[str], list[str]]:
    """
    Resolve a list of user-entered tickers/names into valid Yahoo symbols.

    Returns
    -------
    (resolved_symbols, messages)
    """
    resolved, messages, seen = [], [], set()

    for raw in user_inputs:
        if not raw:
            continue
        sym = resolve_symbol(raw)
        if sym and sym.upper() not in seen:
            resolved.append(sym.upper())
            seen.add(sym.upper())
            if sym.upper() != raw.upper():
                messages.append(f"ℹ️ '{raw}' interpreted as '{sym.upper()}'.")
        else:
            messages.append(f"⚠️ '{raw}' could not be resolved to a valid symbol.")

    return resolved, messages
