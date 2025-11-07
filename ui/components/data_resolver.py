"""
AlphaInsights — Data Resolver Component
----------------------------------------
Resolves human-readable asset names (like 'Microsoft', 'CAC 40', 'Gold')
into canonical tickers compatible with the backend optimizer.

Phase 5.8 — Safe return handling for dashboard integration.
"""

from typing import List, Tuple

def resolve_tickers(user_input: str, start: str, end: str) -> Tuple[List[str], List[str]]:
    """
    Resolve raw input string into a list of usable tickers.

    Parameters
    ----------
    user_input : str
        Comma-separated list of tickers or asset names.
    start : str
        Start date (ISO).
    end : str
        End date (ISO).

    Returns
    -------
    tuple[list[str], list[str]]
        (resolved_tickers, dropped_tickers)
    """
    if not user_input:
        return [], []

    # Basic parsing
    raw = [x.strip() for x in user_input.split(",") if x.strip()]
    resolved = []
    dropped = []

    # Basic mapping examples (expand later)
    name_map = {
        "microsoft": "MSFT",
        "apple": "AAPL",
        "gold": "GC=F",
        "sp500": "^GSPC",
        "spx": "^GSPC",
        "cac 40": "^FCHI",
    }

    for item in raw:
        key = item.lower()
        if key in name_map:
            resolved.append(name_map[key])
        else:
            # Assume it’s already a valid ticker
            resolved.append(item.upper())

    # Deduplicate
    resolved = list(dict.fromkeys(resolved))

    return resolved, dropped
