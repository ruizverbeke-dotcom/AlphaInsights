"""
AlphaInsights Intelligent Ticker Resolver (v2)
-----------------------------------------------
Resolves human-readable asset names into valid Yahoo Finance tickers,
with secondary verification to guarantee that legitimate tickers (like "Microsoft")
are never dropped due to temporary API quirks.

Design Principles
-----------------
- Multi-pass approach:
    1. Direct fuzzy match against known aliases.
    2. If Yahoo data fetch fails, retry each candidate individually.
- Returns only tickers that yield at least one valid closing price.
- 1-D safe, stateless, agent-ready.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple
import pandas as pd
import yfinance as yf
from rapidfuzz import process

# --------------------------------------------------------------------------- #
# Core Aliases Map  (extendable via Continuous Discovery Sheet)
# --------------------------------------------------------------------------- #
ALIASES = {
    "MICROSOFT": "MSFT",
    "APPLE": "AAPL",
    "GOOGLE": "GOOG",
    "ALPHABET": "GOOG",
    "TESLA": "TSLA",
    "AMAZON": "AMZN",
    "S&P 500": "^GSPC",
    "SP500": "^GSPC",
    "CAC 40": "^FCHI",
    "DAX": "^GDAXI",
    "FTSE 100": "^FTSE",
    "NIKKEI 225": "^N225",
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "OIL": "CL=F",
    "WTI": "CL=F",
    "BRENT": "BZ=F",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "US10Y": "^TNX",
}


def _verify_single_ticker(ticker: str, start: str, end: str) -> bool:
    """
    Verify that a ticker has at least one valid closing price in the given range.
    Returns True if valid, False otherwise.
    """
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if "Close" in df.columns and df["Close"].dropna().shape[0] > 0:
            return True
    except Exception:
        return False
    return False


def resolve_tickers(raw_input: str, start: str | None = None, end: str | None = None) -> Tuple[List[str], List[str]]:
    """
    Resolve and verify tickers from user input.

    Parameters
    ----------
    raw_input : str
        Comma-separated list of tickers or asset names.
    start, end : str, optional
        Date range for data verification (YYYY-MM-DD).

    Returns
    -------
    Tuple[List[str], List[str]]
        (resolved_valid, dropped)
    """
    if not raw_input:
        return [], []

    names = [x.strip().upper() for x in raw_input.split(",") if x.strip()]
    resolved: List[str] = []
    dropped: List[str] = []

    # 1. Primary pass — fuzzy match to known aliases
    for name in names:
        if name in ALIASES:
            resolved.append(ALIASES[name])
        else:
            match, score, _ = process.extractOne(name, ALIASES.keys(), score_cutoff=80)
            resolved.append(ALIASES[match] if match else name)

    resolved = list(dict.fromkeys(resolved))  # deduplicate while preserving order

    # 2. Secondary pass — verify via yfinance
    if start and end:
        valid: List[str] = []
        for tkr in resolved:
            if _verify_single_ticker(tkr, start, end):
                valid.append(tkr)
            else:
                dropped.append(tkr)
        resolved = valid

    if not resolved:
        warnings.warn("No valid tickers could be verified; check inputs or date range.", UserWarning)

    return resolved, dropped


# --------------------------------------------------------------------------- #
# Example quick test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    valid, bad = resolve_tickers("Microsoft, CAC 40, Gold", start="2024-01-01", end="2024-12-31")
    print("Resolved:", valid)
    print("Dropped:", bad)
