# ===============================================================
# Sharpe Ratio Module — AlphaInsights Analytics Engine
# ===============================================================
# Author: Ruïz Verbeke
# Created: 2025-10-29
# Description:
#     Provides functions to calculate Sharpe Ratios (daily, monthly,
#     and annualized) for single assets or entire portfolios using
#     historical return data. Supports dynamic weighting, variable
#     time periods, and integration with Streamlit dashboards.
# ===============================================================

# ===============================================================
# Imports
# ===============================================================
import warnings
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union

# Silence future warnings (e.g., yfinance auto_adjust changes)
warnings.simplefilter("ignore", FutureWarning)

# ===============================================================
# Utility Functions
# ===============================================================
def _download_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Internal helper: download Adjusted Close prices for one or more tickers.

    Handles both single- and multi-ticker return formats from yfinance.
    """
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
        group_by="ticker"
    )

    # Case 1: MultiIndex columns (multiple tickers or new yfinance behavior)
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(1):
            # Extract only the Adjusted Close layer
            adj_close = data.xs("Adj Close", axis=1, level=1)
            adj_close = adj_close.loc[:, ~adj_close.columns.duplicated()]  # remove duplicates
            prices = adj_close.copy()
        else:
            raise ValueError("Unexpected data format: 'Adj Close' not found in MultiIndex DataFrame.")
    else:
        # Case 2: Single-level columns (typical for one ticker)
        if "Adj Close" not in data.columns:
            raise ValueError(f"'Adj Close' column not found in data for {tickers}")
        prices = data[["Adj Close"]].copy()
        # Rename column for consistency
        if isinstance(tickers, list) and len(tickers) == 1:
            prices.columns = [tickers[0]]
        elif isinstance(tickers, str):
            prices.columns = [tickers]

    prices.dropna(how="all", inplace=True)
    prices.index = pd.to_datetime(prices.index)
    return prices


# ===============================================================
# Core Function: Calculate Sharpe Ratio
# ===============================================================
def calculate_sharpe_ratio(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: str,
    risk_free_rate_annual: float = 0.02,
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Calculate daily and annualized Sharpe Ratios for single or multiple assets.

    Parameters
    ----------
    tickers : str or list of str
        Single ticker (e.g., 'AAPL') or list of tickers (e.g., ['AAPL','MSFT','SPY']).
    start_date : str
        Start date for analysis ('YYYY-MM-DD').
    end_date : str
        End date for analysis ('YYYY-MM-DD').
    risk_free_rate_annual : float, default 0.02
        Annualized risk-free rate (e.g., 2% = 0.02).
    weights : list of float, optional
        Portfolio weights. Must match number of tickers. If None, equal weighting is used.

    Returns
    -------
    pd.DataFrame
        Sharpe ratio metrics:
            - label
            - tickers
            - weights
            - period_start / period_end
            - risk_free_rate_annual
            - mean_daily_return
            - mean_daily_excess_return
            - daily_volatility
            - sharpe_daily
            - sharpe_annualized
    """

    # --- Step 1: Normalize inputs ---
    if isinstance(tickers, str):
        tickers = [tickers]
    n_assets = len(tickers)

    if weights is None:
        weights = np.repeat(1 / n_assets, n_assets)
    else:
        weights = np.array(weights)
        if not np.isclose(weights.sum(), 1):
            raise ValueError("Portfolio weights must sum to 1.0")

    # --- Step 2: Download prices ---
    prices = _download_prices(tickers, start_date, end_date)
    if prices.empty:
        raise ValueError("No price data returned for selected tickers/date range.")

    # --- Step 3: Compute daily returns ---
    daily_returns = prices.pct_change().dropna()

    # --- Step 4: Convert annual Rf rate to daily ---
    trading_days = 252
    daily_rf_rate = (1 + risk_free_rate_annual) ** (1 / trading_days) - 1

    # --- Step 5: Portfolio-level or single-asset excess returns ---
    excess_returns = daily_returns - daily_rf_rate
    portfolio_returns = (daily_returns * weights).sum(axis=1)
    portfolio_excess = (excess_returns * weights).sum(axis=1)

    # --- Step 6: Statistics ---
    mean_daily_return = portfolio_returns.mean()
    mean_daily_excess = portfolio_excess.mean()
    daily_volatility = portfolio_excess.std()

    # Sharpe Ratios
    sharpe_daily = np.nan if daily_volatility == 0 else mean_daily_excess / daily_volatility
    annualized_excess_return = mean_daily_excess * trading_days
    annualized_volatility = daily_volatility * np.sqrt(trading_days)
    sharpe_annualized = (
        np.nan if annualized_volatility == 0 else annualized_excess_return / annualized_volatility
    )

    # --- Step 7: Return DataFrame ---
    label = "Portfolio" if len(tickers) > 1 else tickers[0]
    result = pd.DataFrame(
        {
            "label": [label],
            "tickers": [", ".join(tickers)],
            "weights": [weights.tolist()],
            "period_start": [start_date],
            "period_end": [end_date],
            "risk_free_rate_annual": [risk_free_rate_annual],
            "mean_daily_return": [mean_daily_return],
            "mean_daily_excess_return": [mean_daily_excess],
            "daily_volatility": [daily_volatility],
            "sharpe_daily": [sharpe_daily],
            "sharpe_annualized": [sharpe_annualized],
        }
    )

    return result


# ===============================================================
# Helper Function: Compare Multiple Portfolios
# ===============================================================
def compare_portfolios(
    portfolios: Dict[str, Dict],
    start_date: str,
    end_date: str,
    risk_free_rate_annual: float = 0.02,
) -> pd.DataFrame:
    """
    Compare Sharpe ratios across multiple portfolios.

    Parameters
    ----------
    portfolios : dict
        Example:
        {
            "Tech Growth": {"tickers": ["AAPL","MSFT","NVDA"], "weights": [0.4,0.4,0.2]},
            "Dividend Core": {"tickers": ["KO","PG","JNJ"], "weights": [0.3,0.4,0.3]},
        }
    start_date, end_date : str
        Analysis period.
    risk_free_rate_annual : float, default 0.02
        Annualized risk-free rate.

    Returns
    -------
    pd.DataFrame
        Comparison table of all portfolio Sharpe metrics.
    """
    results = []
    for name, cfg in portfolios.items():
        tickers = cfg.get("tickers")
        weights = cfg.get("weights")
        df = calculate_sharpe_ratio(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate_annual=risk_free_rate_annual,
            weights=weights,
        )
        df["label"] = name
        results.append(df)

    return pd.concat(results, ignore_index=True)


# ===============================================================
# Manual Test
# ===============================================================
if __name__ == "__main__":
    # --- Single asset example ---
    print("Single Asset Sharpe Ratio:\n")
    single = calculate_sharpe_ratio("AAPL", "2024-01-01", "2024-12-31")
    print(single, "\n")

    # --- Multi-asset example ---
    print("Portfolio Sharpe Ratio:\n")
    portfolio = calculate_sharpe_ratio(
        tickers=["AAPL", "MSFT", "NVDA"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        weights=[0.4, 0.4, 0.2],
    )
    print(portfolio, "\n")

    # --- Portfolio comparison example ---
    print("Portfolio Comparison:\n")
    portfolios = {
        "Tech Growth": {"tickers": ["AAPL", "MSFT", "NVDA"], "weights": [0.4, 0.4, 0.2]},
        "Dividend Core": {"tickers": ["KO", "PG", "JNJ"], "weights": [0.3, 0.4, 0.3]},
    }
    comparison = compare_portfolios(
        portfolios, start_date="2024-01-01", end_date="2024-12-31"
    )
    print(comparison)