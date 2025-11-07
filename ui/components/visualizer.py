"""
AlphaInsights — Portfolio Visualization Component
-------------------------------------------------
Phase 5.9 — Real-Time Portfolio Visualization (Scaffold)

Purpose
-------
Provide reusable, 1D-safe helpers to:
- Compute an optimized portfolio's cumulative performance from prices + weights.
- Optionally overlay a benchmark.
- Return a Plotly figure that Streamlit pages can display.

Design
------
- Pure functions, no Streamlit imports (UI calls these).
- No network calls here: prices must be provided by the caller.
- Fully compatible with AlphaInsights 1D safety rules.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px


# --------------------------------------------------------------------------- #
# 1D Safety Helper
# --------------------------------------------------------------------------- #
def _ensure_2d_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a clean 2-D price DataFrame with aligned 1-D columns.

    - Forces DataFrame type.
    - Squeezes (N,1) shapes.
    - Uses np.ravel to guarantee flat arrays.
    - Drops rows with any NaNs.

    This mirrors the global AlphaInsights 1-D safety contract.
    """
    if not isinstance(prices, pd.DataFrame):
        prices = pd.DataFrame(prices)

    cols = []
    for col in prices.columns:
        s = prices[col]
        if isinstance(s, pd.DataFrame):
            s = s.squeeze("columns")
        arr = np.ravel(s)
        idx = s.index[: len(arr)]
        cols.append(pd.Series(arr, index=idx, name=col))

    df = pd.concat(cols, axis=1).dropna(how="any")
    if df.empty:
        raise ValueError("No valid price data after 1-D cleanup.")
    return df


# --------------------------------------------------------------------------- #
# Core: Portfolio Cumulative Performance
# --------------------------------------------------------------------------- #
def compute_portfolio_cumulative_returns(
    prices: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.Series:
    """
    Compute cumulative portfolio value (starting at 1.0) given prices & weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price data; columns are tickers.
    weights : dict[str, float]
        Portfolio weights keyed by ticker symbol.

    Returns
    -------
    pd.Series
        Cumulative portfolio value over time (1.0 = start).
    """
    if not weights:
        raise ValueError("Weights dictionary is empty.")

    df = _ensure_2d_prices(prices)

    # Filter to tickers present in both
    common = [t for t in df.columns if t in weights]
    if not common:
        raise ValueError("No overlap between price columns and weight keys.")

    # Normalize weights on the intersection
    w_vec = np.array([weights[t] for t in common], dtype=float)
    if w_vec.sum() == 0:
        raise ValueError("Sum of provided weights is zero.")
    w_vec = w_vec / w_vec.sum()

    # Compute log returns
    rets = np.log(df[common] / df[common].shift(1)).dropna()
    # 1-D safety is already enforced via _ensure_2d_prices

    # Portfolio returns and cumulative value
    port_rets = rets.values @ w_vec
    port_series = pd.Series(port_rets, index=rets.index)

    # 1D enforce (belt-and-suspenders)
    port_series = port_series.squeeze()
    port_series = pd.Series(
        np.ravel(port_series),
        index=port_series.index[: len(np.ravel(port_series))],
    )

    cumulative = (1.0 + port_series).cumprod()
    cumulative.name = "Portfolio"
    return cumulative


# --------------------------------------------------------------------------- #
# Optional: Benchmark Handling
# --------------------------------------------------------------------------- #
def align_with_benchmark(
    portfolio: pd.Series,
    benchmark: Optional[pd.Series],
    benchmark_label: str = "Benchmark",
) -> Tuple[pd.Series, Optional[pd.Series]]:
    """
    Align portfolio and optional benchmark series on the intersection of dates.
    """
    if benchmark is None:
        return portfolio, None

    # 1-D safety for benchmark
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.squeeze("columns")
    benchmark = pd.Series(
        np.ravel(benchmark),
        index=benchmark.index[: len(np.ravel(benchmark))],
        name=benchmark_label,
    )

    # Align on common index
    joined = pd.concat([portfolio, benchmark], axis=1, join="inner").dropna(how="any")
    if joined.empty:
        return portfolio, None

    port_aligned = joined.iloc[:, 0]
    bench_aligned = joined.iloc[:, 1]
    return port_aligned, bench_aligned


# --------------------------------------------------------------------------- #
# Plotly Figure Builder
# --------------------------------------------------------------------------- #
def build_performance_figure(
    portfolio: pd.Series,
    benchmark: Optional[pd.Series] = None,
    title: str = "Optimized Portfolio Performance",
) -> "px.line":
    """
    Build a Plotly line chart of portfolio (and optional benchmark) performance.

    Parameters
    ----------
    portfolio : pd.Series
        Cumulative portfolio series (e.g., output of compute_portfolio_cumulative_returns).
    benchmark : pd.Series, optional
        Optional cumulative benchmark series aligned on dates.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Plotly line figure ready for Streamlit display.
    """
    if portfolio is None or portfolio.empty:
        raise ValueError("Portfolio series is empty; cannot plot performance.")

    data = pd.DataFrame({"Date": portfolio.index, "Portfolio": portfolio.values})

    if benchmark is not None and not benchmark.empty:
        # Align again defensively before plotting
        aligned_port, aligned_bench = align_with_benchmark(portfolio, benchmark)
        data = pd.DataFrame(
            {
                "Date": aligned_port.index,
                "Portfolio": aligned_port.values,
                "Benchmark": aligned_bench.values,
            }
        )

    fig = px.line(
        data,
        x="Date",
        y=[c for c in data.columns if c != "Date"],
        title=title,
    )
    fig.update_layout(legend_title_text="Series")
    return fig


# --------------------------------------------------------------------------- #
# End of File
# --------------------------------------------------------------------------- #
