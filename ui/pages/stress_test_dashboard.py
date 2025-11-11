import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Ensure imports resolve correctly when run via Streamlit
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from core.ui_helpers import fetch_backend
from core.ui_config import BACKEND_URL

# ---------------------------------------------------------------------------
# 1. Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Stress Test Dashboard", layout="wide")

st.title("💥 Stress Test Dashboard")
st.caption(
    "Simulate how a portfolio might behave under shocks, historical drawdowns, "
    "and tail-risk scenarios. Scenario-based and educational — not trading advice."
)

# ---------------------------------------------------------------------------
# 2. Input Section
# ---------------------------------------------------------------------------
with st.form("stress_inputs"):
    st.subheader("📊 Input Parameters")

    tickers_raw = st.text_input(
        "Tickers (comma-separated)",
        value=st.session_state.get("last_stress_tickers", "AAPL, MSFT, NVDA, SPY"),
        help="Assets to include in the simulated portfolio.",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    with col2:
        end_date = st.date_input("End Date", datetime.today())

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        submitted = st.form_submit_button("🚀 Run Stress Test", disabled=True)
    else:
        st.markdown("#### ⚠️ Shock Scenarios")
        st.write("Specify hypothetical percentage shocks (negative values = losses).")

        col1_s, col2_s, col3_s = st.columns(3)
        with col1_s:
            mild_shock = st.number_input("Mild Shock (%)", value=-5.0, step=1.0)
        with col2_s:
            moderate_shock = st.number_input("Moderate Shock (%)", value=-15.0, step=1.0)
        with col3_s:
            severe_shock = st.number_input("Severe Shock (%)", value=-30.0, step=1.0)

        st.markdown("Optional weight vector (comma-separated, sums to 1).")
        weights_text = st.text_input(
            "Portfolio Weights",
            value="",
            help="If empty: equal weights across valid tickers."
        )

        use_backend = st.checkbox(
            "Use Backend Stress Simulation (if available)",
            value=False,
            help="Calls /simulate/shock endpoint if supported by backend.",
        )

        submitted = st.form_submit_button("🚀 Run Stress Test")

# ---------------------------------------------------------------------------
# 3. Backend Health
# ---------------------------------------------------------------------------
with st.expander("🩺 Backend Health", expanded=True):
    try:
        health = fetch_backend("health")
        if isinstance(health, dict) and health.get("status") == "ok":
            st.success("✅ Backend reachable")
            st.json(health)
        elif health:
            st.warning("⚠️ Backend responded, but health payload is not 'ok':")
            st.write(health)
        else:
            st.info("No health data returned from backend.")
    except Exception as e:
        st.info(f"Backend not connected or health endpoint unavailable: {e}")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_prices(raw: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Normalize yfinance output into a clean Close-price DataFrame."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Multi-ticker case
    if isinstance(raw.columns, pd.MultiIndex):
        level_1 = list(raw.columns.levels[1])
        field = "Close" if "Close" in level_1 else level_1[0]
        close_df = raw.xs(field, axis=1, level=1).copy()
        valid_cols = [t for t in requested if t in close_df.columns]
        if not valid_cols:
            valid_cols = list(close_df.columns)
        return close_df[valid_cols].dropna(how="all")

    # Single ticker case
    candidates = ["Close", "Adj Close", "price"]
    for c in candidates:
        if c in raw.columns:
            close = raw[c]
            break
    else:
        numeric_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if not numeric_cols:
            return pd.DataFrame()
        close = raw[numeric_cols[0]]
    name = requested[0] if requested else "Asset"
    return pd.DataFrame({name: close}).dropna()


def parse_weights(text: str, n: int) -> np.ndarray:
    """Parse a comma-separated weight vector safely."""
    if not text.strip():
        return np.ones(n) / n
    try:
        raw = [float(x.strip().replace(",", ".")) for x in text.split(",") if x.strip()]
        w = np.array(raw, dtype=float)
    except Exception:
        st.warning("⚠️ Invalid weights format. Using equal weights.")
        return np.ones(n) / n

    if len(w) != n or np.any(np.isnan(w)) or w.sum() <= 0:
        st.warning("⚠️ Invalid or mismatched weights. Using equal weights.")
        return np.ones(n) / n

    return w / w.sum()


def compute_var_cvar(returns: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    """Compute Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall)."""
    if returns.empty:
        return np.nan, np.nan
    sorted_returns = np.sort(returns)
    var_idx = int((1 - alpha) * len(sorted_returns))
    var_value = sorted_returns[var_idx]
    cvar_value = sorted_returns[:var_idx].mean() if var_idx > 0 else var_value
    return var_value, cvar_value

# ---------------------------------------------------------------------------
# 4. Core Simulation Logic
# ---------------------------------------------------------------------------
if submitted:
    tickers_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    st.session_state["last_stress_tickers"] = ", ".join(tickers_list)

    if not tickers_list:
        st.error("❌ Please provide at least one ticker.")
        st.stop()

    st.info(f"📥 Downloading historical data for: {', '.join(tickers_list)}")
    try:
        raw_data = yf.download(
            tickers=tickers_list, start=start_date, end=end_date,
            auto_adjust=True, progress=False
        )
    except Exception as e:
        st.error(f"❌ Failed to download data from Yahoo Finance: {e}")
        st.stop()

    prices = normalize_prices(raw_data, tickers_list)
    if prices.empty:
        st.error("❌ No valid price data retrieved. Try adjusting date range or symbols.")
        st.stop()

    returns = prices.pct_change().dropna(how="all")
    if returns.empty:
        st.error("❌ No valid daily returns computed.")
        st.stop()

    effective_tickers = list(returns.columns)
    w = parse_weights(weights_text, len(effective_tickers))

    # --- Core stats ---
    mean_ret = returns.mean()
    cov = returns.cov()
    port_mean = float(np.dot(mean_ret, w))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

    st.success("✅ Historical portfolio statistics computed.")
    st.write(f"**Assets used:** {', '.join(effective_tickers)}")
    st.write(f"**Expected daily return:** {port_mean:.4%}")
    st.write(f"**Daily volatility:** {port_vol:.4%}")

    # --- Tail risk metrics ---
    port_ret_series = returns.dot(w)
    var95, cvar95 = compute_var_cvar(port_ret_series, 0.95)
    st.markdown("### 🧮 Tail-Risk Metrics (95% confidence)")
    colA, colB = st.columns(2)
    colA.metric("Value at Risk (VaR 95%)", f"{var95 * 100:.2f}%")
    colB.metric("Conditional VaR (CVaR 95%)", f"{cvar95 * 100:.2f}%")

    # -----------------------------------------------------------------------
    # 4.5 Shock Scenarios
    # -----------------------------------------------------------------------
    st.subheader("📉 Stress Test Scenarios")

    scenarios = {
        "Mild Shock": mild_shock / 100.0,
        "Moderate Shock": moderate_shock / 100.0,
        "Severe Shock": severe_shock / 100.0,
    }

    results = []
    for label, shock in scenarios.items():
        stressed_return = port_mean + shock
        stressed_ann = (1 + stressed_return) ** 252 - 1
        results.append({
            "Scenario": label,
            "Shock (%)": shock * 100.0,
            "Baseline Daily Return (%)": port_mean * 100.0,
            "Stressed Daily Return (%)": stressed_return * 100.0,
            "Stressed Annualized Return (%)": stressed_ann * 100.0,
        })
    df_res = pd.DataFrame(results)

    st.dataframe(df_res.style.format({
        "Shock (%)": "{:.1f}",
        "Baseline Daily Return (%)": "{:.3f}",
        "Stressed Daily Return (%)": "{:.3f}",
        "Stressed Annualized Return (%)": "{:.2f}",
    }), use_container_width=True)

    fig = px.bar(
        df_res,
        x="Scenario",
        y="Stressed Daily Return (%)",
        color="Scenario",
        text=df_res["Stressed Daily Return (%)"].map(lambda x: f"{x:.2f}%"),
        title="Portfolio Performance Under Shock Scenarios (1-day impact)",
        labels={"Stressed Daily Return (%)": "Stressed Return (%)"},
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # 4.6 Historical Drawdown
    # -----------------------------------------------------------------------
    st.subheader("📊 Historical Drawdown")
    cumulative = (1 + port_ret_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max

    if drawdown.empty:
        st.info("Not enough data to compute drawdowns.")
    else:
        dd_df = drawdown.to_frame(name="Drawdown").reset_index()
        dd_df.rename(columns={dd_df.columns[0]: "Date"}, inplace=True)
        fig_dd = px.area(
            dd_df,
            x="Date", y="Drawdown",
            title="Historical Portfolio Drawdowns",
            labels={"Drawdown": "Drawdown (relative to peak)"},
        )
        fig_dd.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_dd, use_container_width=True)

    # -----------------------------------------------------------------------
    # 4.7 Backend Stress API (optional future hook)
    # -----------------------------------------------------------------------
    if use_backend:
        try:
            with st.spinner("Calling backend stress simulation endpoint..."):
                payload = {
                    "tickers": effective_tickers,
                    "weights": list(map(float, w)),
                    "start": str(start_date),
                    "end": str(end_date),
                    "shocks": {"mild": mild_shock, "moderate": moderate_shock, "severe": severe_shock},
                }
                backend_result = fetch_backend("simulate/shock", method="POST", json=payload)
                if isinstance(backend_result, dict):
                    st.success("✅ Backend stress simulation result:")
                    st.json(backend_result)
                else:
                    st.warning("⚠️ Unexpected backend response type.")
        except Exception as e:
            st.warning(f"⚠️ Backend stress API unavailable or errored: {e}")

# ---------------------------------------------------------------------------
# 5. Footer
# ---------------------------------------------------------------------------
st.caption(
    "AlphaInsights Stress Simulation Module — for exploratory analytics only. "
    "Includes historical drawdowns, VaR/CVaR, and scenario shocks. "
    "Not investment advice."
)
