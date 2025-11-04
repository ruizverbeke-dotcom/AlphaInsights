\# AlphaInsights — Analytics Core Capsule  

\*\*Branch:\*\* analytics-module  

\*\*Last Updated:\*\* 2025-11-04  



---



\## 🎯 Purpose

This capsule defines the analytical foundation of AlphaInsights — the metrics, methodologies, and safeguards that power portfolio evaluation and optimization.



The Analytics Core ensures that every quantitative insight is \*\*explainable, auditable, and educational\*\*.



---



\## ⚙️ Core Metrics Framework



| Metric | Purpose | Formula | File |

|---------|----------|----------|------|

| \*\*Sharpe Ratio\*\* | Measures risk-adjusted return relative to volatility. | (Rp − Rf) / σp | `/analytics/sharpe\_ratio.py` |

| \*\*Sortino Ratio\*\* | Focuses on downside deviation, penalizing harmful volatility. | (Rp − Rf) / σdown | `/analytics/sortino\_ratio.py` |

| \*\*Treynor Ratio\*\* | Evaluates returns relative to systematic (market) risk. | (Rp − Rf) / βp | `/analytics/treynor\_ratio.py` |

| \*\*Alpha / Beta\*\* | Decomposes performance vs benchmark into active return and sensitivity. | Regression-based | `/analytics/alpha\_beta.py` |

| \*\*Optimization Module\*\* | Determines efficient portfolio weights under constraints. | Mean-variance + CVaR (upcoming) | `/analytics/optimization.py` |



---



\## 🧩 Structural Safeguards



1\. \*\*1-D Safety Rule\*\*  

&nbsp;  ```python

&nbsp;  series = series.squeeze("columns") if isinstance(series, pd.DataFrame) else series

&nbsp;  series = pd.Series(np.ravel(series), index=series.index\[:len(np.ravel(series))])



