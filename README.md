# 🧠 AlphaInsights
**An AI-driven quantitative analytics platform integrating FastAPI, Supabase, and Streamlit.**  
Designed and developed by **Ruïz Verbeke**.

---

## ⚙️ Overview
AlphaInsights is a modular intelligence framework for financial research and portfolio analytics.  
It combines optimization, valuation, and data-logging components into a cohesive, explainable pipeline.

\\\
Streamlit UI  ⇄  FastAPI Backend  ⇄  Supabase Cloud  ⇄  yfinance API
\\\

---

## 🚀 Features
- **Optimization Engine:** Sharpe & CVaR (SLSQP solver)
- **Valuation Intelligence:** Peer-relative scoring (valuation, quality, payout)
- **Cloud Logging:** Integrated with Supabase for persistent data storage
- **UI Dashboards:** Streamlit-based analytics with fallback resilience
- **System Logging:** Auto system snapshot on backend startup

---

## 🧭 Project Timeline & Architecture Overview

| Phase | Dates | Highlights |
|--------|--------|-------------|
| 6.8–6.9 | 2025-11-06 → 2025-11-08 | Supabase integration, backend validation, logging pipeline |
| 7.2 | 2025-11-09 | Backend introspection, automatic system snapshots |
| 7.3 | 2025-11-09 → 2025-11-11 | Valuation signals backend, Streamlit integration, peer-relative scoring |

📂 **Documentation**
- [Phase 6.8–6.9 — Supabase Integration & Validation](docs/progress_logs/Phase_6.8_Supabase_Integration_and_Validation.md)
- [Phase 7.3 — Valuation Signals Integration](docs/progress_logs/Phase_7.3_Valuation_Signals_Integration.md)
- [System Chronicle (Phase 6 → 7)](docs/intelligence_logs/System_Chronicle_Phase6_to7.md)

---

## 🧱 Repository Structure
\\\
backend/          # FastAPI app & routes
core/             # shared logic (config, helpers)
supabase_client/  # cloud integration layer
ui/               # Streamlit dashboards
docs/             # logs, progress reports, memos
\\\

---

## 🧩 Next Milestones
| Phase | Goal | Description |
|--------|------|-------------|
| 7.4 | Supabase caching | Implement caching layer for /valuation/signals |
| 7.5 | Unified Insights Dashboard | Combine valuation + optimization |
| 8.0 | MVP release | Gentrepreneur showcase deployment |

---

## 👨‍💻 Author
**Ruïz Verbeke**  
Founder & Lead Developer — AlphaInsights  
📎 [GitHub Profile](https://github.com/ruizverbeke-dotcom)

---
