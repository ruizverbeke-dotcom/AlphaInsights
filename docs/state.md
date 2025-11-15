# AlphaInsights – state.md
Last Updated: 2025-11-15

---

# 1. Current High-Level Phase
**Phase 7.4 — Supabase Caching Layer (Valuation Signals Cache)**

Phase 7.3 (Valuation & Signal Integration) is COMPLETE and committed on branch:
`phase7.3-valuation-signals`.

The system is now stable and feature-complete for:
- Sharpe optimizer
- CVaR optimizer
- Stress testing
- Symbol resolver
- Full valuation signals engine
- Streamlit dashboards (optimizers, valuation, logs)
- Supabase logging + system snapshots

All architecture memos, design notes, and strategic direction documents for 7.4+ are written and uploaded.

---

# 2. Last Completed Coding Step
**Completed:** Phase 7.3 — Valuation Signals Integration  
**Branch:** `phase7.3-valuation-signals`

### Backend
- Added `/valuation/signals` endpoint in `backend/routes/valuation.py`.
- Built peer-relative valuation scoring:
  - valuation_score
  - quality_score
  - payout_score
  - risk_flags
- Integrated with `symbol_resolver` for human-friendly inputs.
- JSON schema is clean, agent-friendly, and used internally.

### Streamlit UI (`ui/pages/valuation_dashboard.py`)
- Full visualization of valuation metrics:
  - Table of fundamentals + computed scores
  - Valuation score bar chart
  - Quality vs Valuation scatter
  - Risk flag section
- Robust fallback logic:
  1. Backend `/valuation/signals`
  2. Backend `/valuation/summary`
  3. Local yfinance

### System Introspection
- Startup snapshots generated at backend boot:
  `backend/logs/system_status_*.txt`
- Extended /health, /status/summary, /valuation/health endpoints.

### Documentation
- Committed:
  - `phase7.3_valuation_signals_log.md`
  - Requirements snapshot
  - Strategic memos for Phase 7.4–7.6

---

# 3. Next Coding Step (Authoritative)
**Begin Phase 7.4 — Supabase Caching Layer for Valuation Signals**  
**New Branch:** `phase7.4-supabase-cache`

---

## Step 7.4.1 — Implement Supabase Valuation Cache (INITIAL STEP)

### 🎯 Goal
Add a caching layer for `/valuation/signals` so repeated requests with the same set of tickers can be served from Supabase instead of recomputing everything live.

This improves:
- Latency  
- Stability  
- Efficiency  
- Rate-limit resilience  

### 🔧 Scope (IN)
1. **Supabase Table Creation:** `valuation_cache`
   With columns:
   - `id` (uuid, PK)
   - `symbol_set_key` (text, canonicalized list or hashed version)
   - `payload` (jsonb)
   - `computed_at` (timestamptz)
   - `ttl_seconds` (integer) → optional v2

2. **Helper Module:** `supabase_client/cache.py`
   Functions:
   - `get_cached_valuation(symbols: list[str]) -> dict | None`
   - `set_cached_valuation(symbols: list[str], payload: dict) -> None`
   - Internal:
     - canonicalize + sort symbol list
     - generate `symbol_set_key`

3. **Backend Integration (valuation router)**
   In `backend/routes/valuation.py`:
   - On request:
     - Resolve & normalize tickers
     - Check Supabase cache
     - If hit → return cached response (`meta.cache_hit = true`)
     - If miss → compute → store → return (`meta.cache_hit = false`)

4. **Minimal Testing**
   - A pytest OR simple manual script:
     1. Call `/valuation/signals` twice
     2. Confirm 2nd call hits cache

### 🧱 Scope (OUT)
- UI updates  
- TTL tuning  
- Cache hits analytics  
- Integration into optimizers  
- Bundles / insights layer  
- Any Phase 7.6 or later features  

### 📁 Files to Touch
- `supabase_client/cache.py` (new)
- `backend/routes/valuation.py` (update)
- Supabase SQL schema (SQL or via UI)

### 📌 Acceptance Criteria
- Cache table created
- Helpers implemented + stable
- `/valuation/signals` uses cache
- Cache hit/miss detection works
- No breakage to existing logic
- Code pushed on `phase7.4-supabase-cache`

---

# 4. Open Questions / Pending Decisions

### Caching Layer
- Key representation: `symbol_set_key` as text vs hashed digest
- TTL for valuation: 15 min? 1 hour? 1 trading day?
- Add optional `hit_count` table for analytics in later phase

### Branch Discipline
- Continue naming branches as `phase7.x-*`?
- When to create the `phase8.0-mvp-release` freeze branch?

### Bundles (Phase 7.6+)
- Schema: tool-sets, presets, explanations
- Connection to user profile and Supabase usage logs

### Context Intelligence Layer (Phase 7.6)
- External narrative provider TBD
- Router design TBD

---

# 5. Recent Changes (Log)
- 2025-11-11 → Completed valuation signals backend & UI  
- 2025-11-11 → Added startup snapshot system  
- 2025-11-12 → Completed Phase 7.3 valuation documentation  
- 2025-11-13 → Strategic memos added (Phase 7.4–7.6)  
- 2025-11-15 → Planned Supabase caching layer  
- 2025-11-15 → Setup new ChatGPT Project OS + rule documents  
