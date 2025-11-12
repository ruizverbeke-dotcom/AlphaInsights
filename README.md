# 🧭 AlphaInsights — Technical Steering Memo (Phase 7.4+ Direction Adjustment)

**Date:** 2025-11-12  
**Author:** Ruïz Verbeke  
**Context:** Strategic addendum to the Master Technical Brief (2025-11-11)  
**Branch:** phase7.3-valuation-signals → upcoming phase7.4-supabase-cache  

---

## 1️⃣ Purpose of This Memo
This short addendum introduces:
- A set of new design and frontend tools worth evaluating for UI refinement, and  
- A conceptual expansion: the **“AI-Curated Bundles”** system that personalizes analytics access based on user profiles.  

These should **not** interrupt the current backend or Supabase caching work, but rather inform UI evolution, UX planning, and Phase 7.6–8.0 design choices.

---

## 2️⃣ New Tools — Evaluation Summary

| Tool | Type | Potential Use | Priority |
|------|------|----------------|-----------|
| shadcn/ui | Component system (Tailwind + Radix) | Continue using as core UI kit for all React/Streamlit hybrid dashboards. | 🟢 Core |
| ReactBits | Lightweight React utilities | Can accelerate component creation (menus, loaders, popovers) during UI unification. | 🟡 Optional |
| Rive | Motion design / interactive SVGs | For future MVP polish (animated loaders, transitions, or AI assistant avatar). | ⚪ Later |
| 21st.dev | AI-assisted component generator | Could speed up UI prototyping for new dashboards. Not a runtime dependency. | 🟡 Optional |
| barba.js | SPA transition engine | Not relevant to Streamlit; consider only if front-end migrates to full React. | 🔴 Defer |

**Summary**
- Keep **shadcn** as the UI foundation.  
- Experiment with **ReactBits** + **Rive** post-Phase 7.5 for UX polish.  
- Revisit the others once the MVP front-end architecture stabilizes.

---

## 3️⃣ Concept Proposal — “AI-Curated Bundles”

### 3.1 Concept
Introduce an AI-guided recommendation layer that suggests tool and analysis *bundles* based on profile, goals, and sophistication level.

Examples:
- 🧩 Beginner Bundle → Sharpe Dashboard + Simplified Portfolio Summary  
- ⚙️ Professional Bundle → CVaR Optimizer + Stress Testing + Factor Analysis  
- 🧠 Strategist Bundle → Valuation Signals + Insights History + Regime Detection  

Transforms AlphaInsights from a “menu of tools” into a context-aware analytics assistant.

### 3.2 Why It Fits
- Leverages existing profile manager + metadata.  
- Aligns with planned AI/agent roadmap (Phase 8 +).  
- Adds UX value — adapts to user skill and intent.  
- Can start rule-based, later extended with AI reasoning.

### 3.3 Implementation Sketch
- Add tool metadata JSON (id, tags, complexity, dependencies).  
- Extend profile schema with experience_level, risk_tolerance, interests.  
- Create /recommend/bundles endpoint (rule-based scoring initially).  
- Streamlit “Suggested Tools” sidebar → lists recommended bundles.  
- **Phase 8:** upgrade to LLM-driven suggestions using stored profile + usage data.

### 3.4 Roadmap Slot
| Phase | Name | Deliverable |
|--------|------|-------------|
| 7.6 | Tool Metadata + Profile Expansion | Define JSON metadata & extended profile model |
| 7.7 | Bundle Recommendation MVP | Backend /recommend/bundles + rule engine |
| 7.8 | AI-Driven Bundles | Integrate LLM reasoning via API |
| 8.0 | MVP Launch | Include personalized onboarding bundle picker |

---

## 4️⃣ Strategic Impact
| Dimension | Benefit |
|------------|----------|
| User Experience | Reduces cognitive load; guided exploration. |
| Architecture Fit | Uses modular structure; no heavy refactor. |
| AI Alignment | Natural entry point for AI agents. |
| Differentiation | Sets AlphaInsights apart from static dashboards. |

---

## 5️⃣ Actionable Next Steps for Coding Chat
- Track this memo for post-Phase 7.4 planning.  
- Maintain shadcn/ui as core; note optional ReactBits & Rive for UX upgrades.  
- When designing Supabase caching schemas, reserve a JSONB field for tool_metadata → future recommender engine.  
- After Phase 7.4–7.5, open eature/phase7.6-bundles branch to start metadata + profile expansion.

---

## 6️⃣ Closing Summary
AlphaInsights is functionally stable through valuation and optimization analytics.  
The next leap is **personalization** — enabling the system to recommend, not just execute.  
This memo formalizes the vision and path forward without interrupting current deliverables.

**Commit title suggestion:**  
docs: add Phase 7.4+ Steering Memo — New Tools & AI-Curated Bundles Concept
"@ | Out-File -Encoding UTF8 docs\intelligence_logs\Phase_7.4+_Steering_Memo.md

git add docs\intelligence_logs\Phase_7.4+_Steering_Memo.md
git commit -m "Docs: add Phase 7.4+ Steering Memo — New Tools & AI-Curated Bundles Concept"
git push origin phase7.3-valuation-signals


(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> # 1) Confirm the memo file exists locally and show first lines
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights>  = "docs\intelligence_logs\Phase_7.4+_Steering_Memo.md"
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> Write-Host "Exists? " -NoNewline; Test-Path 
Exists? False
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> Write-Host "
--- HEAD of memo ---"

--- HEAD of memo ---
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> Get-Content  -Head 12
Get-Content : Cannot find path 'C:\Users\info\PycharmProjects\AlphaInsights\docs\intelligence_logs\Phase_7.4+_Steering_Memo.md' because it does not exist.
At line:1 char:1
+ Get-Content  -Head 12
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\info\P...teering_Memo.md:String) [Get-Content], ItemNotFoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetContentCommand
 
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> Write-Host "----------------------
"
----------------------

(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> 
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> # 2) Show the last commit with files included
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> git log -1 --stat
commit 38bb0b52bb37c9a96c6b20339bf5edfb3b9ffa89 (HEAD -> phase7.3-valuation-signals, origin/phase7.3-valuation-signals)
Author: Ruïz Verbeke <ruiz.verbeke@gmail.com>
Date:   Wed Nov 12 13:56:07 2025 +0100

    Docs: Added project timeline and architecture overview to README

 README.md | 66 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 66 insertions(+)
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> 
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> # 3) Confirm branch and upstream
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> git rev-parse --abbrev-ref HEAD
phase7.3-valuation-signals
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> git remote -v
origin  https://github.com/ruizverbeke-dotcom/AlphaInsights.git (fetch)
origin  https://github.com/ruizverbeke-dotcom/AlphaInsights.git (push)
(.venv) PS C:\Users\info\PycharmProjects\AlphaInsights> # Create README if missing
if (!(Test-Path README.md)) {
@"
# AlphaInsights

A modular financial intelligence stack:
- **Streamlit** (UI)
- **FastAPI** (backend)
- **Supabase** (cloud logging & storage)
- **yfinance** (market data)

