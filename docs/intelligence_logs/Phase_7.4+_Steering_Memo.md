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
| shadcn/ui | Component system (Tailwind + Radix) | Core UI kit for hybrid dashboards | 🟢 Core |
| ReactBits | Lightweight React utilities | Menus, loaders, popovers | 🟡 Optional |
| Rive | Motion / interactive SVGs | Animated loaders, transitions | ⚪ Later |
| 21st.dev | AI-assisted component generator | Fast UI prototyping | 🟡 Optional |
| barba.js | SPA transition engine | Only if full React front-end | 🔴 Defer |

---

## 3️⃣ Concept Proposal — “AI-Curated Bundles”
(Full section content here — same as before)
