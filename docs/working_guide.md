\# AlphaInsights — Working Guide  

\*\*Purpose:\*\* Help you understand how the Memory Infrastructure works and how to use it day to day.



---



\## 🧠 1. What the Memory System Does



AlphaInsights now has a \*\*self-documenting memory system\*\* — a way for your project to “remember” everything you do.  

Instead of explaining your work to ChatGPT every time, the knowledge now lives \*inside your repository\* as structured Markdown files.



---

\## 🗂️ 2. Overview of the Docs Structure



| Path | Purpose | When You Use It |

|------|----------|-----------------|

| `docs/context\_tracker.json` | Tracks version and structure of the memory system. | Update the version number only when a major docs phase finishes. |

| `docs/ai\_agents\_spec.md` | Blueprint for how future AI agents will manage memory. | Reference only; you don’t edit this. |

| `docs/knowledge\_capsules/` | Permanent documentation of concepts and features. | When a feature or idea is finalized, create a capsule here. |

| `docs/intelligence\_logs/` | Daily logs and memos of what you’re coding or discovering. | Each time you start coding, create a new log here. |

| `docs/innovation\_sparks.md` | “Idea scratchpad” for random or early thoughts. | When you get an idea that might become a feature. |

| `docs/progress\_logs/` | Reports that summarize completed phases or milestones. | When you finish something major (e.g., Phase 1–4). |

| `docs/\_templates/` | Local-only Markdown templates. | Do not commit; copy or generate from them. |

| `tools/new\_memo.ps1` | PowerShell helper that creates a dated memo from a template. | Run it each day or before each coding task. |



---

\## 🧩 3. Daily Workflow (Step by Step)



\### Step 1 — Start a new day or task

Run:

.\\tools\\new\_memo.ps1 2025-11-06 "Continue CVaR Optimizer Development"

This creates a file like:  

`docs/intelligence\_logs/2025-11-06\_grand\_technical\_memo.md`



---



\### Step 2 — Fill in the memo

In that file, write:

\- \*\*Overview:\*\* What you plan to do today  

\- \*\*Observations:\*\* What happens during testing  

\- \*\*Decisions/Actions:\*\* What you’ll change next  

\- \*\*Related Capsules:\*\* Any existing reference docs  



---

\### Step 3 — Code your feature normally

Work in the appropriate Python module (e.g., `/analytics/optimization.py`).



---



\### Step 4 — Capture new ideas

If something new comes to mind:

\- Open `docs/innovation\_sparks.md`  

\- Add an entry with date + short note  



---



\### Step 5 — Finalize work into a capsule

When a concept is stable (e.g., CVaR Optimizer working):



1\. Copy the capsule template:  

&nbsp;  `/docs/\_templates/capsule\_template.md`  

2\. Save it as something like:  

&nbsp;  `docs/knowledge\_capsules/cvar\_optimizer.md`  

3\. Write the final explanation, formulas, and rationale there.



---

\### Step 6 — Mark phase completion

When you finish a big phase, summarize it in `/docs/progress\_logs/`.  

Example: `2025-11-10\_cvar\_phase\_complete.md`



---



\### Step 7 — Update the tracker

Open `docs/context\_tracker.json`  

\- Increment the version number (e.g., `"version": "1.2.0"`)  

\- Add your new commit title under `"recent\_commits"`



Commit that update:

git add docs\\context\_tracker.json

git commit -m "docs: bump context tracker to v1.2.0 after CVaR phase completion"

---

\## 🧭 4. Using Docs to Resume Work in a New Chat



When you open ChatGPT again:

1\. Copy the contents of your \*\*latest memo\*\* (in `intelligence\_logs/`).  

2\. Paste it in the new chat before asking coding questions — this brings the assistant up to speed.  

3\. Optionally link a capsule if it’s relevant (`analytics\_core.md`, etc.).  



You no longer need huge prompts describing everything — your docs \*are\* the context.



---

\## 🧰 5. Maintenance Rules



\- Filenames must be lowercase.  

\- Every file created must be \*\*complete\*\*, not partial.  

\- Never track the database (`database/alphainsights.db`).  

\- Keep `.gitignore` as is — it already protects temp files.  

\- Update `context\_tracker.json` only after meaningful doc changes.  



---



\*\*In short:\*\*  

\- \*Logs\* = what you’re doing now  

\- \*Capsules\* = what you’ve learned permanently  

\- \*Progress logs\* = checkpoints  

\- \*Tracker\* = index of it all  

\- \*Templates\* = your tools to make new ones  



---



\*\*Maintained by:\*\* Ruiz Verbeke  

\*\*Version:\*\* Working Guide v1.0 (Nov 2025)



