🧠 AlphaInsights — Assistant Context Sync Summary



date: 2025-11-06

branch: analytics-module

context version: v2025.11.06a

purpose: capture the full awareness of the active build assistant (this chat) for continuity between sessions.



1️⃣ Overall Project Understanding



AlphaInsights is a Portfolio Intelligence Platform that integrates analytics, optimization, and AI-personalized portfolio construction.

The app uses Streamlit for UI, SQLAlchemy for persistence, and yfinance for data ingestion.

It computes professional-grade metrics (Sharpe, Sortino, Treynor, Alpha/Beta), supports comparisons, and is evolving toward an AI-assisted optimizer (CVaR next).



Current verified dashboards:



Sharpe Ratio Dashboard: fully stable (1-D safety enforced).



Comparison Dashboard: stable (efficient frontier and comparative Sharpe).



Profile Manager: stable (CRUD + atomic Quick Picks + weights unlocked instantly).



Database layer:



ORM: UserProfile model.



CRUD functions implemented and verified in database/queries.py.



Local SQLite DB (database/alphainsights.db) intentionally untracked via .gitignore.



2️⃣ Memory Infrastructure (Docs-Based “Lab Brain”)



The memory system replaces traditional long prompts.

It’s structured and automated to capture work, promote insights, and verify phases.



Core components:



Folder	Purpose

/docs/intelligence\_logs/	Daily or event-based memos (e.g., Grand Technical Memo).

/docs/knowledge\_capsules/	Finalized knowledge modules (e.g., CVaR theory, AI roadmap).

/docs/progress\_logs/	Phase verification \& summaries.

/docs/archive/	Frozen past versions (immutable).

/docs/context\_tracker.json	The current “state checksum” of project memory.



Automation Tools:



tools/new\_memo.ps1 — creates a new memo with standardized headers.



tools/generate\_phase\_summary.ps1 — creates UTF-8-encoded phase summaries.



Templates:



docs/\_templates/capsule\_template.md



docs/\_templates/intelligence\_memo\_template.md



All file names are lowercase; full-file commits only.



3️⃣ Git \& Repo Hygiene



Active branch: analytics-module



Stable tags:



v1.0-stable → first Sharpe/Comparison release.



v1.1-profiles-mvp → Profile Manager finalized.



Local DB untracked (confirmed red in git status).



Encoding normalized to UTF-8 BOM in PowerShell scripts.



4️⃣ Analytics \& UI Rules



1-D Safety Rule (non-negotiable):



series = series.squeeze("columns") if isinstance(series, pd.DataFrame) else series

series = pd.Series(np.ravel(series), index=series.index\[:len(np.ravel(series))])





Streamlit State Safety:



Never mutate st.session\_state\[key] after widget creation.



Use the flag → st.rerun() pattern for Quick Picks or mass updates.



Plotly \& pandas consistency:



Always flatten (N,1) DataFrames before plotting.



Keep consistent date indexes between assets \& benchmarks.



5️⃣ Current Technical Status



✅ Repo clean and synchronized.

✅ Docs structure verified (folders exist, templates ready).

✅ Git functioning normally.

✅ UTF-8 encodings stable.

✅ PowerShell scripts operational.

⚙️ Next phase pending: implement CVaR optimizer.



6️⃣ Pending \& Planned Tasks

Category	Description

🔹 Coding	Implement CVaR optimizer (analytics/optimization.py) and corresponding Streamlit page (ui/pages/optimizer\_dashboard.py).

🔹 Testing	Add CVaR unit test to tests/test\_analytics.py.

🔹 Documentation	Log each coding day via new\_memo.ps1 → promote knowledge to capsule → summarize via generate\_phase\_summary.ps1.

🔹 ML Expansion	Future: Risk forecasting ML (RandomForest, XGBoost, LSTM) + AI agent orchestration (LangGraph / CrewAI / Supabase memory).

7️⃣ Context Checksum (for cross-chat continuity)

Field	Value

latest\_context\_version	v2025.11.06a

last\_updated	2025-11-06

branch	analytics-module

active\_docs	\[super prompt, grand technical memo, context snapshot]

next\_task	CVaR optimizer implementation

repo\_status	clean

infra\_status	operational

8️⃣ Assistant Protocol (How Future Chats Should Behave)



Load the latest context\_tracker.json.



Read the newest file under /docs/intelligence\_logs/ for active work.



Follow the one-step rule: one atomic instruction per message.



Always provide:



Path



Branch



Action type



PowerShell command (from repo root)



Full file content



Never invent new directories; follow verified repo structure.



After code changes, prompt user to:



git add .

git commit -m "describe change"

git push



9️⃣ Notes on Current Environment



User (Ruiz) currently balancing school, work, and volunteer commitments.



Development temporarily paused until available time allows for CVaR implementation.



Focus remains on logging discoveries, learning ecosystem tools, and conceptual refinement.



The repository’s documentation now provides enough scaffolding to resume cleanly at any future point.



🔚 End of Assistant Sync Summary



version: v2025.11.06a

status: active memory synchronized — ready to resume CVaR implementation when user returns.

