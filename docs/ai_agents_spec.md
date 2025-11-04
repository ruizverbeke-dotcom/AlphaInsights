\# AlphaInsights — AI Agents Specification

\*\*Branch:\*\* analytics-module  

\*\*Last Updated:\*\* 2025-11-04  



---



\## 🧠 Purpose

This document defines the architecture, scope, and behavior of the \*\*AI agents\*\* powering the AlphaInsights memory and intelligence layer.



The agents act as the connective tissue between analytics logic, portfolio insights, and the evolving knowledge base in `/docs/`.



---



\## ⚙️ Agent Classes



| Agent Name | Domain | Core Function | Output Target |

|-------------|---------|----------------|----------------|

| \*\*ContextAgent\*\* | System Memory | Maintains and updates `context\_tracker.json`; enforces doc integrity. | `/docs/context\_tracker.json` |

| \*\*AnalyticsAgent\*\* | Portfolio Analytics | Runs and explains metrics (Sharpe, Sortino, Treynor, Alpha/Beta). | `/analytics/` + `/docs/knowledge\_capsules/analytics\_core.md` |

| \*\*KnowledgeAgent\*\* | Knowledge Capsules | Converts analytical results into permanent, explainable insights. | `/docs/knowledge\_capsules/` |

| \*\*InnovationAgent\*\* | R\&D Intelligence | Monitors discoveries, patterns, and hypotheses across logs. | `/docs/intelligence\_logs/` |

| \*\*ArchivistAgent\*\* | Lifecycle Management | Archives outdated memos and keeps repo structure consistent. | `/docs/archive/` |



---



\## 🔄 Communication Protocol



1\. \*\*ContextAgent\*\* triggers synchronization checks on every documentation update.  

2\. \*\*AnalyticsAgent\*\* writes standardized metric summaries to the relevant capsules.  

3\. \*\*KnowledgeAgent\*\* converts summaries to Markdown knowledge capsules with versioned metadata.  

4\. \*\*InnovationAgent\*\* logs experimental or emergent ideas in `intelligence\_logs/`.  

5\. \*\*ArchivistAgent\*\* performs scheduled pruning, moving old items to `/archive/`.



---



\## 🧩 Data Contracts



Each agent must:

\- Output \*\*full files only\*\* — never partial diffs.

\- Adhere to \*\*lowercase filenames\*\*.

\- Include an ISO timestamp in metadata headers.

\- Update `context\_tracker.json` when new capsules are created.



---



\## 🧭 Roadmap Integration



The `ai\_agents\_spec.md` serves as the schema for Phase 3 and 4 automation.  

When the `KnowledgeAgent` is live, it will auto-generate capsule metadata and report changes to the `context\_tracker.json`.



---



\*\*Next Phase:\*\* Initialize `knowledge\_capsules/vision\_and\_philosophy.md`.



