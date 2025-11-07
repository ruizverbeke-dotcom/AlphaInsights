"""
core/sync_rules.py
------------------
Defines the Concept–Synchronization map for AlphaInsights.
Phase 4.5 foundation — converts the architecture.md document
into a machine-legible schema.

Each synchronization describes:
    - source: Concept initiating the flow
    - target: Concept receiving data
    - mode:   communication type ("internal", "http", "db", "ui")
    - purpose: short natural language description
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass(frozen=True)
class SyncRule:
    source: str
    target: str
    mode: str
    purpose: str

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Canonical Synchronization Map (v1)
# --------------------------------------------------------------------------- #

SYNC_RULES: List[SyncRule] = [
    SyncRule("UIFrontend", "BackendAPI", "http", "Send optimizer requests (CVaR, Sharpe)"),
    SyncRule("BackendAPI", "AnalyticsEngine", "internal", "Run mathematical computations"),
    SyncRule("AnalyticsEngine", "DataLoader", "internal", "Fetch & clean price data"),
    SyncRule("AnalyticsEngine", "ProfileManager", "db", "Query user constraints for optimization"),
    SyncRule("ProfileManager", "Database", "orm", "Persist and retrieve profiles"),
    SyncRule("UIFrontend", "ProfileManager", "ui", "Display and update user profile data"),
    SyncRule("BackendAPI", "SystemCore", "config", "Access global metadata & environment"),
    SyncRule("SystemCore", "Supabase", "cloud", "Future persistence and sync layer"),
]


def list_sync_rules(as_dicts: bool = True):
    """Return synchronization rules as list of dicts or objects."""
    return [r.as_dict() for r in SYNC_RULES] if as_dicts else SYNC_RULES


def describe_sync_map() -> str:
    """Return a readable multi-line summary of all synchronizations."""
    lines = ["🔄 AlphaInsights Synchronization Map (Phase 4.5)\n"]
    for rule in SYNC_RULES:
        lines.append(f"{rule.source:15s} → {rule.target:15s} [{rule.mode}] — {rule.purpose}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test for standalone inspection
    print(describe_sync_map())
