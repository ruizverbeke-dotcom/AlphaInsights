"""
AlphaInsights Core Concepts Registry
------------------------------------
Defines all major architectural "Concepts" (modules)
in alignment with MIT’s Legible Modular Software model.

Each Concept declares:
- purpose (what it does)
- location (where it lives)
- dependencies (explicit synchronizations)
"""

CONCEPTS = {
    "DataLoader": {
        "location": "analytics/",
        "purpose": "Fetch and preprocess financial time series.",
        "dependencies": [],
    },
    "AnalyticsEngine": {
        "location": "analytics/",
        "purpose": "Compute ratios, optimizations, and portfolio metrics.",
        "dependencies": ["DataLoader"],
    },
    "ProfileManager": {
        "location": "database/",
        "purpose": "Manage user profiles and risk preferences.",
        "dependencies": [],
    },
    "UIFrontend": {
        "location": "ui/pages/",
        "purpose": "Streamlit dashboards and interaction layer.",
        "dependencies": ["AnalyticsEngine", "BackendAPI"],
    },
    "BackendAPI": {
        "location": "backend/",
        "purpose": "FastAPI gateway exposing analytics endpoints.",
        "dependencies": ["AnalyticsEngine", "ProfileManager"],
    },
    "SystemCore": {
        "location": "core/",
        "purpose": "Central metadata and synchronization registry.",
        "dependencies": [],
    },
}

def list_concepts() -> list[str]:
    """Return list of concept names."""
    return list(CONCEPTS.keys())


def describe_concept(name: str) -> dict:
    """Get description of a specific concept."""
    return CONCEPTS.get(name, {"error": "Concept not found."})
