"""
AlphaInsights Core Metadata
---------------------------
Houses global metadata for versioning, phase tracking,
and concept-level registration. This serves as the
"identity layer" under MIT’s Legible Modular Software model.

Each module reads this metadata for synchronized version context.
"""

from datetime import date

__project__ = "AlphaInsights"
__version__ = "4.2.0"
__phase__ = "Phase 4.2 – Backend Synchronization Online"
__maintainer__ = "Ruïz Verbeke"
__updated__ = date.today().isoformat()

CORE_METADATA = {
    "project": __project__,
    "version": __version__,
    "phase": __phase__,
    "maintainer": __maintainer__,
    "updated": __updated__,
    "description": (
        "AlphaInsights is an AI-native quantitative analytics platform "
        "integrating explainable intelligence, modular architecture, "
        "and adaptive optimization pipelines."
    ),
}

def get_metadata() -> dict:
    """Return current system metadata as a dict."""
    return CORE_METADATA
