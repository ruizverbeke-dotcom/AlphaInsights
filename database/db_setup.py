# database/db_setup.py
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
import os

# ---------------------------------------------------------------------
# Database path setup
# ---------------------------------------------------------------------
# Database will live inside the AlphaInsights/database directory
DB_FILENAME = "alphainsights.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILENAME)
DB_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------
# Base class for ORM models
# ---------------------------------------------------------------------
Base = declarative_base()

# ---------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------
def get_engine():
    """
    Return a SQLAlchemy Engine connected to the local SQLite database.

    Example:
        engine = get_engine()
    """
    return create_engine(DB_URL, echo=False, future=True)