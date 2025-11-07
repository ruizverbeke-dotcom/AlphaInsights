# database/models.py
from sqlalchemy import Column, Integer, String, JSON, DateTime, func
from sqlalchemy.orm import declarative_base
from datetime import datetime

# Important: must match Base from db_setup.py
from .db_setup import Base

class UserProfile(Base):
    """SQLAlchemy ORM model for storing user profiles."""
    __tablename__ = "user_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=True)
    risk_score = Column(Integer, nullable=False)
    asset_class_prefs = Column(JSON, nullable=False, default={})
    sector_prefs = Column(JSON, nullable=False, default={})
    constraints = Column(JSON, nullable=False, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def __repr__(self):
        return f"<UserProfile(id={self.id}, name={self.name}, risk_score={self.risk_score})>"
