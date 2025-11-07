# database/queries.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

from .models import UserProfile
from .db_setup import get_engine

# ---------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------
_engine = get_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)

# ---------------------------------------------------------------------
# CRUD operations for UserProfile
# ---------------------------------------------------------------------
def create_profile(
    *,
    name: Optional[str],
    risk_score: int,
    asset_class_prefs: Dict[str, Any],
    sector_prefs: Dict[str, Any],
    constraints: Dict[str, Any],
) -> UserProfile:
    """
    Create and persist a new UserProfile.

    Args:
        name: Profile name (optional)
        risk_score: Integer 1–10 risk tolerance score
        asset_class_prefs: Dict of asset class preferences
        sector_prefs: Dict of sector preferences
        constraints: Dict of portfolio constraints

    Returns:
        The persisted UserProfile instance
    """
    with SessionLocal() as session:
        profile = UserProfile(
            name=name,
            risk_score=risk_score,
            asset_class_prefs=asset_class_prefs or {},
            sector_prefs=sector_prefs or {},
            constraints=constraints or {},
        )
        session.add(profile)
        session.commit()
        session.refresh(profile)
        return profile


def get_profiles() -> List[UserProfile]:
    """Return all stored user profiles."""
    with SessionLocal() as session:
        return list(session.scalars(select(UserProfile)).all())


def get_profile(profile_id: int) -> Optional[UserProfile]:
    """Return a single profile by ID."""
    with SessionLocal() as session:
        return session.get(UserProfile, profile_id)


def update_profile(
    profile_id: int,
    *,
    name: Optional[str] = None,
    risk_score: Optional[int] = None,
    asset_class_prefs: Optional[Dict[str, Any]] = None,
    sector_prefs: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
) -> Optional[UserProfile]:
    """
    Update fields on an existing profile.
    Returns updated profile or None if not found.
    """
    with SessionLocal() as session:
        profile = session.get(UserProfile, profile_id)
        if not profile:
            return None
        if name is not None:
            profile.name = name
        if risk_score is not None:
            profile.risk_score = risk_score
        if asset_class_prefs is not None:
            profile.asset_class_prefs = asset_class_prefs
        if sector_prefs is not None:
            profile.sector_prefs = sector_prefs
        if constraints is not None:
            profile.constraints = constraints
        session.commit()
        session.refresh(profile)
        return profile


def delete_profile(profile_id: int) -> bool:
    """Delete profile by ID. Returns True if deleted."""
    with SessionLocal() as session:
        profile = session.get(UserProfile, profile_id)
        if not profile:
            return False
        session.delete(profile)
        session.commit()
        return True
