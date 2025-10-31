# tests/test_smoke_database.py
"""
Quick smoke test for UserProfile CRUD operations.
Run once to confirm DB integration works.
"""
from database.queries import (
    create_profile,
    get_profiles,
    get_profile,
    update_profile,
    delete_profile,
)

def run_smoke_test():
    print("▶ Creating profile...")
    p = create_profile(
        name="Test User",
        risk_score=5,
        asset_class_prefs={"equities": 0.6, "bonds": 0.4},
        sector_prefs={"tech": 0.5, "healthcare": 0.5},
        constraints={"max_weight": 0.3},
    )
    print(f"Created profile ID: {p.id}")

    print("▶ Listing profiles...")
    profiles = get_profiles()
    print(f"Total profiles: {len(profiles)}")

    print("▶ Fetching single profile...")
    fetched = get_profile(p.id)
    print(f"Fetched name: {fetched.name}, risk_score: {fetched.risk_score}")

    print("▶ Updating profile...")
    updated = update_profile(p.id, risk_score=7)
    print(f"Updated risk_score: {updated.risk_score}")

    print("▶ Deleting profile...")
    deleted = delete_profile(p.id)
    print(f"Deleted: {deleted}")

    print("✅ Smoke test completed successfully.")

if __name__ == "__main__":
    run_smoke_test()
