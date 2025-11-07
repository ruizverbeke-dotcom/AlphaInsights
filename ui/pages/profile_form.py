# ===============================================================
# Profile Manager — AlphaInsights (FINAL ATOMIC VERSION)
# ===============================================================
# Features:
#   • Profile name first
#   • Additive Quick Picks (Core, Tilt, Growth, Defensive, Sustainable)
#   • "All Equity Sectors" merges instead of overwriting
#   • Atomic state updates → no flicker or loss
#   • Full CRUD with rerun-safe logic
# ===============================================================

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
from database.queries import create_profile, get_profiles, update_profile, delete_profile

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="User Profile Manager — AlphaInsights", page_icon="👤", layout="wide")
st.title("👤 User Profile Manager")
st.caption("Define your investment profile — risk level, horizon, asset mix, sectors, and diversification constraints.")

# -----------------------------
# Constants
# -----------------------------
ASSET_CLASSES = [
    "Stocks", "ETFs", "Bonds (Government)", "Bonds (Corporate)", "High-Yield Bonds",
    "Commodities (Gold, Oil, etc.)", "Real Estate / REITs", "Crypto / Digital Assets",
    "Alternatives (Private Equity, Hedge Funds)", "Cash / Money Market",
]
SECTORS = [
    "Technology", "Healthcare", "Financials", "Energy", "Industrials",
    "Consumer Discretionary", "Consumer Staples", "Materials",
    "Real Estate", "Communication Services", "Utilities", "Renewables",
]
REGIONS = [
    "Global Diversified", "North America", "Europe", "Asia-Pacific",
    "Emerging Markets", "Latin America", "Middle East", "Africa",
]
EXCLUSIONS = [
    "Tobacco", "Weapons", "Gambling", "Fossil Fuels", "Adult Content",
    "Alcohol", "Defense", "Mining", "Oil", "Coal",
]

# -----------------------------
# Helpers
# -----------------------------
def equal_weights(labels):
    """Return equal weights summing to 1.0 rounded to 2dp."""
    if not labels:
        return {}
    n = len(labels)
    base = round(1 / n, 2)
    weights = {lbl: base for lbl in labels}
    diff = round(1.0 - sum(weights.values()), 2)
    if labels:
        weights[labels[-1]] = round(weights[labels[-1]] + diff, 2)
    return weights

def totals_ok(w):
    """Check if weights sum ≈ 1.0."""
    return (not w) or (0.95 <= sum(w.values()) <= 1.05)

def show_progress(label, weights):
    """Show total completion bar for weights."""
    total = sum(weights.values()) if weights else 0.0
    st.progress(min(total, 1.0))
    st.caption(f"**{label} total:** {total:.2f} (should ≈ 1.0)")

# -----------------------------
# Session State
# -----------------------------
defaults = {
    "profile_name": "",
    "assets_sel": [],
    "sectors_sel": [],
    "exclude_ms": [],
    "quick_pick_action": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===============================================================
# Quick Pick Processing (atomic & safe)
# ===============================================================
if st.session_state.quick_pick_action:
    act = st.session_state.quick_pick_action
    updates = {
        "assets_sel": set(st.session_state.assets_sel),
        "sectors_sel": set(st.session_state.sectors_sel),
        "exclude_ms": set(st.session_state.exclude_ms),
    }

    if act == "core":
        updates["assets_sel"] |= {"Stocks", "Bonds (Government)", "Cash / Money Market"}
    elif act == "tilt":
        updates["assets_sel"] |= {"ETFs", "Real Estate / REITs", "Cash / Money Market"}
    elif act == "growth":
        updates["assets_sel"] |= {"ETFs"}
        updates["sectors_sel"] |= {"Technology", "Consumer Discretionary"}
    elif act == "defensive":
        updates["assets_sel"] |= {"Bonds (Government)", "Cash / Money Market"}
        updates["sectors_sel"] |= {"Utilities", "Consumer Staples"}
    elif act == "sustainable":
        updates["assets_sel"] |= {"ETFs", "Stocks"}
        updates["sectors_sel"] |= {"Renewables", "Technology"}
        updates["exclude_ms"] |= {"Fossil Fuels", "Oil", "Coal"}
    elif act == "all_sectors":
        updates["sectors_sel"] |= set(SECTORS)
    elif act == "clear":
        updates = {"assets_sel": set(), "sectors_sel": set(), "exclude_ms": set()}

    # Apply all changes atomically
    st.session_state.assets_sel = sorted(updates["assets_sel"])
    st.session_state.sectors_sel = sorted(updates["sectors_sel"])
    st.session_state.exclude_ms = sorted(updates["exclude_ms"])

    st.session_state.quick_pick_action = None
    st.rerun()

# ===============================================================
# Create / Update Profile
# ===============================================================
st.subheader("🧱 Create or Update Profile")

# --- 1) Profile Name ---
st.text_input("Profile Name", placeholder="e.g., Growth & Innovation Portfolio", key="profile_name")

# --- 2) Investment Focus ---
st.markdown("### ⚙️ Investment Focus")

left, right = st.columns([2, 1])
with left:
    colA, colB = st.columns(2)
    with colA:
        st.multiselect(
            "Select Asset Classes",
            ASSET_CLASSES,
            key="assets_sel",
            help="Pick one or more asset types. Weight inputs will appear below immediately.",
        )
    with colB:
        st.multiselect(
            "Select Sectors (optional)",
            SECTORS,
            key="sectors_sel",
            help="Pick sectors to tilt toward. Weight inputs will appear below immediately.",
        )

with right:
    st.caption("Quick Picks")
    if st.button("Core Assets (Stocks, Bonds, Cash)"):
        st.session_state.quick_pick_action = "core"
        st.rerun()
    if st.button("Passive Tilt (ETFs, REITs)"):
        st.session_state.quick_pick_action = "tilt"
        st.rerun()
    if st.button("Growth Tilt"):
        st.session_state.quick_pick_action = "growth"
        st.rerun()
    if st.button("Defensive Mix"):
        st.session_state.quick_pick_action = "defensive"
        st.rerun()
    if st.button("Sustainable Focus"):
        st.session_state.quick_pick_action = "sustainable"
        st.rerun()
    if st.button("All Equity Sectors"):
        st.session_state.quick_pick_action = "all_sectors"
        st.rerun()
    if st.button("Clear Selections"):
        st.session_state.quick_pick_action = "clear"
        st.rerun()

st.divider()

# ===============================================================
# Main Form
# ===============================================================
with st.form("profile_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        risk = st.slider("Risk Tolerance (1 = Very Conservative, 10 = Very Aggressive)", 1, 10, 5, key="risk_slider")
    with c2:
        horizon = st.number_input("Investment Horizon (years)", min_value=1, max_value=50, value=10, key="horizon_years")

    # --- Asset Weights ---
    st.markdown("### Asset Class Weights")
    asset_weights = {}
    if st.session_state.assets_sel:
        defaults = equal_weights(st.session_state.assets_sel)
        cols = st.columns(min(3, len(st.session_state.assets_sel)))
        for i, a in enumerate(st.session_state.assets_sel):
            col = cols[i % len(cols)]
            with col:
                asset_weights[a] = st.number_input(f"{a}", 0.0, 1.0, defaults[a], 0.01, key=f"w_asset_{a}")
        show_progress("Asset Weights", asset_weights)
    else:
        st.caption("Select asset classes above to define weights.")

    # --- Sector Weights ---
    st.markdown("### Sector Weights (optional)")
    sector_weights = {}
    if st.session_state.sectors_sel:
        defaults = equal_weights(st.session_state.sectors_sel)
        cols = st.columns(min(3, len(st.session_state.sectors_sel)))
        for i, s in enumerate(st.session_state.sectors_sel):
            col = cols[i % len(cols)]
            with col:
                sector_weights[s] = st.number_input(f"{s}", 0.0, 1.0, defaults[s], 0.01, key=f"w_sector_{s}")
        show_progress("Sector Weights", sector_weights)
    else:
        st.caption("Optionally pick sectors above to set tilts.")

    # --- Constraints ---
    st.markdown("### Portfolio Constraints & Preferences")
    col1, col2, col3 = st.columns(3)
    with col1:
        region = st.selectbox("Preferred Region", REGIONS, index=0, key="region_sel")
    with col2:
        custom_region = st.text_input("Custom Region (optional)", placeholder="e.g., Western Europe", key="custom_region")
    with col3:
        max_hold = st.number_input(
            "Max weight per single holding",
            0.0, 1.0, 0.25, 0.01,
            help="Diversification constraint for any single security (e.g., SPY, AAPL).",
            key="max_single_hold",
        )

    excluded = st.multiselect("Exclude sectors or industries", EXCLUSIONS, key="exclude_ms")

    submitted = st.form_submit_button("💾 Save Profile")

# ===============================================================
# Save Logic
# ===============================================================
if submitted:
    name = (st.session_state.profile_name or "").strip()
    if not name:
        st.warning("Please enter a Profile Name.")
    else:
        if not totals_ok(asset_weights):
            st.warning(f"⚠️ Asset weights sum to {sum(asset_weights.values()):.2f} (should ≈ 1.0)")
        if not totals_ok(sector_weights):
            st.warning(f"⚠️ Sector weights sum to {sum(sector_weights.values()):.2f} (should ≈ 1.0)")

        if totals_ok(asset_weights) and totals_ok(sector_weights):
            constraints = {
                "max_weight": st.session_state.max_single_hold,
                "exclude": st.session_state.exclude_ms,
                "preferred_region": st.session_state.custom_region or st.session_state.region_sel,
            }
            data = dict(
                name=name,
                risk_score=st.session_state.risk_slider,
                asset_class_prefs=asset_weights,
                sector_prefs=sector_weights,
                constraints=constraints,
            )
            try:
                active_id = st.session_state.get("active_profile_id")
                if active_id:
                    update_profile(active_id, **data)
                    st.success(f"✅ Profile '{data['name']}' updated.")
                else:
                    new = create_profile(**data)
                    new_id = getattr(new, "id", None) if new is not None else None
                    st.session_state["active_profile_id"] = new_id
                    st.success(f"✅ Profile '{data['name']}' created.")
            except Exception as e:
                st.error(f"Error saving profile: {e}")

st.divider()

# ===============================================================
# Existing Profiles
# ===============================================================
st.subheader("📂 Existing Profiles")
profiles = get_profiles()
if not profiles:
    st.info("No profiles found.")
else:
    for p in profiles:
        with st.expander(f"🗂️ {p.name or 'Unnamed'} (Risk {p.risk_score})"):
            st.json({
                "ID": p.id,
                "Risk Score": p.risk_score,
                "Asset Classes": p.asset_class_prefs,
                "Sectors": p.sector_prefs,
                "Constraints": p.constraints,
                "Created": str(p.created_at),
                "Updated": str(p.updated_at),
            })
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Use in Analysis", key=f"use_{p.id}"):
                    st.session_state["active_profile_id"] = p.id
                    st.success(f"✅ Profile '{p.name}' activated for analysis.")
            with c2:
                if st.button("Delete", key=f"del_{p.id}"):
                    delete_profile(p.id)
                    st.warning(f"🗑️ Profile '{p.name}' deleted.")
                    st.rerun()
