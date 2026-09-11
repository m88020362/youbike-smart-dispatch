# -*- coding: utf-8 -*-
"""Pytest tests for src/intervention.py (R6 + R7) and the R5 probability range.

Locks in the rule-based intervention engine + user-intent Friend Relay 2.0
(design §7, §8, §9, §12; requirements R6, R7) and the R5.2 guarantee that
predicted probabilities fall within [0, 1].

Intervention cases use small hand-built snapshots (shapes identical to
predict.predict_station output, design §9) so they need no trained models.
The probability-range case reuses the real trained artifacts + a real
build_features() row when present, and always asserts the pure clipping
contract. Thresholds/tiers come from src/config.py, never hard-coded.
"""

from __future__ import annotations

import os

import pytest

from src import config, intervention, predict


# --------------------------------------------------------------------------- #
# §9-shaped prediction dict helper                                            #
# --------------------------------------------------------------------------- #
def _mk(station, lat, lon, bikes, docks, sp, fp, risk):
    return {
        "station": station,
        "lat": lat,
        "lon": lon,
        "current_bikes": bikes,
        "current_docks": docks,
        "total_docks": bikes + docks,
        "shortage_prob": sp,
        "full_prob": fp,
        "risk_level": risk,
        "expected_risk_time": "2026-03-30T08:30:00",
    }


REC_KEYS = {
    "station", "current_bikes", "current_docks", "shortage_prob", "full_prob",
    "expected_risk_time", "recommended_action", "options",
}
RELAY_KEYS = {
    "original_station", "partner_station", "mission_type", "predicted_problem",
    "risk_probability", "distance_km", "reward_twd", "mission_text",
}


# =========================================================================== #
# R6 -- intervention recommendation                                           #
# =========================================================================== #
def test_high_shortage_produces_recommendation_with_legal_reward_tier():
    """Shortage-high target -> full §9 recommendation + legal reward tier (R6.1-3)."""
    target = _mk("A_short", 25.0000, 121.5000, 1, 20, 0.80, 0.05, "High")
    snapshot = [
        target,
        _mk("A_donor", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low"),  # ~0.35 km
    ]

    rec = intervention.recommend(target, snapshot)

    assert set(rec.keys()) == REC_KEYS
    assert rec["recommended_action"] in {"A", "B", "C"}
    assert set(rec["options"].keys()) == {"A_truck", "B_relay", "C_none"}
    # Truck option picks the nearby well-stocked donor.
    assert rec["options"]["A_truck"] is not None
    assert rec["options"]["A_truck"]["donor_or_dest"] == "A_donor"
    # Reward tier is always one of the configured tiers.
    assert rec["options"]["B_relay"]["reward_twd"] in set(config.REWARD_TIERS)


def test_high_full_produces_recommendation_with_legal_reward_tier():
    """Full-high target -> destination option + legal reward tier (R6.1, R6.4)."""
    target = _mk("B_full", 25.0100, 121.5100, 22, 0, 0.05, 0.75, "High")
    snapshot = [
        target,
        _mk("B_dest", 25.0120, 121.5110, 2, 19, 0.05, 0.05, "Low"),  # ~0.24 km
    ]

    rec = intervention.recommend(target, snapshot)

    assert rec["options"]["A_truck"] is not None
    assert rec["options"]["A_truck"]["donor_or_dest"] == "B_dest"
    assert rec["options"]["B_relay"]["reward_twd"] in set(config.REWARD_TIERS)


def test_reward_heuristic_only_yields_configured_tiers():
    """Every severity in [0, 1] maps to a value in config.REWARD_TIERS (R7.8)."""
    seen = {intervention._reward_for_severity(s / 20.0) for s in range(21)}
    assert seen.issubset(set(config.REWARD_TIERS))
    assert seen == set(config.REWARD_TIERS)  # spans all tiers


def test_low_risk_is_option_c():
    target = _mk("C_calm", 25.0500, 121.5500, 10, 10, 0.05, 0.05, "Low")
    rec = intervention.recommend(target, [target])
    assert rec["recommended_action"] == "C"
    assert rec["options"]["C_none"] is True
    assert rec["options"]["A_truck"] is None and rec["options"]["B_relay"] is None


def test_missing_required_key_raises_keyerror():
    with pytest.raises(KeyError):
        intervention.recommend({"station": "X"}, [])


# =========================================================================== #
# R6.5 -- Haversine distance                                                  #
# =========================================================================== #
def test_haversine_reference_distance():
    # Taipei 101 -> Taipei Main Station is ~5 km.
    d = intervention.haversine_km(25.0339, 121.5645, 25.0478, 121.5170)
    assert 4.5 < d < 5.5
    assert intervention.haversine_km(25.0, 121.0, 25.0, 121.0) == 0.0


# =========================================================================== #
# R7 -- Friend Relay 2.0 (borrow redirection)                                 #
# =========================================================================== #
def test_recommend_borrow_relay_picks_nearby_future_full_station():
    """Borrow relay selects a nearby future-FULL neighbor within MAX_NEIGHBOR_KM."""
    snapshot = [
        _mk("U_borrow", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full", 25.0030, 121.5010, 20, 1, 0.05, 0.72, "High"),      # ~0.35 km
        _mk("N_far_full", 25.0500, 121.5500, 20, 1, 0.05, 0.90, "High"),  # >1 km
    ]

    mission = intervention.recommend_borrow_relay("U_borrow", snapshot)

    assert mission is not None
    assert RELAY_KEYS.issubset(mission.keys())
    assert mission["partner_station"] == "N_full"          # nearby, not far one
    assert mission["mission_type"] == "borrow"
    assert mission["predicted_problem"] == "full"
    assert mission["distance_km"] <= config.MAX_NEIGHBOR_KM
    assert mission["reward_twd"] in set(config.REWARD_TIERS)


def test_recommend_borrow_relay_excludes_high_shortage_station():
    """Harm guard: never redirect a borrow toward a high-SHORTAGE station (R7.4)."""
    snapshot = [
        _mk("U_borrow", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        # Only nearby candidate is high-shortage -> excluded -> no mission.
        _mk("N_short_hi", 25.0030, 121.5010, 1, 20, 0.85, 0.10, "High"),
    ]
    assert intervention.recommend_borrow_relay("U_borrow", snapshot) is None


# =========================================================================== #
# R7 -- Friend Relay 2.0 (return redirection)                                 #
# =========================================================================== #
def test_recommend_return_relay_picks_nearby_future_shortage_station():
    """Return relay selects a nearby future-SHORTAGE neighbor within MAX_NEIGHBOR_KM."""
    snapshot = [
        _mk("U_return", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_short", 25.0030, 121.5010, 1, 20, 0.70, 0.05, "High"),       # ~0.35 km
        _mk("N_far_short", 25.0500, 121.5500, 1, 20, 0.95, 0.05, "High"),   # >1 km
    ]

    mission = intervention.recommend_return_relay("U_return", snapshot)

    assert mission is not None
    assert RELAY_KEYS.issubset(mission.keys())
    assert mission["partner_station"] == "N_short"
    assert mission["mission_type"] == "return"
    assert mission["predicted_problem"] == "shortage"
    assert mission["distance_km"] <= config.MAX_NEIGHBOR_KM
    assert mission["reward_twd"] in set(config.REWARD_TIERS)


def test_recommend_return_relay_excludes_high_full_station():
    """Harm guard: never redirect a return toward a high-FULL station (R7.4)."""
    snapshot = [
        _mk("U_return", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full_hi", 25.0030, 121.5010, 20, 1, 0.10, 0.85, "High"),
    ]
    assert intervention.recommend_return_relay("U_return", snapshot) is None


def test_relay_returns_none_when_no_suitable_neighbor():
    """No neighbor / only calm neighbors -> no fabricated mission (R7.7)."""
    lone = [_mk("U_alone", 25.0, 121.5, 6, 6, 0.10, 0.10, "Low")]
    assert intervention.recommend_borrow_relay("U_alone", lone) is None
    assert intervention.recommend_return_relay("U_alone", lone) is None

    calm = [
        _mk("U_calm", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_calm", 25.0030, 121.5010, 8, 8, 0.10, 0.10, "Low"),  # nearby, calm
    ]
    assert intervention.recommend_borrow_relay("U_calm", calm) is None
    assert intervention.recommend_return_relay("U_calm", calm) is None


def test_relay_tie_break_prefers_higher_risk_then_nearer():
    """Multiple candidates -> highest relevant risk wins (R7.7)."""
    snapshot = [
        _mk("U_borrow", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full_lo", 25.0030, 121.5010, 20, 1, 0.05, 0.45, "Medium"),  # nearer
        _mk("N_full_hi", 25.0040, 121.5015, 20, 1, 0.05, 0.55, "Medium"),  # riskier
    ]
    mission = intervention.recommend_borrow_relay("U_borrow", snapshot)
    assert mission is not None
    assert mission["partner_station"] == "N_full_hi"  # highest full_prob wins


# =========================================================================== #
# R5.2 -- predicted probabilities fall within [0, 1]                          #
# =========================================================================== #
def test_clip_proba_enforces_unit_interval():
    """Pure contract: probabilities are always clamped into [0, 1] (R5.2)."""
    assert predict._clip_proba(-0.4) == 0.0
    assert predict._clip_proba(1.7) == 1.0
    assert predict._clip_proba(0.5) == 0.5


def test_real_model_predict_proba_in_unit_interval():
    """A real prediction over trained models yields probs within [0, 1] (R5.2)."""
    from src import data_loader, features

    paths = [
        os.path.join(config.MODELS_DIR, "shortage_xgb.json"),
        os.path.join(config.MODELS_DIR, "full_xgb.json"),
        os.path.join(config.MODELS_DIR, "feature_meta.json"),
    ]
    if not all(os.path.exists(p) for p in paths):
        pytest.skip("trained model artifacts not present")
    if not os.path.exists(config.DEFAULT_CSV):
        pytest.skip("representative CSV not present")

    artifacts = predict.load_artifacts()
    df = data_loader.load_clean()
    bundle = features.build_features(features.build_targets(df))
    if len(bundle.X) == 0:
        pytest.skip("no target rows built from CSV")

    # Predict on a small batch of real feature rows.
    for idx in range(min(5, len(bundle.X))):
        feature_row = bundle.X.iloc[idx]
        ts = bundle.timestamp.iloc[idx]
        meta = {
            "station": "(test)",
            "lat": 0.0,
            "lon": 0.0,
            "current_bikes": int(feature_row[config.COL_AVAILABLE_BIKES]),
            "current_docks": int(feature_row[config.COL_AVAILABLE_DOCKS]),
            "total_docks": int(feature_row[config.COL_TOTAL_DOCKS]),
        }
        result = predict.predict_station(feature_row, meta, ts, artifacts=artifacts)
        assert 0.0 <= result["shortage_prob"] <= 1.0
        assert 0.0 <= result["full_prob"] <= 1.0
        assert result["risk_level"] in {"Low", "Medium", "High"}
