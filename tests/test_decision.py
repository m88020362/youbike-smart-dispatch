# -*- coding: utf-8 -*-
"""Pytest tests for intervention.decide_intervention (decision layer).

Focused coverage of the additive decision layer that sits on top of the Task 5
engine (recommend + user-intent Friend Relay 2.0). Uses small hand-built
§9-shaped snapshots (same _mk helper pattern as tests/test_intervention.py) so
no trained models are needed. Thresholds come from src/config.py.
"""

from __future__ import annotations

from src import config, intervention


# --------------------------------------------------------------------------- #
# §9-shaped prediction dict helper (mirrors test_intervention.py)             #
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


DECISION_KEYS = {
    "station",
    "risk_level",
    "risk_probability",
    "predicted_problem",
    "primary_action",
    "secondary_action",
    "urgency",
    "reason",
    "truck",
    "friend_relay",
}


def _assert_shape(decision):
    """Every decision has the exact keys and a non-empty reason string."""
    assert set(decision.keys()) == DECISION_KEYS
    assert isinstance(decision["reason"], str) and decision["reason"]


# =========================================================================== #
# 1. Low risk -> none                                                         #
# =========================================================================== #
def test_low_risk_yields_none_normal():
    target = _mk("Calm", 25.0500, 121.5500, 10, 10, 0.05, 0.05, "Low")
    decision = intervention.decide_intervention(target, [target])

    _assert_shape(decision)
    assert decision["primary_action"] == "none"
    assert decision["secondary_action"] is None
    assert decision["urgency"] == "normal"
    assert decision["truck"] is None
    assert decision["friend_relay"] is None


# =========================================================================== #
# 2. Medium risk + valid Friend Relay -> primary friend_relay                 #
# =========================================================================== #
def test_medium_risk_with_relay_yields_friend_relay():
    # Target is a Medium-full station; its nearby neighbor is future-SHORTAGE,
    # so a RETURN redirection (return intent) qualifies as a relay partner.
    target = _mk("MedFull", 25.0000, 121.5000, 18, 3, 0.10, 0.45, "Medium")
    neighbor = _mk("N_short", 25.0030, 121.5010, 1, 20, 0.45, 0.05, "Medium")
    snapshot = [target, neighbor]

    decision = intervention.decide_intervention(
        target, snapshot, user_context={"intent": intervention.MISSION_RETURN}
    )

    _assert_shape(decision)
    # Relevant risk sits in the Medium band.
    assert config.RISK_LOW_MAX <= decision["risk_probability"] < config.RISK_HIGH_MIN
    assert decision["primary_action"] == "friend_relay"
    assert decision["secondary_action"] is None
    assert decision["urgency"] == "normal"
    assert decision["friend_relay"] is not None


# =========================================================================== #
# 3. High risk + Truck only (no user intent) -> primary truck, no secondary   #
# =========================================================================== #
def test_high_risk_truck_only_yields_truck():
    # Shortage-high target with a nearby well-stocked donor -> truck available.
    # user_context=None -> no Friend Relay is fabricated.
    target = _mk("HiShort", 25.0000, 121.5000, 1, 20, 0.80, 0.05, "High")
    donor = _mk("Donor", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low")
    snapshot = [target, donor]

    decision = intervention.decide_intervention(target, snapshot, user_context=None)

    _assert_shape(decision)
    assert decision["risk_probability"] >= config.RISK_HIGH_MIN
    assert decision["primary_action"] == "truck"
    assert decision["secondary_action"] is None
    assert decision["truck"] is not None
    assert decision["friend_relay"] is None


# =========================================================================== #
# 4. High risk + Friend Relay only (no truck neighbor) -> primary friend_relay#
# =========================================================================== #
def test_high_risk_relay_only_yields_friend_relay():
    # Shortage-high target. The only nearby neighbor is ALSO shortage-heavy
    # (fewer bikes, higher shortage) so it does NOT qualify as a truck donor,
    # but its shortage_prob is Medium/High and full_prob low, so a RETURN
    # redirection relay qualifies. shortage_prob kept below the urgent 0.85.
    target = _mk("HiShort2", 25.0000, 121.5000, 8, 12, 0.70, 0.05, "High")
    neighbor = _mk("N_short2", 25.0030, 121.5010, 2, 18, 0.55, 0.05, "Medium")
    snapshot = [target, neighbor]

    decision = intervention.decide_intervention(
        target, snapshot, user_context={"intent": intervention.MISSION_RETURN}
    )

    _assert_shape(decision)
    assert decision["truck"] is None  # no qualifying donor
    assert decision["friend_relay"] is not None
    assert decision["primary_action"] == "friend_relay"
    assert decision["secondary_action"] is None


# =========================================================================== #
# 5. High risk + both truck and relay -> truck primary, relay secondary       #
# =========================================================================== #
def test_high_risk_both_yields_truck_primary_relay_secondary():
    # Shortage-high target. Donor (more bikes, lower shortage) -> truck.
    # Second neighbor is future-SHORTAGE -> return relay partner.
    target = _mk("HiShort3", 25.0000, 121.5000, 1, 20, 0.75, 0.05, "High")
    donor = _mk("Donor3", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low")
    relay_nb = _mk("N_short3", 25.0035, 121.5012, 2, 18, 0.55, 0.05, "Medium")
    snapshot = [target, donor, relay_nb]

    decision = intervention.decide_intervention(
        target, snapshot, user_context={"intent": intervention.MISSION_RETURN}
    )

    _assert_shape(decision)
    assert decision["truck"] is not None
    assert decision["friend_relay"] is not None
    assert decision["primary_action"] == "truck"
    assert decision["secondary_action"] == "friend_relay"
    # risk below urgent threshold -> normal urgency
    assert decision["risk_probability"] < config.INTERVENTION_URGENT_PROB
    assert decision["urgency"] == "normal"


# =========================================================================== #
# 6. Very high risk + both -> urgency high, truck primary, relay secondary    #
# =========================================================================== #
def test_urgent_risk_both_yields_high_urgency():
    # shortage_prob >= INTERVENTION_URGENT_PROB (0.85) -> urgent.
    target = _mk("Urgent", 25.0000, 121.5000, 1, 20, 0.92, 0.05, "High")
    donor = _mk("DonorU", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low")
    relay_nb = _mk("N_shortU", 25.0035, 121.5012, 2, 18, 0.55, 0.05, "Medium")
    snapshot = [target, donor, relay_nb]

    decision = intervention.decide_intervention(
        target, snapshot, user_context={"intent": intervention.MISSION_RETURN}
    )

    _assert_shape(decision)
    assert decision["risk_probability"] >= config.INTERVENTION_URGENT_PROB
    assert decision["urgency"] == "high"
    assert decision["primary_action"] == "truck"
    assert decision["secondary_action"] == "friend_relay"


# =========================================================================== #
# 7. No user_context -> friend_relay never fabricated                         #
# =========================================================================== #
def test_no_user_context_never_fabricates_relay():
    # A relay-eligible neighbor exists (future-shortage), but with no user
    # intent the decision layer must NOT create a Friend Relay mission.
    target = _mk("HiShort4", 25.0000, 121.5000, 8, 12, 0.70, 0.05, "High")
    relay_nb = _mk("N_short4", 25.0030, 121.5010, 2, 18, 0.55, 0.05, "Medium")
    snapshot = [target, relay_nb]

    decision = intervention.decide_intervention(target, snapshot, user_context=None)

    _assert_shape(decision)
    assert decision["friend_relay"] is None
    # No truck donor either (neighbor has fewer bikes) -> nothing actionable.
    assert decision["primary_action"] == "none"
