# -*- coding: utf-8 -*-
"""Intervention engine + Friend Relay 2.0 (Task 5, R6 + R7).

Given one target station's 30-minute risk prediction (design §9, produced by
predict.predict_station) plus a whole-station current snapshot, this module
produces a rule-based, structured recommendation (design §7, §9) and, when an
imbalance is forecast, a predictive Friend Relay 2.0 incentive mission card
(design §8).

It is deliberately a *simple heuristic*, not an optimizer:
  * distance uses a pure-numpy Haversine (great-circle) formula (R6.5);
  * shortage-high -> pick a nearby donor station (more bikes, own shortage risk
    lower) within config.MAX_NEIGHBOR_KM (R6.3);
  * full-high -> pick a nearby alternative destination (more return spaces, own
    full risk lower) within config.MAX_NEIGHBOR_KM (R6.4);
  * Low risk -> Option C (No Intervention);
  * it does NOT solve a real vehicle routing problem (R6.6) and does NOT build a
    complex incentive optimization model (R7.5).

Friend Relay 2.0 (design §8, R7) -- USER-INTENT-CENTERED (two MVP scenarios):
  * it starts from what the *user* already intends to do (borrow from A, or
    return to C) and, if a nearby station is *forecast* to be imbalanced,
    redirects that same trip to also relieve the network (R7.1);
  * Scenario 1 -- Borrow redirection (recommend_borrow_relay): if a station B
    near the user's borrow origin is forecast Medium/High FULL, suggest
    borrowing from B instead; taking one bike from B lowers B's future full
    risk. It never redirects toward a high-SHORTAGE station (that would harm B);
  * Scenario 2 -- Return redirection (recommend_return_relay): if a station D
    near the user's return origin is forecast Medium/High SHORTAGE, suggest
    returning to D instead; adding one bike to D lowers D's future shortage
    risk. It never redirects toward a high-FULL station (that would harm D);
  * each scenario returns a mission dict with original_station, partner_station,
    mission_type, predicted_problem, risk_probability, distance_km, reward_twd,
    mission_text; or None when no suitable nearby station exists (R7.2);
  * reward is a simple tier from config.REWARD_TIERS = [5, 10, 15] TWD, chosen
    from a severity score combining risk probability and neighbor detour
    distance (R7.3);
  * it involves NO real accounts, payments, coupon APIs, or real money (R7.4).

This module never modifies the raw CSV or any training artifact. It only reads
the prediction dicts / snapshot handed to it and returns new structures.

Design references: design.md §7 (Intervention Engine), §8 (Friend Relay 2.0),
§9 (recommendation data structure). Requirements R6, R7.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config
from . import predict

# Recommended-action codes (design §9).
ACTION_TRUCK = "A"
ACTION_RELAY = "B"
ACTION_NONE = "C"

# Earth mean radius in kilometres, for the Haversine great-circle distance.
EARTH_RADIUS_KM = 6371.0088

# Keys a snapshot / target prediction dict must carry (design §9 prediction
# output; produced by predict.predict_station). These are the facts the
# heuristic reads to pick neighbors and size the intervention.
REQUIRED_PREDICTION_KEYS = (
    "station",
    "lat",
    "lon",
    "current_bikes",
    "current_docks",
    "shortage_prob",
    "full_prob",
    "risk_level",
    "expected_risk_time",
)


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: "float | np.ndarray",
    lon2: "float | np.ndarray",
) -> "float | np.ndarray":
    """Great-circle distance in km between two lat/lon points (pure numpy).

    Scalars or numpy arrays are accepted for the second point, so this can score
    a whole snapshot of candidate neighbors in one vectorized call. No external
    geospatial dependency is used (R6.5).

    Args:
        lat1, lon1: Origin latitude/longitude in decimal degrees.
        lat2, lon2: Target latitude/longitude in decimal degrees. May be numpy
            arrays (element-wise) or scalars.

    Returns:
        Distance in kilometres -- a float for scalar input, or a numpy array
        aligned to the array inputs.
    """
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lon1_r = np.radians(np.asarray(lon1, dtype=float))
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    lon2_r = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    # Clip guards against tiny floating-point excursions outside [0, 1].
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    dist = EARTH_RADIUS_KM * c

    # Return a plain float for scalar inputs, keep arrays as arrays.
    if np.ndim(dist) == 0:
        return float(dist)
    return dist


def _validate_prediction(pred: Dict, what: str) -> None:
    """Ensure a prediction/snapshot dict carries the §9 keys we rely on."""
    missing = [k for k in REQUIRED_PREDICTION_KEYS if k not in pred]
    if missing:
        raise KeyError(
            f"{what} is missing required key(s): {missing}. Expected a "
            f"predict.predict_station() output (design §9): "
            f"{list(REQUIRED_PREDICTION_KEYS)}."
        )


def _risk_probs(pred: Dict) -> "tuple[float, float]":
    """Return (shortage_prob, full_prob) as floats."""
    return float(pred["shortage_prob"]), float(pred["full_prob"])


def _reward_for_severity(severity: float) -> int:
    """Map a severity score in [0, 1] to a reward tier from config.REWARD_TIERS.

    The tiers (default [5, 10, 15] TWD) are split into equal severity bands, so
    the returned value is ALWAYS one of the configured tiers (R7.3). Higher
    severity -> higher tier. This is intentionally a plain lookup, not an
    optimization model (R7.5).
    """
    tiers = sorted(config.REWARD_TIERS)
    n = len(tiers)
    # Clamp severity into [0, 1] then bucket into n equal bands.
    s = min(1.0, max(0.0, float(severity)))
    idx = int(s * n)
    if idx >= n:  # s == 1.0 edge
        idx = n - 1
    return int(tiers[idx])


def _severity_score(risk_prob: float, distance_km: float) -> float:
    """Combine risk probability + neighbor detour distance into [0, 1].

    Design §8: severity blends the risk probability (how likely / how far past
    the risk threshold) with the detour distance to the partner station (a
    closer, easier relay is a "smaller ask"). Nearer + riskier -> higher score
    -> higher reward tier.

    * risk component: how far the probability sits above RISK_LOW_MAX, scaled by
      the span up to certainty. Below RISK_LOW_MAX contributes ~0.
    * proximity component: 1 at the station, decaying to 0 at MAX_NEIGHBOR_KM;
      a farther ask slightly lowers severity so we don't over-reward long trips.
    """
    low = float(config.RISK_LOW_MAX)
    span = max(1e-9, 1.0 - low)
    risk_component = min(1.0, max(0.0, (float(risk_prob) - low) / span))

    max_km = float(config.MAX_NEIGHBOR_KM)
    if max_km <= 0:
        proximity = 1.0
    else:
        proximity = 1.0 - min(1.0, max(0.0, float(distance_km) / max_km))

    # Weight risk more heavily than proximity; risk is the reason to intervene.
    return 0.7 * risk_component + 0.3 * proximity


def _candidate_neighbors(
    target: Dict,
    snapshot: Sequence[Dict],
) -> List[Dict]:
    """Neighbors within MAX_NEIGHBOR_KM of the target (excluding the target).

    Each returned dict is the snapshot entry augmented with "distance_km".
    Vectorized Haversine over the snapshot keeps this a simple, fast scan
    (no routing, R6.6).
    """
    others = [s for s in snapshot if str(s["station"]) != str(target["station"])]
    if not others:
        return []

    lats = np.array([float(s["lat"]) for s in others], dtype=float)
    lons = np.array([float(s["lon"]) for s in others], dtype=float)
    dists = haversine_km(float(target["lat"]), float(target["lon"]), lats, lons)
    dists = np.atleast_1d(dists)

    neighbors: List[Dict] = []
    for s, d in zip(others, dists):
        if d <= config.MAX_NEIGHBOR_KM:
            enriched = dict(s)
            enriched["distance_km"] = float(d)
            neighbors.append(enriched)
    return neighbors


def _pick_donor(target: Dict, neighbors: List[Dict]) -> Optional[Dict]:
    """Pick a donor for a shortage-high target (design §7 step 2).

    A donor should have MORE bikes to spare and a LOWER own shortage risk than
    the target. Among qualifying neighbors, prefer the one with the most
    available bikes, breaking ties by nearer distance.
    """
    target_shortage, _ = _risk_probs(target)
    qualifying = [
        n
        for n in neighbors
        if int(n["current_bikes"]) > int(target["current_bikes"])
        and float(n["shortage_prob"]) < target_shortage
    ]
    if not qualifying:
        return None
    # Most spare bikes first, then nearest.
    qualifying.sort(
        key=lambda n: (-int(n["current_bikes"]), n["distance_km"])
    )
    return qualifying[0]


def _pick_destination(target: Dict, neighbors: List[Dict]) -> Optional[Dict]:
    """Pick an alternative destination for a full-high target (design §7 step 3).

    A destination should have MORE return spaces (docks) and a LOWER own full
    risk than the target. Prefer the most free docks, breaking ties by nearer
    distance.
    """
    _, target_full = _risk_probs(target)
    qualifying = [
        n
        for n in neighbors
        if int(n["current_docks"]) > int(target["current_docks"])
        and float(n["full_prob"]) < target_full
    ]
    if not qualifying:
        return None
    qualifying.sort(
        key=lambda n: (-int(n["current_docks"]), n["distance_km"])
    )
    return qualifying[0]


def _move_bikes_for_shortage(target: Dict, donor: Dict) -> int:
    """How many bikes a truck should move donor -> target (simple heuristic).

    Aim to lift the target just above the shortage threshold while never taking
    a donor below it. This is a back-of-envelope size, not a routing solution
    (R6.6). Always at least 1 when a donor was selected.
    """
    deficit = max(0, (config.SHORTAGE_THRESHOLD + 1) - int(target["current_bikes"]))
    donor_spare = max(0, int(donor["current_bikes"]) - (config.SHORTAGE_THRESHOLD + 1))
    return int(max(1, min(deficit, donor_spare) or 1))


def _move_bikes_for_full(target: Dict, dest: Dict) -> int:
    """How many bikes to move target -> destination to relieve a full station.

    Move enough to restore some free docks at the target without overfilling the
    destination. Simple heuristic, always at least 1 when a destination exists.
    """
    dock_deficit = max(0, (config.FULL_THRESHOLD + 1) - int(target["current_docks"]))
    dest_room = max(0, int(dest["current_docks"]) - (config.FULL_THRESHOLD + 1))
    return int(max(1, min(dock_deficit, dest_room) or 1))


def _empty_options() -> Dict:
    """The options container with every slot empty (Option C default)."""
    return {
        "A_truck": None,
        "B_relay": None,
        "C_none": False,
    }


def _base_recommendation(target: Dict) -> Dict:
    """The §9 recommendation skeleton populated from the target prediction."""
    return {
        "station": str(target["station"]),
        "current_bikes": int(target["current_bikes"]),
        "current_docks": int(target["current_docks"]),
        "shortage_prob": float(target["shortage_prob"]),
        "full_prob": float(target["full_prob"]),
        "expected_risk_time": str(target["expected_risk_time"]),
        "recommended_action": ACTION_NONE,
        "options": _empty_options(),
    }


def recommend(
    target: Dict,
    snapshot: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Produce a structured intervention recommendation for one station.

    This is the module entry point (design §7 flow). It reads the target
    station's prediction dict (design §9, from predict.predict_station) plus the
    whole-station snapshot used to pick a neighbor, and returns the structured
    recommendation described in design §9.

    Flow (design §7):
      1. Low risk -> Option C (No Intervention).
      2. shortage high -> find a nearby donor -> Option A (truck rebalancing:
         move N bikes donor -> target) + Option B (Friend Relay: a return
         redirection whose beneficiary D is the target station -- guide users to
         return here).
      3. full high -> find a nearby alternative destination -> Option A (move
         target's bikes to it) + Option B (Friend Relay: a borrow redirection
         whose beneficiary B is the target station -- guide users to borrow here
         / avoid returning).

    The Friend Relay card (Option B) is the user-intent predictive incentive of
    design §8 / R7: it triggers on a *forecast* imbalance (Medium/High) and
    carries a mission dict (see recommend_borrow_relay / recommend_return_relay)
    with a tier reward from config.REWARD_TIERS. Its §9 keys (mission_text,
    reward_twd, partner_station) remain populated; the user-intent keys are
    added on top.

    Args:
        target: The target station's prediction dict (design §9). Required keys
            per REQUIRED_PREDICTION_KEYS.
        snapshot: The whole-station current snapshot -- a sequence of prediction
            dicts (one per station, same shape as `target`), used only to pick a
            neighbor. May be None/empty; then neighbor-based options are skipped
            and Option B (relay) can still fire only if a partner is found (it
            won't be, without a snapshot), so the recommendation degrades
            gracefully.

    Returns:
        A structured recommendation dict (design §9). For Low risk,
        recommended_action == "C" and options.C_none is True.

    Raises:
        KeyError: If the target (or any snapshot entry inspected) lacks a
            required §9 key.
    """
    _validate_prediction(target, "target prediction")
    rec = _base_recommendation(target)

    risk_level = str(target["risk_level"])

    # Step 1: Low risk -> no intervention (Option C). R6.2.
    if risk_level == predict.RISK_LOW:
        rec["recommended_action"] = ACTION_NONE
        rec["options"]["C_none"] = True
        return rec

    snapshot = list(snapshot) if snapshot else []
    for s in snapshot:
        _validate_prediction(s, "snapshot station")

    neighbors = _candidate_neighbors(target, snapshot)
    shortage_prob, full_prob = _risk_probs(target)

    # Decide which imbalance dominates (higher probability drives the action).
    shortage_dominant = shortage_prob >= full_prob

    donor_or_dest: Optional[Dict] = None
    move_bikes = 0
    if shortage_dominant:
        # Step 2: shortage high -> donor station. R6.3.
        donor_or_dest = _pick_donor(target, neighbors)
        if donor_or_dest is not None:
            move_bikes = _move_bikes_for_shortage(target, donor_or_dest)
    else:
        # Step 3: full high -> alternative destination. R6.4.
        donor_or_dest = _pick_destination(target, neighbors)
        if donor_or_dest is not None:
            move_bikes = _move_bikes_for_full(target, donor_or_dest)

    if donor_or_dest is None:
        # No qualifying neighbor within MAX_NEIGHBOR_KM: nothing actionable to
        # rebalance against. Fall back to Option C so output stays valid (R6.6:
        # we never fabricate a route). The risk is still surfaced in the dict.
        rec["recommended_action"] = ACTION_NONE
        rec["options"]["C_none"] = True
        return rec

    distance_km = float(donor_or_dest["distance_km"])

    # Option A: truck rebalancing (design §7). R6.2. (operator-side, unchanged)
    rec["options"]["A_truck"] = {
        "donor_or_dest": str(donor_or_dest["station"]),
        "distance_km": distance_km,
        "move_bikes": int(move_bikes),
    }

    # Option B: Friend Relay 2.0 predictive incentive (design §8, R7), now
    # sourced from the USER-INTENT scenarios. The target station is the
    # beneficiary the operator wants users to help:
    #   * shortage-high target -> we want users to RETURN bikes here (return
    #     redirection, the target is the beneficiary D);
    #   * full-high target -> we want users to BORROW from here / avoid returning
    #     (borrow redirection, the target is the beneficiary B).
    # We build the mission with the neighbor as the user's origin and the target
    # as the partner/beneficiary, so the two explicit functions and this wiring
    # stay consistent. The §9 schema is preserved: B_relay always carries
    # mission_text, reward_twd, partner_station; the user-intent keys are ADDED.
    if shortage_dominant:
        b_mission = _relay_mission(
            origin=donor_or_dest,
            partner=dict(target, distance_km=distance_km),
            mission_type=MISSION_RETURN,
            predicted_problem=PROBLEM_SHORTAGE,
            risk_probability=shortage_prob,
            action_verb_km="額外騎行",
        )
    else:
        b_mission = _relay_mission(
            origin=donor_or_dest,
            partner=dict(target, distance_km=distance_km),
            mission_type=MISSION_BORROW,
            predicted_problem=PROBLEM_FULL,
            risk_probability=full_prob,
            action_verb_km="額外步行",
        )
    rec["options"]["B_relay"] = b_mission

    # Recommended action: prefer the low-friction Friend Relay when the imbalance
    # is only forecast (Medium); reach for a truck at High severity. Both options
    # are always provided so the operator can choose (R6.2).
    if risk_level == predict.RISK_HIGH:
        rec["recommended_action"] = ACTION_TRUCK
    else:
        rec["recommended_action"] = ACTION_RELAY

    return rec


def recommend_snapshot(snapshot: Sequence[Dict]) -> List[Dict]:
    """Run recommend() for every station in a snapshot (convenience wrapper).

    Each station is scored against the rest of the snapshot as its neighbor
    pool. Useful for the Streamlit demo (Task 6) to show recommendations across
    the network at once.
    """
    snap = list(snapshot)
    return [recommend(station, snap) for station in snap]


# --------------------------------------------------------------------------- #
# Friend Relay 2.0 -- user-intent-centered redirection (design §8, R7)        #
# --------------------------------------------------------------------------- #

# Mission-type / predicted-problem codes for the mission dicts returned by the
# two explicit scenarios below.
MISSION_BORROW = "borrow"
MISSION_RETURN = "return"
PROBLEM_FULL = "full"
PROBLEM_SHORTAGE = "shortage"


def _distance_meters(distance_km: float) -> int:
    """Convert a km distance to an integer metres figure, rounded to 10 m.

    Used only for the human-facing "額外步行/騎行 Xm" mission text.
    """
    meters = float(distance_km) * 1000.0
    return int(round(meters / 10.0) * 10)


def _find_by_station(snapshot: Sequence[Dict], origin_station: str) -> Optional[Dict]:
    """Locate the origin station's own prediction dict inside the snapshot."""
    for s in snapshot:
        if str(s["station"]) == str(origin_station):
            return s
    return None


def _pick_relay_partner(
    candidates: List[Dict],
    prob_key: str,
    exclude_key: str,
) -> Optional[Dict]:
    """Pick the most impactful relay partner among nearby candidates.

    A candidate qualifies when its relevant probability (``prob_key``) is
    Medium/High (>= RISK_LOW_MAX) so the redirection actually relieves a
    *forecast* imbalance, AND its opposite-risk probability (``exclude_key``) is
    below RISK_HIGH_MIN so we never redirect the user toward a station that is
    itself about to hit the opposite failure (the harm-avoidance guard).

    Among qualifying candidates, pick the highest relevant probability (most
    impactful), breaking ties by nearest distance.
    """
    qualifying = [
        n
        for n in candidates
        if float(n[prob_key]) >= config.RISK_LOW_MAX
        and float(n[exclude_key]) < config.RISK_HIGH_MIN
    ]
    if not qualifying:
        return None
    # Most impactful (highest relevant prob) first, then nearest.
    qualifying.sort(key=lambda n: (-float(n[prob_key]), n["distance_km"]))
    return qualifying[0]


def _relay_mission(
    origin: Dict,
    partner: Dict,
    mission_type: str,
    predicted_problem: str,
    risk_probability: float,
    action_verb_km: str,
) -> Dict:
    """Build a Friend Relay 2.0 mission dict (design §8, R7).

    Reward is the simple configurable tier from config.REWARD_TIERS via the
    shared severity heuristic (always one of {5, 10, 15}). ``action_verb_km`` is
    the Chinese phrasing for the detour ("額外步行" for borrow, "額外騎行" for
    return).
    """
    distance_km = float(partner["distance_km"])
    reward = _reward_for_severity(_severity_score(risk_probability, distance_km))
    meters = _distance_meters(distance_km)
    partner_name = str(partner["station"])

    verb = "借車" if mission_type == MISSION_BORROW else "還車"
    mission_text = (
        f"改到 {partner_name} 站{verb}｜{action_verb_km} {meters}m｜獲得 {reward} 元"
    )

    return {
        "original_station": str(origin["station"]),
        "partner_station": partner_name,
        "mission_type": mission_type,
        "predicted_problem": predicted_problem,
        "risk_probability": float(risk_probability),
        "distance_km": distance_km,
        "reward_twd": int(reward),
        "mission_text": mission_text,
    }


def recommend_borrow_relay(
    origin_station: str,
    station_snapshot: Sequence[Dict],
) -> Optional[Dict]:
    """Scenario 1 -- borrow redirection (design §8, R7).

    The user intends to BORROW from ``origin_station`` A. If a nearby station B
    within MAX_NEIGHBOR_KM is forecast to have Medium/High FULL risk, suggest
    the user borrow from B instead: borrowing one bike from B lowers B's future
    full-station risk.

    Harm-avoidance guard: a candidate whose SHORTAGE risk is High
    (shortage_prob >= RISK_HIGH_MIN) is excluded -- borrowing from an
    about-to-be-empty station would only hurt it.

    Args:
        origin_station: The station the user originally plans to borrow from.
        station_snapshot: Whole-station snapshot of prediction dicts (design §9).

    Returns:
        A mission dict (see module docstring for the fields), or None when no
        suitable nearby station exists. Never fabricates a mission.
    """
    snapshot = list(station_snapshot)
    for s in snapshot:
        _validate_prediction(s, "snapshot station")

    origin = _find_by_station(snapshot, origin_station)
    if origin is None:
        return None

    neighbors = _candidate_neighbors(origin, snapshot)
    # Redirect toward a future-FULL station; never toward a high-SHORTAGE one.
    partner = _pick_relay_partner(
        neighbors, prob_key="full_prob", exclude_key="shortage_prob"
    )
    if partner is None:
        return None

    return _relay_mission(
        origin=origin,
        partner=partner,
        mission_type=MISSION_BORROW,
        predicted_problem=PROBLEM_FULL,
        risk_probability=float(partner["full_prob"]),
        action_verb_km="額外步行",
    )


def recommend_return_relay(
    origin_station: str,
    station_snapshot: Sequence[Dict],
) -> Optional[Dict]:
    """Scenario 2 -- return redirection (design §8, R7).

    The user intends to RETURN to ``origin_station`` C. If a nearby station D
    within MAX_NEIGHBOR_KM is forecast to have Medium/High SHORTAGE risk,
    suggest the user return to D instead: returning one bike to D lowers D's
    future shortage risk.

    Harm-avoidance guard: a candidate whose FULL risk is High
    (full_prob >= RISK_HIGH_MIN) is excluded -- returning to an about-to-be-full
    station would only hurt it.

    Args:
        origin_station: The station the user originally plans to return to.
        station_snapshot: Whole-station snapshot of prediction dicts (design §9).

    Returns:
        A mission dict (see module docstring for the fields), or None when no
        suitable nearby station exists. Never fabricates a mission.
    """
    snapshot = list(station_snapshot)
    for s in snapshot:
        _validate_prediction(s, "snapshot station")

    origin = _find_by_station(snapshot, origin_station)
    if origin is None:
        return None

    neighbors = _candidate_neighbors(origin, snapshot)
    # Redirect toward a future-SHORTAGE station; never toward a high-FULL one.
    partner = _pick_relay_partner(
        neighbors, prob_key="shortage_prob", exclude_key="full_prob"
    )
    if partner is None:
        return None

    return _relay_mission(
        origin=origin,
        partner=partner,
        mission_type=MISSION_RETURN,
        predicted_problem=PROBLEM_SHORTAGE,
        risk_probability=float(partner["shortage_prob"]),
        action_verb_km="額外騎行",
    )


# --------------------------------------------------------------------------- #
# Decision layer -- pick a single primary/secondary action (additive, R6/R7)  #
# --------------------------------------------------------------------------- #

# Decision-layer action codes (distinct from the §9 recommended_action codes).
DECISION_TRUCK = "truck"
DECISION_RELAY = "friend_relay"
DECISION_NONE = "none"
URGENCY_NORMAL = "normal"
URGENCY_HIGH = "high"


def _extract_intent(user_context: Optional[object]) -> Optional[str]:
    """Read a borrow/return intent from user_context, tolerating shapes.

    Accepts either a dict shaped like {"intent": "borrow"|"return"} (reads the
    "intent" key) or a bare string "borrow"/"return". Anything unrecognized
    (None, missing key, other value) yields None -> no relay is fabricated.
    """
    if user_context is None:
        return None
    if isinstance(user_context, str):
        intent = user_context
    elif isinstance(user_context, dict):
        intent = user_context.get("intent")
    else:
        return None
    if intent in (MISSION_BORROW, MISSION_RETURN):
        return str(intent)
    return None


def _decision_reason(
    primary_action: str,
    secondary_action: Optional[str],
    risk_level: str,
    risk_prob: float,
    predicted_problem: str,
    urgency: str,
) -> str:
    """Build a deterministic, rule-based Chinese explanation (NO LLM).

    Mentions the rounded risk percentage and the problem type, and describes the
    chosen action(s). Purely a function of the decision inputs -- same inputs
    always produce the same string.
    """
    pct = round(float(risk_prob) * 100)
    problem_zh = "缺車" if predicted_problem == PROBLEM_SHORTAGE else "滿站"

    if primary_action == DECISION_NONE:
        return "目前風險偏低，暫不需介入。"

    # Risk-level phrasing.
    if risk_level == predict.RISK_HIGH:
        risk_phrase = f"30 分鐘後{problem_zh}風險 {pct}%"
    elif risk_level == predict.RISK_MEDIUM:
        risk_phrase = f"目前為中度{problem_zh}風險（{pct}%）"
    else:
        risk_phrase = f"{problem_zh}風險 {pct}%"

    if primary_action == DECISION_TRUCK and secondary_action == DECISION_RELAY:
        if urgency == URGENCY_HIGH:
            return (
                f"{risk_phrase}，風險緊急且同時具備可行派車與使用者協作方案，"
                f"建議立即以派車為主要措施，Friend Relay 2.0 為輔助。"
            )
        return (
            f"{risk_phrase}，且存在可行派車與使用者協作方案，"
            f"建議以派車為主要措施，Friend Relay 2.0 為輔助。"
        )
    if primary_action == DECISION_TRUCK:
        prefix = "風險緊急，" if urgency == URGENCY_HIGH else ""
        return f"{risk_phrase}，{prefix}附近有合適的調度站點，建議派車調度。"
    # primary_action == DECISION_RELAY
    prefix = "風險緊急，" if urgency == URGENCY_HIGH else ""
    return (
        f"{risk_phrase}，{prefix}附近有合適的使用者協作任務，"
        f"建議優先使用 Friend Relay 2.0。"
    )


def decide_intervention(
    target_prediction: Dict,
    snapshot: Optional[Sequence[Dict]] = None,
    user_context: Optional[object] = None,
) -> Dict:
    """Decide a single primary (and optional secondary) intervention action.

    A thin, explainable decision layer *on top of* the existing Task 5 engine.
    It reuses recommend() for truck availability and the user-intent Friend
    Relay 2.0 scenarios for relay availability; it never re-implements
    neighbor search, reward tiers, or Haversine, and never calls an LLM.

    Truck availability comes from recommend(...)["options"]["A_truck"]. Friend
    Relay is only considered when the caller supplies a user intent
    (user_context), because Friend Relay is user-intent-centered -- with no
    intent we do NOT fabricate a mission (truck is still evaluated). An intent
    of "borrow" (MISSION_BORROW) uses recommend_borrow_relay; "return"
    (MISSION_RETURN) uses recommend_return_relay. user_context may be a dict
    {"intent": ...} or a bare "borrow"/"return" string.

    Decision rules (risk_level from target_prediction, risk_prob =
    max(shortage_prob, full_prob)):
      * Low risk -> primary "none", urgency "normal".
      * Medium risk -> primary "friend_relay" if a relay exists, else "none";
        no secondary; urgency "normal".
      * High risk -> primary/secondary chosen from what is available
        (truck+relay -> truck primary, relay secondary; only one -> that one).
      * Urgent (risk_prob >= config.INTERVENTION_URGENT_PROB) -> urgency "high";
        with both available, truck primary + relay secondary; with one, that one
        stays primary.

    Args:
        target_prediction: The target station's §9 prediction dict.
        snapshot: Whole-station snapshot (sequence of §9 dicts) used to find
            neighbors. May be None/empty (then truck/relay simply won't be
            available).
        user_context: Optional user intent ({"intent": "borrow"|"return"} or a
            bare string). None -> no Friend Relay is considered/fabricated.

    Returns:
        A decision dict with the exact keys: station, risk_level,
        risk_probability, predicted_problem, primary_action, secondary_action,
        urgency, reason, truck, friend_relay.

    Raises:
        KeyError: If target_prediction lacks a required §9 key.
    """
    _validate_prediction(target_prediction, "target prediction")

    shortage_prob, full_prob = _risk_probs(target_prediction)
    risk_prob = max(shortage_prob, full_prob)
    risk_level = str(target_prediction["risk_level"])
    predicted_problem = (
        PROBLEM_SHORTAGE if shortage_prob >= full_prob else PROBLEM_FULL
    )
    target_station = str(target_prediction["station"])

    # Truck availability from the existing engine (never duplicated).
    rec = recommend(target_prediction, snapshot)
    truck = rec["options"]["A_truck"]
    truck_available = truck is not None

    # Friend Relay only when the user supplies a usable intent. Reuse the
    # existing user-intent scenarios; never fabricate a mission otherwise.
    friend_relay: Optional[Dict] = None
    intent = _extract_intent(user_context)
    if intent is not None:
        snap = list(snapshot) if snapshot else []
        if intent == MISSION_BORROW:
            friend_relay = recommend_borrow_relay(target_station, snap)
        elif intent == MISSION_RETURN:
            friend_relay = recommend_return_relay(target_station, snap)
    relay_available = friend_relay is not None

    urgent = risk_prob >= float(config.INTERVENTION_URGENT_PROB)

    # --- Choose primary / secondary / urgency by rules ------------------- #
    primary_action = DECISION_NONE
    secondary_action: Optional[str] = None
    urgency = URGENCY_NORMAL

    if risk_level == predict.RISK_LOW:
        primary_action = DECISION_NONE
        secondary_action = None
        urgency = URGENCY_NORMAL
    elif risk_level == predict.RISK_MEDIUM:
        primary_action = DECISION_RELAY if relay_available else DECISION_NONE
        secondary_action = None
        urgency = URGENCY_NORMAL
    else:  # RISK_HIGH
        if truck_available and relay_available:
            primary_action = DECISION_TRUCK
            secondary_action = DECISION_RELAY
        elif truck_available:
            primary_action = DECISION_TRUCK
            secondary_action = None
        elif relay_available:
            primary_action = DECISION_RELAY
            secondary_action = None
        else:
            primary_action = DECISION_NONE
            secondary_action = None
        urgency = URGENCY_NORMAL

    # Rule D: urgent overrides urgency (and confirms truck-primary when both
    # actions are available). A single available action stays primary.
    if urgent and primary_action != DECISION_NONE:
        if truck_available and relay_available:
            primary_action = DECISION_TRUCK
            secondary_action = DECISION_RELAY
        urgency = URGENCY_HIGH

    reason = _decision_reason(
        primary_action=primary_action,
        secondary_action=secondary_action,
        risk_level=risk_level,
        risk_prob=risk_prob,
        predicted_problem=predicted_problem,
        urgency=urgency,
    )

    return {
        "station": target_station,
        "risk_level": str(risk_level),
        "risk_probability": float(risk_prob),
        "predicted_problem": predicted_problem,
        "primary_action": primary_action,
        "secondary_action": secondary_action,
        "urgency": urgency,
        "reason": reason,
        "truck": truck,
        "friend_relay": friend_relay,
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    # Tiny hand-built snapshot demonstrating the three flows without needing the
    # trained models. Shapes match predict.predict_station output (design §9).
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

    snapshot = [
        # Shortage-high target with a nearby well-stocked donor.
        _mk("A_shortage", 25.0000, 121.5000, 1, 20, 0.80, 0.05, "High"),
        _mk("A_donor", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low"),
        # Full-high target with a nearby dock-rich destination.
        _mk("B_full", 25.0100, 121.5100, 22, 0, 0.05, 0.75, "High"),
        _mk("B_dest", 25.0120, 121.5110, 2, 19, 0.05, 0.05, "Low"),
        # Low-risk station -> Option C.
        _mk("C_calm", 25.0500, 121.5500, 10, 10, 0.05, 0.05, "Low"),
    ]

    for r in recommend_snapshot(snapshot):
        print("=" * 64)
        print(f"station={r['station']} action={r['recommended_action']}")
        print(f"  shortage={r['shortage_prob']} full={r['full_prob']} "
              f"risk_time={r['expected_risk_time']}")
        print(f"  options={r['options']}")

    # --- Friend Relay 2.0 user-intent scenarios -----------------------------
    print("=" * 64)
    print("Friend Relay 2.0 -- user-intent scenarios")

    # Scenario 1: user wants to borrow at U_borrow; nearby N_full is future-full.
    borrow_snapshot = [
        _mk("U_borrow", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full", 25.0030, 121.5010, 20, 1, 0.05, 0.72, "High"),   # ~0.35 km
    ]
    borrow_mission = recommend_borrow_relay("U_borrow", borrow_snapshot)
    print(f"  borrow relay: {borrow_mission}")

    # Scenario 2: user wants to return at U_return; nearby N_short is future-short.
    return_snapshot = [
        _mk("U_return", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_short", 25.0030, 121.5010, 1, 20, 0.70, 0.05, "High"),  # ~0.35 km
    ]
    return_mission = recommend_return_relay("U_return", return_snapshot)
    print(f"  return relay: {return_mission}")
