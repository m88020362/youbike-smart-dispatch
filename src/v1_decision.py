# -*- coding: utf-8 -*-
"""V1 temporal risk classification + operator priority (pure functions).

NEW module. src/intervention.py (the V0 decision layer, Friend Relay, reward
tiers, Haversine) is untouched and still drives the stable demo.

Business event definition -- FIXED, never calibrated:
    LOW    ratio < 0.20
    HIGH   ratio > 0.80
    NORMAL 0.20 <= ratio <= 0.80

Boundary convention: exactly 0.20 and exactly 0.80 fall on the NORMAL side,
because the events are strict inequalities.

Temporal states combine the 30m and 60m predictions. Wording rule: with only two
discrete horizons we may say both horizons land in the risk band; we must NOT
claim the station is continuously short/full for the whole intervening hour.
"""

from __future__ import annotations

from typing import Dict, List, Optional

LOW_EVENT = 0.20
HIGH_EVENT = 0.80

PERSISTENT_LOW = "persistent_low"
TRANSIENT_LOW = "transient_low"
EMERGING_LOW = "emerging_low"
PERSISTENT_HIGH = "persistent_high"
TRANSIENT_HIGH = "transient_high"
EMERGING_HIGH = "emerging_high"
NORMAL = "normal"

PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"
PRIORITY_NONE = "none"

# Neutral proximity labels. Deliberately NOT "繁榮區" -- there is no such label
# in the data, so no claim of that kind is made.
TRANSPORT_HUB_MRT_M = 300.0
TRANSPORT_HUB_BUS_M = 100.0
SCHOOL_CONTEXT_M = 500.0

_STATE_ZH = {
    PERSISTENT_LOW: "持續性低車量風險",
    TRANSIENT_LOW: "短暫低車量",
    EMERGING_LOW: "低車量風險形成中",
    PERSISTENT_HIGH: "持續性高占用風險",
    TRANSIENT_HIGH: "短暫高占用",
    EMERGING_HIGH: "高占用風險形成中",
    NORMAL: "正常",
}


def is_low(ratio: float) -> bool:
    """Strictly below the low business event threshold (0.20 is NOT low)."""
    return float(ratio) < LOW_EVENT


def is_high(ratio: float) -> bool:
    """Strictly above the high business event threshold (0.80 is NOT high)."""
    return float(ratio) > HIGH_EVENT


def classify_temporal_risk(ratio_30m: float, ratio_60m: float) -> str:
    """Combine both horizons into one temporal risk state.

    low30 & low60   -> persistent_low
    low30 & !low60  -> transient_low
    !low30 & low60  -> emerging_low
    high30 & high60 -> persistent_high
    high30 & !high60-> transient_high
    !high30 & high60-> emerging_high
    otherwise       -> normal
    """
    r30, r60 = float(ratio_30m), float(ratio_60m)
    low30, low60 = is_low(r30), is_low(r60)
    high30, high60 = is_high(r30), is_high(r60)

    if low30 and low60:
        return PERSISTENT_LOW
    if low30 and not low60:
        return TRANSIENT_LOW
    if (not low30) and low60:
        return EMERGING_LOW
    if high30 and high60:
        return PERSISTENT_HIGH
    if high30 and not high60:
        return TRANSIENT_HIGH
    if (not high30) and high60:
        return EMERGING_HIGH
    return NORMAL


def state_label_zh(state: str) -> str:
    return _STATE_ZH.get(state, state)


def is_persistent(state: str) -> bool:
    return state in (PERSISTENT_LOW, PERSISTENT_HIGH)


def state_family(state: str) -> Optional[str]:
    """'low' / 'high' / None for normal."""
    if state in (PERSISTENT_LOW, TRANSIENT_LOW, EMERGING_LOW):
        return "low"
    if state in (PERSISTENT_HIGH, TRANSIENT_HIGH, EMERGING_HIGH):
        return "high"
    return None


# --------------------------------------------------------------------------- #
# operator priority (deterministic, explainable; NOT a second ML model)
# --------------------------------------------------------------------------- #
def _context_signals(static: Optional[Dict], is_peak: Optional[float]) -> Dict:
    """Neutral context flags derived from static distances + peak indicator."""
    static = static or {}
    mrt = static.get("nearest_mrt_distance")
    bus = static.get("nearest_bus_distance")
    jh = static.get("nearest_junior_high_distance")
    uni = static.get("nearest_university_distance")

    transport_hub = bool(
        (mrt is not None and float(mrt) <= TRANSPORT_HUB_MRT_M)
        or (bus is not None and float(bus) <= TRANSPORT_HUB_BUS_M))
    school_context = bool(
        (jh is not None and float(jh) <= SCHOOL_CONTEXT_M)
        or (uni is not None and float(uni) <= SCHOOL_CONTEXT_M))
    peak = bool(is_peak is not None and float(is_peak) >= 0.5)
    return {
        "transport_hub_priority": transport_hub,
        "high_usage_potential": bool(transport_hub or school_context),
        "school_context": school_context,
        "weekday_peak": peak,
        "nearest_mrt_distance": None if mrt is None else float(mrt),
        "nearest_bus_distance": None if bus is None else float(bus),
    }


def score_priority(state: str, ratio_30m: float, ratio_60m: float,
                   static: Optional[Dict] = None,
                   is_peak: Optional[float] = None,
                   day_type: str = "weekday") -> Dict:
    """Deterministic operator priority for one station.

    Base rank: persistent > emerging/transient > normal. Context can raise the
    band by at most one step, and never lifts a normal station above 'low'.
    Every contributing factor is listed in `reasons`, so the result is fully
    explainable without inspecting code.
    """
    r30, r60 = float(ratio_30m), float(ratio_60m)
    sig = _context_signals(static, is_peak)
    reasons: List[str] = []

    if state == NORMAL:
        base = 0
    elif is_persistent(state):
        base = 3
        fam = "low-bike" if state == PERSISTENT_LOW else "high-occupancy"
        reasons.append(f"30m and 60m both predicted in the {fam} band "
                       f"(persistent risk)")
    else:
        base = 2
        if state in (EMERGING_LOW, EMERGING_HIGH):
            reasons.append("risk appears only at 60m (emerging)")
        else:
            reasons.append("risk appears only at 30m (transient)")

    boost = 0
    if base > 0:
        if sig["weekday_peak"] and day_type == "weekday":
            boost += 1
            reasons.append("weekday peak period (is_peak=1)")
        if sig["transport_hub_priority"]:
            boost += 1
            reasons.append("close to MRT/bus (transport_hub_priority)")
        elif sig["school_context"]:
            reasons.append("near school/university (high_usage_potential)")
        # severity nudge, using the fixed business events
        if state_family(state) == "low" and min(r30, r60) < 0.10:
            boost += 1
            reasons.append("predicted ratio below 0.10 (severe low-bike)")
        if state_family(state) == "high" and max(r30, r60) > 0.90:
            boost += 1
            reasons.append("predicted ratio above 0.90 (severe high-occupancy)")

    if base == 0:
        priority = PRIORITY_NONE
        reasons.append("both horizons within the normal band")
    else:
        rank = base + min(boost, 1)   # context lifts by at most one band
        priority = (PRIORITY_HIGH if rank >= 4
                    else PRIORITY_MEDIUM if rank == 3
                    else PRIORITY_LOW)

    return {
        "risk_state": state,
        "risk_state_zh": state_label_zh(state),
        "priority": priority,
        "reasons": reasons,
        "context": sig,
        "ratio_30m": r30,
        "ratio_60m": r60,
        "business_event_low": LOW_EVENT,
        "business_event_high": HIGH_EVENT,
    }


def assess_station(prediction: Dict, static: Optional[Dict] = None,
                   is_peak: Optional[float] = None,
                   station: Optional[str] = None) -> Dict:
    """Full backend assessment for one station from a v1_predict output dict."""
    state = classify_temporal_risk(prediction["ratio_30m"], prediction["ratio_60m"])
    out = score_priority(state, prediction["ratio_30m"], prediction["ratio_60m"],
                         static=static, is_peak=is_peak,
                         day_type=prediction.get("day_type", "weekday"))
    out["station"] = station
    out["day_type"] = prediction.get("day_type")
    out["model_30m"] = prediction.get("model_30m")
    out["model_60m"] = prediction.get("model_60m")
    out["source"] = prediction.get("source")
    return out


def assess_snapshot(predictions: List[Dict], stations: Optional[List[str]] = None,
                    statics: Optional[List[Optional[Dict]]] = None,
                    is_peaks: Optional[List[Optional[float]]] = None) -> List[Dict]:
    """Assess a whole snapshot, preserving input order."""
    n = len(predictions)
    stations = stations or [None] * n
    statics = statics or [None] * n
    is_peaks = is_peaks or [None] * n
    if not (len(stations) == len(statics) == len(is_peaks) == n):
        raise ValueError("predictions / stations / statics / is_peaks must align")
    return [assess_station(p, static=s, is_peak=k, station=st)
            for p, st, s, k in zip(predictions, stations, statics, is_peaks)]


PRIORITY_RANK = {PRIORITY_HIGH: 0, PRIORITY_MEDIUM: 1,
                 PRIORITY_LOW: 2, PRIORITY_NONE: 3}


def rank_operator_priority(assessments: List[Dict]) -> List[Dict]:
    """Sort worst-first: priority band, then persistence, then severity."""
    def key(a):
        fam = state_family(a["risk_state"])
        severity = (min(a["ratio_30m"], a["ratio_60m"]) if fam == "low"
                    else -max(a["ratio_30m"], a["ratio_60m"]) if fam == "high"
                    else 1.0)
        return (PRIORITY_RANK.get(a["priority"], 9),
                0 if is_persistent(a["risk_state"]) else 1,
                severity)
    return sorted(assessments, key=key)
