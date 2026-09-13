# -*- coding: utf-8 -*-
"""V1 UI logic: risk levels, nearby search, Friend Relay reward. Streamlit-free.

Pure functions so the whole "what the user sees" contract is unit-testable
head-less. app.py only renders what this module computes.

Risk semantics (mode-dependent, per spec):
  borrow mode -> user cares about EMPTY-bike risk (cannot borrow)
  return mode -> user cares about FULL-dock risk (cannot return)

Business events stay FIXED (ratio < 0.20 low, ratio > 0.80 high). The MEDIUM
band is bounded by each model's own calibrated alert threshold.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from . import v1_decision as vd
from . import v1_predict as vp

EARTH_RADIUS_M = 6371008.8

MODE_BORROW = "borrow"
MODE_RETURN = "return"

RISK_HIGH = "高"
RISK_MEDIUM = "中"
RISK_LOW = "低"

RISK_EMOJI = {RISK_HIGH: "🔴", RISK_MEDIUM: "🟡", RISK_LOW: "🟢"}
RISK_COLOR = {RISK_HIGH: "red", RISK_MEDIUM: "orange", RISK_LOW: "green"}

REWARD_MIN = 5
REWARD_MAX = 10
REWARD_MAX_EXTRA_M = 300.0

DEFAULT_RADIUS_M = 500
RADIUS_CHOICES = [100, 300, 500, 1000, 2000]
MAX_USER_STATIONS = 5


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, max(0.0, a))))


# --------------------------------------------------------------------------- #
# risk level conversion
# --------------------------------------------------------------------------- #
def empty_risk_level(ratio: float, low_alert: float) -> str:
    """Risk of NOT being able to BORROW (station runs out of bikes)."""
    r = float(ratio)
    if r < vd.LOW_EVENT:
        return RISK_HIGH
    if r < float(low_alert):
        return RISK_MEDIUM
    return RISK_LOW


def full_risk_level(ratio: float, high_alert: float) -> str:
    """Risk of NOT being able to RETURN (station runs out of docks)."""
    r = float(ratio)
    if r > vd.HIGH_EVENT:
        return RISK_HIGH
    if r > float(high_alert):
        return RISK_MEDIUM
    return RISK_LOW


def alerts_for(day_type: str) -> Dict[str, Dict[str, float]]:
    return {"30": vp.ALERT_THRESHOLDS[f"{day_type}_30"],
            "60": vp.ALERT_THRESHOLDS[f"{day_type}_60"]}


def risk_pair(prediction: Dict, mode: str) -> Dict[str, str]:
    """The two risk levels the USER sees, chosen by mode."""
    a = alerts_for(prediction["day_type"])
    if mode == MODE_BORROW:
        return {"risk_30m": empty_risk_level(prediction["ratio_30m"], a["30"]["low"]),
                "risk_60m": empty_risk_level(prediction["ratio_60m"], a["60"]["low"]),
                "risk_meaning": "借不到車的風險"}
    return {"risk_30m": full_risk_level(prediction["ratio_30m"], a["30"]["high"]),
            "risk_60m": full_risk_level(prediction["ratio_60m"], a["60"]["high"]),
            "risk_meaning": "沒有空位可還的風險"}


def operator_risks(prediction: Dict) -> Dict[str, str]:
    """All four operator risk columns from the same two ratios."""
    a = alerts_for(prediction["day_type"])
    return {
        "empty_risk_30m": empty_risk_level(prediction["ratio_30m"], a["30"]["low"]),
        "empty_risk_60m": empty_risk_level(prediction["ratio_60m"], a["60"]["low"]),
        "full_risk_30m": full_risk_level(prediction["ratio_30m"], a["30"]["high"]),
        "full_risk_60m": full_risk_level(prediction["ratio_60m"], a["60"]["high"]),
    }


# --------------------------------------------------------------------------- #
# nearby search
# --------------------------------------------------------------------------- #
def nearby_stations(snapshot_rows: Sequence[Dict], dest_lat: float,
                    dest_lon: float, radius_m: float = DEFAULT_RADIUS_M) -> List[Dict]:
    """Stations within radius, nearest first, each with distance_m."""
    out = []
    for r in snapshot_rows:
        d = haversine_m(dest_lat, dest_lon, r["lat"], r["lon"])
        if d <= radius_m:
            e = dict(r)
            e["distance_m"] = d
            out.append(e)
    out.sort(key=lambda x: x["distance_m"])
    return out


def baseline_station(rows: Sequence[Dict], mode: str) -> Optional[Dict]:
    """Nearest FEASIBLE station: has a bike to borrow / a dock to return."""
    key = ("current_available_bikes" if mode == MODE_BORROW
           else "current_available_docks")
    for r in sorted(rows, key=lambda x: x["distance_m"]):
        if float(r.get(key, 0)) > 0:
            return r
    return None


# --------------------------------------------------------------------------- #
# Friend Relay reward
# --------------------------------------------------------------------------- #
def reward_for_extra_distance(extra_m: float) -> Optional[int]:
    """5..10 TWD, linear in extra detour. None outside (0, 300] metres."""
    e = float(extra_m)
    if e <= 0 or e > REWARD_MAX_EXTRA_M:
        return None
    reward = REWARD_MIN + REWARD_MIN * (e / REWARD_MAX_EXTRA_M)
    return int(round(min(REWARD_MAX, max(REWARD_MIN, reward))))


def is_reward_candidate(row: Dict, mode: str, risk_state: str,
                        baseline_station_name: Optional[str]) -> bool:
    """Does this station help rebalance the network if the user diverts here?"""
    if baseline_station_name is not None and row["station"] == baseline_station_name:
        return False
    if mode == MODE_RETURN:
        if float(row.get("current_available_docks", 0)) <= 0:
            return False
        return risk_state in (vd.PERSISTENT_LOW, vd.EMERGING_LOW)
    if float(row.get("current_available_bikes", 0)) <= 0:
        return False
    return risk_state in (vd.PERSISTENT_HIGH, vd.EMERGING_HIGH)


def build_user_rows(stations: Sequence[Dict], predictions: Sequence[Dict],
                    mode: str, max_rows: int = MAX_USER_STATIONS) -> List[Dict]:
    """Assemble the user-facing table rows (already sorted by distance).

    `stations` and `predictions` must align positionally.
    """
    if len(stations) != len(predictions):
        raise ValueError("stations and predictions must align")

    base = baseline_station(stations, mode)
    base_name = base["station"] if base else None
    base_dist = base["distance_m"] if base else None

    enriched = []
    for st, pred in zip(stations, predictions):
        state = vd.classify_temporal_risk(pred["ratio_30m"], pred["ratio_60m"])
        rp = risk_pair(pred, mode)
        reward = None
        extra = None
        if base_dist is not None and is_reward_candidate(st, mode, state, base_name):
            extra = st["distance_m"] - base_dist
            reward = reward_for_extra_distance(extra)
        enriched.append({
            "station": st["station"],
            "district": st.get("district", ""),
            "distance_m": st["distance_m"],
            "current_available_bikes": int(st.get("current_available_bikes", 0)),
            "current_available_docks": int(st.get("current_available_docks", 0)),
            "risk_30m": rp["risk_30m"],
            "risk_60m": rp["risk_60m"],
            "risk_meaning": rp["risk_meaning"],
            "risk_state": state,
            "reward_twd": reward,
            "extra_distance_m": extra,
            "is_baseline": st["station"] == base_name,
            "ratio_30m": pred["ratio_30m"],
            "ratio_60m": pred["ratio_60m"],
        })

    # Persistent-imbalance reward candidates should be easy to spot, but the
    # table stays distance-ordered so the user's mental model is unchanged.
    return enriched[:max_rows]


def availability_text(row: Dict, mode: str) -> str:
    if mode == MODE_BORROW:
        return f"可借 {row['current_available_bikes']}"
    return f"可還 {row['current_available_docks']}"


def reward_text(row: Dict) -> str:
    return "—" if row.get("reward_twd") is None else f"+${row['reward_twd']}"


def risk_text(level: str) -> str:
    return f"{RISK_EMOJI.get(level, '')}{level}"


# --------------------------------------------------------------------------- #
# operator aggregation
# --------------------------------------------------------------------------- #
def build_operator_rows(stations: Sequence[Dict], predictions: Sequence[Dict],
                        statics: Optional[Sequence[Optional[Dict]]] = None
                        ) -> List[Dict]:
    """Operator table rows + priority ordering backend."""
    if len(stations) != len(predictions):
        raise ValueError("stations and predictions must align")
    statics = statics or [None] * len(stations)

    rows = []
    for st, pred, static in zip(stations, predictions, statics):
        state = vd.classify_temporal_risk(pred["ratio_30m"], pred["ratio_60m"])
        pri = vd.score_priority(state, pred["ratio_30m"], pred["ratio_60m"],
                                static=static, is_peak=st.get("is_peak"),
                                day_type=pred["day_type"])
        rows.append({
            "station": st["station"],
            "address": station_address(st),
            "district": st.get("district", ""),
            **operator_risks(pred),
            "risk_state": state,
            "risk_state_zh": vd.state_label_zh(state),
            "priority": pri["priority"],
            "reasons": pri["reasons"],
            "current_available_bikes": int(st.get("current_available_bikes", 0)),
            "current_available_docks": int(st.get("current_available_docks", 0)),
            "ratio_30m": pred["ratio_30m"],
            "ratio_60m": pred["ratio_60m"],
        })
    return rows


def station_address(row: Dict) -> str:
    """Best available address. Never fabricated.

    The enriched dataset has no street address field, so the honest fallback is
    city + district + station name.
    """
    city = str(row.get("city") or "").strip()
    district = str(row.get("district") or "").strip()
    station = str(row.get("station") or "").strip()
    parts = [p for p in (city, district, station) if p and p.lower() != "nan"]
    return "".join(parts) if parts else "地址資料未提供"


def rank_operator_rows(rows: Sequence[Dict]) -> List[Dict]:
    """Risk-first ordering: persistent > emerging/transient > normal."""
    rank = {vd.PRIORITY_HIGH: 0, vd.PRIORITY_MEDIUM: 1,
            vd.PRIORITY_LOW: 2, vd.PRIORITY_NONE: 3}

    def key(r):
        fam = vd.state_family(r["risk_state"])
        sev = (min(r["ratio_30m"], r["ratio_60m"]) if fam == "low"
               else -max(r["ratio_30m"], r["ratio_60m"]) if fam == "high" else 1.0)
        return (rank.get(r["priority"], 9),
                0 if vd.is_persistent(r["risk_state"]) else 1, sev)
    return sorted(rows, key=key)


def operator_kpis(rows: Sequence[Dict]) -> Dict[str, int]:
    """Four KPI numbers for the operator header."""
    high = sum(1 for r in rows
               if RISK_HIGH in (r["empty_risk_30m"], r["empty_risk_60m"],
                                r["full_risk_30m"], r["full_risk_60m"]))
    return {
        "high_risk_stations": high,
        "persistent_empty": sum(1 for r in rows
                                if r["risk_state"] == vd.PERSISTENT_LOW),
        "persistent_full": sum(1 for r in rows
                               if r["risk_state"] == vd.PERSISTENT_HIGH),
        "monitored_stations": len(rows),
    }
