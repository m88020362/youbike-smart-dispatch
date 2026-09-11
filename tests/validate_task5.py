# -*- coding: utf-8 -*-
"""Validation script for Task 5 (R6 + R7) of the YouBike dispatch MVP.

Run from the project root:
    python -m tests.validate_task5

Read-only. Exercises src/intervention.py against the Task 5 acceptance criteria
using small, hand-built snapshots (shapes identical to predict.predict_station
output, design §9) so it needs no trained models:

  R6.1 high-risk output includes station, current bikes, current return spaces,
       shortage prob, full prob, expected risk time, recommended action.
  R6.2 Options A (truck), B (Friend Relay), C (No Intervention) exist.
  R6.3 shortage high -> nearby donor (more bikes, own shortage lower).
  R6.4 full high -> nearby destination (more docks, own full lower).
  R6.5 distance via lat/lon Haversine (validated against a known reference).
  R6.6 no routing problem solved (move_bikes is a simple int size, integer >= 1).
  R7.1/7.2 forecast imbalance -> predictive mission card text.
  R7.3 reward only in config.REWARD_TIERS = {5, 10, 15}.
  R7.* neighbor distance <= config.MAX_NEIGHBOR_KM.

Friend Relay 2.0 (corrected, user-intent-centered) checks:
  FR1 borrow redirection selects a nearby FUTURE-FULL station.
  FR2 borrow never redirects toward a high-SHORTAGE station (harm guard).
  FR3 return redirection selects a nearby FUTURE-SHORTAGE station.
  FR4 return never redirects toward a high-FULL station (harm guard).
  FR5 partner station is within config.MAX_NEIGHBOR_KM.
  FR6 reward is one of {5, 10, 15}.
  FR7 no suitable nearby station -> no Friend Relay mission (None).
"""

from __future__ import annotations

import sys

from src import config, intervention, predict


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, condition, detail=""):
        self.checks += 1
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not condition:
            self.failures.append(name)

    def done(self):
        print("-" * 60)
        if self.failures:
            print(f"RESULT: {len(self.failures)}/{self.checks} checks FAILED: "
                  f"{self.failures}")
            return 1
        print(f"RESULT: all {self.checks} checks PASSED")
        return 0


REC_KEYS = {
    "station", "current_bikes", "current_docks", "shortage_prob", "full_prob",
    "expected_risk_time", "recommended_action", "options",
}


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


def main() -> int:
    c = Checker()

    # --- R6.5: Haversine correctness against a known reference --------------
    # Taipei 101 (25.0339, 121.5645) -> Taipei Main Station (25.0478, 121.5170)
    # is ~5.0 km. Assert within a small tolerance.
    d = intervention.haversine_km(25.0339, 121.5645, 25.0478, 121.5170)
    c.check("R6.5 Haversine reference distance ~5km", 4.5 < d < 5.5, f"{d:.3f} km")
    c.check("R6.5 Haversine zero distance == 0",
            intervention.haversine_km(25.0, 121.0, 25.0, 121.0) == 0.0)

    # --- Shortage-high flow (R6.1, R6.2, R6.3) ------------------------------
    shortage_target = _mk("A_shortage", 25.0000, 121.5000, 1, 20, 0.80, 0.05, "High")
    shortage_snapshot = [
        shortage_target,
        _mk("A_donor", 25.0030, 121.5010, 18, 3, 0.05, 0.10, "Low"),   # ~0.35 km
        _mk("A_far", 25.0500, 121.5500, 30, 5, 0.02, 0.02, "Low"),     # far (>1km)
    ]
    r = intervention.recommend(shortage_target, shortage_snapshot)

    c.check("R6.1 recommendation has exactly the §9 keys",
            set(r.keys()) == REC_KEYS, str(set(r.keys())))
    c.check("R6.1 includes recommended_action",
            r["recommended_action"] in {"A", "B", "C"}, r["recommended_action"])
    c.check("R6.1 includes current bikes/docks/probs/risk time",
            r["current_bikes"] == 1 and r["current_docks"] == 20
            and r["shortage_prob"] == 0.80 and r["full_prob"] == 0.05
            and r["expected_risk_time"] == "2026-03-30T08:30:00")

    opts = r["options"]
    c.check("R6.2 options container has A_truck, B_relay, C_none",
            set(opts.keys()) == {"A_truck", "B_relay", "C_none"}, str(set(opts.keys())))
    c.check("R6.3 shortage high produced a truck option (donor picked)",
            opts["A_truck"] is not None)
    c.check("R6.3 donor is the nearby well-stocked station",
            opts["A_truck"]["donor_or_dest"] == "A_donor",
            opts["A_truck"]["donor_or_dest"])
    c.check("R6.3 far donor NOT picked (distance filter)",
            opts["A_truck"]["donor_or_dest"] != "A_far")
    c.check("neighbor distance <= MAX_NEIGHBOR_KM",
            opts["A_truck"]["distance_km"] <= config.MAX_NEIGHBOR_KM,
            f"{opts['A_truck']['distance_km']:.3f} <= {config.MAX_NEIGHBOR_KM}")
    c.check("R6.6 move_bikes is a simple positive int (no routing)",
            isinstance(opts["A_truck"]["move_bikes"], int)
            and opts["A_truck"]["move_bikes"] >= 1,
            str(opts["A_truck"]["move_bikes"]))

    # R7.1/7.2: predictive mission card text present. Under the corrected
    # user-intent Friend Relay, a shortage-dominant target is the BENEFICIARY of
    # a return-relay, so B_relay's partner_station is the target itself
    # (guide users to return here), not the truck donor.
    c.check("R7.2 Friend Relay mission card produced", opts["B_relay"] is not None)
    c.check("R7.2 mission text mentions the beneficiary (target) station",
            "A_shortage" in opts["B_relay"]["mission_text"],
            opts["B_relay"]["mission_text"])
    c.check("R7.2 partner_station is the beneficiary target station",
            opts["B_relay"]["partner_station"] == "A_shortage")
    # R7.3: reward only in configured tiers.
    c.check("R7.3 reward in configured tiers {5,10,15}",
            opts["B_relay"]["reward_twd"] in set(config.REWARD_TIERS),
            str(opts["B_relay"]["reward_twd"]))

    # --- Full-high flow (R6.4) ----------------------------------------------
    full_target = _mk("B_full", 25.0100, 121.5100, 22, 0, 0.05, 0.75, "High")
    full_snapshot = [
        full_target,
        _mk("B_dest", 25.0120, 121.5110, 2, 19, 0.05, 0.05, "Low"),  # ~0.24 km
    ]
    rf = intervention.recommend(full_target, full_snapshot)
    c.check("R6.4 full high produced a destination truck option",
            rf["options"]["A_truck"] is not None)
    c.check("R6.4 destination is the dock-rich neighbor",
            rf["options"]["A_truck"]["donor_or_dest"] == "B_dest")
    c.check("R6.4 destination distance <= MAX_NEIGHBOR_KM",
            rf["options"]["A_truck"]["distance_km"] <= config.MAX_NEIGHBOR_KM)
    c.check("R7.3 full-flow reward in tiers",
            rf["options"]["B_relay"]["reward_twd"] in set(config.REWARD_TIERS))

    # --- Low-risk flow -> Option C ------------------------------------------
    low_target = _mk("C_calm", 25.0500, 121.5500, 10, 10, 0.05, 0.05, "Low")
    rc = intervention.recommend(low_target, [low_target])
    c.check("Low risk -> recommended_action == C",
            rc["recommended_action"] == "C", rc["recommended_action"])
    c.check("Low risk -> C_none True and A/B empty",
            rc["options"]["C_none"] is True
            and rc["options"]["A_truck"] is None
            and rc["options"]["B_relay"] is None)

    # --- Medium risk prefers the low-friction relay (design §7) -------------
    med_target = _mk("D_med", 25.0000, 121.5000, 1, 20, 0.45, 0.05, "Medium")
    med_snapshot = [med_target, _mk("D_donor", 25.0030, 121.5010, 18, 3, 0.05, 0.1, "Low")]
    rm = intervention.recommend(med_target, med_snapshot)
    c.check("Medium risk -> recommended_action == B (relay)",
            rm["recommended_action"] == "B", rm["recommended_action"])

    # --- No qualifying neighbor -> graceful Option C ------------------------
    lonely = _mk("E_lonely", 25.0, 121.5, 1, 20, 0.9, 0.05, "High")
    re = intervention.recommend(lonely, [lonely])  # no other stations
    c.check("High risk but no neighbor -> falls back to Option C",
            re["recommended_action"] == "C" and re["options"]["C_none"] is True)

    # --- Reward tier heuristic: every severity band lands on a valid tier ---
    tiers_seen = {intervention._reward_for_severity(s / 20.0) for s in range(0, 21)}
    c.check("R7.3 reward heuristic only ever yields configured tiers",
            tiers_seen.issubset(set(config.REWARD_TIERS)), str(sorted(tiers_seen)))
    c.check("R7.3 reward heuristic spans all tiers across severity",
            tiers_seen == set(config.REWARD_TIERS), str(sorted(tiers_seen)))

    # --- Missing required key fails clearly ---------------------------------
    raised = False
    try:
        intervention.recommend({"station": "X"}, [])
    except KeyError:
        raised = True
    c.check("missing §9 key raises KeyError", raised)

    # =====================================================================
    # Friend Relay 2.0 -- user-intent-centered scenarios (R7, corrected)
    # =====================================================================
    RELAY_KEYS = {
        "original_station", "partner_station", "mission_type",
        "predicted_problem", "risk_probability", "distance_km",
        "reward_twd", "mission_text",
    }

    # --- FR1: Borrow mission selects a nearby FUTURE-FULL station -----------
    # User wants to borrow at U_borrow. N_full is nearby (~0.35 km) and
    # forecast Medium/High FULL -> borrowing there relieves its full risk.
    # N_far is future-full too but beyond MAX_NEIGHBOR_KM (must be ignored).
    borrow_snap = [
        _mk("U_borrow", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full", 25.0030, 121.5010, 20, 1, 0.05, 0.72, "High"),   # ~0.35 km
        _mk("N_far_full", 25.0500, 121.5500, 20, 1, 0.05, 0.90, "High"),  # >1 km
    ]
    bm = intervention.recommend_borrow_relay("U_borrow", borrow_snap)
    c.check("FR1 borrow mission produced", bm is not None)
    c.check("FR1 borrow mission has all required fields",
            bm is not None and set(bm.keys()) >= RELAY_KEYS,
            str(set(bm.keys()) if bm else None))
    c.check("FR1 borrow selects the nearby future-FULL station",
            bm is not None and bm["partner_station"] == "N_full",
            bm["partner_station"] if bm else None)
    c.check("FR1 borrow mission_type/predicted_problem correct",
            bm is not None and bm["mission_type"] == "borrow"
            and bm["predicted_problem"] == "full")
    c.check("FR1 borrow far future-full station NOT selected (distance filter)",
            bm is not None and bm["partner_station"] != "N_far_full")

    # --- FR2: Borrow never selects a high-SHORTAGE station (harm guard) ------
    # The only nearby candidate is future-SHORTAGE (high). Borrowing there would
    # hurt it -> must be excluded -> no borrow mission.
    borrow_guard_snap = [
        _mk("U_borrow2", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_short_hi", 25.0030, 121.5010, 1, 20, 0.85, 0.10, "High"),  # ~0.35 km
    ]
    bg = intervention.recommend_borrow_relay("U_borrow2", borrow_guard_snap)
    c.check("FR2 borrow never redirects toward a high-SHORTAGE station",
            bg is None, str(bg))

    # --- FR3: Return mission selects a nearby FUTURE-SHORTAGE station --------
    # User wants to return at U_return. N_short is nearby and forecast
    # Medium/High SHORTAGE -> returning there relieves its shortage risk.
    return_snap = [
        _mk("U_return", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_short", 25.0030, 121.5010, 1, 20, 0.70, 0.05, "High"),   # ~0.35 km
        _mk("N_far_short", 25.0500, 121.5500, 1, 20, 0.95, 0.05, "High"),  # >1 km
    ]
    rmn = intervention.recommend_return_relay("U_return", return_snap)
    c.check("FR3 return mission produced", rmn is not None)
    c.check("FR3 return mission has all required fields",
            rmn is not None and set(rmn.keys()) >= RELAY_KEYS,
            str(set(rmn.keys()) if rmn else None))
    c.check("FR3 return selects the nearby future-SHORTAGE station",
            rmn is not None and rmn["partner_station"] == "N_short",
            rmn["partner_station"] if rmn else None)
    c.check("FR3 return mission_type/predicted_problem correct",
            rmn is not None and rmn["mission_type"] == "return"
            and rmn["predicted_problem"] == "shortage")

    # --- FR4: Return never selects a high-FULL station (harm guard) ---------
    # The only nearby candidate is future-FULL (high). Returning there would
    # hurt it -> must be excluded -> no return mission.
    return_guard_snap = [
        _mk("U_return2", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_full_hi", 25.0030, 121.5010, 20, 1, 0.10, 0.85, "High"),  # ~0.35 km
    ]
    rg = intervention.recommend_return_relay("U_return2", return_guard_snap)
    c.check("FR4 return never redirects toward a high-FULL station",
            rg is None, str(rg))

    # --- FR5: Partner station is within MAX_NEIGHBOR_KM ---------------------
    c.check("FR5 borrow partner within MAX_NEIGHBOR_KM",
            bm is not None and bm["distance_km"] <= config.MAX_NEIGHBOR_KM,
            f"{bm['distance_km']:.3f}" if bm else None)
    c.check("FR5 return partner within MAX_NEIGHBOR_KM",
            rmn is not None and rmn["distance_km"] <= config.MAX_NEIGHBOR_KM,
            f"{rmn['distance_km']:.3f}" if rmn else None)

    # --- FR6: Reward is one of {5, 10, 15} ----------------------------------
    c.check("FR6 borrow reward in configured tiers {5,10,15}",
            bm is not None and bm["reward_twd"] in set(config.REWARD_TIERS),
            str(bm["reward_twd"]) if bm else None)
    c.check("FR6 return reward in configured tiers {5,10,15}",
            rmn is not None and rmn["reward_twd"] in set(config.REWARD_TIERS),
            str(rmn["reward_twd"]) if rmn else None)

    # --- FR7: No suitable nearby station -> no Friend Relay mission (None) ---
    # A lone borrow origin (no neighbors) and a borrow origin whose only
    # neighbor is calm (no forecast full risk) both yield no mission.
    lone_snap = [_mk("U_alone", 25.0, 121.5, 6, 6, 0.10, 0.10, "Low")]
    c.check("FR7 no neighbor -> borrow relay None",
            intervention.recommend_borrow_relay("U_alone", lone_snap) is None)
    c.check("FR7 no neighbor -> return relay None",
            intervention.recommend_return_relay("U_alone", lone_snap) is None)
    calm_snap = [
        _mk("U_calm", 25.0000, 121.5000, 6, 6, 0.10, 0.10, "Low"),
        _mk("N_calm", 25.0030, 121.5010, 8, 8, 0.10, 0.10, "Low"),  # nearby, calm
    ]
    c.check("FR7 nearby but no forecast full risk -> borrow relay None",
            intervention.recommend_borrow_relay("U_calm", calm_snap) is None)
    c.check("FR7 nearby but no forecast shortage risk -> return relay None",
            intervention.recommend_return_relay("U_calm", calm_snap) is None)

    # --- B_relay wiring keeps §9 keys AND adds user-intent keys -------------
    c.check("B_relay preserves §9 keys (mission_text/reward_twd/partner_station)",
            {"mission_text", "reward_twd", "partner_station"}.issubset(
                r["options"]["B_relay"].keys()))
    c.check("B_relay enriched with user-intent mission_type/predicted_problem",
            {"mission_type", "predicted_problem"}.issubset(
                r["options"]["B_relay"].keys()))
    # shortage-dominant target -> return-relay style (beneficiary is the target).
    c.check("B_relay for shortage target is a return-relay toward the target",
            r["options"]["B_relay"]["mission_type"] == "return"
            and r["options"]["B_relay"]["partner_station"] == "A_shortage")
    # full-dominant target -> borrow-relay style.
    c.check("B_relay for full target is a borrow-relay toward the target",
            rf["options"]["B_relay"]["mission_type"] == "borrow"
            and rf["options"]["B_relay"]["partner_station"] == "B_full")

    # Informational sample.
    print(f"    info: shortage rec -> action={r['recommended_action']} "
          f"donor={r['options']['A_truck']['donor_or_dest']} "
          f"move={r['options']['A_truck']['move_bikes']} "
          f"reward={r['options']['B_relay']['reward_twd']}")

    return c.done()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
