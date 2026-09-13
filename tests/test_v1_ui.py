# -*- coding: utf-8 -*-
"""V1 UI integration tests. AWS fully mocked; no live endpoint contacted."""

from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from src import v1_decision as vd
from src import v1_predict as vp
from src import v1_ui as ui


class FakeRuntime:
    """Returns caller-supplied ratios per horizon, in row order."""

    def __init__(self, r30, r60):
        self.r30, self.r60 = list(r30), list(r60)
        self.calls = []

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        p = json.loads(Body.decode("utf-8"))
        self.calls.append(p)
        n = len(p["instances"])
        vals = self.r30 if p["horizon_minutes"] == 30 else self.r60
        body = json.dumps({
            "model_name": vp.MODEL_NAMES[(p["day_type"], int(p["horizon_minutes"]))],
            "day_type": p["day_type"], "horizon_minutes": p["horizon_minutes"],
            "n_features_used": len(vp.FEATURE_ORDER[p["day_type"]]),
            "predicted_bike_ratio": vals[:n]})
        return {"Body": types.SimpleNamespace(read=lambda: body.encode("utf-8"))}


def station(name, lat, lon, bikes=10, docks=10, total=20, **over):
    r = {"station": name, "city": "新北市", "district": "板橋區",
         "lat": lat, "lon": lon, "total_docks": float(total),
         "current_available_bikes": float(bikes),
         "current_available_docks": float(docks),
         "current_bike_ratio": bikes / total, "hour": 8.0, "weekday": 1.0,
         "is_peak": 1.0, "rainfall": 0.0,
         "nearest_junior_high_distance": 900.0,
         "nearest_university_distance": 2000.0,
         "nearest_mrt_distance": 5000.0, "nearest_bus_distance": 800.0}
    r.update(over)
    return r


def pred(day_type, r30, r60):
    return {"day_type": day_type, "ratio_30m": r30, "ratio_60m": r60,
            "model_30m": vp.MODEL_NAMES[(day_type, 30)],
            "model_60m": vp.MODEL_NAMES[(day_type, 60)],
            "source": "AWS_SAGEMAKER_V1"}


# =========================================================================== #
# risk semantics per mode
# =========================================================================== #
def test_borrow_mode_uses_empty_bike_risk():
    p = pred("weekday", 0.05, 0.05)          # very few bikes -> cannot borrow
    r = ui.risk_pair(p, ui.MODE_BORROW)
    assert r["risk_30m"] == ui.RISK_HIGH
    assert r["risk_meaning"] == "借不到車的風險"


def test_return_mode_uses_full_dock_risk():
    p = pred("weekday", 0.95, 0.95)          # almost full -> cannot return
    r = ui.risk_pair(p, ui.MODE_RETURN)
    assert r["risk_30m"] == ui.RISK_HIGH
    assert r["risk_meaning"] == "沒有空位可還的風險"


def test_same_ratio_reads_opposite_between_modes():
    p = pred("weekday", 0.05, 0.05)
    assert ui.risk_pair(p, ui.MODE_BORROW)["risk_30m"] == ui.RISK_HIGH
    assert ui.risk_pair(p, ui.MODE_RETURN)["risk_30m"] == ui.RISK_LOW


def test_empty_risk_medium_band_uses_calibrated_alert():
    # weekday/30 low alert = 0.21; 0.20 <= r < 0.21 is MEDIUM
    assert ui.empty_risk_level(0.19, 0.21) == ui.RISK_HIGH
    assert ui.empty_risk_level(0.205, 0.21) == ui.RISK_MEDIUM
    assert ui.empty_risk_level(0.21, 0.21) == ui.RISK_LOW


def test_full_risk_medium_band_uses_calibrated_alert():
    # weekday/30 high alert = 0.62; 0.62 < r <= 0.80 is MEDIUM
    assert ui.full_risk_level(0.85, 0.62) == ui.RISK_HIGH
    assert ui.full_risk_level(0.70, 0.62) == ui.RISK_MEDIUM
    assert ui.full_risk_level(0.62, 0.62) == ui.RISK_LOW


def test_business_event_boundaries_unchanged():
    assert ui.empty_risk_level(0.20, 0.21) == ui.RISK_MEDIUM   # 0.20 not HIGH
    assert ui.full_risk_level(0.80, 0.62) == ui.RISK_MEDIUM    # 0.80 not HIGH


# =========================================================================== #
# reward formula
# =========================================================================== #
def test_reward_zero_extra_distance_is_none():
    assert ui.reward_for_extra_distance(0) is None
    assert ui.reward_for_extra_distance(-10) is None


def test_reward_over_300m_is_none():
    assert ui.reward_for_extra_distance(300.1) is None
    assert ui.reward_for_extra_distance(1000) is None


def test_reward_range_is_5_to_10():
    for e in (1, 50, 150, 299, 300):
        r = ui.reward_for_extra_distance(e)
        assert r is not None and 5 <= r <= 10


def test_reward_scales_with_distance():
    assert ui.reward_for_extra_distance(300) == 10
    assert ui.reward_for_extra_distance(270) == 10   # ~270m -> about +$10
    assert ui.reward_for_extra_distance(1) == 5
    assert (ui.reward_for_extra_distance(150)
            <= ui.reward_for_extra_distance(250))


# =========================================================================== #
# reward candidacy
# =========================================================================== #
def test_return_mode_rewards_low_risk_station():
    row = station("A", 25.0, 121.5, docks=5)
    assert ui.is_reward_candidate(row, ui.MODE_RETURN, vd.PERSISTENT_LOW, "BASE")
    assert ui.is_reward_candidate(row, ui.MODE_RETURN, vd.EMERGING_LOW, "BASE")
    assert not ui.is_reward_candidate(row, ui.MODE_RETURN, vd.PERSISTENT_HIGH, "BASE")
    assert not ui.is_reward_candidate(row, ui.MODE_RETURN, vd.NORMAL, "BASE")


def test_borrow_mode_rewards_high_risk_station():
    row = station("A", 25.0, 121.5, bikes=9)
    assert ui.is_reward_candidate(row, ui.MODE_BORROW, vd.PERSISTENT_HIGH, "BASE")
    assert ui.is_reward_candidate(row, ui.MODE_BORROW, vd.EMERGING_HIGH, "BASE")
    assert not ui.is_reward_candidate(row, ui.MODE_BORROW, vd.PERSISTENT_LOW, "BASE")


def test_baseline_station_never_rewarded():
    row = station("BASE", 25.0, 121.5, docks=5)
    assert not ui.is_reward_candidate(row, ui.MODE_RETURN, vd.PERSISTENT_LOW, "BASE")


def test_return_needs_a_free_dock():
    row = station("A", 25.0, 121.5, docks=0)
    assert not ui.is_reward_candidate(row, ui.MODE_RETURN, vd.PERSISTENT_LOW, "BASE")


def test_borrow_needs_a_bike():
    row = station("A", 25.0, 121.5, bikes=0)
    assert not ui.is_reward_candidate(row, ui.MODE_BORROW, vd.PERSISTENT_HIGH, "BASE")


# =========================================================================== #
# user table assembly
# =========================================================================== #
def _two_station_setup(mode, r30, r60):
    base = station("BASE", 25.0000, 121.5000, bikes=10, docks=10)
    far = station("FAR", 25.0018, 121.5000, bikes=10, docks=10)   # ~200 m
    rows = ui.nearby_stations([base, far], 25.0000, 121.5000, 500)
    preds = [pred("weekday", 0.5, 0.5), pred("weekday", r30, r60)]
    return ui.build_user_rows(rows, preds, mode, max_rows=5)


def test_no_imbalance_means_all_rewards_dash():
    out = _two_station_setup(ui.MODE_RETURN, 0.5, 0.5)
    assert all(r["reward_twd"] is None for r in out)
    assert all(ui.reward_text(r) == "—" for r in out)


def test_return_case_produces_reward_between_5_and_10():
    out = _two_station_setup(ui.MODE_RETURN, 0.05, 0.05)   # persistent_low
    far = [r for r in out if r["station"] == "FAR"][0]
    assert far["risk_state"] == vd.PERSISTENT_LOW
    assert far["reward_twd"] is not None
    assert 5 <= far["reward_twd"] <= 10
    assert ui.reward_text(far).startswith("+$")


def test_borrow_case_produces_reward():
    out = _two_station_setup(ui.MODE_BORROW, 0.95, 0.95)   # persistent_high
    far = [r for r in out if r["station"] == "FAR"][0]
    assert far["risk_state"] == vd.PERSISTENT_HIGH
    assert far["reward_twd"] is not None


def test_baseline_is_flagged_and_table_limited_to_five():
    stations = [station(f"S{i}", 25.0 + i * 0.0002, 121.5) for i in range(9)]
    rows = ui.nearby_stations(stations, 25.0, 121.5, 2000)
    preds = [pred("weekday", 0.5, 0.5)] * len(rows)
    out = ui.build_user_rows(rows, preds, ui.MODE_RETURN)
    assert len(out) == 5
    assert sum(1 for r in out if r["is_baseline"]) == 1


def test_availability_text_follows_mode():
    r = {"current_available_bikes": 7, "current_available_docks": 3}
    assert ui.availability_text(r, ui.MODE_BORROW) == "可借 7"
    assert ui.availability_text(r, ui.MODE_RETURN) == "可還 3"


# =========================================================================== #
# operator table
# =========================================================================== #
def test_operator_generates_four_risk_columns():
    rows = ui.build_operator_rows([station("A", 25.0, 121.5)],
                                  [pred("weekday", 0.05, 0.95)])
    r = rows[0]
    for c in ("full_risk_30m", "full_risk_60m", "empty_risk_30m", "empty_risk_60m"):
        assert c in r and r[c] in (ui.RISK_HIGH, ui.RISK_MEDIUM, ui.RISK_LOW)
    assert r["empty_risk_30m"] == ui.RISK_HIGH      # 0.05 -> cannot borrow
    assert r["full_risk_60m"] == ui.RISK_HIGH       # 0.95 -> cannot return


def test_operator_address_never_fabricated():
    r = ui.build_operator_rows([station("A", 25.0, 121.5)],
                               [pred("weekday", 0.5, 0.5)])[0]
    assert r["address"] == "新北市板橋區A"
    blank = ui.station_address({"city": "", "district": "", "station": ""})
    assert blank == "地址資料未提供"


def test_operator_sorting_puts_persistent_first():
    sts = [station("NORMAL", 25.0, 121.5), station("PERSIST", 25.01, 121.5)]
    prs = [pred("weekday", 0.5, 0.5), pred("weekday", 0.03, 0.03)]
    ranked = ui.rank_operator_rows(ui.build_operator_rows(sts, prs))
    assert ranked[0]["station"] == "PERSIST"


def test_operator_kpis():
    sts = [station("A", 25.0, 121.5), station("B", 25.01, 121.5),
           station("C", 25.02, 121.5)]
    prs = [pred("weekday", 0.03, 0.03),    # persistent low
           pred("weekday", 0.97, 0.97),    # persistent high
           pred("weekday", 0.50, 0.50)]    # normal
    k = ui.operator_kpis(ui.build_operator_rows(sts, prs))
    assert k["persistent_empty"] == 1
    assert k["persistent_full"] == 1
    assert k["monitored_stations"] == 3
    assert k["high_risk_stations"] == 2


# =========================================================================== #
# batch efficiency through the UI path
# =========================================================================== #
def test_500_stations_still_two_requests():
    rows = [{c: 1.0 for c in vp.FEATURE_ORDER["weekday"]} for _ in range(500)]
    c = FakeRuntime([0.4] * 500, [0.4] * 500)
    out = vp.predict_batch(rows, "weekday", client=c)
    assert len(out) == 500
    assert len(c.calls) == 2


# =========================================================================== #
# fallback wiring in app.py
# =========================================================================== #
def test_app_has_v1_primary_and_v0_fallback():
    import io
    src = io.open("app.py", encoding="utf-8").read()
    assert "v1_app.render(st)" in src
    assert "main_v0()" in src
    assert "V1EndpointError" in src
    assert "已切換至 Stable V0" in src
    # V0 helper functions must still exist untouched
    assert "def build_user_view(" in src
    assert "def build_operator_view(" in src


def test_v1_app_module_importable_without_streamlit():
    from src import v1_app
    assert hasattr(v1_app, "render")
    assert hasattr(v1_app, "load_snapshot")
    assert v1_app.TITLE == "YouBike 預測式供需調度系統"


# =========================================================================== #
# demo snapshot integrity
# =========================================================================== #
def test_demo_snapshots_have_required_features():
    from src import v1_app
    for day_type in ("weekday", "weekend"):
        df = v1_app.load_snapshot(day_type)
        assert len(df) > 500
        for c in vp.FEATURE_ORDER[day_type]:
            assert c in df.columns, f"{day_type} snapshot missing {c}"
        if day_type == "weekend":
            # weekend models were trained WITHOUT is_peak; must not be sent
            assert "is_peak" not in vp.FEATURE_ORDER["weekend"]


def test_demo_cases_are_real_and_have_rewards():
    from src import v1_app
    cases = v1_app.demo_cases()
    for k in ("DEMO_CASE_USER_RETURN", "DEMO_CASE_USER_BORROW"):
        c = cases[k]
        assert c is not None
        assert 5 <= c["reward_twd"] <= 10
        assert 0 < c["extra_distance_m"] <= 300
        assert c["baseline_station"] != c["reward_station"]
    assert cases["DEMO_CASE_USER_RETURN"]["reward_station_state"] in (
        vd.PERSISTENT_LOW, vd.EMERGING_LOW)
    assert cases["DEMO_CASE_USER_BORROW"]["reward_station_state"] in (
        vd.PERSISTENT_HIGH, vd.EMERGING_HIGH)
