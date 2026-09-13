# -*- coding: utf-8 -*-
"""Tests for the V1 backend: predictor adapter + decision layer. AWS fully mocked.

No real endpoint is contacted and no credentials are required.
"""

from __future__ import annotations

import json
import types

import pytest

from src import v1_decision as vd
from src import v1_predict as vp


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeRuntime:
    """Records invoke_endpoint calls; replies with the deployed contract shape."""

    def __init__(self, ratio_30=0.10, ratio_60=0.15, raise_exc=None,
                 wrong_len=False, wrong_model=False, bad_json=False,
                 out_of_range=False):
        self.ratio_30, self.ratio_60 = ratio_30, ratio_60
        self.raise_exc = raise_exc
        self.wrong_len, self.wrong_model = wrong_len, wrong_model
        self.bad_json, self.out_of_range = bad_json, out_of_range
        self.calls = []

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        payload = json.loads(Body.decode("utf-8"))
        self.calls.append(payload)
        if self.raise_exc:
            raise self.raise_exc
        n = len(payload["instances"])
        horizon = payload["horizon_minutes"]
        day = payload["day_type"]
        val = self.ratio_30 if horizon == 30 else self.ratio_60
        if self.out_of_range:
            val = 1.7
        ratios = [val] * (n - 1 if self.wrong_len else n)
        model = ("bogus_model" if self.wrong_model
                 else vp.MODEL_NAMES[(day, int(horizon))])
        body = ("{not json" if self.bad_json else json.dumps({
            "model_name": model,
            "day_type": day,
            "horizon_minutes": horizon,
            "n_features_used": len(vp.FEATURE_ORDER[day]),
            "predicted_bike_ratio": ratios,
        }))
        return {"Body": types.SimpleNamespace(read=lambda: body.encode("utf-8"))}


def wk_row(**over):
    r = {c: 1.0 for c in vp.FEATURE_ORDER["weekday"]}
    r.update({"current_available_bikes": 5, "current_available_docks": 7,
              "total_docks": 12, "current_bike_ratio": 5 / 12,
              "hour": 8, "weekday": 1, "lon": 121.5, "lat": 25.0, "is_peak": 1})
    r.update(over)
    return r


def we_row(**over):
    r = {c: 1.0 for c in vp.FEATURE_ORDER["weekend"]}
    r.update({"current_available_bikes": 5, "current_available_docks": 7,
              "total_docks": 12, "current_bike_ratio": 5 / 12,
              "hour": 14, "weekday": 6, "lon": 121.5, "lat": 25.0})
    r.update(over)
    return r


# =========================================================================== #
# 1-4  feature schema
# =========================================================================== #
def test_weekday_payload_has_14_features_in_manifest_order():
    assert len(vp.FEATURE_ORDER["weekday"]) == 14
    assert vp.FEATURE_ORDER["weekday"][-2:] == ["is_peak", "rainfall"]
    c = FakeRuntime()
    vp.predict_batch([wk_row()], "weekday", client=c)
    sent = c.calls[0]["instances"][0]
    assert list(sent.keys()) == vp.FEATURE_ORDER["weekday"]


def test_weekend_payload_has_13_features_and_no_is_peak():
    assert len(vp.FEATURE_ORDER["weekend"]) == 13
    assert "is_peak" not in vp.FEATURE_ORDER["weekend"]
    c = FakeRuntime()
    vp.predict_batch([we_row()], "weekend", client=c)
    sent = c.calls[0]["instances"][0]
    assert list(sent.keys()) == vp.FEATURE_ORDER["weekend"]
    assert "is_peak" not in sent, "is_peak must never be sent for weekend"


def test_weekend_rejects_fabricated_is_peak():
    c = FakeRuntime()
    with pytest.raises(vp.FeatureSchemaError) as e:
        vp.predict_batch([we_row(is_peak=0)], "weekend", client=c)
    assert "is_peak" in str(e.value)
    assert c.calls == [], "must reject before spending an AWS call"


def test_weekday_missing_is_peak_is_rejected():
    row = wk_row()
    row.pop("is_peak")
    c = FakeRuntime()
    with pytest.raises(vp.FeatureSchemaError) as e:
        vp.predict_batch([row], "weekday", client=c)
    assert "is_peak" in str(e.value)
    assert c.calls == []


def test_unknown_extra_feature_rejected():
    c = FakeRuntime()
    with pytest.raises(vp.FeatureSchemaError):
        vp.predict_batch([wk_row(bogus=1.0)], "weekday", client=c)
    assert c.calls == []


def test_invalid_day_type_rejected():
    with pytest.raises(vp.FeatureSchemaError):
        vp.predict_batch([wk_row()], "holiday", client=FakeRuntime())


# =========================================================================== #
# 5  output contract
# =========================================================================== #
def test_output_schema_and_values():
    c = FakeRuntime(ratio_30=0.11, ratio_60=0.22)
    out = vp.predict_batch([wk_row()], "weekday", client=c)
    assert len(out) == 1
    o = out[0]
    assert set(o) == {"day_type", "ratio_30m", "ratio_60m",
                      "model_30m", "model_60m", "source"}
    assert o["day_type"] == "weekday"
    assert o["ratio_30m"] == pytest.approx(0.11)
    assert o["ratio_60m"] == pytest.approx(0.22)
    assert o["model_30m"] == "weekday_30m_bike_ratio"
    assert o["model_60m"] == "weekday_60m_bike_ratio"
    assert o["source"] == "AWS_SAGEMAKER_V1"


def test_batch_preserves_row_order():
    rows = [wk_row(current_bike_ratio=i / 100.0) for i in range(10)]
    c = FakeRuntime()
    vp.predict_batch(rows, "weekday", client=c)
    sent = [r["current_bike_ratio"] for r in c.calls[0]["instances"]]
    assert sent == [i / 100.0 for i in range(10)]


# =========================================================================== #
# 6-14  temporal risk states + boundaries
# =========================================================================== #
@pytest.mark.parametrize("r30,r60,expected", [
    (0.05, 0.05, vd.PERSISTENT_LOW),
    (0.05, 0.50, vd.TRANSIENT_LOW),
    (0.50, 0.05, vd.EMERGING_LOW),
    (0.95, 0.95, vd.PERSISTENT_HIGH),
    (0.95, 0.50, vd.TRANSIENT_HIGH),
    (0.50, 0.95, vd.EMERGING_HIGH),
    (0.50, 0.50, vd.NORMAL),
])
def test_temporal_states(r30, r60, expected):
    assert vd.classify_temporal_risk(r30, r60) == expected


def test_exact_020_boundary_is_normal_side():
    assert not vd.is_low(0.20)
    assert vd.classify_temporal_risk(0.20, 0.20) == vd.NORMAL
    assert vd.classify_temporal_risk(0.20, 0.19) == vd.EMERGING_LOW
    assert vd.classify_temporal_risk(0.19, 0.20) == vd.TRANSIENT_LOW


def test_exact_080_boundary_is_normal_side():
    assert not vd.is_high(0.80)
    assert vd.classify_temporal_risk(0.80, 0.80) == vd.NORMAL
    assert vd.classify_temporal_risk(0.80, 0.81) == vd.EMERGING_HIGH
    assert vd.classify_temporal_risk(0.81, 0.80) == vd.TRANSIENT_HIGH


def test_state_families_and_persistence():
    assert vd.state_family(vd.PERSISTENT_LOW) == "low"
    assert vd.state_family(vd.EMERGING_HIGH) == "high"
    assert vd.state_family(vd.NORMAL) is None
    assert vd.is_persistent(vd.PERSISTENT_HIGH)
    assert not vd.is_persistent(vd.TRANSIENT_LOW)


# =========================================================================== #
# 15  endpoint unavailable / response validation
# =========================================================================== #
def test_invoke_failure_raises_endpoint_unavailable():
    c = FakeRuntime(raise_exc=RuntimeError("ValidationError: endpoint not found"))
    with pytest.raises(vp.EndpointUnavailableError) as e:
        vp.predict_batch([wk_row()], "weekday", client=c)
    assert "invoke_endpoint failed" in str(e.value)


def test_endpoint_unavailable_is_catchable_for_fallback():
    c = FakeRuntime(raise_exc=RuntimeError("boom"))
    used_fallback = False
    try:
        vp.predict_batch([wk_row()], "weekday", client=c)
    except vp.EndpointUnavailableError:
        used_fallback = True
    assert used_fallback, "UI must be able to fall back to stable V0"


def test_length_mismatch_rejected_not_misaligned():
    c = FakeRuntime(wrong_len=True)
    with pytest.raises(vp.ResponseValidationError) as e:
        vp.predict_batch([wk_row(), wk_row()], "weekday", client=c)
    assert "cannot be trusted" in str(e.value)


def test_routing_mismatch_rejected():
    c = FakeRuntime(wrong_model=True)
    with pytest.raises(vp.ResponseValidationError) as e:
        vp.predict_batch([wk_row()], "weekday", client=c)
    assert "routing mismatch" in str(e.value)


def test_bad_json_rejected():
    with pytest.raises(vp.ResponseValidationError):
        vp.predict_batch([wk_row()], "weekday", client=FakeRuntime(bad_json=True))


def test_out_of_range_prediction_rejected():
    with pytest.raises(vp.ResponseValidationError):
        vp.predict_batch([wk_row()], "weekday", client=FakeRuntime(out_of_range=True))


def test_all_errors_share_one_base_class():
    for exc in (vp.EndpointUnavailableError, vp.FeatureSchemaError,
                vp.ResponseValidationError):
        assert issubclass(exc, vp.V1EndpointError)


# =========================================================================== #
# 16  batch efficiency: request count must not scale with N
# =========================================================================== #
@pytest.mark.parametrize("n", [1, 10, 500])
def test_request_count_is_two_regardless_of_n(n):
    rows = [wk_row() for _ in range(n)]
    c = FakeRuntime()
    out = vp.predict_batch(rows, "weekday", client=c)
    assert len(out) == n
    assert len(c.calls) == 2, f"{n} stations must still be exactly 2 calls"
    assert {c.calls[0]["horizon_minutes"], c.calls[1]["horizon_minutes"]} == {30, 60}
    for call in c.calls:
        assert len(call["instances"]) == n


def test_weekend_snapshot_also_two_calls():
    c = FakeRuntime()
    vp.predict_batch([we_row() for _ in range(50)], "weekend", client=c)
    assert len(c.calls) == 2


# =========================================================================== #
# day type routing + alert metadata
# =========================================================================== #
def test_day_type_rule_and_documented_limitation():
    import pandas as pd
    assert vp.resolve_day_type(pd.Timestamp("2026-05-04")) == "weekday"   # Mon
    assert vp.resolve_day_type(pd.Timestamp("2026-05-08")) == "weekday"   # Fri
    assert vp.resolve_day_type(pd.Timestamp("2026-05-09")) == "weekend"   # Sat
    assert vp.resolve_day_type(pd.Timestamp("2026-05-10")) == "weekend"   # Sun
    assert vp.DAY_TYPE_SOURCE == "calendar_weekday_demo_rule"


def test_alert_thresholds_match_deployed_manifest():
    assert vp.ALERT_THRESHOLDS["weekday_30"] == {"low": 0.21, "high": 0.62}
    assert vp.ALERT_THRESHOLDS["weekday_60"] == {"low": 0.25, "high": 0.65}
    assert vp.ALERT_THRESHOLDS["weekend_30"] == {"low": 0.22, "high": 0.65}
    assert vp.ALERT_THRESHOLDS["weekend_60"] == {"low": 0.25, "high": 0.55}


def test_alert_is_metadata_only_and_separate_from_business_event():
    c = FakeRuntime(ratio_30=0.205, ratio_60=0.10)
    pred = vp.predict_batch([wk_row()], "weekday", client=c)[0]
    flags = vp.alert_flags(pred)
    # 0.205 is NOT a business low event (>= 0.20) but IS a calibrated alert (<0.21)
    assert flags["alert_low_30m"] is True
    assert vd.is_low(0.205) is False
    assert flags["purpose"] == "early_warning_metadata_only"
    assert flags["business_event_low"] == 0.20


# =========================================================================== #
# operator priority backend
# =========================================================================== #
NEAR_MRT = {"nearest_mrt_distance": 120.0, "nearest_bus_distance": 400.0,
            "nearest_junior_high_distance": 900.0,
            "nearest_university_distance": 2000.0}
FAR = {"nearest_mrt_distance": 5000.0, "nearest_bus_distance": 900.0,
       "nearest_junior_high_distance": 3000.0,
       "nearest_university_distance": 8000.0}


def test_priority_persistent_beats_transient():
    p = vd.score_priority(vd.PERSISTENT_LOW, 0.05, 0.05, static=FAR)
    t = vd.score_priority(vd.TRANSIENT_LOW, 0.05, 0.50, static=FAR)
    assert vd.PRIORITY_RANK[p["priority"]] < vd.PRIORITY_RANK[t["priority"]]


def test_priority_normal_is_none_and_cannot_be_boosted():
    n = vd.score_priority(vd.NORMAL, 0.50, 0.50, static=NEAR_MRT, is_peak=1)
    assert n["priority"] == vd.PRIORITY_NONE


def test_peak_and_transport_hub_appear_in_reasons():
    r = vd.score_priority(vd.PERSISTENT_LOW, 0.05, 0.05,
                          static=NEAR_MRT, is_peak=1, day_type="weekday")
    joined = " ".join(r["reasons"])
    assert "peak" in joined
    assert "MRT" in joined or "transport_hub" in joined
    assert r["priority"] == vd.PRIORITY_HIGH
    assert r["context"]["transport_hub_priority"] is True
    assert r["context"]["high_usage_potential"] is True


def test_neutral_naming_only():
    r = vd.score_priority(vd.PERSISTENT_LOW, 0.05, 0.05, static=NEAR_MRT, is_peak=1)
    blob = json.dumps(r, ensure_ascii=False)
    assert "繁榮" not in blob
    assert "transport_hub_priority" in blob
    assert "high_usage_potential" in blob


def test_assess_and_rank_snapshot_order():
    c = FakeRuntime(ratio_30=0.05, ratio_60=0.05)
    preds = vp.predict_batch([wk_row(), wk_row()], "weekday", client=c)
    a = vd.assess_snapshot(preds, stations=["A", "B"],
                           statics=[NEAR_MRT, FAR], is_peaks=[1, 0])
    assert [x["station"] for x in a] == ["A", "B"]
    assert all(x["risk_state"] == vd.PERSISTENT_LOW for x in a)
    ranked = vd.rank_operator_priority(a)
    assert ranked[0]["station"] == "A", "near-MRT peak station should rank first"


def test_assess_station_carries_model_provenance():
    c = FakeRuntime()
    pred = vp.predict_batch([wk_row()], "weekday", client=c)[0]
    a = vd.assess_station(pred, static=FAR, is_peak=0, station="S1")
    assert a["source"] == "AWS_SAGEMAKER_V1"
    assert a["model_30m"] == "weekday_30m_bike_ratio"
    assert a["model_60m"] == "weekday_60m_bike_ratio"


# =========================================================================== #
# static lookup (backend only)
# =========================================================================== #
def test_static_lookup_is_small_and_has_expected_columns():
    df = vp.load_station_static()
    assert len(df) > 1000
    for c in vp.STATIC_COLUMNS:
        assert c in df.columns
    st = df.index[0]
    d = vp.static_for_station(st)
    assert set(vp.STATIC_COLUMNS).issubset(set(d))
    assert d["total_docks"] > 0


def test_static_lookup_unknown_station_raises():
    with pytest.raises(KeyError):
        vp.static_for_station("___no_such_station___")


# =========================================================================== #
# app wiring (V1 is now the primary UI path, with explicit V0 fallback)
# =========================================================================== #
def test_app_wires_v1_with_explicit_v0_fallback():
    """app.py renders V1 first and falls back to V0 loudly, never silently."""
    import io
    src = io.open("app.py", encoding="utf-8").read()
    assert "v1_app" in src and "v1_predict" in src
    assert "v1_app.render(st)" in src
    assert "V1EndpointError" in src
    assert "main_v0()" in src
    # the stable V0 view builders must survive untouched for that fallback
    assert "def build_user_view(" in src
    assert "def build_operator_view(" in src
