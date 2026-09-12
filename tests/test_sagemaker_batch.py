# -*- coding: utf-8 -*-
"""Pytest tests for SageMaker BATCH inference (mocked runtime).

Verifies the whole-snapshot path issues exactly TWO invoke_endpoint calls and
maps probabilities back to the right stations in the right order.
No live endpoint, no credentials, no trained models required.
"""

from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from src import config, intervention, predict, sagemaker_predict

FEATURE_COLUMNS = list(config.FEATURE_COLUMNS)
FEATURE_META = {"feature_columns": FEATURE_COLUMNS}


def _row(**over):
    base = {
        "hour": 8.0, "weekday": 1.0, "is_weekend": 0.0, "station_id": 0.0,
        "district_id": 3.0, "total_docks": 20.0, "lon": 121.5, "lat": 25.0,
        "available_bikes": 10.0, "available_docks": 10.0,
        "bike_ratio": 0.5, "dock_ratio": 0.5,
        "prev_available_bikes": 10.0, "bike_change": 0.0, "dock_change": 0.0,
    }
    base.update(over)
    return base


def _frame(n):
    return pd.DataFrame([_row(station_id=float(i)) for i in range(n)])


def _metas(n):
    return [
        {
            "station": f"S{i:03d}", "lat": 25.0 + i * 0.001, "lon": 121.5,
            "current_bikes": i, "current_docks": 20 - i, "total_docks": 20,
        }
        for i in range(n)
    ]


def _timestamps(n):
    return ["2026-03-29 08:00:00"] * n


class BatchRuntime:
    """Fake sagemaker-runtime returning per-endpoint probability sequences."""

    def __init__(self, shortage_seq=None, full_seq=None, raise_for=None,
                 wrong_length_for=None):
        self.shortage_seq = shortage_seq
        self.full_seq = full_seq
        self.raise_for = raise_for or set()
        self.wrong_length_for = wrong_length_for or set()
        self.calls = []

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        payload = json.loads(Body.decode("utf-8"))
        rows = payload["instances"]
        self.calls.append({"endpoint": EndpointName, "rows": rows,
                           "content_type": ContentType})
        if EndpointName in self.raise_for:
            raise RuntimeError("simulated endpoint failure")
        n = len(rows)
        if EndpointName in self.wrong_length_for:
            probs = [0.5] * (n - 1)
        elif EndpointName == config.SAGEMAKER_ENDPOINT_SHORTAGE:
            probs = self.shortage_seq if self.shortage_seq is not None else [0.1] * n
        else:
            probs = self.full_seq if self.full_seq is not None else [0.2] * n
        return {"Body": types.SimpleNamespace(
            read=lambda: json.dumps({"probabilities": list(probs)}).encode("utf-8")
        )}


# =========================================================================== #
# 1. exactly two requests for N stations                                      #
# =========================================================================== #
@pytest.mark.parametrize("n", [1, 5, 50, 763])
def test_batch_issues_exactly_two_requests(n):
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(_frame(n), FEATURE_META, client=client)
    assert len(client.calls) == 2, f"{n} stations must still be 2 requests"
    assert {c["endpoint"] for c in client.calls} == {
        config.SAGEMAKER_ENDPOINT_SHORTAGE,
        config.SAGEMAKER_ENDPOINT_FULL,
    }


def test_batch_sends_all_rows_in_one_payload():
    n = 100
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(_frame(n), FEATURE_META, client=client)
    for call in client.calls:
        assert len(call["rows"]) == n


def test_build_snapshot_batch_is_two_requests_for_763_stations():
    n = 763
    client = BatchRuntime()
    snap = sagemaker_predict.build_snapshot_batch(
        _frame(n), _metas(n), _timestamps(n),
        feature_meta=FEATURE_META, client=client,
    )
    assert len(snap) == n
    assert len(client.calls) == 2


# =========================================================================== #
# 2 + 3. probabilities map to the correct model slot                          #
# =========================================================================== #
def test_shortage_and_full_probabilities_map_to_correct_slots():
    n = 4
    shortage = [0.91, 0.10, 0.55, 0.02]
    full = [0.03, 0.80, 0.20, 0.01]
    client = BatchRuntime(shortage_seq=shortage, full_seq=full)
    got_short, got_full = sagemaker_predict.predict_probs_batch(
        _frame(n), FEATURE_META, client=client
    )
    assert got_short == pytest.approx(shortage)
    assert got_full == pytest.approx(full)


def test_snapshot_assigns_each_probability_to_its_own_station():
    n = 4
    shortage = [0.91, 0.10, 0.55, 0.02]
    full = [0.03, 0.80, 0.20, 0.01]
    client = BatchRuntime(shortage_seq=shortage, full_seq=full)
    snap = sagemaker_predict.build_snapshot_batch(
        _frame(n), _metas(n), _timestamps(n),
        feature_meta=FEATURE_META, client=client,
    )
    for i, pred in enumerate(snap):
        assert pred["station"] == f"S{i:03d}"
        assert pred["shortage_prob"] == pytest.approx(shortage[i])
        assert pred["full_prob"] == pytest.approx(full[i])


def test_batch_clips_out_of_range_probabilities():
    client = BatchRuntime(shortage_seq=[1.7, -0.4], full_seq=[-1.0, 2.0])
    short, full = sagemaker_predict.predict_probs_batch(
        _frame(2), FEATURE_META, client=client
    )
    assert short == [1.0, 0.0]
    assert full == [0.0, 1.0]


# =========================================================================== #
# 4. row order is preserved                                                   #
# =========================================================================== #
def test_row_order_is_preserved_in_payload():
    n = 10
    frame = _frame(n)
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(frame, FEATURE_META, client=client)
    sent = client.calls[0]["rows"]
    # station_id was set to the row index, so order is directly checkable.
    assert [r["station_id"] for r in sent] == [float(i) for i in range(n)]


def test_every_row_is_sent_in_feature_meta_column_order():
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(_frame(5), FEATURE_META, client=client)
    for row in client.calls[0]["rows"]:
        assert list(row.keys()) == FEATURE_COLUMNS


def test_shuffled_frame_columns_are_normalized():
    frame = _frame(3)[list(reversed(FEATURE_COLUMNS))]
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(frame, FEATURE_META, client=client)
    for row in client.calls[0]["rows"]:
        assert list(row.keys()) == FEATURE_COLUMNS


def test_both_endpoints_receive_the_same_rows():
    client = BatchRuntime()
    sagemaker_predict.predict_probs_batch(_frame(6), FEATURE_META, client=client)
    assert client.calls[0]["rows"] == client.calls[1]["rows"]


# =========================================================================== #
# 5. schema equals the local backend                                          #
# =========================================================================== #
LOCAL_SCHEMA = {
    "station", "lat", "lon", "current_bikes", "current_docks", "total_docks",
    "shortage_prob", "full_prob", "risk_level", "expected_risk_time",
}


def test_batch_snapshot_schema_matches_local_backend():
    n = 3
    client = BatchRuntime()
    snap = sagemaker_predict.build_snapshot_batch(
        _frame(n), _metas(n), _timestamps(n),
        feature_meta=FEATURE_META, client=client,
    )
    for pred in snap:
        assert set(pred.keys()) == LOCAL_SCHEMA


def test_batch_risk_level_and_horizon_match_local_semantics():
    client = BatchRuntime(shortage_seq=[0.05, 0.45, 0.95], full_seq=[0.01, 0.02, 0.03])
    snap = sagemaker_predict.build_snapshot_batch(
        _frame(3), _metas(3), _timestamps(3),
        feature_meta=FEATURE_META, client=client,
    )
    assert [p["risk_level"] for p in snap] == [
        predict.RISK_LOW, predict.RISK_MEDIUM, predict.RISK_HIGH,
    ]
    # 30-minute horizon applied identically to predict.predict_station.
    assert snap[0]["expected_risk_time"] == "2026-03-29T08:30:00"


def test_batch_snapshot_feeds_decision_layer_and_relay():
    """AWS batch output drives the unchanged decision + Friend Relay logic."""
    frame = _frame(2)
    metas = [
        {"station": "T", "lat": 25.000, "lon": 121.5,
         "current_bikes": 1, "current_docks": 19, "total_docks": 20},
        {"station": "D", "lat": 25.003, "lon": 121.5,
         "current_bikes": 18, "current_docks": 2, "total_docks": 20},
    ]
    client = BatchRuntime(shortage_seq=[0.91, 0.05], full_seq=[0.02, 0.10])
    snap = sagemaker_predict.build_snapshot_batch(
        frame, metas, _timestamps(2), feature_meta=FEATURE_META, client=client
    )
    decision = intervention.decide_intervention(snap[0], snap)
    assert decision["station"] == "T"
    assert decision["primary_action"] == intervention.DECISION_TRUCK
    assert decision["urgency"] == intervention.URGENCY_HIGH

    rec = intervention.recommend(snap[0], snap)
    assert rec["options"]["A_truck"]["donor_or_dest"] == "D"


# =========================================================================== #
# 6. failures                                                                 #
# =========================================================================== #
def test_batch_endpoint_failure_raises_endpoint_unavailable():
    client = BatchRuntime(raise_for={config.SAGEMAKER_ENDPOINT_SHORTAGE})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError) as exc:
        sagemaker_predict.predict_probs_batch(
            _frame(3), FEATURE_META, client=client
        )
    assert "deploy_endpoints.py" in str(exc.value)


def test_batch_rejects_length_mismatch_rather_than_misalign():
    """A short response must fail loudly, never silently mis-map stations."""
    client = BatchRuntime(wrong_length_for={config.SAGEMAKER_ENDPOINT_FULL})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError) as exc:
        sagemaker_predict.predict_probs_batch(
            _frame(5), FEATURE_META, client=client
        )
    assert "cannot be trusted" in str(exc.value)


def test_batch_missing_feature_column_raises_before_calling_aws():
    frame = _frame(3).drop(columns=["bike_ratio"])
    client = BatchRuntime()
    with pytest.raises(predict.MissingFeatureError) as exc:
        sagemaker_predict.predict_probs_batch(frame, FEATURE_META, client=client)
    assert "bike_ratio" in str(exc.value)
    assert client.calls == []


def test_empty_frame_makes_no_requests():
    client = BatchRuntime()
    short, full = sagemaker_predict.predict_probs_batch(
        pd.DataFrame(columns=FEATURE_COLUMNS), FEATURE_META, client=client
    )
    assert short == [] and full == []
    assert client.calls == []


def test_length_mismatch_between_metas_and_rows_raises():
    client = BatchRuntime()
    with pytest.raises(ValueError):
        sagemaker_predict.build_snapshot_batch(
            _frame(3), _metas(2), _timestamps(3),
            feature_meta=FEATURE_META, client=client,
        )


# =========================================================================== #
# 7. app wiring: build_snapshot uses batch for SageMaker                      #
# =========================================================================== #
def test_app_build_snapshot_sagemaker_uses_two_requests_only():
    import app

    artifacts = predict.load_artifacts()
    bundle = app.build_feature_bundle()
    ts = app.available_timestamps(bundle)[0]

    client = BatchRuntime()
    snap = app.build_snapshot(
        bundle, ts, artifacts,
        backend=config.BACKEND_SAGEMAKER, sm_client=client,
    )
    assert len(snap) > 100, "expected a realistic multi-station snapshot"
    assert len(client.calls) == 2, "whole snapshot must be exactly 2 requests"


def test_app_local_and_sagemaker_snapshots_have_identical_schema():
    import app

    artifacts = predict.load_artifacts()
    bundle = app.build_feature_bundle()
    ts = app.available_timestamps(bundle)[0]

    local = app.build_snapshot(bundle, ts, artifacts)
    client = BatchRuntime()
    remote = app.build_snapshot(
        bundle, ts, artifacts,
        backend=config.BACKEND_SAGEMAKER, sm_client=client,
    )
    assert len(local) == len(remote)
    assert set(local[0].keys()) == set(remote[0].keys()) == LOCAL_SCHEMA
