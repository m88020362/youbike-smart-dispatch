# -*- coding: utf-8 -*-
"""Pytest tests for the SageMaker prediction backend (mocked runtime).

No live endpoint, no AWS credentials and no trained-model files are required:
the sagemaker-runtime client is replaced by a fake that records what was sent
and returns canned responses. This covers:

  1. exactly the 15 features are sent, in feature_meta order
  2. shortage / full responses are parsed into the right slots
  3. AWS probabilities flow correctly into the existing decision layer
  4. endpoint failures raise EndpointUnavailableError with a clear message
  5. the §9 output shape matches the local backend exactly
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config, intervention, predict, sagemaker_predict


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = list(config.FEATURE_COLUMNS)
FEATURE_META = {"feature_columns": FEATURE_COLUMNS}

# The exact row used by every deployment parity check.
PARITY_VALUES = {
    "hour": 0.0,
    "weekday": 6.0,
    "is_weekend": 1.0,
    "station_id": 0.0,
    "district_id": 14.0,
    "total_docks": 12.0,
    "lon": 121.41264,
    "lat": 25.00694,
    "available_bikes": 5.0,
    "available_docks": 7.0,
    "bike_ratio": 0.4166666666666667,
    "dock_ratio": 0.5833333333333334,
    "prev_available_bikes": 0.0,
    "bike_change": 0.0,
    "dock_change": 0.0,
}
PARITY_SHORTAGE = 0.02430786751210690
PARITY_FULL = 0.00598118081688881


class _FakeBody:
    def __init__(self, payload: str):
        self._payload = payload.encode("utf-8")

    def read(self):
        return self._payload


class FakeRuntime:
    """Records invocations and returns a per-endpoint canned probability."""

    def __init__(self, probs=None, raise_for=None, bad_body_for=None):
        self.probs = probs or {
            config.SAGEMAKER_ENDPOINT_SHORTAGE: PARITY_SHORTAGE,
            config.SAGEMAKER_ENDPOINT_FULL: PARITY_FULL,
        }
        self.raise_for = raise_for or set()
        self.bad_body_for = bad_body_for or set()
        self.calls = []

    def invoke_endpoint(self, EndpointName, ContentType, Accept, Body):
        self.calls.append(
            {
                "endpoint": EndpointName,
                "content_type": ContentType,
                "accept": Accept,
                "body": Body,
            }
        )
        if EndpointName in self.raise_for:
            raise RuntimeError("simulated ValidationError: endpoint not found")
        if EndpointName in self.bad_body_for:
            return {"Body": _FakeBody('{"unexpected": true}')}
        return {
            "Body": _FakeBody(
                json.dumps({"probabilities": [self.probs[EndpointName]]})
            )
        }


def _feature_series(**overrides) -> pd.Series:
    values = dict(PARITY_VALUES)
    values.update(overrides)
    return pd.Series(values)


def _meta(station="TestStation"):
    return {
        "station": station,
        "lat": 25.00694,
        "lon": 121.41264,
        "current_bikes": 5,
        "current_docks": 7,
        "total_docks": 12,
    }


# =========================================================================== #
# 1. exactly 15 features, in feature_meta order                               #
# =========================================================================== #
def test_sends_exactly_15_features_in_meta_order():
    client = FakeRuntime()
    sagemaker_predict.predict_probs(_feature_series(), FEATURE_META, client=client)

    assert len(client.calls) == 2, "both endpoints must be invoked once each"
    sent = json.loads(client.calls[0]["body"].decode("utf-8"))

    assert len(sent) == 15
    assert list(sent.keys()) == FEATURE_COLUMNS, "feature order must match meta"
    for name, value in PARITY_VALUES.items():
        assert sent[name] == pytest.approx(value)


def test_reorders_shuffled_input_to_meta_order():
    """A row whose index is shuffled is still sent in the trained order."""
    shuffled = _feature_series()[list(reversed(FEATURE_COLUMNS))]
    client = FakeRuntime()
    sagemaker_predict.predict_probs(shuffled, FEATURE_META, client=client)

    sent = json.loads(client.calls[0]["body"].decode("utf-8"))
    assert list(sent.keys()) == FEATURE_COLUMNS


def test_both_endpoints_receive_identical_payload_and_json_content_type():
    client = FakeRuntime()
    sagemaker_predict.predict_probs(_feature_series(), FEATURE_META, client=client)

    bodies = {c["body"] for c in client.calls}
    assert len(bodies) == 1, "shortage and full must score the same feature row"
    for call in client.calls:
        assert call["content_type"] == "application/json"
        assert call["accept"] == "application/json"


def test_missing_feature_raises_and_names_column():
    incomplete = _feature_series().drop("bike_ratio")
    client = FakeRuntime()
    with pytest.raises(predict.MissingFeatureError) as exc:
        sagemaker_predict.predict_probs(incomplete, FEATURE_META, client=client)
    assert "bike_ratio" in str(exc.value)
    assert client.calls == [], "must not call AWS with an incomplete row"


# =========================================================================== #
# 2. responses parsed into the right slots                                    #
# =========================================================================== #
def test_parses_shortage_and_full_into_correct_slots():
    client = FakeRuntime(
        probs={
            config.SAGEMAKER_ENDPOINT_SHORTAGE: 0.91,
            config.SAGEMAKER_ENDPOINT_FULL: 0.07,
        }
    )
    shortage, full = sagemaker_predict.predict_probs(
        _feature_series(), FEATURE_META, client=client
    )
    assert shortage == pytest.approx(0.91)
    assert full == pytest.approx(0.07)


def test_parses_parity_baseline_values():
    client = FakeRuntime()
    shortage, full = sagemaker_predict.predict_probs(
        _feature_series(), FEATURE_META, client=client
    )
    assert shortage == pytest.approx(PARITY_SHORTAGE, abs=1e-12)
    assert full == pytest.approx(PARITY_FULL, abs=1e-12)


def test_probabilities_are_clipped_into_unit_range():
    client = FakeRuntime(
        probs={
            config.SAGEMAKER_ENDPOINT_SHORTAGE: 1.4,
            config.SAGEMAKER_ENDPOINT_FULL: -0.3,
        }
    )
    shortage, full = sagemaker_predict.predict_probs(
        _feature_series(), FEATURE_META, client=client
    )
    assert shortage == 1.0
    assert full == 0.0


def test_predict_station_returns_same_shape_as_local_backend():
    client = FakeRuntime()
    pred = sagemaker_predict.predict_station(
        _feature_series(),
        _meta(),
        "2026-03-29 00:00:52",
        feature_meta=FEATURE_META,
        client=client,
    )
    # Exactly the design §9 keys the local backend emits.
    assert set(pred.keys()) == {
        "station", "lat", "lon", "current_bikes", "current_docks",
        "total_docks", "shortage_prob", "full_prob", "risk_level",
        "expected_risk_time",
    }
    assert pred["shortage_prob"] == pytest.approx(PARITY_SHORTAGE, abs=1e-12)
    assert pred["full_prob"] == pytest.approx(PARITY_FULL, abs=1e-12)
    # Low probabilities -> Low risk, and the 30-minute horizon is applied.
    assert pred["risk_level"] == predict.RISK_LOW
    assert pred["expected_risk_time"] == "2026-03-29T00:30:52"


def test_risk_level_uses_max_of_the_two_probabilities():
    client = FakeRuntime(
        probs={
            config.SAGEMAKER_ENDPOINT_SHORTAGE: 0.05,
            config.SAGEMAKER_ENDPOINT_FULL: 0.88,
        }
    )
    pred = sagemaker_predict.predict_station(
        _feature_series(), _meta(), "2026-03-29 08:00:00",
        feature_meta=FEATURE_META, client=client,
    )
    assert pred["risk_level"] == predict.RISK_HIGH


# =========================================================================== #
# 3. AWS probabilities flow into the existing decision layer                  #
# =========================================================================== #
def _sagemaker_pred(station, shortage, full, client_probs=None, **meta_over):
    """Build a §9 dict for `station` through the SageMaker backend."""
    client = FakeRuntime(
        probs={
            config.SAGEMAKER_ENDPOINT_SHORTAGE: shortage,
            config.SAGEMAKER_ENDPOINT_FULL: full,
        }
    )
    meta = _meta(station)
    meta.update(meta_over)
    return sagemaker_predict.predict_station(
        _feature_series(), meta, "2026-03-29 08:00:00",
        feature_meta=FEATURE_META, client=client,
    )


def test_aws_probabilities_feed_decide_intervention():
    """A high-shortage AWS prediction drives the unchanged decision layer."""
    target = _sagemaker_pred(
        "AWS_Target", 0.91, 0.03, current_bikes=1, current_docks=20
    )
    donor = _sagemaker_pred(
        "AWS_Donor", 0.05, 0.10, current_bikes=18, current_docks=3,
        lat=25.0097, lon=121.4126,
    )
    decision = intervention.decide_intervention(target, [target, donor])

    assert decision["station"] == "AWS_Target"
    assert decision["risk_level"] == predict.RISK_HIGH
    assert decision["predicted_problem"] == intervention.PROBLEM_SHORTAGE
    assert decision["risk_probability"] == pytest.approx(0.91)
    # Urgent threshold (0.85) crossed -> high urgency, truck primary.
    assert decision["urgency"] == intervention.URGENCY_HIGH
    assert decision["primary_action"] == intervention.DECISION_TRUCK
    assert isinstance(decision["reason"], str) and decision["reason"]


def test_aws_low_risk_prediction_yields_no_intervention():
    target = _sagemaker_pred("AWS_Calm", 0.04, 0.05)
    decision = intervention.decide_intervention(target, [target])
    assert decision["primary_action"] == intervention.DECISION_NONE
    assert decision["urgency"] == intervention.URGENCY_NORMAL


def test_aws_prediction_works_with_recommend():
    """intervention.recommend consumes the AWS §9 dict unchanged."""
    target = _sagemaker_pred(
        "AWS_Full", 0.03, 0.80, current_bikes=22, current_docks=0
    )
    dest = _sagemaker_pred(
        "AWS_Dest", 0.05, 0.05, current_bikes=2, current_docks=19,
        lat=25.0090, lon=121.4130,
    )
    rec = intervention.recommend(target, [target, dest])
    assert rec["options"]["A_truck"] is not None
    assert rec["options"]["A_truck"]["donor_or_dest"] == "AWS_Dest"


# =========================================================================== #
# 4. endpoint unavailable -> clear error (caller can fall back)                #
# =========================================================================== #
def test_shortage_endpoint_failure_raises_endpoint_unavailable():
    client = FakeRuntime(raise_for={config.SAGEMAKER_ENDPOINT_SHORTAGE})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError) as exc:
        sagemaker_predict.predict_probs(
            _feature_series(), FEATURE_META, client=client
        )
    message = str(exc.value)
    assert config.SAGEMAKER_ENDPOINT_SHORTAGE in message
    assert "deploy_endpoints.py" in message


def test_full_endpoint_failure_raises_endpoint_unavailable():
    client = FakeRuntime(raise_for={config.SAGEMAKER_ENDPOINT_FULL})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError) as exc:
        sagemaker_predict.predict_probs(
            _feature_series(), FEATURE_META, client=client
        )
    assert config.SAGEMAKER_ENDPOINT_FULL in str(exc.value)


def test_unparseable_response_raises_endpoint_unavailable():
    client = FakeRuntime(bad_body_for={config.SAGEMAKER_ENDPOINT_SHORTAGE})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError) as exc:
        sagemaker_predict.predict_probs(
            _feature_series(), FEATURE_META, client=client
        )
    assert "could not be parsed" in str(exc.value)


def test_predict_station_propagates_endpoint_unavailable():
    client = FakeRuntime(raise_for={config.SAGEMAKER_ENDPOINT_SHORTAGE})
    with pytest.raises(sagemaker_predict.EndpointUnavailableError):
        sagemaker_predict.predict_station(
            _feature_series(), _meta(), "2026-03-29 08:00:00",
            feature_meta=FEATURE_META, client=client,
        )


def test_endpoint_unavailable_is_catchable_for_fallback():
    """The app-level fallback pattern works: catch, then use the local path."""
    client = FakeRuntime(raise_for={config.SAGEMAKER_ENDPOINT_FULL})
    used_fallback = False
    try:
        sagemaker_predict.predict_probs(
            _feature_series(), FEATURE_META, client=client
        )
    except sagemaker_predict.EndpointUnavailableError:
        used_fallback = True
    assert used_fallback, "app must be able to fall back instead of crashing"


# =========================================================================== #
# 5. backend labels / config wiring                                           #
# =========================================================================== #
def test_default_backend_is_local_so_stable_demo_is_unchanged():
    assert config.PREDICTION_BACKEND == config.BACKEND_LOCAL


def test_backend_labels_resolve():
    import app

    assert app.backend_label(config.BACKEND_LOCAL) == app.BACKEND_LABEL_LOCAL
    assert (
        app.backend_label(config.BACKEND_SAGEMAKER) == app.BACKEND_LABEL_SAGEMAKER
    )


def test_build_snapshot_accepts_backend_argument():
    """build_snapshot exposes the backend switch (wiring contract)."""
    import inspect

    import app

    params = inspect.signature(app.build_snapshot).parameters
    assert "backend" in params
    assert params["backend"].default == config.BACKEND_LOCAL
    assert "sm_client" in params
