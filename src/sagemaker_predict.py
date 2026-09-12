# -*- coding: utf-8 -*-
"""Thin SageMaker real-time prediction backend (v0 cloud demo).

Purpose
-------
Swap ONLY the probability source. Everything else -- feature engineering, the
15-feature column order, risk-level thresholds, the design §9 output shape, the
intervention / Friend Relay rules and the Streamlit UI -- is reused unchanged
from the existing stable modules.

    already-built 15-feature row  (features.build_features output)
      -> reorder to feature_meta["feature_columns"]   (predict.py helper, reused)
        -> POST to youbike-shortage-v1-endpoint  }  sagemaker-runtime
        -> POST to youbike-full-v1-endpoint      }
          -> shortage_prob / full_prob (clipped to [0, 1])
            -> same §9 dict as predict.predict_station()

This module deliberately does NOT:
  * rebuild or engineer any feature (it consumes a prepared row);
  * re-implement risk classification (it calls predict._classify_risk);
  * re-implement the §9 assembly (it mirrors predict.predict_station exactly by
    reusing that module's helpers);
  * contain any intervention / Friend Relay logic.

The deployed endpoints run deploy/code/inference.py, which enforces the same
feature order and the same [0, 1] clip server-side. Local parity was verified at
an absolute difference of ~1e-18 for both models.

Error handling
--------------
Any AWS-side problem (endpoint absent, not InService, throttled, expired
credentials, boto3 missing) raises EndpointUnavailableError with an actionable
message. Callers are expected to catch it and fall back to the local backend
rather than crashing the app.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Dict, Optional, Tuple, Union

import pandas as pd

from . import config
from . import predict

# Content types the deployed inference.py accepts / returns.
CONTENT_TYPE = "application/json"


class EndpointUnavailableError(RuntimeError):
    """Raised when a SageMaker endpoint cannot serve a prediction.

    Carries a human-readable, actionable message (e.g. "endpoint not found --
    run deploy/deploy_endpoints.py"), so the UI can show it and fall back to the
    local backend instead of crashing.
    """


def get_runtime_client(region: Optional[str] = None):
    """Create a ``sagemaker-runtime`` boto3 client.

    Args:
        region: AWS region; defaults to config.AWS_REGION.

    Raises:
        EndpointUnavailableError: If boto3 is not installed or a client cannot
            be constructed (e.g. no credentials resolvable at all).
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise EndpointUnavailableError(
            "boto3 is not installed, so the SageMaker backend cannot be used. "
            "Install boto3 or switch the prediction backend back to 'local'."
        ) from exc

    try:
        return boto3.client(
            "sagemaker-runtime", region_name=region or config.AWS_REGION
        )
    except Exception as exc:
        raise EndpointUnavailableError(
            f"Could not create a sagemaker-runtime client: {exc}"
        ) from exc


def _feature_payload(
    feature_row: Union[pd.Series, pd.DataFrame],
    feature_meta: Dict,
) -> Tuple[bytes, list]:
    """Reorder the prepared row to the trained order and JSON-encode it.

    Reuses predict._prepare_feature_frame so the column order, the "never fill a
    missing feature" rule and the resulting error messages are byte-for-byte the
    same as the local backend.

    Returns:
        (utf-8 encoded JSON body, the ordered feature column list).
    """
    frame = predict._prepare_feature_frame(feature_row, feature_meta)
    columns = list(frame.columns)
    # Single observation -> one JSON object of feature name -> value.
    values = {c: float(frame.iloc[0][c]) for c in columns}
    return json.dumps(values).encode("utf-8"), columns


def _invoke(client, endpoint_name: str, body: bytes) -> float:
    """Invoke one endpoint and return its single positive-class probability.

    Raises:
        EndpointUnavailableError: On any AWS error or unparseable response, with
            a message naming the endpoint.
    """
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=CONTENT_TYPE,
            Accept=CONTENT_TYPE,
            Body=body,
        )
    except Exception as exc:
        raise EndpointUnavailableError(
            f"Endpoint '{endpoint_name}' could not be invoked: {exc}. "
            f"Confirm it exists and is InService "
            f"(run deploy/deploy_endpoints.py to create it), and that AWS "
            f"credentials are still valid."
        ) from exc

    try:
        raw = response["Body"].read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
        probability = float(parsed["probabilities"][0])
    except Exception as exc:
        raise EndpointUnavailableError(
            f"Endpoint '{endpoint_name}' returned an unexpected response "
            f"that could not be parsed: {exc}"
        ) from exc

    # Same [0, 1] guarantee the local backend gives (server also clips).
    return predict._clip_proba(probability)


def predict_probs(
    feature_row: Union[pd.Series, pd.DataFrame],
    feature_meta: Dict,
    client=None,
    shortage_endpoint: Optional[str] = None,
    full_endpoint: Optional[str] = None,
) -> Tuple[float, float]:
    """Return (shortage_prob, full_prob) from the two deployed endpoints.

    Args:
        feature_row: An already-built 15-feature row (pandas Series or one-row
            DataFrame) from features.build_features().X. NOT re-engineered here.
        feature_meta: Parsed feature_meta.json (supplies the column order).
        client: A sagemaker-runtime client. Created on demand when None.
        shortage_endpoint / full_endpoint: Endpoint name overrides; default to
            the config values.

    Returns:
        (shortage_prob, full_prob), each clipped into [0, 1].

    Raises:
        EndpointUnavailableError: If either endpoint cannot serve the request.
        predict.MissingFeatureError: If a required feature column is absent.
    """
    if client is None:
        client = get_runtime_client()

    body, _ = _feature_payload(feature_row, feature_meta)

    shortage_prob = _invoke(
        client, shortage_endpoint or config.SAGEMAKER_ENDPOINT_SHORTAGE, body
    )
    full_prob = _invoke(
        client, full_endpoint or config.SAGEMAKER_ENDPOINT_FULL, body
    )
    return shortage_prob, full_prob


def predict_station(
    feature_row: Union[pd.Series, pd.DataFrame],
    meta: Dict,
    observation_ts,
    feature_meta: Dict,
    client=None,
) -> Dict:
    """SageMaker-backed twin of predict.predict_station -- identical §9 output.

    The ONLY difference from the local backend is where shortage_prob /
    full_prob come from. Metadata validation, risk classification and
    expected_risk_time all reuse predict.py, so the returned dict is
    indistinguishable in shape and semantics from the local one and can be fed
    straight into intervention.recommend / decide_intervention / the view
    builders.

    Args:
        feature_row: Prepared 15-feature row (see predict_probs).
        meta: Observation metadata (design §9 keys, see predict.REQUIRED_META_KEYS).
        observation_ts: The observation timestamp.
        feature_meta: Parsed feature_meta.json.
        client: Optional sagemaker-runtime client (created on demand).

    Returns:
        The design §9 prediction dict.

    Raises:
        EndpointUnavailableError: If the endpoints cannot serve the request.
        KeyError: If a required metadata key is absent.
    """
    predict._validate_meta(meta)
    ts = pd.Timestamp(observation_ts)

    shortage_prob, full_prob = predict_probs(
        feature_row, feature_meta, client=client
    )

    # Identical downstream semantics to predict.predict_station.
    risk_level = predict._classify_risk(max(shortage_prob, full_prob))
    expected_risk_time = ts + timedelta(
        minutes=predict.PREDICTION_HORIZON_MINUTES
    )

    return {
        "station": str(meta["station"]),
        "lat": float(meta["lat"]),
        "lon": float(meta["lon"]),
        "current_bikes": int(meta["current_bikes"]),
        "current_docks": int(meta["current_docks"]),
        "total_docks": int(meta["total_docks"]),
        "shortage_prob": shortage_prob,
        "full_prob": full_prob,
        "risk_level": risk_level,
        "expected_risk_time": expected_risk_time.isoformat(),
    }


def check_endpoints(client=None) -> Tuple[bool, str]:
    """Cheap reachability probe for both endpoints.

    Used by the UI to decide whether to offer the SageMaker backend or fall back
    to local, WITHOUT raising. Uses describe_endpoint via a sagemaker client
    (not sagemaker-runtime) so it does not consume an inference call.

    Returns:
        (ok, message). ok is True only when both endpoints report InService.
    """
    try:
        import boto3
    except ImportError:
        return False, "boto3 未安裝，無法使用 SageMaker backend。"

    try:
        sm = boto3.client("sagemaker", region_name=config.AWS_REGION)
    except Exception as exc:
        return False, f"無法建立 SageMaker client：{exc}"

    names = (
        config.SAGEMAKER_ENDPOINT_SHORTAGE,
        config.SAGEMAKER_ENDPOINT_FULL,
    )
    statuses = {}
    for name in names:
        try:
            statuses[name] = sm.describe_endpoint(EndpointName=name)[
                "EndpointStatus"
            ]
        except Exception as exc:
            return False, (
                f"Endpoint '{name}' 無法查詢（可能尚未建立）：{exc}. "
                f"請先執行 deploy/deploy_endpoints.py。"
            )

    not_ready = {n: s for n, s in statuses.items() if s != "InService"}
    if not_ready:
        return False, f"Endpoint 尚未就緒：{not_ready}。請等待 InService。"
    return True, "兩個 Endpoint 均為 InService。"


# --------------------------------------------------------------------------- #
# Batch inference -- ONE invoke_endpoint per model for a whole snapshot        #
# --------------------------------------------------------------------------- #
#
# The deployed deploy/code/inference.py accepts {"instances": [ {...}, ... ]}
# and returns {"probabilities": [p0, p1, ...]} in the SAME row order. Verified
# locally: a 3-row batch returned 3 probabilities and batch[0] was bit-identical
# to the equivalent single-row call.
#
# This turns an N-station snapshot from 2*N requests into exactly 2 requests.


def _rows_payload(
    feature_frame: pd.DataFrame,
    feature_meta: Dict,
) -> Tuple[bytes, int]:
    """Reorder every row to the trained order and encode one batch payload.

    Reuses predict._prepare_feature_frame per row so the column order and the
    "never fill a missing feature" rule stay identical to the local backend and
    to the single-row path.

    Returns:
        (utf-8 encoded {"instances": [...]} body, row count).
    """
    columns = predict._feature_columns(feature_meta)

    available = set(feature_frame.columns)
    missing = [c for c in columns if c not in available]
    if missing:
        raise predict.MissingFeatureError(
            f"Snapshot feature frame is missing required feature column(s): "
            f"{missing}. Provide valid features.build_features() rows; "
            f"the SageMaker backend never fills missing features."
        )

    ordered = feature_frame[columns]
    instances = [
        {c: float(row[c]) for c in columns}
        for _, row in ordered.iterrows()
    ]
    body = json.dumps({"instances": instances}).encode("utf-8")
    return body, len(instances)


def _invoke_batch(client, endpoint_name: str, body: bytes, expected: int) -> list:
    """Invoke one endpoint once and return `expected` probabilities in order.

    Raises:
        EndpointUnavailableError: On any AWS error, unparseable response, or a
            response whose length does not match the number of rows sent (which
            would mean the order/row mapping could not be trusted).
    """
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType=CONTENT_TYPE,
            Accept=CONTENT_TYPE,
            Body=body,
        )
    except Exception as exc:
        raise EndpointUnavailableError(
            f"Endpoint '{endpoint_name}' could not be invoked: {exc}. "
            f"Confirm it exists and is InService "
            f"(run deploy/deploy_endpoints.py to create it), and that AWS "
            f"credentials are still valid."
        ) from exc

    try:
        raw = response["Body"].read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        probabilities = json.loads(raw)["probabilities"]
    except Exception as exc:
        raise EndpointUnavailableError(
            f"Endpoint '{endpoint_name}' returned an unexpected response "
            f"that could not be parsed: {exc}"
        ) from exc

    if len(probabilities) != expected:
        raise EndpointUnavailableError(
            f"Endpoint '{endpoint_name}' returned {len(probabilities)} "
            f"probabilities for {expected} rows. Row-to-station mapping cannot "
            f"be trusted, so the batch was rejected."
        )

    return [predict._clip_proba(float(p)) for p in probabilities]


def predict_probs_batch(
    feature_frame: pd.DataFrame,
    feature_meta: Dict,
    client=None,
    shortage_endpoint: Optional[str] = None,
    full_endpoint: Optional[str] = None,
) -> Tuple[list, list]:
    """Score a whole snapshot with exactly TWO invoke_endpoint calls.

    Args:
        feature_frame: One row per station, columns covering the 15 trained
            feature columns (extra columns are ignored, order is normalized).
            Rows are NOT re-engineered here.
        feature_meta: Parsed feature_meta.json (supplies the column order).
        client: sagemaker-runtime client; created on demand when None.
        shortage_endpoint / full_endpoint: Endpoint name overrides.

    Returns:
        (shortage_probs, full_probs) -- two lists of floats in [0, 1], each
        aligned positionally to the input rows.

    Raises:
        EndpointUnavailableError: If either endpoint fails or returns a
            mismatched number of probabilities.
        predict.MissingFeatureError: If a required feature column is absent.
    """
    if client is None:
        client = get_runtime_client()

    body, count = _rows_payload(feature_frame, feature_meta)
    if count == 0:
        return [], []

    shortage_probs = _invoke_batch(
        client, shortage_endpoint or config.SAGEMAKER_ENDPOINT_SHORTAGE,
        body, count,
    )
    full_probs = _invoke_batch(
        client, full_endpoint or config.SAGEMAKER_ENDPOINT_FULL,
        body, count,
    )
    return shortage_probs, full_probs


def build_snapshot_batch(
    feature_frame: pd.DataFrame,
    metas: list,
    timestamps: list,
    feature_meta: Dict,
    client=None,
) -> list:
    """Build a full §9 snapshot from a batch of rows -- 2 requests total.

    The returned list is positionally aligned to ``feature_frame`` /
    ``metas`` / ``timestamps``, and each entry is the SAME design §9 dict the
    local backend produces (same keys, same risk-level thresholds, same
    30-minute horizon), so intervention.* and the view builders consume it
    without any change.

    Args:
        feature_frame: One row per station (see predict_probs_batch).
        metas: Per-station metadata dicts (design §9 keys), same length/order.
        timestamps: Per-station observation timestamps, same length/order.
        feature_meta: Parsed feature_meta.json.
        client: Optional sagemaker-runtime client.

    Returns:
        A list of §9 prediction dicts in input row order.

    Raises:
        EndpointUnavailableError: If the endpoints cannot serve the request.
        ValueError: If the three inputs have mismatched lengths.
        KeyError: If a metadata dict lacks a required key.
    """
    n = len(feature_frame)
    if not (n == len(metas) == len(timestamps)):
        raise ValueError(
            f"Mismatched lengths: {n} feature rows, {len(metas)} metas, "
            f"{len(timestamps)} timestamps. They must align positionally."
        )
    for meta in metas:
        predict._validate_meta(meta)

    shortage_probs, full_probs = predict_probs_batch(
        feature_frame, feature_meta, client=client
    )

    snapshot = []
    for shortage_prob, full_prob, meta, ts in zip(
        shortage_probs, full_probs, metas, timestamps
    ):
        # Identical downstream semantics to predict.predict_station.
        risk_level = predict._classify_risk(max(shortage_prob, full_prob))
        expected_risk_time = pd.Timestamp(ts) + timedelta(
            minutes=predict.PREDICTION_HORIZON_MINUTES
        )
        snapshot.append(
            {
                "station": str(meta["station"]),
                "lat": float(meta["lat"]),
                "lon": float(meta["lon"]),
                "current_bikes": int(meta["current_bikes"]),
                "current_docks": int(meta["current_docks"]),
                "total_docks": int(meta["total_docks"]),
                "shortage_prob": shortage_prob,
                "full_prob": full_prob,
                "risk_level": risk_level,
                "expected_risk_time": expected_risk_time.isoformat(),
            }
        )
    return snapshot
