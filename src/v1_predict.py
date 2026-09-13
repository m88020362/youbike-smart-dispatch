# -*- coding: utf-8 -*-
"""V1 predictor adapter for the deployed multimodel SageMaker endpoint.

NEW module. src/sagemaker_predict.py (the V0 XGBoost adapter) is untouched and
still works; this sits alongside it.

Endpoint : youbike-v1-multimodel-endpoint  (one endpoint, four LightGBM models)
Contract : exactly the deployed deploy_v1/code/inference.py contract. No new
           payload shape is invented here.

    request  {"day_type": ..., "horizon_minutes": 30|60, "instances": [{...}]}
    response {"predicted_bike_ratio": [...], "model_name": ..., ...}

The authoritative feature order comes from the bundle manifest, mirrored in
FEATURE_ORDER below (weekday 14 incl. is_peak, weekend 13 without). Schema is
validated BEFORE any AWS call so a bad payload never costs an invocation:

  * weekday payload missing is_peak            -> FeatureSchemaError
  * weekend payload carrying is_peak           -> FeatureSchemaError
  * is_peak is NEVER fabricated for weekend

Batch behaviour: one snapshot of N stations costs exactly TWO invoke_endpoint
calls (30m + 60m) regardless of N. Row order is preserved end to end.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

ENDPOINT_NAME = "youbike-v1-multimodel-endpoint"
REGION = "us-west-2"
CONTENT_TYPE = "application/json"
SOURCE_TAG = "AWS_SAGEMAKER_V1"

DAY_WEEKDAY = "weekday"
DAY_WEEKEND = "weekend"
HORIZONS = (30, 60)

# Business event definition -- FIXED, never calibrated.
LOW_EVENT = 0.20
HIGH_EVENT = 0.80

# Authoritative order, copied from the deployed bundle manifest.
_BASE = [
    "current_available_bikes",
    "current_available_docks",
    "total_docks",
    "current_bike_ratio",
    "hour",
    "weekday",
    "lon",
    "lat",
    "nearest_junior_high_distance",
    "nearest_university_distance",
    "nearest_mrt_distance",
    "nearest_bus_distance",
]
FEATURE_ORDER: Dict[str, List[str]] = {
    DAY_WEEKDAY: _BASE + ["is_peak", "rainfall"],   # 14
    DAY_WEEKEND: _BASE + ["rainfall"],              # 13
}

# Per-model calibrated ALERT triggers (early-warning metadata only; the decision
# layer's persistent-state logic uses the fixed 0.20 / 0.80 business events).
ALERT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "weekday_30": {"low": 0.21, "high": 0.62},
    "weekday_60": {"low": 0.25, "high": 0.65},
    "weekend_30": {"low": 0.22, "high": 0.65},
    "weekend_60": {"low": 0.25, "high": 0.55},
}

MODEL_NAMES = {
    ("weekday", 30): "weekday_30m_bike_ratio",
    ("weekday", 60): "weekday_60m_bike_ratio",
    ("weekend", 30): "weekend_30m_bike_ratio",
    ("weekend", 60): "weekend_60m_bike_ratio",
}

DAY_TYPE_SOURCE = "calendar_weekday_demo_rule"


class V1EndpointError(RuntimeError):
    """Base class for every V1 adapter failure. Never silently swallowed."""


class EndpointUnavailableError(V1EndpointError):
    """Endpoint missing, not InService, credentials absent, or invoke failed."""


class FeatureSchemaError(V1EndpointError):
    """Payload does not match the model's exact feature schema."""


class ResponseValidationError(V1EndpointError):
    """Endpoint replied with something unusable."""


# --------------------------------------------------------------------------- #
# day type
# --------------------------------------------------------------------------- #
def resolve_day_type(timestamp) -> str:
    """Mon-Fri -> weekday, Sat/Sun -> weekend.

    KNOWN LIMITATION: this is a calendar rule only. The training weekend dataset
    also contains national holidays (假日與國定假日), but runtime routing here
    does NOT detect them. Recorded as day_type_source =
    'calendar_weekday_demo_rule'; do not claim full holiday handling.
    """
    import pandas as pd

    ts = pd.Timestamp(timestamp)
    return DAY_WEEKEND if ts.weekday() >= 5 else DAY_WEEKDAY


# --------------------------------------------------------------------------- #
# schema validation
# --------------------------------------------------------------------------- #
def validate_rows(rows: Sequence[Dict], day_type: str) -> List[Dict]:
    """Validate + reorder rows to the exact per-day-type feature order.

    Runs BEFORE any AWS call. Rejects both directions of schema mismatch and
    never invents a value.
    """
    if day_type not in FEATURE_ORDER:
        raise FeatureSchemaError(
            f"day_type must be {DAY_WEEKDAY!r} or {DAY_WEEKEND!r}, got {day_type!r}")
    if not rows:
        raise FeatureSchemaError("no feature rows supplied")

    expected = FEATURE_ORDER[day_type]
    expected_set = set(expected)
    out: List[Dict] = []

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise FeatureSchemaError(f"row {i} is {type(row).__name__}, expected dict")
        keys = set(row)

        missing = [c for c in expected if c not in keys]
        if missing:
            raise FeatureSchemaError(
                f"row {i} ({day_type}) missing required feature(s) {missing}. "
                f"This route needs exactly {len(expected)} features: {expected}. "
                f"No feature is fabricated.")

        # weekend must NOT carry is_peak: the weekend models were trained
        # without it, and fabricating is_peak=0 is explicitly forbidden.
        if day_type == DAY_WEEKEND and "is_peak" in keys:
            raise FeatureSchemaError(
                "weekend payload carries 'is_peak', but the weekend models were "
                "trained WITHOUT it (the weekend enriched source has no 尖峰時段 "
                "column). Remove is_peak; it must not be fabricated or passed.")

        extra = keys - expected_set
        if extra:
            raise FeatureSchemaError(
                f"row {i} ({day_type}) has unexpected feature(s) {sorted(extra)}; "
                f"expected exactly {expected}")

        try:
            out.append({c: float(row[c]) for c in expected})
        except (TypeError, ValueError) as e:
            raise FeatureSchemaError(f"row {i}: non-numeric feature value: {e}") from e

    return out


# --------------------------------------------------------------------------- #
# AWS plumbing
# --------------------------------------------------------------------------- #
def get_runtime_client(region: Optional[str] = None):
    try:
        import boto3
    except ImportError as e:
        raise EndpointUnavailableError(
            "boto3 is not installed, so the V1 SageMaker endpoint cannot be "
            "used. Install boto3 or fall back to the stable V0 predictor.") from e
    try:
        return boto3.client("sagemaker-runtime", region_name=region or REGION)
    except Exception as e:
        raise EndpointUnavailableError(
            f"could not create sagemaker-runtime client: {e}") from e


def check_endpoint(endpoint_name: str = ENDPOINT_NAME,
                   region: Optional[str] = None) -> "tuple[bool, str]":
    """Health probe. Returns (ok, message); never raises.

    Uses describe_endpoint (control plane), so it consumes no inference call.
    """
    try:
        import boto3
    except ImportError:
        return False, "boto3 未安裝，無法使用 AWS V1 endpoint。"
    try:
        sm = boto3.client("sagemaker", region_name=region or REGION)
    except Exception as e:
        return False, f"無法建立 SageMaker client：{e}"
    try:
        d = sm.describe_endpoint(EndpointName=endpoint_name)
    except Exception as e:
        return False, (f"Endpoint '{endpoint_name}' 無法查詢（可能不存在或憑證失效）：{e}")
    status = d.get("EndpointStatus")
    if status != "InService":
        return False, f"Endpoint '{endpoint_name}' 狀態為 {status}，尚未 InService。"
    return True, f"Endpoint '{endpoint_name}' InService。"


def _invoke(client, day_type: str, horizon: int, rows: List[Dict],
            endpoint_name: str) -> List[float]:
    """ONE invoke_endpoint call for a whole batch. Order preserved."""
    body = json.dumps({"day_type": day_type, "horizon_minutes": int(horizon),
                       "instances": rows}).encode("utf-8")
    try:
        resp = client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType=CONTENT_TYPE,
            Accept=CONTENT_TYPE, Body=body)
    except Exception as e:
        raise EndpointUnavailableError(
            f"invoke_endpoint failed for {day_type}/{horizon}m on "
            f"'{endpoint_name}': {e}. Confirm the endpoint exists and is "
            f"InService, and that AWS credentials are valid.") from e

    try:
        raw = resp["Body"].read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        parsed = json.loads(raw)
    except Exception as e:
        raise ResponseValidationError(
            f"{day_type}/{horizon}m: response was not valid JSON: {e}") from e

    if "predicted_bike_ratio" not in parsed:
        raise ResponseValidationError(
            f"{day_type}/{horizon}m: response missing 'predicted_bike_ratio'; "
            f"keys={sorted(parsed)}")
    ratios = parsed["predicted_bike_ratio"]
    if len(ratios) != len(rows):
        raise ResponseValidationError(
            f"{day_type}/{horizon}m: got {len(ratios)} predictions for "
            f"{len(rows)} rows; row-to-station mapping cannot be trusted.")

    expected_model = MODEL_NAMES[(day_type, int(horizon))]
    got_model = parsed.get("model_name")
    if got_model != expected_model:
        raise ResponseValidationError(
            f"routing mismatch: asked {day_type}/{horizon}m (expected "
            f"{expected_model}) but endpoint answered as {got_model}")

    n_expected = len(FEATURE_ORDER[day_type])
    n_used = parsed.get("n_features_used")
    if n_used is not None and int(n_used) != n_expected:
        raise ResponseValidationError(
            f"{expected_model}: endpoint used {n_used} features, expected "
            f"{n_expected}")

    out = []
    for j, v in enumerate(ratios):
        try:
            f = float(v)
        except (TypeError, ValueError) as e:
            raise ResponseValidationError(
                f"{expected_model}: prediction {j} is not numeric: {v!r}") from e
        if not (0.0 <= f <= 1.0):
            raise ResponseValidationError(
                f"{expected_model}: prediction {j} = {f} outside [0, 1]")
        out.append(f)
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def predict_batch(rows: Sequence[Dict], day_type: str, client=None,
                  endpoint_name: str = ENDPOINT_NAME) -> List[Dict]:
    """Score N stations for BOTH horizons using exactly TWO endpoint calls.

    Returns one dict per input row, in the SAME order:

        {"day_type", "ratio_30m", "ratio_60m", "model_30m", "model_60m",
         "source": "AWS_SAGEMAKER_V1"}

    Raises:
        FeatureSchemaError / EndpointUnavailableError / ResponseValidationError
    """
    prepared = validate_rows(rows, day_type)      # before any AWS call
    if client is None:
        client = get_runtime_client()

    r30 = _invoke(client, day_type, 30, prepared, endpoint_name)
    r60 = _invoke(client, day_type, 60, prepared, endpoint_name)

    m30 = MODEL_NAMES[(day_type, 30)]
    m60 = MODEL_NAMES[(day_type, 60)]
    return [
        {"day_type": day_type,
         "ratio_30m": a, "ratio_60m": b,
         "model_30m": m30, "model_60m": m60,
         "source": SOURCE_TAG}
        for a, b in zip(r30, r60)
    ]


def predict_one(row: Dict, day_type: str, client=None,
                endpoint_name: str = ENDPOINT_NAME) -> Dict:
    """Single-station convenience wrapper (still two endpoint calls)."""
    return predict_batch([row], day_type, client=client,
                         endpoint_name=endpoint_name)[0]


def alert_flags(prediction: Dict) -> Dict:
    """Per-model calibrated early-warning flags for one prediction dict.

    Alert thresholds are METADATA ONLY. They are deliberately separate from the
    business events (0.20 / 0.80) that drive the decision layer, and must not be
    treated as a dispatch decision on their own.
    """
    dt = prediction["day_type"]
    a30 = ALERT_THRESHOLDS[f"{dt}_30"]
    a60 = ALERT_THRESHOLDS[f"{dt}_60"]
    r30, r60 = prediction["ratio_30m"], prediction["ratio_60m"]
    return {
        "alert_low_30m": bool(r30 < a30["low"]),
        "alert_high_30m": bool(r30 > a30["high"]),
        "alert_low_60m": bool(r60 < a60["low"]),
        "alert_high_60m": bool(r60 > a60["high"]),
        "thresholds": {"30m": dict(a30), "60m": dict(a60)},
        "purpose": "early_warning_metadata_only",
        "business_event_low": LOW_EVENT,
        "business_event_high": HIGH_EVENT,
    }


# --------------------------------------------------------------------------- #
# static feature lookup (backend only; not wired to any UI)
# --------------------------------------------------------------------------- #
_STATIC_CACHE = None
STATIC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "v1_training", "runtime", "station_features.parquet")

STATIC_COLUMNS = ["total_docks", "lon", "lat",
                  "nearest_junior_high_distance", "nearest_university_distance",
                  "nearest_mrt_distance", "nearest_bus_distance"]


def load_station_static(path: Optional[str] = None):
    """Load the small per-station static feature table (cached in-process).

    ~100 KB parquet for 1,583 stations, so runtime never touches the 1.1 GB
    enriched CSV.
    """
    global _STATIC_CACHE
    if _STATIC_CACHE is not None:
        return _STATIC_CACHE
    import pandas as pd

    p = path or STATIC_PATH
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"station static features not found at {p}. Build it with "
            f"v1_training/build_station_features.py")
    df = pd.read_parquet(p).set_index("station")
    _STATIC_CACHE = df
    return df


def static_for_station(station: str, path: Optional[str] = None) -> Dict:
    """Static feature dict for one station, or raise if unknown."""
    df = load_station_static(path)
    if station not in df.index:
        raise KeyError(
            f"station {station!r} has no static enriched features; "
            f"{len(df)} stations available")
    row = df.loc[station]
    return {c: float(row[c]) for c in STATIC_COLUMNS if c in df.columns}
