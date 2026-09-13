# -*- coding: utf-8 -*-
"""Single-endpoint inference handler for all four V1 LightGBM models.

One endpoint, four routed models. The request selects a model by day_type +
horizon; the handler applies that model's OWN feature order and returns the
predicted future_bike_ratio plus the model's calibrated alert flags.

    day_type  horizon  -> model                    features
    weekday   30       -> weekday_30m_bike_ratio   14 (includes is_peak)
    weekday   60       -> weekday_60m_bike_ratio   14 (includes is_peak)
    weekend   30       -> weekend_30m_bike_ratio   13 (NO is_peak)
    weekend   60       -> weekend_60m_bike_ratio   13 (NO is_peak)

The weekend enriched source has no 尖峰時段 column, so the weekend models were
trained WITHOUT is_peak. The handler therefore keeps a per-model feature list
and never injects a fabricated is_peak value.

Business event definitions are FIXED and not calibrated:
    low_bike_event       = actual future_bike_ratio < 0.20
    high_occupancy_event = actual future_bike_ratio > 0.80
Only the alert trigger is per-model calibrated (from each model's
threshold_analysis / metadata).

Predictions are clipped to [0, 1] for interpretation, matching training-time
evaluation.

Loads with the low-level lightgbm.Booster API only: no scikit-learn dependency,
so it is immune to sklearn version drift in the serving container.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional, Union

import pandas as pd
from lightgbm import Booster

MANIFEST_FILENAME = "manifest.json"
DAY_WEEKDAY = "weekday"
DAY_WEEKEND = "weekend"
VALID_HORIZONS = (30, 60)


class ModelRoutingError(ValueError):
    """Raised when a request does not identify exactly one available model."""


def _clip(p: float) -> float:
    return float(min(1.0, max(0.0, p)))


def model_fn(model_dir: str) -> Dict:
    """Load the manifest plus all four boosters once at container start."""
    manifest_path = os.path.join(model_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"missing {MANIFEST_FILENAME} in {model_dir}; contents="
            f"{sorted(os.listdir(model_dir))}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    models = {}
    for key, spec in manifest["models"].items():
        path = os.path.join(model_dir, spec["model_file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"model file missing for {key}: {path}")
        booster = Booster(model_file=path)
        n_expected = len(spec["feature_columns"])
        if booster.num_feature() != n_expected:
            raise RuntimeError(
                f"{key}: booster expects {booster.num_feature()} features but "
                f"manifest declares {n_expected}")
        models[key] = {
            "booster": booster,
            "feature_columns": list(spec["feature_columns"]),
            "model_name": spec["model_name"],
            "day_type": spec["day_type"],
            "horizon_minutes": int(spec["horizon_minutes"]),
            "alert_low": float(spec["alert_low_threshold"]),
            "alert_high": float(spec["alert_high_threshold"]),
            "training_job_name": spec.get("training_job_name"),
        }
        print(f"loaded {key}: {spec['model_name']} "
              f"({booster.num_feature()} features, {booster.num_trees()} trees)",
              flush=True)

    return {"manifest": manifest, "models": models,
            "event_low": float(manifest["event_definition"]["low_bike_event_threshold"]),
            "event_high": float(manifest["event_definition"]["high_occupancy_event_threshold"])}


def _route_key(day_type: str, horizon: int) -> str:
    return f"{day_type}_{int(horizon)}m"


def input_fn(request_body: Union[str, bytes],
             request_content_type: str = "application/json") -> Dict:
    """Parse a routed prediction request.

    Accepted JSON shapes:
        {"day_type": "weekday", "horizon_minutes": 30, "features": {...}}
        {"day_type": "weekday", "horizon_minutes": 30, "instances": [{...}, ...]}
    """
    if request_content_type not in ("application/json", None):
        raise ValueError(f"unsupported content type {request_content_type!r}; "
                         f"use application/json")
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object carrying day_type, "
                         "horizon_minutes and features/instances")

    day_type = payload.get("day_type")
    horizon = payload.get("horizon_minutes")
    if day_type not in (DAY_WEEKDAY, DAY_WEEKEND):
        raise ModelRoutingError(
            f"day_type must be {DAY_WEEKDAY!r} or {DAY_WEEKEND!r}, got {day_type!r}")
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        raise ModelRoutingError(f"horizon_minutes must be an int, got {horizon!r}")
    if horizon not in VALID_HORIZONS:
        raise ModelRoutingError(
            f"horizon_minutes must be one of {VALID_HORIZONS}, got {horizon}")

    if "instances" in payload:
        rows = payload["instances"]
    elif "features" in payload:
        rows = [payload["features"]]
    else:
        raise ValueError("payload needs 'features' (single row) or 'instances' (list)")
    if not isinstance(rows, list) or not rows:
        raise ValueError("'instances' must be a non-empty list of objects")

    return {"day_type": day_type, "horizon_minutes": horizon,
            "frame": pd.DataFrame(rows)}


def _prepare(frame: pd.DataFrame, feature_columns: List[str],
             model_name: str) -> pd.DataFrame:
    """Reorder to the model's own training order; never fill a missing feature."""
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{model_name}: input missing required feature column(s) {missing}. "
            f"This model expects exactly: {feature_columns}. "
            f"No feature is invented or filled.")
    return frame[feature_columns].astype("float64").reset_index(drop=True)


def predict_fn(parsed: Dict, bundle: Dict) -> Dict:
    """Route to the requested model and return ratios plus alert flags."""
    key = _route_key(parsed["day_type"], parsed["horizon_minutes"])
    entry = bundle["models"].get(key)
    if entry is None:
        raise ModelRoutingError(
            f"no model for {key}; available: {sorted(bundle['models'])}")

    X = _prepare(parsed["frame"], entry["feature_columns"], entry["model_name"])
    raw = entry["booster"].predict(X.to_numpy())
    ratios = [_clip(p) for p in raw]

    low_t, high_t = entry["alert_low"], entry["alert_high"]
    return {
        "model_name": entry["model_name"],
        "day_type": entry["day_type"],
        "horizon_minutes": entry["horizon_minutes"],
        "training_job_name": entry["training_job_name"],
        "n_features_used": len(entry["feature_columns"]),
        "predicted_bike_ratio": ratios,
        "low_bike_alert": [bool(r < low_t) for r in ratios],
        "high_occupancy_alert": [bool(r > high_t) for r in ratios],
        "alert_thresholds": {"low_bike_alert_below": low_t,
                             "high_occupancy_alert_above": high_t},
        "event_definition": {
            "low_bike_event": f"actual future_bike_ratio < {bundle['event_low']}",
            "high_occupancy_event": f"actual future_bike_ratio > {bundle['event_high']}",
            "calibrated": False},
    }


def output_fn(prediction: Dict, accept: str = "application/json") -> str:
    if accept not in ("application/json", None):
        raise ValueError(f"unsupported accept type {accept!r}; use application/json")
    return json.dumps(prediction, ensure_ascii=False)
