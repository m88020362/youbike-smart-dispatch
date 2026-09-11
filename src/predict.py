# -*- coding: utf-8 -*-
"""Prediction output for the YouBike predictive dispatch MVP (Task 4, R5).

V1 inference contract
---------------------
V1 prediction operates on an *already-built* feature row produced by
``features.build_features()``. In that pipeline the lag features
(``prev_available_bikes`` / ``bike_change`` / ``dock_change``) are historically
valid: they come from the same station's previous observation, and
``config.LAG_FILL_VALUE`` is only ever used for a genuine first historical
observation *inside* ``build_features()``.

This module therefore does NOT reconstruct lag features. It consumes a prepared
feature row (a pandas Series or one-row DataFrame from ``build_features().X``)
plus the observation's metadata (station / lat / lon / current inventory /
timestamp), enforces the exact ``feature_meta["feature_columns"]`` order, runs
both trained XGBoost models, and returns a structured prediction dict:

    load models + feature_meta
      -> take prepared feature row from build_features()
        -> reorder to feature_meta["feature_columns"] (fail if any missing)
          -> predict_proba (positive class), clipped to [0, 1]
            -> shortage_prob / full_prob
              -> risk level (config thresholds, max of the two probs)
                -> structured dict (incl. expected_risk_time = obs ts + 30 min)

Design references:
  * design.md §6 (Prediction), §9 (data structures), §13 (error handling).
  * Requirements R5 criteria 1-3.

Key constraints:
  * All thresholds come from src/config.py (RISK_LOW_MAX, RISK_HIGH_MIN,
    MODELS_DIR, TARGET_*_MINUTES, ...). Nothing hard-coded here.
  * The feature vector is NOT rebuilt here; it must already be a valid
    build_features() row so it is column-for-column identical to training.
  * LAG_FILL_VALUE is NEVER substituted here. If a required prepared feature
    (any column in feature_meta["feature_columns"]) is missing from the
    provided feature row, prediction fails clearly, naming the missing
    column(s) -- nothing is invented or filled.
  * Missing model / feature_meta artifacts raise a clear error telling the user
    to run training first (train.py), per §13.

This module never modifies the raw CSV or any training artifact.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from xgboost import XGBClassifier

from . import config

# Risk-level labels (kept as constants; the numeric thresholds live in config).
RISK_LOW = "Low"
RISK_MEDIUM = "Medium"
RISK_HIGH = "High"

# The 30-minute prediction horizon, taken from the target window midpoint so it
# stays consistent with how targets were defined (config.TARGET_MIN/MAX_MINUTES).
PREDICTION_HORIZON_MINUTES = (
    config.TARGET_MIN_MINUTES + config.TARGET_MAX_MINUTES
) // 2

# Metadata keys the caller must supply alongside a prepared feature row. These
# are the observation facts downstream tasks (Task 5) need; they are NOT model
# features and are never used to rebuild the feature vector.
REQUIRED_META_KEYS = (
    "station",
    "lat",
    "lon",
    "current_bikes",
    "current_docks",
    "total_docks",
)


class ModelArtifactsMissingError(FileNotFoundError):
    """Raised when a required model / feature_meta artifact is missing.

    The message points the user to run training first, per design §13.
    """


class MissingFeatureError(ValueError):
    """Raised when a prepared feature row lacks required feature column(s).

    The V1 contract forbids inventing or filling missing features (no
    LAG_FILL_VALUE substitution at predict time), so this fails clearly and
    names the offending column(s).
    """


def _artifact_paths() -> Dict[str, str]:
    """Absolute paths to the three artifacts produced by train.py."""
    return {
        "shortage_model": os.path.join(config.MODELS_DIR, "shortage_xgb.json"),
        "full_model": os.path.join(config.MODELS_DIR, "full_xgb.json"),
        "feature_meta": os.path.join(config.MODELS_DIR, "feature_meta.json"),
    }


def load_artifacts() -> Dict:
    """Load trained models + feature meta from config.MODELS_DIR.

    Returns:
        A dict with keys: "shortage_model", "full_model" (fitted
        XGBClassifier instances) and "feature_meta" (the parsed
        feature_meta.json dict).

    Raises:
        ModelArtifactsMissingError: If any of the three artifacts is absent,
            with a message instructing the user to train first (train.py).
    """
    paths = _artifact_paths()
    missing = [name for name, p in paths.items() if not os.path.exists(p)]
    if missing:
        missing_files = ", ".join(os.path.basename(paths[m]) for m in missing)
        raise ModelArtifactsMissingError(
            f"Missing model artifact(s): {missing_files}. "
            f"Expected them under '{config.MODELS_DIR}'. "
            f"Run training first (python -m src.train) to produce the models "
            f"and feature_meta before predicting."
        )

    with open(paths["feature_meta"], "r", encoding="utf-8") as f:
        feature_meta = json.load(f)

    shortage_model = XGBClassifier()
    shortage_model.load_model(paths["shortage_model"])
    full_model = XGBClassifier()
    full_model.load_model(paths["full_model"])

    return {
        "shortage_model": shortage_model,
        "full_model": full_model,
        "feature_meta": feature_meta,
    }


def _classify_risk(prob: float) -> str:
    """Map a probability to Low / Medium / High using config thresholds.

    prob < RISK_LOW_MAX (0.3)                    -> Low
    RISK_LOW_MAX <= prob < RISK_HIGH_MIN (0.6)   -> Medium
    prob >= RISK_HIGH_MIN (0.6)                  -> High
    """
    if prob < config.RISK_LOW_MAX:
        return RISK_LOW
    if prob < config.RISK_HIGH_MIN:
        return RISK_MEDIUM
    return RISK_HIGH


def _clip_proba(p: float) -> float:
    """Clamp a probability into [0, 1] to guarantee the R5-2 range contract."""
    return float(min(1.0, max(0.0, p)))


def _feature_columns(feature_meta: Dict) -> List[str]:
    """The exact, ordered feature columns the models were trained on.

    Prefers the persisted feature_meta["feature_columns"] and falls back to
    config.FEATURE_COLUMNS only when the meta lacks it (prior convention).
    """
    return list(feature_meta.get("feature_columns", config.FEATURE_COLUMNS))


def _prepare_feature_frame(
    feature_row: Union[pd.Series, pd.DataFrame],
    feature_meta: Dict,
) -> pd.DataFrame:
    """Validate + reorder a prepared feature row to the exact training order.

    The row must already be a valid build_features() output (historically valid
    lag features included). This selects/reorders columns to match
    feature_meta["feature_columns"] and NEVER fills a missing feature.

    Args:
        feature_row: A prepared feature vector -- a pandas Series (index ==
            feature columns) or a one-row DataFrame (columns == feature
            columns) taken from features.build_features().X.
        feature_meta: Parsed feature_meta.json (provides column order).

    Returns:
        A one-row DataFrame with columns == feature_meta["feature_columns"],
        in that exact order.

    Raises:
        MissingFeatureError: If any required feature column is absent, naming
            the missing column(s). No value is invented/filled.
        ValueError: If a DataFrame with != 1 row is provided.
    """
    columns = _feature_columns(feature_meta)

    if isinstance(feature_row, pd.Series):
        available = set(feature_row.index)
        missing = [c for c in columns if c not in available]
        if missing:
            raise MissingFeatureError(
                f"Prepared feature row is missing required feature column(s): "
                f"{missing}. Provide a valid features.build_features() row; "
                f"predict.py never fills missing features."
            )
        # Reorder to the exact training column order.
        return feature_row.reindex(columns).to_frame().T[columns]

    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError(
                f"Expected a one-row feature DataFrame, got {len(feature_row)} "
                f"rows. Pass a single prepared observation at a time."
            )
        available = set(feature_row.columns)
        missing = [c for c in columns if c not in available]
        if missing:
            raise MissingFeatureError(
                f"Prepared feature row is missing required feature column(s): "
                f"{missing}. Provide a valid features.build_features() row; "
                f"predict.py never fills missing features."
            )
        return feature_row[columns].reset_index(drop=True)

    raise TypeError(
        f"feature_row must be a pandas Series or one-row DataFrame, got "
        f"{type(feature_row).__name__}."
    )


def _validate_meta(meta: Dict) -> None:
    """Ensure observation metadata carries the §9 keys downstream tasks need."""
    missing = [k for k in REQUIRED_META_KEYS if k not in meta]
    if missing:
        raise KeyError(
            f"Observation metadata is missing required key(s): {missing}. "
            f"Expected: {list(REQUIRED_META_KEYS)}."
        )


def predict_station(
    feature_row: Union[pd.Series, pd.DataFrame],
    meta: Dict,
    observation_ts: Union[str, datetime, pd.Timestamp],
    artifacts: Optional[Dict] = None,
) -> Dict:
    """Predict 30-minute shortage/full risk for one prepared observation.

    V1 contract: ``feature_row`` MUST be an already-built feature vector from
    features.build_features() (lag features historically valid). This function
    does not reconstruct any features; it only reorders to the trained column
    order, runs the models, and packages the §9 output.

    Args:
        feature_row: A prepared feature vector for a single observation --
            a pandas Series (index == feature columns) or a one-row DataFrame
            from features.build_features().X.
        meta: Observation metadata / current-state facts (design §9). Required
            keys: "station", "lat", "lon", "current_bikes", "current_docks",
            "total_docks".
        observation_ts: The observation's timestamp (str/datetime/Timestamp),
            i.e. build_features()'s aligned `timestamp` for this row. Used for
            expected_risk_time = observation_ts + 30 min.
        artifacts: Pre-loaded artifacts dict from load_artifacts(). If None,
            artifacts are loaded on demand (raises ModelArtifactsMissingError
            if absent).

    Returns:
        A structured prediction dict (design §9):
            {
              "station", "lat", "lon",
              "current_bikes", "current_docks", "total_docks",
              "shortage_prob", "full_prob",
              "risk_level": "Low|Medium|High",
              "expected_risk_time": str  # observation_ts + 30 min (ISO)
            }

    Raises:
        ModelArtifactsMissingError: If artifacts are missing (via load_artifacts).
        MissingFeatureError: If the prepared row lacks a required feature column.
        KeyError: If a required metadata key is absent.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    _validate_meta(meta)

    ts = pd.Timestamp(observation_ts)

    feature_meta = artifacts["feature_meta"]
    X = _prepare_feature_frame(feature_row, feature_meta)

    shortage_prob = _clip_proba(
        artifacts["shortage_model"].predict_proba(X)[:, 1][0]
    )
    full_prob = _clip_proba(artifacts["full_model"].predict_proba(X)[:, 1][0])

    # Overall risk is driven by the more severe of the two probabilities.
    risk_level = _classify_risk(max(shortage_prob, full_prob))

    expected_risk_time = ts + timedelta(minutes=PREDICTION_HORIZON_MINUTES)

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


def predict_batch(
    observations: List[Tuple[Union[pd.Series, pd.DataFrame], Dict, Union[str, datetime, pd.Timestamp]]],
    artifacts: Optional[Dict] = None,
) -> List[Dict]:
    """Predict for a batch of prepared observations (loads artifacts once).

    Convenience wrapper over predict_station; the whole-snapshot output is what
    the intervention engine (Task 5) consumes to pick neighbor stations.

    Args:
        observations: A list of (feature_row, meta, observation_ts) tuples, each
            a prepared build_features() row plus its metadata and timestamp
            (see predict_station).
        artifacts: Optional pre-loaded artifacts (loaded once if None).

    Returns:
        A list of structured prediction dicts, aligned to `observations`.
    """
    if artifacts is None:
        artifacts = load_artifacts()
    return [
        predict_station(
            feature_row,
            meta,
            observation_ts,
            artifacts=artifacts,
        )
        for feature_row, meta, observation_ts in observations
    ]


if __name__ == "__main__":
    import sys

    from . import data_loader, features

    sys.stdout.reconfigure(encoding="utf-8")

    # Manual smoke: build real feature rows via the pipeline, then predict on
    # one of them (V1 contract: predict consumes build_features() output).
    artifacts = load_artifacts()
    df = data_loader.load_clean()
    targets = features.build_targets(df)
    bundle = features.build_features(targets)

    idx = 0
    feature_row = bundle.X.iloc[idx]
    ts = bundle.timestamp.iloc[idx]
    meta = {
        "station": "(demo)",
        "lat": 0.0,
        "lon": 0.0,
        "current_bikes": int(feature_row[config.COL_AVAILABLE_BIKES]),
        "current_docks": int(feature_row[config.COL_AVAILABLE_DOCKS]),
        "total_docks": int(feature_row[config.COL_TOTAL_DOCKS]),
    }
    result = predict_station(feature_row, meta, ts, artifacts=artifacts)
    print("=" * 64)
    print("Prediction demo (real build_features() row):")
    for k, v in result.items():
        print(f"  {k}: {v}")
