# -*- coding: utf-8 -*-
"""SageMaker inference handler for a single YouBike XGBoost model.

Packaged inside each model.tar.gz under ``code/inference.py`` and used by the
SageMaker XGBoost framework container (3.0-5). It reproduces the inference
contract of ``src/predict.py`` for ONE model at a time (shortage OR full).

Why the low-level Booster API instead of XGBClassifier
------------------------------------------------------
The SageMaker XGBoost 3.0-5 container ships xgboost 3.0.5 together with
scikit-learn 1.8.0. scikit-learn 1.8 removed the ``_estimator_type`` attribute
(replaced by ``__sklearn_tags__``), which xgboost 3.0.5's sklearn wrapper still
relies on. As a result ``XGBClassifier().load_model()`` raises:

    TypeError: `_estimator_type` undefined.
               Please use appropriate mixin to define estimator type.

The ``xgboost.Booster`` API does not depend on scikit-learn at all, so it is
immune to sklearn version drift inside the container. It was verified locally to
produce probabilities bit-identical to ``XGBClassifier.predict_proba()[:, 1]``
(absolute difference ~1e-18, i.e. float64 rounding noise).

Contract reproduced from src/predict.py:
  * Enforce the exact ``feature_meta["feature_columns"]`` order (15 features).
  * Never invent / fill a missing feature: fail clearly, naming missing cols.
  * Positive-class probability (binary:logistic Booster.predict output).
  * Clip probability into [0, 1].

The model_dir (``/opt/ml/model``) after extraction contains:
    <name>_xgb.json      (shortage_xgb.json OR full_xgb.json)
    feature_meta.json
    code/inference.py

Only ONE ``*_xgb.json`` is expected per artifact (single-model endpoint).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Union

import pandas as pd
from xgboost import Booster, DMatrix

FEATURE_META_FILENAME = "feature_meta.json"


def _find_model_json(model_dir: str) -> str:
    """Locate the single *_xgb.json model file in the extracted artifact."""
    candidates = sorted(glob.glob(os.path.join(model_dir, "*_xgb.json")))
    if not candidates:
        raise FileNotFoundError(
            f"No '*_xgb.json' model file found in model_dir '{model_dir}'. "
            f"Expected exactly one (shortage_xgb.json or full_xgb.json)."
        )
    if len(candidates) > 1:
        names = ", ".join(os.path.basename(c) for c in candidates)
        raise RuntimeError(
            f"Expected exactly one '*_xgb.json' in '{model_dir}', found: {names}. "
            f"This is a single-model endpoint; package one model per artifact."
        )
    return candidates[0]


def _clip_proba(p: float) -> float:
    """Clamp a probability into [0, 1] (matches src/predict.py._clip_proba)."""
    return float(min(1.0, max(0.0, p)))


def model_fn(model_dir: str) -> Dict:
    """Load the Booster + feature_meta from the extracted artifact.

    Returns a dict: {"booster": Booster, "feature_columns": [...],
    "model_name": "shortage" | "full" | <basename>}.
    """
    model_path = _find_model_json(model_dir)
    meta_path = os.path.join(model_dir, FEATURE_META_FILENAME)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing '{FEATURE_META_FILENAME}' in model_dir '{model_dir}'. "
            f"It must be packaged alongside the model."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        feature_meta = json.load(f)
    feature_columns = list(feature_meta["feature_columns"])

    booster = Booster()
    booster.load_model(model_path)

    base = os.path.basename(model_path)
    model_name = base.replace("_xgb.json", "")

    return {
        "booster": booster,
        "feature_columns": feature_columns,
        "model_name": model_name,
    }


def input_fn(request_body: Union[str, bytes], request_content_type: str = "application/json") -> pd.DataFrame:
    """Parse the request body into a DataFrame of feature rows.

    Accepts application/json in either of these shapes:
      * a single object:            {"hour": 8, "weekday": 1, ...}
      * a list of objects:          [{"hour": 8, ...}, {"hour": 9, ...}]
      * an object with "instances": {"instances": [{...}, {...}]}
    """
    if request_content_type not in ("application/json", "application/jsonlines", None):
        raise ValueError(
            f"Unsupported content type '{request_content_type}'. Use application/json."
        )

    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    payload = json.loads(request_body)

    if isinstance(payload, dict) and "instances" in payload:
        rows = payload["instances"]
    elif isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
            "JSON payload must be an object, a list of objects, or "
            "{'instances': [...]}."
        )

    return pd.DataFrame(rows)


def _prepare_feature_frame(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Validate + reorder to the exact training feature column order.

    Mirrors src/predict.py._prepare_feature_frame: never fills a missing
    feature; fails naming the missing column(s).
    """
    available = set(df.columns)
    missing = [c for c in feature_columns if c not in available]
    if missing:
        raise ValueError(
            f"Input is missing required feature column(s): {missing}. "
            f"Expected all of: {feature_columns}. No feature is invented/filled."
        )
    return df[feature_columns].reset_index(drop=True)


def predict_fn(input_df: pd.DataFrame, model_bundle: Dict) -> List[float]:
    """Run Booster.predict, clipped to [0, 1] (one probability per row).

    For the binary:logistic objective used in training, Booster.predict returns
    the positive-class probability directly, which is numerically identical to
    XGBClassifier.predict_proba()[:, 1].
    """
    booster = model_bundle["booster"]
    feature_columns = model_bundle["feature_columns"]

    X = _prepare_feature_frame(input_df, feature_columns)
    dmatrix = DMatrix(X)
    proba = booster.predict(dmatrix)
    return [_clip_proba(p) for p in proba]


def output_fn(prediction: List[float], accept: str = "application/json") -> str:
    """Serialize probabilities to JSON."""
    if accept not in ("application/json", None):
        raise ValueError(f"Unsupported accept type '{accept}'. Use application/json.")
    return json.dumps({"probabilities": [float(p) for p in prediction]})
