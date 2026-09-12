# -*- coding: utf-8 -*-
"""Local parity check: src/predict.py vs deploy/code/inference.py.

Builds real feature rows from the stable dataset through the existing pipeline
(data_loader -> build_targets -> build_features), picks one observation, and
compares the shortage/full positive-class probabilities produced by:

  (A) src/predict.py.predict_station  (the stable local path)
  (B) deploy/code/inference.py        (the SageMaker handler), simulated by
      pointing model_fn at a temp dir holding one *_xgb.json + feature_meta.json

No model, dataset, or core src/ code is modified. Read-only over stable
artifacts; only creates temp dirs for the simulation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src import config, data_loader, features, predict  # noqa: E402

# Load deploy/code/inference.py as an isolated module (not on sys.path).
_INFER_PATH = os.path.join(PROJECT_ROOT, "deploy", "code", "inference.py")
_spec = importlib.util.spec_from_file_location("sm_inference", _INFER_PATH)
inference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inference)


def _simulate_endpoint(model_filename: str, feature_row: pd.Series) -> float:
    """Simulate one single-model SageMaker artifact + full handler roundtrip."""
    tmp = tempfile.mkdtemp(prefix="sm_parity_")
    try:
        shutil.copy(
            os.path.join(config.MODELS_DIR, model_filename),
            os.path.join(tmp, model_filename),
        )
        shutil.copy(
            os.path.join(config.MODELS_DIR, "feature_meta.json"),
            os.path.join(tmp, "feature_meta.json"),
        )
        bundle = inference.model_fn(tmp)
        # Build a JSON request exactly like a real client would.
        payload = json.dumps(feature_row.to_dict())
        df = inference.input_fn(payload, "application/json")
        preds = inference.predict_fn(df, bundle)
        return preds[0]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    df = data_loader.load_clean()
    targets = features.build_targets(df)
    bundle = features.build_features(targets)

    idx = 0
    feature_row = bundle.X.iloc[idx]
    ts = bundle.timestamp.iloc[idx]

    meta = {
        "station": "(parity)",
        "lat": 0.0,
        "lon": 0.0,
        "current_bikes": int(feature_row[config.COL_AVAILABLE_BIKES]),
        "current_docks": int(feature_row[config.COL_AVAILABLE_DOCKS]),
        "total_docks": int(feature_row[config.COL_TOTAL_DOCKS]),
    }

    # (A) stable local path
    local = predict.predict_station(feature_row, meta, ts)
    local_shortage = local["shortage_prob"]
    local_full = local["full_prob"]

    # (B) SageMaker handler path
    ep_shortage = _simulate_endpoint("shortage_xgb.json", feature_row)
    ep_full = _simulate_endpoint("full_xgb.json", feature_row)

    print("=" * 68)
    print("Parity check: src/predict.py  vs  deploy/code/inference.py")
    print(f"observation timestamp: {ts}")
    print(f"feature columns ({len(feature_row)}): {list(feature_row.index)}")
    print("-" * 68)
    print(f"{'model':<10}{'src/predict.py':>22}{'inference.py':>22}")
    print(f"{'shortage':<10}{local_shortage:>22.17f}{ep_shortage:>22.17f}")
    print(f"{'full':<10}{local_full:>22.17f}{ep_full:>22.17f}")
    print("-" * 68)

    tol = 1e-9
    d_short = abs(local_shortage - ep_shortage)
    d_full = abs(local_full - ep_full)
    print(f"abs diff shortage: {d_short:.3e}")
    print(f"abs diff full    : {d_full:.3e}")

    ok = d_short <= tol and d_full <= tol
    print("=" * 68)
    print(f"RESULT: {'PASS - probabilities identical' if ok else 'FAIL - mismatch'} (tol={tol:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
