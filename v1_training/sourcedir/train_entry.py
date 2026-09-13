# -*- coding: utf-8 -*-
"""SageMaker script-mode training entry point (runs INSIDE the container).

Reads the preprocessed parquet/csv training + validation channels prepared by
overnight_train.py, fits LightGBM regression with the fixed conservative
parameters, evaluates, and writes model + metrics + metadata to /opt/ml/model
so SageMaker packages them into model.tar.gz.

LightGBM is pip-installed at container start via requirements.txt (script-mode
installs it automatically when present in the source dir).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd


def _read_channel(path: str) -> pd.DataFrame:
    """Read a channel directory. CSV is the expected format.

    Parquet is attempted only if a .parquet file is actually present, and a
    missing engine is reported clearly instead of surfacing as a bare
    ImportError (this container has no pyarrow/fastparquet).
    """
    names = sorted(os.listdir(path))
    print(f"channel {path} contains: {names}", flush=True)

    csvs = [n for n in names if n.endswith(".csv")]
    parquets = [n for n in names if n.endswith(".parquet")]

    # CSV wins outright. A stale .parquet left over from an earlier run in the
    # same S3 channel prefix must never break the job: this container has no
    # parquet engine, so any parquet is skipped whenever CSV data is present.
    if csvs:
        if parquets:
            print(f"ignoring non-CSV files in channel (no parquet engine "
                  f"in this container): {parquets}", flush=True)
        frames = [pd.read_csv(os.path.join(path, n)) for n in csvs]
        print(f"read {len(csvs)} csv file(s) from {path}", flush=True)
        return pd.concat(frames, ignore_index=True)

    if parquets:
        try:
            frames = [pd.read_parquet(os.path.join(path, n)) for n in parquets]
            return pd.concat(frames, ignore_index=True)
        except ImportError as e:
            raise RuntimeError(
                f"channel {path} only has parquet {parquets} but this container "
                f"has no parquet engine (pyarrow/fastparquet). Upload CSV "
                f"channels instead. Original error: {e}"
            ) from e

    raise RuntimeError(f"no .csv/.parquet data files in channel {path}; found {names}")


def main() -> int:
    import lightgbm as lgb
    from sklearn.metrics import (f1_score, mean_absolute_error,
                                 precision_score, r2_score, recall_score,
                                 roc_auc_score)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "model"))
    ap.add_argument("--target", default="future_bike_ratio")
    ap.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    ap.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    ap.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    args, _ = ap.parse_known_args()

    LOW, HIGH = 0.20, 0.80
    PARAMS = dict(objective="regression", n_estimators=800, learning_rate=0.05,
                  num_leaves=31, max_depth=-1, subsample=0.8,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=-1)

    train = _read_channel(args.train)
    valid = _read_channel(args.validation)
    print(f"train={len(train):,} valid={len(valid):,}", flush=True)

    # feature_columns.json is uploaded to the model prefix, NOT into an input
    # channel, so it is normally ABSENT inside the container. The channel data
    # already contains exactly the feature columns plus the target, so deriving
    # the feature list from the dataframe is authoritative. The JSON is honoured
    # only if it happens to be present and consistent.
    features = [c for c in train.columns if c != args.target]
    for candidate in (
        os.path.join(args.train, "feature_columns.json"),
        os.path.join(args.train, "..", "feature_columns.json"),
        "/opt/ml/input/data/feature_columns.json",
    ):
        try:
            if os.path.exists(candidate):
                with open(candidate, "r", encoding="utf-8") as f:
                    declared = json.load(f)["feature_columns"]
                if set(declared).issubset(train.columns):
                    features = declared
                    print(f"feature order taken from {candidate}", flush=True)
                break
        except Exception as e:
            print(f"ignoring unusable {candidate}: {e}", flush=True)
    print(f"features n={len(features)}: {features}", flush=True)

    Xtr, ytr = train[features].astype("float64"), train[args.target].astype("float64")
    Xva, yva = valid[features].astype("float64"), valid[args.target].astype("float64")

    model = lgb.LGBMRegressor(**PARAMS)
    try:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    except TypeError:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
                  early_stopping_rounds=50, verbose=False)

    pred = np.clip(model.predict(Xva), 0.0, 1.0)
    yv = yva.to_numpy()

    def binm(actual, predicted, score):
        out = {"precision": float(precision_score(actual, predicted, zero_division=0)),
               "recall": float(recall_score(actual, predicted, zero_division=0)),
               "f1": float(f1_score(actual, predicted, zero_division=0))}
        out["auc"] = None if len(np.unique(actual)) < 2 else float(roc_auc_score(actual, score))
        return out

    low = binm((yv < LOW).astype(int), (pred < LOW).astype(int), 1.0 - pred)
    high = binm((yv > HIGH).astype(int), (pred > HIGH).astype(int), pred)

    metrics = {
        "MAE": float(mean_absolute_error(yv, pred)),
        "RMSE": float(np.sqrt(np.mean((yv - pred) ** 2))),
        "R2": float(r2_score(yv, pred)),
        "threshold_low": LOW, "threshold_high": HIGH,
        "low_bike_precision": low["precision"], "low_bike_recall": low["recall"],
        "low_bike_f1": low["f1"], "low_bike_auc": low["auc"],
        "high_occupancy_precision": high["precision"], "high_occupancy_recall": high["recall"],
        "high_occupancy_f1": high["f1"], "high_occupancy_auc": high["auc"],
        "best_iteration": getattr(model, "best_iteration_", None),
        "n_evaluated": int(len(yv)),
        "prediction_all_finite": bool(np.isfinite(pred).all()),
    }
    print(json.dumps(metrics, indent=2), flush=True)

    os.makedirs(args.model_dir, exist_ok=True)
    model.booster_.save_model(os.path.join(args.model_dir, f"{args.model_name}_lgbm.txt"))
    with open(os.path.join(args.model_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": features, "target": args.target}, f, indent=2)
    with open(os.path.join(args.model_dir, "container_env.json"), "w", encoding="utf-8") as f:
        json.dump({"LightGBM_version": lgb.__version__,
                   "Python_version": platform.python_version(),
                   "train_rows": int(len(train)), "validation_rows": int(len(valid))},
                  f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
