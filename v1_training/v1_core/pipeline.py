# -*- coding: utf-8 -*-
"""One-model end-to-end pipeline: audit -> target -> features -> split -> train.

Shared by the local fallback and the SageMaker entry point so both paths produce
byte-identical metadata structure and metrics semantics.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import columns as C
from . import dataio, modeling, prep

WEEKDAY = "weekday"
WEEKEND = "weekend"


def filter_day_type(df: pd.DataFrame, day_type: Optional[str],
                    reasons: Dict[str, str]) -> pd.DataFrame:
    """Optionally restrict to Mon-Fri / Sat-Sun by timestamp.

    Only used when the source file is NOT already split by day type. The
    limitation (no national-holiday calendar) is recorded in `reasons` so the
    metadata never overstates what was done.
    """
    if day_type is None:
        return df
    wd = df[C.TIMESTAMP].dt.weekday
    if day_type == WEEKDAY:
        out = df[wd <= 4]
    else:
        out = df[wd >= 5]
    reasons["day_type_filter"] = (
        f"{day_type} derived from timestamp weekday only (Mon-Fri / Sat-Sun). "
        f"No national-holiday calendar was applied, so this is NOT identical to "
        f"the pre-split 平日 / 假日與國定假日 S3 files."
    )
    return out


def run_model(
    model_name: str,
    csv_path: str,
    horizon_minutes: int,
    day_type: Optional[str],
    out_dir: str,
    training_backend: str,
    months_wanted: int = 1,
    smoke_rows: int = 200_000,
    raw_s3_uri: Optional[str] = None,
    include_rainfall: bool = False,
) -> Dict:
    """Train one model and write artifact + metrics + metadata. Returns metadata."""
    import lightgbm as lgb

    os.makedirs(out_dir, exist_ok=True)
    log: List[str] = []

    def say(msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.append(line)

    # ---------- Phase A: audit -------------------------------------------
    say(f"AUDIT {csv_path}")
    audit = dataio.audit_csv(csv_path)
    say(f"  rows_scanned={audit['rows_scanned']:,} "
        f"range={audit['first_timestamp']} -> {audit['last_timestamp']}")
    say(f"  complete_months={audit['complete_months']}")
    say(f"  stations~{audit['station_count_estimate']}")

    months = dataio.pick_months(audit["complete_months"], months_wanted)
    if not months:
        raise RuntimeError(
            f"No complete month found in {csv_path}; month_row_counts="
            f"{audit['month_row_counts']}"
        )
    say(f"  using months={months}")

    # ---------- load only the chosen range ------------------------------
    say("LOAD range (chunked)")
    raw = dataio.load_range(csv_path, audit["encoding"], audit["column_mapping"], months)
    say(f"  loaded rows={len(raw):,}")

    df = prep.normalize_frame(raw, audit["column_mapping"])
    del raw
    df, drop_counts = prep.drop_invalid_rows(df)
    say(f"  after validity drops rows={len(df):,} {drop_counts}")

    reasons: Dict[str, str] = {}
    df = filter_day_type(df, day_type, reasons)
    say(f"  after day_type={day_type} rows={len(df):,}")
    if len(df) == 0:
        raise RuntimeError("no rows left after day-type filter")

    actual_start = str(df[C.TIMESTAMP].min())
    actual_end = str(df[C.TIMESTAMP].max())

    # ---------- target ----------------------------------------------------
    say(f"TARGET horizon={horizon_minutes}m window=+/-5m")
    labeled, tstats = prep.build_target(df, horizon_minutes)
    del df
    say(f"  target rows={len(labeled):,} {tstats}")
    if len(labeled) == 0:
        raise RuntimeError("target construction produced 0 rows")

    # ---------- features --------------------------------------------------
    feat_df, features, excluded = prep.build_features(
        labeled, include_rainfall=include_rainfall
    )
    excluded.update(reasons)
    say(f"FEATURES n={len(features)}: {features}")

    # ---------- chronological split ---------------------------------------
    train_df, valid_df, split_info = prep.chronological_split(feat_df)
    say(f"SPLIT {split_info.get('split_strategy')} "
        f"train={len(train_df):,} valid={len(valid_df):,}")
    if len(valid_df) == 0:
        raise RuntimeError(f"validation split empty: {split_info}")

    Xtr = train_df[features].astype("float64")
    ytr = train_df[prep.TARGET].astype("float64")
    Xva = valid_df[features].astype("float64")
    yva = valid_df[prep.TARGET].astype("float64")

    # ---------- SMOKE TEST (pipeline validity, not quality) --------------
    say("SMOKE TEST on subset")
    s_tr = min(len(Xtr), smoke_rows)
    s_va = min(len(Xva), max(1000, smoke_rows // 10))
    assert len(Xtr) > 0 and len(Xva) > 0, "smoke: empty split"
    assert np.isfinite(Xtr.iloc[:s_tr].to_numpy()).all(), "smoke: non-finite features"
    assert np.isfinite(ytr.iloc[:s_tr].to_numpy()).all(), "smoke: non-finite target"
    assert ytr.between(0, 1).all(), "smoke: target outside [0,1]"
    smoke = lgb.LGBMRegressor(
        **{**modeling.LGBM_PARAMS, "n_estimators": 40}
    )
    smoke.fit(Xtr.iloc[:s_tr], ytr.iloc[:s_tr])
    sp = smoke.predict(Xva.iloc[:s_va])
    assert sp.shape == (s_va,), f"smoke: bad shape {sp.shape}"
    assert np.isfinite(sp).all(), "smoke: non-finite predictions"
    _ = modeling.evaluate(yva.iloc[:s_va].to_numpy(), sp)
    say("  SMOKE PASS")

    # ---------- full train ------------------------------------------------
    say(f"TRAIN LightGBM rows={len(Xtr):,} features={len(features)}")
    model = modeling.train_lightgbm(Xtr, ytr, Xva, yva)
    best_iter = getattr(model, "best_iteration_", None)
    say(f"  fitted best_iteration={best_iter}")

    say("EVALUATE")
    preds = model.predict(Xva)
    metrics = modeling.evaluate(yva.to_numpy(), preds)
    metrics["best_iteration"] = best_iter
    say(f"  MAE={metrics['MAE']:.5f} RMSE={metrics['RMSE']:.5f} R2={metrics['R2']:.5f}")
    say(f"  low_bike  P={metrics['low_bike_precision']:.4f} "
        f"R={metrics['low_bike_recall']:.4f} F1={metrics['low_bike_f1']:.4f} "
        f"AUC={metrics['low_bike_auc']}")
    say(f"  high_occ  P={metrics['high_occupancy_precision']:.4f} "
        f"R={metrics['high_occupancy_recall']:.4f} F1={metrics['high_occupancy_f1']:.4f} "
        f"AUC={metrics['high_occupancy_auc']}")

    # ---------- persist ---------------------------------------------------
    model_path = os.path.join(out_dir, f"{model_name}_lgbm.txt")
    model.booster_.save_model(model_path)
    joblib_path = os.path.join(out_dir, f"{model_name}_lgbm.pkl")
    try:
        import joblib
        joblib.dump(model, joblib_path)
    except Exception as e:
        joblib_path = None
        say(f"  (joblib dump skipped: {e})")

    importance = dict(
        sorted(
            zip(features, [int(v) for v in model.booster_.feature_importance("gain")]),
            key=lambda kv: -kv[1],
        )
    )

    metadata = {
        "model_name": model_name,
        "model_type": "LightGBM Regression (LGBMRegressor)",
        "training_backend": training_backend,
        "LightGBM_version": lgb.__version__,
        "Python_version": platform.python_version(),
        "container_image": None if training_backend == "LOCAL_FALLBACK" else "see training_job",
        "training_job_name": None,
        "training_job_arn": None,
        "instance_type": None if training_backend == "LOCAL_FALLBACK" else None,
        "raw_s3_uri": raw_s3_uri,
        "raw_local_path": csv_path,
        "actual_start_date": actual_start,
        "actual_end_date": actual_end,
        "months_used": months,
        "day_type": day_type,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "features": features,
        "excluded_features": sorted(excluded.keys()),
        "excluded_reasons": excluded,
        "target_definition": "future_available_bikes / future_total_docks (future observation's own capacity)",
        "target_window_minutes": [horizon_minutes - 5, horizon_minutes + 5],
        "horizon_minutes": horizon_minutes,
        "threshold_low": modeling.LOW_BIKE_THRESHOLD,
        "threshold_high": modeling.HIGH_OCCUPANCY_THRESHOLD,
        "hyperparameters": modeling.LGBM_PARAMS,
        "early_stopping_rounds": modeling.EARLY_STOPPING_ROUNDS,
        "split": split_info,
        "drop_counts": drop_counts,
        "target_stats": tstats,
        "feature_importance_gain": importance,
        "audit": {
            k: audit[k] for k in (
                "encoding", "header", "column_mapping", "column_mapping_method",
                "unmapped_columns", "rows_scanned", "first_timestamp",
                "last_timestamp", "month_row_counts", "complete_months",
                "station_count_estimate", "roughly_time_sorted",
                "has_is_peak", "has_rainfall", "has_temperature",
                "has_nearest_junior_high", "has_nearest_university",
                "has_nearest_mrt", "has_nearest_bus",
            )
        },
        "model_artifact_local_path": model_path,
        "model_artifact_s3_uri": None,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    metadata.update({k: metrics[k] for k in (
        "MAE", "RMSE", "R2",
        "low_bike_precision", "low_bike_recall", "low_bike_f1", "low_bike_auc",
        "high_occupancy_precision", "high_occupancy_recall",
        "high_occupancy_f1", "high_occupancy_auc",
    )})

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(out_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_columns": features, "target": prep.TARGET}, f,
                  indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    say(f"SAVED -> {out_dir}")
    return metadata
