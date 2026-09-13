# -*- coding: utf-8 -*-
"""Bundle the four Completed SageMaker LightGBM models into ONE artifact.

Reads each model's real metadata.json / feature_columns.json (no retraining, no
value invented) and emits:

    deploy_v1/artifacts/youbike-v1-multimodel.tar.gz
        manifest.json
        weekday_30m_bike_ratio_lgbm.txt
        weekday_60m_bike_ratio_lgbm.txt
        weekend_30m_bike_ratio_lgbm.txt
        weekend_60m_bike_ratio_lgbm.txt
        code/inference.py
        code/requirements.txt

Source artifacts under v1_training/enriched/ are read-only.
"""

from __future__ import annotations

import json, os, shutil, tarfile
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # project root, parent of deploy_v1/
SRC = os.path.join(ROOT, "v1_training", "enriched")
OUT = HERE
STAGE = os.path.join(OUT, "_stage")
ARTIFACTS = os.path.join(OUT, "artifacts")

MODELS = [
    ("weekday_30m", "weekday_30m_bike_ratio", "weekday", 30),
    ("weekday_60m", "weekday_60m_bike_ratio", "weekday", 60),
    ("weekend_30m", "weekend_30m_bike_ratio", "weekend", 30),
    ("weekend_60m", "weekend_60m_bike_ratio", "weekend", 60),
]

EVENT_LOW, EVENT_HIGH = 0.20, 0.80


def main():
    if os.path.isdir(STAGE):
        shutil.rmtree(STAGE)
    os.makedirs(os.path.join(STAGE, "code"), exist_ok=True)
    os.makedirs(ARTIFACTS, exist_ok=True)

    manifest = {
        "bundle_name": "youbike-v1-multimodel",
        "training_backend": "SAGEMAKER_ENRICHED",
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "description": ("Four LightGBM regression models behind ONE endpoint, "
                        "routed by day_type + horizon_minutes."),
        "target_definition": "future_available_bikes / future_total_docks",
        "event_definition": {
            "low_bike_event_threshold": EVENT_LOW,
            "high_occupancy_event_threshold": EVENT_HIGH,
            "calibrated": False,
            "note": "business definition, fixed; only alert triggers are calibrated",
        },
        "routing": {"day_type": ["weekday", "weekend"], "horizon_minutes": [30, 60]},
        "models": {},
    }

    for key, name, day_type, horizon in MODELS:
        d = os.path.join(SRC, name)
        md = json.load(open(os.path.join(d, "metadata.json"), encoding="utf-8"))
        fc = json.load(open(os.path.join(d, "feature_columns.json"), encoding="utf-8"))
        src_model = os.path.join(d, f"{name}_lgbm.txt")
        shutil.copy(src_model, os.path.join(STAGE, f"{name}_lgbm.txt"))

        assert md["day_type"] == day_type, f"{name}: day_type mismatch"
        assert int(md["horizon_minutes"]) == horizon, f"{name}: horizon mismatch"

        manifest["models"][key] = {
            "model_name": name,
            "model_file": f"{name}_lgbm.txt",
            "day_type": day_type,
            "horizon_minutes": horizon,
            "target_window_minutes": md["target_window_minutes"],
            "feature_columns": fc["feature_columns"],
            "n_features": len(fc["feature_columns"]),
            "alert_low_threshold": md["chosen_low_threshold"]["alert_threshold"],
            "alert_high_threshold": md["chosen_high_threshold"]["alert_threshold"],
            "training_job_name": md["training_job_name"],
            "model_artifact_s3_uri": md["model_artifact_s3_uri"],
            "MAE": md["MAE"], "RMSE": md["RMSE"], "R2": md["R2"],
            "train_rows": md["train_rows"], "validation_rows": md["validation_rows"],
            "actual_start_date": md["actual_start_date"],
            "actual_end_date": md["actual_end_date"],
            "LightGBM_version": md.get("LightGBM_version"),
        }

    with open(os.path.join(STAGE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    shutil.copy(os.path.join(OUT, "code", "inference.py"),
                os.path.join(STAGE, "code", "inference.py"))
    shutil.copy(os.path.join(OUT, "code", "requirements.txt"),
                os.path.join(STAGE, "code", "requirements.txt"))

    tar_path = os.path.join(ARTIFACTS, "youbike-v1-multimodel.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        for fn in sorted(os.listdir(STAGE)):
            full = os.path.join(STAGE, fn)
            if os.path.isdir(full):
                for sub in sorted(os.listdir(full)):
                    tar.add(os.path.join(full, sub), arcname=f"{fn}/{sub}")
            else:
                tar.add(full, arcname=fn)

    print(f"built {tar_path} ({os.path.getsize(tar_path):,} B)")
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            print(f"   {m.name:<40} {m.size:>10,} B")
    print("\nrouting table:")
    for k, v in manifest["models"].items():
        print(f"   {k:<12} -> {v['model_name']:<24} nfeat={v['n_features']} "
              f"alert low<{v['alert_low_threshold']} high>{v['alert_high_threshold']}")


if __name__ == "__main__":
    main()
