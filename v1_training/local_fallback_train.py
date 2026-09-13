# -*- coding: utf-8 -*-
"""LOCAL FALLBACK training (no AWS required).

Runs the SAME target / features / chronological-split / LightGBM / metrics
pipeline as the SageMaker path, on the local monthly CSVs, in an isolated venv.
Artifacts are clearly labelled training_backend = LOCAL_FALLBACK so they are
never mistaken for a SageMaker-trained model.

P0 is weekday_30m_bike_ratio. The three bonus models only start after P0 has
fully succeeded.

Usage:
    v1_training/venv/Scripts/python.exe v1_training/local_fallback_train.py
"""

from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_ROOT = os.path.dirname(HERE)

from v1_core import pipeline, status  # noqa: E402

# Latest complete local month. Local files are NOT pre-split by day type, so the
# weekday / weekend split is derived from the timestamp (documented in metadata).
JUNE = os.path.join(PROJECT_ROOT, "dataset", "YouBike 六月資料.csv 的副本.csv")

# (model_name, horizon_minutes, day_type) in strict priority order.
PLAN = [
    ("weekday_30m_bike_ratio", 30, pipeline.WEEKDAY),   # P0 - must succeed first
    ("weekday_60m_bike_ratio", 60, pipeline.WEEKDAY),
    ("weekend_30m_bike_ratio", 30, pipeline.WEEKEND),
    ("weekend_60m_bike_ratio", 60, pipeline.WEEKEND),
]
P0 = PLAN[0][0]

OUT_ROOT = os.path.join(HERE, "local_fallback")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    if not os.path.exists(JUNE):
        status.update_status(current_phase="local_fallback_failed",
                            last_error=f"missing local dataset: {JUNE}")
        status.write_summary()
        print("STOPPED: local dataset not found:", JUNE)
        return 1

    status.update_status(
        current_phase="local_fallback_running",
        branch="stable-v1-training",
        dataset_range="latest complete local month (2026-06)",
        training_job_name=None,
        training_job_status="LOCAL_FALLBACK",
    )

    p0_ok = False
    blockers = []

    for model_name, horizon, day_type in PLAN:
        if model_name != P0 and not p0_ok:
            status.record_model(model_name, status="NOT RUN",
                                note="skipped because P0 did not succeed")
            continue

        out_dir = os.path.join(OUT_ROOT, model_name)
        print("=" * 72)
        print("MODEL:", model_name, "| horizon:", horizon, "| day_type:", day_type)
        print("=" * 72)
        status.update_status(current_phase=f"training:{model_name}")
        status.record_model(model_name, status="RUNNING")
        try:
            meta = pipeline.run_model(
                model_name=model_name,
                csv_path=JUNE,
                horizon_minutes=horizon,
                day_type=day_type,
                out_dir=out_dir,
                training_backend="LOCAL_FALLBACK",
                months_wanted=1,          # brief: 1 month for local fallback
                raw_s3_uri=None,
            )
            status.record_model(
                model_name,
                status="SUCCESS",
                training_backend="LOCAL_FALLBACK",
                actual_start_date=meta["actual_start_date"],
                actual_end_date=meta["actual_end_date"],
                train_rows=meta["train_rows"],
                validation_rows=meta["validation_rows"],
                artifact_uri=meta["model_artifact_local_path"],
                metrics_path=os.path.join(out_dir, "metrics.json"),
                metrics={k: meta[k] for k in (
                    "MAE", "RMSE", "R2",
                    "low_bike_precision", "low_bike_recall", "low_bike_f1",
                    "low_bike_auc",
                    "high_occupancy_precision", "high_occupancy_recall",
                    "high_occupancy_f1", "high_occupancy_auc",
                )},
            )
            status.update_status(
                latest_success=model_name,
                artifact_uri=meta["model_artifact_local_path"],
                metrics_path=os.path.join(out_dir, "metrics.json"),
            )
            status.write_summary()
            if model_name == P0:
                p0_ok = True
                print("\n*** P0 SUCCESS -- continuing to bonus models ***\n")
        except Exception as exc:
            tb = traceback.format_exc(limit=6)
            print("MODEL FAILED:", model_name)
            print(tb)
            blockers.append(f"{model_name}: {type(exc).__name__}: {exc}")
            status.record_model(model_name, status="FAILED",
                                error=f"{type(exc).__name__}: {exc}")
            status.update_status(last_error=f"{model_name}: {exc}",
                                blockers=blockers)
            status.write_summary()
            if model_name == P0:
                print("P0 FAILED -- not starting bonus models.")
                break

    status.update_status(
        current_phase="local_fallback_done",
        blockers=blockers,
    )
    print(status.write_summary())
    return 0 if p0_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
