# -*- coding: utf-8 -*-
"""Train the three remaining SAGEMAKER_ENRICHED models (max 2 concurrent).

Reuses the EXACT pipeline that produced the successful P0
(weekday-30m-bike-ratio-enr-1789263106) by importing run_enriched_aws and
v1_core directly: same container, instance, hyperparameters, target design,
CSV-only channels, retry wrapper and calibration code.

weekday_30m_bike_ratio is NEVER retrained or overwritten.

Wave 1 (concurrent): weekday_60m_bike_ratio + weekend_30m_bike_ratio
Wave 2:              weekend_60m_bike_ratio

Each job gets a fresh run-tagged CSV-only channel prefix, so a stale parquet can
never appear in a channel. On ResourceLimitExceeded the wave degrades to
sequential WITHOUT changing the instance type.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Import the proven P0 orchestrator as a library.
_spec = importlib.util.spec_from_file_location(
    "ra", os.path.join(HERE, "run_enriched_aws.py"))
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

from v1_core import columns as C   # noqa: E402
from v1_core import dataio, modeling, prep  # noqa: E402

RUN_TAG = str(int(time.time()))
MONTHS = ["2026-05"]

RAW = {
    "weekday": (os.path.join(HERE, "_enriched_raw", "weekday_enriched.csv"),
                "s3://ubike-data-final/平日_含最近距離_含天氣.csv"),
    "weekend": (os.path.join(HERE, "_enriched_raw", "weekend_enriched.csv"),
                "s3://ubike-data-final/假日與國定假日_含最近距離_含天氣.csv"),
}

WAVE1 = [("weekday_60m_bike_ratio", 60, "weekday"),
         ("weekend_30m_bike_ratio", 30, "weekend")]
WAVE2 = [("weekend_60m_bike_ratio", 60, "weekend")]

PROTECTED = "weekday_30m_bike_ratio"


def say(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def prepare(model_name, horizon, day_type, s3):
    """Audit -> load 2026-05 -> target -> features -> split -> upload CSV."""
    assert model_name != PROTECTED, "refusing to touch the successful P0"
    raw_local, raw_uri = RAW[day_type]
    out_dir = os.path.join(HERE, "enriched", model_name)
    os.makedirs(out_dir, exist_ok=True)

    say(f"{model_name}: audit")
    audit = dataio.audit_csv(raw_local)
    if MONTHS[0] not in audit["complete_months"]:
        raise RuntimeError(f"{MONTHS[0]} not complete; {audit['complete_months']}")
    say(f"{model_name}: months={MONTHS} is_peak={audit['has_is_peak']} "
        f"rainfall={audit['has_rainfall']}")

    raw = dataio.load_range(raw_local, audit["encoding"], audit["column_mapping"], MONTHS)
    df = prep.normalize_frame(raw, audit["column_mapping"])
    del raw
    df, drops = prep.drop_invalid_rows(df)

    labeled, tstats = prep.build_target(df, horizon)
    del df
    if len(labeled) == 0:
        raise RuntimeError("0 target rows")
    say(f"{model_name}: target rows={len(labeled):,} "
        f"window={tstats['target_window_low_minutes']}-"
        f"{tstats['target_window_high_minutes']}m "
        f"anomalies={tstats['target_anomaly_count_out_of_unit_range']}")

    feat_df, features, excluded = prep.build_features(
        labeled, include_rainfall=bool(audit["has_rainfall"]))
    excluded["day_type_source"] = (
        f"'{day_type}' from the pre-split source object {raw_uri}; provider's "
        f"平日 / 假日與國定假日 semantics preserved, not re-derived")
    say(f"{model_name}: features n={len(features)}")

    train_df, valid_df, split_info = prep.chronological_split(feat_df)
    if len(valid_df) == 0:
        raise RuntimeError(f"empty validation split {split_info}")
    say(f"{model_name}: split={split_info.get('split_strategy')} "
        f"train={len(train_df):,} valid={len(valid_df):,}")

    actual_start = str(feat_df[C.TIMESTAMP].min())
    actual_end = str(feat_df[C.TIMESTAMP].max())

    # Fresh, CSV-only channel prefix (run-tagged => no stale objects possible).
    base = f"{ra.ENRICHED_PREFIX}/{model_name}/csv-{RUN_TAG}"
    tmp = os.path.join(HERE, "_enriched_channels", f"{model_name}-{RUN_TAG}")
    os.makedirs(tmp, exist_ok=True)
    cols = features + [prep.TARGET]
    tr = os.path.join(tmp, "train.csv")
    va = os.path.join(tmp, "validation.csv")
    train_df[cols].to_csv(tr, index=False)
    valid_df[cols].to_csv(va, index=False)
    s3.upload_file(tr, ra.WORK_BUCKET, f"{base}/train/train.csv")
    s3.upload_file(va, ra.WORK_BUCKET, f"{base}/validation/validation.csv")
    say(f"{model_name}: channels -> s3://{ra.WORK_BUCKET}/{base}/ (CSV only)")

    # keep validation locally for metrics + calibration after Completed
    valid_df[cols].to_parquet(os.path.join(out_dir, "_valid_local.parquet"),
                              index=False)

    return {
        "model_name": model_name, "horizon": horizon, "day_type": day_type,
        "raw_uri": raw_uri, "audit": audit, "features": features,
        "excluded": excluded, "drops": drops, "tstats": tstats,
        "split_info": split_info, "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "actual_start": actual_start, "actual_end": actual_end,
        "channel_base": f"s3://{ra.WORK_BUCKET}/{base}",
        "out_dir": out_dir,
    }


def submit_one(spec, sm, sourcedir_uri):
    ra.with_retry("reachability", lambda: sm.list_training_jobs(MaxResults=1))
    for cand in ([ra.IMAGE_OVERRIDE] if ra.IMAGE_OVERRIDE else ra.IMAGE_CANDIDATES):
        try:
            job, arn, out = ra.with_retry(
                "create_training_job",
                lambda c=cand: ra.submit(sm, spec["model_name"],
                                         spec["channel_base"], sourcedir_uri,
                                         c, ra.INSTANCE_PRIMARY))
            spec.update({"job": job, "arn": arn, "out": out, "image": cand,
                         "submitted_at": time.time()})
            ra.put_model(spec["model_name"], status="RUNNING",
                         training_job_name=job, training_job_arn=arn)
            return spec
        except Exception as e:
            code = getattr(e, "response", {}).get("Error", {}).get("Code", "")
            if "ResourceLimitExceeded" in str(code) or "ResourceLimitExceeded" in str(e):
                raise
            say(f"  image rejected {cand}: {e}")
    raise RuntimeError("no usable training image")


def poll_wave(specs, sm):
    """Poll several jobs until all terminal."""
    pending = {s["job"]: s for s in specs if s.get("job")}
    last = {}
    while pending:
        for job, spec in list(pending.items()):
            d = ra.with_retry(f"describe {job}",
                              lambda j=job: sm.describe_training_job(TrainingJobName=j))
            st = d["TrainingJobStatus"]
            if last.get(job) != st:
                say(f"  {job}: {st} ({d.get('SecondaryStatus')})")
                ra.put_model(spec["model_name"], training_job_status=st)
                last[job] = st
            if st in ("Completed", "Failed", "Stopped"):
                spec["desc"] = d
                pending.pop(job)
        if pending:
            time.sleep(30)
    return specs


def finalize(spec, s3):
    """Download artifact, compute metrics + calibration, persist everything."""
    import lightgbm as lgb
    d = spec["desc"]
    name = spec["model_name"]
    out_dir = spec["out_dir"]
    if d["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"{spec['job']} {d['TrainingJobStatus']}: "
                           f"{d.get('FailureReason')}")
    artifact = d["ModelArtifacts"]["S3ModelArtifacts"]
    say(f"{name}: COMPLETED {artifact}")

    import tarfile
    tarp = os.path.join(out_dir, "model.tar.gz")
    b, k = artifact.replace("s3://", "").split("/", 1)
    s3.download_file(b, k, tarp)
    with tarfile.open(tarp, "r:gz") as tar:
        tar.extractall(out_dir)

    metrics, cenv = {}, {}
    for fn in ("metrics.json", "container_env.json"):
        p = os.path.join(out_dir, fn)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                (metrics if fn == "metrics.json" else cenv).update(json.load(f))

    mfile = next((os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.endswith("_lgbm.txt")), None)
    valid = pd.read_parquet(os.path.join(out_dir, "_valid_local.parquet"))
    feats = spec["features"]
    booster = lgb.Booster(model_file=mfile)
    yp = np.clip(booster.predict(valid[feats].to_numpy()), 0, 1)
    yt = valid[prep.TARGET].to_numpy()

    if not metrics:
        metrics = modeling.evaluate(yt, yp)
    sweep = ra.sweep_thresholds(yt, yp)
    cl = ra.pick_threshold(sweep["low"], "low")
    ch = ra.pick_threshold(sweep["high"], "high")
    pd.DataFrame({"y_true": yt, "y_pred": yp}).to_parquet(
        os.path.join(out_dir, "validation_predictions.parquet"), index=False)

    dur = None
    if d.get("TrainingStartTime") and d.get("TrainingEndTime"):
        dur = (d["TrainingEndTime"] - d["TrainingStartTime"]).total_seconds()

    meta = {
        "model_name": name,
        "model_type": "LightGBM Regression (LGBMRegressor)",
        "training_backend": ra.TRAINING_BACKEND,
        "LightGBM_version": cenv.get("LightGBM_version"),
        "Python_version": cenv.get("Python_version"),
        "container_image": spec["image"],
        "training_job_name": spec["job"],
        "training_job_arn": spec["arn"],
        "instance_type": ra.INSTANCE_PRIMARY,
        "training_duration_seconds": dur,
        "raw_s3_uri": spec["raw_uri"],
        "actual_start_date": spec["actual_start"],
        "actual_end_date": spec["actual_end"],
        "months_used": MONTHS,
        "day_type": spec["day_type"],
        "train_rows": spec["train_rows"],
        "validation_rows": spec["validation_rows"],
        "feature_columns": feats,
        "excluded_features": sorted(spec["excluded"].keys()),
        "excluded_reasons": spec["excluded"],
        "target_definition": "future_available_bikes / future_total_docks",
        "target_window_minutes": [spec["horizon"] - 5, spec["horizon"] + 5],
        "horizon_minutes": spec["horizon"],
        "hyperparameters": modeling.LGBM_PARAMS,
        "split": spec["split_info"],
        "drop_counts": spec["drops"],
        "target_stats": spec["tstats"],
        "event_definition_fixed": {
            "low_bike_event": f"actual future_bike_ratio < {ra.LOW_BIKE_EVENT}",
            "high_occupancy_event": f"actual future_bike_ratio > {ra.HIGH_OCCUPANCY_EVENT}",
            "calibrated": False},
        "threshold_sweep": sweep,
        "chosen_low_threshold": cl,
        "chosen_high_threshold": ch,
        "audit": spec["audit"],
        "model_artifact_s3_uri": artifact,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    meta.update({kk: metrics.get(kk) for kk in ("MAE", "RMSE", "R2")})

    for fn, obj in (("metadata.json", meta), ("metrics.json", metrics),
                    ("feature_columns.json", {"feature_columns": feats,
                                              "target": prep.TARGET}),
                    ("threshold_analysis.json", {"sweep": sweep,
                                                 "chosen_low": cl,
                                                 "chosen_high": ch})):
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        s3.upload_file(os.path.join(out_dir, fn), ra.WORK_BUCKET,
                       f"{ra.ENRICHED_PREFIX}/{name}/{fn}")
    s3.upload_file(os.path.join(out_dir, "validation_predictions.parquet"),
                   ra.WORK_BUCKET,
                   f"{ra.ENRICHED_PREFIX}/{name}/validation_predictions.parquet")

    ra.put_model(name, status="SUCCESS", training_job_status="Completed",
                 training_job_name=spec["job"], artifact_uri=artifact,
                 train_rows=spec["train_rows"],
                 validation_rows=spec["validation_rows"],
                 actual_start_date=spec["actual_start"],
                 actual_end_date=spec["actual_end"],
                 features=feats, chosen_low=cl, chosen_high=ch,
                 metrics={kk: meta[kk] for kk in ("MAE", "RMSE", "R2")},
                 training_duration_seconds=dur)
    say(f"{name}: MAE={metrics.get('MAE'):.6f} RMSE={metrics.get('RMSE'):.6f} "
        f"R2={metrics.get('R2'):.6f} | LOW alert {cl['alert_threshold']} "
        f"(R={cl['recall']:.4f}) HIGH alert {ch['alert_threshold']} "
        f"(R={ch['recall']:.4f})")
    return meta


def diagnose(spec):
    """CloudWatch root exception for a failed job."""
    import boto3
    logs = boto3.client("logs", region_name=ra.REGION)
    job = spec.get("job")
    say(f"{spec['model_name']}: FAILED - pulling CloudWatch root cause")
    d = spec.get("desc", {})
    say(f"  FailureReason: {str(d.get('FailureReason'))[:400]}")
    G = "/aws/sagemaker/TrainingJobs"
    try:
        st = logs.describe_log_streams(logGroupName=G, logStreamNamePrefix=job)["logStreams"]
        for s in st:
            ev = logs.get_log_events(logGroupName=G, logStreamName=s["logStreamName"],
                                     limit=10000, startFromHead=True)["events"]
            keep = [e["message"].rstrip() for e in ev
                    if any(k in e["message"] for k in
                           ("Error", "error", "Traceback", "Exception", "line ",
                            "contains", "ignoring", "features n="))]
            say(f"  --- {s['logStreamName']} ---")
            for m in keep[-40:]:
                print("    " + m, flush=True)
    except Exception as e:
        say(f"  (log fetch failed: {e})")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        say("STOPPED: no AWS credentials in this process")
        return 1
    import boto3
    s3 = boto3.client("s3", region_name=ra.REGION)
    sm = boto3.client("sagemaker", region_name=ra.REGION)
    ident = boto3.client("sts", region_name=ra.REGION).get_caller_identity()
    say(f"STS OK {ident['Account']} run_tag={RUN_TAG}")

    sourcedir_uri = ra.upload_sourcedir(s3)
    results, failures = {}, {}

    def do_wave(plan, allow_parallel=True):
        specs = []
        for name, hz, dt in plan:
            try:
                specs.append(prepare(name, hz, dt, s3))
            except Exception as e:
                say(f"{name}: PREPARE FAILED {e}")
                print(traceback.format_exc(limit=5))
                failures[name] = f"prepare: {type(e).__name__}: {e}"
        live = []
        for sp in specs:
            try:
                live.append(submit_one(sp, sm, sourcedir_uri))
                say(f"SUBMITTED {sp['job']} -> InProgress")
                if not allow_parallel:
                    poll_wave([sp], sm)
            except Exception as e:
                if "ResourceLimitExceeded" in str(e):
                    say("quota hit -> finishing current job before submitting next "
                        "(instance type unchanged)")
                    if live:
                        poll_wave(live, sm)
                        for x in live:
                            _handle(x)
                        live = []
                    try:
                        live.append(submit_one(sp, sm, sourcedir_uri))
                        say(f"SUBMITTED {sp['job']} -> InProgress")
                    except Exception as e2:
                        failures[sp["model_name"]] = f"submit: {e2}"
                else:
                    failures[sp["model_name"]] = f"submit: {e}"
        if live:
            poll_wave(live, sm)
            for x in live:
                _handle(x)

    def _handle(sp):
        try:
            results[sp["model_name"]] = finalize(sp, s3)
        except Exception as e:
            failures[sp["model_name"]] = str(e)
            try:
                diagnose(sp)
            except Exception:
                pass

    say("=== WAVE 1 (up to 2 concurrent) ===")
    do_wave(WAVE1)
    say("=== WAVE 2 ===")
    do_wave(WAVE2)

    ra.put_status(current_phase="remaining_models_done",
                  blockers=[f"{k}: {v}" for k, v in failures.items()])
    ra.write_summary()

    say("=== RESULT ===")
    for k in ("weekday_60m_bike_ratio", "weekend_30m_bike_ratio",
              "weekend_60m_bike_ratio"):
        say(f"  {k}: {'SUCCESS' if k in results else failures.get(k, 'NOT RUN')}")
    with open(os.path.join(HERE, "_remaining_results.json"), "w",
              encoding="utf-8") as f:
        json.dump({"results": {k: {"job": v["training_job_name"],
                                   "artifact": v["model_artifact_s3_uri"]}
                               for k, v in results.items()},
                   "failures": failures}, f, indent=2, ensure_ascii=False)
    return 0 if len(results) == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
