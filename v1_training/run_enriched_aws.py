# -*- coding: utf-8 -*-
"""SINGLE ENTRY POINT -- formal V1 enriched training on AWS (SAGEMAKER_ENRICHED).

Run this ONCE in a PowerShell where you have already set and verified the four
AWS_* environment variables. Credentials are read ONLY from the environment;
nothing is written to disk and no credential value is printed.

Pipeline:
    Phase 3  sts.get_caller_identity() + read-only audit of the enriched S3 CSVs
    Phase 4  target: future_available_bikes / future_total_docks, 25-35 / 55-65 min
    Phase 5  enriched features (is_peak, nearest distances, rainfall when safe)
    Phase 6  latest 2 complete months, auto-degrade to 1 on memory/quota pressure
    Phase 7  chronological split (earlier month -> train, latest -> validation)
    Phase 8  LightGBM regression, fixed params, no tuning
    Phase 9  P0 weekday_30m first, alone, must fully succeed
    Phase 10 MAE / RMSE / R2
    Phase 11 threshold sweep on the FORMAL validation set (overrides provisional)
    Phase 12 artifacts to s3://.../v1-enriched/  (never touches stable prefixes)
    Phase 13 bonus models only after P0, max 2 concurrent
    Phase 14 ml.m5.xlarge, 1 instance, MaxRuntime 7200, no endpoint
    Phase 15 v1_training/enriched_status.json + ENRICHED_SUMMARY.md

Guards: creates no endpoint, no IAM change, no Bedrock, deletes nothing, and
writes only under the v1-enriched/ prefix.

Usage:
    v1_training/venv/Scripts/python.exe v1_training/run_enriched_aws.py
    v1_training/venv/Scripts/python.exe v1_training/run_enriched_aws.py --p0-only
    v1_training/venv/Scripts/python.exe v1_training/run_enriched_aws.py --audit-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from v1_core import columns as C          # noqa: E402
from v1_core import dataio, modeling, prep  # noqa: E402

REGION = "us-west-2"
RAW_BUCKET = "ubike-data-final"
WORK_BUCKET = "youbike-dispatch-502837994229-usw2"
ENRICHED_PREFIX = "v1-enriched"          # Phase 12: isolated, never overwrites
ROLE_ARN = "arn:aws:iam::502837994229:role/kiro-sagemaker-execution-role"

IMAGE_CANDIDATES = [
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1",
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1",
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1",
]
IMAGE_OVERRIDE = os.environ.get("V1_TRAINING_IMAGE")

INSTANCE_PRIMARY = "ml.m5.xlarge"
INSTANCE_FALLBACK = "ml.m5.large"
MAX_RUNTIME = 7200

TRAINING_BACKEND = "SAGEMAKER_ENRICHED"

PLAN = [
    ("weekday_30m_bike_ratio", 30, "weekday"),
    ("weekday_60m_bike_ratio", 60, "weekday"),
    ("weekend_30m_bike_ratio", 30, "weekend"),
    ("weekend_60m_bike_ratio", 60, "weekend"),
]
P0 = PLAN[0][0]

# Phase 11 calibration contract.
#
# The BUSINESS EVENT DEFINITION IS FIXED and never participates in calibration:
#     LOW_BIKE_EVENT       = actual future_bike_ratio < 0.20
#     HIGH_OCCUPANCY_EVENT = actual future_bike_ratio > 0.80
# Only the PREDICTION ALERT TRIGGER is tuned. This keeps the operational meaning
# of 0.20 / 0.80 stable while letting recall be recovered, and avoids the earlier
# flaw where sweeping a shared value also changed event prevalence (which made
# F1 monotone in prevalence and always selected the most permissive grid edge).
LOW_BIKE_EVENT = 0.20
HIGH_OCCUPANCY_EVENT = 0.80

LOW_ALERT_CANDIDATES = [0.15, 0.18, 0.20, 0.21, 0.22, 0.25, 0.30]
HIGH_ALERT_CANDIDATES = [0.55, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]

RECALL_TARGET = 0.75

# LOCAL_FALLBACK reference only (8-feature June model). NOT a final threshold.
LOCAL_REFERENCE = {
    "low_event": 0.20,
    "low_alert_reference": 0.21,
    "low_reference_precision": 0.81,
    "low_reference_recall": 0.80,
    "low_reference_f1": 0.80,
    "high_event": 0.80,
    "high_alert_reference": 0.62,
    "high_reference_precision": 0.20,
    "high_reference_recall": 0.77,
    "note": ("LOCAL_FALLBACK reference from the local 9-column June CSV. The "
             "formal enriched validation result overrides these."),
}

STATUS_PATH = os.path.join(HERE, "enriched_status.json")
SUMMARY_PATH = os.path.join(HERE, "ENRICHED_SUMMARY.md")
LOCAL_ANALYSIS = os.path.join(HERE, "local_threshold_analysis.json")


# ----------------------------- status ------------------------------------- #
def _now():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_status():
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_phase": "init", "timestamp": _now(),
            "branch": "stable-v1-training",
            "training_backend": TRAINING_BACKEND,
            "sts_identity_status": None, "s3_audit_status": None,
            "dataset_range": None, "selected_features": None,
            "provisional_thresholds": None, "training_job_name": None,
            "training_job_status": None, "artifact_uri": None,
            "metrics_path": None, "last_error": None, "models": {}}


def put_status(**kw):
    st = load_status()
    st.update(kw)
    st["timestamp"] = _now()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False, default=str)
    return st


def put_model(name, **kw):
    st = load_status()
    st.setdefault("models", {})
    e = st["models"].get(name, {})
    e.update(kw)
    e["updated"] = _now()
    e["training_backend"] = TRAINING_BACKEND
    st["models"][name] = e
    st["timestamp"] = _now()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False, default=str)
    return st


def say(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def with_retry(label, fn, attempts=5, base_delay=4):
    """Retry a control-plane call through transient connectivity failures.

    Targets EndpointConnectionError / ConnectTimeout / ThrottlingException, which
    are transient. Credential and validation errors are re-raised immediately so
    a real misconfiguration is not masked. Exponential backoff, capped attempts,
    and the call is idempotent-safe because create_training_job is invoked only
    after a successful reachability probe.
    """
    from botocore.exceptions import (ClientError, ConnectTimeoutError,
                                     EndpointConnectionError, NoCredentialsError)
    delay = base_delay
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except NoCredentialsError:
            raise
        except (EndpointConnectionError, ConnectTimeoutError) as e:
            last = e
            say(f"  {label}: transient connectivity failure "
                f"(attempt {i}/{attempts}): {type(e).__name__}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException", "TooManyRequestsException",
                        "RequestTimeout", "ServiceUnavailable",
                        "InternalFailure", "InternalServerError"):
                last = e
                say(f"  {label}: retryable API error {code} "
                    f"(attempt {i}/{attempts})")
            else:
                raise
        if i < attempts:
            say(f"  {label}: backing off {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


def fail(m):
    say("STOPPED: " + m)
    put_status(current_phase="stopped", last_error=m)
    write_summary()
    return 1


# ----------------------------- metrics ------------------------------------ #
def prf(actual, predicted):
    tp = int(np.sum(actual & predicted)); fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "tp": tp, "fp": fp, "fn": fn}


def auc_or_na(actual, score):
    if len(np.unique(actual)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(actual, score))


def sweep_thresholds(y_true, y_pred):
    """Phase 11 alert-trigger calibration against FIXED business events.

    The actual event is held constant (< 0.20 / > 0.80) for every candidate, so
    event prevalence and the ROC-AUC are identical across the sweep and only the
    alert rule varies. That makes Precision / Recall / F1 directly comparable
    between candidates.
    """
    low_event = y_true < LOW_BIKE_EVENT
    high_event = y_true > HIGH_OCCUPANCY_EVENT

    out = {
        "event_definition": {
            "low_bike_event": f"actual future_bike_ratio < {LOW_BIKE_EVENT}",
            "high_occupancy_event": f"actual future_bike_ratio > {HIGH_OCCUPANCY_EVENT}",
            "fixed": True,
            "note": "business definition; NOT calibrated",
        },
        "low_event_prevalence": float(low_event.mean()),
        "high_event_prevalence": float(high_event.mean()),
        # AUC is a property of the fixed event + score, independent of the alert.
        "low_event_auc": auc_or_na(low_event.astype(int), 1.0 - y_pred),
        "high_event_auc": auc_or_na(high_event.astype(int), y_pred),
        "low": {},
        "high": {},
    }

    for t in LOW_ALERT_CANDIDATES:
        alert = y_pred < t
        m = prf(low_event, alert)
        m["alert_threshold"] = float(t)
        m["alert_prevalence"] = float(alert.mean())
        m["actual_event_prevalence"] = out["low_event_prevalence"]
        out["low"][f"{t:.2f}"] = m

    for t in HIGH_ALERT_CANDIDATES:
        alert = y_pred > t
        m = prf(high_event, alert)
        m["alert_threshold"] = float(t)
        m["alert_prevalence"] = float(alert.mean())
        m["actual_event_prevalence"] = out["high_event_prevalence"]
        out["high"][f"{t:.2f}"] = m

    return out


def pick_threshold(side_sweep, side):
    """Recall-first alert selection; the event definition is never changed."""
    candidates = {k: v for k, v in side_sweep.items() if isinstance(v, dict)
                  and "recall" in v}
    elig = {k: v for k, v in candidates.items() if v["recall"] >= RECALL_TARGET}
    pool = elig or candidates
    best = max(pool.items(), key=lambda kv: kv[1]["f1"])
    event = LOW_BIKE_EVENT if side == "low" else HIGH_OCCUPANCY_EVENT
    sym = "<" if side == "low" else ">"
    return {
        "event_definition": f"actual future_bike_ratio {sym} {event}",
        "event_threshold_fixed": event,
        "alert_threshold": float(best[0]),
        "alert_rule": f"predicted_ratio {sym} {float(best[0])}",
        "precision": best[1]["precision"],
        "recall": best[1]["recall"],
        "f1": best[1]["f1"],
        "alert_prevalence": best[1]["alert_prevalence"],
        "actual_event_prevalence": best[1]["actual_event_prevalence"],
        "recall_meets_target": bool(elig),
        "reason": (
            f"highest F1 among alert candidates reaching Recall >= {RECALL_TARGET}"
            if elig else
            f"NO alert candidate reached Recall >= {RECALL_TARGET}; highest F1 "
            f"chosen instead (Recall below target)"),
        "candidates_evaluated": sorted(float(k) for k in candidates),
    }


def local_provisional():
    """LOCAL_FALLBACK numbers, carried purely as a reference for comparison."""
    ref = dict(LOCAL_REFERENCE)
    ref["training_backend"] = "LOCAL_FALLBACK"
    ref["is_final"] = False
    if os.path.exists(LOCAL_ANALYSIS):
        try:
            with open(LOCAL_ANALYSIS, encoding="utf-8") as f:
                a = json.load(f)
            dec = a.get("decoupled_alert_calibration") or {}
            if dec:
                ref["local_decoupled_low_alert"] = (
                    dec.get("low", {}).get("recommended_alert_threshold"))
                ref["local_decoupled_high_alert"] = (
                    dec.get("high", {}).get("recommended_alert_threshold"))
            ref["local_shared_sweep_low"] = a.get("provisional_low_threshold")
            ref["local_shared_sweep_high"] = a.get("provisional_high_threshold")
            ref["local_shared_sweep_note"] = (
                "superseded: that sweep also moved the event definition")
        except Exception:
            pass
    return ref


# ----------------------------- AWS helpers -------------------------------- #
def classify_objects(objs):
    """Identify the weekday / weekend enriched objects from their keys."""
    weekend = [k for k, _ in objs if ("假日" in k or "國定" in k)]
    weekday = [k for k, _ in objs
               if "平日" in k and k not in weekend]
    return (weekday[0] if weekday else None), (weekend[0] if weekend else None)


def download_once(s3, key, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        say(f"  reuse cached {os.path.basename(dest)} ({os.path.getsize(dest):,} B)")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    say(f"  downloading s3://{RAW_BUCKET}/{key}")
    s3.download_file(RAW_BUCKET, key, dest)
    say(f"  done ({os.path.getsize(dest):,} B)")
    return dest


def upload_sourcedir(s3):
    tar_path = os.path.join(HERE, "_enriched_sourcedir.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(os.path.join(HERE, "sourcedir", "train_entry.py"), arcname="train_entry.py")
        tar.add(os.path.join(HERE, "sourcedir", "requirements.txt"), arcname="requirements.txt")
    key = f"{ENRICHED_PREFIX}/code/sourcedir.tar.gz"
    s3.upload_file(tar_path, WORK_BUCKET, key)
    uri = f"s3://{WORK_BUCKET}/{key}"
    say(f"  sourcedir -> {uri}")
    return uri


def upload_channels(s3, model_name, train_df, valid_df, features):
    # Dedicated CSV sub-prefix. The original {model_name}/train/ prefix still
    # holds a stale train.parquet from an earlier run, and SageMaker downloads
    # EVERY object under a channel prefix. Using a clean csv/ sub-prefix
    # guarantees the channel contains CSV only, without deleting anything.
    base = f"{ENRICHED_PREFIX}/{model_name}/csv"
    tmp = os.path.join(HERE, "_enriched_channels", model_name)
    os.makedirs(tmp, exist_ok=True)
    cols = features + [prep.TARGET]
    # CSV, not parquet. The sagemaker-scikit-learn:1.2-1 container (Python 3.8)
    # ships pandas WITHOUT pyarrow/fastparquet, so pd.read_parquet raises
    # ImportError inside the training script. Writing CSV keeps the container
    # dependency set untouched (only lightgbm is installed) and removes the
    # failure entirely. Verified root cause of job
    # weekday-30m-bike-ratio-enr-1789257242.
    tr, va, ext = (os.path.join(tmp, "train.csv"),
                   os.path.join(tmp, "validation.csv"), "csv")
    train_df[cols].to_csv(tr, index=False)
    valid_df[cols].to_csv(va, index=False)
    say(f"  wrote CSV channels ({len(train_df):,} train / {len(valid_df):,} valid rows)")
    fc = os.path.join(tmp, "feature_columns.json")
    with open(fc, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": features, "target": prep.TARGET}, f, indent=2)
    s3.upload_file(tr, WORK_BUCKET, f"{base}/train/train.{ext}")
    s3.upload_file(va, WORK_BUCKET, f"{base}/validation/validation.{ext}")
    s3.upload_file(fc, WORK_BUCKET, f"{base}/feature_columns.json")
    say(f"  channels -> s3://{WORK_BUCKET}/{base}/")
    return f"s3://{WORK_BUCKET}/{base}", va, cols


def submit(sm, model_name, channel_base, sourcedir_uri, image, instance):
    job = f"{model_name.replace('_','-')}-enr-{int(time.time())}"[:63]
    out = f"s3://{WORK_BUCKET}/{ENRICHED_PREFIX}/output"
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {"ChannelName": "train", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"{channel_base}/train/",
                "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "validation", "DataSource": {"S3DataSource": {
                "S3DataType": "S3Prefix", "S3Uri": f"{channel_base}/validation/",
                "S3DataDistributionType": "FullyReplicated"}}},
        ],
        OutputDataConfig={"S3OutputPath": out},
        ResourceConfig={"InstanceType": instance, "InstanceCount": 1,
                        "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": MAX_RUNTIME},
        HyperParameters={"sagemaker_program": "train_entry.py",
                         "sagemaker_submit_directory": sourcedir_uri,
                         "model-name": model_name},
        Environment={"MODEL_NAME": model_name},
    )
    d = sm.describe_training_job(TrainingJobName=job)
    say(f"  SUBMITTED {job} -> {d['TrainingJobStatus']}")
    return job, d["TrainingJobArn"], out


def poll(sm, job, model_name):
    last = None
    while True:
        d = with_retry(f"describe {job}",
                       lambda: sm.describe_training_job(TrainingJobName=job))
        s = d["TrainingJobStatus"]
        if s != last:
            say(f"  {job}: {s} ({d.get('SecondaryStatus')})")
            put_model(model_name, training_job_status=s)
            put_status(training_job_name=job, training_job_status=s)
            last = s
        if s in ("Completed", "Failed", "Stopped"):
            return d
        time.sleep(30)


# ----------------------------- one model ---------------------------------- #
def run_one(model_name, horizon, day_type, raw_local, raw_uri,
            s3, sm, sourcedir_uri, instance, months_wanted, audit_only=False,
            force_months=None):
    out_dir = os.path.join(HERE, "enriched", model_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Phase 3: audit ------------------------------------------------
    say(f"AUDIT {os.path.basename(raw_local)}")
    audit = dataio.audit_csv(raw_local)
    say(f"  encoding={audit['encoding']}")
    say(f"  header={audit['header']}")
    say(f"  range {audit['first_timestamp']} -> {audit['last_timestamp']}")
    say(f"  months={audit['month_row_counts']}")
    say(f"  complete_months={audit['complete_months']} stations~{audit['station_count_estimate']}")
    say(f"  is_peak={audit['has_is_peak']} rainfall={audit['has_rainfall']} "
        f"temperature={audit['has_temperature']}")
    say(f"  dist jh={audit['has_nearest_junior_high']} uni={audit['has_nearest_university']} "
        f"mrt={audit['has_nearest_mrt']} bus={audit['has_nearest_bus']}")
    say(f"  mapping={json.dumps(audit['column_mapping'], ensure_ascii=False)}")
    say(f"  UNMAPPED (left out) = {audit['unmapped_columns']}")
    with open(os.path.join(out_dir, "audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    put_status(s3_audit_status=f"OK:{model_name}")
    if audit_only:
        return None

    if force_months:
        requested = [m.strip() for m in force_months.split(",") if m.strip()]
        missing = [m for m in requested if m not in audit["complete_months"]]
        if missing:
            raise RuntimeError(
                f"forced month(s) {missing} are not complete months in this "
                f"file. complete_months={audit['complete_months']}")
        months = requested
        say(f"  Phase 6 months={months} (FORCED)")
    else:
        months = dataio.pick_months(audit["complete_months"], months_wanted)
        if not months:
            raise RuntimeError(f"no complete month; months={audit['month_row_counts']}")
        say(f"  Phase 6 months={months}")

    raw = dataio.load_range(raw_local, audit["encoding"], audit["column_mapping"], months)
    df = prep.normalize_frame(raw, audit["column_mapping"])
    del raw
    df, drops = prep.drop_invalid_rows(df)
    say(f"  rows after validity={len(df):,} {drops}")

    # Phase 13: respect the provider's own weekday/weekend split; do NOT re-derive.
    reasons = {"day_type_source":
               f"'{day_type}' taken from the pre-split source object {raw_uri}; "
               f"the provider's 平日 / 假日與國定假日 semantics are preserved and "
               f"NOT re-derived from timestamp.weekday()"}

    labeled, tstats = prep.build_target(df, horizon)
    del df
    if len(labeled) == 0:
        raise RuntimeError("0 target rows")
    say(f"  Phase 4 target rows={len(labeled):,} "
        f"anomalies={tstats['target_anomaly_count_out_of_unit_range']} "
        f"window={tstats['target_window_low_minutes']}-{tstats['target_window_high_minutes']}min")

    # Phase 5: rainfall only when the audit resolved it.
    include_rain = bool(audit["has_rainfall"])
    feat_df, features, excluded = prep.build_features(labeled, include_rainfall=include_rain)
    excluded.update(reasons)
    say(f"  Phase 5 features n={len(features)}: {features}")
    put_status(selected_features=features)

    train_df, valid_df, split_info = prep.chronological_split(feat_df)
    say(f"  Phase 7 split={split_info.get('split_strategy')} "
        f"train={len(train_df):,} valid={len(valid_df):,}")
    if len(valid_df) == 0:
        raise RuntimeError(f"empty validation split: {split_info}")

    actual_start = str(feat_df[C.TIMESTAMP].min())
    actual_end = str(feat_df[C.TIMESTAMP].max())
    put_status(dataset_range=f"{actual_start} -> {actual_end}")

    # ---- smoke test before spending a training job ---------------------
    say("  SMOKE TEST")
    import lightgbm as lgb
    Xs = train_df[features].astype("float64").iloc[:150_000]
    ys = train_df[prep.TARGET].astype("float64").iloc[:150_000]
    Xv = valid_df[features].astype("float64").iloc[:20_000]
    yv = valid_df[prep.TARGET].astype("float64").iloc[:20_000]
    assert np.isfinite(Xs.to_numpy()).all(), "smoke: non-finite features"
    assert ys.between(0, 1).all(), "smoke: target outside [0,1]"
    sm_model = lgb.LGBMRegressor(**{**modeling.LGBM_PARAMS, "n_estimators": 40})
    sm_model.fit(Xs, ys)
    sp = sm_model.predict(Xv)
    assert sp.shape == (len(Xv),) and np.isfinite(sp).all(), "smoke: bad predictions"
    _ = modeling.evaluate(yv.to_numpy(), sp)
    say("  SMOKE PASS")

    channel_base, local_valid_path, cols = upload_channels(
        s3, model_name, train_df, valid_df, features)
    valid_local = valid_df[features + [prep.TARGET]]

    # ---- submit --------------------------------------------------------
    job = arn = out = None
    errors = []
    # Reachability probe first, so a transient control-plane outage does not
    # leave us guessing whether a job was actually created.
    with_retry("sagemaker reachability",
               lambda: sm.list_training_jobs(MaxResults=1))
    say("  control plane reachable")
    for cand in ([IMAGE_OVERRIDE] if IMAGE_OVERRIDE else IMAGE_CANDIDATES):
        try:
            job, arn, out = with_retry(
                "create_training_job",
                lambda c=cand: submit(sm, model_name, channel_base,
                                      sourcedir_uri, c, instance))
            image = cand
            break
        except Exception as e:
            errors.append(f"{cand}: {e}")
            say(f"  image rejected: {cand} -> {e}")
    if job is None:
        raise RuntimeError("no usable training image; tried " + " | ".join(errors))

    artifact_guess = f"{out}/{job}/output/model.tar.gz"
    put_model(model_name, status="RUNNING", training_job_name=job,
              training_job_arn=arn, artifact_uri=artifact_guess)
    put_status(current_phase=f"training:{model_name}", training_job_name=job,
               training_job_status="InProgress", artifact_uri=artifact_guess)
    write_summary()

    d = poll(sm, job, model_name)
    if d["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"{job} {d['TrainingJobStatus']}: {d.get('FailureReason')}")
    artifact = d["ModelArtifacts"]["S3ModelArtifacts"]
    say(f"  COMPLETED {artifact}")

    # ---- fetch artifact, recompute Phase 10/11 locally -----------------
    metrics, cenv = {}, {}
    try:
        tarp = os.path.join(out_dir, "model.tar.gz")
        b, k = artifact.replace("s3://", "").split("/", 1)
        s3.download_file(b, k, tarp)
        with tarfile.open(tarp, "r:gz") as tar:
            tar.extractall(out_dir)
        for fn in ("metrics.json", "container_env.json"):
            p = os.path.join(out_dir, fn)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    (metrics if fn == "metrics.json" else cenv).update(json.load(f))
    except Exception as e:
        say(f"  (artifact read issue: {e})")

    # Phase 11: sweep using the trained booster on the local validation copy.
    sweep = None
    chosen_low = chosen_high = None
    try:
        mfile = None
        for fn in os.listdir(out_dir):
            if fn.endswith("_lgbm.txt"):
                mfile = os.path.join(out_dir, fn)
        if mfile:
            booster = lgb.Booster(model_file=mfile)
            yp = np.clip(booster.predict(valid_local[features].to_numpy()), 0, 1)
            yt = valid_local[prep.TARGET].to_numpy()
            sweep = sweep_thresholds(yt, yp)
            chosen_low = pick_threshold(sweep["low"], "low")
            chosen_high = pick_threshold(sweep["high"], "high")
            pd.DataFrame({"y_true": yt, "y_pred": yp}).to_parquet(
                os.path.join(out_dir, "validation_predictions.parquet"), index=False)
            say(f"  Phase 11 event LOW <{LOW_BIKE_EVENT} (prevalence "
                f"{sweep['low_event_prevalence']:.4f}, AUC {sweep['low_event_auc']}) "
                f"-> alert <{chosen_low['alert_threshold']} "
                f"P={chosen_low['precision']:.4f} R={chosen_low['recall']:.4f} "
                f"F1={chosen_low['f1']:.4f} target_met={chosen_low['recall_meets_target']}")
            say(f"  Phase 11 event HIGH >{HIGH_OCCUPANCY_EVENT} (prevalence "
                f"{sweep['high_event_prevalence']:.4f}, AUC {sweep['high_event_auc']}) "
                f"-> alert >{chosen_high['alert_threshold']} "
                f"P={chosen_high['precision']:.4f} R={chosen_high['recall']:.4f} "
                f"F1={chosen_high['f1']:.4f} target_met={chosen_high['recall_meets_target']}")
            if not metrics:
                metrics = modeling.evaluate(yt, yp)
    except Exception as e:
        say(f"  (threshold sweep skipped: {e})")

    metadata = {
        "model_name": model_name,
        "model_type": "LightGBM Regression (LGBMRegressor)",
        "training_backend": TRAINING_BACKEND,
        "LightGBM_version": cenv.get("LightGBM_version"),
        "Python_version": cenv.get("Python_version"),
        "container_image": image,
        "training_job_name": job,
        "training_job_arn": arn,
        "instance_type": instance,
        "raw_s3_uri": raw_uri,
        "actual_start_date": actual_start,
        "actual_end_date": actual_end,
        "months_used": months,
        "day_type": day_type,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "feature_columns": features,
        "excluded_features": sorted(excluded.keys()),
        "excluded_reasons": excluded,
        "target_definition": "future_available_bikes / future_total_docks (future observation capacity)",
        "target_window_minutes": [horizon - 5, horizon + 5],
        "horizon_minutes": horizon,
        "hyperparameters": modeling.LGBM_PARAMS,
        "split": split_info,
        "drop_counts": drops,
        "target_stats": tstats,
        "audit": audit,
        "MAE": metrics.get("MAE"), "RMSE": metrics.get("RMSE"), "R2": metrics.get("R2"),
        "event_definition_fixed": {
            "low_bike_event": f"actual future_bike_ratio < {LOW_BIKE_EVENT}",
            "high_occupancy_event": f"actual future_bike_ratio > {HIGH_OCCUPANCY_EVENT}",
            "calibrated": False,
            "note": ("business event definition, held fixed; only the prediction "
                     "alert trigger is calibrated"),
        },
        "alert_candidates": {"low": LOW_ALERT_CANDIDATES,
                             "high": HIGH_ALERT_CANDIDATES},
        "threshold_sweep": sweep,
        "chosen_low_threshold": chosen_low,
        "chosen_high_threshold": chosen_high,
        "local_provisional_reference": local_provisional(),
        "model_artifact_s3_uri": artifact,
        "generated_at": _now(),
    }
    for fn, obj in (("metadata.json", metadata),
                    ("metrics.json", metrics),
                    ("feature_columns.json", {"feature_columns": features,
                                              "target": prep.TARGET}),
                    ("threshold_analysis.json", {
                        "training_backend": TRAINING_BACKEND,
                        "event_definition_fixed": {
                            "low_bike_event": f"actual future_bike_ratio < {LOW_BIKE_EVENT}",
                            "high_occupancy_event": f"actual future_bike_ratio > {HIGH_OCCUPANCY_EVENT}",
                            "calibrated": False},
                        "alert_candidates": {"low": LOW_ALERT_CANDIDATES,
                                             "high": HIGH_ALERT_CANDIDATES},
                        "recall_target": RECALL_TARGET,
                        "sweep": sweep,
                        "chosen_low": chosen_low,
                        "chosen_high": chosen_high,
                        "local_reference": local_provisional()})):
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

    for fn in ("metadata.json", "metrics.json", "feature_columns.json",
               "threshold_analysis.json"):
        s3.upload_file(os.path.join(out_dir, fn), WORK_BUCKET,
                       f"{ENRICHED_PREFIX}/{model_name}/{fn}")

    put_model(model_name, status="SUCCESS", training_job_status="Completed",
              training_job_name=job, training_job_arn=arn,
              actual_start_date=actual_start, actual_end_date=actual_end,
              train_rows=metadata["train_rows"],
              validation_rows=metadata["validation_rows"],
              artifact_uri=artifact,
              metrics_path=os.path.join(out_dir, "metrics.json"),
              metrics={k: metadata[k] for k in ("MAE", "RMSE", "R2")},
              chosen_low=chosen_low, chosen_high=chosen_high,
              features=features)
    put_status(artifact_uri=artifact,
               metrics_path=os.path.join(out_dir, "metrics.json"))
    write_summary()
    return metadata


# ----------------------------- summary ------------------------------------ #
def _alert_line(chosen):
    """One-line rendering of a calibrated alert rule."""
    if not chosen:
        return "N/A"
    flag = "" if chosen.get("recall_meets_target") else "  [Recall BELOW 0.75 target]"
    return (f"{chosen.get('alert_rule')}  P={chosen.get('precision'):.4f} "
            f"R={chosen.get('recall'):.4f} F1={chosen.get('f1'):.4f}{flag}")


def write_summary():
    st = load_status()
    models = st.get("models", {})
    L = ["# V1 ENRICHED (formal) Training Summary", "",
         f"Generated: {_now()}", f"Branch: {st.get('branch')}", "",
         "> This file covers **SAGEMAKER_ENRICHED** (formal) models only.",
         "> The four LOCAL_FALLBACK models are a separate, clearly-labelled",
         "> baseline documented in LOCAL_MODEL_COMPARISON.md / OVERNIGHT_SUMMARY.md.",
         "> LOCAL_FALLBACK is NOT the formal enriched AWS model.", "",
         f"current_phase: {st.get('current_phase')}",
         f"STS identity: {st.get('sts_identity_status')}",
         f"S3 audit: {st.get('s3_audit_status')}",
         f"dataset range: {st.get('dataset_range')}", ""]
    L.append("## Result")
    L.append("")
    for name, _, _ in PLAN:
        e = models.get(name)
        L.append(f"{'P0 ' if name == P0 else ''}{name}: "
                 f"{e.get('status', 'NOT RUN') if e else 'NOT RUN'}")
    L.append("")
    for name, _, _ in PLAN:
        e = models.get(name)
        if not e or e.get("status") != "SUCCESS":
            continue
        m = e.get("metrics", {}) or {}
        L += [f"## {name}", "",
              f"training_backend: {e.get('training_backend')}",
              f"training_job_name: {e.get('training_job_name')}",
              f"data: {e.get('actual_start_date')} -> {e.get('actual_end_date')}",
              f"train_rows: {e.get('train_rows')}  validation_rows: {e.get('validation_rows')}",
              f"MAE: {m.get('MAE')}", f"RMSE: {m.get('RMSE')}", f"R2: {m.get('R2')}",
              "",
              f"event definition (FIXED, not calibrated): "
              f"low < {LOW_BIKE_EVENT} / high > {HIGH_OCCUPANCY_EVENT}",
              f"calibrated low alert : {_alert_line(e.get('chosen_low'))}",
              f"calibrated high alert: {_alert_line(e.get('chosen_high'))}",
              f"artifact: {e.get('artifact_uri')}", ""]
    L += ["## Errors / blockers", ""]
    b = st.get("blockers") or []
    if st.get("last_error"):
        b = list(b) + [str(st["last_error"])]
    L += [f"- {x}" for x in b] if b else ["- none"]
    L.append("")
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------- main --------------------------------------- #
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--months", type=int, default=2)
    ap.add_argument("--force-months", default=None,
                    help="comma-separated exact months to use, e.g. 2026-05. "
                         "Overrides --months and the latest-N selection.")
    ap.add_argument("--instance", default=INSTANCE_PRIMARY)
    args = ap.parse_args()

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return fail("Missing AWS_ACCESS_KEY_ID. Set and verify the four AWS_* "
                    "variables in THIS PowerShell process first.")

    import boto3
    from botocore.exceptions import ClientError

    # ---- Phase 3a: identity -------------------------------------------
    try:
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    except Exception as e:
        return fail(f"sts.get_caller_identity failed: {e}")
    say("STS OK")
    say(f"  Account: {ident['Account']}")
    say(f"  ARN    : {ident['Arn']}")
    say(f"  Region : {REGION}")
    put_status(current_phase="audit", sts_identity_status=f"OK {ident['Account']}",
               provisional_thresholds=local_provisional())

    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)

    # ---- Phase 3b: list + classify raw objects ------------------------
    try:
        resp = s3.list_objects_v2(Bucket=RAW_BUCKET)
        objs = [(o["Key"], o["Size"]) for o in resp.get("Contents", [])]
    except Exception as e:
        return fail(f"cannot list s3://{RAW_BUCKET}/: {e}")
    say(f"s3://{RAW_BUCKET}/ contains {len(objs)} object(s):")
    for k, s in objs:
        say(f"  {k}  ContentLength={s:,}")
    wd_key, we_key = classify_objects(objs)
    say(f"  weekday object -> {wd_key}")
    say(f"  weekend object -> {we_key}")
    if not wd_key:
        return fail("could not identify the 平日 enriched object among: "
                    + str([k for k, _ in objs])
                    + ". Refusing to guess which file is weekday.")

    cache = os.path.join(HERE, "_enriched_raw")
    raw_wd = download_once(s3, wd_key, os.path.join(cache, "weekday_enriched.csv"))
    raw_we = None

    if args.audit_only:
        run_one(P0, 30, "weekday", raw_wd, f"s3://{RAW_BUCKET}/{wd_key}",
                s3, sm, None, args.instance, args.months, audit_only=True)
        if we_key:
            raw_we = download_once(s3, we_key, os.path.join(cache, "weekend_enriched.csv"))
            run_one("weekend_30m_bike_ratio", 30, "weekend", raw_we,
                    f"s3://{RAW_BUCKET}/{we_key}", s3, sm, None,
                    args.instance, args.months, audit_only=True)
        put_status(current_phase="audit_only_done")
        write_summary()
        say("AUDIT ONLY complete - no training job submitted")
        return 0

    sourcedir_uri = upload_sourcedir(s3)

    instance, months = args.instance, args.months
    plan = PLAN[:1] if args.p0_only else PLAN
    p0_ok = False
    blockers = []

    for name, horizon, day_type in plan:
        if name != P0 and not p0_ok:
            put_model(name, status="NOT RUN", note="P0 did not succeed")
            continue
        if day_type == "weekend":
            if not we_key:
                put_model(name, status="NOT RUN",
                          note="weekend enriched object not identified")
                continue
            if raw_we is None:
                raw_we = download_once(s3, we_key,
                                       os.path.join(cache, "weekend_enriched.csv"))
        raw_local = raw_wd if day_type == "weekday" else raw_we
        raw_uri = f"s3://{RAW_BUCKET}/{wd_key if day_type == 'weekday' else we_key}"

        print("=" * 74)
        say(f"MODEL {name} horizon={horizon}m day_type={day_type} instance={instance}")
        print("=" * 74)
        try:
            run_one(name, horizon, day_type, raw_local, raw_uri, s3, sm,
                    sourcedir_uri, instance, months,
                    force_months=args.force_months)
            if name == P0:
                p0_ok = True
                say("*** P0 SUCCESS (SAGEMAKER_ENRICHED) ***")
        except (ClientError, MemoryError, Exception) as e:
            code = ""
            if isinstance(e, ClientError):
                code = e.response.get("Error", {}).get("Code", "")
            msg = f"{name}: {type(e).__name__} {code} {e}"
            say("FAILED " + msg)
            if not isinstance(e, ClientError):
                print(traceback.format_exc(limit=6))
            blockers.append(msg)
            put_model(name, status="FAILED", error=msg)
            put_status(last_error=msg, blockers=blockers)
            write_summary()

            # Phase 14: degrade data volume / instance ONCE, never scale up.
            degraded = False
            if isinstance(e, MemoryError) and months > 1:
                say("memory pressure -> retry with 1 month")
                months, degraded = 1, True
            elif code == "ResourceLimitExceeded" and instance != INSTANCE_FALLBACK:
                say(f"quota -> retry on {INSTANCE_FALLBACK} with 1 month")
                instance, months, degraded = INSTANCE_FALLBACK, 1, True
            if degraded:
                try:
                    run_one(name, horizon, day_type, raw_local, raw_uri, s3, sm,
                            sourcedir_uri, instance, months,
                            force_months=args.force_months)
                    if name == P0:
                        p0_ok = True
                        say("*** P0 SUCCESS after degrade ***")
                except Exception as e2:
                    blockers.append(f"{name} degraded retry: {e2}")
                    put_model(name, status="FAILED", error=str(e2))
                    put_status(last_error=str(e2), blockers=blockers)

            if name == P0 and not p0_ok:
                say("P0 failed -> stopping. LOCAL_FALLBACK models remain available.")
                break

    put_status(current_phase="done", blockers=blockers)
    write_summary()
    say("=" * 60)
    say(f"P0 ({P0}) {'SUCCESS' if p0_ok else 'NOT SUCCESSFUL'}")
    say(f"status  : {STATUS_PATH}")
    say(f"summary : {SUMMARY_PATH}")
    return 0 if p0_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
