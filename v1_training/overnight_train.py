# -*- coding: utf-8 -*-
"""SINGLE ENTRY POINT for the SageMaker V1 training path.

Run this once in a PowerShell that already has the four AWS_* environment
variables set. It does everything end to end:

    STS verify
      -> Phase A streaming audit of the enriched S3 CSV (never loads 1.1 GB)
        -> pick latest complete month(s)
          -> preprocess (target + features + chronological split)
            -> upload channels to s3://.../v1-training/
              -> boto3 create_training_job (LightGBM in a managed container)
                -> poll to Completed / Failed
                  -> write metrics + metadata + status

Design decisions (deliberate, per mission brief):
  * boto3 create_training_job is used directly instead of the SageMaker Python
    SDK, because the installed SDK is v3 which removed the framework estimator
    classes. This avoids all SDK/plumbing risk (brief section 15 timebox).
  * The container is an AWS-managed scikit-learn script-mode image; LightGBM is
    pip-installed from sourcedir/requirements.txt. LightGBM is still what trains.
  * MaxRuntimeInSeconds = 7200, one instance, ml.m5.xlarge with an automatic
    fallback to ml.m5.large (which also reduces the data range to 1 month).
  * P0 (weekday_30m) runs alone and must fully succeed before any bonus model.

IMPORTANT HONESTY NOTE: this script could not be executed against real AWS while
it was written (the agent shell had no credentials), so the AWS calls are
UNVERIFIED. The local fallback models in v1_training/local_fallback/ were
actually trained and ARE verified. If anything here fails, that fallback already
satisfies P0.

Credentials are read only from the environment and never printed or written.

Usage:
    v1_training/venv/Scripts/python.exe v1_training/overnight_train.py
    v1_training/venv/Scripts/python.exe v1_training/overnight_train.py --p0-only
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_ROOT = os.path.dirname(HERE)

from v1_core import columns as C          # noqa: E402
from v1_core import dataio, modeling, prep, status  # noqa: E402

REGION = "us-west-2"
RAW_BUCKET = "ubike-data-final"
WEEKDAY_KEY_HINTS = ["平日"]
WEEKEND_KEY_HINTS = ["假日", "國定"]

WORK_BUCKET = "youbike-dispatch-502837994229-usw2"
WORK_PREFIX = "v1-training"
ROLE_ARN = "arn:aws:iam::502837994229:role/kiro-sagemaker-execution-role"

# AWS-managed script-mode images (us-west-2). Tried in order.
IMAGE_CANDIDATES = [
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1",
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.0-1",
    "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1",
]
IMAGE_OVERRIDE = os.environ.get("V1_TRAINING_IMAGE")

INSTANCE_PRIMARY = "ml.m5.xlarge"
INSTANCE_FALLBACK = "ml.m5.large"
MAX_RUNTIME = 7200

PLAN = [
    ("weekday_30m_bike_ratio", 30, "weekday"),
    ("weekday_60m_bike_ratio", 60, "weekday"),
    ("weekend_30m_bike_ratio", 30, "weekend"),
    ("weekend_60m_bike_ratio", 60, "weekend"),
]
P0 = PLAN[0][0]


def say(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg: str) -> int:
    say("STOPPED: " + msg)
    status.update_status(current_phase="sagemaker_stopped", last_error=msg)
    status.write_summary()
    return 1


# --------------------------------------------------------------------------- #
def verify_identity():
    import boto3
    ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    say(f"STS OK Account={ident['Account']}")
    say(f"  ARN={ident['Arn']}")
    say(f"  Region={REGION}")
    return ident


def find_raw_keys(s3):
    """List the raw bucket and classify the weekday / weekend objects."""
    resp = s3.list_objects_v2(Bucket=RAW_BUCKET)
    objs = [(o["Key"], o["Size"]) for o in resp.get("Contents", [])]
    say(f"raw bucket s3://{RAW_BUCKET}/ objects:")
    for k, s in objs:
        say(f"  {k}  ({s:,} B)")
    weekday = [k for k, _ in objs if any(h in k for h in WEEKDAY_KEY_HINTS)
               and not any(h in k for h in WEEKEND_KEY_HINTS)]
    weekend = [k for k, _ in objs if any(h in k for h in WEEKEND_KEY_HINTS)]
    return objs, (weekday[0] if weekday else None), (weekend[0] if weekend else None)


def download_raw(s3, key: str, dest: str) -> str:
    """Download the raw object once so the audit + load passes can stream it."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        say(f"  reuse cached {dest} ({os.path.getsize(dest):,} B)")
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    say(f"  downloading s3://{RAW_BUCKET}/{key} -> {dest}")
    s3.download_file(RAW_BUCKET, key, dest)
    say(f"  downloaded ({os.path.getsize(dest):,} B)")
    return dest


def upload_channels(s3, model_name: str, train_df, valid_df, features):
    """Write train/validation parquet + feature_columns.json to the work prefix."""
    base = f"{WORK_PREFIX}/{model_name}"
    tmp = os.path.join(HERE, "_channels", model_name)
    os.makedirs(tmp, exist_ok=True)

    cols = features + [prep.TARGET]
    tr_path = os.path.join(tmp, "train.parquet")
    va_path = os.path.join(tmp, "validation.parquet")
    try:
        train_df[cols].to_parquet(tr_path, index=False)
        valid_df[cols].to_parquet(va_path, index=False)
        ext = "parquet"
    except Exception as e:
        say(f"  parquet unavailable ({e}); falling back to csv")
        tr_path = os.path.join(tmp, "train.csv")
        va_path = os.path.join(tmp, "validation.csv")
        train_df[cols].to_csv(tr_path, index=False)
        valid_df[cols].to_csv(va_path, index=False)
        ext = "csv"

    fc_path = os.path.join(tmp, "feature_columns.json")
    with open(fc_path, "w", encoding="utf-8") as f:
        json.dump({"feature_columns": features, "target": prep.TARGET}, f, indent=2)

    s3.upload_file(tr_path, WORK_BUCKET, f"{base}/train/train.{ext}")
    s3.upload_file(va_path, WORK_BUCKET, f"{base}/validation/validation.{ext}")
    s3.upload_file(fc_path, WORK_BUCKET, f"{base}/feature_columns.json")
    say(f"  uploaded channels to s3://{WORK_BUCKET}/{base}/")
    return f"s3://{WORK_BUCKET}/{base}"


def upload_sourcedir(s3) -> str:
    """Tar + upload the script-mode source directory."""
    tar_path = os.path.join(HERE, "_sourcedir.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(os.path.join(HERE, "sourcedir", "train_entry.py"), arcname="train_entry.py")
        tar.add(os.path.join(HERE, "sourcedir", "requirements.txt"), arcname="requirements.txt")
    key = f"{WORK_PREFIX}/code/sourcedir.tar.gz"
    s3.upload_file(tar_path, WORK_BUCKET, key)
    uri = f"s3://{WORK_BUCKET}/{key}"
    say(f"  sourcedir -> {uri}")
    return uri


def submit_job(sm, model_name, channel_base, sourcedir_uri, image, instance_type):
    job = f"{model_name.replace('_', '-')}-{int(time.time())}"[:63]
    out = f"s3://{WORK_BUCKET}/{WORK_PREFIX}/output"
    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": image, "TrainingInputMode": "File"},
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {"ChannelName": "train",
             "DataSource": {"S3DataSource": {
                 "S3DataType": "S3Prefix",
                 "S3Uri": f"{channel_base}/train/",
                 "S3DataDistributionType": "FullyReplicated"}}},
            {"ChannelName": "validation",
             "DataSource": {"S3DataSource": {
                 "S3DataType": "S3Prefix",
                 "S3Uri": f"{channel_base}/validation/",
                 "S3DataDistributionType": "FullyReplicated"}}},
        ],
        OutputDataConfig={"S3OutputPath": out},
        ResourceConfig={"InstanceType": instance_type, "InstanceCount": 1,
                        "VolumeSizeInGB": 50},
        StoppingCondition={"MaxRuntimeInSeconds": MAX_RUNTIME},
        HyperParameters={
            "sagemaker_program": "train_entry.py",
            "sagemaker_submit_directory": sourcedir_uri,
            "model-name": model_name,
        },
        Environment={"MODEL_NAME": model_name},
    )
    desc = sm.describe_training_job(TrainingJobName=job)
    say(f"  submitted {job} status={desc['TrainingJobStatus']}")
    return job, desc["TrainingJobArn"], out


def poll_job(sm, job: str, model_name: str):
    """Poll until terminal. Persists identifiers immediately so an interrupted
    shell never loses the job."""
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=job)
        st = d["TrainingJobStatus"]
        if st != last:
            say(f"  {job}: {st} ({d.get('SecondaryStatus')})")
            status.record_model(model_name, training_job_name=job,
                                training_job_status=st)
            status.update_status(training_job_name=job, training_job_status=st)
            last = st
        if st in ("Completed", "Failed", "Stopped"):
            return d
        time.sleep(30)


def run_one(model_name, horizon, day_type, raw_local, raw_s3_uri,
            s3, sm, sourcedir_uri, instance_type, months_wanted):
    out_dir = os.path.join(HERE, "sagemaker", model_name)
    os.makedirs(out_dir, exist_ok=True)

    say(f"AUDIT {raw_local}")
    audit = dataio.audit_csv(raw_local)
    say(f"  range {audit['first_timestamp']} -> {audit['last_timestamp']}")
    say(f"  complete_months={audit['complete_months']} stations~{audit['station_count_estimate']}")
    say(f"  is_peak={audit['has_is_peak']} rainfall={audit['has_rainfall']} "
        f"temp={audit['has_temperature']} mrt={audit['has_nearest_mrt']}")
    say(f"  mapping={json.dumps(audit['column_mapping'], ensure_ascii=False)}")
    say(f"  unmapped={audit['unmapped_columns']}")

    months = dataio.pick_months(audit["complete_months"], months_wanted)
    if not months:
        raise RuntimeError(f"no complete month; months={audit['month_row_counts']}")
    say(f"  months={months}")

    raw = dataio.load_range(raw_local, audit["encoding"], audit["column_mapping"], months)
    df = prep.normalize_frame(raw, audit["column_mapping"])
    del raw
    df, drops = prep.drop_invalid_rows(df)

    reasons = {}
    # The S3 files are ALREADY split by day type, so no extra filter is applied;
    # the file's own 平日 / 假日與國定假日 definition is preserved and recorded.
    reasons["day_type_source"] = (
        f"day_type '{day_type}' comes from the pre-split source object "
        f"({raw_s3_uri}); its 平日 / 假日與國定假日 semantics are the data "
        f"provider's, not re-derived here."
    )

    labeled, tstats = prep.build_target(df, horizon)
    del df
    if len(labeled) == 0:
        raise RuntimeError("0 target rows")
    say(f"  target rows={len(labeled):,} anomalies="
        f"{tstats['target_anomaly_count_out_of_unit_range']}")

    feat_df, features, excluded = prep.build_features(labeled, include_rainfall=False)
    excluded.update(reasons)
    train_df, valid_df, split_info = prep.chronological_split(feat_df)
    say(f"  split={split_info.get('split_strategy')} train={len(train_df):,} valid={len(valid_df):,}")
    if len(valid_df) == 0:
        raise RuntimeError("empty validation split")

    channel_base = upload_channels(s3, model_name, train_df, valid_df, features)

    image = IMAGE_OVERRIDE or None
    job = arn = out = None
    errors = []
    for candidate in ([image] if image else IMAGE_CANDIDATES):
        try:
            job, arn, out = submit_job(sm, model_name, channel_base,
                                       sourcedir_uri, candidate, instance_type)
            image = candidate
            break
        except Exception as e:
            errors.append(f"{candidate}: {e}")
            say(f"  image rejected {candidate}: {e}")
    if job is None:
        raise RuntimeError("no usable training image. tried: " + " | ".join(errors))

    status.record_model(model_name, status="RUNNING", training_job_name=job,
                        training_job_arn=arn, artifact_uri=f"{out}/{job}/output/model.tar.gz")
    status.update_status(current_phase=f"sagemaker_training:{model_name}",
                        training_job_name=job, training_job_status="InProgress",
                        artifact_uri=f"{out}/{job}/output/model.tar.gz")
    status.write_summary()

    desc = poll_job(sm, job, model_name)
    if desc["TrainingJobStatus"] != "Completed":
        raise RuntimeError(f"training job {job} {desc['TrainingJobStatus']}: "
                           f"{desc.get('FailureReason')}")

    artifact = desc["ModelArtifacts"]["S3ModelArtifacts"]
    say(f"  COMPLETED artifact={artifact}")

    # Pull metrics out of the produced model.tar.gz.
    metrics = {}
    container_env = {}
    try:
        local_tar = os.path.join(out_dir, "model.tar.gz")
        bkt, key = artifact.replace("s3://", "").split("/", 1)
        s3.download_file(bkt, key, local_tar)
        with tarfile.open(local_tar, "r:gz") as tar:
            tar.extractall(out_dir)
        for fn, sink in (("metrics.json", "metrics"), ("container_env.json", "env")):
            p = os.path.join(out_dir, fn)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    if sink == "metrics":
                        metrics = json.load(f)
                    else:
                        container_env = json.load(f)
    except Exception as e:
        say(f"  (could not read metrics from artifact: {e})")

    metadata = {
        "model_name": model_name,
        "model_type": "LightGBM Regression (LGBMRegressor)",
        "training_backend": "SAGEMAKER_TRAINING_JOB",
        "LightGBM_version": container_env.get("LightGBM_version"),
        "Python_version": container_env.get("Python_version"),
        "container_image": image,
        "training_job_name": job,
        "training_job_arn": arn,
        "instance_type": instance_type,
        "raw_s3_uri": raw_s3_uri,
        "actual_start_date": str(feat_df[C.TIMESTAMP].min()),
        "actual_end_date": str(feat_df[C.TIMESTAMP].max()),
        "months_used": months,
        "day_type": day_type,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "features": features,
        "excluded_features": sorted(excluded.keys()),
        "excluded_reasons": excluded,
        "target_definition": "future_available_bikes / future_total_docks",
        "target_window_minutes": [horizon - 5, horizon + 5],
        "threshold_low": modeling.LOW_BIKE_THRESHOLD,
        "threshold_high": modeling.HIGH_OCCUPANCY_THRESHOLD,
        "split": split_info,
        "drop_counts": drops,
        "target_stats": tstats,
        "audit": audit,
        "model_artifact_s3_uri": artifact,
    }
    metadata.update({k: metrics.get(k) for k in (
        "MAE", "RMSE", "R2",
        "low_bike_precision", "low_bike_recall", "low_bike_f1", "low_bike_auc",
        "high_occupancy_precision", "high_occupancy_recall",
        "high_occupancy_f1", "high_occupancy_auc")})

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    status.record_model(
        model_name, status="SUCCESS", training_backend="SAGEMAKER_TRAINING_JOB",
        training_job_name=job, training_job_arn=arn,
        training_job_status="Completed",
        actual_start_date=metadata["actual_start_date"],
        actual_end_date=metadata["actual_end_date"],
        train_rows=metadata["train_rows"], validation_rows=metadata["validation_rows"],
        artifact_uri=artifact, metrics_path=os.path.join(out_dir, "metrics.json"),
        metrics={k: metadata[k] for k in (
            "MAE", "RMSE", "R2", "low_bike_precision", "low_bike_recall",
            "low_bike_f1", "low_bike_auc", "high_occupancy_precision",
            "high_occupancy_recall", "high_occupancy_f1", "high_occupancy_auc")},
    )
    status.write_summary()
    return metadata


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--p0-only", action="store_true")
    ap.add_argument("--months", type=int, default=2)
    ap.add_argument("--instance", default=INSTANCE_PRIMARY)
    args = ap.parse_args()

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return fail("Missing AWS_ACCESS_KEY_ID. Set the four AWS_* variables "
                    "in this same PowerShell process first.")

    import boto3
    from botocore.exceptions import ClientError

    try:
        verify_identity()
    except Exception as e:
        return fail(f"STS failed: {e}")

    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)

    status.update_status(current_phase="sagemaker_audit", branch="stable-v1-training")

    try:
        objs, weekday_key, weekend_key = find_raw_keys(s3)
    except Exception as e:
        return fail(f"cannot list s3://{RAW_BUCKET}/: {e}")
    if not weekday_key:
        return fail(f"could not identify the 平日 object among: {[k for k,_ in objs]}")

    cache = os.path.join(HERE, "_raw_cache")
    raw_weekday = download_raw(s3, weekday_key, os.path.join(cache, "weekday.csv"))
    raw_weekend = None

    sourcedir_uri = upload_sourcedir(s3)

    instance = args.instance
    months_wanted = args.months
    plan = PLAN[:1] if args.p0_only else PLAN
    p0_ok = False
    blockers = []

    for model_name, horizon, day_type in plan:
        if model_name != P0 and not p0_ok:
            status.record_model(model_name, status="NOT RUN",
                                note="P0 did not succeed")
            continue
        if day_type == "weekend":
            if not weekend_key:
                status.record_model(model_name, status="NOT RUN",
                                    note="weekend object not identified")
                continue
            if raw_weekend is None:
                raw_weekend = download_raw(s3, weekend_key,
                                           os.path.join(cache, "weekend.csv"))
        raw_local = raw_weekday if day_type == "weekday" else raw_weekend
        raw_uri = f"s3://{RAW_BUCKET}/{weekday_key if day_type=='weekday' else weekend_key}"

        print("=" * 72)
        say(f"MODEL {model_name} horizon={horizon} day_type={day_type} instance={instance}")
        print("=" * 72)
        try:
            run_one(model_name, horizon, day_type, raw_local, raw_uri,
                    s3, sm, sourcedir_uri, instance, months_wanted)
            if model_name == P0:
                p0_ok = True
                say("*** P0 SUCCESS ***")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = f"{model_name}: {code} {e}"
            say("FAILED " + msg)
            blockers.append(msg)
            status.record_model(model_name, status="FAILED", error=msg)
            status.update_status(last_error=msg, blockers=blockers)
            status.write_summary()
            if "ResourceLimitExceeded" in code and instance != INSTANCE_FALLBACK:
                say(f"quota hit -> retrying once on {INSTANCE_FALLBACK} with 1 month")
                instance, months_wanted = INSTANCE_FALLBACK, 1
                try:
                    run_one(model_name, horizon, day_type, raw_local, raw_uri,
                            s3, sm, sourcedir_uri, instance, months_wanted)
                    if model_name == P0:
                        p0_ok = True
                except Exception as e2:
                    blockers.append(f"{model_name} fallback: {e2}")
                    status.record_model(model_name, status="FAILED", error=str(e2))
            if model_name == P0 and not p0_ok:
                say("P0 failed -> stopping. Local fallback models already satisfy P0.")
                break
        except Exception as e:
            tb = traceback.format_exc(limit=6)
            say("FAILED " + model_name)
            print(tb)
            blockers.append(f"{model_name}: {type(e).__name__}: {e}")
            status.record_model(model_name, status="FAILED", error=str(e))
            status.update_status(last_error=str(e), blockers=blockers)
            status.write_summary()
            if model_name == P0:
                say("P0 failed -> stopping. Local fallback models already satisfy P0.")
                break

    status.update_status(current_phase="sagemaker_done", blockers=blockers)
    print(status.write_summary())
    return 0 if p0_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
