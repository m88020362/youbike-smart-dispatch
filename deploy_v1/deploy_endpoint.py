# -*- coding: utf-8 -*-
"""Deploy the V1 multimodel artifact to ONE SageMaker real-time endpoint.

Upload -> Model -> Endpoint Config -> Endpoint -> AWS parity test.

Container: sagemaker-scikit-learn:1.2-1 (Python 3.8.10). Chosen because that
exact image already pip-installed lightgbm==4.6.0 and trained all four models,
so the LightGBM wheel compatibility is proven rather than assumed. inference.py
needs only pandas (in the image) plus lightgbm (from code/requirements.txt).

Creates ONE Model holding all four boosters, routed at request time.
Credentials come only from environment variables; none are printed or written.
"""

from __future__ import annotations

import json, os, sys, time, tarfile
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

REGION = "us-west-2"
BUCKET = "youbike-dispatch-502837994229-usw2"
KEY = "v1-deploy/youbike-v1-multimodel.tar.gz"
LOCAL_TAR = os.path.join(HERE, "artifacts", "youbike-v1-multimodel.tar.gz")
ROLE = "arn:aws:iam::502837994229:role/kiro-sagemaker-execution-role"
IMAGE = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:1.2-1"

MODEL_NAME = "youbike-v1-multimodel"
CONFIG_NAME = "youbike-v1-multimodel-config"
ENDPOINT_NAME = "youbike-v1-multimodel-endpoint"
INSTANCE = "ml.m5.large"
VARIANT = "AllTraffic"

SRC = os.path.join(ROOT, "v1_training", "enriched")
CH = os.path.join(ROOT, "v1_training", "_enriched_channels")
CASES = [
    ("weekday", 30, "weekday_30m_bike_ratio", "weekday_30m_bike_ratio"),
    ("weekday", 60, "weekday_60m_bike_ratio", "weekday_60m_bike_ratio-1789266742"),
    ("weekend", 30, "weekend_30m_bike_ratio", "weekend_30m_bike_ratio-1789266742"),
    ("weekend", 60, "weekend_60m_bike_ratio", "weekend_60m_bike_ratio-1789266742"),
]
PARITY_ROWS = 500
TOL = 1e-9

LOG = []
def say(m):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True); LOG.append(line)

def dump(path="_deploy_report.json", **kw):
    with open(os.path.join(HERE, path), "w", encoding="utf-8") as f:
        json.dump(kw, f, indent=2, ensure_ascii=False, default=str)


def retry(label, fn, attempts=5, delay=4):
    from botocore.exceptions import (ClientError, ConnectTimeoutError,
                                     EndpointConnectionError, NoCredentialsError)
    d = delay; last = None
    for i in range(1, attempts+1):
        try:
            return fn()
        except NoCredentialsError:
            raise
        except (EndpointConnectionError, ConnectTimeoutError) as e:
            last = e; say(f"  {label}: transient connectivity {i}/{attempts}")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("ThrottlingException","TooManyRequestsException",
                        "ServiceUnavailable","InternalFailure","RequestTimeout"):
                last = e; say(f"  {label}: retryable {code} {i}/{attempts}")
            else:
                raise
        if i < attempts:
            time.sleep(d); d = min(d*2, 60)
    raise RuntimeError(f"{label} failed after {attempts}: {last}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        say("STOPPED: no AWS credentials in this process")
        return 1
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)
    rt = boto3.client("sagemaker-runtime", region_name=REGION)

    # ---- 1 preflight -----------------------------------------------------
    ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    say(f"STS OK account={ident['Account']} region={REGION}")
    if ident["Account"] != "502837994229":
        say("STOPPED: unexpected account"); return 1
    boto3.client("iam").get_role(RoleName="kiro-sagemaker-execution-role")
    say("execution role present (unmodified)")

    # ---- 2 upload --------------------------------------------------------
    local_size = os.path.getsize(LOCAL_TAR)
    say(f"uploading {LOCAL_TAR} ({local_size:,} B) -> s3://{BUCKET}/{KEY}")
    retry("upload", lambda: s3.upload_file(LOCAL_TAR, BUCKET, KEY))
    head = retry("head_object", lambda: s3.head_object(Bucket=BUCKET, Key=KEY))
    remote_size = head["ContentLength"]
    say(f"head_object: size={remote_size:,} B match={remote_size == local_size}")
    if remote_size != local_size:
        say("STOPPED: size mismatch"); return 1
    artifact_uri = f"s3://{BUCKET}/{KEY}"

    # ---- 4 one Model -----------------------------------------------------
    for name, fn, kw in (("model", sm.describe_model, {"ModelName": MODEL_NAME}),
                         ("config", sm.describe_endpoint_config, {"EndpointConfigName": CONFIG_NAME}),
                         ("endpoint", sm.describe_endpoint, {"EndpointName": ENDPOINT_NAME})):
        try:
            fn(**kw); say(f"NOTE: {name} '{list(kw.values())[0]}' already exists")
        except ClientError:
            pass

    try:
        r = retry("create_model", lambda: sm.create_model(
            ModelName=MODEL_NAME, ExecutionRoleArn=ROLE,
            PrimaryContainer={
                "Image": IMAGE,
                "ModelDataUrl": artifact_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                    "SAGEMAKER_REGION": REGION,
                }}))
        say(f"created Model {MODEL_NAME} -> {r['ModelArn']}")
    except ClientError as e:
        if "Cannot create already existing" in str(e):
            say(f"Model {MODEL_NAME} already exists, reusing")
        else:
            raise

    # ---- 5 config --------------------------------------------------------
    try:
        r = retry("create_endpoint_config", lambda: sm.create_endpoint_config(
            EndpointConfigName=CONFIG_NAME,
            ProductionVariants=[{"VariantName": VARIANT, "ModelName": MODEL_NAME,
                                 "InitialInstanceCount": 1,
                                 "InstanceType": INSTANCE,
                                 "InitialVariantWeight": 1.0}]))
        say(f"created EndpointConfig {CONFIG_NAME} ({INSTANCE} x1)")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if "already existing" in str(e):
            say(f"EndpointConfig {CONFIG_NAME} already exists, reusing")
        elif code == "ResourceLimitExceeded":
            say(f"STOPPED: ResourceLimitExceeded for {INSTANCE} - not upgrading instance")
            say(f"  {e}")
            dump(status="ResourceLimitExceeded", instance=INSTANCE, error=str(e))
            return 1
        else:
            raise

    # ---- 6 endpoint ------------------------------------------------------
    try:
        r = retry("create_endpoint", lambda: sm.create_endpoint(
            EndpointName=ENDPOINT_NAME, EndpointConfigName=CONFIG_NAME))
        say(f"creating Endpoint {ENDPOINT_NAME} -> {r['EndpointArn']}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if "already existing" in str(e):
            say(f"Endpoint {ENDPOINT_NAME} already exists, will poll it")
        elif code == "ResourceLimitExceeded":
            say(f"STOPPED: ResourceLimitExceeded creating endpoint on {INSTANCE}")
            dump(status="ResourceLimitExceeded", instance=INSTANCE, error=str(e))
            return 1
        else:
            raise

    say("polling for InService (30s interval)")
    deadline = time.time() + 45*60
    last = None
    while True:
        d = retry("describe_endpoint",
                  lambda: sm.describe_endpoint(EndpointName=ENDPOINT_NAME))
        st = d["EndpointStatus"]
        if st != last:
            say(f"  {ENDPOINT_NAME}: {st}")
            last = st
        if st in ("InService","Failed"):
            break
        if time.time() > deadline:
            say("poll timeout"); break
        time.sleep(30)

    if st != "InService":
        say(f"ENDPOINT NOT HEALTHY: {st}")
        say(f"FailureReason: {d.get('FailureReason')}")
        logs = boto3.client("logs", region_name=REGION)
        G = f"/aws/sagemaker/Endpoints/{ENDPOINT_NAME}"
        try:
            streams = logs.describe_log_streams(logGroupName=G, orderBy="LastEventTime",
                                                descending=True, limit=2)["logStreams"]
            for s_ in streams:
                ev = logs.get_log_events(logGroupName=G, logStreamName=s_["logStreamName"],
                                         limit=10000, startFromHead=True)["events"]
                say(f"--- {s_['logStreamName']} ---")
                for e in ev:
                    m = e["message"].rstrip()
                    if any(k in m for k in ("Error","error","Traceback","Exception",
                                            "line ","ModuleNotFound","ImportError",
                                            "lightgbm","loaded ","manifest")):
                        print("   " + m, flush=True); LOG.append("   " + m)
        except Exception as ex:
            say(f"(log fetch failed: {ex})")
        dump(status=st, failure_reason=d.get("FailureReason"), log="\n".join(LOG))
        return 1

    say("ENDPOINT InService")

    # ---- 7 AWS parity ----------------------------------------------------
    import lightgbm as lgb
    say("AWS parity test (same real validation rows as local test)")
    rows = []
    all_ok = True
    for day_type, horizon, model_name, chan in CASES:
        vcsv = os.path.join(CH, chan, "validation.csv")
        fc = json.load(open(os.path.join(SRC, model_name, "feature_columns.json"),
                            encoding="utf-8"))["feature_columns"]
        df = pd.read_csv(vcsv, nrows=PARITY_ROWS)
        X = df[fc].astype("float64")

        booster = lgb.Booster(model_file=os.path.join(
            SRC, model_name, f"{model_name}_lgbm.txt"))
        local = np.clip(booster.predict(X.to_numpy()), 0.0, 1.0)

        payload = json.dumps({"day_type": day_type, "horizon_minutes": horizon,
                              "instances": X.to_dict(orient="records")})
        resp = retry(f"invoke {model_name}", lambda p=payload: rt.invoke_endpoint(
            EndpointName=ENDPOINT_NAME, ContentType="application/json",
            Accept="application/json", Body=p.encode("utf-8")))
        body = json.loads(resp["Body"].read().decode("utf-8"))
        aws = np.asarray(body["predicted_bike_ratio"], dtype="float64")

        routed_ok = body["model_name"] == model_name
        nfeat_ok = body["n_features_used"] == len(fc)
        diff = float(np.abs(local - aws).max())
        ok = routed_ok and nfeat_ok and diff <= TOL and len(aws) == len(local)
        all_ok = all_ok and ok
        rows.append({"model": model_name, "route": f"{day_type}/{horizon}",
                     "rows": int(len(aws)), "n_features": body["n_features_used"],
                     "local_first": float(local[0]), "aws_first": float(aws[0]),
                     "max_abs_diff": diff, "routed_correctly": routed_ok,
                     "pass": ok, "alert_thresholds": body["alert_thresholds"]})
        say(f"  {model_name}: rows={len(aws)} nfeat={body['n_features_used']} "
            f"max|diff|={diff:.3e} routed={routed_ok} PASS={ok}")

    # guard checks against the LIVE endpoint
    guards = {}
    wfc = json.load(open(os.path.join(SRC, "weekend_30m_bike_ratio",
                                      "feature_columns.json"),
                         encoding="utf-8"))["feature_columns"]
    wdf = pd.read_csv(os.path.join(CH, "weekend_30m_bike_ratio-1789266742",
                                   "validation.csv"), nrows=3)[wfc]
    for label, body_obj in (
        ("weekend_rows_to_weekday_route",
         {"day_type": "weekday", "horizon_minutes": 30,
          "instances": wdf.to_dict(orient="records")}),
        ("invalid_day_type",
         {"day_type": "holiday", "horizon_minutes": 30, "features": {}}),
        ("invalid_horizon",
         {"day_type": "weekday", "horizon_minutes": 45, "features": {}})):
        try:
            rt.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                               ContentType="application/json",
                               Accept="application/json",
                               Body=json.dumps(body_obj).encode("utf-8"))
            guards[label] = "NOT REJECTED (unexpected)"
            all_ok = False
        except Exception as e:
            guards[label] = f"rejected: {type(e).__name__}"
        say(f"  guard {label}: {guards[label]}")

    dump(status="InService", artifact_uri=artifact_uri, model=MODEL_NAME,
         config=CONFIG_NAME, endpoint=ENDPOINT_NAME, image=IMAGE,
         instance=INSTANCE, parity=rows, guards=guards,
         parity_all_pass=all_ok, tolerance=TOL)
    say("=" * 60)
    say(f"PARITY: {'PASS - all four routes match local' if all_ok else 'FAIL'}")
    say(f"ENDPOINT RUNNING / BILLING: {ENDPOINT_NAME} ({INSTANCE})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
