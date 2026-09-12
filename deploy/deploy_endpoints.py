# -*- coding: utf-8 -*-
"""Create Endpoint Configs + Endpoints, wait, then verify inference parity.

Flow:
  1. sts.get_caller_identity()
  2. pre-flight (refuse to clobber existing configs/endpoints)
  3. create 2 Endpoint Configs (ml.m5.large, count 1)
  4. create 2 Endpoints
  5. poll until InService / Failed
  6. on Failed  -> print FailureReason + CloudWatch logs, then STOP
     on InService -> invoke both with the exact parity feature row and compare
                     against the stable local baselines

Endpoints are left running on purpose (caller will confirm deletion later).

COST NOTE: two ml.m5.large real-time endpoints bill continuously while they
exist (roughly $0.23/hour combined in us-west-2). Delete them when done.

Credentials are read ONLY from the process environment; none are written or
printed. Does not modify stable code, models, dataset, or Git.

Usage (set the four AWS_* env vars first, in the SAME PowerShell process):
    .venv/Scripts/python.exe deploy/deploy_endpoints.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-west-2"
INSTANCE_TYPE = "ml.m5.large"
INSTANCE_COUNT = 1
VARIANT = "AllTraffic"

# (model name, config name, endpoint name, baseline probability)
PLAN = [
    ("youbike-shortage-v1", "youbike-shortage-v1-config",
     "youbike-shortage-v1-endpoint", 0.02430786751210690),
    ("youbike-full-v1", "youbike-full-v1-config",
     "youbike-full-v1-endpoint", 0.00598118081688881),
]

TOL = 1e-9
POLL_SECONDS = 30
MAX_WAIT_SECONDS = 45 * 60

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_ROW = os.path.join(PROJECT_ROOT, "deploy", "test-xgb305", "feature_row.json")


def fail(msg: str) -> int:
    print("\nSTOPPED: " + msg)
    return 1


def load_feature_row() -> dict:
    """The exact 15-feature row used by every previous parity test."""
    if os.path.exists(FEATURE_ROW):
        with open(FEATURE_ROW, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fallback: rebuild from the stable pipeline (read-only).
    sys.path.insert(0, PROJECT_ROOT)
    from src import data_loader, features
    df = data_loader.load_clean()
    bundle = features.build_features(features.build_targets(df))
    row = bundle.X.iloc[0]
    return {
        "timestamp": str(bundle.timestamp.iloc[0]),
        "features": {k: float(v) for k, v in row.items()},
        "feature_order": list(row.index),
    }


def dump_logs(logs_client, endpoint_name: str, max_events: int = 80) -> None:
    group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    print(f"  CloudWatch log group: {group}")
    try:
        streams = logs_client.describe_log_streams(
            logGroupName=group, orderBy="LastEventTime", descending=True, limit=1
        )["logStreams"]
    except ClientError as e:
        print("    (no log group / not accessible:",
              e.response["Error"]["Code"], ")")
        return
    if not streams:
        print("    (no log streams)")
        return
    stream = streams[0]["logStreamName"]
    print(f"  stream: {stream}")
    try:
        events = logs_client.get_log_events(
            logGroupName=group, logStreamName=stream,
            limit=max_events, startFromHead=False,
        )["events"]
    except ClientError as e:
        print("    (cannot read events:", e.response["Error"]["Code"], ")")
        return
    print("  --- last %d log lines ---" % len(events))
    for ev in events:
        print("   ", ev["message"].rstrip())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return fail("Missing AWS_ACCESS_KEY_ID. Set the four AWS_* variables "
                    "in this same PowerShell process first.")

    print("=" * 72)
    print("STEP 1 - sts.get_caller_identity()")
    try:
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    except (ClientError, NoCredentialsError) as e:
        return fail(f"STS failed ({type(e).__name__}): {e}")
    print("  Account:", ident["Account"])
    print("  ARN    :", ident["Arn"])
    print("  Region :", REGION)

    sm = boto3.client("sagemaker", region_name=REGION)
    smrt = boto3.client("sagemaker-runtime", region_name=REGION)
    logs = boto3.client("logs", region_name=REGION)

    # --- pre-flight ------------------------------------------------------
    print("=" * 72)
    print("STEP 2 - pre-flight")
    for model, cfg, ep, _ in PLAN:
        try:
            sm.describe_model(ModelName=model)
            print(f"  model '{model}' exists - OK")
        except ClientError:
            return fail(f"Model '{model}' does not exist. Create the Models first.")
        for kind, fn, kw in (
            ("endpoint config", sm.describe_endpoint_config, {"EndpointConfigName": cfg}),
            ("endpoint", sm.describe_endpoint, {"EndpointName": ep}),
        ):
            try:
                fn(**kw)
                return fail(f"{kind} '{list(kw.values())[0]}' already exists. "
                            f"Delete it first; this script will not overwrite.")
            except ClientError:
                pass
        print(f"  '{cfg}' / '{ep}' do not exist - OK to create")

    # --- create configs --------------------------------------------------
    print("=" * 72)
    print("STEP 3 - create_endpoint_config")
    for model, cfg, ep, _ in PLAN:
        try:
            r = sm.create_endpoint_config(
                EndpointConfigName=cfg,
                ProductionVariants=[{
                    "VariantName": VARIANT,
                    "ModelName": model,
                    "InitialInstanceCount": INSTANCE_COUNT,
                    "InstanceType": INSTANCE_TYPE,
                    "InitialVariantWeight": 1.0,
                }],
            )
        except ClientError as e:
            return fail(f"create_endpoint_config '{cfg}' failed: "
                        f"{e.response['Error']['Code']} | {e.response['Error']['Message']}")
        print(f"  created {cfg} ({INSTANCE_TYPE} x{INSTANCE_COUNT})")
        print(f"    {r['EndpointConfigArn']}")

    # --- create endpoints ------------------------------------------------
    print("=" * 72)
    print("STEP 4 - create_endpoint  (billing starts once InService)")
    for model, cfg, ep, _ in PLAN:
        try:
            r = sm.create_endpoint(EndpointName=ep, EndpointConfigName=cfg)
        except ClientError as e:
            return fail(f"create_endpoint '{ep}' failed: "
                        f"{e.response['Error']['Code']} | {e.response['Error']['Message']}")
        print(f"  creating {ep}")
        print(f"    {r['EndpointArn']}")

    # --- poll ------------------------------------------------------------
    print("=" * 72)
    print("STEP 5 - waiting for InService (polling every %ds)" % POLL_SECONDS)
    names = [ep for _, _, ep, _ in PLAN]
    deadline = time.time() + MAX_WAIT_SECONDS
    statuses = {}
    while True:
        statuses = {}
        for ep in names:
            statuses[ep] = sm.describe_endpoint(EndpointName=ep)["EndpointStatus"]
        print("  " + time.strftime("%H:%M:%S"), statuses, flush=True)
        if all(s in ("InService", "Failed") for s in statuses.values()):
            break
        if time.time() > deadline:
            print("  poll timeout reached")
            break
        time.sleep(POLL_SECONDS)

    failed = [ep for ep, s in statuses.items() if s == "Failed"]
    not_ready = [ep for ep, s in statuses.items() if s not in ("InService", "Failed")]

    if failed or not_ready:
        print("=" * 72)
        print("ENDPOINT NOT HEALTHY - diagnostics")
        for ep in failed:
            d = sm.describe_endpoint(EndpointName=ep)
            print("-" * 72)
            print("  Endpoint:", ep, "| Status: Failed")
            print("  FailureReason:")
            print("   ", d.get("FailureReason", "(none)"))
            dump_logs(logs, ep)
        for ep in not_ready:
            print("-" * 72)
            print("  Endpoint:", ep, "| Status:", statuses[ep], "(still not ready)")
            dump_logs(logs, ep)
        return fail("One or more endpoints did not reach InService. "
                    "No model / container / dependency / architecture change made. "
                    "Endpoints left in place for inspection.")

    # --- invoke + parity -------------------------------------------------
    print("=" * 72)
    print("STEP 6 - inference parity check against AWS endpoints")
    payload = load_feature_row()
    body = json.dumps(payload["features"]).encode("utf-8")
    print("  observation timestamp:", payload["timestamp"])
    print("  feature count        :", len(payload["feature_order"]))

    print("-" * 72)
    print(f"{'model':<10}{'local stable':>24}{'AWS endpoint':>24}{'abs diff':>14}")
    all_ok = True
    for model, cfg, ep, baseline in PLAN:
        try:
            resp = smrt.invoke_endpoint(
                EndpointName=ep,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
        except ClientError as e:
            return fail(f"invoke_endpoint '{ep}' failed: "
                        f"{e.response['Error']['Code']} | {e.response['Error']['Message']}")
        raw = resp["Body"].read().decode("utf-8")
        try:
            got = float(json.loads(raw)["probabilities"][0])
        except Exception:
            return fail(f"Unexpected response body from '{ep}': {raw}")
        diff = abs(got - baseline)
        if diff > TOL:
            all_ok = False
        label = "shortage" if "shortage" in ep else "full"
        print(f"{label:<10}{baseline:>24.17f}{got:>24.17f}{diff:>14.3e}")

    print("=" * 72)
    print("TOLERANCE:", f"{TOL:.0e}")
    print("RESULT:", "PASS - AWS endpoints match local stable predictions"
          if all_ok else "FAIL - probabilities differ")
    print()
    print("Endpoints are STILL RUNNING and billing (~$0.23/hour combined).")
    print("Endpoints:", ", ".join(names))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
