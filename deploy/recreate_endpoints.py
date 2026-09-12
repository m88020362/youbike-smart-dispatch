# -*- coding: utf-8 -*-
"""Demo-day fast path: recreate the two already-verified Endpoints.

Reuses the EXISTING, already-validated Endpoint Configs. It does NOT create or
touch Models, Endpoint Configs, S3 artifacts or the IAM role -- those were
verified earlier and are deliberately left alone. The only AWS resources this
creates are the two Endpoints themselves.

    verify STS identity
      -> confirm youbike-shortage-v1-config / youbike-full-v1-config exist
        -> confirm the same-named Endpoints do NOT exist
          -> create_endpoint from the existing configs
            -> wait for InService (or stop on Failed with FailureReason)

Separate from deploy/deploy_endpoints.py on purpose: that script is the verified
first-time deployment path (it creates configs and refuses to touch existing
ones). This one is the idempotent-ish restore path for demo day.

Credentials are read ONLY from the process environment; none are written or
printed.

COST NOTE: two ml.m5.large endpoints bill continuously once InService (roughly
$0.23/hour combined in us-west-2). Run deploy/delete_endpoints.py when done.

Usage (set the four AWS_* env vars first, in the SAME PowerShell process):
    .venv/Scripts/python.exe deploy/recreate_endpoints.py
"""

from __future__ import annotations

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-west-2"

# (endpoint name, existing endpoint-config name)
PAIRS = [
    ("youbike-shortage-v1-endpoint", "youbike-shortage-v1-config"),
    ("youbike-full-v1-endpoint", "youbike-full-v1-config"),
]

POLL_SECONDS = 20
MAX_WAIT_SECONDS = 45 * 60


def fail(msg: str) -> int:
    print("\nSTOPPED: " + msg)
    return 1


def _exists(fn, **kw) -> bool:
    try:
        fn(**kw)
        return True
    except ClientError:
        return False


def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60} 分 {total % 60} 秒"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return fail(
            "Missing AWS_ACCESS_KEY_ID. Set the four AWS_* variables in this "
            "same PowerShell process first."
        )

    # --- 1. identity -----------------------------------------------------
    print("=" * 70)
    print("STEP 1 - sts.get_caller_identity()")
    try:
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    except (ClientError, NoCredentialsError) as e:
        return fail(f"STS failed ({type(e).__name__}): {e}")
    print("  Account:", ident["Account"])
    print("  ARN    :", ident["Arn"])
    print("  Region :", REGION)

    sm = boto3.client("sagemaker", region_name=REGION)

    # --- 2. configs must already exist -----------------------------------
    print("=" * 70)
    print("STEP 2 - verifying existing Endpoint Configs (not recreated)")
    for endpoint_name, config_name in PAIRS:
        try:
            desc = sm.describe_endpoint_config(EndpointConfigName=config_name)
        except ClientError as e:
            return fail(
                f"Endpoint Config '{config_name}' not found "
                f"({e.response['Error']['Code']}). This script only reuses "
                f"existing configs -- run deploy/deploy_endpoints.py for a "
                f"first-time deployment."
            )
        variant = desc["ProductionVariants"][0]
        print(
            f"  OK  {config_name}  "
            f"({variant['InstanceType']} x{variant['InitialInstanceCount']}, "
            f"model={variant['ModelName']})"
        )

    # --- 3. endpoints must NOT already exist -----------------------------
    print("=" * 70)
    print("STEP 3 - confirming Endpoints do not already exist")
    for endpoint_name, _ in PAIRS:
        if _exists(sm.describe_endpoint, EndpointName=endpoint_name):
            status = sm.describe_endpoint(EndpointName=endpoint_name)[
                "EndpointStatus"
            ]
            return fail(
                f"Endpoint '{endpoint_name}' already exists (status={status}). "
                f"Delete it first with deploy/delete_endpoints.py, or just use "
                f"it as-is if it is already InService."
            )
        print(f"  '{endpoint_name}' absent - OK to create")

    # --- 4. create endpoints ---------------------------------------------
    print("=" * 70)
    print("STEP 4 - create_endpoint from existing configs")
    started = time.monotonic()
    for endpoint_name, config_name in PAIRS:
        try:
            resp = sm.create_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=config_name
            )
        except ClientError as e:
            return fail(
                f"create_endpoint '{endpoint_name}' failed: "
                f"{e.response['Error']['Code']} | {e.response['Error']['Message']}"
            )
        print(f"  creating {endpoint_name}  (config: {config_name})")
        print(f"    {resp['EndpointArn']}")

    # --- 5. wait for InService -------------------------------------------
    print("=" * 70)
    print(f"STEP 5 - waiting for InService (polling every {POLL_SECONDS}s)")
    names = [name for name, _ in PAIRS]
    deadline = time.time() + MAX_WAIT_SECONDS
    statuses = {}
    while True:
        statuses = {
            name: sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
            for name in names
        }
        print("  " + time.strftime("%H:%M:%S"), statuses, flush=True)
        if all(s in ("InService", "Failed") for s in statuses.values()):
            break
        if time.time() > deadline:
            print("  poll timeout reached")
            break
        time.sleep(POLL_SECONDS)

    elapsed = time.monotonic() - started

    failed = [n for n, s in statuses.items() if s == "Failed"]
    pending = [n for n, s in statuses.items() if s not in ("InService", "Failed")]

    if failed or pending:
        print("=" * 70)
        print("ENDPOINT NOT HEALTHY")
        for name in failed:
            desc = sm.describe_endpoint(EndpointName=name)
            print("-" * 70)
            print(f"  Endpoint: {name}  | Status: Failed")
            print("  FailureReason:")
            print("   ", desc.get("FailureReason", "(none)"))
            print(
                f"  CloudWatch log group: /aws/sagemaker/Endpoints/{name}"
            )
        for name in pending:
            print("-" * 70)
            print(f"  Endpoint: {name}  | Status: {statuses[name]} (not ready)")
        return fail(
            "One or more endpoints did not reach InService. Nothing was "
            "changed in the Models, Configs, artifacts or IAM role."
        )

    # --- 6. summary ------------------------------------------------------
    print("=" * 70)
    print("STEP 6 - summary")
    for endpoint_name, config_name in PAIRS:
        desc = sm.describe_endpoint(EndpointName=endpoint_name)
        print("-" * 70)
        print("  Endpoint name :", desc["EndpointName"])
        print("  Status        :", desc["EndpointStatus"])
        print("  Config name   :", config_name)
        print("  Endpoint ARN  :", desc["EndpointArn"])
    print("-" * 70)
    print("  建立耗時       :", _fmt_duration(elapsed))

    print("=" * 70)
    print("RESULT: PASS - both endpoints InService from existing configs")
    print("No Model, Endpoint Config, artifact or IAM change was made.")
    print()
    print("Next:")
    print("  1) .venv\\Scripts\\streamlit run app.py")
    print("  2) sidebar 開發者選項 -> 預測來源 -> AWS SageMaker")
    print("  3) demo 結束後: .venv\\Scripts\\python.exe deploy\\delete_endpoints.py")
    print()
    print("Endpoints are billing now (~$0.23/hour combined).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
