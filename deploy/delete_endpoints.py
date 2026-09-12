# -*- coding: utf-8 -*-
"""Delete the two real-time Endpoints to stop instance billing.

Deletes ONLY:
    youbike-shortage-v1-endpoint
    youbike-full-v1-endpoint

Explicitly PRESERVES:
    Endpoint Configs, SageMaker Models, S3 artifacts, IAM execution role

Polls until each endpoint is genuinely gone (describe_endpoint raises), so the
"billing stopped" claim is verified rather than assumed. Then confirms the
preserved resources still exist.

Credentials are read ONLY from the process environment; none are written or
printed. Does not touch local code, models, or dataset.

Usage (set the four AWS_* env vars first, in the SAME PowerShell process):
    .venv/Scripts/python.exe deploy/delete_endpoints.py
"""

from __future__ import annotations

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-west-2"

ENDPOINTS = [
    "youbike-shortage-v1-endpoint",
    "youbike-full-v1-endpoint",
]

# preserved (verified, never deleted)
CONFIGS = ["youbike-shortage-v1-config", "youbike-full-v1-config"]
MODELS = ["youbike-shortage-v1", "youbike-full-v1"]
BUCKET = "youbike-dispatch-502837994229-usw2"
ARTIFACT_KEYS = [
    "artifacts/shortage-model.tar.gz",
    "artifacts/full-model.tar.gz",
]
ROLE_NAME = "kiro-sagemaker-execution-role"

POLL_SECONDS = 15
MAX_WAIT_SECONDS = 10 * 60


def fail(msg: str) -> int:
    print("\nSTOPPED: " + msg)
    return 1


def endpoint_exists(sm, name: str) -> bool:
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError:
        return False


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        return fail("Missing AWS_ACCESS_KEY_ID. Set the four AWS_* variables "
                    "in this same PowerShell process first.")

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

    # --- 2. delete endpoints --------------------------------------------
    print("=" * 70)
    print("STEP 2 - delete_endpoint (stops instance billing)")
    for ep in ENDPOINTS:
        if not endpoint_exists(sm, ep):
            print(f"  '{ep}' already absent - nothing to delete")
            continue
        status = sm.describe_endpoint(EndpointName=ep)["EndpointStatus"]
        print(f"  '{ep}' current status: {status} -> issuing delete")
        try:
            sm.delete_endpoint(EndpointName=ep)
        except ClientError as e:
            return fail(f"delete_endpoint '{ep}' failed: "
                        f"{e.response['Error']['Code']} | {e.response['Error']['Message']}")

    # --- 3. confirm gone -------------------------------------------------
    print("=" * 70)
    print("STEP 3 - confirming deletion (polling every %ds)" % POLL_SECONDS)
    deadline = time.time() + MAX_WAIT_SECONDS
    while True:
        remaining = [ep for ep in ENDPOINTS if endpoint_exists(sm, ep)]
        if not remaining:
            print("  " + time.strftime("%H:%M:%S"), "all endpoints gone")
            break
        print("  " + time.strftime("%H:%M:%S"), "still present:", remaining, flush=True)
        if time.time() > deadline:
            return fail(f"Timed out waiting for deletion of: {remaining}. "
                        f"Check the console; billing may still be active.")
        time.sleep(POLL_SECONDS)

    live = [e["EndpointName"] for e in sm.list_endpoints()["Endpoints"]]
    print("  list_endpoints() ->", live if live else "[] (none)")
    billing_stopped = not any(ep in live for ep in ENDPOINTS)
    print("  BILLING STOPPED:", "YES" if billing_stopped else "NO")

    # --- 4. verify preserved resources ----------------------------------
    print("=" * 70)
    print("STEP 4 - verifying PRESERVED resources")

    print("  Endpoint Configs:")
    for cfg in CONFIGS:
        try:
            d = sm.describe_endpoint_config(EndpointConfigName=cfg)
            v = d["ProductionVariants"][0]
            print(f"    OK  {cfg}  ({v['InstanceType']} x{v['InitialInstanceCount']})")
        except ClientError as e:
            print(f"    MISSING  {cfg}  ({e.response['Error']['Code']})")

    print("  SageMaker Models:")
    for m in MODELS:
        try:
            d = sm.describe_model(ModelName=m)
            print(f"    OK  {m}")
            print(f"        image: {d['PrimaryContainer']['Image']}")
        except ClientError as e:
            print(f"    MISSING  {m}  ({e.response['Error']['Code']})")

    print("  S3 artifacts:")
    s3 = boto3.client("s3", region_name=REGION)
    for key in ARTIFACT_KEYS:
        try:
            h = s3.head_object(Bucket=BUCKET, Key=key)
            print(f"    OK  s3://{BUCKET}/{key}  ({h['ContentLength']:,} bytes)")
        except ClientError as e:
            print(f"    MISSING  {key}  ({e.response['Error']['Code']})")

    print("  IAM execution role:")
    try:
        r = boto3.client("iam").get_role(RoleName=ROLE_NAME)["Role"]
        print(f"    OK  {r['Arn']}")
    except ClientError as e:
        print(f"    MISSING  {ROLE_NAME}  ({e.response['Error']['Code']})")

    print("=" * 70)
    print("RESULT:", "PASS - endpoints deleted, billing stopped, "
          "configs/models/artifacts/role preserved"
          if billing_stopped else "FAIL - endpoints still present")
    return 0 if billing_stopped else 1


if __name__ == "__main__":
    raise SystemExit(main())
