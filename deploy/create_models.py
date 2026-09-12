# -*- coding: utf-8 -*-
"""Create the two SageMaker Models (shortage / full) - nothing else.

Uses the SageMaker XGBoost 3.0-5 framework container and the code/inference.py
already packaged inside each model.tar.gz. Key fix vs the earlier failed
attempt: SAGEMAKER_SUBMIT_DIRECTORY points at the container-local path
/opt/ml/model/code (where the extracted artifact's code/ directory lands),
NOT at the S3 tar URI. That is what caused "No module named 'inference'".

No requirements.txt is used: the container's built-in xgboost 3.0.5 is what the
Booster-based inference.py needs, and it was verified locally to reproduce the
stable probabilities exactly.

Credentials are read ONLY from the process environment. Nothing is written to
disk and no credential value is printed.

Creates ONLY SageMaker Models. No Endpoint Config, no Endpoint.

Usage (set the four AWS_* env vars first, in the SAME PowerShell process):
    .venv/Scripts/python.exe deploy/create_models.py
"""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

REGION = "us-west-2"
ROLE_ARN = "arn:aws:iam::502837994229:role/kiro-sagemaker-execution-role"
BUCKET = "youbike-dispatch-502837994229-usw2"

# us-west-2 SageMaker XGBoost registry account (empirically confirmed: the
# 1.7-1 image pulled and started from this same registry path).
IMAGE_URI = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:3.0-5"

MODELS = [
    ("youbike-shortage-v1", f"s3://{BUCKET}/artifacts/shortage-model.tar.gz"),
    ("youbike-full-v1", f"s3://{BUCKET}/artifacts/full-model.tar.gz"),
]


def fail(msg: str) -> int:
    print("\nSTOPPED: " + msg)
    return 1


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

    # --- 2. container image ---------------------------------------------
    print("=" * 70)
    print("STEP 2 - container image")
    print("  XGBoost framework container (3.0-5):")
    print("   ", IMAGE_URI)
    print("  built-in xgboost 3.0.5 / python 3.10 / sklearn 1.8.0")
    print("  custom inference: code/inference.py (Booster API, no sklearn dep)")

    sm = boto3.client("sagemaker", region_name=REGION)

    # --- 3. pre-flight: refuse to clobber existing models ---------------
    print("=" * 70)
    print("STEP 3 - pre-flight check")
    for name, _ in MODELS:
        try:
            sm.describe_model(ModelName=name)
            return fail(
                f"Model '{name}' already exists. Delete it first if you intend "
                f"to recreate it; this script will not overwrite."
            )
        except ClientError as e:
            if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
                print(f"  '{name}' does not exist yet - OK to create")
            else:
                return fail(f"describe_model check failed for {name}: {e}")

    # --- 4. create models ------------------------------------------------
    print("=" * 70)
    print("STEP 4 - create_model")
    for name, artifact in MODELS:
        container = {
            "Image": IMAGE_URI,
            "ModelDataUrl": artifact,
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                # container-local path where model.tar.gz's code/ is extracted
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_REGION": REGION,
            },
        }
        try:
            resp = sm.create_model(
                ModelName=name,
                ExecutionRoleArn=ROLE_ARN,
                PrimaryContainer=container,
            )
        except ClientError as e:
            return fail(
                f"create_model failed for '{name}': "
                f"{e.response['Error']['Code']} | {e.response['Error']['Message']}"
            )
        print(f"  created {name} -> {resp['ModelArn']}")

    # --- 5. verify -------------------------------------------------------
    print("=" * 70)
    print("STEP 5 - describe_model verification")
    for name, _ in MODELS:
        try:
            d = sm.describe_model(ModelName=name)
        except ClientError as e:
            return fail(f"describe_model failed for {name}: {e}")
        c = d["PrimaryContainer"]
        print("-" * 70)
        print("  Model name        :", d["ModelName"])
        print("  Model ARN         :", d["ModelArn"])
        print("  ModelDataUrl      :", c["ModelDataUrl"])
        print("  ExecutionRoleArn  :", d["ExecutionRoleArn"])
        print("  Container image   :", c["Image"])
        print("  Environment variables:")
        for k in sorted(c.get("Environment", {})):
            print(f"      {k} = {c['Environment'][k]}")

    print("=" * 70)
    print("RESULT: PASS - both SageMaker Models created and verified")
    print("No Endpoint Config and no Endpoint was created.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
