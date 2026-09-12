# -*- coding: utf-8 -*-
"""Verify AWS identity, upload the two model artifacts, verify their sizes.

Credentials are read ONLY from the current process environment variables
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN /
AWS_DEFAULT_REGION). Nothing is written to disk, and no credential value is
printed. Creates no AWS resources other than overwriting the two S3 objects.

Usage (set the four env vars first, in the SAME PowerShell process):
    .venv/Scripts/python.exe deploy/upload_artifacts.py
"""

from __future__ import annotations

import os
import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

BUCKET = "youbike-dispatch-502837994229-usw2"
REGION = "us-west-2"

UPLOADS = [
    ("deploy/artifacts/shortage-model.tar.gz", "artifacts/shortage-model.tar.gz"),
    ("deploy/artifacts/full-model.tar.gz", "artifacts/full-model.tar.gz"),
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fail(msg: str) -> int:
    print("\nSTOPPED: " + msg)
    return 1


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")

    # --- 0. sanity: are the env vars present at all? ---------------------
    missing = [
        v
        for v in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(v)
    ]
    if missing:
        return fail(
            f"Missing environment variable(s): {missing}. "
            f"Set the four AWS_* variables in this same PowerShell process first."
        )

    # --- 1. verify identity ---------------------------------------------
    print("=" * 68)
    print("STEP 1 - sts.get_caller_identity()")
    try:
        ident = boto3.client("sts", region_name=REGION).get_caller_identity()
    except (ClientError, NoCredentialsError) as e:
        return fail(f"STS failed ({type(e).__name__}): {e}")

    print("  Account:", ident["Account"])
    print("  ARN    :", ident["Arn"])
    print("  UserId :", ident["UserId"])
    print("  Region :", REGION)

    # --- 2. upload -------------------------------------------------------
    print("=" * 68)
    print("STEP 2 - upload artifacts (overwrite)")
    s3 = boto3.client("s3", region_name=REGION)

    local_sizes = {}
    for local_rel, key in UPLOADS:
        local_path = os.path.join(PROJECT_ROOT, local_rel)
        if not os.path.exists(local_path):
            return fail(f"Local artifact not found: {local_path}")
        local_sizes[key] = os.path.getsize(local_path)
        try:
            s3.upload_file(local_path, BUCKET, key)
        except (ClientError, NoCredentialsError) as e:
            return fail(f"Upload of {key} failed ({type(e).__name__}): {e}")
        print(f"  uploaded {key}  (local {local_sizes[key]:,} bytes)")

    # --- 3. verify with head_object --------------------------------------
    print("=" * 68)
    print("STEP 3 - head_object verification")
    all_ok = True
    for _, key in UPLOADS:
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key)
        except ClientError as e:
            return fail(f"head_object for {key} failed: {e}")
        remote = head["ContentLength"]
        expected = local_sizes[key]
        match = remote == expected
        all_ok = all_ok and match
        print(f"  s3://{BUCKET}/{key}")
        print(f"      size on S3   : {remote:,} bytes")
        print(f"      local size   : {expected:,} bytes")
        print(f"      match        : {'YES' if match else 'NO'}")
        print(f"      last modified: {head['LastModified']}")
        print(f"      etag         : {head['ETag'].strip(chr(34))}")

    print("=" * 68)
    print("RESULT:", "PASS - both artifacts uploaded and sizes match"
          if all_ok else "FAIL - size mismatch")
    print("No SageMaker Model / Endpoint Config / Endpoint was created.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
