# -*- coding: utf-8 -*-
"""Delete the V1 multimodel Endpoint to stop instance billing.

Deletes ONLY youbike-v1-multimodel-endpoint. Preserves the Endpoint Config,
the SageMaker Model, the S3 artifact and the IAM role, so the endpoint can be
recreated later without rebuilding or re-uploading anything.

Usage (four AWS_* env vars set in the SAME PowerShell process):
    v1_training/venv/Scripts/python.exe deploy_v1/delete_endpoint.py
"""
from __future__ import annotations
import os, sys, time
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
ENDPOINT = "youbike-v1-multimodel-endpoint"
CONFIG = "youbike-v1-multimodel-config"
MODEL = "youbike-v1-multimodel"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("STOPPED: set the four AWS_* env vars in this process first")
        return 1
    sm = boto3.client("sagemaker", region_name=REGION)
    i = boto3.client("sts", region_name=REGION).get_caller_identity()
    print("Account:", i["Account"], "Region:", REGION)

    try:
        st = sm.describe_endpoint(EndpointName=ENDPOINT)["EndpointStatus"]
        print(f"{ENDPOINT} status={st} -> deleting")
        sm.delete_endpoint(EndpointName=ENDPOINT)
    except ClientError as e:
        print(f"{ENDPOINT} not present ({e.response['Error']['Code']}) - nothing to delete")
        return 0

    deadline = time.time() + 10*60
    while True:
        try:
            sm.describe_endpoint(EndpointName=ENDPOINT)
            if time.time() > deadline:
                print("TIMEOUT waiting for deletion - check console, billing may continue")
                return 1
            time.sleep(15)
        except ClientError:
            break
    live = [e["EndpointName"] for e in sm.list_endpoints()["Endpoints"]]
    print("endpoint deleted. remaining endpoints:", live if live else "[] none")
    print("BILLING STOPPED:", "YES" if ENDPOINT not in live else "NO")

    for label, fn, kw in (("EndpointConfig", sm.describe_endpoint_config,
                           {"EndpointConfigName": CONFIG}),
                          ("Model", sm.describe_model, {"ModelName": MODEL})):
        try:
            fn(**kw); print(f"PRESERVED {label}: {list(kw.values())[0]}")
        except ClientError:
            print(f"NOTE {label} missing: {list(kw.values())[0]}")
    print("PRESERVED S3 artifact + IAM role (untouched)")
    print("\nTo bring it back: v1_training/venv/Scripts/python.exe deploy_v1/deploy_endpoint.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
