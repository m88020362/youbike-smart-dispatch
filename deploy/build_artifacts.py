# -*- coding: utf-8 -*-
"""Package shortage-model.tar.gz and full-model.tar.gz for SageMaker.

Each archive contains one model JSON + feature_meta.json + code/inference.py,
matching the SageMaker XGBoost framework container layout:

    <model>_xgb.json
    feature_meta.json
    code/inference.py

No requirements.txt is shipped: the XGBoost 3.0-5 container already provides
xgboost 3.0.5 and pandas, and the Booster-based inference.py needs nothing else.

Reads stable artifacts read-only from models/; writes tar.gz into
deploy/artifacts/. Does not modify src/, models/, or dataset/.
"""

from __future__ import annotations

import os
import tarfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CODE_DIR = os.path.join(PROJECT_ROOT, "deploy", "code")
OUT_DIR = os.path.join(PROJECT_ROOT, "deploy", "artifacts")

ARTIFACTS = {
    "shortage-model.tar.gz": "shortage_xgb.json",
    "full-model.tar.gz": "full_xgb.json",
}


def build_one(archive_name: str, model_filename: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, archive_name)

    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(os.path.join(MODELS_DIR, model_filename), arcname=model_filename)
        tar.add(
            os.path.join(MODELS_DIR, "feature_meta.json"),
            arcname="feature_meta.json",
        )
        tar.add(os.path.join(CODE_DIR, "inference.py"), arcname="code/inference.py")
    return out_path


def main() -> int:
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for archive_name, model_filename in ARTIFACTS.items():
        path = build_one(archive_name, model_filename)
        size = os.path.getsize(path)
        print(f"built {path}  ({size:,} bytes)")
        with tarfile.open(path, "r:gz") as tar:
            for m in tar.getmembers():
                print(f"    {m.name:<24} {m.size:>10,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
