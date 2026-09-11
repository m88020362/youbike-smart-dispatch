# -*- coding: utf-8 -*-
"""Validation script for Task 3 (R4) of the YouBike dispatch MVP.

Run from the project root:
    python -m tests.validate_task3

Read-only w.r.t. the raw CSV. Trains the models (writing to models/ and
outputs/) and then asserts the Task 3 acceptance criteria:
  * baseline + XGBoost trained for both shortage and full (R4.1),
  * train/test split is by TIME, not random (R4.2),
  * understandable metrics emitted: precision/recall/F1/ROC-AUC/confusion (R4.3),
  * class imbalance handled via scale_pos_weight (R4.4),
  * models + metrics persisted to models/ and outputs/ (R4.6),
  * predicted probabilities lie in [0, 1].
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from xgboost import XGBClassifier

from src import config, data_loader, features, train


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, condition, detail=""):
        self.checks += 1
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not condition:
            self.failures.append(name)

    def done(self):
        print("-" * 60)
        if self.failures:
            print(f"RESULT: {len(self.failures)}/{self.checks} checks FAILED: "
                  f"{self.failures}")
            return 1
        print(f"RESULT: all {self.checks} checks PASSED")
        return 0


def main() -> int:
    c = Checker()

    # --- Train end-to-end ---------------------------------------------------
    metrics = train.train()

    # R4.6: artifacts persisted.
    shortage_model_path = os.path.join(config.MODELS_DIR, "shortage_xgb.json")
    full_model_path = os.path.join(config.MODELS_DIR, "full_xgb.json")
    feature_meta_path = os.path.join(config.MODELS_DIR, "feature_meta.json")
    metrics_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")

    c.check("R4.6 shortage model saved", os.path.exists(shortage_model_path))
    c.check("R4.6 full model saved", os.path.exists(full_model_path))
    c.check("R4.6 feature_meta.json saved", os.path.exists(feature_meta_path))
    c.check("R4.6 metrics.json saved", os.path.exists(metrics_path))
    c.check(
        "R4.3/R4.6 confusion-matrix + ROC plots saved",
        os.path.exists(os.path.join(config.OUTPUTS_DIR, "shortage_confusion_matrix.png"))
        and os.path.exists(os.path.join(config.OUTPUTS_DIR, "full_confusion_matrix.png"))
        and os.path.exists(os.path.join(config.OUTPUTS_DIR, "shortage_roc.png"))
        and os.path.exists(os.path.join(config.OUTPUTS_DIR, "full_roc.png")),
    )

    # metrics.json parses and matches the returned dict.
    with open(metrics_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    c.check("metrics.json is valid JSON with both labels",
            "shortage" in on_disk and "full" in on_disk)

    # R4.1: baseline + XGBoost metrics present for both labels.
    for label in ("shortage", "full"):
        res = metrics[label]
        c.check(f"R4.1 {label}: baseline metrics present",
                {"precision", "recall", "f1", "roc_auc", "confusion_matrix"}
                .issubset(res["baseline"].keys()))
        c.check(f"R4.1 {label}: xgboost metrics present",
                {"precision", "recall", "f1", "roc_auc", "confusion_matrix"}
                .issubset(res["xgboost"].keys()))
        # R4.4: imbalance handled -> scale_pos_weight recorded and >= 1 here.
        c.check(f"R4.4 {label}: scale_pos_weight applied",
                res["xgboost"]["scale_pos_weight"] >= 1.0,
                f"spw={res['xgboost']['scale_pos_weight']:.2f}")

    # R4.2: TIME-based split -> train max timestamp <= test min timestamp.
    split = metrics["split"]
    c.check("R4.2 split is time-based (train_max <= test_min)",
            split["boundary_ok"] is True,
            f"train_max={split['train_time_max']}, test_min={split['test_time_min']}")
    c.check("R4.2 split honors TRAIN_FRACTION",
            abs(split["n_train"] / split["n_total"] - config.TRAIN_FRACTION) < 0.001,
            f"train_frac={split['n_train'] / split['n_total']:.4f}")

    # --- Probabilities in [0,1] from the persisted models -------------------
    df = data_loader.load_clean()
    tgt = features.build_targets(df)
    bundle = features.build_features(tgt)

    for label, path in (("shortage", shortage_model_path), ("full", full_model_path)):
        m = XGBClassifier()
        m.load_model(path)
        proba = m.predict_proba(bundle.X)[:, 1]
        c.check(f"{label}: predict_proba in [0,1]",
                float(np.min(proba)) >= 0.0 and float(np.max(proba)) <= 1.0,
                f"range=[{np.min(proba):.4f}, {np.max(proba):.4f}]")

    # feature_meta column order matches config.
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        fm = json.load(f)
    c.check("feature_meta feature_columns matches config",
            fm["feature_columns"] == list(config.FEATURE_COLUMNS))
    c.check("feature_meta has station/district encodings",
            len(fm["station_encoding"]) > 0 and len(fm["district_encoding"]) > 0,
            f"stations={len(fm['station_encoding'])}, "
            f"districts={len(fm['district_encoding'])}")

    # Informational: headline metrics.
    for label in ("shortage", "full"):
        x = metrics[label]["xgboost"]
        print(f"    info: {label} XGB  P={x['precision']:.3f} R={x['recall']:.3f} "
              f"F1={x['f1']:.3f} AUC={x['roc_auc']}")

    return c.done()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
