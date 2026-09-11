# -*- coding: utf-8 -*-
"""Validation script for Task 4 (R5) of the YouBike dispatch MVP.

Run from the project root:
    python -m tests.validate_task4

Read-only w.r.t. the raw CSV and the training artifacts. Uses the models
persisted by Task 3 (models/shortage_xgb.json, full_xgb.json, feature_meta.json)
and asserts the Task 4 acceptance criteria against SEVERAL real observations
produced by the existing pipeline (load_clean -> build_targets -> build_features):
  * shortage probability and full probability are output (R5.1),
  * both probabilities lie in [0, 1] (R5.2),
  * risk level is classified Low / Medium / High using config thresholds (R5.3),
  * the structured dict matches design §9,
  * expected_risk_time = observation timestamp + 30 min,
  * a prepared row missing a required feature FAILS CLEARLY (V1 contract:
    predict.py never invents/fills a feature),
  * missing model artifacts raise a clear "train first" error (design §13).

V1 contract note: prediction operates on already-built build_features() rows;
there is NO arbitrary current-state reconstruction and NO LAG_FILL_VALUE
substitution at predict time.
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config, data_loader, features, predict


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


EXPECTED_KEYS = {
    "station", "lat", "lon",
    "current_bikes", "current_docks", "total_docks",
    "shortage_prob", "full_prob", "risk_level", "expected_risk_time",
}


def _build_observations(n: int):
    """Build several real prepared observations via the existing pipeline.

    Returns a list of (feature_row, meta, observation_ts) tuples where each
    feature_row is a genuine build_features() row (historically valid lag
    features) and meta carries the §9 current-state facts.
    """
    df = data_loader.load_clean()
    targets = features.build_targets(df)
    bundle = features.build_features(targets)

    total = len(bundle.X)
    # Spread the sample across the matrix so we exercise varied real rows.
    step = max(1, total // n)
    idxs = list(range(0, total, step))[:n]

    observations = []
    for i in idxs:
        feature_row = bundle.X.iloc[i]
        ts = bundle.timestamp.iloc[i]
        meta = {
            # station id is encoded in the feature row; use it as a stable label.
            "station": f"station_id={int(feature_row[config.COL_STATION_ID])}",
            "lat": float(feature_row[config.COL_LAT]),
            "lon": float(feature_row[config.COL_LON]),
            "current_bikes": int(feature_row[config.COL_AVAILABLE_BIKES]),
            "current_docks": int(feature_row[config.COL_AVAILABLE_DOCKS]),
            "total_docks": int(feature_row[config.COL_TOTAL_DOCKS]),
        }
        observations.append((feature_row, meta, ts))
    return observations


def main() -> int:
    c = Checker()

    artifacts = predict.load_artifacts()

    # --- Several real observations from the existing pipeline ---------------
    observations = _build_observations(50)
    c.check("built real observations from pipeline", len(observations) > 0,
            f"{len(observations)} observations")

    results = predict.predict_batch(observations, artifacts=artifacts)

    # R5.1: shortage + full probability output; structured dict per §9.
    c.check("R5.1 batch returns one result per observation",
            len(results) == len(observations),
            f"{len(results)} results for {len(observations)} observations")
    all_keys_ok = all(set(r.keys()) == EXPECTED_KEYS for r in results)
    c.check("§9 result dict has exactly the specified keys", all_keys_ok)

    probs_ok = all(
        isinstance(r["shortage_prob"], float) and isinstance(r["full_prob"], float)
        for r in results
    )
    c.check("R5.1 shortage_prob and full_prob present as floats", probs_ok)

    # R5.2: probabilities in [0, 1].
    range_ok = all(
        0.0 <= r["shortage_prob"] <= 1.0 and 0.0 <= r["full_prob"] <= 1.0
        for r in results
    )
    mins = min(min(r["shortage_prob"], r["full_prob"]) for r in results)
    maxs = max(max(r["shortage_prob"], r["full_prob"]) for r in results)
    c.check("R5.2 probabilities in [0,1]", range_ok, f"range=[{mins:.4f}, {maxs:.4f}]")

    # R5.3: risk level valid + correctly derived from max(shortage, full) using
    # the config thresholds (Low/Medium/High).
    levels_valid = all(r["risk_level"] in {"Low", "Medium", "High"} for r in results)
    c.check("R5.3 risk_level in {Low, Medium, High}", levels_valid)

    def expected_level(p):
        if p < config.RISK_LOW_MAX:
            return "Low"
        if p < config.RISK_HIGH_MIN:
            return "Medium"
        return "High"

    classification_ok = all(
        r["risk_level"] == expected_level(max(r["shortage_prob"], r["full_prob"]))
        for r in results
    )
    c.check("R5.3 risk_level = threshold(max(shortage, full))", classification_ok)

    # --- Boundary checks on the pure classifier (thresholds from config) ----
    boundary_cases = [
        (config.RISK_LOW_MAX - 0.001, "Low"),
        (config.RISK_LOW_MAX, "Medium"),
        (config.RISK_HIGH_MIN - 0.001, "Medium"),
        (config.RISK_HIGH_MIN, "High"),
        (0.99, "High"),
    ]
    boundary_ok = all(predict._classify_risk(p) == exp for p, exp in boundary_cases)
    c.check("R5.3 threshold boundaries classify correctly", boundary_ok,
            f"low_max={config.RISK_LOW_MAX}, high_min={config.RISK_HIGH_MIN}")

    # --- expected_risk_time = observation_ts + 30 min -----------------------
    row0, meta0, _ = observations[0]
    fixed_ts = pd.Timestamp("2026-03-30 08:00:00")
    single = predict.predict_station(row0, meta0, fixed_ts, artifacts=artifacts)
    delta_min = (pd.Timestamp(single["expected_risk_time"]) - fixed_ts).total_seconds() / 60.0
    c.check("§9 expected_risk_time = observation_ts + 30 min",
            abs(delta_min - 30.0) < 1e-6, f"delta_min={delta_min}")

    # --- Series input also accepted (build_features().X.iloc[i] is a Series) -
    c.check("prepared Series row accepted", isinstance(row0, pd.Series),
            f"type={type(row0).__name__}")

    # --- V1 contract: a missing required feature FAILS CLEARLY --------------
    incomplete_row = row0.drop(labels=[config.COL_PREV_BIKES])
    raised_missing = False
    missing_msg = ""
    try:
        predict.predict_station(incomplete_row, meta0, fixed_ts, artifacts=artifacts)
    except predict.MissingFeatureError as e:
        raised_missing = True
        missing_msg = str(e)
    c.check("V1 missing feature raises MissingFeatureError (no fill)", raised_missing)
    c.check("V1 missing-feature error names the missing column",
            config.COL_PREV_BIKES in missing_msg, missing_msg)

    # --- §13: missing model artifacts -> clear "train first" error ----------
    import os

    saved = predict._artifact_paths
    try:
        predict._artifact_paths = lambda: {
            "shortage_model": os.path.join(config.MODELS_DIR, "__does_not_exist__.json"),
            "full_model": os.path.join(config.MODELS_DIR, "__does_not_exist2__.json"),
            "feature_meta": os.path.join(config.MODELS_DIR, "__does_not_exist3__.json"),
        }
        raised = False
        message = ""
        try:
            predict.load_artifacts()
        except predict.ModelArtifactsMissingError as e:
            raised = True
            message = str(e)
        c.check("§13 missing artifacts raise ModelArtifactsMissingError", raised)
        c.check("§13 error message tells user to train first",
                "train" in message.lower(), message)
    finally:
        predict._artifact_paths = saved

    # Informational: show one prediction.
    print(f"    info: sample prediction -> {single['station']}: "
          f"shortage={single['shortage_prob']:.3f} full={single['full_prob']:.3f} "
          f"risk={single['risk_level']} at {single['expected_risk_time']}")

    return c.done()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
