# -*- coding: utf-8 -*-
"""Local parity test: bundled single-endpoint handler vs the 4 original models.

For each model, real validation feature rows (the exact CSV channel uploaded to
the successful SageMaker job) are scored two ways:

  A) baseline  - the ORIGINAL per-model booster from v1_training/enriched/
  B) bundled   - through the single-endpoint handler extracted from
                 deploy_v1/artifacts/youbike-v1-multimodel.tar.gz, routed only
                 by day_type + horizon_minutes (JSON in, JSON out)

Both must agree exactly. No endpoint is created; nothing is uploaded.
"""

from __future__ import annotations

import importlib.util, json, os, shutil, tarfile, tempfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(ROOT) if os.path.basename(ROOT) == "deploy_v1" else ROOT
SRC = os.path.join(ROOT, "v1_training", "enriched")
CH = os.path.join(ROOT, "v1_training", "_enriched_channels")
TAR = os.path.join(ROOT, "deploy_v1", "artifacts", "youbike-v1-multimodel.tar.gz")

CASES = [
    ("weekday", 30, "weekday_30m_bike_ratio", "weekday_30m_bike_ratio"),
    ("weekday", 60, "weekday_60m_bike_ratio", "weekday_60m_bike_ratio-1789266742"),
    ("weekend", 30, "weekend_30m_bike_ratio", "weekend_30m_bike_ratio-1789266742"),
    ("weekend", 60, "weekend_60m_bike_ratio", "weekend_60m_bike_ratio-1789266742"),
]
N_ROWS = 5000
TOL = 0.0   # demand exact equality

L = []
def say(m):
    print(m, flush=True); L.append(m)

# ---- extract the bundle exactly as a container would -------------------
work = tempfile.mkdtemp(prefix="v1_bundle_")
with tarfile.open(TAR, "r:gz") as tar:
    tar.extractall(work)
say(f"extracted bundle -> {sorted(os.listdir(work))}")

spec = importlib.util.spec_from_file_location(
    "bundled_inf", os.path.join(work, "code", "inference.py"))
inf = importlib.util.module_from_spec(spec); spec.loader.exec_module(inf)

say("")
say("=== model_fn: loading all four models from ONE artifact ===")
bundle = inf.model_fn(work)
say(f"routes available: {sorted(bundle['models'])}")
say(f"fixed events: low<{bundle['event_low']} high>{bundle['event_high']}")
say("")

import lightgbm as lgb
all_ok = True
rows = []

for day_type, horizon, model_name, chan_dir in CASES:
    vcsv = os.path.join(CH, chan_dir, "validation.csv")
    if not os.path.exists(vcsv):
        say(f"!! missing validation channel for {model_name}: {vcsv}")
        all_ok = False
        continue
    df = pd.read_csv(vcsv, nrows=N_ROWS)
    fc = json.load(open(os.path.join(SRC, model_name, "feature_columns.json"),
                        encoding="utf-8"))["feature_columns"]
    X = df[fc].astype("float64")

    # A) baseline: original booster, direct
    base_booster = lgb.Booster(
        model_file=os.path.join(SRC, model_name, f"{model_name}_lgbm.txt"))
    base = np.clip(base_booster.predict(X.to_numpy()), 0.0, 1.0)

    # B) bundled: full JSON round trip through the routed handler
    payload = json.dumps({"day_type": day_type, "horizon_minutes": horizon,
                          "instances": df[fc].to_dict(orient="records")})
    parsed = inf.input_fn(payload, "application/json")
    pred = inf.predict_fn(parsed, bundle)
    out = json.loads(inf.output_fn(pred))
    got = np.asarray(out["predicted_bike_ratio"], dtype="float64")

    assert out["model_name"] == model_name, \
        f"ROUTING ERROR: asked {day_type}/{horizon} got {out['model_name']}"
    diff = np.abs(base - got)
    ok = bool(diff.max() <= TOL) and len(got) == len(base)
    all_ok = all_ok and ok

    rows.append((model_name, out["n_features_used"], len(base),
                 float(diff.max()), float(diff.mean()), ok,
                 int(sum(out["low_bike_alert"])), int(sum(out["high_occupancy_alert"])),
                 out["alert_thresholds"]))
    say(f"{model_name}")
    say(f"   routed via day_type={day_type} horizon={horizon} -> {out['model_name']} "
        f"(job {out['training_job_name']})")
    say(f"   features used     : {out['n_features_used']}")
    say(f"   rows scored       : {len(got):,}")
    say(f"   max |diff|        : {diff.max():.20f}")
    say(f"   mean |diff|       : {diff.mean():.20f}")
    say(f"   identical         : {ok}")
    say(f"   alert thresholds  : {out['alert_thresholds']}")
    say(f"   low alerts fired  : {sum(out['low_bike_alert']):,} / {len(got):,}")
    say(f"   high alerts fired : {sum(out['high_occupancy_alert']):,} / {len(got):,}")
    say("")

# ---- negative controls: routing + feature guards -----------------------
say("=== guard checks ===")
try:
    d = pd.read_csv(os.path.join(CH, "weekend_30m_bike_ratio-1789266742",
                                 "validation.csv"), nrows=5)
    fcw = json.load(open(os.path.join(SRC, "weekend_30m_bike_ratio",
                                      "feature_columns.json"),
                         encoding="utf-8"))["feature_columns"]
    # weekend rows (13 features, no is_peak) sent to a WEEKDAY route (needs 14)
    p = json.dumps({"day_type": "weekday", "horizon_minutes": 30,
                    "instances": d[fcw].to_dict(orient="records")})
    inf.predict_fn(inf.input_fn(p, "application/json"), bundle)
    say("!! FAIL: weekend rows accepted by weekday route (should have raised)")
    all_ok = False
except ValueError as e:
    say(f"OK  weekday route rejects 13-feature weekend rows: {str(e)[:90]}...")
for bad in ({"day_type": "holiday", "horizon_minutes": 30, "features": {}},
            {"day_type": "weekday", "horizon_minutes": 45, "features": {}}):
    try:
        inf.input_fn(json.dumps(bad), "application/json")
        say(f"!! FAIL: accepted invalid route {bad}")
        all_ok = False
    except inf.ModelRoutingError as e:
        say(f"OK  rejected invalid route: {str(e)[:70]}...")

say("")
say("=" * 68)
say(f"{'model':<24}{'nf':>4}{'rows':>8}{'max|diff|':>14}  identical")
for r in rows:
    say(f"{r[0]:<24}{r[1]:>4}{r[2]:>8}{r[3]:>14.1e}  {r[5]}")
say("=" * 68)
say(f"PARITY RESULT: {'PASS - all four models bit-identical to originals' if all_ok else 'FAIL'}")
shutil.rmtree(work, ignore_errors=True)
open(os.path.join(ROOT, "deploy_v1", "_parity.txt"), "w",
     encoding="utf-8").write("\n".join(L))
