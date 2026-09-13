# -*- coding: utf-8 -*-
"""Find real DEMO_CASE_USER_RETURN / DEMO_CASE_USER_BORROW from the snapshot.

Predictions come from the LOCAL bundled boosters, which were verified
bit-identical (max abs diff 0.0) to the deployed endpoint, so any case found
here behaves identically in the live app. Nothing is fabricated.
"""
from __future__ import annotations
import json, os, sys, tarfile, tempfile, shutil, importlib.util
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import v1_ui as ui           # noqa: E402
from src import v1_decision as vd     # noqa: E402
from src import v1_predict as vp      # noqa: E402

RUNTIME = os.path.join(ROOT, "v1_training", "runtime")
TAR = os.path.join(ROOT, "deploy_v1", "artifacts", "youbike-v1-multimodel.tar.gz")

PRESETS = {"SOGO 忠孝館": (25.041, 121.5435)}


def local_predict(rows, day_type, work):
    import lightgbm as lgb
    feats = vp.FEATURE_ORDER[day_type]
    X = pd.DataFrame([{c: float(r[c]) for c in feats} for r in rows])[feats]
    out = []
    for h in (30, 60):
        name = vp.MODEL_NAMES[(day_type, h)]
        b = lgb.Booster(model_file=os.path.join(work, f"{name}_lgbm.txt"))
        out.append(np.clip(b.predict(X.to_numpy()), 0, 1))
    m30, m60 = vp.MODEL_NAMES[(day_type, 30)], vp.MODEL_NAMES[(day_type, 60)]
    return [{"day_type": day_type, "ratio_30m": float(a), "ratio_60m": float(c),
             "model_30m": m30, "model_60m": m60, "source": "LOCAL_BUNDLE_PARITY"}
            for a, c in zip(out[0], out[1])]


def snapshot_rows(day_type):
    fn = f"demo_{day_type}_snapshot.parquet"
    df = pd.read_parquet(os.path.join(RUNTIME, fn))
    if day_type == "weekday" and "is_peak" not in df.columns:
        raise RuntimeError("weekday snapshot lacks is_peak")
    return df


def feature_rows(df, day_type):
    feats = vp.FEATURE_ORDER[day_type]
    return [{c: float(r[c]) for c in feats} for _, r in df.iterrows()]


def find_case(day_type, mode, work, radius=500):
    df = snapshot_rows(day_type)
    preds = local_predict(feature_rows(df, day_type), day_type, work)
    recs = df.to_dict(orient="records")
    for p, r in zip(preds, recs):
        r["_pred"] = p
        r["_state"] = vd.classify_temporal_risk(p["ratio_30m"], p["ratio_60m"])

    want = ((vd.PERSISTENT_LOW, vd.EMERGING_LOW) if mode == ui.MODE_RETURN
            else (vd.PERSISTENT_HIGH, vd.EMERGING_HIGH))
    seeds = [r for r in recs if r["_state"] in want]
    seeds.sort(key=lambda r: 0 if r["_state"] in (vd.PERSISTENT_LOW,
                                                  vd.PERSISTENT_HIGH) else 1)

    # prefer a destination near the preset if it works, else any real cluster
    candidates = []
    for name, (plat, plon) in PRESETS.items():
        candidates.append((name, plat, plon))
    for s in seeds[:400]:
        candidates.append((f"{s['station']} 周邊", float(s["lat"]), float(s["lon"])))

    for dest_name, dlat, dlon in candidates:
        near = ui.nearby_stations(recs, dlat, dlon, radius)
        if len(near) < 2:
            continue
        near_preds = [n["_pred"] for n in near]
        rows = ui.build_user_rows(near, near_preds, mode, max_rows=99)
        rewarded = [r for r in rows if r.get("reward_twd")]
        if rewarded:
            best = max(rewarded, key=lambda r: r["reward_twd"])
            base = next((r for r in rows if r["is_baseline"]), None)
            return {
                "mode": mode, "day_type": day_type,
                "destination": dest_name,
                "destination_lat": dlat, "destination_lon": dlon,
                "radius_m": radius,
                "stations_in_radius": len(near),
                "baseline_station": base["station"] if base else None,
                "baseline_distance_m": round(base["distance_m"], 1) if base else None,
                "reward_station": best["station"],
                "reward_station_state": best["risk_state"],
                "reward_station_distance_m": round(best["distance_m"], 1),
                "extra_distance_m": round(best["extra_distance_m"], 1),
                "reward_twd": best["reward_twd"],
                "reward_station_risk_30m": best["risk_30m"],
                "reward_station_risk_60m": best["risk_60m"],
                "rewarded_station_count": len(rewarded),
                "is_preset_destination": dest_name in PRESETS,
            }
    return None


def main():
    work = tempfile.mkdtemp(prefix="bundle_")
    with tarfile.open(TAR, "r:gz") as t:
        t.extractall(work)
    try:
        meta = json.load(open(os.path.join(RUNTIME, "demo_snapshot_meta.json"),
                              encoding="utf-8"))
        cases = {
            "DEMO_CASE_USER_RETURN": find_case("weekday", ui.MODE_RETURN, work),
            "DEMO_CASE_USER_BORROW": find_case("weekday", ui.MODE_BORROW, work),
        }
        cases["snapshot_meta"] = meta
        cases["note"] = ("predictions for case discovery came from the local "
                         "bundled boosters, verified bit-identical (0.0 diff) to "
                         "the deployed endpoint")
        with open(os.path.join(RUNTIME, "demo_cases.json"), "w",
                  encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(json.dumps(cases, indent=2, ensure_ascii=False)[:2600], flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
