# -*- coding: utf-8 -*-
"""Build small runtime demo snapshots from the cached enriched CSVs.

Picks one real timestamp with high station coverage (weekday: is_peak=1) and
keeps every station at that instant, including rainfall / is_peak which the V0
runtime snapshot lacks. Nothing is fabricated.

Outputs:
    v1_training/runtime/demo_weekday_snapshot.parquet
    v1_training/runtime/demo_weekend_snapshot.parquet
"""
from __future__ import annotations
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v1_training"))
from v1_core import columns as C, dataio   # noqa: E402

OUT = os.path.join(ROOT, "v1_training", "runtime")
MONTH = "2026-05"

SPECS = [
    ("weekday", os.path.join(ROOT, "v1_training", "_enriched_raw", "weekday_enriched.csv"),
     "demo_weekday_snapshot.parquet", True),
    ("weekend", os.path.join(ROOT, "v1_training", "_enriched_raw", "weekend_enriched.csv"),
     "demo_weekend_snapshot.parquet", False),
]


def build(day_type, raw, outname, require_peak):
    audit = dataio.audit_csv(raw)
    mp = audit["column_mapping"]
    want = [C.TIMESTAMP, C.CITY, C.DISTRICT, C.STATION, C.TOTAL_DOCKS,
            C.AVAILABLE_BIKES, C.AVAILABLE_DOCKS, C.LON, C.LAT,
            C.DIST_JUNIOR_HIGH, C.DIST_UNIVERSITY, C.DIST_MRT, C.DIST_BUS,
            C.RAINFALL]
    if require_peak and C.IS_PEAK in mp:
        want.append(C.IS_PEAK)
    cols = [mp[k] for k in want if k in mp]
    print(f"[{day_type}] reading {len(cols)} cols", flush=True)

    df = dataio.load_range(raw, audit["encoding"], mp, [MONTH], usecols=cols)
    inv = {real: log for log, real in mp.items() if real in df.columns}
    df = df.rename(columns=inv)
    df[C.TIMESTAMP] = pd.to_datetime(df[C.TIMESTAMP], format="mixed", errors="coerce")
    for c in (C.TOTAL_DOCKS, C.AVAILABLE_BIKES, C.AVAILABLE_DOCKS, C.LON, C.LAT,
              C.RAINFALL, C.DIST_JUNIOR_HIGH, C.DIST_UNIVERSITY, C.DIST_MRT,
              C.DIST_BUS) + ((C.IS_PEAK,) if C.IS_PEAK in df.columns else ()):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[C.TIMESTAMP, C.STATION, C.TOTAL_DOCKS,
                           C.AVAILABLE_BIKES, C.AVAILABLE_DOCKS, C.LON, C.LAT])
    df = df[df[C.TOTAL_DOCKS] > 0]
    df[C.STATION] = df[C.STATION].astype(str)
    print(f"[{day_type}] rows={len(df):,}", flush=True)

    cand = df
    if require_peak and C.IS_PEAK in df.columns:
        cand = df[df[C.IS_PEAK] >= 0.5]
        print(f"[{day_type}] is_peak=1 rows={len(cand):,}", flush=True)

    cov = cand.groupby(C.TIMESTAMP)[C.STATION].nunique().sort_values(ascending=False)
    if cov.empty:
        raise RuntimeError(f"no candidate timestamp for {day_type}")
    best_ts = cov.index[0]
    print(f"[{day_type}] chosen ts={best_ts} stations={cov.iloc[0]}", flush=True)

    snap = cand[cand[C.TIMESTAMP] == best_ts].drop_duplicates(
        subset=[C.STATION], keep="last").copy()
    snap["current_available_bikes"] = snap[C.AVAILABLE_BIKES].astype("float64")
    snap["current_available_docks"] = snap[C.AVAILABLE_DOCKS].astype("float64")
    snap["current_bike_ratio"] = (snap["current_available_bikes"]
                                  / snap[C.TOTAL_DOCKS].astype("float64"))
    snap["hour"] = snap[C.TIMESTAMP].dt.hour.astype("float64")
    snap["weekday"] = snap[C.TIMESTAMP].dt.weekday.astype("float64")
    if C.IS_PEAK not in snap.columns:
        pass  # weekend has no is_peak; never fabricated
    snap = snap.sort_values(C.STATION).reset_index(drop=True)

    path = os.path.join(OUT, outname)
    snap.to_parquet(path, index=False)
    print(f"[{day_type}] wrote {path} ({os.path.getsize(path):,} B) "
          f"stations={len(snap)}", flush=True)
    return {"day_type": day_type, "timestamp": str(best_ts),
            "stations": int(len(snap)), "file": outname,
            "has_is_peak": bool(C.IS_PEAK in snap.columns),
            "rainfall_mean": float(snap[C.RAINFALL].mean()) if C.RAINFALL in snap else None,
            "columns": list(snap.columns)}


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for day_type, raw, outname, peak in SPECS:
        if not os.path.exists(raw):
            print(f"skip {day_type}: {raw} missing", flush=True); continue
        meta[day_type] = build(day_type, raw, outname, peak)
    with open(os.path.join(OUT, "demo_snapshot_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
