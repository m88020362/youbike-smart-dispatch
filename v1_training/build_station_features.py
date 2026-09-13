# -*- coding: utf-8 -*-
"""Build a small static per-station feature lookup from the cached enriched CSV.

Backend data access only. Produces a compact artifact so nothing at runtime ever
needs to open the 1.1 GB enriched file.

Static per station: total_docks, lon, lat and the four nearest-POI distances.
Taken as the most common (mode) value per station, which is robust to occasional
capacity edits mid-month.
"""
from __future__ import annotations
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v1_training"))
from v1_core import columns as C, dataio, prep   # noqa: E402

RAW = os.path.join(ROOT, "v1_training", "_enriched_raw", "weekday_enriched.csv")
OUTDIR = os.path.join(ROOT, "v1_training", "runtime")
STATIC = [C.TOTAL_DOCKS, C.LON, C.LAT, C.DIST_JUNIOR_HIGH,
          C.DIST_UNIVERSITY, C.DIST_MRT, C.DIST_BUS]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    audit = dataio.audit_csv(RAW)
    mp = audit["column_mapping"]
    cols = [mp[k] for k in ([C.STATION, C.DISTRICT] + STATIC) if k in mp]
    print("reading columns:", cols, flush=True)

    frames = []
    for chunk in pd.read_csv(RAW, encoding=audit["encoding"],
                             usecols=cols, chunksize=400_000):
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    # Rename only the columns we actually read; normalize_frame expects the
    # full observation schema (bikes/docks/timestamp) which is not needed here.
    inv = {real: logical for logical, real in mp.items() if real in df.columns}
    df = df.rename(columns=inv)
    df[C.STATION] = df[C.STATION].astype(str)
    for c in STATIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print("rows:", len(df), flush=True)

    agg = {}
    for c in STATIC:
        if c in df.columns:
            agg[c] = (c, lambda s: s.dropna().mode().iloc[0]
                      if not s.dropna().empty else float("nan"))
    g = df.groupby(C.STATION, sort=True).agg(**{k: v for k, v in agg.items()})
    if C.DISTRICT in df.columns:
        g["district"] = df.groupby(C.STATION)[C.DISTRICT].agg(
            lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else "")
    g = g.reset_index().rename(columns={C.STATION: "station"})

    before = len(g)
    g = g.dropna(subset=[C.TOTAL_DOCKS, C.LON, C.LAT])
    g = g[g[C.TOTAL_DOCKS] > 0]
    print(f"stations: {before} -> {len(g)} after requiring capacity+coords", flush=True)

    pq = os.path.join(OUTDIR, "station_features.parquet")
    g.to_parquet(pq, index=False)
    g.to_csv(os.path.join(OUTDIR, "station_features.csv"), index=False,
             encoding="utf-8-sig")
    meta = {
        "source_object": "s3://ubike-data-final/平日_含最近距離_含天氣.csv",
        "built_from_local_cache": RAW,
        "station_count": int(len(g)),
        "static_columns": [c for c in STATIC if c in g.columns],
        "aggregation": "per-station mode (most frequent value)",
        "note": ("static features only; dynamic features (bikes, docks, hour, "
                 "weekday, rainfall, is_peak) come from the live snapshot"),
    }
    with open(os.path.join(OUTDIR, "station_features_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"wrote {pq} ({os.path.getsize(pq):,} B) stations={len(g)}", flush=True)
    print(g.head(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
