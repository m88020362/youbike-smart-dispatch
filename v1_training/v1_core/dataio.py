# -*- coding: utf-8 -*-
"""Streaming CSV audit + range-filtered loading.

Never reads a whole multi-GB file into RAM. Two passes:
  1. audit_csv()   -- header, encoding, sample rows, min/max timestamp, month
                      histogram, station count estimate (chunked).
  2. load_range()  -- second chunked pass keeping only rows inside the chosen
                      month window.

Works on a local path or any file-like object (so an S3 streaming body can be
passed straight through without downloading to disk first).
"""

from __future__ import annotations

import io
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import columns as C
from . import prep

CANDIDATE_ENCODINGS = ("utf-8-sig", "cp950", "utf-8", "big5")
DEFAULT_CHUNKSIZE = 400_000


def sniff_encoding_and_header(path: str) -> Tuple[str, List[str], List[str]]:
    """Detect a workable encoding and read the header + 2 sample lines."""
    last_err = None
    for enc in CANDIDATE_ENCODINGS:
        try:
            with io.open(path, encoding=enc) as fh:
                header_line = fh.readline()
                samples = [fh.readline().rstrip("\r\n") for _ in range(2)]
            header = next(pd.read_csv(io.StringIO(header_line), nrows=0)).__class__ if False else \
                list(pd.read_csv(io.StringIO(header_line), nrows=0).columns)
            if len(header) >= 5:
                return enc, header, samples
        except Exception as e:  # try the next encoding
            last_err = e
            continue
    raise RuntimeError(f"Could not decode {path} with {CANDIDATE_ENCODINGS}: {last_err}")


def audit_csv(
    path: str,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_chunks: Optional[int] = None,
) -> Dict:
    """Low-cost streaming audit. Returns a JSON-serializable report."""
    encoding, header, samples = sniff_encoding_and_header(path)
    mapping, how, unmapped = C.map_columns(header)
    C.require_fields(mapping, header)

    ts_col = mapping[C.TIMESTAMP]
    station_col = mapping[C.STATION]

    month_counts: Counter = Counter()
    stations: set = set()
    n_rows = 0
    ts_min = None
    ts_max = None
    monotonic_violations = 0
    prev_last_ts = None

    reader = pd.read_csv(
        path, encoding=encoding, chunksize=chunksize,
        usecols=[ts_col, station_col], dtype={station_col: "string"},
    )
    for i, chunk in enumerate(reader):
        if max_chunks is not None and i >= max_chunks:
            break
        ts = prep.parse_timestamps(chunk[ts_col])
        ts = ts.dropna()
        if len(ts) == 0:
            continue
        n_rows += len(chunk)
        cmin, cmax = ts.min(), ts.max()
        ts_min = cmin if ts_min is None else min(ts_min, cmin)
        ts_max = cmax if ts_max is None else max(ts_max, cmax)
        month_counts.update(ts.dt.to_period("M").astype(str).tolist())
        if len(stations) < 20000:
            stations.update(chunk[station_col].dropna().astype(str).unique().tolist())
        if prev_last_ts is not None and cmin < prev_last_ts:
            monotonic_violations += 1
        prev_last_ts = cmax

    complete = complete_months(month_counts, ts_min, ts_max)
    return {
        "path": path,
        "encoding": encoding,
        "header": header,
        "sample_rows": samples,
        "column_mapping": mapping,
        "column_mapping_method": how,
        "unmapped_columns": unmapped,
        "rows_scanned": n_rows,
        "first_timestamp": None if ts_min is None else str(ts_min),
        "last_timestamp": None if ts_max is None else str(ts_max),
        "month_row_counts": dict(sorted(month_counts.items())),
        "complete_months": complete,
        "station_count_estimate": len(stations),
        "roughly_time_sorted": monotonic_violations == 0,
        "chunk_order_violations": monotonic_violations,
        "has_is_peak": C.IS_PEAK in mapping,
        "has_rainfall": C.RAINFALL in mapping,
        "has_temperature": C.TEMPERATURE in mapping,
        "has_nearest_junior_high": C.DIST_JUNIOR_HIGH in mapping,
        "has_nearest_university": C.DIST_UNIVERSITY in mapping,
        "has_nearest_mrt": C.DIST_MRT in mapping,
        "has_nearest_bus": C.DIST_BUS in mapping,
    }


def complete_months(month_counts: Counter, ts_min, ts_max) -> List[str]:
    """Months that are not truncated by the dataset's own boundaries.

    A month is treated as complete when the data starts on/before its first day
    and ends on/after its last day. Conservative: partial edge months are
    excluded so we never train on a half month by accident.
    """
    if not month_counts or ts_min is None or ts_max is None:
        return []
    # Tolerance: these feeds are 30-minute granularity, so a complete month
    # ends at 23:30 on the last day, not 23:59:59. Requiring exact coverage to
    # period.end_time would reject every month. One day of slack on each edge
    # keeps genuinely truncated edge months out while accepting real ones.
    tol = pd.Timedelta(days=1)
    out = []
    for m in sorted(month_counts):
        period = pd.Period(m, freq="M")
        starts_early_enough = ts_min <= period.start_time + tol
        ends_late_enough = ts_max >= period.end_time - tol
        if starts_early_enough and ends_late_enough:
            out.append(m)
    return out


def pick_months(complete: List[str], want: int = 2) -> List[str]:
    """Latest `want` complete months (or fewer if that is all there is)."""
    if not complete:
        return []
    return complete[-want:]


def load_range(
    path: str,
    encoding: str,
    mapping: Dict[str, str],
    months: List[str],
    chunksize: int = DEFAULT_CHUNKSIZE,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Second pass: keep only rows whose timestamp falls in `months`."""
    periods = [pd.Period(m, freq="M") for m in months]
    lo = min(p.start_time for p in periods)
    hi = max(p.end_time for p in periods)

    keep_logical = [
        C.TIMESTAMP, C.STATION, C.TOTAL_DOCKS, C.AVAILABLE_BIKES,
        C.AVAILABLE_DOCKS, C.LON, C.LAT, C.DISTRICT, C.IS_PEAK, C.RAINFALL,
        C.DIST_JUNIOR_HIGH, C.DIST_UNIVERSITY, C.DIST_MRT, C.DIST_BUS,
    ]
    cols = usecols or [mapping[k] for k in keep_logical if k in mapping]
    ts_col = mapping[C.TIMESTAMP]

    frames = []
    reader = pd.read_csv(path, encoding=encoding, chunksize=chunksize, usecols=cols)
    for chunk in reader:
        ts = prep.parse_timestamps(chunk[ts_col])
        mask = ts.notna() & (ts >= lo) & (ts <= hi)
        if mask.any():
            frames.append(chunk.loc[mask])
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)
