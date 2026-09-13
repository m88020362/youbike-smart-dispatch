# -*- coding: utf-8 -*-
"""V1 preprocessing: target construction, features, chronological split.

Target (regression):
    future_bike_ratio = future_available_bikes / future_total_docks

built strictly WITHIN a station, using the FUTURE observation's own total_docks.
The future observation is located with a tolerance window around the horizon:
    30-minute model -> delta in [25, 35]
    60-minute model -> delta in [55, 65]
Implemented with pandas.merge_asof(direction="nearest", tolerance=5min, by=station),
which picks the observation closest to the horizon and rejects anything outside
the window. Rows without a valid future observation are DROPPED, never imputed.

Anomalies (ratio outside [0, 1]) are counted and dropped -- never silently
clipped, so a data problem stays visible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import columns as C

# Feature column names produced here (stable contract for training + metadata).
F_BIKES = "current_available_bikes"
F_DOCKS = "current_available_docks"
F_TOTAL = "total_docks"
F_RATIO = "current_bike_ratio"
F_HOUR = "hour"
F_WEEKDAY = "weekday"
F_IS_PEAK = "is_peak"
F_LON = "lon"
F_LAT = "lat"
F_D_JH = "nearest_junior_high_distance"
F_D_UNI = "nearest_university_distance"
F_D_MRT = "nearest_mrt_distance"
F_D_BUS = "nearest_bus_distance"
F_RAINFALL = "rainfall"

TARGET = "future_bike_ratio"

# Always excluded, with the reason recorded in metadata.
EXCLUSIONS = {
    "temperature": "not confirmed available at prediction timestamp -> temporal leakage risk",
    "period": "coarse duplicate of hour; excluded per mission brief (時段)",
    "source": "provenance label, not predictive (資料來源)",
    "station_raw_string": "non-numeric identity not fed to the numeric trainer",
    "timestamp_raw": "split into hour / weekday instead of numeric timestamp",
}


def parse_timestamps(series: pd.Series) -> pd.Series:
    """Parse mixed timestamp spellings seen across the datasets.

    Observed formats include '2026/01/11 18:00' and '2026-05-01 23:30:41'.
    Parsed with format='mixed' so both work; unparseable values become NaT and
    are dropped by the caller.
    """
    return pd.to_datetime(series, format="mixed", errors="coerce")


def normalize_frame(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename mapped columns to logical names and coerce dtypes."""
    inv = {real: logical for logical, real in mapping.items()}
    out = df.rename(columns=inv)

    out[C.TIMESTAMP] = parse_timestamps(out[C.TIMESTAMP])

    for col in (C.TOTAL_DOCKS, C.AVAILABLE_BIKES, C.AVAILABLE_DOCKS):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in (C.LON, C.LAT, C.RAINFALL, C.IS_PEAK,
                C.DIST_JUNIOR_HIGH, C.DIST_UNIVERSITY, C.DIST_MRT, C.DIST_BUS):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out[C.STATION] = out[C.STATION].astype(str)
    return out


def drop_invalid_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop rows unusable for target construction. Returns (df, drop_counts)."""
    counts = {}
    n0 = len(df)

    df = df[df[C.TIMESTAMP].notna()]
    counts["dropped_bad_timestamp"] = n0 - len(df)

    n1 = len(df)
    df = df[df[C.STATION].notna() & (df[C.STATION] != "") & (df[C.STATION] != "nan")]
    counts["dropped_missing_station"] = n1 - len(df)

    n2 = len(df)
    df = df[
        df[C.AVAILABLE_BIKES].notna()
        & df[C.AVAILABLE_DOCKS].notna()
        & df[C.TOTAL_DOCKS].notna()
        & (df[C.TOTAL_DOCKS] > 0)
    ]
    counts["dropped_missing_or_zero_capacity"] = n2 - len(df)

    return df, counts


def build_target(
    df: pd.DataFrame,
    horizon_minutes: int,
    tolerance_minutes: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach future_bike_ratio using a per-station forward window.

    horizon 30 + tolerance 5 => accepted delta in [25, 35].
    horizon 60 + tolerance 5 => accepted delta in [55, 65].
    """
    stats: Dict[str, int] = {}

    df = df.sort_values([C.STATION, C.TIMESTAMP], kind="mergesort").reset_index(drop=True)

    left = df.copy()
    left["_probe"] = left[C.TIMESTAMP] + pd.Timedelta(minutes=horizon_minutes)

    right = df[[C.STATION, C.TIMESTAMP, C.AVAILABLE_BIKES, C.TOTAL_DOCKS]].copy()
    right = right.rename(
        columns={
            C.TIMESTAMP: "future_timestamp",
            C.AVAILABLE_BIKES: "future_available_bikes",
            C.TOTAL_DOCKS: "future_total_docks",
        }
    )

    left = left.sort_values("_probe", kind="mergesort")
    right = right.sort_values("future_timestamp", kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on="_probe",
        right_on="future_timestamp",
        by=C.STATION,
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )

    n_all = len(merged)
    merged = merged[merged["future_timestamp"].notna()]
    stats["dropped_no_future_observation"] = n_all - len(merged)

    delta = (merged["future_timestamp"] - merged[C.TIMESTAMP]).dt.total_seconds() / 60.0
    merged["delta_minutes"] = delta
    lo = horizon_minutes - tolerance_minutes
    hi = horizon_minutes + tolerance_minutes
    n_before = len(merged)
    merged = merged[(delta >= lo) & (delta <= hi)]
    stats["dropped_delta_out_of_window"] = n_before - len(merged)
    stats["target_window_low_minutes"] = lo
    stats["target_window_high_minutes"] = hi

    n_before = len(merged)
    merged = merged[merged["future_total_docks"] > 0]
    stats["dropped_future_capacity_not_positive"] = n_before - len(merged)

    merged[TARGET] = (
        merged["future_available_bikes"] / merged["future_total_docks"]
    )

    n_before = len(merged)
    merged = merged[np.isfinite(merged[TARGET])]
    stats["dropped_target_not_finite"] = n_before - len(merged)

    # Anomalies are reported and dropped, NOT clipped.
    anomaly_mask = (merged[TARGET] < 0.0) | (merged[TARGET] > 1.0)
    stats["target_anomaly_count_out_of_unit_range"] = int(anomaly_mask.sum())
    merged = merged[~anomaly_mask]

    stats["target_rows"] = len(merged)
    return merged.reset_index(drop=True), stats


def build_features(
    df: pd.DataFrame,
    include_rainfall: bool = False,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Build the minimal, leakage-free feature matrix.

    Returns (frame_with_features, feature_columns, excluded_reasons).
    Optional columns are only included when actually present in the data.
    """
    out = df.copy()

    out[F_BIKES] = out[C.AVAILABLE_BIKES].astype("float64")
    out[F_DOCKS] = out[C.AVAILABLE_DOCKS].astype("float64")
    out[F_TOTAL] = out[C.TOTAL_DOCKS].astype("float64")
    out[F_RATIO] = out[F_BIKES] / out[F_TOTAL]
    out[F_HOUR] = out[C.TIMESTAMP].dt.hour.astype("int16")
    out[F_WEEKDAY] = out[C.TIMESTAMP].dt.weekday.astype("int16")

    features = [F_BIKES, F_DOCKS, F_TOTAL, F_RATIO, F_HOUR, F_WEEKDAY]
    excluded = dict(EXCLUSIONS)

    for logical, name in (
        (C.LON, F_LON),
        (C.LAT, F_LAT),
        (C.DIST_JUNIOR_HIGH, F_D_JH),
        (C.DIST_UNIVERSITY, F_D_UNI),
        (C.DIST_MRT, F_D_MRT),
        (C.DIST_BUS, F_D_BUS),
    ):
        if logical in out.columns and out[logical].notna().any():
            out[name] = out[logical].astype("float64")
            features.append(name)

    # is_peak: use only if the dataset already provides it (never re-invented).
    if C.IS_PEAK in out.columns and out[C.IS_PEAK].notna().any():
        vals = set(pd.unique(out[C.IS_PEAK].dropna()))
        if vals.issubset({0, 1, 0.0, 1.0, True, False}):
            out[F_IS_PEAK] = out[C.IS_PEAK].astype("float64")
            features.append(F_IS_PEAK)
        else:
            excluded["is_peak"] = (
                f"present but not 0/1 (found {sorted(map(str, vals))[:6]}); "
                f"peak definition NOT re-invented per mission brief"
            )
    else:
        excluded["is_peak"] = "column absent; peak rules not invented per mission brief"

    if include_rainfall and C.RAINFALL in out.columns and out[C.RAINFALL].notna().any():
        out[F_RAINFALL] = out[C.RAINFALL].astype("float64").fillna(0.0)
        features.append(F_RAINFALL)
    else:
        excluded["rainfall"] = (
            "cannot confirm it is observable at the prediction timestamp; "
            "excluded from the P0 model to avoid temporal leakage"
        )

    # Guard: never leak the future into X.
    for forbidden in ("future_available_bikes", "future_total_docks",
                      "future_timestamp", "delta_minutes", TARGET):
        assert forbidden not in features, f"leakage: {forbidden} in features"

    return out, features, excluded


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Split past -> future by timestamp, never randomly.

    Uses a timestamp CUT so the same timestamp cannot appear on both sides.
    Prefers a month boundary when two or more distinct months are present
    (earlier month(s) -> train, latest month -> validation).
    """
    df = df.sort_values(C.TIMESTAMP, kind="mergesort").reset_index(drop=True)
    months = sorted(df[C.TIMESTAMP].dt.to_period("M").unique())
    info: Dict[str, str] = {}

    if len(months) >= 2:
        cut_period = months[-1]
        cut_ts = cut_period.to_timestamp()
        train = df[df[C.TIMESTAMP] < cut_ts]
        valid = df[df[C.TIMESTAMP] >= cut_ts]
        info["split_strategy"] = "month_boundary"
        info["split_cut_timestamp"] = str(cut_ts)
        info["train_months"] = ",".join(str(m) for m in months[:-1])
        info["validation_months"] = str(cut_period)
        if len(train) > 0 and len(valid) > 0:
            return train.reset_index(drop=True), valid.reset_index(drop=True), info

    # Single month (or degenerate month split): 80/20 by timestamp cut.
    idx = int(len(df) * train_fraction)
    if idx <= 0 or idx >= len(df):
        info["split_strategy"] = "insufficient_rows"
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True), info
    cut_ts = df[C.TIMESTAMP].iloc[idx]
    train = df[df[C.TIMESTAMP] < cut_ts]
    valid = df[df[C.TIMESTAMP] >= cut_ts]
    info["split_strategy"] = "timestamp_quantile_80_20"
    info["split_cut_timestamp"] = str(cut_ts)
    return train.reset_index(drop=True), valid.reset_index(drop=True), info
