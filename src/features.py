# -*- coding: utf-8 -*-
"""Target construction (Task 1, R2) for the YouBike predictive dispatch MVP.

build_targets(): for each station, use the next observation as the future state,
keep only pairs spaced ~25-35 minutes apart (excluding ~60-minute gaps), and
label 30-minute shortage / full risk.

build_features(): from the target-bearing DataFrame, construct an interpretable,
leakage-free feature matrix (time / station / inventory / lag features) plus the
shortage / full labels and the retained timestamp for later time-based splitting.
Only current-and-past information is used, so no future leakage is introduced.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from . import config


class FeatureBundle(NamedTuple):
    """Return contract for build_features().

    Fields:
        X: Numeric/encoded feature DataFrame (columns == config.FEATURE_COLUMNS,
            in that fixed order). Contains no future information and no NaN/inf.
        y_shortage: Series (0/1) aligned to X, the 30-minute shortage label.
        y_full: Series (0/1) aligned to X, the 30-minute full/no-return label.
        timestamp: Series of the current-observation timestamps, aligned to X,
            retained for downstream time-based train/test splitting.
        station_encoding: dict mapping station name -> integer id used in X.
        district_encoding: dict mapping district name -> integer id used in X.

    The two encoding dicts are deterministic (built from sorted unique values)
    so downstream train.py / predict.py can reproduce the exact same mapping,
    and are also serializable for persistence in feature_meta.
    """

    X: pd.DataFrame
    y_shortage: pd.Series
    y_full: pd.Series
    timestamp: pd.Series
    station_encoding: dict
    district_encoding: dict


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build 30-minute shortage/full targets from cleaned snapshot data.

    For each station (sorted by timestamp), the "next" observation is taken as
    the future state via shift(-1). Only rows whose gap to that next observation
    falls within [TARGET_MIN_MINUTES, TARGET_MAX_MINUTES] are kept as valid
    30-minute target samples. Rows whose next observation is ~60 minutes away
    (a gap) are dropped.

    Args:
        df: Cleaned DataFrame from data_loader.load_clean(), with internal
            English column names, sorted by (station, timestamp).

    Returns:
        DataFrame of valid target rows with added columns:
            future_timestamp, future_available_bikes, future_available_docks,
            delta_min, shortage_target (int 0/1), full_target (int 0/1).
    """
    if df.empty:
        # Return an empty frame with the expected target columns present.
        out = df.copy()
        for col in (
            config.COL_FUTURE_TIMESTAMP,
            config.COL_FUTURE_BIKES,
            config.COL_FUTURE_DOCKS,
            config.COL_DELTA_MIN,
            config.COL_SHORTAGE_TARGET,
            config.COL_FULL_TARGET,
        ):
            out[col] = pd.Series(dtype="float64")
        return out

    # Ensure correct ordering before shifting within each station.
    work = df.sort_values(
        [config.COL_STATION, config.COL_TIMESTAMP]
    ).reset_index(drop=True)

    grp = work.groupby(config.COL_STATION, sort=False)

    # Next observation = future state for this station.
    work[config.COL_FUTURE_TIMESTAMP] = grp[config.COL_TIMESTAMP].shift(-1)
    work[config.COL_FUTURE_BIKES] = grp[config.COL_AVAILABLE_BIKES].shift(-1)
    work[config.COL_FUTURE_DOCKS] = grp[config.COL_AVAILABLE_DOCKS].shift(-1)

    # Gap in minutes between current and future observation.
    work[config.COL_DELTA_MIN] = (
        work[config.COL_FUTURE_TIMESTAMP] - work[config.COL_TIMESTAMP]
    ).dt.total_seconds() / 60.0

    # Keep only pairs within the 30-minute window; excludes last-per-station
    # rows (NaN delta) and ~60-minute gaps.
    mask = (
        work[config.COL_DELTA_MIN] >= config.TARGET_MIN_MINUTES
    ) & (
        work[config.COL_DELTA_MIN] <= config.TARGET_MAX_MINUTES
    )
    out = work[mask].copy()

    # Labels based on the future state.
    out[config.COL_SHORTAGE_TARGET] = (
        out[config.COL_FUTURE_BIKES] <= config.SHORTAGE_THRESHOLD
    ).astype(int)
    out[config.COL_FULL_TARGET] = (
        out[config.COL_FUTURE_DOCKS] <= config.FULL_THRESHOLD
    ).astype(int)

    out[config.COL_FUTURE_BIKES] = out[config.COL_FUTURE_BIKES].astype(int)
    out[config.COL_FUTURE_DOCKS] = out[config.COL_FUTURE_DOCKS].astype(int)

    return out.reset_index(drop=True)


def _deterministic_encoding(values: pd.Series) -> dict:
    """Build a deterministic label encoding: sorted unique value -> int id.

    Using sorted unique values (rather than order-of-appearance) makes the
    mapping reproducible regardless of row ordering, so downstream modules can
    reconstruct the exact same integer codes from the persisted mapping.

    Missing values (NaN) are coerced to the literal string "nan" via a
    per-element ``str`` conversion, so a column containing NaNs (e.g. a small
    number of rows with an unknown district) still yields an all-string,
    sortable key set instead of a mixed str/float list that ``sorted`` rejects.
    """
    uniques = sorted({str(v) for v in values.tolist()})
    return {name: i for i, name in enumerate(uniques)}


def build_features(df: pd.DataFrame) -> FeatureBundle:
    """Build an interpretable, leakage-free feature matrix from target rows.

    Input is the DataFrame produced by build_targets(): cleaned columns plus
    future_timestamp / future_available_bikes / future_available_docks /
    delta_min / shortage_target / full_target. Only current-and-past information
    is used to construct features; the future_* columns are used ONLY to read
    off the labels, never as features (no leakage).

    Features produced (see design.md §4, requirements R3 criteria 1-5, 7):
        Time (from `timestamp`): hour, weekday, is_weekend.
        Station/static: station_id (deterministic label-encode), district_id
            (deterministic label-encode), total_docks, lon, lat.
        Inventory: available_bikes, available_docks, bike_ratio, dock_ratio.
            Ratios use total_docks as denominator; when total_docks == 0 the
            ratio is set to 0.0 (safe default, never inf/NaN).
        Lag (same station, shift(1) on time-sorted data): prev_available_bikes,
            bike_change = available_bikes - prev_available_bikes,
            dock_change = available_docks - prev_available_docks. The first row
            per station has no predecessor; those lag NaNs are filled with
            config.LAG_FILL_VALUE (default 0), so prev_available_bikes = 0 and
            bike_change / dock_change = 0 on a station's first observation.

    All per-station operations run on data sorted by (station, timestamp) so the
    shift(1) lag only ever references the previous (past) observation.

    Args:
        df: Target-bearing DataFrame from build_targets().

    Returns:
        FeatureBundle(X, y_shortage, y_full, timestamp, station_encoding,
        district_encoding). See the FeatureBundle docstring for the contract.
    """
    if df.empty:
        empty_X = pd.DataFrame(columns=config.FEATURE_COLUMNS)
        empty_y = pd.Series(dtype="int64")
        empty_ts = pd.Series(dtype="datetime64[ns]")
        return FeatureBundle(
            X=empty_X,
            y_shortage=empty_y.copy(),
            y_full=empty_y.copy(),
            timestamp=empty_ts,
            station_encoding={},
            district_encoding={},
        )

    # Sort by (station, timestamp) so shift(1) references the past observation.
    work = df.sort_values(
        [config.COL_STATION, config.COL_TIMESTAMP]
    ).reset_index(drop=True)

    ts = work[config.COL_TIMESTAMP]

    # --- Time features (current row only) -----------------------------------
    hour = ts.dt.hour.astype(int)
    weekday = ts.dt.weekday.astype(int)          # Monday=0 .. Sunday=6
    is_weekend = (weekday >= 5).astype(int)      # Sat/Sun -> 1

    # --- Station / district deterministic label encoding --------------------
    station_encoding = _deterministic_encoding(work[config.COL_STATION])
    district_encoding = _deterministic_encoding(work[config.COL_DISTRICT])
    # Coerce per element with str() (matching _deterministic_encoding keys) so
    # missing values map to the literal "nan" key rather than propagating NA;
    # this is robust across numpy-object and Arrow-backed string dtypes.
    station_key = work[config.COL_STATION].map(lambda v: str(v))
    district_key = work[config.COL_DISTRICT].map(lambda v: str(v))
    station_id = station_key.map(station_encoding).astype(int)
    district_id = district_key.map(district_encoding).astype(int)

    # --- Inventory + safe ratios (divide-by-zero -> 0.0) --------------------
    total_docks = work[config.COL_TOTAL_DOCKS].astype(float)
    available_bikes = work[config.COL_AVAILABLE_BIKES].astype(float)
    available_docks = work[config.COL_AVAILABLE_DOCKS].astype(float)

    safe_total = total_docks.where(total_docks > 0, other=pd.NA)
    bike_ratio = (available_bikes / safe_total).fillna(0.0).astype(float)
    dock_ratio = (available_docks / safe_total).fillna(0.0).astype(float)

    # --- Lag features (same station, shift(1)) ------------------------------
    grp = work.groupby(config.COL_STATION, sort=False)
    prev_bikes = grp[config.COL_AVAILABLE_BIKES].shift(1)
    prev_docks = grp[config.COL_AVAILABLE_DOCKS].shift(1)

    bike_change = work[config.COL_AVAILABLE_BIKES] - prev_bikes
    dock_change = work[config.COL_AVAILABLE_DOCKS] - prev_docks

    # First row per station has NaN lag -> documented default fill.
    prev_bikes = prev_bikes.fillna(config.LAG_FILL_VALUE).astype(int)
    bike_change = bike_change.fillna(config.LAG_FILL_VALUE).astype(int)
    dock_change = dock_change.fillna(config.LAG_FILL_VALUE).astype(int)

    # --- Assemble X in the fixed, documented column order -------------------
    X = pd.DataFrame(
        {
            config.COL_HOUR: hour,
            config.COL_WEEKDAY: weekday,
            config.COL_IS_WEEKEND: is_weekend,
            config.COL_STATION_ID: station_id,
            config.COL_DISTRICT_ID: district_id,
            config.COL_TOTAL_DOCKS: work[config.COL_TOTAL_DOCKS].astype(int),
            # lon/lat are not in CRITICAL_COLUMNS, so a few rows may carry NaN
            # coordinates. Fill with 0.0 to keep X NaN-free per the FeatureBundle
            # contract (these rows are a tiny fraction and unusable for spatial
            # neighbor logic, which is out of scope for the model features).
            config.COL_LON: work[config.COL_LON].astype(float).fillna(0.0),
            config.COL_LAT: work[config.COL_LAT].astype(float).fillna(0.0),
            config.COL_AVAILABLE_BIKES: work[config.COL_AVAILABLE_BIKES].astype(int),
            config.COL_AVAILABLE_DOCKS: work[config.COL_AVAILABLE_DOCKS].astype(int),
            config.COL_BIKE_RATIO: bike_ratio,
            config.COL_DOCK_RATIO: dock_ratio,
            config.COL_PREV_BIKES: prev_bikes,
            config.COL_BIKE_CHANGE: bike_change,
            config.COL_DOCK_CHANGE: dock_change,
        }
    )[config.FEATURE_COLUMNS].reset_index(drop=True)

    y_shortage = work[config.COL_SHORTAGE_TARGET].astype(int).reset_index(drop=True)
    y_full = work[config.COL_FULL_TARGET].astype(int).reset_index(drop=True)
    timestamp = ts.reset_index(drop=True)

    return FeatureBundle(
        X=X,
        y_shortage=y_shortage,
        y_full=y_full,
        timestamp=timestamp,
        station_encoding=station_encoding,
        district_encoding=district_encoding,
    )
