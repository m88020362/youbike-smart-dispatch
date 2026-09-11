# -*- coding: utf-8 -*-
"""Pytest tests for src/features.py (R2 target + R3 features).

Locks in the leakage-free 30-minute target construction and the interpretable
feature matrix (design §3, §4, §12; requirements R2, R3). All cases use small
synthetic in-memory DataFrames so they are deterministic and need no dataset.

Thresholds/intervals come from src/config.py, never hard-coded here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, features


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _clean_df(rows: list[dict]) -> pd.DataFrame:
    """Build a cleaned-shape DataFrame (internal English columns, sorted)."""
    df = pd.DataFrame(rows)
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP])
    return df.sort_values(
        [config.COL_STATION, config.COL_TIMESTAMP]
    ).reset_index(drop=True)


def _row(ts, station, bikes, docks, *, total=20, district="板橋區",
         lon=121.46, lat=25.01, city="新北市") -> dict:
    return {
        config.COL_TIMESTAMP: ts,
        config.COL_CITY: city,
        config.COL_DISTRICT: district,
        config.COL_STATION: station,
        config.COL_TOTAL_DOCKS: total,
        config.COL_AVAILABLE_BIKES: bikes,
        config.COL_AVAILABLE_DOCKS: docks,
        config.COL_LON: lon,
        config.COL_LAT: lat,
    }


# =========================================================================== #
# R2 -- 30-minute target construction                                         #
# =========================================================================== #
def test_60min_gap_not_treated_as_target():
    """A ~60-minute gap produces no label (R2.3)."""
    df = _clean_df([
        _row("2026-03-29 08:00:00", "S", 10, 10),  # next @ 09:00 -> 60 min GAP
        _row("2026-03-29 09:00:00", "S", 8, 12),   # last row -> no next
    ])

    tgt = features.build_targets(df)

    assert len(tgt) == 0


def test_25_to_35_minute_pairs_retained():
    """Pairs spaced within [TARGET_MIN, TARGET_MAX] are kept; others dropped."""
    df = _clean_df([
        _row("2026-03-29 08:00:00", "S", 10, 10),  # +25 min -> kept
        _row("2026-03-29 08:25:00", "S", 9, 11),   # +35 min -> kept
        _row("2026-03-29 09:00:00", "S", 8, 12),   # +60 min -> GAP, dropped
        _row("2026-03-29 10:00:00", "S", 7, 13),   # last row -> dropped
    ])

    tgt = features.build_targets(df)

    # Exactly the two in-window pairs survive.
    assert len(tgt) == 2
    deltas = sorted(round(d) for d in tgt[config.COL_DELTA_MIN])
    assert deltas == [25, 35]
    # Every retained delta obeys the configured window bounds.
    assert tgt[config.COL_DELTA_MIN].min() >= config.TARGET_MIN_MINUTES
    assert tgt[config.COL_DELTA_MIN].max() <= config.TARGET_MAX_MINUTES


def test_boundary_intervals_just_outside_window_excluded():
    """24-min (below min) and 36-min (above max) pairs are excluded."""
    df = _clean_df([
        _row("2026-03-29 08:00:00", "S", 10, 10),  # +24 min -> excluded
        _row("2026-03-29 08:24:00", "S", 9, 11),   # +36 min -> excluded
        _row("2026-03-29 09:00:00", "S", 8, 12),   # last -> excluded
    ])

    tgt = features.build_targets(df)

    assert len(tgt) == 0


def test_shortage_and_full_label_values_correct():
    """Labels use the FUTURE observation vs. config thresholds (R2.4, R2.5)."""
    st = config.SHORTAGE_THRESHOLD
    ft = config.FULL_THRESHOLD
    df = _clean_df([
        # pair 0: future bikes = st (<=) -> shortage 1 ; future docks = ft+5 -> full 0
        _row("2026-03-29 08:00:00", "S", 10, 10),
        _row("2026-03-29 08:30:00", "S", st, ft + 5),
        # pair 1: future bikes = st+5 -> shortage 0 ; future docks = ft (<=) -> full 1
        _row("2026-03-29 09:00:00", "S", st + 5, ft),
    ])

    tgt = features.build_targets(df).sort_values(
        config.COL_TIMESTAMP).reset_index(drop=True)

    assert len(tgt) == 2
    assert int(tgt[config.COL_SHORTAGE_TARGET].iloc[0]) == 1
    assert int(tgt[config.COL_FULL_TARGET].iloc[0]) == 0
    assert int(tgt[config.COL_SHORTAGE_TARGET].iloc[1]) == 0
    assert int(tgt[config.COL_FULL_TARGET].iloc[1]) == 1
    # General invariant: label == (future value <= threshold).
    exp_short = (tgt[config.COL_FUTURE_BIKES] <= st).astype(int)
    exp_full = (tgt[config.COL_FUTURE_DOCKS] <= ft).astype(int)
    assert exp_short.equals(tgt[config.COL_SHORTAGE_TARGET])
    assert exp_full.equals(tgt[config.COL_FULL_TARGET])


def test_target_uses_next_observation_per_station_independently():
    """Future state is the same station's next row, not another station's."""
    df = _clean_df([
        _row("2026-03-29 08:00:00", "A", 10, 10),
        _row("2026-03-29 08:30:00", "A", 3, 17),
        _row("2026-03-29 08:00:00", "B", 5, 15),
        _row("2026-03-29 08:30:00", "B", 6, 14),
    ])

    tgt = features.build_targets(df)

    a = tgt[tgt[config.COL_STATION] == "A"].iloc[0]
    b = tgt[tgt[config.COL_STATION] == "B"].iloc[0]
    assert int(a[config.COL_FUTURE_BIKES]) == 3   # A's own next row
    assert int(b[config.COL_FUTURE_BIKES]) == 6   # B's own next row


# =========================================================================== #
# R3 -- feature matrix                                                        #
# =========================================================================== #
def _two_pair_frame():
    """A station with three 30-min-spaced rows -> two target pairs."""
    return _clean_df([
        _row("2026-03-29 08:00:00", "S", 10, 10, total=20),
        _row("2026-03-29 08:30:00", "S", 6, 14, total=20),
        _row("2026-03-29 09:00:00", "S", 3, 17, total=20),
    ])


def test_feature_columns_exact_order_and_no_nan_inf():
    df = _two_pair_frame()
    bundle = features.build_features(features.build_targets(df))

    assert list(bundle.X.columns) == config.FEATURE_COLUMNS
    assert not bundle.X.isna().any().any()
    assert np.isfinite(bundle.X.to_numpy(dtype=float)).all()
    # Labels align 1:1 with X rows.
    assert len(bundle.y_shortage) == len(bundle.X)
    assert len(bundle.y_full) == len(bundle.X)
    assert len(bundle.timestamp) == len(bundle.X)


def test_no_future_information_leakage_in_features():
    """No feature column equals or is derived from the future_* columns."""
    df = _two_pair_frame()
    tgt = features.build_targets(df)
    bundle = features.build_features(tgt)

    # The future_* columns must NOT leak into the feature matrix.
    leaky = {
        config.COL_FUTURE_TIMESTAMP,
        config.COL_FUTURE_BIKES,
        config.COL_FUTURE_DOCKS,
        config.COL_DELTA_MIN,
        config.COL_SHORTAGE_TARGET,
        config.COL_FULL_TARGET,
    }
    assert leaky.isdisjoint(set(bundle.X.columns))

    # Concretely: current available_bikes in X must equal the current (not
    # future) values from the target rows, row-for-row.
    tgt_sorted = tgt.sort_values(
        [config.COL_STATION, config.COL_TIMESTAMP]).reset_index(drop=True)
    assert list(bundle.X[config.COL_AVAILABLE_BIKES]) == list(
        tgt_sorted[config.COL_AVAILABLE_BIKES])


def test_ratio_safe_when_total_docks_zero():
    """bike_ratio / dock_ratio are 0.0 (never inf/NaN) when total_docks == 0."""
    df = _clean_df([
        _row("2026-03-29 08:00:00", "Z", 0, 0, total=0),
        _row("2026-03-29 08:30:00", "Z", 0, 0, total=0),
        _row("2026-03-29 09:00:00", "Z", 0, 0, total=0),
    ])

    bundle = features.build_features(features.build_targets(df))

    assert (bundle.X[config.COL_BIKE_RATIO] == 0.0).all()
    assert (bundle.X[config.COL_DOCK_RATIO] == 0.0).all()
    assert np.isfinite(bundle.X[config.COL_BIKE_RATIO].to_numpy()).all()
    assert np.isfinite(bundle.X[config.COL_DOCK_RATIO].to_numpy()).all()


def test_lag_nan_filled_on_first_row_per_station():
    """First row per station has no predecessor -> lag filled with LAG_FILL_VALUE."""
    df = _clean_df([
        # Station A: first kept pair is A's earliest row -> lag must be filled.
        _row("2026-03-29 08:00:00", "A", 10, 10),
        _row("2026-03-29 08:30:00", "A", 7, 13),
        _row("2026-03-29 09:00:00", "A", 4, 16),
    ])

    bundle = features.build_features(features.build_targets(df))

    # X is sorted by (station, timestamp); the first row is A's earliest kept
    # observation, whose lag has no predecessor and is filled.
    first = bundle.X.iloc[0]
    assert int(first[config.COL_PREV_BIKES]) == config.LAG_FILL_VALUE
    assert int(first[config.COL_BIKE_CHANGE]) == config.LAG_FILL_VALUE
    assert int(first[config.COL_DOCK_CHANGE]) == config.LAG_FILL_VALUE
    # And a later row has a real (non-fabricated) change computed from history:
    # second kept row current bikes = 7, its predecessor within the frame = 10.
    second = bundle.X.iloc[1]
    assert int(second[config.COL_BIKE_CHANGE]) == 7 - 10


def test_empty_input_returns_empty_bundle():
    empty = pd.DataFrame(columns=[
        config.COL_TIMESTAMP, config.COL_CITY, config.COL_DISTRICT,
        config.COL_STATION, config.COL_TOTAL_DOCKS, config.COL_AVAILABLE_BIKES,
        config.COL_AVAILABLE_DOCKS, config.COL_LON, config.COL_LAT,
    ])
    tgt = features.build_targets(empty)
    bundle = features.build_features(tgt)

    assert len(bundle.X) == 0
    assert list(bundle.X.columns) == config.FEATURE_COLUMNS
    assert len(bundle.y_shortage) == 0
    assert len(bundle.y_full) == 0
