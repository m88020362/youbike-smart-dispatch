# -*- coding: utf-8 -*-
"""Data loading and cleaning for the YouBike predictive dispatch MVP (Task 1, R1).

Reads a raw cp950 CSV, validates required columns, renames to internal English
column names, parses timestamps, coerces numeric columns, drops unusable rows,
and returns a DataFrame sorted by (station, timestamp).

Read-only with respect to the raw CSV: this module never writes to it.
"""

from __future__ import annotations

import os

import pandas as pd

from . import config


def load_clean(csv_path: str | None = None, encoding: str | None = None) -> pd.DataFrame:
    """Load and clean a raw YouBike snapshot CSV.

    Args:
        csv_path: Path to the CSV. Defaults to config.DEFAULT_CSV.
        encoding: File encoding. Defaults to config.DATA_ENCODING (cp950).

    Returns:
        Cleaned DataFrame with internal English column names, parsed timestamp,
        numeric inventory columns, sorted by (station, timestamp).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If any required column is missing.
    """
    csv_path = csv_path or config.DEFAULT_CSV
    encoding = encoding or config.DATA_ENCODING

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding=encoding)

    # Validate required columns before any transformation.
    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only required columns, then rename to internal English names.
    df = df[config.REQUIRED_COLUMNS].copy()
    df = df.rename(columns=config.COLUMN_RENAME)

    n_raw = len(df)

    # Parse timestamp.
    df[config.COL_TIMESTAMP] = pd.to_datetime(
        df[config.COL_TIMESTAMP], errors="coerce"
    )

    # Coerce numeric columns; unparseable values become NaN.
    for col in config.NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing any critical value (explicit strategy, no silent NaN).
    before_drop = len(df)
    df = df.dropna(subset=config.CRITICAL_COLUMNS)
    n_dropped = before_drop - len(df)

    # Cast integer-like inventory columns to nullable-safe ints.
    for col in (
        config.COL_TOTAL_DOCKS,
        config.COL_AVAILABLE_BIKES,
        config.COL_AVAILABLE_DOCKS,
    ):
        df[col] = df[col].round().astype(int)

    # Strip whitespace on station/district string keys.
    for col in (config.COL_STATION, config.COL_DISTRICT, config.COL_CITY):
        df[col] = df[col].astype(str).str.strip()

    # Sort by (station, timestamp) for downstream temporal logic.
    df = df.sort_values(
        [config.COL_STATION, config.COL_TIMESTAMP]
    ).reset_index(drop=True)

    if n_dropped:
        print(
            f"[data_loader] loaded {n_raw} rows, dropped {n_dropped} invalid, "
            f"kept {len(df)}."
        )

    return df
