# -*- coding: utf-8 -*-
"""Pytest tests for src/data_loader.py (R1) of the YouBike dispatch MVP.

Locks in the most important loading/cleaning behaviour to prevent regression
(design §2, §12; requirements R1). Most cases use small synthetic cp950 CSVs
written to a tmp dir; a single case exercises the real representative CSV to
prove genuine cp950 decoding.

Read-only guarantee: these tests NEVER modify, move, or rename the real dataset
CSV. Synthetic CSVs are written only under pytest's tmp_path.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from src import config, data_loader


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _write_cp950_csv(path: str, rows: list[dict]) -> None:
    """Write a raw-header (Chinese columns) cp950 CSV from a list of row dicts."""
    df = pd.DataFrame(rows, columns=config.REQUIRED_COLUMNS)
    df.to_csv(path, index=False, encoding=config.DATA_ENCODING)


def _sample_rows() -> list[dict]:
    """Two stations, out-of-order timestamps, valid numeric values."""
    return [
        {
            "日期": "2026-03-29 08:30:00", "城市": "新北市", "行政區": "板橋區",
            "場站名稱": "乙站", "總車柱數": 20, "可借車數": 5, "可還位數": 15,
            "經度": 121.46, "緯度": 25.01,
        },
        {
            "日期": "2026-03-29 08:00:00", "城市": "新北市", "行政區": "板橋區",
            "場站名稱": "乙站", "總車柱數": 20, "可借車數": 8, "可還位數": 12,
            "經度": 121.46, "緯度": 25.01,
        },
        {
            "日期": "2026-03-29 08:00:00", "城市": "新北市", "行政區": "中和區",
            "場站名稱": "甲站", "總車柱數": 30, "可借車數": 10, "可還位數": 20,
            "經度": 121.50, "緯度": 25.00,
        },
    ]


# --------------------------------------------------------------------------- #
# R1.1 -- real cp950 CSV decodes without mojibake                              #
# --------------------------------------------------------------------------- #
def test_real_cp950_csv_reads_without_mojibake():
    """The representative dataset CSV decodes as cp950 (R1.1)."""
    if not os.path.exists(config.DEFAULT_CSV):
        pytest.skip(f"representative CSV not present: {config.DEFAULT_CSV}")

    df = data_loader.load_clean()  # defaults to config.DEFAULT_CSV / cp950

    assert len(df) > 0
    # Internal English columns present after rename.
    assert set(config.COLUMN_RENAME.values()).issubset(df.columns)
    # No Unicode replacement char (mojibake) in the station names.
    joined = "".join(df[config.COL_STATION].astype(str).head(200))
    assert "\ufffd" not in joined


# --------------------------------------------------------------------------- #
# R1.2 -- required columns present / missing column raises ValueError          #
# --------------------------------------------------------------------------- #
def test_required_columns_present_after_load(tmp_path):
    csv = tmp_path / "ok.csv"
    _write_cp950_csv(str(csv), _sample_rows())

    df = data_loader.load_clean(str(csv))

    assert set(config.COLUMN_RENAME.values()).issubset(df.columns)


def test_missing_required_column_raises_valueerror(tmp_path):
    """Dropping one required column must raise a clear ValueError (R1.2)."""
    rows = _sample_rows()
    df = pd.DataFrame(rows, columns=config.REQUIRED_COLUMNS)
    # Drop a single required column ("可還位數").
    df = df.drop(columns=["可還位數"])
    csv = tmp_path / "missing_col.csv"
    df.to_csv(str(csv), index=False, encoding=config.DATA_ENCODING)

    with pytest.raises(ValueError):
        data_loader.load_clean(str(csv))


def test_missing_csv_raises_filenotfounderror(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        data_loader.load_clean(str(missing))


# --------------------------------------------------------------------------- #
# R1.3 -- timestamp parsed to datetime                                         #
# --------------------------------------------------------------------------- #
def test_timestamp_parsed_to_datetime(tmp_path):
    csv = tmp_path / "ts.csv"
    _write_cp950_csv(str(csv), _sample_rows())

    df = data_loader.load_clean(str(csv))

    assert pd.api.types.is_datetime64_any_dtype(df[config.COL_TIMESTAMP])
    # And the values round-trip to the expected instants.
    assert df[config.COL_TIMESTAMP].min() == pd.Timestamp("2026-03-29 08:00:00")


# --------------------------------------------------------------------------- #
# R1.4 -- invalid numeric rows dropped (no silent NaN in critical columns)     #
# --------------------------------------------------------------------------- #
def test_unparseable_numeric_row_is_dropped(tmp_path):
    rows = _sample_rows()
    # Corrupt the "可借車數" of one row with a non-numeric token.
    rows[0]["可借車數"] = "N/A"
    csv = tmp_path / "bad_numeric.csv"
    _write_cp950_csv(str(csv), rows)

    df = data_loader.load_clean(str(csv))

    # The corrupted row is dropped; no NaN remains in critical columns.
    assert int(df[config.CRITICAL_COLUMNS].isna().any(axis=1).sum()) == 0
    assert len(df) == len(rows) - 1


# --------------------------------------------------------------------------- #
# R1.5 -- output sorted by (station, timestamp)                                #
# --------------------------------------------------------------------------- #
def test_output_sorted_by_station_then_timestamp(tmp_path):
    csv = tmp_path / "sort.csv"
    _write_cp950_csv(str(csv), _sample_rows())

    df = data_loader.load_clean(str(csv))

    expected = df.sort_values([config.COL_STATION, config.COL_TIMESTAMP])
    assert df.reset_index(drop=True).equals(expected.reset_index(drop=True))
    # Concretely: 甲站 rows come before 乙站? Sorting is lexicographic on the
    # station string; within a station, timestamps ascend.
    lower = df[df[config.COL_STATION] == "乙站"]
    assert list(lower[config.COL_TIMESTAMP]) == sorted(lower[config.COL_TIMESTAMP])


# --------------------------------------------------------------------------- #
# R1.6 -- the source CSV is never modified                                     #
# --------------------------------------------------------------------------- #
def test_source_csv_not_modified(tmp_path):
    """load_clean must be read-only w.r.t. the source file (R1.6)."""
    csv = tmp_path / "readonly.csv"
    _write_cp950_csv(str(csv), _sample_rows())

    before_bytes = csv.read_bytes()
    before_mtime = os.path.getmtime(str(csv))

    _ = data_loader.load_clean(str(csv))

    assert csv.read_bytes() == before_bytes
    assert os.path.getmtime(str(csv)) == before_mtime
