# -*- coding: utf-8 -*-
"""Validation script for Task 1 (R1 + R2) of the YouBike dispatch MVP.

Run from the project root:
    python -m tests.validate_task1

Read-only: does not modify any raw CSV. Prints a PASS/FAIL summary of the
Task 1 acceptance criteria.
"""

from __future__ import annotations

import sys

import pandas as pd

from src import config, data_loader, features


class Checker:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, condition, detail=""):
        self.checks += 1
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" -- {detail}"
        print(line)
        if not condition:
            self.failures.append(name)

    def done(self):
        print("-" * 60)
        if self.failures:
            print(f"RESULT: {len(self.failures)}/{self.checks} checks FAILED: "
                  f"{self.failures}")
            return 1
        print(f"RESULT: all {self.checks} checks PASSED")
        return 0


def main() -> int:
    c = Checker()

    # --- R1: load & clean ---------------------------------------------------
    df = data_loader.load_clean()

    c.check("R1.1 cp950 decoded, rows loaded", len(df) > 0,
            f"rows={len(df)}")

    # Internal English columns present after rename.
    expected_internal = set(config.COLUMN_RENAME.values())
    c.check("R1.2 required columns present (renamed)",
            expected_internal.issubset(set(df.columns)),
            f"columns={list(df.columns)}")

    # No mojibake in station names (a Chinese char should be present, no
    # replacement char U+FFFD).
    sample_station = str(df[config.COL_STATION].iloc[0])
    c.check("R1.1b no replacement char (mojibake) in station name",
            "\ufffd" not in "".join(df[config.COL_STATION].astype(str).head(50)),
            f"sample station={sample_station!r}")

    # Timestamp parsed to datetime.
    c.check("R1.3 timestamp is datetime dtype",
            pd.api.types.is_datetime64_any_dtype(df[config.COL_TIMESTAMP]),
            str(df[config.COL_TIMESTAMP].dtype))

    # No NaN in critical columns after cleaning.
    crit_na = int(df[config.CRITICAL_COLUMNS].isna().any(axis=1).sum())
    c.check("R1.4 no NaN in critical columns", crit_na == 0,
            f"rows with NaN={crit_na}")

    # Sorted by (station, timestamp).
    sorted_df = df.sort_values([config.COL_STATION, config.COL_TIMESTAMP])
    is_sorted = df.reset_index(drop=True).equals(sorted_df.reset_index(drop=True))
    c.check("R1.5 sorted by (station, timestamp)", is_sorted)

    # Report confirmed dataset facts (informational, sanity vs. probe).
    print(f"    info: station count = {df[config.COL_STATION].nunique()}")
    print(f"    info: distinct timestamps = {df[config.COL_TIMESTAMP].nunique()}")
    print(f"    info: ts range = {df[config.COL_TIMESTAMP].min()} .. "
          f"{df[config.COL_TIMESTAMP].max()}")

    # --- R2: 30-minute target ----------------------------------------------
    tgt = features.build_targets(df)

    c.check("R2 target rows > 0", len(tgt) > 0, f"target rows={len(tgt)}")

    # R2.2: all kept deltas within [25, 35].
    dmin = float(tgt[config.COL_DELTA_MIN].min())
    dmax = float(tgt[config.COL_DELTA_MIN].max())
    c.check("R2.2 all kept intervals within 25-35 min",
            dmin >= config.TARGET_MIN_MINUTES and dmax <= config.TARGET_MAX_MINUTES,
            f"delta_min range=[{dmin:.2f}, {dmax:.2f}]")

    # R2.3: no ~60-minute gaps kept.
    c.check("R2.3 no ~60-min gap kept as target",
            dmax <= config.TARGET_MAX_MINUTES,
            f"max delta_min={dmax:.2f}")

    # R2.4/R2.5: labels correct vs. future values.
    exp_shortage = (tgt[config.COL_FUTURE_BIKES] <= config.SHORTAGE_THRESHOLD).astype(int)
    exp_full = (tgt[config.COL_FUTURE_DOCKS] <= config.FULL_THRESHOLD).astype(int)
    c.check("R2.4 shortage_target matches future_bikes <= threshold",
            exp_shortage.equals(tgt[config.COL_SHORTAGE_TARGET]),
            f"positives={int(tgt[config.COL_SHORTAGE_TARGET].sum())}")
    c.check("R2.5 full_target matches future_docks <= threshold",
            exp_full.equals(tgt[config.COL_FULL_TARGET]),
            f"positives={int(tgt[config.COL_FULL_TARGET].sum())}")

    # --- Synthetic gap-exclusion test (deterministic) -----------------------
    # Build a tiny station series: 3 obs at 30-min spacing, then a 60-min gap.
    ts = pd.to_datetime([
        "2026-03-29 08:00:00",  # -> next at 08:30 (30 min) : valid
        "2026-03-29 08:30:00",  # -> next at 09:30 (60 min) : GAP, excluded
        "2026-03-29 09:30:00",  # -> next at 10:00 (30 min) : valid
        "2026-03-29 10:00:00",  # -> last, no next : excluded
    ])
    synth = pd.DataFrame({
        config.COL_TIMESTAMP: ts,
        config.COL_CITY: ["新北市"] * 4,
        config.COL_DISTRICT: ["板橋區"] * 4,
        config.COL_STATION: ["TEST_STN"] * 4,
        config.COL_TOTAL_DOCKS: [20, 20, 20, 20],
        config.COL_AVAILABLE_BIKES: [10, 8, 1, 0],   # future of row0=8, row2=0
        config.COL_AVAILABLE_DOCKS: [10, 12, 19, 20],
        config.COL_LON: [121.0] * 4,
        config.COL_LAT: [25.0] * 4,
    })
    synth_tgt = features.build_targets(synth)

    # Only 2 valid pairs expected (rows 0 and 2); the 60-min gap and the last
    # row are excluded.
    c.check("R2 synthetic: exactly 2 valid target rows (gap+last excluded)",
            len(synth_tgt) == 2, f"got {len(synth_tgt)} rows")

    # Row starting 08:00 -> future bikes=8 -> shortage 0 ; future docks=12 -> full 0
    # Row starting 09:30 -> future bikes=0 -> shortage 1 ; future docks=20 -> full 0
    if len(synth_tgt) == 2:
        r0 = synth_tgt.iloc[0]
        r1 = synth_tgt.iloc[1]
        c.check("R2 synthetic: first pair delta=30",
                abs(r0[config.COL_DELTA_MIN] - 30.0) < 1e-6,
                f"delta={r0[config.COL_DELTA_MIN]}")
        c.check("R2 synthetic: gap row excluded (second kept starts 09:30)",
                str(r1[config.COL_TIMESTAMP]) == "2026-03-29 09:30:00",
                f"second kept ts={r1[config.COL_TIMESTAMP]}")
        # Second pair future state: bikes=0 (<=2 -> shortage 1), docks=20 (>2 -> full 0).
        c.check("R2 synthetic: labels correct on second pair",
                int(r1[config.COL_SHORTAGE_TARGET]) == 1
                and int(r1[config.COL_FULL_TARGET]) == 0,
                f"shortage={int(r1[config.COL_SHORTAGE_TARGET])}, "
                f"full={int(r1[config.COL_FULL_TARGET])}")
        # First pair future state: bikes=8 (>2 -> shortage 0), docks=12 (>2 -> full 0).
        c.check("R2 synthetic: labels correct on first pair",
                int(r0[config.COL_SHORTAGE_TARGET]) == 0
                and int(r0[config.COL_FULL_TARGET]) == 0,
                f"shortage={int(r0[config.COL_SHORTAGE_TARGET])}, "
                f"full={int(r0[config.COL_FULL_TARGET])}")

    return c.done()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
