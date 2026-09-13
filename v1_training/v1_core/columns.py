# -*- coding: utf-8 -*-
"""Adaptive column mapping for the V1 enriched YouBike datasets.

Why adaptive: the enriched S3 files (平日_含最近距離_含天氣.csv) carry extra
weather / nearest-distance / peak columns whose EXACT header spellings were not
observable when this code was written. Rather than hard-coding guesses, every
logical field declares a list of accepted spellings plus a loose token-based
matcher. Mapping is resolved at runtime against the real header and recorded in
metadata so the choice is auditable.

Policy (per mission brief):
  * Obvious synonyms are mapped automatically and logged.
  * Only genuinely undeterminable semantics raise.
  * Fields excluded on purpose (temperature, 時段, 資料來源) are never mapped
    into features; they are recorded in `excluded_reasons`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# --- logical field names --------------------------------------------------- #
TIMESTAMP = "timestamp"
CITY = "city"
DISTRICT = "district"
STATION = "station"
TOTAL_DOCKS = "total_docks"
AVAILABLE_BIKES = "available_bikes"
AVAILABLE_DOCKS = "available_docks"
LON = "lon"
LAT = "lat"
SOURCE = "source"
PERIOD = "period"
IS_PEAK = "is_peak"
RAINFALL = "rainfall"
TEMPERATURE = "temperature"
DIST_JUNIOR_HIGH = "nearest_junior_high_distance"
DIST_UNIVERSITY = "nearest_university_distance"
DIST_MRT = "nearest_mrt_distance"
DIST_BUS = "nearest_bus_distance"

# Fields that MUST resolve or we cannot build the target at all.
REQUIRED_FIELDS = (
    TIMESTAMP,
    STATION,
    TOTAL_DOCKS,
    AVAILABLE_BIKES,
    AVAILABLE_DOCKS,
)

# Exact / near-exact accepted spellings, most specific first.
CANDIDATES: Dict[str, List[str]] = {
    TIMESTAMP: ["日期", "時間", "資料時間", "datetime", "timestamp", "date", "時間戳"],
    CITY: ["城市", "縣市", "city"],
    DISTRICT: ["行政區", "區", "district", "鄉鎮市區"],
    STATION: ["場站名稱", "站點名稱", "站名", "場站", "station", "station_name"],
    TOTAL_DOCKS: ["總車柱數", "總停車柱數", "車柱數", "總數", "total_docks", "capacity"],
    AVAILABLE_BIKES: ["可借車數", "可借車輛數", "可借", "available_bikes", "bikes"],
    AVAILABLE_DOCKS: ["可還位數", "可還空位數", "可還", "available_docks", "docks"],
    LON: ["經度", "lon", "longitude", "lng"],
    LAT: ["緯度", "lat", "latitude"],
    SOURCE: ["資料來源", "來源", "source"],
    PERIOD: ["時段", "period", "time_period"],
    IS_PEAK: ["是否尖峰", "尖峰", "is_peak", "peak", "尖峰時段"],
    RAINFALL: ["降雨量", "雨量", "rainfall", "precipitation", "降水量"],
    TEMPERATURE: ["平均溫度_C", "平均溫度", "溫度", "temperature", "temp"],
    DIST_JUNIOR_HIGH: ["最近國中距離", "最近國中", "國中距離", "nearest_junior_high"],
    DIST_UNIVERSITY: ["最近大專距離", "最近大專", "大專距離", "最近大學距離", "nearest_university"],
    DIST_MRT: ["最近捷運距離", "最近捷運", "捷運距離", "nearest_mrt"],
    DIST_BUS: ["最近公車距離", "最近公車", "公車距離", "nearest_bus"],
}

# Loose token matcher: ALL tokens must appear in the normalized header.
TOKEN_RULES: Dict[str, List[List[str]]] = {
    IS_PEAK: [["尖峰"], ["peak"]],
    RAINFALL: [["雨"], ["rain"], ["precip"]],
    TEMPERATURE: [["溫度"], ["temp"]],
    DIST_JUNIOR_HIGH: [["國中"], ["junior"]],
    DIST_UNIVERSITY: [["大專"], ["大學"], ["univ"], ["college"]],
    DIST_MRT: [["捷運"], ["mrt"], ["metro"]],
    DIST_BUS: [["公車"], ["bus"]],
    TOTAL_DOCKS: [["總", "柱"], ["total", "dock"], ["capacity"]],
    AVAILABLE_BIKES: [["可借"], ["available", "bike"]],
    AVAILABLE_DOCKS: [["可還"], ["available", "dock"]],
}


def _norm(name: str) -> str:
    """Normalize a header for comparison (NFKC, strip quotes/space, lowercase)."""
    s = unicodedata.normalize("NFKC", str(name))
    s = s.replace("\ufeff", "").strip().strip('"').strip("'").strip()
    s = re.sub(r"\s+", "", s)
    return s.lower()


def map_columns(header: List[str]) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    """Resolve logical field -> actual header name.

    Returns:
        (mapping, how, unmapped_headers)
          mapping: logical field -> real column name
          how:     logical field -> "exact:<pattern>" | "token:<tokens>"
          unmapped_headers: real columns not claimed by any logical field
    """
    norm_to_real: Dict[str, str] = {}
    for real in header:
        norm_to_real.setdefault(_norm(real), real)

    mapping: Dict[str, str] = {}
    how: Dict[str, str] = {}
    claimed = set()

    # Pass 1: exact / accepted spellings.
    for field, patterns in CANDIDATES.items():
        for pat in patterns:
            npat = _norm(pat)
            if npat in norm_to_real and norm_to_real[npat] not in claimed:
                mapping[field] = norm_to_real[npat]
                how[field] = f"exact:{pat}"
                claimed.add(norm_to_real[npat])
                break

    # Pass 2: loose token match for anything still missing.
    for field, rulesets in TOKEN_RULES.items():
        if field in mapping:
            continue
        for tokens in rulesets:
            hit = None
            for nreal, real in norm_to_real.items():
                if real in claimed:
                    continue
                if all(_norm(t) in nreal for t in tokens):
                    hit = real
                    break
            if hit is not None:
                mapping[field] = hit
                how[field] = f"token:{'+'.join(tokens)}"
                claimed.add(hit)
                break

    unmapped = [h for h in header if h not in claimed]
    return mapping, how, unmapped


class ColumnMappingError(RuntimeError):
    """Raised only when a REQUIRED field cannot be identified at all."""


def require_fields(mapping: Dict[str, str], header: List[str]) -> None:
    """Fail loudly if a field needed to build the target is unresolvable."""
    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        raise ColumnMappingError(
            "Could not identify required column(s): "
            + ", ".join(missing)
            + ". Actual header was: "
            + ", ".join(map(str, header))
            + ". Target construction is impossible without these, so the run "
            "was stopped rather than guessing."
        )
