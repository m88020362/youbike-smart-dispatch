# -*- coding: utf-8 -*-
"""Status / summary writers so the overnight run is readable at a glance."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Optional

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATUS_PATH = os.path.join(HERE, "overnight_status.json")
SUMMARY_PATH = os.path.join(HERE, "OVERNIGHT_SUMMARY.md")

MODEL_ORDER = [
    "weekday_30m_bike_ratio",
    "weekday_60m_bike_ratio",
    "weekend_30m_bike_ratio",
    "weekend_60m_bike_ratio",
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_status() -> Dict:
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "current_phase": "init",
        "timestamp": _now(),
        "branch": "stable-v1-training",
        "dataset_range": None,
        "latest_success": None,
        "training_job_name": None,
        "training_job_status": None,
        "artifact_uri": None,
        "metrics_path": None,
        "last_error": None,
        "models": {},
    }


def update_status(**fields) -> Dict:
    st = load_status()
    st.update(fields)
    st["timestamp"] = _now()
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
    return st


def record_model(name: str, **fields) -> Dict:
    st = load_status()
    st.setdefault("models", {})
    entry = st["models"].get(name, {})
    entry.update(fields)
    entry["updated"] = _now()
    st["models"][name] = entry
    if fields.get("status") == "SUCCESS":
        st["latest_success"] = name
    st["timestamp"] = _now()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
    return st


def _fmt(v, nd=4):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_summary() -> str:
    st = load_status()
    models = st.get("models", {})
    lines = []
    lines.append("# V1 Overnight Training Summary")
    lines.append("")
    lines.append(f"Generated: {_now()}")
    lines.append(f"Branch: {st.get('branch')}")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    for name in MODEL_ORDER:
        entry = models.get(name)
        status = entry.get("status", "NOT RUN") if entry else "NOT RUN"
        label = "P0 " + name if name.startswith("weekday_30m") else name
        lines.append(f"{label}: {status}")
    lines.append("")

    for name in MODEL_ORDER:
        entry = models.get(name)
        if not entry or entry.get("status") != "SUCCESS":
            continue
        m = entry.get("metrics", {}) or {}
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"training_backend: {entry.get('training_backend')}")
        lines.append(f"使用資料期間: {entry.get('actual_start_date')} -> {entry.get('actual_end_date')}")
        lines.append(f"train_rows: {entry.get('train_rows')}  validation_rows: {entry.get('validation_rows')}")
        lines.append("")
        lines.append(f"MAE:  {_fmt(m.get('MAE'))}")
        lines.append(f"RMSE: {_fmt(m.get('RMSE'))}")
        lines.append(f"R2:   {_fmt(m.get('R2'))}")
        lines.append("")
        lines.append("Low-bike (< 0.20):")
        lines.append(f"  Precision {_fmt(m.get('low_bike_precision'))}")
        lines.append(f"  Recall    {_fmt(m.get('low_bike_recall'))}")
        lines.append(f"  F1        {_fmt(m.get('low_bike_f1'))}")
        lines.append(f"  AUC       {_fmt(m.get('low_bike_auc'))}")
        lines.append("")
        lines.append("High-occupancy (> 0.80):")
        lines.append(f"  Precision {_fmt(m.get('high_occupancy_precision'))}")
        lines.append(f"  Recall    {_fmt(m.get('high_occupancy_recall'))}")
        lines.append(f"  F1        {_fmt(m.get('high_occupancy_f1'))}")
        lines.append(f"  AUC       {_fmt(m.get('high_occupancy_auc'))}")
        lines.append("")
        lines.append(f"Artifact: {entry.get('artifact_uri')}")
        lines.append("")

    lines.append("## Errors / blockers")
    lines.append("")
    blockers = st.get("blockers") or []
    if st.get("last_error"):
        blockers = list(blockers) + [str(st["last_error"])]
    if blockers:
        for b in blockers:
            lines.append(f"- {b}")
    else:
        lines.append("- none")
    lines.append("")

    text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    return text
