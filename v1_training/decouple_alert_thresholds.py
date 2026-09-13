# -*- coding: utf-8 -*-
"""Addendum: decouple the ALERT trigger from the EVENT definition.

Finding that motivates this: sweeping a single shared threshold for both the
event definition and the alert rule makes F1 a monotone function of event
prevalence, so "max F1" always selects the most permissive grid edge (LOW 0.25,
HIGH 0.65) rather than a genuine optimum. Meanwhile ROC-AUC is nearly flat
across the HIGH grid (0.956 -> 0.936), which shows the ranking quality is fine;
recall collapses only because a regression point estimate rarely reaches extreme
values.

Correct fix: keep the OPERATIONAL event definition fixed (0.20 / 0.80) and tune
the ALERT threshold applied to the prediction. This preserves the operational
meaning while recovering recall. Results are appended to
local_threshold_analysis.json under "decoupled_alert_calibration".
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "local_threshold_analysis.json")
MODELS = ["weekday_30m_bike_ratio", "weekday_60m_bike_ratio",
          "weekend_30m_bike_ratio", "weekend_60m_bike_ratio"]
RECALL_TARGET = 0.75
EVENT_LOW, EVENT_HIGH = 0.20, 0.80


def prf(a, p):
    tp = int(np.sum(a & p)); fp = int(np.sum(~a & p)); fn = int(np.sum(a & ~p))
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0
    return pr, rc, f1


def main():
    preds = {m: pd.read_parquet(
        os.path.join(HERE, "local_fallback", m, "validation_predictions.parquet"))
        for m in MODELS}

    grid = [round(x, 3) for x in np.arange(0.05, 0.96, 0.01)]
    result = {"event_low": EVENT_LOW, "event_high": EVENT_HIGH,
              "recall_target": RECALL_TARGET, "low": {}, "high": {}}

    for side, event, direction in (("low", EVENT_LOW, "lt"), ("high", EVENT_HIGH, "gt")):
        rows = []
        for g in grid:
            prs, rcs, f1s = [], [], []
            for m in MODELS:
                yt, yp = preds[m]["y_true"].to_numpy(), preds[m]["y_pred"].to_numpy()
                actual = (yt < event) if direction == "lt" else (yt > event)
                alert = (yp < g) if direction == "lt" else (yp > g)
                pr, rc, f1 = prf(actual, alert)
                prs.append(pr); rcs.append(rc); f1s.append(f1)
            rows.append({"alert_threshold": g,
                         "mean_precision": float(np.mean(prs)),
                         "mean_recall": float(np.mean(rcs)),
                         "mean_f1": float(np.mean(f1s)),
                         "min_recall": float(np.min(rcs))})
        elig = [r for r in rows if r["mean_recall"] >= RECALL_TARGET]
        best = max(elig, key=lambda r: r["mean_f1"]) if elig else \
            max(rows, key=lambda r: r["mean_f1"])
        result[side] = {
            "grid_result": rows,
            "recommended_alert_threshold": best["alert_threshold"],
            "at_recommended": best,
            "recall_target_met": bool(elig),
            "interpretation": (
                f"event stays defined as y_true "
                f"{'<' if direction == 'lt' else '>'} {event} "
                f"(operational meaning unchanged); the alert fires when "
                f"predicted ratio {'<' if direction == 'lt' else '>'} "
                f"{best['alert_threshold']}"),
        }
        print(f"{side}: event {'<' if direction=='lt' else '>'} {event} -> "
              f"alert {'<' if direction=='lt' else '>'} {best['alert_threshold']} "
              f"| meanP={best['mean_precision']:.4f} meanR={best['mean_recall']:.4f} "
              f"meanF1={best['mean_f1']:.4f} minR={best['min_recall']:.4f} "
              f"target_met={bool(elig)}")

    with open(OUT, encoding="utf-8") as f:
        a = json.load(f)
    a["decoupled_alert_calibration"] = result
    a["methodology_caveat"] = (
        "Sweeping one shared threshold for BOTH the event definition and the "
        "alert rule makes F1 a monotone function of event prevalence, so max-F1 "
        "always lands on the most permissive grid edge (LOW 0.25 / HIGH 0.65). "
        "ROC-AUC is nearly flat across the HIGH grid (0.956 -> 0.936), showing "
        "ranking quality is not the limiter. Recommendation: keep the "
        "operational event definition at 0.20 / 0.80 and use the decoupled "
        "alert thresholds in 'decoupled_alert_calibration' instead of moving "
        "the operational definition to 0.65."
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2, ensure_ascii=False)
    print("appended decoupled_alert_calibration ->", OUT)


if __name__ == "__main__":
    main()
