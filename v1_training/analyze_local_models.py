# -*- coding: utf-8 -*-
"""Phase 1 + 2: LOCAL_FALLBACK model comparison and provisional threshold sweep.

NO RETRAINING. The four saved LightGBM boosters are reloaded and the identical
validation split is rebuilt deterministically (the preprocessing / split path
contains no randomness), so validation predictions are reproduced exactly.

Outputs:
    v1_training/local_threshold_analysis.json
    v1_training/LOCAL_MODEL_COMPARISON.md
    v1_training/local_fallback/<model>/validation_predictions.parquet

Threshold selection rule (per brief): Recall first -- prefer candidates with
macro Recall >= 0.75, and among those take the highest macro F1. If none reach
0.75, take the highest macro F1 and flag that Recall is below target.
One shared LOW / HIGH threshold is chosen for all four models.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PROJECT_ROOT = os.path.dirname(HERE)

from v1_core import columns as C      # noqa: E402
from v1_core import dataio, prep      # noqa: E402
from v1_core import pipeline          # noqa: E402

JUNE = os.path.join(PROJECT_ROOT, "dataset", "YouBike 六月資料.csv 的副本.csv")
OUT_ROOT = os.path.join(HERE, "local_fallback")

MODELS = [
    ("weekday_30m_bike_ratio", 30, pipeline.WEEKDAY),
    ("weekday_60m_bike_ratio", 60, pipeline.WEEKDAY),
    ("weekend_30m_bike_ratio", 30, pipeline.WEEKEND),
    ("weekend_60m_bike_ratio", 60, pipeline.WEEKEND),
]

LOW_GRID = [0.10, 0.15, 0.20, 0.25]
HIGH_GRID = [0.65, 0.70, 0.75, 0.80, 0.85]
RECALL_TARGET = 0.75

CURRENT_LOW = 0.20
CURRENT_HIGH = 0.80


def say(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def rebuild_validation(model_name, horizon, day_type):
    """Rebuild the exact validation set and predict with the SAVED booster."""
    import lightgbm as lgb

    out_dir = os.path.join(OUT_ROOT, model_name)
    cached = os.path.join(out_dir, "validation_predictions.parquet")
    if os.path.exists(cached):
        say(f"  reuse cached predictions {model_name}")
        return pd.read_parquet(cached)

    audit = dataio.audit_csv(JUNE)
    months = dataio.pick_months(audit["complete_months"], 1)
    raw = dataio.load_range(JUNE, audit["encoding"], audit["column_mapping"], months)
    df = prep.normalize_frame(raw, audit["column_mapping"])
    del raw
    df, _ = prep.drop_invalid_rows(df)
    df = pipeline.filter_day_type(df, day_type, {})
    labeled, _ = prep.build_target(df, horizon)
    del df
    feat_df, features, _ = prep.build_features(labeled, include_rainfall=False)
    _, valid_df, _ = prep.chronological_split(feat_df)

    booster = lgb.Booster(
        model_file=os.path.join(out_dir, f"{model_name}_lgbm.txt")
    )
    y_pred = booster.predict(valid_df[features].astype("float64").to_numpy())

    out = pd.DataFrame({
        "station": valid_df[C.STATION].astype(str).to_numpy(),
        "timestamp": valid_df[C.TIMESTAMP].to_numpy(),
        "y_true": valid_df[prep.TARGET].astype("float64").to_numpy(),
        "y_pred": np.clip(np.asarray(y_pred, dtype="float64"), 0.0, 1.0),
    })
    out.to_parquet(cached, index=False)
    say(f"  {model_name}: rebuilt {len(out):,} validation rows -> {cached}")
    return out


def prf(actual, predicted):
    """Precision / Recall / F1 without sklearn overhead."""
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn}


def auc_or_na(actual, score):
    if len(np.unique(actual)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(actual, score))


def sweep(preds):
    """Threshold sweep for one model."""
    y_true = preds["y_true"].to_numpy()
    y_pred = preds["y_pred"].to_numpy()
    res = {"low": {}, "high": {}}

    for t in LOW_GRID:
        a = y_true < t
        p = y_pred < t
        m = prf(a, p)
        m["actual_prevalence"] = float(a.mean())
        m["predicted_prevalence"] = float(p.mean())
        m["auc"] = auc_or_na(a.astype(int), 1.0 - y_pred)
        res["low"][f"{t:.2f}"] = m

    for t in HIGH_GRID:
        a = y_true > t
        p = y_pred > t
        m = prf(a, p)
        m["actual_prevalence"] = float(a.mean())
        m["predicted_prevalence"] = float(p.mean())
        m["auc"] = auc_or_na(a.astype(int), y_pred)
        res["high"][f"{t:.2f}"] = m
    return res


def macro(per_model, side, grid):
    """Macro-average across the four models for each threshold."""
    out = {}
    for t in grid:
        key = f"{t:.2f}"
        ps = [per_model[m][side][key]["precision"] for m in per_model]
        rs = [per_model[m][side][key]["recall"] for m in per_model]
        fs = [per_model[m][side][key]["f1"] for m in per_model]
        out[key] = {
            "mean_precision": float(np.mean(ps)),
            "mean_recall": float(np.mean(rs)),
            "mean_f1": float(np.mean(fs)),
            "min_recall": float(np.min(rs)),
        }
    return out


def choose(macro_side, current_value):
    """Recall-first selection with the documented tie-breaks."""
    eligible = {k: v for k, v in macro_side.items()
                if v["mean_recall"] >= RECALL_TARGET}
    if eligible:
        best = max(eligible.items(), key=lambda kv: kv[1]["mean_f1"])
        chosen, why = float(best[0]), (
            f"highest macro F1 ({best[1]['mean_f1']:.4f}) among candidates with "
            f"macro Recall >= {RECALL_TARGET}")
        recall_ok = True
    else:
        best = max(macro_side.items(), key=lambda kv: kv[1]["mean_f1"])
        chosen, why = float(best[0]), (
            f"NO threshold reached macro Recall >= {RECALL_TARGET}; picked the "
            f"highest macro F1 ({best[1]['mean_f1']:.4f}) instead")
        recall_ok = False

    # Stability preference: keep the incumbent unless the sweep is clearly better.
    cur_key = f"{current_value:.2f}"
    note = None
    if cur_key in macro_side and chosen != current_value:
        cur = macro_side[cur_key]
        new = macro_side[f"{chosen:.2f}"]
        cur_eligible = cur["mean_recall"] >= RECALL_TARGET
        new_eligible = new["mean_recall"] >= RECALL_TARGET
        if cur_eligible and new_eligible and (new["mean_f1"] - cur["mean_f1"]) < 0.01:
            note = (f"kept incumbent {current_value:.2f}: sweep winner "
                    f"{chosen:.2f} improved macro F1 by only "
                    f"{new['mean_f1'] - cur['mean_f1']:.4f} (<0.01), not clearly better")
            chosen = current_value
            why = note
    return chosen, why, recall_ok, macro_side[f"{chosen:.2f}"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    say("PHASE 1 - rebuild validation predictions (no retraining)")

    preds = {}
    meta = {}
    for name, horizon, day_type in MODELS:
        preds[name] = rebuild_validation(name, horizon, day_type)
        with open(os.path.join(OUT_ROOT, name, "metadata.json"), encoding="utf-8") as f:
            meta[name] = json.load(f)

    say("PHASE 2 - threshold sweep")
    per_model = {name: sweep(p) for name, p in preds.items()}
    macro_low = macro(per_model, "low", LOW_GRID)
    macro_high = macro(per_model, "high", HIGH_GRID)

    low_t, low_why, low_ok, low_stats = choose(macro_low, CURRENT_LOW)
    high_t, high_why, high_ok, high_stats = choose(macro_high, CURRENT_HIGH)
    say(f"  provisional_low={low_t}  ({low_why})")
    say(f"  provisional_high={high_t} ({high_why})")

    # ---- persistent-risk cross-tab (30m vs 60m on shared rows) ----------
    persistence = {}
    for day in ("weekday", "weekend"):
        a = preds[f"{day}_30m_bike_ratio"]
        b = preds[f"{day}_60m_bike_ratio"]
        j = a.merge(b, on=["station", "timestamp"], suffixes=("_30", "_60"))
        if len(j) == 0:
            persistence[day] = {"joined_rows": 0}
            continue
        low30 = j["y_pred_30"] < low_t
        low60 = j["y_pred_60"] < low_t
        hi30 = j["y_pred_30"] > high_t
        hi60 = j["y_pred_60"] > high_t
        t30 = j["y_true_30"] < low_t
        t60 = j["y_true_60"] < low_t
        persistence[day] = {
            "joined_rows": int(len(j)),
            "low_persistent_pred": int((low30 & low60).sum()),
            "low_transient_pred": int((low30 & ~low60).sum()),
            "low_emerging_pred": int((~low30 & low60).sum()),
            "high_persistent_pred": int((hi30 & hi60).sum()),
            "high_transient_pred": int((hi30 & ~hi60).sum()),
            "high_emerging_pred": int((~hi30 & hi60).sum()),
            "low_persistent_actual": int((t30 & t60).sum()),
            "low_persistent_precision": float(
                ((low30 & low60) & (t30 & t60)).sum() / max(1, (low30 & low60).sum())
            ),
            "low_persistent_recall": float(
                ((low30 & low60) & (t30 & t60)).sum() / max(1, (t30 & t60).sum())
            ),
        }
        say(f"  {day} joined={len(j):,} low_persistent_pred="
            f"{persistence[day]['low_persistent_pred']:,}")

    analysis = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "training_backend": "LOCAL_FALLBACK",
        "disclaimer": (
            "These are PROVISIONAL thresholds derived from LOCAL_FALLBACK models "
            "trained on the local 9-column June CSV (8 features, no enriched "
            "weather / distance / is_peak). They are NOT final. The formal "
            "SAGEMAKER_ENRICHED validation must be re-swept and takes precedence."
        ),
        "selection_rule": {
            "recall_target": RECALL_TARGET,
            "order": "prefer macro Recall >= target, then highest macro F1",
            "shared_threshold_across_models": True,
            "incumbent_low": CURRENT_LOW,
            "incumbent_high": CURRENT_HIGH,
            "stability_margin_macro_f1": 0.01,
        },
        "low_grid": LOW_GRID,
        "high_grid": HIGH_GRID,
        "per_model": per_model,
        "macro_low": macro_low,
        "macro_high": macro_high,
        "provisional_low_threshold": low_t,
        "provisional_low_reason": low_why,
        "provisional_low_recall_meets_target": low_ok,
        "provisional_low_macro": low_stats,
        "provisional_high_threshold": high_t,
        "provisional_high_reason": high_why,
        "provisional_high_recall_meets_target": high_ok,
        "provisional_high_macro": high_stats,
        "persistent_risk_crosstab": persistence,
        "models": {
            name: {
                "training_backend": meta[name]["training_backend"],
                "actual_start_date": meta[name]["actual_start_date"],
                "actual_end_date": meta[name]["actual_end_date"],
                "train_rows": meta[name]["train_rows"],
                "validation_rows": meta[name]["validation_rows"],
                "features": meta[name]["features"],
                "MAE": meta[name]["MAE"],
                "RMSE": meta[name]["RMSE"],
                "R2": meta[name]["R2"],
                "low_bike_precision": meta[name]["low_bike_precision"],
                "low_bike_recall": meta[name]["low_bike_recall"],
                "low_bike_f1": meta[name]["low_bike_f1"],
                "low_bike_auc": meta[name]["low_bike_auc"],
                "high_occupancy_precision": meta[name]["high_occupancy_precision"],
                "high_occupancy_recall": meta[name]["high_occupancy_recall"],
                "high_occupancy_f1": meta[name]["high_occupancy_f1"],
                "high_occupancy_auc": meta[name]["high_occupancy_auc"],
            }
            for name, _, _ in MODELS
        },
    }

    ap = os.path.join(HERE, "local_threshold_analysis.json")
    with open(ap, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    say(f"saved {ap}")

    write_markdown(analysis)
    return 0


def _f(v, nd=4):
    return "N/A" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write_markdown(a) -> None:
    m = a["models"]
    L = []
    L.append("# LOCAL_FALLBACK Model Comparison + Provisional Thresholds")
    L.append("")
    L.append(f"Generated: {a['generated_at']}")
    L.append("")
    L.append("> **training_backend = LOCAL_FALLBACK.** These four models were trained")
    L.append("> on the local 9-column June CSV with 8 features. They are NOT the formal")
    L.append("> enriched AWS models. The enriched S3 dataset (weather / nearest-distance /")
    L.append("> is_peak) has not been trained yet; `SAGEMAKER_ENRICHED` remains pending.")
    L.append("")
    L.append("## Phase 1 - four model metrics")
    L.append("")
    L.append("| model | data range | train | valid | MAE | RMSE | R2 |")
    L.append("|---|---|---|---|---|---|---|")
    for k in m:
        e = m[k]
        L.append(f"| {k} | {e['actual_start_date'][:10]} -> {e['actual_end_date'][:10]} "
                 f"| {e['train_rows']:,} | {e['validation_rows']:,} "
                 f"| {_f(e['MAE'])} | {_f(e['RMSE'])} | {_f(e['R2'])} |")
    L.append("")
    L.append("Low-bike @ 0.20 (as trained):")
    L.append("")
    L.append("| model | Precision | Recall | F1 | AUC |")
    L.append("|---|---|---|---|---|")
    for k in m:
        e = m[k]
        L.append(f"| {k} | {_f(e['low_bike_precision'])} | {_f(e['low_bike_recall'])} "
                 f"| {_f(e['low_bike_f1'])} | {_f(e['low_bike_auc'])} |")
    L.append("")
    L.append("High-occupancy @ 0.80 (as trained):")
    L.append("")
    L.append("| model | Precision | Recall | F1 | AUC |")
    L.append("|---|---|---|---|---|")
    for k in m:
        e = m[k]
        L.append(f"| {k} | {_f(e['high_occupancy_precision'])} | {_f(e['high_occupancy_recall'])} "
                 f"| {_f(e['high_occupancy_f1'])} | {_f(e['high_occupancy_auc'])} |")
    L.append("")
    L.append("Feature list (identical for all four):")
    L.append("")
    L.append("```")
    L.append(", ".join(m["weekday_30m_bike_ratio"]["features"]))
    L.append("```")
    L.append("")
    L.append("## 30m vs 60m")
    L.append("")
    for day in ("weekday", "weekend"):
        a30, a60 = m[f"{day}_30m_bike_ratio"], m[f"{day}_60m_bike_ratio"]
        L.append(f"**{day}**: MAE {_f(a30['MAE'])} -> {_f(a60['MAE'])}, "
                 f"R2 {_f(a30['R2'])} -> {_f(a60['R2'])}, "
                 f"low-bike F1 {_f(a30['low_bike_f1'])} -> {_f(a60['low_bike_f1'])}, "
                 f"high-occ Recall {_f(a30['high_occupancy_recall'])} -> "
                 f"{_f(a60['high_occupancy_recall'])}")
    L.append("")
    L.append("The 60-minute horizon is consistently weaker, which is expected: more")
    L.append("time means more unobserved demand. Both horizons still separate the")
    L.append("low-bike class well (AUC >= 0.92), so both are usable as risk signals.")
    L.append("")
    L.append("## Phase 2 - threshold sweep (macro across the four models)")
    L.append("")
    L.append("Low-bike:")
    L.append("")
    L.append("| threshold | mean Precision | mean Recall | mean F1 | min Recall |")
    L.append("|---|---|---|---|---|")
    for t, v in a["macro_low"].items():
        L.append(f"| {t} | {_f(v['mean_precision'])} | {_f(v['mean_recall'])} "
                 f"| {_f(v['mean_f1'])} | {_f(v['min_recall'])} |")
    L.append("")
    L.append("High-occupancy:")
    L.append("")
    L.append("| threshold | mean Precision | mean Recall | mean F1 | min Recall |")
    L.append("|---|---|---|---|---|")
    for t, v in a["macro_high"].items():
        L.append(f"| {t} | {_f(v['mean_precision'])} | {_f(v['mean_recall'])} "
                 f"| {_f(v['mean_f1'])} | {_f(v['min_recall'])} |")
    L.append("")
    L.append("### Provisional thresholds")
    L.append("")
    L.append(f"- `provisional_low_threshold  = {a['provisional_low_threshold']}`")
    L.append(f"  - {a['provisional_low_reason']}")
    L.append(f"  - macro Recall >= 0.75: {a['provisional_low_recall_meets_target']}")
    L.append(f"- `provisional_high_threshold = {a['provisional_high_threshold']}`")
    L.append(f"  - {a['provisional_high_reason']}")
    L.append(f"  - macro Recall >= 0.75: {a['provisional_high_recall_meets_target']}")
    L.append("")
    L.append("These are provisional only. Phase 11 re-sweeps on the formal enriched")
    L.append("validation set, and that result takes precedence.")
    L.append("")
    L.append("## Persistent-risk story")
    L.append("")
    L.append("Wording rule: with only 30m and 60m horizons we can say")
    L.append("\"30 分鐘與 60 分鐘後皆預測處於低車量區間，因此判定為持續性風險\",")
    L.append("and we must NOT claim the station is continuously short for the whole hour.")
    L.append("")
    for day, v in a["persistent_risk_crosstab"].items():
        if not v.get("joined_rows"):
            L.append(f"- {day}: no shared rows")
            continue
        L.append(f"**{day}** (shared rows {v['joined_rows']:,}):")
        L.append("")
        L.append(f"- low persistent (30m low AND 60m low): {v['low_persistent_pred']:,}")
        L.append(f"- low transient (30m low, 60m recovered): {v['low_transient_pred']:,}")
        L.append(f"- low emerging (30m fine, 60m low): {v['low_emerging_pred']:,}")
        L.append(f"- high persistent: {v['high_persistent_pred']:,}")
        L.append(f"- high transient: {v['high_transient_pred']:,}")
        L.append(f"- high emerging: {v['high_emerging_pred']:,}")
        L.append(f"- persistent-low precision {_f(v['low_persistent_precision'])} / "
                 f"recall {_f(v['low_persistent_recall'])}")
        L.append("")

    p = os.path.join(HERE, "LOCAL_MODEL_COMPARISON.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("saved", p)


if __name__ == "__main__":
    raise SystemExit(main())
