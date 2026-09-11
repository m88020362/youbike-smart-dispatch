# -*- coding: utf-8 -*-
"""Model training for the YouBike predictive dispatch MVP (Task 3, R4).

Pipeline entry:
    data_loader.load_clean()
      -> features.build_targets()
        -> features.build_features()
          -> train.train()

Trains, for each risk label (shortage / full):
  A) a simple, interpretable baseline (inventory-ratio-based probability), and
  B) an XGBoost classifier (XGBClassifier), taking predict_proba's positive
     class as the risk probability.

Key design constraints (R4):
  * Train/test split is by TIME only (sort by timestamp, first TRAIN_FRACTION
    is train, the rest is test). No random split -> no temporal leakage.
  * Class imbalance is handled via scale_pos_weight, with attention to recall
    on the rare high-risk positive class.
  * Emits understandable metrics: precision, recall, F1, ROC-AUC (when both
    classes are present in test), and confusion matrix.
  * No deep learning / GNN / RL.
  * On completion, persists models + feature meta to MODELS_DIR and metrics +
    plots to OUTPUTS_DIR.

This module never modifies the raw CSV.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from . import config, data_loader, features


# --- Baseline --------------------------------------------------------------
def _baseline_shortage_proba(X: pd.DataFrame) -> np.ndarray:
    """Interpretable shortage baseline.

    Intuition: the emptier a station is now (low bike_ratio), the more likely it
    runs short in 30 minutes. Map bike_ratio -> probability with p = 1 - ratio,
    clipped to [0, 1]. No training required; purely a transparent reference.
    """
    ratio = X[config.COL_BIKE_RATIO].to_numpy(dtype=float)
    return np.clip(1.0 - ratio, 0.0, 1.0)


def _baseline_full_proba(X: pd.DataFrame) -> np.ndarray:
    """Interpretable full/no-return baseline.

    Intuition: the fuller a station is now (low dock_ratio -> few free docks),
    the more likely it becomes full. Map dock_ratio -> probability with
    p = 1 - dock_ratio, clipped to [0, 1].
    """
    ratio = X[config.COL_DOCK_RATIO].to_numpy(dtype=float)
    return np.clip(1.0 - ratio, 0.0, 1.0)


# --- Metrics ---------------------------------------------------------------
def _compute_scale_pos_weight(y_train: pd.Series) -> float:
    """scale_pos_weight = (#negatives / #positives) to counter imbalance.

    Falls back to 1.0 when there are no positives (degenerate) to keep training
    stable. Higher weight pushes the model toward recalling the rare positive
    (high-risk) class.
    """
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def _metrics_from_proba(
    y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> Dict:
    """Compute precision/recall/F1/ROC-AUC/confusion-matrix from probabilities.

    ROC-AUC is only meaningful when both classes appear in y_true; otherwise it
    is reported as None. All probability arrays are validated to lie in [0, 1].
    """
    y_true_arr = y_true.to_numpy(dtype=int)
    proba = np.asarray(proba, dtype=float)

    # Guard: probabilities must be valid.
    if proba.size and (proba.min() < 0.0 or proba.max() > 1.0):
        raise ValueError(
            f"probabilities out of [0,1]: min={proba.min()}, max={proba.max()}"
        )

    y_pred = (proba >= threshold).astype(int)

    both_classes = len(np.unique(y_true_arr)) == 2
    try:
        auc = float(roc_auc_score(y_true_arr, proba)) if both_classes else None
    except ValueError:
        auc = None

    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": threshold,
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "support": {
            "n": int(len(y_true_arr)),
            "positives": int((y_true_arr == 1).sum()),
            "negatives": int((y_true_arr == 0).sum()),
        },
    }


# --- Plots -----------------------------------------------------------------
def _plot_confusion_matrix(cm_dict: Dict, title: str, out_path: str) -> None:
    """Save a 2x2 confusion-matrix heatmap for a single model/label."""
    cm = np.array(
        [
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["neg", "pos"])
    ax.set_yticks([0, 1], labels=["neg", "pos"])
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > vmax / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_roc(y_true: pd.Series, proba: np.ndarray, title: str, out_path: str) -> bool:
    """Save a ROC curve. Returns False (and skips) if only one class present."""
    y_true_arr = y_true.to_numpy(dtype=int)
    if len(np.unique(y_true_arr)) < 2:
        return False
    fpr, tpr, _ = roc_curve(y_true_arr, proba)
    auc = roc_auc_score(y_true_arr, proba)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# --- XGBoost ---------------------------------------------------------------
def _make_xgb(scale_pos_weight: float) -> XGBClassifier:
    """Construct an XGBClassifier tuned for a small, imbalanced tabular task."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )


def _time_split_index(timestamp: pd.Series, train_fraction: float) -> np.ndarray:
    """Return a boolean mask (True=train) for a TIME-based split.

    Rows are ordered by timestamp; the earliest `train_fraction` fraction is the
    training set, the remainder is test. This never shuffles, so no future
    information leaks into training.
    """
    order = np.argsort(timestamp.to_numpy(), kind="stable")
    n = len(order)
    n_train = int(np.floor(n * train_fraction))
    train_positions = set(order[:n_train].tolist())
    mask = np.array([i in train_positions for i in range(n)], dtype=bool)
    return mask


def _train_one_label(
    label_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    train_mask: np.ndarray,
    baseline_fn,
    models_dir: str,
    outputs_dir: str,
) -> Dict:
    """Train baseline + XGBoost for a single label and collect metrics/artifacts."""
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # --- Baseline (no fit needed) --
    base_proba_test = baseline_fn(X_test)
    baseline_metrics = _metrics_from_proba(y_test, base_proba_test)

    # --- XGBoost --
    spw = _compute_scale_pos_weight(y_train)
    model = _make_xgb(spw)
    model.fit(X_train, y_train)

    xgb_proba_test = model.predict_proba(X_test)[:, 1]
    xgb_metrics = _metrics_from_proba(y_test, xgb_proba_test)
    xgb_metrics["scale_pos_weight"] = spw

    # Persist model.
    model_path = os.path.join(models_dir, f"{label_name}_xgb.json")
    model.save_model(model_path)

    # Plots (XGBoost).
    cm_png = os.path.join(outputs_dir, f"{label_name}_confusion_matrix.png")
    _plot_confusion_matrix(
        xgb_metrics["confusion_matrix"],
        f"{label_name} XGBoost confusion matrix",
        cm_png,
    )
    roc_png = os.path.join(outputs_dir, f"{label_name}_roc.png")
    roc_drawn = _plot_roc(
        y_test, xgb_proba_test, f"{label_name} XGBoost ROC", roc_png
    )

    # Feature importance (interpretability; ordered by config.FEATURE_COLUMNS).
    importances = model.feature_importances_.tolist()
    feature_importance = {
        col: float(imp) for col, imp in zip(config.FEATURE_COLUMNS, importances)
    }

    return {
        "label": label_name,
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
        "train_positive_rate": float((y_train == 1).mean()) if len(y_train) else 0.0,
        "test_positive_rate": float((y_test == 1).mean()) if len(y_test) else 0.0,
        "baseline": baseline_metrics,
        "xgboost": xgb_metrics,
        "feature_importance": feature_importance,
        "artifacts": {
            "model": os.path.relpath(model_path, config.PROJECT_ROOT),
            "confusion_matrix_png": os.path.relpath(cm_png, config.PROJECT_ROOT),
            "roc_png": (
                os.path.relpath(roc_png, config.PROJECT_ROOT) if roc_drawn else None
            ),
        },
    }


def train(
    csv_path: str | None = None,
    train_fraction: float | None = None,
) -> Dict:
    """Run the full training pipeline and persist all artifacts.

    Args:
        csv_path: Optional CSV override (defaults to config.DEFAULT_CSV).
        train_fraction: Optional time-split fraction (defaults to
            config.TRAIN_FRACTION).

    Returns:
        The metrics dictionary that is also written to
        outputs/metrics.json.

    Side effects (per R4 criterion 6):
        models/shortage_xgb.json, models/full_xgb.json, models/feature_meta.json
        outputs/metrics.json, outputs/*_confusion_matrix.png, outputs/*_roc.png
    """
    train_fraction = (
        config.TRAIN_FRACTION if train_fraction is None else train_fraction
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    # --- Data pipeline ------------------------------------------------------
    df = data_loader.load_clean(csv_path)
    targets = features.build_targets(df)
    bundle = features.build_features(targets)

    X = bundle.X
    if len(X) == 0:
        raise ValueError("No training samples produced (empty feature matrix).")

    # --- TIME-based split (no random split) ---------------------------------
    train_mask = _time_split_index(bundle.timestamp, train_fraction)
    ts_sorted = bundle.timestamp.sort_values()
    n_train = int(train_mask.sum())
    # Boundary timestamp facts for reporting/validation.
    train_ts = bundle.timestamp[train_mask]
    test_ts = bundle.timestamp[~train_mask]
    split_info = {
        "train_fraction": train_fraction,
        "n_total": int(len(X)),
        "n_train": n_train,
        "n_test": int(len(X) - n_train),
        "train_time_min": str(train_ts.min()) if len(train_ts) else None,
        "train_time_max": str(train_ts.max()) if len(train_ts) else None,
        "test_time_min": str(test_ts.min()) if len(test_ts) else None,
        "test_time_max": str(test_ts.max()) if len(test_ts) else None,
        "boundary_ok": (
            bool(train_ts.max() <= test_ts.min())
            if len(train_ts) and len(test_ts)
            else None
        ),
    }

    # --- Train each label ---------------------------------------------------
    shortage_result = _train_one_label(
        "shortage",
        X,
        bundle.y_shortage,
        train_mask,
        _baseline_shortage_proba,
        config.MODELS_DIR,
        config.OUTPUTS_DIR,
    )
    full_result = _train_one_label(
        "full",
        X,
        bundle.y_full,
        train_mask,
        _baseline_full_proba,
        config.MODELS_DIR,
        config.OUTPUTS_DIR,
    )

    # --- Persist feature meta (column order + encoders) ---------------------
    feature_meta = {
        "feature_columns": list(config.FEATURE_COLUMNS),
        "station_encoding": bundle.station_encoding,
        "district_encoding": bundle.district_encoding,
        "config": {
            "shortage_threshold": config.SHORTAGE_THRESHOLD,
            "full_threshold": config.FULL_THRESHOLD,
            "target_min_minutes": config.TARGET_MIN_MINUTES,
            "target_max_minutes": config.TARGET_MAX_MINUTES,
            "train_fraction": train_fraction,
        },
    }
    feature_meta_path = os.path.join(config.MODELS_DIR, "feature_meta.json")
    with open(feature_meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)

    # --- Persist metrics ----------------------------------------------------
    metrics = {
        "csv": os.path.relpath(csv_path or config.DEFAULT_CSV, config.PROJECT_ROOT),
        "split": split_info,
        "shortage": shortage_result,
        "full": full_result,
    }
    metrics_path = os.path.join(config.OUTPUTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    result = train()
    s = result["shortage"]
    fl = result["full"]
    print("=" * 64)
    print("Training complete.")
    print(f"Split: {result['split']}")
    for name, res in (("shortage", s), ("full", fl)):
        print("-" * 64)
        print(f"[{name}] n_train={res['n_train']} n_test={res['n_test']} "
              f"test_pos_rate={res['test_positive_rate']:.4f}")
        b = res["baseline"]
        x = res["xgboost"]
        print(f"  baseline: P={b['precision']:.3f} R={b['recall']:.3f} "
              f"F1={b['f1']:.3f} AUC={b['roc_auc']}")
        print(f"  xgboost : P={x['precision']:.3f} R={x['recall']:.3f} "
              f"F1={x['f1']:.3f} AUC={x['roc_auc']} spw={x['scale_pos_weight']:.2f}")
