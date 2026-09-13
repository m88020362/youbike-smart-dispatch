# -*- coding: utf-8 -*-
"""LightGBM regression training + regression / operational metrics.

Fixed conservative hyperparameters (NO tuning tonight, per mission brief).
Operational thresholds are fixed at 0.20 / 0.80 and represent a SERVICE RISK
buffer after normalizing by station capacity -- they are NOT claims that 0.20
means physically empty or 0.80 means physically full.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

LOW_BIKE_THRESHOLD = 0.20
HIGH_OCCUPANCY_THRESHOLD = 0.80

# Conservative, fixed. No tuning / CV / Optuna tonight.
LGBM_PARAMS = dict(
    objective="regression",
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
EARLY_STOPPING_ROUNDS = 50


def train_lightgbm(X_train, y_train, X_valid, y_valid):
    """Fit LGBMRegressor with validation early stopping.

    Handles the lightgbm>=4 callback API and falls back to the older
    early_stopping_rounds kwarg if needed. Model TYPE is never changed.
    """
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    use_valid = X_valid is not None and len(X_valid) > 0

    if use_valid:
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            )
        except TypeError:
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l1",
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False,
            )
    else:
        model.fit(X_train, y_train)
    return model


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray,
                    auc_score: np.ndarray) -> Dict[str, Optional[float]]:
    """Precision / Recall / F1 / ROC-AUC with a single-class guard.

    A single-class validation set yields AUC = None (reported as N/A) instead of
    raising, so a degenerate split can never crash the run.
    """
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 roc_auc_score)

    out: Dict[str, Optional[float]] = {
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }
    if len(np.unique(actual)) < 2:
        out["auc"] = None
        out["auc_note"] = "N/A - validation actual events are a single class"
    else:
        out["auc"] = float(roc_auc_score(actual, auc_score))
    return out


def evaluate(y_true: np.ndarray, y_pred_raw: np.ndarray) -> Dict:
    """Regression + operational metrics.

    Clipping is applied ONLY for prediction interpretation, as instructed.
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.clip(np.asarray(y_pred_raw, dtype="float64"), 0.0, 1.0)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = float(r2_score(y_true, y_pred))

    low = _binary_metrics(
        (y_true < LOW_BIKE_THRESHOLD).astype(int),
        (y_pred < LOW_BIKE_THRESHOLD).astype(int),
        1.0 - y_pred,           # higher score => more likely low-bike
    )
    high = _binary_metrics(
        (y_true > HIGH_OCCUPANCY_THRESHOLD).astype(int),
        (y_pred > HIGH_OCCUPANCY_THRESHOLD).astype(int),
        y_pred,                 # higher score => more likely high-occupancy
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "threshold_low": LOW_BIKE_THRESHOLD,
        "threshold_high": HIGH_OCCUPANCY_THRESHOLD,
        "low_bike_precision": low["precision"],
        "low_bike_recall": low["recall"],
        "low_bike_f1": low["f1"],
        "low_bike_auc": low["auc"],
        "low_bike_auc_note": low.get("auc_note"),
        "low_bike_actual_event_rate": float((y_true < LOW_BIKE_THRESHOLD).mean()),
        "high_occupancy_precision": high["precision"],
        "high_occupancy_recall": high["recall"],
        "high_occupancy_f1": high["f1"],
        "high_occupancy_auc": high["auc"],
        "high_occupancy_auc_note": high.get("auc_note"),
        "high_occupancy_actual_event_rate": float(
            (y_true > HIGH_OCCUPANCY_THRESHOLD).mean()
        ),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
        "prediction_all_finite": bool(np.isfinite(y_pred).all()),
        "n_evaluated": int(len(y_true)),
    }
