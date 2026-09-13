# LOCAL_FALLBACK Model Comparison + Provisional Thresholds

Generated: 2026-09-13T07:01:17+08:00

> **training_backend = LOCAL_FALLBACK.** These four models were trained
> on the local 9-column June CSV with 8 features. They are NOT the formal
> enriched AWS models. The enriched S3 dataset (weather / nearest-distance /
> is_peak) has not been trained yet; `SAGEMAKER_ENRICHED` remains pending.

## Phase 1 - four model metrics

| model | data range | train | valid | MAE | RMSE | R2 |
|---|---|---|---|---|---|---|
| weekday_30m_bike_ratio | 2026-06-01 -> 2026-06-30 | 1,306,659 | 328,499 | 0.0570 | 0.0983 | 0.8176 |
| weekday_60m_bike_ratio | 2026-06-01 -> 2026-06-30 | 1,300,443 | 326,915 | 0.0803 | 0.1257 | 0.7018 |
| weekend_30m_bike_ratio | 2026-06-06 -> 2026-06-28 | 472,568 | 118,712 | 0.0558 | 0.0912 | 0.8290 |
| weekend_60m_bike_ratio | 2026-06-06 -> 2026-06-28 | 467,906 | 117,150 | 0.0788 | 0.1180 | 0.7143 |

Low-bike @ 0.20 (as trained):

| model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| weekday_30m_bike_ratio | 0.8588 | 0.8461 | 0.8524 | 0.9547 |
| weekday_60m_bike_ratio | 0.8240 | 0.7321 | 0.7753 | 0.9223 |
| weekend_30m_bike_ratio | 0.8486 | 0.8167 | 0.8323 | 0.9573 |
| weekend_60m_bike_ratio | 0.8052 | 0.6926 | 0.7447 | 0.9248 |

High-occupancy @ 0.80 (as trained):

| model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| weekday_30m_bike_ratio | 0.7690 | 0.3612 | 0.4915 | 0.9412 |
| weekday_60m_bike_ratio | 0.7739 | 0.2124 | 0.3333 | 0.8897 |
| weekend_30m_bike_ratio | 0.7358 | 0.3199 | 0.4459 | 0.9517 |
| weekend_60m_bike_ratio | 0.6933 | 0.1466 | 0.2421 | 0.9044 |

Feature list (identical for all four):

```
current_available_bikes, current_available_docks, total_docks, current_bike_ratio, hour, weekday, lon, lat
```

## 30m vs 60m

**weekday**: MAE 0.0570 -> 0.0803, R2 0.8176 -> 0.7018, low-bike F1 0.8524 -> 0.7753, high-occ Recall 0.3612 -> 0.2124
**weekend**: MAE 0.0558 -> 0.0788, R2 0.8290 -> 0.7143, low-bike F1 0.8323 -> 0.7447, high-occ Recall 0.3199 -> 0.1466

The 60-minute horizon is consistently weaker, which is expected: more
time means more unobserved demand. Both horizons still separate the
low-bike class well (AUC >= 0.92), so both are usable as risk signals.

## Phase 2 - threshold sweep (macro across the four models)

Low-bike:

| threshold | mean Precision | mean Recall | mean F1 | min Recall |
|---|---|---|---|---|
| 0.10 | 0.8490 | 0.5438 | 0.6586 | 0.4441 |
| 0.15 | 0.8396 | 0.6751 | 0.7460 | 0.5685 |
| 0.20 | 0.8342 | 0.7719 | 0.8012 | 0.6926 |
| 0.25 | 0.8508 | 0.8250 | 0.8375 | 0.7680 |

High-occupancy:

| threshold | mean Precision | mean Recall | mean F1 | min Recall |
|---|---|---|---|---|
| 0.65 | 0.8111 | 0.6057 | 0.6902 | 0.5023 |
| 0.70 | 0.7835 | 0.5046 | 0.6088 | 0.3803 |
| 0.75 | 0.7616 | 0.3909 | 0.5082 | 0.2582 |
| 0.80 | 0.7430 | 0.2600 | 0.3782 | 0.1466 |
| 0.85 | 0.7458 | 0.1454 | 0.2393 | 0.0741 |

### Provisional thresholds

- `provisional_low_threshold  = 0.25`
  - highest macro F1 (0.8375) among candidates with macro Recall >= 0.75
  - macro Recall >= 0.75: True
- `provisional_high_threshold = 0.65`
  - NO threshold reached macro Recall >= 0.75; picked the highest macro F1 (0.6902) instead
  - macro Recall >= 0.75: False

These are provisional only. Phase 11 re-sweeps on the formal enriched
validation set, and that result takes precedence.

## Persistent-risk story

Wording rule: with only 30m and 60m horizons we can say
"30 分鐘與 60 分鐘後皆預測處於低車量區間，因此判定為持續性風險",
and we must NOT claim the station is continuously short for the whole hour.

**weekday** (shared rows 325,361):

- low persistent (30m low AND 60m low): 106,521
- low transient (30m low, 60m recovered): 6,272
- low emerging (30m fine, 60m low): 905
- high persistent: 24,020
- high transient: 8,077
- high emerging: 149
- persistent-low precision 0.8145 / recall 0.8725

**weekend** (shared rows 117,150):

- low persistent (30m low AND 60m low): 33,227
- low transient (30m low, 60m recovered): 2,650
- low emerging (30m fine, 60m low): 532
- high persistent: 8,696
- high transient: 2,820
- high emerging: 123
- persistent-low precision 0.8021 / recall 0.8498

## Methodology caveat (read before locking any threshold)

The sweep above uses ONE shared value for both the event definition and
the alert rule, as specified. That makes F1 a monotone function of event
prevalence, so max-F1 always lands on the most permissive edge of the grid.
Evidence (weekday_30m):

| threshold | LOW prevalence | LOW F1 | HIGH prevalence | HIGH F1 | HIGH AUC |
|---|---|---|---|---|---|
| 0.10 | 0.1569 | 0.7403 | - | - | - |
| 0.15 | 0.2217 | 0.8148 | - | - | - |
| 0.20 | 0.2791 | 0.8524 | - | - | - |
| 0.25 | 0.3473 | 0.8773 | - | - | - |
| 0.65 | - | - | 0.1171 | 0.7663 | 0.9555 |
| 0.70 | - | - | 0.0793 | 0.6988 | 0.9522 |
| 0.75 | - | - | 0.0500 | 0.6159 | 0.9469 |
| 0.80 | - | - | 0.0309 | 0.4915 | 0.9412 |
| 0.85 | - | - | 0.0188 | 0.3438 | 0.9357 |

HIGH ROC-AUC is nearly flat (0.9555 -> 0.9357) while recall collapses
(0.71 -> 0.22). Ranking quality is therefore NOT the limiter; the model is a
conservative point estimate that rarely predicts extreme values, and true
>0.80 events are only ~3% of rows.

### Decoupled alert calibration (recommended alternative)

Keep the operational event definition fixed at 0.20 / 0.80 and tune only the
alert trigger applied to the prediction:

| side | event | recommended alert | mean P | mean R | mean F1 | min R |
|---|---|---|---|---|---|---|
| low | y_true < 0.2 | y_pred < 0.21 | 0.8067 | 0.8017 | 0.8037 | 0.7280 |
| high | y_true > 0.8 | y_pred > 0.62 | 0.2001 | 0.7651 | 0.3165 | 0.6478 |

Reading this honestly:

- LOW works well. Event `<0.20` with alert `<0.21` reaches mean Recall 0.8017
  at mean Precision 0.8067. The operational meaning of 0.20 is preserved and
  the Recall target is met, so the low-bike definition needs no change.
- HIGH has no clean answer. True `>0.80` events are only ~3% of rows.
  Reaching Recall 0.7651 forces mean Precision to 0.2001, i.e. about four
  false alerts per real one. Three honest options:

  1. keep 0.80, accept low recall (macro P 0.7430 / R 0.2600)
  2. keep 0.80, accept low precision (macro P 0.2001 / R 0.7651)
  3. lower the operational definition toward 0.65, where events are ~12%
     prevalent and both sides are reasonable (weekday_30m P 0.8374 / R 0.7062)

  This is an operations policy call about tolerable false dispatches, not
  something the data settles alone. The mechanical max-F1 rule picked option 3
  (0.65); it is recorded as `provisional_high_threshold` but must be confirmed
  against the formal enriched validation in Phase 11 before being treated as
  final.

Both provisional values remain LOCAL_FALLBACK-derived and non-final.
