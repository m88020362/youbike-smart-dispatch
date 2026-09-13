# V1 Overnight Training Summary

Generated: 2026-09-13T01:40:09+08:00
Branch: stable-v1-training

## Result

P0 weekday_30m_bike_ratio: SUCCESS
weekday_60m_bike_ratio: SUCCESS
weekend_30m_bike_ratio: SUCCESS
weekend_60m_bike_ratio: SUCCESS

## weekday_30m_bike_ratio

training_backend: LOCAL_FALLBACK
使用資料期間: 2026-06-01 00:00:00 -> 2026-06-30 23:30:00
train_rows: 1306659  validation_rows: 328499

MAE:  0.0570
RMSE: 0.0983
R2:   0.8176

Low-bike (< 0.20):
  Precision 0.8588
  Recall    0.8461
  F1        0.8524
  AUC       0.9547

High-occupancy (> 0.80):
  Precision 0.7690
  Recall    0.3612
  F1        0.4915
  AUC       0.9412

Artifact: C:\Users\user\Desktop\hackathon\v1_training\local_fallback\weekday_30m_bike_ratio\weekday_30m_bike_ratio_lgbm.txt

## weekday_60m_bike_ratio

training_backend: LOCAL_FALLBACK
使用資料期間: 2026-06-01 00:00:00 -> 2026-06-30 23:30:00
train_rows: 1300443  validation_rows: 326915

MAE:  0.0803
RMSE: 0.1257
R2:   0.7018

Low-bike (< 0.20):
  Precision 0.8240
  Recall    0.7321
  F1        0.7753
  AUC       0.9223

High-occupancy (> 0.80):
  Precision 0.7739
  Recall    0.2124
  F1        0.3333
  AUC       0.8897

Artifact: C:\Users\user\Desktop\hackathon\v1_training\local_fallback\weekday_60m_bike_ratio\weekday_60m_bike_ratio_lgbm.txt

## weekend_30m_bike_ratio

training_backend: LOCAL_FALLBACK
使用資料期間: 2026-06-06 00:00:00 -> 2026-06-28 23:30:00
train_rows: 472568  validation_rows: 118712

MAE:  0.0558
RMSE: 0.0912
R2:   0.8290

Low-bike (< 0.20):
  Precision 0.8486
  Recall    0.8167
  F1        0.8323
  AUC       0.9573

High-occupancy (> 0.80):
  Precision 0.7358
  Recall    0.3199
  F1        0.4459
  AUC       0.9517

Artifact: C:\Users\user\Desktop\hackathon\v1_training\local_fallback\weekend_30m_bike_ratio\weekend_30m_bike_ratio_lgbm.txt

## weekend_60m_bike_ratio

training_backend: LOCAL_FALLBACK
使用資料期間: 2026-06-06 00:00:00 -> 2026-06-28 23:30:00
train_rows: 467906  validation_rows: 117150

MAE:  0.0788
RMSE: 0.1180
R2:   0.7143

Low-bike (< 0.20):
  Precision 0.8052
  Recall    0.6926
  F1        0.7447
  AUC       0.9248

High-occupancy (> 0.80):
  Precision 0.6933
  Recall    0.1466
  F1        0.2421
  AUC       0.9044

Artifact: C:\Users\user\Desktop\hackathon\v1_training\local_fallback\weekend_60m_bike_ratio\weekend_60m_bike_ratio_lgbm.txt

## Errors / blockers

- SageMaker Training Job not submitted: the agent shell had no AWS credentials. Run v1_training/overnight_train.py in a credentialed PowerShell to execute the cloud path. All four LOCAL_FALLBACK models already succeeded.
