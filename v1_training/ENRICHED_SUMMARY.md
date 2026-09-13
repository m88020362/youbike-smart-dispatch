# V1 ENRICHED (formal) Training Summary

Generated: 2026-09-13T10:42:52+08:00
Branch: stable-v1-training

> This file covers **SAGEMAKER_ENRICHED** (formal) models only.
> The four LOCAL_FALLBACK models are a separate, clearly-labelled
> baseline documented in LOCAL_MODEL_COMPARISON.md / OVERNIGHT_SUMMARY.md.
> LOCAL_FALLBACK is NOT the formal enriched AWS model.

current_phase: remaining_models_done
STS identity: OK 502837994229
S3 audit: OK:weekday_30m_bike_ratio
dataset range: 2026-05-04 06:00:41 -> 2026-05-29 23:00:37

## Result

P0 weekday_30m_bike_ratio: SUCCESS
weekday_60m_bike_ratio: SUCCESS
weekend_30m_bike_ratio: SUCCESS
weekend_60m_bike_ratio: SUCCESS

## weekday_30m_bike_ratio

training_backend: SAGEMAKER_ENRICHED
training_job_name: weekday-30m-bike-ratio-enr-1789263106
data: 2026-05-04 06:00:41 -> 2026-05-29 23:00:37
train_rows: 863994  validation_rows: 216855
MAE: 0.07540173731110925
RMSE: 0.11575216238952823
R2: 0.7664355529573589

event definition (FIXED, not calibrated): low < 0.2 / high > 0.8
calibrated low alert : predicted_ratio < 0.21  P=0.8283 R=0.8441 F1=0.8361
calibrated high alert: predicted_ratio > 0.62  P=0.2396 R=0.7525 F1=0.3635
artifact: s3://youbike-dispatch-502837994229-usw2/v1-enriched/output/weekday-30m-bike-ratio-enr-1789263106/output/model.tar.gz

## weekday_60m_bike_ratio

training_backend: SAGEMAKER_ENRICHED
training_job_name: weekday-60m-bike-ratio-enr-1789266895
data: 2026-05-04 06:00:41 -> 2026-05-29 22:30:35
train_rows: 839308  validation_rows: 210659
MAE: 0.10236814106498505
RMSE: 0.14499349231672307
R2: 0.6344120020089989

event definition (FIXED, not calibrated): low < 0.2 / high > 0.8
calibrated low alert : predicted_ratio < 0.25  P=0.7294 R=0.8158 F1=0.7702
calibrated high alert: predicted_ratio > 0.65  P=0.2853 R=0.4169 F1=0.3387  [Recall BELOW 0.75 target]
artifact: s3://youbike-dispatch-502837994229-usw2/v1-enriched/output/weekday-60m-bike-ratio-enr-1789266895/output/model.tar.gz

## weekend_30m_bike_ratio

training_backend: SAGEMAKER_ENRICHED
training_job_name: weekend-30m-bike-ratio-enr-1789266896
data: 2026-05-01 06:00:41 -> 2026-05-31 23:00:36
train_rows: 474887  validation_rows: 119273
MAE: 0.07759328889801391
RMSE: 0.11517324546039669
R2: 0.7662583247646336

event definition (FIXED, not calibrated): low < 0.2 / high > 0.8
calibrated low alert : predicted_ratio < 0.22  P=0.7860 R=0.8410 F1=0.8126
calibrated high alert: predicted_ratio > 0.65  P=0.3195 R=0.7800 F1=0.4533
artifact: s3://youbike-dispatch-502837994229-usw2/v1-enriched/output/weekend-30m-bike-ratio-enr-1789266896/output/model.tar.gz

## weekend_60m_bike_ratio

training_backend: SAGEMAKER_ENRICHED
training_job_name: weekend-60m-bike-ratio-enr-1789267182
data: 2026-05-01 06:00:41 -> 2026-05-31 22:30:33
train_rows: 461009  validation_rows: 116175
MAE: 0.10551492011932476
RMSE: 0.14603864162867694
R2: 0.6241852077548284

event definition (FIXED, not calibrated): low < 0.2 / high > 0.8
calibrated low alert : predicted_ratio < 0.25  P=0.6965 R=0.7937 F1=0.7419
calibrated high alert: predicted_ratio > 0.55  P=0.1804 R=0.7523 F1=0.2911
artifact: s3://youbike-dispatch-502837994229-usw2/v1-enriched/output/weekend-60m-bike-ratio-enr-1789267182/output/model.tar.gz

## Errors / blockers

- weekday_30m_bike_ratio: EndpointConnectionError  Could not connect to the endpoint URL: "https://api.sagemaker.us-west-2.amazonaws.com/"
