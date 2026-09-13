# V1 ENRICHED (formal) Training Summary

Generated: 2026-09-13T09:34:57+08:00
Branch: stable-v1-training

> This file covers **SAGEMAKER_ENRICHED** (formal) models only.
> The four LOCAL_FALLBACK models are a separate, clearly-labelled
> baseline documented in LOCAL_MODEL_COMPARISON.md / OVERNIGHT_SUMMARY.md.
> LOCAL_FALLBACK is NOT the formal enriched AWS model.

current_phase: done
STS identity: OK 502837994229
S3 audit: OK:weekday_30m_bike_ratio
dataset range: 2026-05-04 06:00:41 -> 2026-05-29 23:00:37

## Result

P0 weekday_30m_bike_ratio: SUCCESS
weekday_60m_bike_ratio: NOT RUN
weekend_30m_bike_ratio: NOT RUN
weekend_60m_bike_ratio: NOT RUN

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

## Errors / blockers

- weekday_30m_bike_ratio: EndpointConnectionError  Could not connect to the endpoint URL: "https://api.sagemaker.us-west-2.amazonaws.com/"
