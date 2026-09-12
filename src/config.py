# -*- coding: utf-8 -*-
"""Centralized configuration for the YouBike predictive dispatch MVP.

All tunable thresholds, interval boundaries, and paths live here so they are
easy to adjust and never hard-coded across modules.
"""

import os

# --- Paths -----------------------------------------------------------------
# Project root = parent of this src/ directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# First-version vertical slice uses one small, date-continuous CSV.
DEFAULT_CSV_NAME = "新北AWS黑克松競賽0329-31.csv 的副本.csv"
DEFAULT_CSV = os.path.join(DATASET_DIR, DEFAULT_CSV_NAME)

# --- Data loading ----------------------------------------------------------
DATA_ENCODING = "cp950"

# Required columns in the raw CSV (original Chinese headers).
REQUIRED_COLUMNS = [
    "日期",
    "城市",
    "行政區",
    "場站名稱",
    "總車柱數",
    "可借車數",
    "可還位數",
    "經度",
    "緯度",
]

# Mapping from raw Chinese headers to internal English column names.
COLUMN_RENAME = {
    "日期": "timestamp",
    "城市": "city",
    "行政區": "district",
    "場站名稱": "station",
    "總車柱數": "total_docks",
    "可借車數": "available_bikes",
    "可還位數": "available_docks",
    "經度": "lon",
    "緯度": "lat",
}

# Internal column names (post-rename), for downstream reference.
COL_TIMESTAMP = "timestamp"
COL_CITY = "city"
COL_DISTRICT = "district"
COL_STATION = "station"
COL_TOTAL_DOCKS = "total_docks"
COL_AVAILABLE_BIKES = "available_bikes"
COL_AVAILABLE_DOCKS = "available_docks"
COL_LON = "lon"
COL_LAT = "lat"

# Columns that must be numeric after loading.
NUMERIC_COLUMNS = [
    COL_TOTAL_DOCKS,
    COL_AVAILABLE_BIKES,
    COL_AVAILABLE_DOCKS,
    COL_LON,
    COL_LAT,
]

# Key columns that must not be NaN for a row to be usable.
CRITICAL_COLUMNS = [
    COL_TIMESTAMP,
    COL_STATION,
    COL_TOTAL_DOCKS,
    COL_AVAILABLE_BIKES,
    COL_AVAILABLE_DOCKS,
]

# --- 30-minute target ------------------------------------------------------
# Only observation pairs spaced within this window count as a 30-minute target.
# ~60-minute gaps are excluded.
TARGET_MIN_MINUTES = 25
TARGET_MAX_MINUTES = 35

# Future-state risk thresholds (inclusive).
SHORTAGE_THRESHOLD = 2  # future available_bikes <= this  -> shortage risk
FULL_THRESHOLD = 2      # future available_docks <= this  -> full/no-return risk

# Internal target/derived column names.
COL_FUTURE_TIMESTAMP = "future_timestamp"
COL_FUTURE_BIKES = "future_available_bikes"
COL_FUTURE_DOCKS = "future_available_docks"
COL_DELTA_MIN = "delta_min"
COL_SHORTAGE_TARGET = "shortage_target"
COL_FULL_TARGET = "full_target"

# --- Feature engineering (Task 2, R3) --------------------------------------
# Time features derived from `timestamp`.
COL_HOUR = "hour"
COL_WEEKDAY = "weekday"
COL_IS_WEEKEND = "is_weekend"

# Encoded station / district identity columns (deterministic label encoding).
COL_STATION_ID = "station_id"
COL_DISTRICT_ID = "district_id"

# Inventory ratio features.
COL_BIKE_RATIO = "bike_ratio"
COL_DOCK_RATIO = "dock_ratio"

# Lag features (same station, shift(1)).
COL_PREV_BIKES = "prev_available_bikes"
COL_BIKE_CHANGE = "bike_change"
COL_DOCK_CHANGE = "dock_change"

# Value used to fill per-station first-row lag NaNs (documented default).
LAG_FILL_VALUE = 0

# Ordered list of feature columns that make up X. Downstream train.py MUST use
# this ordering (persisted via feature_meta) so features stay reproducible.
FEATURE_COLUMNS = [
    # time
    COL_HOUR,
    COL_WEEKDAY,
    COL_IS_WEEKEND,
    # station identity / static
    COL_STATION_ID,
    COL_DISTRICT_ID,
    COL_TOTAL_DOCKS,
    COL_LON,
    COL_LAT,
    # current inventory
    COL_AVAILABLE_BIKES,
    COL_AVAILABLE_DOCKS,
    COL_BIKE_RATIO,
    COL_DOCK_RATIO,
    # lag / temporal change
    COL_PREV_BIKES,
    COL_BIKE_CHANGE,
    COL_DOCK_CHANGE,
]

# --- Model / risk (used by later tasks; kept here for a single source) -----
TRAIN_FRACTION = 0.7
RISK_LOW_MAX = 0.3
RISK_HIGH_MIN = 0.6

# --- Intervention (used by later tasks) ------------------------------------
MAX_NEIGHBOR_KM = 1.0
REWARD_TIERS = [5, 10, 15]
INTERVENTION_URGENT_PROB = 0.85  # max(shortage_prob, full_prob) >= this => urgency "high"

# --- Prediction backend (Task 8 / v0 cloud demo) ---------------------------
# Which predictor produces shortage_prob / full_prob.
#   "local"     -> load models from MODELS_DIR and predict in-process (default,
#                  the stable v0 behaviour; works with no AWS access at all).
#   "sagemaker" -> call the deployed real-time endpoints via sagemaker-runtime.
# Feature engineering, risk thresholds, intervention rules and the Streamlit UI
# are IDENTICAL for both backends; only the probability source changes.
BACKEND_LOCAL = "local"
BACKEND_SAGEMAKER = "sagemaker"
PREDICTION_BACKEND = BACKEND_LOCAL

# Deployed SageMaker real-time endpoint names (created by deploy/create_models.py
# + deploy/deploy_endpoints.py). Only used when PREDICTION_BACKEND == "sagemaker".
AWS_REGION = "us-west-2"
SAGEMAKER_ENDPOINT_SHORTAGE = "youbike-shortage-v1-endpoint"
SAGEMAKER_ENDPOINT_FULL = "youbike-full-v1-endpoint"

# --- Optional Bedrock explain layer (Task C) -------------------------------
# Bedrock is a PURELY OPTIONAL natural-language wrapper around an ALREADY
# FINISHED deterministic decision. It never predicts, classifies risk, ranks
# donor stations, or chooses an action, and it is never called automatically.
BEDROCK_ENABLED = False           # master off-switch; UI button also required
BEDROCK_REGION = "us-west-2"
# NOTE: confirm this model id is enabled in the workshop account before relying
# on it. Any failure (AccessDenied / throttle / timeout / wrong id) falls back
# to the deterministic template, so a wrong value cannot break the demo.
BEDROCK_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
BEDROCK_MAX_TOKENS = 300
BEDROCK_TIMEOUT_SECONDS = 8
# Competition constraint: strictly fewer than 1 request per second.
BEDROCK_MIN_INTERVAL_SECONDS = 1.1
