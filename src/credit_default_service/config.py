"""Shared project configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "api_requests.jsonl"

TARGET_COLUMN = "default.payment.next.month"
ID_COLUMN = "ID"

RAW_CATEGORICAL_FEATURES = ["SEX", "EDUCATION", "MARRIAGE"]
RAW_NUMERIC_FEATURES = [
    "LIMIT_BAL",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
]
FEATURE_COLUMNS = RAW_NUMERIC_FEATURES[:1] + RAW_CATEGORICAL_FEATURES + RAW_NUMERIC_FEATURES[1:]

ENGINEERED_NUMERIC_FEATURES = [
    "avg_payment_status",
    "max_payment_status",
    "late_payment_count",
    "avg_bill_amt",
    "max_bill_amt",
    "bill_amt_trend",
    "avg_pay_amt",
    "max_pay_amt",
    "pay_amt_trend",
    "payment_ratio",
    "credit_utilization",
]
ENGINEERED_CATEGORICAL_FEATURES = ["age_group", "credit_limit_group"]

CATEGORICAL_FEATURES = RAW_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_NUMERIC_FEATURES + ENGINEERED_CATEGORICAL_FEATURES

FEATURE_RULES = {
    "LIMIT_BAL": {"min": 0, "max": 1_000_000},
    "SEX": {"allowed": [1, 2]},
    "EDUCATION": {"allowed": [0, 1, 2, 3, 4, 5, 6]},
    "MARRIAGE": {"allowed": [0, 1, 2, 3]},
    "AGE": {"min": 18, "max": 100},
    "PAY_0": {"min": -2, "max": 9},
    "PAY_2": {"min": -2, "max": 9},
    "PAY_3": {"min": -2, "max": 9},
    "PAY_4": {"min": -2, "max": 9},
    "PAY_5": {"min": -2, "max": 9},
    "PAY_6": {"min": -2, "max": 9},
    "PAY_AMT1": {"min": 0},
    "PAY_AMT2": {"min": 0},
    "PAY_AMT3": {"min": 0},
    "PAY_AMT4": {"min": 0},
    "PAY_AMT5": {"min": 0},
    "PAY_AMT6": {"min": 0},
}

MODEL_FILENAMES = {
    "v1": "credit_default_v1.joblib",
    "v2": "credit_default_v2.joblib",
}

DEFAULT_THRESHOLD = 0.5
