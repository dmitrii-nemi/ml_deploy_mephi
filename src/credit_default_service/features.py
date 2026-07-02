"""Feature engineering used consistently during training and inference."""

from __future__ import annotations

import pandas as pd


PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMOUNT_COLUMNS = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
]
PAY_AMOUNT_COLUMNS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact behavioral and demographic features to raw UCI rows."""

    engineered = df.copy()

    engineered["avg_payment_status"] = engineered[PAY_STATUS_COLUMNS].mean(axis=1)
    engineered["max_payment_status"] = engineered[PAY_STATUS_COLUMNS].max(axis=1)
    engineered["late_payment_count"] = (engineered[PAY_STATUS_COLUMNS] > 0).sum(axis=1)

    engineered["avg_bill_amt"] = engineered[BILL_AMOUNT_COLUMNS].mean(axis=1)
    engineered["max_bill_amt"] = engineered[BILL_AMOUNT_COLUMNS].max(axis=1)
    engineered["bill_amt_trend"] = engineered["BILL_AMT1"] - engineered["BILL_AMT6"]

    engineered["avg_pay_amt"] = engineered[PAY_AMOUNT_COLUMNS].mean(axis=1)
    engineered["max_pay_amt"] = engineered[PAY_AMOUNT_COLUMNS].max(axis=1)
    engineered["pay_amt_trend"] = engineered["PAY_AMT1"] - engineered["PAY_AMT6"]

    engineered["payment_ratio"] = engineered["avg_pay_amt"] / (engineered["avg_bill_amt"].abs() + 1.0)
    engineered["credit_utilization"] = engineered["avg_bill_amt"] / (engineered["LIMIT_BAL"] + 1.0)

    engineered["age_group"] = pd.cut(
        engineered["AGE"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    ).astype(str)
    engineered["credit_limit_group"] = pd.cut(
        engineered["LIMIT_BAL"],
        bins=[0, 50000, 140000, 240000, 360000, float("inf")],
        labels=["very_low", "low", "medium", "high", "very_high"],
    ).astype(str)

    return engineered
