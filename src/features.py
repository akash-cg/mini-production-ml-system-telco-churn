"""Shared preprocessing and feature engineering used by training and serving."""
from __future__ import annotations
import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
SECURITY_SUPPORT_COLUMNS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_COLUMNS = ["StreamingTV", "StreamingMovies"]

def _yes(s: pd.Series) -> pd.Series:
    return s.astype(str).eq("Yes").astype(int)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"] * out["tenure"])
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic, stateless features to minimize training-serving skew."""
    out = clean_raw_data(df)
    safe_tenure = out["tenure"].clip(lower=1)
    out["avg_charge_per_month"] = out["TotalCharges"] / safe_tenure
    out["charge_gap"] = out["TotalCharges"] - (out["MonthlyCharges"] * out["tenure"])
    out["service_count"] = sum(_yes(out[c]) for c in SERVICE_COLUMNS)
    out["security_support_count"] = sum(_yes(out[c]) for c in SECURITY_SUPPORT_COLUMNS)
    out["streaming_count"] = sum(_yes(out[c]) for c in STREAMING_COLUMNS)
    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
    out["monthly_charge_per_service"] = out["MonthlyCharges"] / out["service_count"].clip(lower=1)
    out["automatic_payment"] = out["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
    out["contract_risk_score"] = out["Contract"].map({"Month-to-month": 2, "One year": 1, "Two year": 0}).fillna(1)
    return out

def split_xy(df: pd.DataFrame):
    featured = build_features(df)
    y = featured["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    X = featured.drop(columns=["Churn", "customerID"], errors="ignore")
    return X, y
