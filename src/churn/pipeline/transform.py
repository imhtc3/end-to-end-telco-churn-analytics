from __future__ import annotations

import re

import numpy as np
import pandas as pd

from churn.config import get_logger

log = get_logger(__name__)

ADDON_SERVICES = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]
BINARY_COLS = ["partner", "dependents", "phone_service", "paperless_billing", "churn"]

TENURE_BINS = [-1, 6, 12, 24, 48, 60, np.inf]
TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49-60m", "61m+"]


def to_snake(name: str) -> str:
    name = name.replace("customerID", "customer_id")
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return name.lower()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [to_snake(c) for c in df.columns]
    df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)

    df["senior_citizen"] = df["senior_citizen"].astype(int)
    df["tenure"] = df["tenure"].astype(int)
    df["monthly_charges"] = df["monthly_charges"].astype(float)

    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
    n_blank = df["total_charges"].isna().sum()
    if n_blank:
        # these are customers in their first billing cycle - nothing billed yet
        df.loc[df["total_charges"].isna() & (df["tenure"] == 0), "total_charges"] = 0.0
        log.info("imputed %d blank total_charges for tenure-0 customers", n_blank)

    for col in BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["has_internet"] = (out["internet_service"] != "No").astype(int)
    out["multiple_lines_flag"] = (out["multiple_lines"] == "Yes").astype(int)

    addon_flags = out[ADDON_SERVICES].eq("Yes").astype(int)
    out["n_addons"] = addon_flags.sum(axis=1)
    out["has_protection_bundle"] = (
        addon_flags[["online_security", "tech_support", "device_protection"]].sum(axis=1) >= 2
    ).astype(int)
    out["n_services"] = out["phone_service"] + out["multiple_lines_flag"] + out["has_internet"] + out["n_addons"]

    out["auto_pay"] = out["payment_method"].str.contains("automatic").astype(int)
    out["contract_months"] = out["contract"].map({"Month-to-month": 1, "One year": 12, "Two year": 24})

    out["tenure_band"] = pd.cut(out["tenure"], TENURE_BINS, labels=TENURE_LABELS).astype(str)

    # average of what they actually paid vs. what they pay now. A positive gap
    # means the bill went up over time -- price increases are a classic trigger.
    months = out["tenure"].clip(lower=1)
    out["avg_historic_charge"] = np.where(out["tenure"] > 0, out["total_charges"] / months, out["monthly_charges"])
    out["charge_drift"] = out["monthly_charges"] - out["avg_historic_charge"]
    out["charge_per_service"] = out["monthly_charges"] / out["n_services"].clip(lower=1)

    out["is_new_customer"] = (out["tenure"] <= 6).astype(int)
    out["family"] = ((out["partner"] == 1) | (out["dependents"] == 1)).astype(int)
    out["fiber_no_support"] = (
        (out["internet_service"] == "Fiber optic") & (out["tech_support"] != "Yes")
    ).astype(int)

    return out


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_features(clean(raw))
    log.info("transformed frame: %d rows, %d columns", *df.shape)
    return df
