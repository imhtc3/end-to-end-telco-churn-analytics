"""Lightweight data contract for the raw extract.

Not trying to replace great_expectations here, just catching the things that
would silently break downstream (renamed columns, new category labels, dupes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn.config import get_logger

log = get_logger(__name__)

YES_NO = {"Yes", "No"}
INTERNET_ADDON = {"Yes", "No", "No internet service"}

EXPECTED_DOMAINS: dict[str, set[str]] = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {"0", "1"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": INTERNET_ADDON,
    "OnlineBackup": INTERNET_ADDON,
    "DeviceProtection": INTERNET_ADDON,
    "TechSupport": INTERNET_ADDON,
    "StreamingTV": INTERNET_ADDON,
    "StreamingMovies": INTERNET_ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
    "Churn": YES_NO,
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
REQUIRED = ["customerID", *EXPECTED_DOMAINS, *NUMERIC_COLS]


class DataContractError(Exception):
    pass


@dataclass
class ValidationReport:
    n_rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"rows": self.n_rows, "errors": self.errors, "warnings": self.warnings}


def validate_raw(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    rep = ValidationReport(n_rows=len(df))

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        # nothing else is meaningful without the columns
        if strict:
            raise DataContractError("; ".join(rep.errors))
        return rep

    if df["customerID"].duplicated().any():
        rep.errors.append(f"{df['customerID'].duplicated().sum()} duplicate customerIDs")

    for col, allowed in EXPECTED_DOMAINS.items():
        seen = set(df[col].str.strip().unique())
        unexpected = seen - allowed
        if unexpected:
            rep.errors.append(f"{col}: unexpected values {sorted(unexpected)}")

    for col in NUMERIC_COLS:
        as_num = pd.to_numeric(df[col].str.strip(), errors="coerce")
        n_bad = as_num.isna().sum()
        if n_bad:
            # TotalCharges is blank for brand-new customers (tenure 0); that's known
            msg = f"{col}: {n_bad} non-numeric values"
            if col == "TotalCharges":
                zero_tenure = (pd.to_numeric(df["tenure"], errors="coerce") == 0)
                if (as_num.isna() <= zero_tenure).all():
                    rep.warnings.append(msg + " (all tenure == 0, will impute 0)")
                    continue
            rep.errors.append(msg)
        elif (as_num < 0).any():
            rep.errors.append(f"{col}: negative values")

    churn_rate = (df["Churn"] == "Yes").mean()
    if not 0.05 < churn_rate < 0.60:
        rep.warnings.append(f"churn rate {churn_rate:.1%} looks off vs. history (~26%)")

    for w in rep.warnings:
        log.warning(w)
    if rep.errors:
        for e in rep.errors:
            log.error(e)
        if strict:
            raise DataContractError("; ".join(rep.errors))
    else:
        log.info("raw extract passed %d checks", len(EXPECTED_DOMAINS) + len(NUMERIC_COLS) + 2)
    return rep
