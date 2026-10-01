from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import load_config
from churn.pipeline.load import read_query

CATEGORICAL = [
    "gender",
    "contract_type",
    "payment_method",
    "internet_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

NUMERIC = [
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "phone_service",
    "paperless_billing",
    "auto_pay",
    "contract_months",
    "n_addons",
    "n_services",
    "monthly_charges",
    "log_total_charges",
    "charge_drift",
    "price_vs_peers",
]

TARGET = "churn"
ID = "customer_id"


@dataclass
class Split:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    ids: dict[str, pd.Series]
    full: pd.DataFrame

    def assignment(self) -> pd.Series:
        """customer_id -> which split it landed in (handy for the BI layer)."""
        parts = [pd.Series(name, index=ids.values) for name, ids in self.ids.items()]
        return pd.concat(parts).rename("split")


def load_mart() -> pd.DataFrame:
    df = read_query("SELECT * FROM vw_feature_mart")
    df["log_total_charges"] = np.log1p(df["total_charges"])
    return df


def make_split(df: pd.DataFrame | None = None, seed: int | None = None) -> Split:
    cfg = load_config()
    seed = cfg["random_seed"] if seed is None else seed
    df = load_mart() if df is None else df

    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET].to_numpy()
    ids = df[ID]

    # test first, then carve validation out of what is left
    X_tmp, X_test, y_tmp, y_test, id_tmp, id_test = train_test_split(
        X, y, ids, test_size=cfg["model"]["test_size"], stratify=y, random_state=seed
    )
    valid_frac = cfg["model"]["valid_size"] / (1 - cfg["model"]["test_size"])
    X_train, X_valid, y_train, y_valid, id_train, id_valid = train_test_split(
        X_tmp, y_tmp, id_tmp, test_size=valid_frac, stratify=y_tmp, random_state=seed
    )
    return Split(
        X_train, X_valid, X_test, y_train, y_valid, y_test,
        ids={"train": id_train, "valid": id_valid, "test": id_test},
        full=df,
    )
