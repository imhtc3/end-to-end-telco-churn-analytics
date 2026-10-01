import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    """A handful of rows in the exact shape of the IBM extract (all strings)."""
    rows = [
        # id, gender, senior, partner, dependents, tenure, phone, lines, internet,
        # sec, backup, protect, support, tv, movies, contract, paperless, payment, monthly, total, churn
        ["0001-A", "Female", "0", "Yes", "No", "1", "No", "No phone service", "DSL",
         "No", "Yes", "No", "No", "No", "No", "Month-to-month", "Yes", "Electronic check", "29.85", "29.85", "No"],
        ["0002-B", "Male", "0", "No", "No", "34", "Yes", "No", "DSL",
         "Yes", "No", "Yes", "No", "No", "No", "One year", "No", "Mailed check", "56.95", "1889.5", "No"],
        ["0003-C", "Male", "1", "No", "No", "2", "Yes", "Yes", "Fiber optic",
         "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", "99.65", "199.3", "Yes"],
        ["0004-D", "Female", "0", "Yes", "Yes", "0", "Yes", "No", "No",
         "No internet service", "No internet service", "No internet service", "No internet service",
         "No internet service", "No internet service", "Two year", "No", "Bank transfer (automatic)", "20.25", " ", "No"],
        ["0005-E", "Female", "0", "Yes", "No", "72", "Yes", "Yes", "Fiber optic",
         "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Two year", "Yes", "Credit card (automatic)", "115.50", "8312.75", "No"],
        ["0006-F", "Male", "0", "No", "No", "8", "Yes", "Yes", "Fiber optic",
         "No", "No", "Yes", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", "99.65", "820.5", "Yes"],
    ]
    cols = ["customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def rng():
    return np.random.default_rng(7)
