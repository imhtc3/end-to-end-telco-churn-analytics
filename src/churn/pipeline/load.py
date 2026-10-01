from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from churn.config import ROOT, get_logger, load_config, resolve
from churn.pipeline.transform import ADDON_SERVICES

log = get_logger(__name__)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or resolve(load_config()["data"]["warehouse"])
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    return con


def run_sql_file(con: sqlite3.Connection, path: Path) -> None:
    con.executescript(path.read_text())


def _dim(values: pd.Series, key: str, name: str) -> pd.DataFrame:
    d = pd.DataFrame({name: sorted(values.unique())})
    d.insert(0, key, range(1, len(d) + 1))
    return d


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    dim_contract = _dim(df["contract"], "contract_key", "contract_type")
    dim_contract["contract_months"] = dim_contract["contract_type"].map(
        {"Month-to-month": 1, "One year": 12, "Two year": 24}
    )

    dim_payment = _dim(df["payment_method"], "payment_key", "payment_method")
    dim_payment["auto_pay"] = dim_payment["payment_method"].str.contains("automatic").astype(int)

    dim_internet = _dim(df["internet_service"], "internet_key", "internet_service")

    dim_customer = df[
        ["customer_id", "gender", "senior_citizen", "partner", "dependents", "family"]
    ].copy()

    fact = (
        df.merge(dim_contract[["contract_key", "contract_type"]], left_on="contract", right_on="contract_type")
        .merge(dim_payment[["payment_key", "payment_method"]], on="payment_method")
        .merge(dim_internet, on="internet_service")
    )
    fact = fact[
        [
            "customer_id", "contract_key", "payment_key", "internet_key",
            "tenure", "tenure_band", "phone_service", "multiple_lines",
            "paperless_billing", "n_addons", "n_services",
            "monthly_charges", "total_charges", "charge_drift", "churn",
        ]
    ]

    bridge = df.melt(
        id_vars="customer_id", value_vars=ADDON_SERVICES, var_name="service_name", value_name="status"
    )

    return {
        "dim_contract": dim_contract,
        "dim_payment": dim_payment,
        "dim_internet": dim_internet,
        "dim_customer": dim_customer,
        "fact_subscription": fact,
        "bridge_customer_service": bridge,
    }


def load(df: pd.DataFrame, db_path: Path | None = None) -> Path:
    cfg = load_config()
    db_path = db_path or resolve(cfg["data"]["warehouse"])
    sql_dir = ROOT / cfg["paths"]["sql_dir"]
    tables = build_tables(df)

    with closing(connect(db_path)) as con:
        run_sql_file(con, sql_dir / "schema.sql")
        # order matters because of FKs
        for name, frame in tables.items():
            frame.to_sql(name, con, if_exists="append", index=False)
            log.info("loaded %-24s %6d rows", name, len(frame))
        run_sql_file(con, sql_dir / "views.sql")
        con.commit()

        n_fact = con.execute("SELECT COUNT(*) FROM fact_subscription").fetchone()[0]
        n_view = con.execute("SELECT COUNT(*) FROM vw_feature_mart").fetchone()[0]
        if n_fact != len(df) or n_view != len(df):
            raise RuntimeError(f"row count mismatch after load: src={len(df)} fact={n_fact} mart={n_view}")

    log.info("warehouse ready at %s", db_path)
    return db_path


def read_query(sql: str, db_path: Path | None = None, **params) -> pd.DataFrame:
    with closing(connect(db_path)) as con:
        return pd.read_sql_query(sql, con, params=params or None)


def write_scores(scores: pd.DataFrame, db_path: Path | None = None) -> None:
    with closing(connect(db_path)) as con:
        con.execute("DELETE FROM model_scores WHERE model_name IN (%s)" % ",".join("?" * scores["model_name"].nunique()),
                    tuple(scores["model_name"].unique()))
        scores.to_sql("model_scores", con, if_exists="append", index=False)
        con.commit()
