"""Flat star-schema extracts that the Power BI report in powerbi/ is built on."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pandas as pd

from churn.config import ROOT, get_logger, load_config
from churn.pipeline.load import connect

log = get_logger(__name__)

TABLES = {
    "dim_customer": "SELECT * FROM dim_customer",
    "dim_contract": "SELECT * FROM dim_contract",
    "dim_payment": "SELECT * FROM dim_payment",
    "dim_internet": "SELECT * FROM dim_internet",
    "fact_subscription": "SELECT * FROM fact_subscription",
    "bridge_customer_service": "SELECT * FROM bridge_customer_service",
}


def _dim_tenure_band() -> pd.DataFrame:
    from churn.pipeline.transform import TENURE_LABELS

    return pd.DataFrame({"tenure_band": TENURE_LABELS, "band_order": range(1, len(TENURE_LABELS) + 1)})


def export(segments: pd.DataFrame | None = None) -> Path:
    out = ROOT / load_config()["data"]["powerbi_dir"]
    out.mkdir(parents=True, exist_ok=True)

    with closing(connect()) as con:
        for name, sql in TABLES.items():
            pd.read_sql_query(sql, con).to_csv(out / f"{name}.csv", index=False)

    _dim_tenure_band().to_csv(out / "dim_tenure_band.csv", index=False)

    scores_path = ROOT / "data" / "processed" / "churn_scores.csv"
    if scores_path.exists():
        pd.read_csv(scores_path).to_csv(out / "fact_churn_score.csv", index=False)

    if segments is not None:
        segments[["customer_id", "segment_id", "segment_name"]].to_csv(out / "dim_segment.csv", index=False)

    log.info("Power BI extracts written to %s", out.relative_to(ROOT))
    return out
