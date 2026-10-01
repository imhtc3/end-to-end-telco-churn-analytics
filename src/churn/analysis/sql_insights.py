"""Runs every query in sql/analysis and drops the result next to the reports."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import ROOT, get_logger, load_config
from churn.pipeline.load import read_query

log = get_logger(__name__)


def run_all(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    cfg = load_config()
    sql_dir = ROOT / cfg["paths"]["sql_dir"] / "analysis"
    out_dir = out_dir or ROOT / cfg["paths"]["reports"] / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for path in sorted(sql_dir.glob("*.sql")):
        df = read_query(path.read_text())
        if df.empty:
            log.info("%s returned no rows, skipping", path.name)
            continue
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        results[path.stem] = df
        log.info("%-28s -> %d rows", path.name, len(df))
    return results
