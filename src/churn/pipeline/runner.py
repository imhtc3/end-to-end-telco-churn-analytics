"""Tiny DAG runner. Each step declares what it needs; results are passed along
in a shared context dict and a run manifest is written at the end."""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from churn.config import ROOT, get_logger, load_config, resolve

log = get_logger("pipeline")


@dataclass
class Step:
    name: str
    fn: Callable[[dict], object]
    needs: list[str] = field(default_factory=list)


def _extract(ctx):
    from churn.pipeline.extract import extract
    ctx["raw"] = extract(force=ctx.get("force_download", False))


def _validate(ctx):
    from churn.pipeline.validate import validate_raw
    ctx["validation"] = validate_raw(ctx["raw"]).as_dict()


def _transform(ctx):
    from churn.pipeline.transform import transform
    df = transform(ctx["raw"])
    df.to_csv(resolve(load_config()["data"]["clean_file"]), index=False)
    ctx["clean"] = df


def _load(ctx):
    from churn.pipeline.load import load
    load(ctx["clean"])


def _sql(ctx):
    from churn.analysis.sql_insights import run_all
    ctx["sql_results"] = list(run_all())


def _eda(ctx):
    from churn.analysis.eda import run
    run(ctx["clean"])


def _segment(ctx):
    from churn.analysis import style
    from churn.models.segment import pick_k, plot_segments, segment

    cfg = load_config()
    style.apply()
    scan = pick_k(ctx["clean"], seed=cfg["random_seed"])
    labelled, profile = segment(ctx["clean"], k=cfg["segmentation"]["n_clusters"], seed=cfg["random_seed"])
    profile.round(3).to_csv(ROOT / cfg["paths"]["reports"] / "segment_profile.csv", index=False)
    style.save(plot_segments(labelled, profile, scan), ROOT / cfg["paths"]["figures"] / "15_segments.png")
    ctx["segments"] = labelled


def _train(ctx):
    from churn.models.train import run
    ctx["model_summary"] = run()


def _excel(ctx):
    from churn.reporting.excel_report import build
    thr = ctx.get("model_summary", {}).get("threshold", 0.5)
    build(segments=ctx.get("segments"), threshold=thr)


def _powerbi(ctx):
    from churn.reporting.powerbi_export import export
    export(segments=ctx.get("segments"))


STEPS = [
    Step("extract", _extract),
    Step("validate", _validate, ["extract"]),
    Step("transform", _transform, ["validate"]),
    Step("load", _load, ["transform"]),
    Step("sql", _sql, ["load"]),
    Step("eda", _eda, ["transform"]),
    Step("segment", _segment, ["transform"]),
    Step("train", _train, ["load"]),
    Step("excel", _excel, ["train", "segment"]),
    Step("powerbi", _powerbi, ["train", "segment"]),
]
STEP_NAMES = [s.name for s in STEPS]


def _with_dependencies(selected: list[str]) -> list[str]:
    by_name = {s.name: s for s in STEPS}
    needed: set[str] = set()

    def visit(name):
        if name in needed:
            return
        needed.add(name)
        for dep in by_name[name].needs:
            visit(dep)

    for name in selected:
        visit(name)
    return [s for s in STEP_NAMES if s in needed]  # keep declared order (already topological)


def run(steps: list[str] | None = None, force_download: bool = False) -> dict:
    plan = _with_dependencies(steps or STEP_NAMES)
    log.info("plan: %s", " -> ".join(plan))
    ctx: dict = {"force_download": force_download}
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "steps": []}

    status = "success"
    for step in [s for s in STEPS if s.name in plan]:
        t0 = time.perf_counter()
        log.info("▶ %s", step.name)
        try:
            step.fn(ctx)
            manifest["steps"].append({"step": step.name, "status": "ok", "seconds": round(time.perf_counter() - t0, 2)})
        except Exception as exc:
            manifest["steps"].append({"step": step.name, "status": "failed", "error": repr(exc),
                                      "seconds": round(time.perf_counter() - t0, 2)})
            log.error("step %s failed:\n%s", step.name, traceback.format_exc())
            status = "failed"
            break

    manifest["status"] = status
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if "validation" in ctx:
        manifest["validation"] = ctx["validation"]
    if isinstance(ctx.get("clean"), pd.DataFrame):
        manifest["rows"] = len(ctx["clean"])
    if "model_summary" in ctx:
        manifest["best_model"] = ctx["model_summary"]["best_model"]

    path = ROOT / load_config()["paths"]["reports"] / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    log.info("pipeline %s in %.1fs", status, sum(s["seconds"] for s in manifest["steps"]))
    if status != "success":
        raise SystemExit(1)
    return manifest
