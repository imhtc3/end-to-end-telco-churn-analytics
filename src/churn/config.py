from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config" / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve(rel: str) -> Path:
    """Paths in the config are relative to the repo root."""
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger
