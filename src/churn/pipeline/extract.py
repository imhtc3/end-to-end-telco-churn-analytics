"""Pull the raw IBM Telco churn extract and keep a local copy under data/raw."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from churn.config import get_logger, load_config, resolve

log = get_logger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def download(urls: list[str], dest: Path, timeout: int = 30) -> Path:
    tmp = dest.with_suffix(".part")
    last_err = None
    for url in urls:
        try:
            log.info("downloading %s", url)
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(dest)
            return dest
        except Exception as exc:  # network errors, 404s, etc. -> try the next mirror
            log.warning("failed (%s): %s", type(exc).__name__, exc)
            last_err = exc
    raise RuntimeError(
        f"Could not fetch the dataset from any mirror. Put the CSV at {dest} and rerun."
    ) from last_err


def extract(force: bool = False) -> pd.DataFrame:
    cfg = load_config()
    dest = resolve(cfg["data"]["raw_file"])

    if force or not dest.exists():
        download(cfg["data"]["source_urls"], dest)
    else:
        log.info("using cached raw file %s", dest.name)

    # read everything as string first -- TotalCharges has blanks and we want
    # validation to see the file exactly as delivered
    df = pd.read_csv(dest, dtype=str, keep_default_na=False)
    log.info("raw extract: %d rows x %d cols (sha256 %s…)", *df.shape, _sha256(dest)[:12])
    return df
