#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from churn.pipeline.runner import STEP_NAMES, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Telco churn analytics pipeline")
    parser.add_argument("--steps", nargs="+", choices=STEP_NAMES,
                        help="run only these steps (plus whatever they depend on)")
    parser.add_argument("--force-download", action="store_true", help="re-download the raw extract")
    args = parser.parse_args()
    run(args.steps, force_download=args.force_download)


if __name__ == "__main__":
    main()
