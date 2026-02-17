#!/usr/bin/env python
"""YAML-driven benchmark runner for MoveBench baselines.

Usage:
  python benchmark.py --config configs/benchmarks/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MoveBench benchmarks from a YAML config.")
    p.add_argument(
        "--config",
        type=str,
        default="configs/benchmarks/default.yaml",
        help="Benchmark YAML config path.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        # Allow running without installing as a package.
        sys.path.insert(0, str(repo_root))

    from movebench import benchmark as mb_benchmark

    mb_benchmark.main(config_path=str(args.config), overrides=None)


if __name__ == "__main__":
    main()

