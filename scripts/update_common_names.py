#!/usr/bin/env python3
"""Utility script to refresh common-name assignments in metadata CSVs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.metadata import get_common_names  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute study_species_common using latest taxonomic mapping."
    )
    parser.add_argument(
        "--input",
        default="mbr_studies_metadata.csv",
        help="Existing metadata CSV to patch (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. When omitted, overwrites the input file after backing it up to *.bak.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def refresh_common_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if "study_species" not in df.columns:
        raise ValueError("Input CSV missing required column 'study_species'")
    if "study_species_common" not in df.columns:
        df["study_species_common"] = ""

    updates = 0
    new_common_names = []
    for species, current_common in zip(df["study_species"], df["study_species_common"]):
        recomputed = get_common_names(species if isinstance(species, str) else "")
        new_common_names.append(recomputed)
        if recomputed != current_common:
            updates += 1
    df["study_species_common"] = new_common_names
    return df, updates


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata file not found: {input_path}")

    df = pd.read_csv(input_path)
    df, updates = refresh_common_names(df)

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        input_path.replace(backup_path)
        logging.info("Backed up original file to %s", backup_path)
        output_path = input_path

    df.to_csv(output_path, index=False)
    logging.info("Wrote updated metadata to %s (updated %s rows)", output_path, updates)


if __name__ == "__main__":
    main()

