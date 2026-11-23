from __future__ import annotations

import argparse
import os
from pathlib import Path

from movebench.ingest import ingest


def _parse_int_with_underscores(value: str) -> int:
    s = str(value).strip()
    if not s:
        raise argparse.ArgumentTypeError("Expected an integer")
    try:
        return int(s.replace("_", ""))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer: {value!r}") from e


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "ingest",
        help="Ingest Movebank CSV(s) into movebench.",
        description="Ingest Movebank-formatted CSVs into movebench as new studies.",
    )
    parser.add_argument(
        "source_path",
        type=Path,
        help="Path to a Movebank-formatted CSV or a directory of CSVs.",
    )
    parser.add_argument(
        "--study-id", type=str, default=None, help="Study id (default: CSV stem)."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override MOVEBENCH_DATA_ROOT for this run.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing study directory."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing.",
    )
    parser.add_argument(
        "--max-individuals",
        type=_parse_int_with_underscores,
        default=1000,
        help="Maximum number of individuals to keep (default: 10).",
    )
    parser.add_argument(
        "--max-points",
        type=_parse_int_with_underscores,
        default=100_000_000,
        help="Stop after cumulative points exceeds this threshold (default: 100_000).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for individual subsampling (default: 0).",
    )
    parser.set_defaults(func=run, _parser=parser)


def run(args: argparse.Namespace) -> int:
    if args.data_root is not None:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    try:
        return ingest(
            args.source_path,
            study_id=args.study_id,
            max_individuals=args.max_individuals,
            max_points=args.max_points,
            random_seed=args.random_seed,
            dry_run=args.dry_run,
            force=args.force,
        )
    except ValueError as exc:
        args._parser.error(str(exc))
