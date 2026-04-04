from __future__ import annotations

import argparse
from movebench.extract import (
    build_extraction_runs,
    normalize_split_values,
    parse_multi_path_arg,
    resolve_requested_studies,
    run_extract_job,
    split_cli_values,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract point features for one or more studies.",
        description="Extract point features for one or more studies.",
    )
    parser.add_argument(
        "--layers-config",
        dest="layers_configs",
        action="append",
        type=lambda value: parse_multi_path_arg(value, flag_name="--layers-config"),
        default=None,
        help="Layer config path(s). Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--sampler-config",
        dest="sampler_configs",
        action="append",
        type=lambda value: parse_multi_path_arg(value, flag_name="--sampler-config"),
        default=None,
        help="Sampler config path(s). Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional list of studies or named sets like 'mini'. Pass 'all' or omit for all available studies.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional list of splits to extract (e.g. --splits train test). Default: all splits.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers for feature extraction (default: 16).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing point stores on disk.",
    )
    parser.add_argument(
        "--allow-known-download-failures",
        action="store_true",
        help="Proceed even if the latest matching chip download report is marked failed.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    runs = build_extraction_runs(
        layers_configs=args.layers_configs,
        sampler_configs=args.sampler_configs,
    )
    studies = resolve_requested_studies(_collapse_cli_values(args.studies))
    splits = normalize_split_values(args.splits)
    any_failed = False

    print(f"[extract] combos={len(runs)} studies={len(studies)}")
    for study_id in studies:
        for run in runs:
            rc = run_extract_job(
                study_id=str(study_id),
                run=run,
                n_workers=int(args.n_workers),
                splits=splits,
                skip_existing=not bool(args.no_skip_existing),
                allow_known_download_failures=bool(
                    args.allow_known_download_failures
                ),
            )
            if rc != 0:
                any_failed = True
    return 2 if any_failed else 0


def _collapse_cli_values(values: list[str] | None) -> str | None:
    resolved = split_cli_values(values)
    if not resolved:
        return None
    return ",".join(resolved)
