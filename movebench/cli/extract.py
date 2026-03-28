from __future__ import annotations

import argparse
import os

from movebench.download_reports import (
    load_chip_request_status_for_extract,
    summarize_failed_chip_report,
)
from movebench.features.config import load_sampler_config
from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.study import Study


def _default_splits(study: Study) -> list[str]:
    split_keys = [str(split) for split in study.split_inventory()["split_keys"]]
    return split_keys or ["train"]


def _known_download_failure_suffix(status: dict[str, object] | None) -> str:
    if not isinstance(status, dict):
        return ""
    parts: list[str] = []
    report_path = status.get("report_path")
    if report_path:
        parts.append(f"report={report_path}")
    report = status.get("report")
    if isinstance(report, dict):
        summary = summarize_failed_chip_report(report)
        if summary:
            parts.append(summary)
    return " ".join(parts)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract point features for one or more studies.",
        description="Extract point features for one or more studies.",
    )
    parser.add_argument(
        "--layers-config",
        default="configs/layers/all.yaml",
        help="Path to a layers YAML config (default: configs/layers/all.yaml).",
    )
    parser.add_argument(
        "--sampler-config",
        default="configs/samplers/issf_defaults.yaml",
        help=(
            "Path to a sampler YAML config "
            "(default: configs/samplers/issf_defaults.yaml)."
        ),
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional list of studies to extract. Pass 'all' or omit for all available studies.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
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
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override MOVEBENCH_DATA_ROOT for this run.",
    )
    parser.add_argument(
        "--paths-config",
        default=None,
        help="Override MOVEBENCH_PATHS_CONFIG for this run.",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    if args.data_root:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    if args.paths_config:
        os.environ["MOVEBENCH_PATHS_CONFIG"] = str(args.paths_config)

    sampler = load_sampler_config(args.sampler_config)
    plan = REGISTRY.resolve(load_layers_config(args.layers_config))

    all_names = Study.list_available()

    studies = args.studies
    if studies is None or (
        len(studies) == 1 and str(studies[0]).strip().lower() == "all"
    ):
        names = all_names
    else:
        names = [str(s) for s in studies]

    print(f"Studies to extract: {len(names)}")
    print(f"Layers config: {args.layers_config}")
    print(f"Sampler config: {args.sampler_config}")
    if args.splits:
        print(f"Splits: {args.splits}")
    print(f"Workers: {args.n_workers}")
    print(f"Sampler type: {sampler.type}")
    print(f"Candidates: {sampler.num_candidates}")
    print(f"Seed: {sampler.random_seed}")
    print()

    any_failed = False
    for i, name in enumerate(names, 1):
        if name not in all_names:
            print(f"[{i}/{len(names)}] SKIP {name} (not found)")
            continue

        print(f"[{i}/{len(names)}] Extracting {name}...")
        request_status = None
        try:
            study = Study(name)
            resolved_splits = (
                [str(split) for split in args.splits]
                if args.splits
                else _default_splits(study)
            )
            request_status = load_chip_request_status_for_extract(
                study=study,
                plan=plan,
                splits=resolved_splits,
            )
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and not bool(args.allow_known_download_failures)
            ):
                any_failed = True
                print(
                    "  ERROR: known download failures for this extract request; "
                    + _known_download_failure_suffix(request_status)
                )
                continue
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and bool(args.allow_known_download_failures)
            ):
                print(
                    "  WARN: proceeding despite known download failures; "
                    + _known_download_failure_suffix(request_status)
                )
            study.extract_point_features(
                layers_config=args.layers_config,
                sampler_config=args.sampler_config,
                splits=resolved_splits,
                n_workers=int(args.n_workers),
                skip_existing=not bool(args.no_skip_existing),
            )
        except Exception as e:
            any_failed = True
            suffix = _known_download_failure_suffix(request_status)
            if suffix:
                print(f"  ERROR: {e} ({suffix})")
            else:
                print(f"  ERROR: {e}")

    print("\nDone.")
    return 2 if any_failed else 0
