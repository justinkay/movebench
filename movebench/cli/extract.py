from __future__ import annotations

import argparse
import os

from movebench.features.config import load_sampler_config
from movebench.study import Study


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
        try:
            study = Study(name)
            study.extract_point_features(
                layers_config=args.layers_config,
                sampler_config=args.sampler_config,
                splits=args.splits,
                n_workers=int(args.n_workers),
                skip_existing=not bool(args.no_skip_existing),
            )
        except Exception as e:
            any_failed = True
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 2 if any_failed else 0
