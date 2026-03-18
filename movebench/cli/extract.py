from __future__ import annotations

import argparse
import os


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Extract point features for one or more studies.",
        description="Extract point features for one or more studies.",
    )
    parser.add_argument(
        "--config",
        default="configs/layers_all.yaml",
        help="Path to a layers YAML config (default: configs/layers_all.yaml).",
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
        "--parametric",
        action="store_true",
        help="Use parametric iSSF sampling (default: empirical).",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=100,
        help="Number of candidate points per step (default: 100).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for candidate sampling (default: 0).",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing point stores on disk.",
    )
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=2.0,
        help="Step length multiplier used when fitting the sampler (default: 2.0).",
    )
    parser.add_argument(
        "--step-q",
        type=float,
        default=0.95,
        help="Quantile for step length cutoff used when fitting the sampler (default: 0.95).",
    )
    parser.add_argument(
        "--gap-q",
        type=float,
        default=0.999,
        help="Quantile for time gap cutoff used when fitting the sampler (default: 0.999).",
    )
    parser.add_argument(
        "--max-step-q",
        type=float,
        default=0.99,
        help="Upper quantile cap for valid step lengths used during extraction (default: 0.99).",
    )
    parser.add_argument(
        "--bounds-buffer-deg",
        type=float,
        default=0.01,
        help="Bounds buffer (degrees) for local raster reads (default: 0.01).",
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

    try:
        from movebench.study import Study
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")
        print("Make sure movebench dependencies are installed in your environment.")
        return 1

    all_names = Study.list_available()

    studies = args.studies
    if studies is None or (len(studies) == 1 and str(studies[0]).strip().lower() == "all"):
        names = all_names
    else:
        names = [str(s) for s in studies]

    print(f"Studies to extract: {len(names)}")
    print(f"Config: {args.config}")
    if args.splits:
        print(f"Splits: {args.splits}")
    print(f"Workers: {args.n_workers}")
    print(f"Parametric: {bool(args.parametric)}")
    print(f"Candidates: {args.num_candidates}")
    print(f"Seed: {args.random_seed}")
    print(f"Max step q: {args.max_step_q}")
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
                config=args.config,
                splits=args.splits,
                parametric=bool(args.parametric),
                num_candidates=int(args.num_candidates),
                random_seed=int(args.random_seed),
                n_workers=int(args.n_workers),
                skip_existing=not bool(args.no_skip_existing),
                step_multiplier=float(args.step_multiplier),
                step_q=float(args.step_q),
                max_step_q=float(args.max_step_q),
                gap_q=float(args.gap_q),
                bounds_buffer_deg=float(args.bounds_buffer_deg),
            )
        except Exception as e:
            any_failed = True
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 2 if any_failed else 0
