from __future__ import annotations

import argparse
import os


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sample-points",
        help="Sample GEE covariates at exact GPS coordinates (no buffer required).",
        description=(
            "Sample GEE chip layers at exact GPS coordinates for all points in one "
            "or more studies.  Writes a Parquet file per study without requiring a "
            "buffer or any outlier pre-cleaning step."
        ),
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
        help="Studies to process (by name). Default: all available studies.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help=(
            "Optional list of splits to process (e.g. --splits train test). "
            "Default: all splits."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=32,
        help="Number of multiprocessing workers (default: 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help=(
            "Maximum number of points per GEE sampleRegions call (default: 500). "
            "Reduce if GEE returns memory errors."
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download all points even if the parquet file already exists.",
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
    names = args.studies if args.studies else all_names

    print(f"Studies to sample: {len(names)}")
    print(f"Config:  {args.config}")
    print(f"Workers: {args.n_workers}")
    print(f"Batch:   {args.batch_size}")
    print()

    any_failed = False
    for i, name in enumerate(names, 1):
        if name not in all_names:
            print(f"[{i}/{len(names)}] SKIP {name} (not found)")
            continue

        print(f"[{i}/{len(names)}] Sampling {name}...")
        try:
            study = Study(name)
            result = study.download_point_values(
                config=args.config,
                splits=args.splits,
                n_workers=int(args.n_workers),
                batch_size=int(args.batch_size),
                skip_existing=not bool(args.no_skip_existing),
            )
            print(
                f"  config_hash={result['config_hash']} "
                f"n_points={result['n_points']} "
                f"downloaded={result['n_downloaded']} "
                f"existing={result['n_existing']} "
                f"failed={result['n_failed']}"
            )
            if result["n_failed"] > 0:
                any_failed = True
                for f in result["failures"][:5]:
                    print(
                        f"    FAIL chip_group={f['chip_group']} "
                        f"bucket={f['bucket_key']} error={f['error']}"
                    )
        except Exception as e:
            any_failed = True
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 2 if any_failed else 0
