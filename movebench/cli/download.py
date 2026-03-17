from __future__ import annotations

import argparse
import os
from typing import Any


def _count_present_tiles(tile_stats: dict[str, Any] | None) -> int:
    if not isinstance(tile_stats, dict):
        return 0

    total = 0
    for layer_stats in tile_stats.values():
        if not isinstance(layer_stats, dict):
            continue
        years = layer_stats.get("years")
        if isinstance(years, dict):
            for year_stats in years.values():
                if isinstance(year_stats, dict):
                    total += int(year_stats.get("downloaded", 0)) + int(year_stats.get("existing", 0))
            continue
        total += int(layer_stats.get("downloaded", 0)) + int(layer_stats.get("existing", 0))
    return total


def _format_study_download_summary(name: str, result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        chip_present = 0
        tile_present = 0
    else:
        chip_stats = result.get("chips")
        chip_present = (
            int(chip_stats.get("downloaded", 0)) + int(chip_stats.get("existing", 0))
            if isinstance(chip_stats, dict)
            else 0
        )
        tile_present = _count_present_tiles(result.get("tiles"))
    return f"OK {name}: {chip_present} chips, {tile_present} tiles present"


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Download rasters for all available studies.",
        description="Download rasters for all available studies.",
    )
    parser.add_argument(
        "--config",
        default="configs/layers_all.yaml",
        help="Path to a layers YAML config (default: configs/layers_all.yaml).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to download (e.g. --splits train test). Default: all splits.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=32,
        help="Number of multiprocessing workers for chip downloads (default: 32).",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Only download these studies (by name). Default: all available studies.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing rasters on disk.",
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

    print(f"Studies to download: {len(names)}")
    if args.config:
        print(f"Config: {args.config}")
    if args.splits:
        print(f"Splits: {args.splits}")
    print(f"Workers: {args.n_workers}")
    print()

    for i, name in enumerate(names, 1):
        if name not in all_names:
            print(f"[{i}/{len(names)}] SKIP {name} (not found)")
            continue

        print(f"[{i}/{len(names)}] Downloading {name}...")
        try:
            study = Study(name)
            result = study.download_rasters(
                config=args.config,
                splits=args.splits,
                n_workers=args.n_workers,
                skip_existing=not args.no_skip_existing,
            )
            print(f"[{i}/{len(names)}] {_format_study_download_summary(name, result)}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 0
