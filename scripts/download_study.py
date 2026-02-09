from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Download movebench rasters for a single study (integration test)."
    )
    parser.add_argument("study_id", help="Study id (directory name under studies root).")
    parser.add_argument(
        "--config",
        default="configs/layers_all.yaml",
        help="Path to a layers YAML config (default: configs/layers_all.yaml).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to download (e.g. --splits train test). Default: all splits present in trajectories.csv",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers for chip downloads.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing rasters on disk.",
    )
    parser.add_argument(
        "--px-per-side",
        type=int,
        default=500,
        help="Tile download grid pixel size per side (default: 500).",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override MOVEBENCH_DATA_ROOT for this run.",
    )
    parser.add_argument(
        "--paths-config",
        default=None,
        help="Override MOVEBENCH_PATHS_CONFIG for this run (not required for GEE downloads).",
    )

    args = parser.parse_args(argv[1:])

    if args.data_root:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    if args.paths_config:
        os.environ["MOVEBENCH_PATHS_CONFIG"] = str(args.paths_config)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    try:
        from movebench.study import Study
    except ModuleNotFoundError as e:
        # Common missing deps: PyYAML (yaml), earthengine-api (ee), rasterio, pandas, requests, tqdm.
        print(f"Import failed: {e}")
        print("Make sure movebench dependencies are installed in your environment.")
        return 1

    study = Study(args.study_id)
    print(f"study_dir: {study.study_dir}")

    result = study.download_rasters(
        config=args.config,
        splits=args.splits,
        n_workers=int(args.n_workers),
        skip_existing=not bool(args.no_skip_existing),
        px_per_side=int(args.px_per_side),
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
