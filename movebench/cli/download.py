from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.study import Study, resolve_study_selection
from movebench.layers.download_zip import download_public_studies


def _count_tiles(tile_stats: dict[str, Any] | None, *fields: str) -> int:
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
                    total += sum(int(year_stats.get(field, 0)) for field in fields)
            continue
        total += sum(int(layer_stats.get(field, 0)) for field in fields)
    return total


def _download_failure_count(result: dict[str, Any] | None) -> int:
    if not isinstance(result, dict):
        return 0
    chips = result.get("chips")
    failed = int(chips.get("failed", 0)) if isinstance(chips, dict) else 0
    return failed + _count_tiles(result.get("tiles"), "failed")


def _format_study_download_summary(name: str, result: dict[str, Any] | None) -> str:
    failed = _download_failure_count(result)
    report_path: str | None = None
    if not isinstance(result, dict):
        chip_present = 0
        tile_present = 0
    else:
        chip_stats = result.get("chips")
        if isinstance(chip_stats, dict) and chip_stats.get("report_path"):
            report_path = str(chip_stats["report_path"])
        chip_present = (
            int(chip_stats.get("downloaded", 0)) + int(chip_stats.get("existing", 0))
            if isinstance(chip_stats, dict)
            else 0
        )
        tile_stats = result.get("tiles")
        tile_present = _count_tiles(tile_stats, "downloaded", "existing")
    status = "WARN" if failed > 0 else "OK"
    msg = f"{status} {name}: {chip_present} chips, {tile_present} tiles present"
    if failed > 0:
        msg += f", {failed} failures"
        if report_path:
            msg += f", report={report_path}"
    return msg


def _study_failure_details(result: dict[str, Any] | None) -> list[str]:
    if not isinstance(result, dict):
        return []

    details: list[str] = []

    chip_stats = result.get("chips")
    if isinstance(chip_stats, dict):
        chip_failures = chip_stats.get("failures")
        if isinstance(chip_failures, list) and chip_failures:
            seen: set[tuple[str, str]] = set()
            for item in chip_failures:
                if not isinstance(item, dict):
                    continue
                group = str(item.get("group", "unknown"))
                error = str(item.get("error", "unknown"))
                key = (group, error)
                if key in seen:
                    continue
                seen.add(key)
                details.append(f"chip {group}: {error}")

    tile_stats = result.get("tiles")
    if not isinstance(tile_stats, dict):
        return details

    def _append_tile_details(label: str, stats: dict[str, Any]) -> None:
        if int(stats.get("failed", 0)) <= 0:
            return
        index_path = stats.get("index_path")
        if not index_path:
            details.append(f"{label}: {int(stats.get('failed', 0))} tile failures")
            return

        try:
            with open(Path(index_path), "r") as f:
                payload = json.load(f)
        except Exception as e:
            details.append(
                f"{label}: {int(stats.get('failed', 0))} tile failures ({e})"
            )
            return

        reasons: list[str] = []
        seen_reasons: set[str] = set()
        for item in payload.get("errors", []):
            if not isinstance(item, dict):
                continue
            error = str(item.get("error", "unknown"))
            if error in seen_reasons:
                continue
            seen_reasons.add(error)
            reasons.append(error)
        for item in payload.get("tiles", []):
            if not isinstance(item, dict) or "error" not in item:
                continue
            error = str(item["error"])
            if error in seen_reasons:
                continue
            seen_reasons.add(error)
            reasons.append(error)

        if reasons:
            details.append(f"{label}: " + " | ".join(reasons[:2]))
        else:
            details.append(f"{label}: {int(stats.get('failed', 0))} tile failures")

    for layer_name, layer_stats in sorted(
        tile_stats.items(), key=lambda kv: str(kv[0])
    ):
        if not isinstance(layer_stats, dict):
            continue
        years = layer_stats.get("years")
        if isinstance(years, dict):
            for year, year_stats in sorted(years.items(), key=lambda kv: int(kv[0])):
                if isinstance(year_stats, dict):
                    _append_tile_details(f"{layer_name}[{int(year)}]", year_stats)
            continue
        _append_tile_details(str(layer_name), layer_stats)

    return details


def _apply_common_path_overrides(args: argparse.Namespace) -> None:
    if getattr(args, "data_root", None):
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    if getattr(args, "paths_config", None):
        os.environ["MOVEBENCH_PATHS_CONFIG"] = str(args.paths_config)


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Download public MoveBench study snapshots or use the GEE downloader.",
        description=(
            "Download public MoveBench study snapshots from the published ZIP release. "
            "Use `movebench download gee ...` for GEE raster downloads."
        ),
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Studies to download, or a named set like 'mini'. Default: all public studies.",
    )
    parser.add_argument(
        "--layers-config",
        default=None,
        help=(
            "Optional layers config filter for public ZIP downloads. "
            "Only rasters needed for this config will be extracted."
        ),
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Number of parallel HTTP workers for public ZIP downloads "
            "(default: 1; use >1 to enable parallel fetching)."
        ),
    )
    parser.set_defaults(func=run)

    download_subparsers = parser.add_subparsers(dest="download_command", required=False)
    gee_parser = download_subparsers.add_parser(
        "gee",
        help="Download rasters for existing local studies via Google Earth Engine.",
        description="Download rasters for existing local studies via Google Earth Engine.",
    )
    gee_parser.add_argument(
        "--layers-config",
        default=argparse.SUPPRESS,
        help="Path to a layers YAML config (default: configs/layers/all.yaml).",
    )
    gee_parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to download (e.g. --splits train test). Default: all splits.",
    )
    gee_parser.add_argument(
        "--n-workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of multiprocessing workers for chip downloads (default: 32).",
    )
    gee_parser.add_argument(
        "--studies",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Only download these studies or a named set like 'mini'. Default: all available studies.",
    )
    gee_parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing rasters on disk.",
    )
    for target, default in ((parser, None), (gee_parser, argparse.SUPPRESS)):
        target.add_argument(
            "--data-root", default=default,
            help="Override MOVEBENCH_DATA_ROOT for this run.",
        )
        target.add_argument(
            "--paths-config", default=default,
            help="Override MOVEBENCH_PATHS_CONFIG for this run.",
        )
    gee_parser.set_defaults(func=run_gee)


def run(args: argparse.Namespace) -> int:
    _apply_common_path_overrides(args)
    workers = 1 if args.n_workers is None else args.n_workers
    return int(
        download_public_studies(
            studies=args.studies,
            range_workers=workers,
            range_prefetch_chunks=workers,
            layers_config=args.layers_config,
        )
    )


def run_gee(args: argparse.Namespace) -> int:
    _apply_common_path_overrides(args)
    args.n_workers = 32 if args.n_workers is None else args.n_workers
    args.layers_config = args.layers_config or "configs/layers/all.yaml"

    all_names = Study.list_available()
    names = resolve_study_selection(args.studies)

    print(f"Studies to download: {len(names)}")
    if args.layers_config:
        print(f"Layers config: {args.layers_config}")
        print(
            f"Rasters hash: {REGISTRY.resolve(load_layers_config(args.layers_config)).rasters_hash}"
        )
    if args.splits:
        print(f"Splits: {args.splits}")
    print(f"Workers: {args.n_workers}")
    print()

    any_failed = False
    for i, name in enumerate(names, 1):
        if name not in all_names:
            print(f"[{i}/{len(names)}] SKIP {name} (not found)")
            continue

        print(f"[{i}/{len(names)}] Downloading {name}...")
        try:
            study = Study(name)
            result = study.download_rasters(
                layers_config=args.layers_config,
                splits=args.splits,
                n_workers=args.n_workers,
                skip_existing=not args.no_skip_existing,
            )
            any_failed |= _download_failure_count(result) > 0
            print(f"[{i}/{len(names)}] {_format_study_download_summary(name, result)}")
            for detail in _study_failure_details(result):
                print(f"  {detail}")
        except Exception as e:
            any_failed = True
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 2 if any_failed else 0
