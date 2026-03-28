from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.study import Study


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
                    total += int(year_stats.get("downloaded", 0)) + int(
                        year_stats.get("existing", 0)
                    )
            continue
        total += int(layer_stats.get("downloaded", 0)) + int(
            layer_stats.get("existing", 0)
        )
    return total


def _format_study_download_summary(name: str, result: dict[str, Any] | None) -> str:
    failed = 0
    report_path: str | None = None
    if not isinstance(result, dict):
        chip_present = 0
        tile_present = 0
    else:
        chip_stats = result.get("chips")
        if isinstance(chip_stats, dict) and chip_stats.get("report_path"):
            report_path = str(chip_stats["report_path"])
        failed += (
            int(chip_stats.get("failed", 0)) if isinstance(chip_stats, dict) else 0
        )
        chip_present = (
            int(chip_stats.get("downloaded", 0)) + int(chip_stats.get("existing", 0))
            if isinstance(chip_stats, dict)
            else 0
        )
        tile_stats = result.get("tiles")
        tile_present = _count_present_tiles(tile_stats)
        if isinstance(tile_stats, dict):
            for layer_stats in tile_stats.values():
                if not isinstance(layer_stats, dict):
                    continue
                years = layer_stats.get("years")
                if isinstance(years, dict):
                    for year_stats in years.values():
                        if isinstance(year_stats, dict):
                            failed += int(year_stats.get("failed", 0))
                else:
                    failed += int(layer_stats.get("failed", 0))
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


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "download",
        help="Download rasters for all available studies.",
        description="Download rasters for all available studies.",
    )
    parser.add_argument(
        "--layers-config",
        default="configs/layers/all.yaml",
        help="Path to a layers YAML config (default: configs/layers/all.yaml).",
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

    all_names = Study.list_available()
    names = args.studies if args.studies else all_names

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
            print(f"[{i}/{len(names)}] {_format_study_download_summary(name, result)}")
            for detail in _study_failure_details(result):
                print(f"  {detail}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")
    return 0
