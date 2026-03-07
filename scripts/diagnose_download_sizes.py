#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _insert_repo_root_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _default_config_path() -> str:
    ae = _repo_root() / "configs" / "layers_all_ae.yaml"
    if ae.exists():
        return str(ae)
    return str(_repo_root() / "configs" / "layers_all.yaml")


def _format_bytes(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n_f = float(n)
    if n_f < 0:
        return f"-{_format_bytes(-n_f)}"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n_f >= 1024.0 and i < len(units) - 1:
        n_f /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n_f)}{units[i]}"
    return f"{n_f:.2f}{units[i]}"


def _du_apparent_size_bytes(path: Path) -> int | None:
    if os.name != "posix":
        return None
    if shutil.which("du") is None:
        return None
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True).strip()
        # Typical: "<bytes>\t<path>"
        return int(out.split()[0])
    except Exception:
        return None


def _dir_size_bytes(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    du = _du_apparent_size_bytes(path)
    if du is not None:
        return int(du)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += int((Path(root) / f).stat().st_size)
            except FileNotFoundError:
                continue
    return int(total)


def _count_files(path: Path, *, suffix: str = ".tif") -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1 if path.name.endswith(suffix) else 0
    n = 0
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_file() and e.name.endswith(suffix):
                    n += 1
        return int(n)
    except FileNotFoundError:
        return 0


def closest_years(requested: list[int], available: list[int]) -> list[int]:
    """Resolve requested years to the closest available years (ties choose the smaller year)."""
    if not requested:
        return []
    if not available:
        return sorted(set(int(y) for y in requested))

    def _closest(y: int) -> int:
        return min(available, key=lambda a: (abs(int(a) - int(y)), int(a)))

    return sorted({_closest(int(y)) for y in requested})


@dataclass(frozen=True)
class SizeModel:
    """Heuristic disk-size model: bytes ~= intercept*n_files + slope*pixel_bands."""

    intercept_bytes_per_file: float
    slope_bytes_per_pixel_band: float

    def estimate(self, *, n_files: int, pixel_bands: float) -> float:
        return (float(self.intercept_bytes_per_file) * float(n_files)) + (
            float(self.slope_bytes_per_pixel_band) * float(pixel_bands)
        )


DEFAULT_FALLBACK_MODEL = SizeModel(intercept_bytes_per_file=8_192.0, slope_bytes_per_pixel_band=2.5)

# Default heuristic models tuned for the current default configs (chips+tiles written as float32 GeoTIFFs).
DEFAULT_SIZE_MODELS: dict[str, SizeModel] = {
    # Chip groups (configs/layers_all_ae.yaml)
    "era5": SizeModel(intercept_bytes_per_file=8_000.0, slope_bytes_per_pixel_band=2.9),
    "modis": SizeModel(intercept_bytes_per_file=0.0, slope_bytes_per_pixel_band=2.0),
    "high_res": SizeModel(intercept_bytes_per_file=25_000.0, slope_bytes_per_pixel_band=3.0),
    # Tile layers (configs/layers_all_ae.yaml)
    "alphaearth_embeddings": SizeModel(intercept_bytes_per_file=0.0, slope_bytes_per_pixel_band=1.25),
    "dem": SizeModel(intercept_bytes_per_file=0.0, slope_bytes_per_pixel_band=3.0),
    "dist_water": SizeModel(intercept_bytes_per_file=0.0, slope_bytes_per_pixel_band=2.5),
}


def _load_size_models(models_json: str | None) -> dict[str, SizeModel]:
    models = dict(DEFAULT_SIZE_MODELS)
    if not models_json:
        return models
    p = Path(models_json)
    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"--models-json must contain an object mapping type->params; got {type(payload)!r}")
    for k, v in payload.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        if "intercept_bytes_per_file" not in v or "slope_bytes_per_pixel_band" not in v:
            continue
        models[k] = SizeModel(
            intercept_bytes_per_file=float(v["intercept_bytes_per_file"]),
            slope_bytes_per_pixel_band=float(v["slope_bytes_per_pixel_band"]),
        )
    return models


def _years_on_disk(tile_dir: Path) -> list[int]:
    tile_dir = Path(tile_dir)
    if not tile_dir.exists():
        return []
    years: set[int] = set()
    pat_idx = re.compile(r"^index_(\d{4})\.json$")
    pat_tile = re.compile(r"^tile_\d{3}_(\d{4})\.tif$")
    for p in tile_dir.iterdir():
        m = pat_idx.match(p.name)
        if m:
            years.add(int(m.group(1)))
            continue
        m = pat_tile.match(p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")


def _parse_duration_seconds(raw: str, *, field_name: str) -> int:
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string duration like '7d'")
    match = _DURATION_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"{field_name} must match <int><unit> with unit in {{s,m,h,d}}, got {raw!r}")

    value = int(match.group("value"))
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {raw!r}")

    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group("unit")]
    return int(value) * int(multiplier)


def _expanded_bounds_to_rects(
    min_lon_exp: float,
    min_lat_exp: float,
    max_lon_exp: float,
    max_lat_exp: float,
) -> list[tuple[float, float, float, float]]:
    min_lat_exp = max(float(min_lat_exp), -89.9999)
    max_lat_exp = min(float(max_lat_exp), 89.9999)
    if min_lat_exp >= max_lat_exp:
        return []

    span = float(max_lon_exp) - float(min_lon_exp)
    if span <= 0.0:
        return []
    if span >= 360.0:
        return [(-180.0, min_lat_exp, 180.0, max_lat_exp)]

    start = ((float(min_lon_exp) + 180.0) % 360.0) - 180.0
    end = start + span

    rects: list[tuple[float, float, float, float]]
    if end > 180.0:
        rects = [
            (start, min_lat_exp, 180.0, max_lat_exp),
            (-180.0, min_lat_exp, end - 360.0, max_lat_exp),
        ]
    else:
        rects = [(start, min_lat_exp, end, max_lat_exp)]

    out: list[tuple[float, float, float, float]] = []
    for a, b, c, d in rects:
        a = max(float(a), -180.0)
        c = min(float(c), 180.0)
        b = float(b)
        d = float(d)
        if a >= c or b >= d:
            continue
        out.append((a, b, c, d))
    return out


def compute_dynamic_extent_buffers_km(
    df,
    *,
    train_df,
    min_extent_buffer_m: float,
    max_extent_buffer_m: float,
    max_step_time_s: int,
    step_q: float,
    step_multiplier: float,
    gap_q: float,
    time_col: str = "timestamp",
    window_s: int = 30 * 86400,
):
    import numpy as np
    import pandas as pd

    _insert_repo_root_on_path()
    from movebench.trajectories import prepare_interval_step_starts

    min_extent_buffer_m = float(min_extent_buffer_m)
    max_extent_buffer_m = float(max_extent_buffer_m)
    if min_extent_buffer_m <= 0.0:
        raise ValueError(f"min_extent_buffer_m must be > 0, got {min_extent_buffer_m}")
    if max_extent_buffer_m < min_extent_buffer_m:
        raise ValueError(
            f"max_extent_buffer_m must be >= min_extent_buffer_m ({min_extent_buffer_m}), got {max_extent_buffer_m}"
        )
    window_s = int(window_s)
    if window_s <= 0:
        raise ValueError(f"window_s must be > 0, got {window_s}")

    query_times = pd.to_datetime(df[time_col], errors="coerce")
    query_mask = query_times.notna()
    if not bool(query_mask.any()):
        raise ValueError(f"No valid {time_col!r} values available for dynamic extent estimation")

    out = pd.Series(np.nan, index=df.index, dtype=np.float64, name="extent_buffer_km")

    train = train_df.copy()
    train[time_col] = pd.to_datetime(train[time_col], errors="coerce")
    train = train.dropna(subset=[time_col]).copy()

    prepared = prepare_interval_step_starts(train, step_time_s=int(max_step_time_s), gap_q=float(gap_q))
    valid = prepared.loc[prepared["valid_step_start"], [time_col, "step_length_m"]].copy()
    valid = valid.dropna(subset=[time_col, "step_length_m"]).sort_values(time_col).reset_index(drop=True)

    if len(valid) == 0:
        out.loc[query_mask] = max_extent_buffer_m / 1000.0
    else:
        step_times = valid[time_col].astype("int64").to_numpy(dtype=np.int64, copy=False)
        step_lengths_m = valid["step_length_m"].to_numpy(dtype=np.float64, copy=False)

        half_window_ns = int(round(float(window_s) * 1_000_000_000.0 / 2.0))
        query_ns = query_times.loc[query_mask].astype("int64").to_numpy(dtype=np.int64, copy=False)
        unique_query_ns, inverse = np.unique(query_ns, return_inverse=True)
        unique_buffers_m = np.empty(len(unique_query_ns), dtype=np.float64)

        for idx, ts_ns in enumerate(unique_query_ns):
            left = int(np.searchsorted(step_times, ts_ns - half_window_ns, side="left"))
            right = int(np.searchsorted(step_times, ts_ns + half_window_ns, side="right"))
            if right <= left:
                buffer_m = max_extent_buffer_m
            else:
                buffer_m = float(step_multiplier) * float(np.nanquantile(step_lengths_m[left:right], q=float(step_q)))
                if not np.isfinite(buffer_m):
                    buffer_m = max_extent_buffer_m
            unique_buffers_m[idx] = min(max(buffer_m, min_extent_buffer_m), max_extent_buffer_m)

        out.loc[query_mask] = unique_buffers_m[inverse] / 1000.0

    valid_out = out.dropna()
    if len(valid_out) == 0:
        raise ValueError("Dynamic extent estimation produced no valid extent buffers")

    summary = {
        "mode": "rolling_window",
        "min_km": float(valid_out.min()),
        "median_km": float(valid_out.median()),
        "max_km": float(valid_out.max()),
        "lower_bound_km": float(min_extent_buffer_m) / 1000.0,
        "upper_bound_km": float(max_extent_buffer_m) / 1000.0,
        "window_s": int(window_s),
    }
    return out, summary


def estimate_tile_count(
    df,
    *,
    extent_buffer_km: float | None = None,
    extent_buffer_km_col: str | None = None,
    grid_size_m: float,
    set_col: str,
    lon_col: str,
    lat_col: str,
    max_buckets: int = 5000,
    max_coarsen_iters: int = 4,
) -> int:
    """Estimate number of (lon_bin, lat_bin) grid cells intersecting the union rectangles.

    This mirrors the rectangle construction in `movebench.layers.download.download_tile_layer`, but replaces
    Earth Engine's `coveringGrid(...).filter(intersects)` with a pure-Python grid index set.

    IMPORTANT: grid origin is (0° lon, 0° lat). Empirically this matches on-disk tile bounds.
    """
    import numpy as np
    import pandas as pd

    lon = df[lon_col].astype(float)
    lat = df[lat_col].astype(float)
    if lon.isna().any() or lat.isna().any():
        raise ValueError("trajectories df contains NaN lon/lat values")

    ref_lat = float(lat.median())
    lon_deg_to_m_ref = 111320.0 * math.cos(math.radians(ref_lat))
    bucket_deg_lon = 360.0 if abs(lon_deg_to_m_ref) < 1e-6 else float(grid_size_m) / lon_deg_to_m_ref
    bucket_deg_lat = float(grid_size_m) / 110540.0
    bucket_deg_lon = max(float(bucket_deg_lon), 1e-6)
    bucket_deg_lat = max(float(bucket_deg_lat), 1e-6)

    if extent_buffer_km_col is not None:
        if extent_buffer_km_col not in df.columns:
            raise ValueError(f"trajectories df missing extent buffer column: {extent_buffer_km_col!r}")
        extent_values_km = df[extent_buffer_km_col].astype(float)
        if extent_values_km.isna().any():
            raise ValueError(f"trajectories df contains NaN values in {extent_buffer_km_col!r}")
        if (extent_values_km <= 0).any():
            raise ValueError(f"trajectories df contains non-positive values in {extent_buffer_km_col!r}")
    elif extent_buffer_km is None:
        raise ValueError("estimate_tile_count requires extent_buffer_km or extent_buffer_km_col")

    base = pd.DataFrame({"split": df[set_col].astype(str), "lon": lon, "lat": lat, "lat_abs": lat.abs()})
    if extent_buffer_km_col is not None:
        base["extent_buffer_km"] = extent_values_km.to_numpy(dtype=float, copy=False)

    grouped = None
    b_lon = float(bucket_deg_lon)
    b_lat = float(bucket_deg_lat)
    for _ in range(int(max_coarsen_iters) + 1):
        lon_bin = np.floor((base["lon"] + 180.0) / b_lon).astype("int64")
        lat_bin = np.floor((base["lat"] + 90.0) / b_lat).astype("int64")
        g = (
            base.assign(lon_bin=lon_bin, lat_bin=lat_bin)
            .groupby(["split", "lon_bin", "lat_bin"], sort=False)
            .agg(
                min_lon=("lon", "min"),
                max_lon=("lon", "max"),
                min_lat=("lat", "min"),
                max_lat=("lat", "max"),
                max_abs_lat=("lat_abs", "max"),
            )
            .reset_index()
        )
        if len(g) <= int(max_buckets):
            grouped = g
            break
        b_lon *= 2.0
        b_lat *= 2.0

    grouped_base = None
    group_cols: list[str]

    if grouped is None:
        grouped = (
            base.groupby(["split"], sort=False)
            .agg(
                min_lon=("lon", "min"),
                max_lon=("lon", "max"),
                min_lat=("lat", "min"),
                max_lat=("lat", "max"),
                max_abs_lat=("lat_abs", "max"),
            )
            .reset_index()
        )
        grouped_base = base
        group_cols = ["split"]
    else:
        grouped_base = base.assign(
            lon_bin=np.floor((base["lon"] + 180.0) / b_lon).astype("int64"),
            lat_bin=np.floor((base["lat"] + 90.0) / b_lat).astype("int64"),
        )
        group_cols = ["split", "lon_bin", "lat_bin"]

    cell_deg = float(grid_size_m) / 111320.0
    if cell_deg <= 0:
        raise ValueError(f"Invalid cell_deg computed from grid_size_m={grid_size_m}")

    cells: set[tuple[int, int]] = set()

    if extent_buffer_km_col is None:
        for _, r in grouped.iterrows():
            buffer_m = float(extent_buffer_km) * 1000.0

            buffer_deg_lat = buffer_m / 110540.0
            lon_deg_to_m = 111320.0 * math.cos(math.radians(float(r["max_abs_lat"])))
            buffer_deg_lon = 180.0 if abs(lon_deg_to_m) < 1e-6 else buffer_m / lon_deg_to_m

            min_lon_exp = float(r["min_lon"]) - buffer_deg_lon
            max_lon_exp = float(r["max_lon"]) + buffer_deg_lon
            min_lat_exp = float(r["min_lat"]) - buffer_deg_lat
            max_lat_exp = float(r["max_lat"]) + buffer_deg_lat

            rects = _expanded_bounds_to_rects(min_lon_exp, min_lat_exp, max_lon_exp, max_lat_exp)

            for a, b, c, d in rects:
                i_min = math.floor(a / cell_deg)
                i_max = math.ceil(c / cell_deg) - 1
                j_min = math.floor(b / cell_deg)
                j_max = math.ceil(d / cell_deg) - 1

                for i in range(int(i_min), int(i_max) + 1):
                    for j in range(int(j_min), int(j_max) + 1):
                        cells.add((i, j))
        return int(len(cells))

    for _, group_df in grouped_base.groupby(group_cols, sort=False):
        lon_vals = group_df["lon"].to_numpy(dtype=float, copy=False)
        lat_vals = group_df["lat"].to_numpy(dtype=float, copy=False)
        buffer_m = group_df["extent_buffer_km"].to_numpy(dtype=float, copy=False) * 1000.0
        buffer_deg_lat = buffer_m / 110540.0
        lon_deg_to_m = 111320.0 * np.cos(np.deg2rad(lat_vals))
        buffer_deg_lon = np.where(np.abs(lon_deg_to_m) < 1e-6, 180.0, buffer_m / lon_deg_to_m)

        ref_lon = float(np.median(lon_vals))
        lon_unwrapped = ref_lon + (((lon_vals - ref_lon) + 180.0) % 360.0) - 180.0

        min_lon_exp = float(np.min(lon_unwrapped - buffer_deg_lon))
        max_lon_exp = float(np.max(lon_unwrapped + buffer_deg_lon))
        min_lat_exp = float(np.min(lat_vals - buffer_deg_lat))
        max_lat_exp = float(np.max(lat_vals + buffer_deg_lat))

        rects = _expanded_bounds_to_rects(min_lon_exp, min_lat_exp, max_lon_exp, max_lat_exp)

        for a, b, c, d in rects:
            i_min = math.floor(a / cell_deg)
            i_max = math.ceil(c / cell_deg) - 1
            j_min = math.floor(b / cell_deg)
            j_max = math.ceil(d / cell_deg) - 1

            for i in range(int(i_min), int(i_max) + 1):
                for j in range(int(j_min), int(j_max) + 1):
                    cells.add((i, j))

    return int(len(cells))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Estimate per-study MoveBench raster download sizes. "
            "Default mode is pre-download planning; enable --compare-on-disk to verify against existing downloads."
        )
    )
    p.add_argument("--config", default=_default_config_path(), help="Path to a layers YAML config.")
    p.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Optional list of study ids to scan. Default: all available studies.",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional list of splits to include (e.g. --splits train test). Default: all splits present.",
    )
    p.add_argument("--rasters-hash", default=None, help="Override rasters_hash (default: derived from config).")
    p.add_argument(
        "--models-json",
        default=None,
        help=(
            "Optional JSON file overriding size models. "
            "Format: {\"type\": {\"intercept_bytes_per_file\": <float>, \"slope_bytes_per_pixel_band\": <float>}, ...}"
        ),
    )
    p.add_argument(
        "--annual-available-years",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional available years list for annual tile layers; requested years are mapped to closest available "
            "years using the downloader tie-break (ties choose the smaller year)."
        ),
    )
    p.add_argument(
        "--compare-on-disk",
        action="store_true",
        help="If set, compute actual on-disk bytes and print estimate vs actual.",
    )
    p.add_argument(
        "--extent-mode",
        choices=["study_wide", "rolling_window", "rolling_month"],
        default="study_wide",
        help=(
            "How to estimate download extent. 'study_wide' uses one study-wide extent buffer for all points; "
            "'rolling_window' estimates a per-point extent from a centered time window around each point. "
            "'rolling_month' is kept as a compatibility alias for 'rolling_window'."
        ),
    )
    p.add_argument(
        "--extent-window",
        default="30d",
        help=(
            "Duration string for rolling-window extent estimation, using the same syntax as config durations "
            "(for example: 1h, 1d, 7d, 30d). Ignored unless --extent-mode rolling_window is used."
        ),
    )
    p.add_argument(
        "--no-file-counts",
        action="store_true",
        help="With --compare-on-disk, skip counting .tif files (faster; disables file-count mismatch warnings).",
    )
    p.add_argument("--out-json", default=None, help="Optional path to write a JSON report.")
    p.add_argument("--out-csv", default=None, help="Optional path to write a CSV report.")
    p.add_argument("--data-root", default=None, help="Override MOVEBENCH_DATA_ROOT for this run.")
    p.add_argument("--paths-config", default=None, help="Override MOVEBENCH_PATHS_CONFIG for this run.")
    args = p.parse_args(argv[1:])

    if args.data_root:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    if args.paths_config:
        os.environ["MOVEBENCH_PATHS_CONFIG"] = str(args.paths_config)

    _insert_repo_root_on_path()

    try:
        import numpy as np
        import pandas as pd

        from movebench.layers.config import load_layers_config
        from movebench.layers.download import (
            LAT_COL,
            LON_COL,
            POINT_ID_COL,
            SET_COL,
            TIME_COL,
            compute_group_scale,
            get_buffer_m_for_study,
        )
        from movebench.layers.registry import REGISTRY
        from movebench.study import Study
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")
        print("Make sure movebench dependencies are installed (notably pandas and numpy).")
        return 1

    extent_mode = "rolling_window" if str(args.extent_mode) == "rolling_month" else str(args.extent_mode)
    extent_window = str(args.extent_window).strip()
    extent_window_s: int | None = None
    if extent_mode == "rolling_window":
        try:
            extent_window_s = _parse_duration_seconds(extent_window, field_name="--extent-window")
        except ValueError as e:
            print(f"Invalid --extent-window: {e}")
            return 1

    cfg = load_layers_config(args.config)
    plan = REGISTRY.resolve(cfg)
    rasters_hash = str(args.rasters_hash) if args.rasters_hash else str(plan.rasters_hash)

    chip_groups = list(plan.chip_groups.keys())
    tile_layers = [l for l in plan.layers_in_order if REGISTRY.get(l).source == "gee_tile"]

    clamp = cfg.raster_params["scale_clamp"]
    min_pixels = int(clamp["min_pixels"])
    max_pixels = int(clamp["max_pixels"])
    px_per_side = int(cfg.raster_params["tiles"]["px_per_side"])
    buf_cfg = cfg.raster_params["buffer"]
    step_multiplier = float(buf_cfg["step_multiplier"])
    step_q = float(buf_cfg["step_q"])
    gap_q = float(buf_cfg["gap_q"])
    min_step_time_s = buf_cfg.get("min_step_time_s")
    max_step_time_s = buf_cfg.get("max_step_time_s")

    all_studies = Study.list_available()
    study_ids = [str(x) for x in (args.studies if args.studies else all_studies)]
    study_ids = [s for s in study_ids if s in set(all_studies)]
    if not study_ids:
        print("No studies found under the configured studies root.")
        return 0

    try:
        size_models = _load_size_models(args.models_json)
    except Exception as e:
        print(f"Failed to load --models-json: {e}")
        return 1

    report: dict[str, Any] = {
        "config": str(args.config),
        "rasters_hash": str(rasters_hash),
        "compare_on_disk": bool(args.compare_on_disk),
        "extent_mode": str(extent_mode),
        "extent_window": str(extent_window) if extent_mode == "rolling_window" else None,
        "no_file_counts": bool(args.no_file_counts),
        "chip_groups": list(chip_groups),
        "tile_layers": list(tile_layers),
        "models": {
            k: {
                "intercept_bytes_per_file": float(v.intercept_bytes_per_file),
                "slope_bytes_per_pixel_band": float(v.slope_bytes_per_pixel_band),
            }
            for k, v in size_models.items()
        },
        "fallback_model": {
            "intercept_bytes_per_file": float(DEFAULT_FALLBACK_MODEL.intercept_bytes_per_file),
            "slope_bytes_per_pixel_band": float(DEFAULT_FALLBACK_MODEL.slope_bytes_per_pixel_band),
        },
        "annual_available_years": list(args.annual_available_years) if args.annual_available_years else None,
        "studies": [],
    }

    failures = 0
    rel_errors: list[float] = []
    rel_errors_complete: list[float] = []
    total_est_payload_bytes = 0
    total_est_disk_bytes = 0
    total_actual_bytes = 0
    total_processed = 0

    for i, sid in enumerate(study_ids, 1):
        print(f"[{i}/{len(study_ids)}] {sid}")
        try:
            study = Study(str(sid))
        except Exception as e:
            failures += 1
            report["studies"].append({"study_id": str(sid), "status": "skip", "error": str(e)})
            print(f"  SKIP: {e}")
            continue

        try:
            df = pd.read_csv(study.trajectories_path, low_memory=False)
        except Exception as e:
            failures += 1
            report["studies"].append({"study_id": str(sid), "status": "skip", "error": f"read_csv_failed: {e}"})
            print(f"  SKIP: read_csv_failed: {e}")
            continue

        # Match downloader cleaning.
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
        df = df.dropna(subset=[TIME_COL]).copy()

        used_splits = [str(s) for s in (args.splits if args.splits else sorted(df[SET_COL].dropna().unique().tolist()))]
        df = df[df[SET_COL].isin(used_splits)].copy()

        if len(df) == 0:
            failures += 1
            report["studies"].append({"study_id": study.study_id, "status": "skip", "error": "no_rows_after_split_filter"})
            print("  SKIP: no rows after split filter")
            continue

        # Compute buffer from train split only (same as download_rasters).
        train_df = df[df[SET_COL] == "train"].copy()
        if len(train_df) == 0:
            failures += 1
            report["studies"].append({"study_id": study.study_id, "status": "skip", "error": "no_train_split"})
            print("  SKIP: no train split rows (cannot compute buffer)")
            continue

        try:
            resolution_buffer_m = float(
                get_buffer_m_for_study(
                    train_df,
                    step_time_s=min_step_time_s,
                    step_multiplier=float(step_multiplier),
                    step_q=float(step_q),
                    gap_q=float(gap_q),
                )
            )
            if max_step_time_s == min_step_time_s:
                extent_buffer_m = float(resolution_buffer_m)
            else:
                extent_buffer_m = float(
                    get_buffer_m_for_study(
                        train_df,
                        step_time_s=max_step_time_s,
                        step_multiplier=float(step_multiplier),
                        step_q=float(step_q),
                        gap_q=float(gap_q),
                    )
                )
        except Exception as e:
            failures += 1
            report["studies"].append({"study_id": study.study_id, "status": "skip", "error": f"buffer_failed: {e}"})
            print(f"  SKIP: buffer_failed: {e}")
            continue

        resolution_buffer_km = float(resolution_buffer_m) / 1000.0
        extent_buffer_km = float(extent_buffer_m) / 1000.0
        dynamic_extent_col = "_extent_buffer_km_est"
        extent_buffer_summary: dict[str, Any] = {
            "mode": "study_wide",
            "min_km": float(extent_buffer_km),
            "median_km": float(extent_buffer_km),
            "max_km": float(extent_buffer_km),
            "lower_bound_km": float(extent_buffer_km),
            "upper_bound_km": float(extent_buffer_km),
        }

        if extent_mode == "rolling_window":
            if min_step_time_s is None or max_step_time_s is None:
                failures += 1
                report["studies"].append(
                    {
                        "study_id": study.study_id,
                        "status": "skip",
                        "error": "rolling_window extent mode requires both min_step_time and max_step_time",
                    }
                )
                print("  SKIP: rolling_window extent mode requires both min_step_time and max_step_time")
                continue

            try:
                extent_buffer_series_km, extent_buffer_summary = compute_dynamic_extent_buffers_km(
                    df,
                    train_df=train_df,
                    min_extent_buffer_m=float(resolution_buffer_m),
                    max_extent_buffer_m=float(extent_buffer_m),
                    max_step_time_s=int(max_step_time_s),
                    step_q=float(step_q),
                    step_multiplier=float(step_multiplier),
                    gap_q=float(gap_q),
                    time_col=TIME_COL,
                    window_s=int(extent_window_s),
                )
                extent_buffer_summary["window"] = str(extent_window)
            except Exception as e:
                failures += 1
                report["studies"].append(
                    {"study_id": study.study_id, "status": "skip", "error": f"dynamic_extent_failed: {e}"}
                )
                print(f"  SKIP: dynamic_extent_failed: {e}")
                continue

            df = df.copy()
            df[dynamic_extent_col] = extent_buffer_series_km.reindex(df.index).to_numpy(dtype=float, copy=False)

        # Compute unique point set per split (chip files keyed by point_id).
        split_points: dict[str, pd.DataFrame] = {}
        n_points_total = 0
        for s in used_splits:
            sub = df[df[SET_COL] == s].copy()
            if POINT_ID_COL in sub.columns:
                sub = sub.drop_duplicates(subset=[POINT_ID_COL])
            split_points[str(s)] = sub
            n_points_total += int(len(sub))

        chips_payload = 0.0
        chips_disk = 0.0
        tiles_payload = 0.0
        tiles_disk = 0.0

        chips_out: dict[str, Any] = {}
        for g in chip_groups:
            group = plan.chip_groups[g]
            layers = list(group["layers"])
            band_count = int(sum(len(REGISTRY.get(l).bands) for l in layers))
            target_scale_m = float(group["target_scale_m"])
            scale = float(
                compute_group_scale(
                    float(resolution_buffer_km),
                    float(target_scale_m),
                    min_pixels=int(min_pixels),
                    max_pixels=int(max_pixels),
                )
            )
            base = (2.0 * float(extent_buffer_m)) / float(scale)

            pixel_bands_total = 0.0
            files_expected = 0
            for _, sub in split_points.items():
                if len(sub) == 0:
                    continue
                lats = sub[LAT_COL].astype(float).to_numpy()
                sec_lat = 1.0 / np.maximum(np.cos(np.deg2rad(lats)), 1e-6)
                if dynamic_extent_col in sub.columns:
                    buffer_m = sub[dynamic_extent_col].astype(float).to_numpy(dtype=float, copy=False) * 1000.0
                    side = (2.0 * buffer_m) / float(scale)
                    pixel_bands_total += float(np.sum((side * side) * sec_lat * float(band_count)))
                else:
                    sec_sum = float(np.sum(sec_lat))
                    pixel_bands_total += float((base * base) * sec_sum * float(band_count))
                files_expected += int(len(sub))

            payload_bytes = float(pixel_bands_total) * 4.0
            model = size_models.get(g, DEFAULT_FALLBACK_MODEL)
            disk_bytes = float(model.estimate(n_files=int(files_expected), pixel_bands=float(pixel_bands_total)))

            chips_payload += float(payload_bytes)
            chips_disk += float(disk_bytes)

            chips_out[g] = {
                "band_count": int(band_count),
                "expected_files": int(files_expected),
                "pixel_bands": float(pixel_bands_total),
                "est_payload_bytes": int(round(payload_bytes)),
                "est_disk_bytes": int(round(disk_bytes)),
                "model_used": str(g) if g in size_models else "fallback",
            }

        tiles_out: dict[str, Any] = {}
        for layer in tile_layers:
            spec = REGISTRY.get(layer)
            if spec.native_scale_m is None:
                continue
            layer_scale = float(
                compute_group_scale(
                    float(resolution_buffer_km),
                    float(spec.native_scale_m),
                    min_pixels=int(min_pixels),
                    max_pixels=int(max_pixels),
                )
            )
            grid_size_m = float(layer_scale) * float(px_per_side)
            tile_count = int(
                estimate_tile_count(
                    df,
                    extent_buffer_km=float(extent_buffer_km),
                    extent_buffer_km_col=dynamic_extent_col if dynamic_extent_col in df.columns else None,
                    grid_size_m=float(grid_size_m),
                    set_col=SET_COL,
                    lon_col=LON_COL,
                    lat_col=LAT_COL,
                )
            )

            years: list[int] | None = None
            n_years = 1
            if str(spec.temporal) == "annual":
                yrs = df[TIME_COL].dt.year.dropna().astype("int64")
                requested_years = sorted(set(int(y) for y in yrs.tolist()))
                resolved_years: list[int] = list(requested_years)
                if args.compare_on_disk:
                    tile_dir = study.study_dir / "rasters" / rasters_hash / "tiles" / str(layer)
                    on_disk = _years_on_disk(tile_dir)
                    if on_disk:
                        resolved_years = list(on_disk)
                if (not args.compare_on_disk) and args.annual_available_years:
                    resolved_years = closest_years(requested_years, list(args.annual_available_years))
                years = list(resolved_years)
                n_years = int(len(resolved_years))

            band_count = int(len(spec.bands))
            files_expected = int(tile_count) * int(n_years)
            pixel_bands_total = (
                float(tile_count) * float(n_years) * float(px_per_side) * float(px_per_side) * float(band_count)
            )
            payload_bytes = float(pixel_bands_total) * 4.0
            model = size_models.get(layer, DEFAULT_FALLBACK_MODEL)
            disk_bytes = float(model.estimate(n_files=int(files_expected), pixel_bands=float(pixel_bands_total)))

            tiles_payload += float(payload_bytes)
            tiles_disk += float(disk_bytes)

            tiles_out[layer] = {
                "tile_count": int(tile_count),
                "band_count": int(band_count),
                "years": years,
                "expected_files": int(files_expected),
                "pixel_bands": float(pixel_bands_total),
                "est_payload_bytes": int(round(payload_bytes)),
                "est_disk_bytes": int(round(disk_bytes)),
                "model_used": str(layer) if layer in size_models else "fallback",
            }

        est_payload_total = float(chips_payload) + float(tiles_payload)
        est_disk_total = float(chips_disk) + float(tiles_disk)
        est_payload_total_i = int(round(est_payload_total))
        est_disk_total_i = int(round(est_disk_total))

        # Optional on-disk comparison.
        actual_total: int | None = None
        actual_by_type: dict[str, Any] | None = None
        file_count_mismatches: list[str] = []
        tile_index_errors: dict[str, Any] = {}
        rel_err: float | None = None

        if args.compare_on_disk:
            rasters_root = study.study_dir / "rasters" / rasters_hash
            actual_by_type = {"chips": {}, "tiles": {}}
            actual_total = 0

            for g in chip_groups:
                d = rasters_root / "chips" / g
                b = _dir_size_bytes(d)
                c = None if args.no_file_counts else _count_files(d, suffix=".tif")
                actual_by_type["chips"][g] = {"bytes": int(b), "files": int(c) if c is not None else None}
                actual_total += int(b)

                expected = int(chips_out.get(g, {}).get("expected_files", 0))
                if (not args.no_file_counts) and c is not None and expected != int(c) and (expected > 0 or int(c) > 0):
                    file_count_mismatches.append(f"chips/{g} expected={expected} actual={int(c)}")

            for layer in tile_layers:
                d = rasters_root / "tiles" / layer
                b = _dir_size_bytes(d)
                c = None if args.no_file_counts else _count_files(d, suffix=".tif")
                actual_by_type["tiles"][layer] = {"bytes": int(b), "files": int(c) if c is not None else None}
                actual_total += int(b)

                # Tile index errors (static: index.json, annual: index_YYYY.json).
                spec = REGISTRY.get(layer)
                tdir = d
                total_errors = 0
                if str(spec.temporal) == "annual":
                    pat = re.compile(r"^index_(\d{4})\.json$")
                    per_year: dict[int, Any] = {}
                    for pth in sorted(tdir.glob("index_*.json")):
                        m = pat.match(pth.name)
                        if not m:
                            continue
                        try:
                            payload = json.loads(pth.read_text())
                        except Exception:
                            payload = None
                        errs = payload.get("errors", []) if isinstance(payload, dict) else []
                        n_err = int(len(errs)) if isinstance(errs, list) else 0
                        total_errors += int(n_err)
                        per_year[int(m.group(1))] = {
                            "errors": int(n_err),
                            "n_tiles_reported": payload.get("n_tiles_reported") if isinstance(payload, dict) else None,
                            "n_tiles_effective": payload.get("n_tiles_effective") if isinstance(payload, dict) else None,
                        }
                    if per_year:
                        tile_index_errors[layer] = {"total_errors": int(total_errors), "per_year": per_year}
                else:
                    idx_path = tdir / "index.json"
                    if idx_path.exists():
                        try:
                            payload = json.loads(idx_path.read_text())
                        except Exception:
                            payload = None
                        errs = payload.get("errors", []) if isinstance(payload, dict) else []
                        n_err = int(len(errs)) if isinstance(errs, list) else 0
                        total_errors = int(n_err)
                        if n_err:
                            tile_index_errors[layer] = {"total_errors": int(n_err)}

                expected = int(tiles_out.get(layer, {}).get("expected_files", 0))
                mismatch_parts: list[str] = []
                if (not args.no_file_counts) and c is not None and expected != int(c) and (expected > 0 or int(c) > 0):
                    mismatch_parts.append(f"expected={expected} actual={int(c)}")
                if total_errors:
                    mismatch_parts.append(f"index_errors={int(total_errors)}")
                if mismatch_parts:
                    file_count_mismatches.append(f"tiles/{layer} " + " ".join(mismatch_parts))

            if int(actual_total) > 0:
                rel_err = float(est_disk_total - float(actual_total)) / float(actual_total)
                rel_errors.append(abs(float(rel_err)))
                if not file_count_mismatches:
                    rel_errors_complete.append(abs(float(rel_err)))

        total_processed += 1
        total_est_payload_bytes += int(est_payload_total_i)
        total_est_disk_bytes += int(est_disk_total_i)
        if args.compare_on_disk and actual_total is not None:
            total_actual_bytes += int(actual_total)

        study_out = {
            "study_id": study.study_id,
            "study_dir": str(study.study_dir),
            "n_points": int(n_points_total),
            "n_splits": int(len(used_splits)),
            "extent_mode": str(extent_mode),
            "extent_window": str(extent_window) if extent_mode == "rolling_window" else None,
            "buffer_km": float(extent_buffer_km),
            "resolution_buffer_km": float(resolution_buffer_km),
            "extent_buffer_km": float(extent_buffer_km),
            "extent_buffer_summary": extent_buffer_summary,
            "est_payload_bytes": int(est_payload_total_i),
            "est_disk_bytes": int(est_disk_total_i),
            "chips": chips_out,
            "tiles": tiles_out,
            "actual_bytes": int(actual_total) if actual_total is not None else None,
            "rel_error_disk": float(rel_err) if rel_err is not None else None,
            "actual": actual_by_type,
            "file_count_mismatches": file_count_mismatches,
            "tile_index_errors": tile_index_errors,
        }
        report["studies"].append(study_out)

        print(
            f"  points: {n_points_total}  resolution_buffer_km: {resolution_buffer_km:.3f}"
            f"  extent_buffer_km: {extent_buffer_km:.3f}"
        )
        if extent_mode == "rolling_window":
            print(
                "  extent_buffer_summary_km: "
                f"window={extent_buffer_summary['window']} "
                f"min={float(extent_buffer_summary['min_km']):.3f} "
                f"median={float(extent_buffer_summary['median_km']):.3f} "
                f"max={float(extent_buffer_summary['max_km']):.3f}"
            )
        print(f"  est_payload: {_format_bytes(est_payload_total)}")
        print(f"  est_disk:    {_format_bytes(est_disk_total)}")
        if args.compare_on_disk:
            print(f"  actual:      {_format_bytes(actual_total)}")
            if rel_err is not None:
                print(f"  rel_error:   {rel_err:+.2%}")
        if args.compare_on_disk and file_count_mismatches:
            preview = "; ".join(file_count_mismatches[:3])
            more = "" if len(file_count_mismatches) <= 3 else f" (+{len(file_count_mismatches) - 3} more)"
            print(f"  WARN: {preview}{more}")
        print()

    if args.compare_on_disk and rel_errors:
        mean_abs = float(sum(rel_errors)) / float(len(rel_errors))
        worst_abs = float(max(rel_errors))
        print(f"studies: {len(study_ids)}  skipped+failed: {failures}")
        print(f"abs(rel_error) disk_est: mean={mean_abs:.2%}  worst={worst_abs:.2%}")
        if rel_errors_complete and len(rel_errors_complete) != len(rel_errors):
            mean_abs_c = float(sum(rel_errors_complete)) / float(len(rel_errors_complete))
            worst_abs_c = float(max(rel_errors_complete))
            print(f"abs(rel_error) disk_est (no mismatches): mean={mean_abs_c:.2%}  worst={worst_abs_c:.2%}")
    else:
        print(f"studies: {len(study_ids)}  skipped+failed: {failures}")

    if total_processed:
        print(f"total (processed studies): {total_processed}")
        print(f"total_est_payload: {_format_bytes(total_est_payload_bytes)}")
        print(f"total_est_disk:    {_format_bytes(total_est_disk_bytes)}")
        if args.compare_on_disk:
            print(f"total_actual:      {_format_bytes(total_actual_bytes)}")
        print()

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "study_id",
                    "n_points",
                    "resolution_buffer_km",
                    "extent_buffer_km",
                    "est_payload_bytes",
                    "est_disk_bytes",
                    "actual_bytes",
                    "rel_error_disk",
                ],
            )
            w.writeheader()
            for s in report["studies"]:
                if s.get("status") == "skip":
                    continue
                w.writerow(
                    {
                        "study_id": s.get("study_id"),
                        "n_points": s.get("n_points"),
                        "resolution_buffer_km": s.get("resolution_buffer_km"),
                        "extent_buffer_km": s.get("extent_buffer_km"),
                        "est_payload_bytes": s.get("est_payload_bytes"),
                        "est_disk_bytes": s.get("est_disk_bytes"),
                        "actual_bytes": s.get("actual_bytes"),
                        "rel_error_disk": s.get("rel_error_disk"),
                    }
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
