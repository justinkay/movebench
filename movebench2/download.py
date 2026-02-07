from __future__ import annotations

import math
import multiprocessing as mp
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from movebench2.layers import REGISTRY
from movebench2.layers.plan import LayerPlan
from movebench2.movement import LAT_COL, LON_COL, POINT_ID_COL, SET_COL, TIME_COL, get_buffer_m_for_study


_ee_initialized = False
_ee = None


def ensure_ee_initialized():
    """Lazily initialize Google Earth Engine.

    This keeps imports working in environments without `earthengine-api`.
    """
    global _ee_initialized, _ee
    if not _ee_initialized:
        import ee

        _ee = ee
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
        _ee_initialized = True
    return _ee


def compute_group_scale(
    buffer_km: float,
    target_scale_m: float,
    *,
    min_pixels: int = 3,
    max_pixels: int = 100,
) -> float:
    """Clamp scale so chips have a reasonable pixel count per side."""
    r_m = buffer_km * 1000.0
    scale_for_max = (2.0 * r_m) / max_pixels
    scale_for_min = (2.0 * r_m) / min_pixels

    if target_scale_m < scale_for_max:
        return float(scale_for_max)
    if target_scale_m > scale_for_min:
        return float(scale_for_min)
    return float(target_scale_m)


def square_region_for_point(lon: float, lat: float, buffer_km: float):
    ee = ensure_ee_initialized()
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(buffer_km * 1000.0).bounds()


def _sleep_backoff_s(attempt: int, *, base_delay_s: float, backoff: float) -> float:
    return base_delay_s * (backoff**attempt)


def download_tiff(
    url: str,
    out_path: Path,
    *,
    tries: int = 5,
    base_delay_s: float = 2.0,
    backoff: float = 2.0,
    timeout_s: float = 300.0,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(tries):
        try:
            r = requests.get(url, stream=True, timeout=timeout_s)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
            return
        except requests.RequestException as e:
            last_exc = e
            if attempt == tries - 1:
                break
            time.sleep(_sleep_backoff_s(attempt, base_delay_s=base_delay_s, backoff=backoff))

    assert last_exc is not None
    raise last_exc


def download_image(img, region, scale_m: float, out_path: Path, *, crs: str = "EPSG:4326") -> None:
    """Download an ee.Image directly to GeoTIFF (no tiling/merge)."""
    params = {"scale": scale_m, "region": region, "format": "GEO_TIFF", "crs": crs}
    url = img.getDownloadURL(params)
    download_tiff(url, out_path)


def download_tiled_image(img, region, scale_m: float, out_path: Path, *, crs: str = "EPSG:4326") -> None:
    """Download a large ee.Image by splitting into tiles and merging when needed."""
    ee = ensure_ee_initialized()

    bounds_info = region.bounds().getInfo()
    coords = bounds_info["coordinates"][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    center_lat = (min_lat + max_lat) / 2.0
    lon_deg_to_m = 111320.0 * math.cos(math.radians(center_lat))
    lat_deg_to_m = 110540.0

    width_m = (max_lon - min_lon) * lon_deg_to_m
    height_m = (max_lat - min_lat) * lat_deg_to_m
    width_px = width_m / scale_m
    height_px = height_m / scale_m

    band_names = img.bandNames().getInfo()
    num_bands = len(band_names)
    bytes_per_pixel = 4

    estimated_size_bytes = width_px * height_px * num_bands * bytes_per_pixel
    gee_limit_bytes = 50_331_648

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if estimated_size_bytes <= gee_limit_bytes * 0.7:
        params = {"scale": scale_m, "region": region, "format": "GEO_TIFF", "crs": crs}
        url = img.getDownloadURL(params)
        download_tiff(url, out_path)
        return

    pixels_per_tile = (gee_limit_bytes * 0.35) / (num_bands * bytes_per_pixel)
    tile_side_px = math.sqrt(pixels_per_tile)
    tile_side_m = tile_side_px * scale_m

    tile_width_deg = tile_side_m / lon_deg_to_m
    tile_height_deg = tile_side_m / lat_deg_to_m

    n_cols = math.ceil((max_lon - min_lon) / tile_width_deg)
    n_rows = math.ceil((max_lat - min_lat) / tile_height_deg)

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    with tempfile.TemporaryDirectory() as tmp_dir:
        tile_paths: list[Path] = []
        tile_idx = 0
        for row in range(n_rows):
            for col in range(n_cols):
                tile_min_lon = min_lon + col * tile_width_deg
                tile_max_lon = min(tile_min_lon + tile_width_deg, max_lon)
                tile_min_lat = min_lat + row * tile_height_deg
                tile_max_lat = min(tile_min_lat + tile_height_deg, max_lat)

                tile_region = ee.Geometry.Rectangle([tile_min_lon, tile_min_lat, tile_max_lon, tile_max_lat])
                tile_path = Path(tmp_dir) / f"tile_{tile_idx:03d}.tif"
                params = {"scale": scale_m, "region": tile_region, "format": "GEO_TIFF", "crs": crs}

                url = img.getDownloadURL(params)
                download_tiff(url, tile_path)
                tile_paths.append(tile_path)
                tile_idx += 1

        if not tile_paths:
            raise RuntimeError("All tile downloads failed")

        old_cpl_log = os.environ.get("CPL_LOG", "")
        os.environ["CPL_LOG"] = "/dev/null"
        src_files = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, mosaic_transform = merge(src_files, resampling=Resampling.nearest)
            profile = src_files[0].profile.copy()
            profile.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": mosaic_transform})
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(mosaic)
        finally:
            for src in src_files:
                src.close()
            if old_cpl_log:
                os.environ["CPL_LOG"] = old_cpl_log
            else:
                os.environ.pop("CPL_LOG", None)


def _nearest_hourly_image(collection_id: str, timestamp: str, band_list: list[str]):
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    start = ts.advance(-1, "hour")
    end = ts.advance(1, "hour")
    col = (
        ee.ImageCollection(collection_id)
        .select(band_list)
        .filterDate(start, end)
        .sort("system:time_start")
    )
    return ee.Image(col.first())


def _nearest_daily_image(collection_id: str, timestamp: str, band_list: list[str]):
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    start = ts.advance(-1, "day")
    end = ts.advance(1, "day")
    col = (
        ee.ImageCollection(collection_id)
        .select(band_list)
        .filterDate(start, end)
        .sort("system:time_start")
    )
    return ee.Image(col.first())


def _nearest_modis_image(collection_id: str, timestamp: str, band_list: list[str], *, days_window: int):
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    start = ts.advance(-days_window, "day")
    end = ts.advance(days_window, "day")
    col = (
        ee.ImageCollection(collection_id)
        .select(band_list)
        .filterDate(start, end)
        .sort("system:time_start")
    )
    return ee.Image(col.first())


def _dynamic_world_image(spec, timestamp: str, region):
    ee = ensure_ee_initialized()

    bands = list(spec.bands)
    DW_START = "2015-06-23"

    try:
        ts_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        dw_start_dt = datetime(2015, 6, 23, tzinfo=ts_dt.tzinfo)
        was_clamped = ts_dt < dw_start_dt
    except Exception:
        was_clamped = False

    if was_clamped:
        col = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(DW_START, "2016-06-23")
            .filterBounds(region)
            .select(bands)
        )
    else:
        ts = ee.Date(timestamp)
        col_30 = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(ts.advance(-15, "day"), ts.advance(15, "day"))
            .filterBounds(region)
            .select(bands)
        )
        col_60 = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(ts.advance(-30, "day"), ts.advance(30, "day"))
            .filterBounds(region)
            .select(bands)
        )
        col = ee.ImageCollection(ee.Algorithms.If(col_30.size().gt(0), col_30, col_60))

    zero_img = ee.Image([0] * len(bands)).rename(bands).toFloat()
    img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), zero_img)).unmask(0)
    return img.rename(bands).toFloat()


def _landsat_ndwi_image(spec, timestamp: str, region):
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    bands = list(spec.bands)

    col = (
        ee.ImageCollection(spec.gee_collection)
        .filterDate(ts.advance(-32, "day"), ts.advance(32, "day"))
        .filterBounds(region)
        .select(bands)
    )

    zero_img = ee.Image(0).rename(bands).toFloat()
    img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), zero_img)).unmask(0)
    return img.rename(bands).toFloat()


def build_chip_layer_image(spec, timestamp: str, region):
    ee = ensure_ee_initialized()

    if spec.name == "dynamic_world":
        img = _dynamic_world_image(spec, timestamp, region)
    elif spec.name == "landsat_ndwi":
        img = _landsat_ndwi_image(spec, timestamp, region)
    elif spec.name in ("modis_lst", "modis_vi"):
        days_window = 4 if spec.name == "modis_lst" else 10
        img = _nearest_modis_image(spec.gee_collection, timestamp, spec.bands, days_window=days_window)
    elif spec.name in ("modis_ndsi", "modis_bai"):
        img = _nearest_modis_image(spec.gee_collection, timestamp, spec.bands, days_window=10)
    elif spec.temporal == "hourly":
        img = _nearest_hourly_image(spec.gee_collection, timestamp, spec.bands)
    elif spec.temporal == "daily":
        img = _nearest_daily_image(spec.gee_collection, timestamp, spec.bands)
    else:
        raise ValueError(f"Unsupported gee_chip temporal for layer {spec.name!r}: {spec.temporal!r}")

    return ee.Image(img).rename(list(spec.bands)).toFloat()


def build_chip_group_image(layer_names: list[str], timestamp: str, region):
    ee = ensure_ee_initialized()

    stack = None
    for layer_name in layer_names:
        spec = REGISTRY.get(layer_name)
        layer_img = build_chip_layer_image(spec, timestamp, region)
        renamed = [f"{layer_name}__{b}" for b in spec.bands]
        layer_img = layer_img.rename(renamed)
        stack = layer_img if stack is None else ee.Image(stack).addBands(layer_img)

    if stack is None:
        raise ValueError("Empty chip group layer list")
    return ee.Image(stack).toFloat()


def build_tile_layer_image(layer_name: str):
    ee = ensure_ee_initialized()

    spec = REGISTRY.get(layer_name)
    if spec.source != "gee_tile":
        raise ValueError(f"Not a gee_tile layer: {layer_name!r}")

    if layer_name == "dem":
        # Copernicus DEM uses band "DEM".
        img = ee.ImageCollection(spec.gee_collection).mosaic().select("DEM").rename("elevation")
        return img.toFloat()

    if layer_name == "dist_water":
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        water = gsw.select("max_extent").unmask(0).gt(0).selfMask()
        max_dist = 30_000
        dist = (
            ee.Image(1)
            .cumulativeCost(source=water, maxDistance=max_dist, geodeticDistance=True)
            .unmask(max_dist)
            .rename("dist_water")
        )
        return dist.toFloat()

    raise ValueError(f"No tile image builder for layer: {layer_name!r}")


@dataclass(frozen=True)
class ChipDownloadRequest:
    point_id: int
    lon: float
    lat: float
    timestamp: str
    study_dir: str
    chip_groups: dict[str, dict[str, Any]]
    buffer_km: float
    skip_existing: bool


def _download_chip_worker(args: tuple[int, ChipDownloadRequest]) -> dict[str, Any]:
    idx, req = args
    _ = idx  # unused but keeps signature compatible with imap style

    ensure_ee_initialized()

    study_dir = Path(req.study_dir)
    region = square_region_for_point(req.lon, req.lat, req.buffer_km)

    results: dict[str, Any] = {"point_id": req.point_id, "files": {}}

    for group_name, group in req.chip_groups.items():
        out_path = study_dir / "rasters" / "chips" / group_name / f"{req.point_id}.tif"
        if req.skip_existing and out_path.exists():
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
            continue

        layer_names = list(group["layers"])
        target_scale_m = float(group["target_scale_m"])
        scale_m = compute_group_scale(req.buffer_km, target_scale_m)

        try:
            img = build_chip_group_image(layer_names, req.timestamp, region)
            download_image(img, region, scale_m, out_path, crs="EPSG:4326")
            results["files"][group_name] = {
                "path": str(out_path),
                "status": "downloaded",
                "scale_m": float(scale_m),
            }
        except Exception as e:
            results["files"][group_name] = {"path": None, "status": "failed", "error": str(e)}

    return results


def download_chips_for_plan(
    *,
    df: Any,
    study_dir: Path,
    plan: LayerPlan,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download all chip groups needed by a plan.

    Chips are stored at: study_dir/rasters/chips/<chip_group>/<point_id>.tif
    """
    if plan.chip_groups is None or not plan.chip_groups:
        return {"total_points": 0, "downloaded": 0, "existing": 0, "failed": 0, "failed_points": []}

    if SET_COL in df.columns:
        train_df = df[df[SET_COL] == "train"]
    else:
        train_df = df
    buffer_km = get_buffer_m_for_study(train_df) / 1000.0

    requests: list[ChipDownloadRequest] = []
    seen: set[int] = set()
    for _, row in df.iterrows():
        point_id = int(row[POINT_ID_COL])
        if point_id in seen:
            continue
        seen.add(point_id)

        ts = row[TIME_COL]
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        requests.append(
            ChipDownloadRequest(
                point_id=point_id,
                lon=float(row[LON_COL]),
                lat=float(row[LAT_COL]),
                timestamp=ts_iso,
                study_dir=str(study_dir),
                chip_groups=plan.chip_groups,
                buffer_km=float(buffer_km),
                skip_existing=bool(skip_existing),
            )
        )

    # Ensure directories exist
    for group_name in plan.chip_groups.keys():
        (study_dir / "rasters" / "chips" / group_name).mkdir(parents=True, exist_ok=True)

    if n_workers <= 1:
        results = [_download_chip_worker((i, r)) for i, r in enumerate(requests)]
    else:
        with mp.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(_download_chip_worker, enumerate(requests)))

    stats = {"total_points": len(results), "downloaded": 0, "existing": 0, "failed": 0, "failed_points": []}
    for r in results:
        for group_name, info in r["files"].items():
            if info["status"] == "downloaded":
                stats["downloaded"] += 1
            elif info["status"] == "existing":
                stats["existing"] += 1
            else:
                stats["failed"] += 1
                stats["failed_points"].append({"point_id": r["point_id"], "group": group_name, "error": info.get("error")})

    stats["buffer_km"] = float(buffer_km)
    stats["chip_groups"] = {k: {"layers": list(v["layers"]), "target_scale_m": float(v["target_scale_m"])} for k, v in plan.chip_groups.items()}
    return stats


def _bounds_with_buffer_deg(df: Any, *, buffer_km: float) -> tuple[float, float, float, float]:
    import numpy as np

    min_lon = float(np.min(df[LON_COL].values))
    max_lon = float(np.max(df[LON_COL].values))
    min_lat = float(np.min(df[LAT_COL].values))
    max_lat = float(np.max(df[LAT_COL].values))

    center_lat = (min_lat + max_lat) / 2.0
    lat_buf_deg = (buffer_km * 1000.0) / 110540.0
    lon_buf_deg = (buffer_km * 1000.0) / (111320.0 * math.cos(math.radians(center_lat)))

    return (
        max(-180.0, min_lon - lon_buf_deg),
        max(-90.0, min_lat - lat_buf_deg),
        min(180.0, max_lon + lon_buf_deg),
        min(90.0, max_lat + lat_buf_deg),
    )


def download_tiles(
    *,
    df: Any,
    study_dir: Path,
    tile_layers: list[str],
    buffer_km: float,
    scale_m: float = 30.0,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Download study-wide tile rasters for gee_tile layers."""
    if not tile_layers:
        return {"downloaded": 0, "existing": 0, "failed": 0, "layers": {}}

    ee = ensure_ee_initialized()

    bounds = _bounds_with_buffer_deg(df, buffer_km=buffer_km)
    region = ee.Geometry.Rectangle(list(bounds))

    (study_dir / "rasters" / "tiles").mkdir(parents=True, exist_ok=True)

    out: dict[str, Any] = {"downloaded": 0, "existing": 0, "failed": 0, "layers": {}, "bounds": bounds, "scale_m": float(scale_m)}

    for layer_name in tile_layers:
        out_path = study_dir / "rasters" / "tiles" / f"{layer_name}.tif"
        if skip_existing and out_path.exists():
            out["existing"] += 1
            out["layers"][layer_name] = {"path": str(out_path), "status": "existing"}
            continue

        try:
            img = build_tile_layer_image(layer_name)
            download_tiled_image(img, region, scale_m, out_path, crs="EPSG:4326")
            out["downloaded"] += 1
            out["layers"][layer_name] = {"path": str(out_path), "status": "downloaded"}
        except Exception as e:
            out["failed"] += 1
            out["layers"][layer_name] = {"path": None, "status": "failed", "error": str(e)}

    return out
