"""Earth Engine raster downloads for movebench.

This module writes rasters to the on-disk layout expected by `movebench.layers.io.LayerIO`:

- Chips (per-point): `study_dir/rasters/<rasters_hash>/chips/<chip_group>/<point_id:05d>.tif`
- Tiles (per-study):
  - Static: `study_dir/rasters/<rasters_hash>/tiles/<layer>/tile_XXX.tif` + `index.json`
  - Annual: `study_dir/rasters/<rasters_hash>/tiles/<layer>/tile_XXX_<year>.tif` + `index_<year>.json` (also writes `index.json`)

All masked pixels are materialized as the numeric sentinel `-9999` in GeoTIFFs.
`LayerIO` converts this sentinel to NaN immediately after reading.
"""

from __future__ import annotations

import os
import json
import hashlib
import math
import multiprocessing as mp
import requests
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ee
import rasterio
from rasterio.enums import ColorInterp
from rasterio.merge import merge as rasterio_merge
from tqdm import tqdm

from movebench.layers.constants import (
    GEE_NODATA_SENTINEL,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON_EQUATOR,
    WGS84_CRS,
    WGS84_MAX_LAT,
    WGS84_MAX_LON,
    WGS84_MIN_LAT,
    WGS84_MIN_LON,
)
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import LayerRegistry, REGISTRY
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    compute_step_length_extent_m,
    haversine_distance_m,
    prepare_interval_step_starts,
)


_ee_initialized = False


def ensure_ee_initialized() -> None:
    """Initialize the Earth Engine client once for this process."""
    global _ee_initialized
    if not _ee_initialized:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
        _ee_initialized = True


def compute_group_scale(
    buffer_km: float,
    native_scale_m: float,
    *,
    min_pixels: int,
    max_pixels: int,
) -> float:
    """Compute meters/pixel scale, clamped so chips are not too big/small."""
    if buffer_km <= 0:
        raise ValueError(f"buffer_km must be > 0, got {buffer_km}")
    if native_scale_m <= 0:
        raise ValueError(f"native_scale_m must be > 0, got {native_scale_m}")

    R_m = buffer_km * 1000.0

    # Coarsest allowed scale (fewest pixels).
    scale_for_max = (2.0 * R_m) / float(max_pixels)
    # Finest allowed scale (most pixels).
    scale_for_min = (2.0 * R_m) / float(min_pixels)

    if native_scale_m < scale_for_max:
        return float(scale_for_max)
    if native_scale_m > scale_for_min:
        return float(scale_for_min)
    return float(native_scale_m)


def download_tiff(url: str, out_path: Path, *, tries: int = 5) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delay_s = 2.0
    for attempt in range(tries):
        tmp_path: Path | None = None
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                dir=str(out_path.parent),
                prefix=f"{out_path.name}.",
                suffix=".tmp",
            ) as tmp:
                tmp_path = Path(tmp.name)
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        tmp.write(chunk)
            tmp_path.replace(out_path)
            return
        except requests.RequestException:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt == tries - 1:
                raise
            time.sleep(delay_s)
            delay_s *= 2.0


def _fix_geotiff_colorinterp(path: Path, *, num_bands: int | None = None) -> None:
    """Best-effort fix for GDAL/libtiff ExtraSamples warnings on multi-band GeoTIFFs.

    Some writers (including some GEE GeoTIFF exports) can omit ExtraSamples
    metadata for multi-band rasters. GDAL then emits warnings like:
        TIFFReadDirectory: ... ExtraSamples doesn't match SamplesPerPixel
    and heuristically treats remaining bands as extra samples.

    We explicitly mark band 1 as grayscale and all remaining bands as
    undefined, which prompts GDAL to write consistent ExtraSamples metadata.
    """
    path = Path(path)
    try:
        # Route GDAL warnings to a temp log file (avoids noisy stderr spam).
        with tempfile.NamedTemporaryFile(prefix="gdal_", suffix=".log") as tf:
            with rasterio.Env(CPL_LOG=str(tf.name)):
                with rasterio.open(path, "r+") as ds:
                    count = int(ds.count)
                    n = count if num_bands is None else int(num_bands)
                    if n != count:
                        n = count
                    if n <= 0:
                        return
                    if n == 1:
                        ds.colorinterp = (ColorInterp.gray,)
                    else:
                        ds.colorinterp = (ColorInterp.gray,) + (ColorInterp.undefined,) * (n - 1)
    except Exception:
        return


def square_region_for_point(lon: float, lat: float, buffer_km: float):
    ensure_ee_initialized()
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(float(buffer_km) * 1000.0).bounds()


def union_regions(regions: list):
    ensure_ee_initialized()
    return ee.Geometry.MultiPolygon(regions)


def _wrap_lon_to_wgs84(lon: float) -> float:
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    if wrapped == -180.0 and float(lon) > 0.0:
        return 180.0
    return wrapped


def _clamp_tile_bounds_latlon(bounds_latlon: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Clamp tile bounds to a valid EPSG:4326 lon/lat range.

    Earth Engine can snap coveringGrid cells slightly beyond [-180,180] near the dateline. Since all
    downstream readers expect lon in [-180,180], clamp bounds here to avoid invalid requests.
    """
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bounds_latlon)
    min_lon = max(min_lon, WGS84_MIN_LON)
    max_lon = min(max_lon, WGS84_MAX_LON)
    min_lat = max(min_lat, WGS84_MIN_LAT)
    max_lat = min(max_lat, WGS84_MAX_LAT)
    return (min_lon, min_lat, max_lon, max_lat)


def _bounds_from_ee_feature_info(feature_info: Any) -> tuple[float, float, float, float]:
    """Extract (min_lon, min_lat, max_lon, max_lat) from an EE Feature `getInfo()` dict."""
    if not isinstance(feature_info, dict):
        raise TypeError(f"Expected feature_info dict, got {type(feature_info)!r}")

    geom = feature_info.get("geometry")
    if isinstance(geom, dict) and "coordinates" in geom:
        coords = geom.get("coordinates")
    elif "coordinates" in feature_info:
        # Some callers may pass a geometry dict directly.
        coords = feature_info.get("coordinates")
    else:
        raise KeyError("Feature info missing geometry coordinates")

    if not isinstance(coords, list) or not coords:
        raise ValueError("Invalid geometry coordinates (expected non-empty list)")

    ring = coords[0]
    if not isinstance(ring, list) or not ring:
        raise ValueError("Invalid geometry coordinates (expected polygon ring)")

    min_lon = min(float(c[0]) for c in ring)
    max_lon = max(float(c[0]) for c in ring)
    min_lat = min(float(c[1]) for c in ring)
    max_lat = max(float(c[1]) for c in ring)
    return (min_lon, min_lat, max_lon, max_lat)


def _enumerate_covering_grid_tile_bounds(
    filtered_fc: Any,
    *,
    n_tiles_reported: int,
    page_size: int = 250,
) -> tuple[list[tuple[float, float, float, float]], list[int], list[dict[str, Any]], int]:
    """Enumerate tile bounds from a coveringGrid FeatureCollection.

    This intentionally avoids per-index `toList(1, i)` calls, which have proven unstable for some
    computed FeatureCollections; instead it pages `toList(limit, offset)` and parses client-side.
    """
    n_tiles_reported_i = int(n_tiles_reported)
    page_size_i = max(int(page_size), 1)

    tile_bounds_latlon: list[tuple[float, float, float, float]] = []
    tile_indices: list[int] = []
    geometry_errors: list[dict[str, Any]] = []
    n_tiles_effective = int(n_tiles_reported_i)

    for offset in range(0, n_tiles_reported_i, page_size_i):
        expected = min(int(page_size_i), int(n_tiles_reported_i) - int(offset))
        if expected <= 0:
            break

        try:
            page = filtered_fc.toList(int(expected), int(offset)).getInfo()
        except Exception as e:
            geometry_errors.append({"tile_index": int(offset), "error": f"tile_page_failed: {type(e).__name__}: {e}"})
            n_tiles_effective = int(offset)
            break

        if not isinstance(page, list):
            geometry_errors.append(
                {
                    "tile_index": int(offset),
                    "error": f"tile_page_invalid: expected list, got {type(page).__name__}",
                }
            )
            n_tiles_effective = int(offset)
            break

        if len(page) < int(expected):
            n_tiles_effective = int(offset) + int(len(page))

        if not page:
            break

        for j, feat in enumerate(page):
            i = int(offset) + int(j)
            try:
                bounds = _bounds_from_ee_feature_info(feat)
                bounds = _clamp_tile_bounds_latlon(bounds)
                if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                    raise ValueError(f"degenerate_bounds: {bounds}")
            except Exception as e:
                geometry_errors.append(
                    {
                        "tile_index": int(i),
                        "error": f"tile_geometry_failed: {type(e).__name__}: {e}",
                    }
                )
                continue
            tile_bounds_latlon.append(bounds)
            tile_indices.append(int(i))

        if len(page) < int(expected):
            break

    return tile_bounds_latlon, tile_indices, geometry_errors, int(n_tiles_effective)


def _fully_masked_image(bands: list[str]):
    ensure_ee_initialized()
    base = ee.Image.constant([0] * len(bands)).rename(bands).toFloat()
    return base.updateMask(ee.Image.constant(0))


def _parse_iso_timestamp(ts: str) -> datetime:
    # Accept "Z" suffix for UTC.
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _collection_image_or_masked(collection, bands: list[str], *, reducer: str = "first"):
    if reducer == "first":
        image = ee.Image(collection.first())
    elif reducer == "median":
        image = collection.median()
    elif reducer == "mosaic":
        image = collection.mosaic()
    else:
        raise ValueError(f"Unsupported reducer: {reducer!r}")
    return ee.Image(ee.Algorithms.If(collection.size().gt(0), image, _fully_masked_image(bands)))


def _nearest_from_window(collection_id: str, timestamp: str, *, bands: list[str], start: int, end: int, unit: str):
    ensure_ee_initialized()
    ts = ee.Date(timestamp)
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(ts.advance(start, unit), ts.advance(end, unit))
        .select(bands)
        .sort("system:time_start")
    )
    return _collection_image_or_masked(col, bands, reducer="first")


def _latest_from_past_days(
    collection_id: str,
    timestamp: str,
    *,
    bands: list[str],
    days: int,
    region
):
    ensure_ee_initialized()
    ts = ee.Date(timestamp)
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(ts.advance(-int(days), "day"), ts.advance(1, "second"))
        .filterBounds(region)
        .select(bands)
        .sort("system:time_start", False)
    )
    return _collection_image_or_masked(col, bands, reducer="first")


def _apply_chip_output_conventions(layer: str, spec, img) -> Any:
    if getattr(spec, "scale_factor", 1.0) not in (None, 1.0):
        img = img.multiply(float(spec.scale_factor))
    return img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()


def _nearest_temporal_chip_image(*, spec, timestamp: str):
    temporal_windows: dict[str, tuple[int, int, str]] = {
        "hourly": (-1, 1, "hour"),
        "daily": (-1, 1, "day"),
        "8day": (-4, 4, "day"),
        "16day": (-10, 10, "day"),
    }

    try:
        start, end, unit = temporal_windows[str(spec.temporal)]
    except KeyError as e:
        raise ValueError(f"Unsupported gee_chip temporal={spec.temporal!r}") from e
    return _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=start, end=end, unit=unit)


def _chip_layer_image(layer: str, *, spec, timestamp: str, region) -> Any:
    """Build an ee.Image for one gee_chip layer, with uniquely-named bands."""
    ensure_ee_initialized()

    if layer == "landsat":
        img8 = _latest_from_past_days(
            spec.gee_collection,
            timestamp,
            bands=spec.bands,
            days=8,
            region=region
        )
        img32 = _latest_from_past_days(
            "LANDSAT/COMPOSITES/C02/T1_L2_32DAY",
            timestamp,
            bands=spec.bands,
            days=32,
            region=region
        )
        return _apply_chip_output_conventions(layer, spec, img8.unmask(img32, sameFootprint=False))

    if layer == "dynamic_world":
        ts_dt = _parse_iso_timestamp(timestamp)
        dw_start_dt = datetime(2015, 6, 23, tzinfo=ts_dt.tzinfo)
        if ts_dt < dw_start_dt:
            col = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate("2015-06-23", "2016-06-23")
                .filterBounds(region)
                .select(spec.bands)
            )
        else:
            ts = ee.Date(timestamp)
            col30 = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(ts.advance(-15, "day"), ts.advance(15, "day"))
                .filterBounds(region)
                .select(spec.bands)
            )
            col60 = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(ts.advance(-30, "day"), ts.advance(30, "day"))
                .filterBounds(region)
                .select(spec.bands)
            )
            col = ee.ImageCollection(ee.Algorithms.If(col30.size().gt(0), col30, col60))
        return _apply_chip_output_conventions(layer, spec, _collection_image_or_masked(col, spec.bands, reducer="median"))

    img = _nearest_temporal_chip_image(spec=spec, timestamp=timestamp)

    if layer == "era5_land_hourly":
        era5_hourly_fallback = _nearest_from_window(
            "ECMWF/ERA5/HOURLY",
            timestamp,
            bands=spec.bands,
            start=-1,
            end=1,
            unit="hour",
        )
        img = img.unmask(era5_hourly_fallback, sameFootprint=False)

    if layer == "era5_land_daily":
        src = [
            "mean_2m_air_temperature",
            "dewpoint_2m_temperature",
            "total_precipitation",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ]
        dst = [
            "temperature_2m",
            "dewpoint_temperature_2m",
            "total_precipitation_sum",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
        ]
        era5_daily = _nearest_from_window("ECMWF/ERA5/DAILY", timestamp, bands=src, start=-1, end=1, unit="day").rename(dst)
        fallback = _fully_masked_image(spec.bands).addBands(era5_daily, overwrite=True).select(spec.bands)
        img = img.unmask(fallback, sameFootprint=False)

    return _apply_chip_output_conventions(layer, spec, img)


def build_chip_group_image(
    plan: LayerPlan,
    *,
    registry: LayerRegistry,
    chip_group: str,
    timestamp: str,
    region,
) -> Any:
    """Build an ee.Image stack for a chip group, in plan order."""
    ensure_ee_initialized()

    group = plan.chip_groups.get(chip_group)
    if group is None:
        raise KeyError(f"Unknown chip_group: {chip_group!r}")

    img = None
    for layer in group["layers"]:
        spec = registry.get(layer)
        if spec.source != "gee_chip":
            raise ValueError(f"chip group {chip_group!r} includes non-gee_chip layer {layer!r}")
        layer_img = _chip_layer_image(layer, spec=spec, timestamp=timestamp, region=region)
        img = layer_img if img is None else img.addBands(layer_img)

    if img is None:
        raise ValueError(f"chip group {chip_group!r} has no layers")

    return ee.Image(img).unmask(GEE_NODATA_SENTINEL).toFloat()


def download_tiled_image(
    img,
    region,
    *,
    scale: float,
    out_path: Path,
    num_bands: int,
    crs: str = WGS84_CRS,
    bounds_latlon: tuple[float, float, float, float] | None = None,
) -> None:
    """Download an ee.Image, tiling if needed, and write GeoTIFF to out_path."""
    ensure_ee_initialized()

    if bounds_latlon is None:
        # Get region bounds.
        bounds_info = region.bounds().getInfo()
        coords = bounds_info["coordinates"][0]
        min_lon = min(c[0] for c in coords)
        max_lon = max(c[0] for c in coords)
        min_lat = min(c[1] for c in coords)
        max_lat = max(c[1] for c in coords)
    else:
        min_lon, min_lat, max_lon, max_lat = (float(x) for x in bounds_latlon)

    # Estimate size (rough).
    center_lat = (min_lat + max_lat) / 2.0
    lon_deg_to_m = METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(center_lat))
    lat_deg_to_m = METERS_PER_DEG_LAT

    width_m = (max_lon - min_lon) * lon_deg_to_m
    height_m = (max_lat - min_lat) * lat_deg_to_m
    width_px = width_m / float(scale)
    height_px = height_m / float(scale)

    bytes_per_pixel = 4  # float32
    estimated_size_bytes = width_px * height_px * float(num_bands) * bytes_per_pixel
    gee_limit_bytes = 50_331_648

    # If small enough, download directly.
    if estimated_size_bytes <= gee_limit_bytes * 0.7:
        params = {
            "scale": float(scale),
            "region": region,
            "format": "GEO_TIFF",
            "crs": crs,
        }
        url = img.getDownloadURL(params)
        download_tiff(url, out_path)
        _fix_geotiff_colorinterp(out_path, num_bands=int(num_bands))
        return

    # Otherwise, split into tiles and merge.
    pixels_per_tile = (gee_limit_bytes * 0.35) / (float(num_bands) * bytes_per_pixel)
    tile_side_px = math.sqrt(pixels_per_tile)
    tile_side_m = tile_side_px * float(scale)

    tile_width_deg = tile_side_m / lon_deg_to_m
    tile_height_deg = tile_side_m / lat_deg_to_m

    n_cols = math.ceil((max_lon - min_lon) / tile_width_deg)
    n_rows = math.ceil((max_lat - min_lat) / tile_height_deg)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

                params = {
                    "scale": float(scale),
                    "region": tile_region,
                    "format": "GEO_TIFF",
                    "crs": crs,
                }

                url = img.getDownloadURL(params)
                download_tiff(url, tile_path)
                tile_paths.append(tile_path)
                tile_idx += 1

        if not tile_paths:
            raise RuntimeError("No tiles were downloaded")

        log_path = str(Path(tmp_dir) / "gdal.log")
        with rasterio.Env(CPL_LOG=log_path):
            src_files = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic, mosaic_transform = rasterio_merge(src_files)
                profile = src_files[0].profile.copy()
                profile.update(
                    {
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": mosaic_transform,
                    }
                )
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(mosaic)
            finally:
                for ds in src_files:
                    ds.close()

        _fix_geotiff_colorinterp(out_path, num_bands=int(num_bands))


def _tile_image(layer: str, *, registry: LayerRegistry, year: int | None = None):
    """Build an ee.Image for a gee_tile layer."""
    ensure_ee_initialized()
    spec = registry.get(layer)

    if layer == "dem":
        # COPERNICUS/DEM/GLO30 uses band "DEM"; we export one band.
        return ee.ImageCollection(spec.gee_collection).mosaic().select("DEM").toFloat()

    annual_reducers = {
        "alphaearth_embeddings": "mosaic",
        "modis_lc": "first",
        "modis_cvf_tree": "first",
        "modis_cvf_nontree": "first",
        "modis_cvf_nonveg": "first",
    }
    reducer = annual_reducers.get(layer)
    if reducer is not None:
        if year is None:
            raise ValueError(f"{layer} requires a year for gee_tile export")
        col = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
            .select(spec.bands)
        )
        return _collection_image_or_masked(col, spec.bands, reducer=reducer).toFloat()

    if layer == "dist_water":
        if spec.clip_max is None:
            raise ValueError("dist_water requires LayerSpec.clip_max (meters)")
        max_dist = float(spec.clip_max)
        gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        water = gsw.select("max_extent").unmask(0).gt(0).selfMask()
        dist = (
            ee.Image(1)
            .cumulativeCost(source=water, maxDistance=max_dist, geodeticDistance=True)
            .clamp(0, max_dist)
            .unmask(max_dist)
            .rename("dist_water")
        )
        return dist.toFloat()

    raise ValueError(f"Unsupported gee_tile layer {layer!r}")


def _rasters_root(study_dir: Path, rasters_hash: str) -> Path:
    return Path(study_dir) / "rasters" / str(rasters_hash)


def _chip_path(study_dir: Path, rasters_hash: str, chip_group: str, point_id: int) -> Path:
    return _rasters_root(study_dir, rasters_hash) / "chips" / chip_group / f"{int(point_id):05d}.tif"


def _tile_dir(study_dir: Path, rasters_hash: str, layer: str) -> Path:
    return _rasters_root(study_dir, rasters_hash) / "tiles" / layer


def _expected_band_count_for_chip_group(plan: LayerPlan, chip_group: str, registry: LayerRegistry) -> int:
    g = plan.chip_groups.get(chip_group)
    if g is None:
        raise KeyError(f"Unknown chip_group: {chip_group!r}")
    return sum(len(registry.get(layer).bands) for layer in g["layers"])

def _stable_hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def _write_json_atomic(path: Path, payload: dict[str, Any], *, sort_keys: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=sort_keys)
        Path(tmp.name).replace(path)


def _chips_meta_path(study_dir: Path, rasters_hash: str) -> Path:
    return _rasters_root(study_dir, rasters_hash) / "chips_meta.json"


def _clear_chip_group_dir(group_dir: Path) -> None:
    group_dir = Path(group_dir)
    group_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.tif", "*.tif.*.tmp"):
        for p in group_dir.glob(pattern):
            p.unlink()


def _resolve_buffer_km_args(
    *,
    buffer_km: float | None,
    resolution_buffer_km: float | None,
    extent_buffer_km: float | None,
    caller_name: str,
) -> tuple[float, float]:
    if resolution_buffer_km is None:
        if buffer_km is None:
            raise ValueError(f"{caller_name} requires resolution_buffer_km or legacy buffer_km")
        resolution_buffer_km = float(buffer_km)
    if extent_buffer_km is None:
        if buffer_km is None:
            raise ValueError(f"{caller_name} requires extent_buffer_km or legacy buffer_km")
        extent_buffer_km = float(buffer_km)
    return float(resolution_buffer_km), float(extent_buffer_km)


def _require_columns(df: pd.DataFrame, required: list[str], *, label: str = "trajectories df") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _chip_request_hash(point_ids: list[int]) -> str:
    payload = {"point_ids": sorted({int(pid) for pid in point_ids})}
    return _stable_hash_dict(payload)


def _normalize_chip_group_completions(meta: dict[str, Any], *, meta_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    raw = meta.get("chip_group_completions", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid chips_meta.json chip_group_completions (expected mapping): {meta_path}")

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for group_name, group_entries in raw.items():
        if not isinstance(group_name, str) or not isinstance(group_entries, dict):
            raise ValueError(f"Invalid chip_group_completions entry for {group_name!r} in {meta_path}")
        normalized_group = {str(request_hash): dict(record) for request_hash, record in group_entries.items() if isinstance(record, dict)}
        if len(normalized_group) != len(group_entries):
            raise ValueError(f"Invalid chip_group_completions[{group_name!r}] records in {meta_path}")
        if normalized_group:
            out[group_name] = normalized_group
    return out


def _tile_filename(tile_index: int, *, year: int | None = None) -> str:
    if year is None:
        return f"tile_{int(tile_index):03d}.tif"
    return f"tile_{int(tile_index):03d}_{int(year)}.tif"


def _tile_index_is_complete(tile_dir: Path, idx_path: Path) -> tuple[bool, int]:
    if not idx_path.exists():
        return (False, 0)
    with open(idx_path, "r") as f:
        existing = json.load(f)
    tiles = existing.get("tiles", [])
    errors = existing.get("errors", [])
    if not tiles or errors or any(("error" in t) for t in tiles):
        return (False, len(tiles))
    if not all((tile_dir / t["file"]).exists() for t in tiles):
        return (False, len(tiles))
    return (True, len(tiles))


def _build_tile_meta(
    tile_index: int,
    bounds_latlon: tuple[float, float, float, float],
    *,
    year: int | None = None,
) -> dict[str, Any]:
    return {
        "file": _tile_filename(tile_index, year=year),
        "tile_index": int(tile_index),
        "bounds": {
            "min_lon": float(bounds_latlon[0]),
            "min_lat": float(bounds_latlon[1]),
            "max_lon": float(bounds_latlon[2]),
            "max_lat": float(bounds_latlon[3]),
        },
    }


def _build_geometry_errors(
    geometry_errors: list[dict[str, Any]],
    *,
    year: int | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "tile_index": int(e["tile_index"]),
            "file": _tile_filename(int(e["tile_index"]), year=year),
            "error": str(e["error"]),
        }
        for e in geometry_errors
    ]


def _empty_tile_stats(*, total: int, failed: int = 0, index_path: Path | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {"total": int(total), "downloaded": 0, "existing": 0, "failed": int(failed)}
    if index_path is not None:
        stats["index_path"] = str(index_path)
    return stats


def _record_tile_result(
    stats: dict[str, Any],
    tile_meta: dict[str, Any],
    info: dict[str, Any] | None,
) -> None:
    if info is None:
        stats["failed"] += 1
        tile_meta["error"] = "tile_missing_result"
        return

    status = str(info.get("status", "failed"))
    if status == "downloaded":
        stats["downloaded"] += 1
    elif status == "existing":
        stats["existing"] += 1
    else:
        stats["failed"] += 1
        tile_meta["error"] = str(info.get("error", "unknown"))


def _build_tile_index_payload(
    *,
    layer: str,
    layer_scale: float,
    num_bands: int,
    tiles_out: list[dict[str, Any]],
    errors_out: list[dict[str, Any]],
    n_tiles_reported: int,
    n_tiles_effective: int,
    temporal: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "layer": layer,
        "scale": float(layer_scale),
        "num_bands": int(num_bands),
        "tiles": tiles_out,
    }
    if temporal is not None:
        payload["temporal"] = temporal
    if year is not None:
        payload["year"] = int(year)
    if errors_out:
        payload["errors"] = errors_out
        payload["n_tiles_reported"] = int(n_tiles_reported)
    if int(n_tiles_effective) != int(n_tiles_reported):
        payload["n_tiles_reported"] = int(n_tiles_reported)
        payload["n_tiles_effective"] = int(n_tiles_effective)
    return payload


def _expected_chip_group_meta(plan: LayerPlan, chip_group: str, registry: LayerRegistry) -> dict[str, Any]:
    g = plan.chip_groups.get(chip_group)
    if g is None:
        raise KeyError(f"Unknown chip_group: {chip_group!r}")
    layers = list(g["layers"])
    band_ids_in_order: list[str] = []
    for layer in layers:
        spec = registry.get(layer)
        band_ids_in_order.extend([f"{layer}.{b}" for b in spec.bands])
    payload = {
        "target_scale_m": float(g["target_scale_m"]),
        "layers": layers,
        "band_ids_in_order": band_ids_in_order,
    }
    return {
        "signature": _stable_hash_dict(payload),
        **payload,
    }


@dataclass(frozen=True)
class ChipRequest:
    point_id: int
    lon: float
    lat: float
    timestamp: str


@dataclass(frozen=True)
class TileRequest:
    tile_index: int
    bounds_latlon: tuple[float, float, float, float]


def _download_tile_worker(args: tuple[TileRequest, dict[str, Any]]) -> dict[str, Any]:
    req, ctx = args

    study_dir = Path(ctx["study_dir"])
    rasters_hash = str(ctx["rasters_hash"])
    layer = str(ctx["layer"])
    year = ctx.get("year")
    scale = float(ctx["scale"])
    num_bands = int(ctx["num_bands"])
    crs = str(ctx["crs"])
    skip_existing = bool(ctx["skip_existing"])
    registry: LayerRegistry = ctx["registry"]

    year_int = int(year) if year is not None else None
    tile_dir = _tile_dir(study_dir, rasters_hash, layer)
    tile_index = int(req.tile_index)
    out_path = tile_dir / _tile_filename(tile_index, year=year_int)

    if skip_existing and out_path.exists():
        return {"tile_index": tile_index, "year": year_int, "status": "existing", "file": out_path.name}

    try:
        ensure_ee_initialized()
        img = _tile_image(layer, registry=registry, year=year_int).unmask(
            GEE_NODATA_SENTINEL, sameFootprint=False
        )
        bounds_latlon = _clamp_tile_bounds_latlon(req.bounds_latlon)
        min_lon, min_lat, max_lon, max_lat = bounds_latlon
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(f"Invalid tile bounds: {bounds_latlon}")
        region = ee.Geometry.Rectangle(
            [min_lon, min_lat, max_lon, max_lat], proj=crs, geodesic=False
        )
        download_tiled_image(
            img,
            region,
            scale=scale,
            out_path=out_path,
            num_bands=num_bands,
            crs=crs,
            bounds_latlon=bounds_latlon,
        )
        return {"tile_index": tile_index, "year": year_int, "status": "downloaded", "file": out_path.name}
    except Exception as e:
        return {
            "tile_index": tile_index,
            "year": year_int,
            "status": "failed",
            "file": out_path.name,
            "error": str(e),
        }


def _run_tile_workers(
    reqs: list[TileRequest],
    ctx: dict[str, Any],
    *,
    desc: str,
    n_workers: int,
    pool: mp.Pool | None = None,
) -> tuple[dict[int, dict[str, Any]], mp.Pool | None]:
    if not reqs:
        return {}, pool

    worker_args = [(req, ctx) for req in reqs]
    if pool is not None:
        results = list(
            tqdm(
                pool.imap_unordered(_download_tile_worker, worker_args),
                total=len(reqs),
                desc=desc,
                unit="tile",
                dynamic_ncols=True,
            )
        )
        return {int(r["tile_index"]): r for r in results}, pool

    with mp.Pool(int(n_workers)) as local_pool:
        results = list(
            tqdm(
                local_pool.imap_unordered(_download_tile_worker, worker_args),
                total=len(reqs),
                desc=desc,
                unit="tile",
                dynamic_ncols=True,
            )
        )
    return {int(r["tile_index"]): r for r in results}, pool


def _build_tile_index_result(
    *,
    tile_indices: list[int],
    tile_bounds_latlon: list[tuple[float, float, float, float]],
    results_by_index: dict[int, dict[str, Any]],
    geometry_errors: list[dict[str, Any]],
    layer: str,
    layer_scale: float,
    num_bands: int,
    n_tiles_reported: int,
    n_tiles_effective: int,
    index_path: Path,
    year: int | None = None,
) -> dict[str, Any]:
    errors_out = _build_geometry_errors(geometry_errors, year=year)
    stats = _empty_tile_stats(total=n_tiles_reported, failed=len(errors_out), index_path=index_path)
    tiles_out: list[dict[str, Any]] = []

    for tile_index, bounds_latlon in zip(tile_indices, tile_bounds_latlon):
        tile_meta = _build_tile_meta(int(tile_index), bounds_latlon, year=year)
        _record_tile_result(stats, tile_meta, results_by_index.get(int(tile_index)))
        tiles_out.append(tile_meta)

    stats["total"] = int(n_tiles_effective)
    payload = _build_tile_index_payload(
        layer=layer,
        layer_scale=float(layer_scale),
        num_bands=num_bands,
        temporal="annual" if year is not None else None,
        year=year,
        tiles_out=tiles_out,
        errors_out=errors_out,
        n_tiles_reported=n_tiles_reported,
        n_tiles_effective=n_tiles_effective,
    )
    _write_json_atomic(index_path, payload, sort_keys=False)
    return stats


def _write_compat_index_if_missing(index_path: Path, source_index_path: Path) -> None:
    if index_path.exists() or (not source_index_path.exists()):
        return
    with open(source_index_path, "r") as f:
        payload = json.load(f)
    with open(index_path, "w") as f:
        json.dump(payload, f, indent=2)


def _download_chip_worker(args: tuple[ChipRequest, dict[str, Any]]) -> dict[str, Any]:
    req, ctx = args

    study_dir = Path(ctx["study_dir"])
    resolution_buffer_km = float(ctx["resolution_buffer_km"])
    extent_buffer_km = float(ctx["extent_buffer_km"])
    skip_existing = bool(ctx["skip_existing"])
    plan: LayerPlan = ctx["plan"]
    registry: LayerRegistry = ctx["registry"]
    min_pixels = int(ctx["min_pixels"])
    max_pixels = int(ctx["max_pixels"])
    group_current: dict[str, bool] = ctx["group_current"]

    results: dict[str, Any] = {"point_id": int(req.point_id), "files": {}}
    all_exist = bool(skip_existing) and all(
        bool(group_current.get(group_name, False))
        and _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id).exists()
        for group_name in plan.chip_groups.keys()
    )

    if all_exist:
        results["files"] = {
            group_name: {
                "path": str(_chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)),
                "status": "existing",
            }
            for group_name in plan.chip_groups.keys()
        }
        return results

    ensure_ee_initialized()

    region = square_region_for_point(req.lon, req.lat, extent_buffer_km)

    for group_name, group in plan.chip_groups.items():
        out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
        expected = _expected_band_count_for_chip_group(plan, group_name, registry)
        if skip_existing and bool(group_current.get(group_name, False)) and out_path.exists():
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
            continue

        scale = compute_group_scale(
            resolution_buffer_km,
            float(group["target_scale_m"]),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        try:
            img = build_chip_group_image(
                plan,
                registry=registry,
                chip_group=group_name,
                timestamp=req.timestamp,
                region=region,
            )
            download_tiled_image(
                img,
                region,
                scale=scale,
                out_path=out_path,
                num_bands=expected,
                crs=WGS84_CRS,
            )
            results["files"][group_name] = {"path": str(out_path), "status": "downloaded"}
        except Exception as e:
            results["files"][group_name] = {"path": None, "status": "failed", "error": str(e)}

    return results


def download_chips(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry = REGISTRY,
    buffer_km: float | None = None,
    resolution_buffer_km: float | None = None,
    extent_buffer_km: float | None = None,
    min_pixels: int,
    max_pixels: int,
    splits: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    study_dir = Path(study_dir)
    resolution_buffer_km, extent_buffer_km = _resolve_buffer_km_args(
        buffer_km=buffer_km,
        resolution_buffer_km=resolution_buffer_km,
        extent_buffer_km=extent_buffer_km,
        caller_name="download_chips",
    )
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    if min_pixels <= 0:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")
    if max_pixels < min_pixels:
        raise ValueError(f"max_pixels must be >= min_pixels ({min_pixels}), got {max_pixels}")

    if splits is None:
        splits = sorted(df[SET_COL].dropna().unique().tolist())

    _require_columns(df, [LON_COL, LAT_COL, TIME_COL, SET_COL, POINT_ID_COL])

    sub = df[df[SET_COL].isin(splits)].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows matched splits={splits}")

    meta_path = _chips_meta_path(study_dir, plan.rasters_hash)
    meta: dict[str, Any]
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid chips_meta.json (expected mapping): {meta_path}")
        if str(meta.get("schema_version", "")) != "1":
            raise ValueError(
                f"Unsupported chips_meta.json schema_version={meta.get('schema_version')!r} (expected '1'): {meta_path}"
            )
    else:
        # Without metadata we cannot trust existing chip files, so rebuild them
        # instead of aborting the whole study download.
        meta = {"schema_version": "1", "chip_groups": {}}

    existing_groups = meta.get("chip_groups", {})
    if not isinstance(existing_groups, dict):
        raise ValueError(f"Invalid chips_meta.json chip_groups (expected mapping): {meta_path}")
    chip_group_completions = _normalize_chip_group_completions(meta, meta_path=meta_path)
    expected_meta_by_group = {
        group_name: _expected_chip_group_meta(plan, group_name, registry) for group_name in plan.chip_groups.keys()
    }
    group_current = {
        group_name: bool(
            isinstance(existing_groups.get(group_name), dict)
            and existing_groups.get(group_name, {}).get("signature") == expected["signature"]
        )
        for group_name, expected in expected_meta_by_group.items()
    }

    for group_name, is_current in group_current.items():
        if is_current:
            continue
        group_dir = _rasters_root(study_dir, plan.rasters_hash) / "chips" / group_name
        _clear_chip_group_dir(group_dir)
        chip_group_completions.pop(group_name, None)

    reqs = [
        ChipRequest(
            point_id=int(row[POINT_ID_COL]),
            lon=float(row[LON_COL]),
            lat=float(row[LAT_COL]),
            timestamp=row[TIME_COL].isoformat() if hasattr(row[TIME_COL], "isoformat") else str(row[TIME_COL]),
        )
        for _, row in sub.iterrows()
    ]

    requested_point_ids = [int(req.point_id) for req in reqs]
    request_hash = _chip_request_hash(requested_point_ids)
    requested_splits = sorted({str(x) for x in sub[SET_COL].dropna().astype(str).tolist()})

    if skip_existing:
        can_skip_all = all(
            bool(group_current.get(group_name, False))
            and request_hash in chip_group_completions.get(group_name, {})
            for group_name in plan.chip_groups.keys()
        )
        if can_skip_all:
            return {
                "total_points": len(reqs),
                "downloaded": 0,
                "existing": len(reqs) * len(plan.chip_groups),
                "failed": 0,
                "failures": [],
            }

    # Ensure output dirs exist up front (avoids races on first write).
    for group_name in plan.chip_groups.keys():
        (_rasters_root(study_dir, plan.rasters_hash) / "chips" / group_name).mkdir(parents=True, exist_ok=True)

    ctx = {
        "study_dir": str(study_dir),
        "resolution_buffer_km": float(resolution_buffer_km),
        "extent_buffer_km": float(extent_buffer_km),
        "skip_existing": bool(skip_existing),
        "plan": plan,
        "registry": registry,
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "group_current": group_current,
    }

    with mp.Pool(int(n_workers)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_download_chip_worker, [(r, ctx) for r in reqs]),
                total=len(reqs),
                desc=f"Downloading chips ({os.path.basename(study_dir)})",
                unit="point",
                dynamic_ncols=True
            )
        )

    stats = {"total_points": len(reqs), "downloaded": 0, "existing": 0, "failed": 0, "failures": []}
    group_success_counts = {group_name: 0 for group_name in plan.chip_groups.keys()}
    group_failed = {group_name: False for group_name in plan.chip_groups.keys()}
    for r in results:
        files = r.get("files", {})
        for group_name in plan.chip_groups.keys():
            info = files.get(group_name)
            status = str(info.get("status", "failed")) if isinstance(info, dict) else "failed"
            if status == "downloaded":
                stats["downloaded"] += 1
                group_success_counts[group_name] += 1
            elif status == "existing":
                stats["existing"] += 1
                group_success_counts[group_name] += 1
            else:
                stats["failed"] += 1
                group_failed[group_name] = True
                stats["failures"].append(
                    {
                        "point_id": r.get("point_id"),
                        "group": group_name,
                        "error": info.get("error", "unknown") if isinstance(info, dict) else "missing_result",
                    }
                )

    meta["chip_groups"] = dict(existing_groups)
    for group_name, expected in expected_meta_by_group.items():
        meta["chip_groups"][group_name] = expected

    completed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for group_name in plan.chip_groups.keys():
        group_records = chip_group_completions.get(group_name)
        is_complete = (group_success_counts.get(group_name, 0) == len(reqs)) and not group_failed.get(group_name, False)
        if is_complete:
            if group_records is None:
                group_records = {}
                chip_group_completions[group_name] = group_records
            group_records[request_hash] = {
                "point_count": int(len(reqs)),
                "splits": list(requested_splits),
                "completed_at": completed_at,
            }
        elif group_records is not None:
            group_records.pop(request_hash, None)
            if not group_records:
                chip_group_completions.pop(group_name, None)

    if chip_group_completions:
        meta["chip_group_completions"] = chip_group_completions
    else:
        meta.pop("chip_group_completions", None)
    _write_json_atomic(meta_path, meta)

    return stats


def get_buffer_m_for_study(
    df: pd.DataFrame,
    *,
    step_time_s: int | None = None,
    step_q: float = 0.95,
    max_step_q: float = 0.99,
    step_multiplier: float = 2.0,
    gap_q: float = 0.999,
) -> float:
    """Compute buffer size (meters) from movement data (train split only)."""
    if TIME_COL not in df.columns:
        raise RuntimeError(f"Expected a '{TIME_COL}' column in the data.")

    if step_time_s is not None:
        d = prepare_interval_step_starts(df, step_time_s=int(step_time_s), gap_q=float(gap_q))
        valid = d.loc[d["valid_step_start"], "step_length_m"].to_numpy(dtype=np.float64, copy=False)
        if len(valid) == 0:
            raise ValueError("No valid interval step lengths computed")
        return compute_step_length_extent_m(
            valid,
            step_q=step_q,
            max_step_q=max_step_q,
            step_multiplier=step_multiplier,
        )

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL])
    d = d.sort_values([INDIVIDUAL_COL, TIME_COL])

    d["time_since_last"] = d.groupby(INDIVIDUAL_COL)[TIME_COL].diff()
    per_indiv_cutoff = d.groupby(INDIVIDUAL_COL)["time_since_last"].transform(lambda s: s.dropna().quantile(gap_q))
    d = d[d["time_since_last"].notna() & (d["time_since_last"] <= per_indiv_cutoff)].copy()

    g = d.groupby(INDIVIDUAL_COL)
    lat_prev_deg = g[LAT_COL].shift()
    lon_prev_deg = g[LON_COL].shift()
    d["step_length_m"] = haversine_distance_m(
        lat_prev_deg.to_numpy(dtype=np.float64, copy=False),
        lon_prev_deg.to_numpy(dtype=np.float64, copy=False),
        d[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        d[LON_COL].to_numpy(dtype=np.float64, copy=False),
    )

    valid = d["step_length_m"].dropna()
    valid = valid[valid > 0]
    if len(valid) == 0:
        raise ValueError("No valid step lengths computed")

    return compute_step_length_extent_m(
        valid,
        step_q=step_q,
        max_step_q=max_step_q,
        step_multiplier=step_multiplier,
    )


def download_tile_layer(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry,
    layer: str,
    buffer_km: float | None = None,
    resolution_buffer_km: float | None = None,
    extent_buffer_km: float | None = None,
    min_pixels: int,
    max_pixels: int,
    px_per_side: int,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if layer not in plan.layers_in_order:
        raise ValueError(f"Layer {layer!r} not present in plan.layers_in_order")

    resolution_buffer_km, extent_buffer_km = _resolve_buffer_km_args(
        buffer_km=buffer_km,
        resolution_buffer_km=resolution_buffer_km,
        extent_buffer_km=extent_buffer_km,
        caller_name="download_tile_layer",
    )
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    px_per_side = int(px_per_side)
    n_workers = int(n_workers)
    if n_workers <= 0:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    spec = registry.get(layer)
    if spec.source != "gee_tile":
        raise ValueError(f"download_tile_layer only supports gee_tile layers, got {layer!r} source={spec.source!r}")
    if spec.native_scale_m is None:
        raise ValueError(f"gee_tile layer {layer!r} missing spec.native_scale_m")

    tile_dir = _tile_dir(study_dir, plan.rasters_hash, layer)
    index_path = tile_dir / "index.json"

    is_annual = str(spec.temporal) == "annual"

    required_cols = [LON_COL, LAT_COL, SET_COL]
    if is_annual:
        required_cols.append(TIME_COL)
    _require_columns(df, required_cols)

    resolved_years: list[int] = []
    requested_years: list[int] = []

    if (not is_annual) and skip_existing and index_path.exists():
        ok, n = _tile_index_is_complete(tile_dir, index_path)
        if ok:
            stats = _empty_tile_stats(total=n, index_path=index_path)
            stats["existing"] = int(n)
            return stats

    if is_annual:
        ts = pd.to_datetime(df[TIME_COL], errors="coerce")
        yrs = ts.dropna().dt.year.astype("int64")
        if len(yrs) == 0:
            raise ValueError(f"trajectories df has no valid {TIME_COL!r} values for annual layer {layer!r}")
        requested_years = sorted(set(int(y) for y in yrs.tolist()))

        ensure_ee_initialized()

        try:
            ts_list = (
                ee.ImageCollection(spec.gee_collection).aggregate_array("system:time_start").getInfo()
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to query available years for annual gee_tile layer {layer!r} from {spec.gee_collection!r}: {e}"
            ) from e

        available_years = sorted(
            {
                datetime.utcfromtimestamp(int(x) / 1000.0).year
                for x in (ts_list or [])
                if x is not None
            }
        )
        if not available_years:
            raise RuntimeError(
                f"No available years found for annual gee_tile layer {layer!r} collection {spec.gee_collection!r}"
            )

        def _closest_year(y: int) -> int:
            return min(available_years, key=lambda a: (abs(int(a) - int(y)), int(a)))

        resolved_years = sorted({_closest_year(y) for y in requested_years})

        if skip_existing and resolved_years:
            years_out: dict[int, dict[str, Any]] = {}
            all_complete = True
            for y in resolved_years:
                idx_path_y = tile_dir / f"index_{int(y)}.json"
                ok, n = _tile_index_is_complete(tile_dir, idx_path_y)
                if not ok:
                    all_complete = False
                    break
                years_out[int(y)] = _empty_tile_stats(total=n, index_path=idx_path_y)
                years_out[int(y)]["existing"] = int(n)

            if all_complete:
                if not index_path.exists():
                    src = tile_dir / f"index_{int(resolved_years[0])}.json"
                    with open(src, "r") as f:
                        payload = json.load(f)
                    with open(index_path, "w") as f:
                        json.dump(payload, f, indent=2)
                return {
                    "temporal": "annual",
                    "years": years_out,
                    "index_path": str(index_path),
                }
    else:
        ensure_ee_initialized()

    layer_scale = compute_group_scale(
        resolution_buffer_km,
        float(spec.native_scale_m),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    crs = WGS84_CRS
    grid_size_m = float(layer_scale) * float(px_per_side)

    # Avoid building a huge Earth Engine union geometry from per-point buffers:
    # bucket points into coarse spatial bins and build one conservative rectangle per bucket.
    lon = df[LON_COL].astype(float)
    lat = df[LAT_COL].astype(float)
    if lon.isna().any() or lat.isna().any():
        raise ValueError("trajectories df contains NaN lon/lat values")

    ref_lat = float(lat.median())
    lon_deg_to_m_ref = METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(ref_lat))
    bucket_deg_lon = 360.0 if abs(lon_deg_to_m_ref) < 1e-6 else grid_size_m / lon_deg_to_m_ref
    bucket_deg_lat = grid_size_m / METERS_PER_DEG_LAT

    bucket_deg_lon = max(float(bucket_deg_lon), 1e-6)
    bucket_deg_lat = max(float(bucket_deg_lat), 1e-6)

    MAX_BUCKETS = 5000
    MAX_COARSEN_ITERS = 4

    base = pd.DataFrame(
        {
            "split": df[SET_COL].astype(str),
            "lon": lon,
            "lat": lat,
            "lat_abs": lat.abs(),
        }
    )

    grouped = None
    b_lon = bucket_deg_lon
    b_lat = bucket_deg_lat
    for _ in range(MAX_COARSEN_ITERS + 1):
        lon_bin = np.floor((base["lon"] + WGS84_MAX_LON) / b_lon).astype("int64")
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
        if len(g) <= MAX_BUCKETS:
            grouped = g
            break
        b_lon *= 2.0
        b_lat *= 2.0

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

    rect_geoms = []
    rect_bounds: list[tuple[float, float, float, float]] = []

    for _, r in grouped.iterrows():
        buffer_m = float(extent_buffer_km) * 1000.0

        buffer_deg_lat = buffer_m / METERS_PER_DEG_LAT
        lon_deg_to_m = METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(float(r["max_abs_lat"])))
        buffer_deg_lon = WGS84_MAX_LON if abs(lon_deg_to_m) < 1e-6 else buffer_m / lon_deg_to_m

        min_lon_exp = float(r["min_lon"]) - buffer_deg_lon
        max_lon_exp = float(r["max_lon"]) + buffer_deg_lon
        min_lat_exp = float(r["min_lat"]) - buffer_deg_lat
        max_lat_exp = float(r["max_lat"]) + buffer_deg_lat

        min_lat_exp = max(min_lat_exp, WGS84_MIN_LAT)
        max_lat_exp = min(max_lat_exp, WGS84_MAX_LAT)

        rects: list[tuple[float, float, float, float]]
        if (max_lon_exp - min_lon_exp) >= 360.0:
            rects = [(WGS84_MIN_LON, min_lat_exp, WGS84_MAX_LON, max_lat_exp)]
        elif max_lon_exp > WGS84_MAX_LON:
            rects = [
                (min_lon_exp, min_lat_exp, WGS84_MAX_LON, max_lat_exp),
                (WGS84_MIN_LON, min_lat_exp, max_lon_exp - 360.0, max_lat_exp),
            ]
        elif min_lon_exp < WGS84_MIN_LON:
            rects = [
                (min_lon_exp + 360.0, min_lat_exp, WGS84_MAX_LON, max_lat_exp),
                (WGS84_MIN_LON, min_lat_exp, max_lon_exp, max_lat_exp),
            ]
        else:
            rects = [(min_lon_exp, min_lat_exp, max_lon_exp, max_lat_exp)]

        for a, b, c, d in rects:
            a = max(float(a), WGS84_MIN_LON)
            c = min(float(c), WGS84_MAX_LON)
            b = float(b)
            d = float(d)
            if a >= c or b >= d:
                continue
            rect_bounds.append((a, b, c, d))
            rect_geoms.append(ee.Geometry.Rectangle([a, b, c, d], proj=crs, geodesic=False))

    if not rect_geoms:
        raise RuntimeError("No bucket rectangles were constructed for tiling")

    min_rect_lon = min(b[0] for b in rect_bounds)
    min_rect_lat = min(b[1] for b in rect_bounds)
    max_rect_lon = max(b[2] for b in rect_bounds)
    max_rect_lat = max(b[3] for b in rect_bounds)

    bounds_rect = ee.Geometry.Rectangle([min_rect_lon, min_rect_lat, max_rect_lon, max_rect_lat], proj=crs, geodesic=False)
    union_region = union_regions(rect_geoms)

    grid = bounds_rect.coveringGrid(proj=crs, scale=grid_size_m)
    filtered = grid.filter(ee.Filter.intersects(".geo", union_region)).sort("system:index")

    n_tiles_reported = int(filtered.size().getInfo())

    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_bounds_latlon, tile_indices, geometry_errors, n_tiles_effective = _enumerate_covering_grid_tile_bounds(
        filtered,
        n_tiles_reported=int(n_tiles_reported),
        page_size=250,
    )

    if is_annual:
        years_out: dict[int, dict[str, Any]] = {}
        reqs = [
            TileRequest(tile_index=int(tile_index), bounds_latlon=bounds_latlon)
            for tile_index, bounds_latlon in zip(tile_indices, tile_bounds_latlon)
        ]
        ctx_base = {
            "study_dir": str(study_dir),
            "rasters_hash": str(plan.rasters_hash),
            "layer": str(layer),
            "scale": float(layer_scale),
            "num_bands": int(len(spec.bands)),
            "crs": str(crs),
            "skip_existing": bool(skip_existing),
            "registry": registry,
        }

        pending_years: list[int] = []
        worker_args: list[tuple[TileRequest, dict[str, Any]]] = []
        for year in resolved_years:
            idx_path_y = tile_dir / f"index_{int(year)}.json"
            if skip_existing and idx_path_y.exists():
                ok, n_existing = _tile_index_is_complete(tile_dir, idx_path_y)
                if ok:
                    years_out[int(year)] = _empty_tile_stats(total=n_existing, index_path=idx_path_y)
                    years_out[int(year)]["existing"] = int(n_existing)
                    continue

            pending_years.append(int(year))
            ctx_y = {**ctx_base, "year": int(year)}
            worker_args.extend((req, ctx_y) for req in reqs)

        results_by_year: dict[int, dict[int, dict[str, Any]]] = {}
        if worker_args:
            with mp.Pool(int(n_workers)) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(_download_tile_worker, worker_args),
                        total=len(worker_args),
                        desc=f"Downloading tiles ({layer}) ({os.path.basename(study_dir)})",
                        unit="tile",
                        dynamic_ncols=True,
                    )
                )
            for result in results:
                year = result.get("year")
                if year is None:
                    continue
                year_results = results_by_year.setdefault(int(year), {})
                year_results[int(result["tile_index"])] = result

        for year in pending_years:
            idx_path_y = tile_dir / f"index_{int(year)}.json"
            years_out[int(year)] = _build_tile_index_result(
                tile_indices=tile_indices,
                tile_bounds_latlon=tile_bounds_latlon,
                results_by_index=results_by_year.get(int(year), {}),
                geometry_errors=geometry_errors,
                layer=layer,
                layer_scale=float(layer_scale),
                num_bands=len(spec.bands),
                n_tiles_reported=n_tiles_reported,
                n_tiles_effective=n_tiles_effective,
                index_path=idx_path_y,
                year=int(year),
            )

        # Compatibility index.json used for bounds-only lookups.
        if resolved_years:
            _write_compat_index_if_missing(index_path, tile_dir / f"index_{int(resolved_years[0])}.json")

        return {
            "temporal": "annual",
            "years": years_out,
            "index_path": str(index_path),
        }

    reqs = [
        TileRequest(tile_index=int(tile_index), bounds_latlon=bounds_latlon)
        for tile_index, bounds_latlon in zip(tile_indices, tile_bounds_latlon)
    ]
    ctx = {
        "study_dir": str(study_dir),
        "rasters_hash": str(plan.rasters_hash),
        "layer": str(layer),
        "year": None,
        "scale": float(layer_scale),
        "num_bands": int(len(spec.bands)),
        "crs": str(crs),
        "skip_existing": bool(skip_existing),
        "registry": registry,
    }

    results_by_index, _ = _run_tile_workers(
        reqs,
        ctx,
        desc=f"Downloading tiles ({layer}) ({os.path.basename(study_dir)})",
        n_workers=n_workers,
    )
    return _build_tile_index_result(
        tile_indices=tile_indices,
        tile_bounds_latlon=tile_bounds_latlon,
        results_by_index=results_by_index,
        geometry_errors=geometry_errors,
        layer=layer,
        layer_scale=float(layer_scale),
        num_bands=len(spec.bands),
        n_tiles_reported=n_tiles_reported,
        n_tiles_effective=n_tiles_effective,
        index_path=index_path,
    )


def download_tiles(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry = REGISTRY,
    buffer_km: float | None = None,
    resolution_buffer_km: float | None = None,
    extent_buffer_km: float | None = None,
    min_pixels: int,
    max_pixels: int,
    px_per_side: int,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    tile_layers = [l for l in plan.layers_in_order if registry.get(l).source == "gee_tile"]
    out: dict[str, Any] = {}
    for layer in tile_layers:
        out[layer] = download_tile_layer(
            df,
            study_dir=study_dir,
            plan=plan,
            registry=registry,
            layer=layer,
            buffer_km=buffer_km,
            resolution_buffer_km=resolution_buffer_km,
            extent_buffer_km=extent_buffer_km,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            px_per_side=px_per_side,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )
    return out
