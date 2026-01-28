"""
Google Earth Engine download functionality for MoveBench.

This module provides functions for downloading environmental raster data:
- ERA5 / ERA5-Land atmospheric data (hourly, ~11km)
- MODIS products: LST, VI, NDSI, BAI (~250m-1km)
- DEM tiles (static, 30m)

Downloads are organized per-study and use the naming convention from config.py.
"""

import json
import math
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
import requests
from retry import retry

from movebench.config import (
    PATHS,
    CHIP_GROUPS,
    GEE_LAYERS,
    LAYERS,
    REGION_MULTIPLIER,
    GEE_MAX_DOWNLOAD_SIZE_MB,
    LON_COL,
    LAT_COL,
    TIME_COL,
    SET_COL,
    POINT_ID_COL,
    compute_group_scale,
    get_study_dir,
    get_chip_path,
    get_tile_dir,
)


# =============================================================================
# LAZY GEE INITIALIZATION
# =============================================================================

_ee_initialized = False
_ee = None


def ensure_ee_initialized():
    """
    Lazily initialize Earth Engine with high-volume endpoint.

    This avoids breaking imports when GEE is not available.
    """
    global _ee_initialized, _ee
    if not _ee_initialized:
        import ee
        _ee = ee
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
        _ee_initialized = True
    return _ee


# =============================================================================
# DOWNLOAD UTILITIES
# =============================================================================

@retry((requests.RequestException,), tries=5, delay=2, backoff=2)
def download_tiff(url: str, out_path: Path) -> None:
    """
    Download a GeoTIFF from URL with retry logic.

    Args:
        url: GEE download URL
        out_path: Local file path to save to
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1 << 20):  # 1MB chunks
            if chunk:
                f.write(chunk)


def download_tiled_image(img, region, scale: float, out_path: Path, crs: str = "EPSG:4326") -> None:
    """
    Download a large GEE image by splitting into tiles and merging.

    When the full region exceeds GEE's download size limit, this function
    splits the region into a grid of smaller tiles, downloads each tile,
    and merges them into a single output file.

    Args:
        img: ee.Image to download
        region: ee.Geometry defining the region to download
        scale: Resolution in meters per pixel
        out_path: Output file path
        crs: Coordinate reference system (default EPSG:4326)
    """
    ee = ensure_ee_initialized()

    # Get region bounds
    bounds_info = region.bounds().getInfo()
    coords = bounds_info['coordinates'][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    # Estimate image size
    # At equator: 1 degree ≈ 111km. Use cos(lat) correction for longitude.
    center_lat = (min_lat + max_lat) / 2
    lon_deg_to_m = 111320 * math.cos(math.radians(center_lat))
    lat_deg_to_m = 110540

    width_m = (max_lon - min_lon) * lon_deg_to_m
    height_m = (max_lat - min_lat) * lat_deg_to_m
    width_px = width_m / scale
    height_px = height_m / scale

    # Get band count from image
    band_names = img.bandNames().getInfo()
    num_bands = len(band_names)
    bytes_per_pixel = 4  # float32

    estimated_size_bytes = width_px * height_px * num_bands * bytes_per_pixel
    # GEE's actual limit is 48MB (50,331,648 bytes), use that directly
    gee_limit_bytes = 50_331_648

    # If small enough, download directly
    if estimated_size_bytes <= gee_limit_bytes * 0.7:  # 30% safety margin
        params = {
            "scale": scale,
            "region": region,
            "format": "GEO_TIFF",
            "crs": crs,
        }
        url = img.getDownloadURL(params)
        download_tiff(url, out_path)
        return

    # Calculate grid dimensions
    # Each tile should be well under the limit - use 35% to account for GEE overhead
    pixels_per_tile = (gee_limit_bytes * 0.35) / (num_bands * bytes_per_pixel)
    tile_side_px = math.sqrt(pixels_per_tile)
    tile_side_m = tile_side_px * scale

    # Convert to degrees
    tile_width_deg = tile_side_m / lon_deg_to_m
    tile_height_deg = tile_side_m / lat_deg_to_m

    n_cols = math.ceil((max_lon - min_lon) / tile_width_deg)
    n_rows = math.ceil((max_lat - min_lat) / tile_height_deg)

    # Download tiles to temp directory and merge
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tile_paths = []
        tile_idx = 0

        for row in range(n_rows):
            for col in range(n_cols):
                tile_min_lon = min_lon + col * tile_width_deg
                tile_max_lon = min(tile_min_lon + tile_width_deg, max_lon)
                tile_min_lat = min_lat + row * tile_height_deg
                tile_max_lat = min(tile_min_lat + tile_height_deg, max_lat)

                tile_region = ee.Geometry.Rectangle([
                    tile_min_lon, tile_min_lat,
                    tile_max_lon, tile_max_lat
                ])

                tile_path = Path(tmp_dir) / f"tile_{tile_idx:03d}.tif"

                params = {
                    "scale": scale,
                    "region": tile_region,
                    "format": "GEO_TIFF",
                    "crs": crs,
                }

                try:
                    url = img.getDownloadURL(params)
                    download_tiff(url, tile_path)
                    tile_paths.append(tile_path)
                except Exception as e:
                    print(f"[WARN] Tile {tile_idx} download failed: {e}")

                tile_idx += 1

        if not tile_paths:
            raise RuntimeError("All tile downloads failed")

        # Merge tiles using rasterio
        # Suppress GDAL warnings about TIFF photometric interpretation for multi-band GEE files
        old_cpl_log = os.environ.get('CPL_LOG', '')
        os.environ['CPL_LOG'] = '/dev/null'
        src_files = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, mosaic_transform = merge(src_files)

            # Write merged output
            profile = src_files[0].profile.copy()
            profile.update({
                'height': mosaic.shape[1],
                'width': mosaic.shape[2],
                'transform': mosaic_transform,
            })

            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(mosaic)
        finally:
            for src in src_files:
                src.close()
            # Restore GDAL logging
            if old_cpl_log:
                os.environ['CPL_LOG'] = old_cpl_log
            else:
                os.environ.pop('CPL_LOG', None)


# =============================================================================
# EARTH ENGINE HELPERS
# =============================================================================

def square_region_for_point(lon: float, lat: float, buffer_km: float):
    """Create a square bounding box region around a point."""
    ee = ensure_ee_initialized()
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(buffer_km * 1000.0).bounds()


def union_regions(regions: list):
    """Create a union of multiple regions."""
    ee = ensure_ee_initialized()
    return ee.Geometry.MultiPolygon(regions)


# =============================================================================
# TEMPORAL HELPERS
# =============================================================================

def _nearest_hourly_image(collection_id: str, timestamp: str, band_list: list):
    """Get nearest hourly image from a collection."""
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    start = ts.advance(-1, "hour")
    end = ts.advance(1, "hour")

    col = (ee.ImageCollection(collection_id)
           .select(band_list)
           .filterDate(start, end)
           .sort("system:time_start"))
    return ee.Image(col.first())


def _nearest_modis_collection(collection_id: str, timestamp, days_window: int, select_bands: list):
    """Get nearest MODIS composite image."""
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    start = ts.advance(-days_window, "day")
    end = ts.advance(days_window, "day")

    col = (ee.ImageCollection(collection_id)
           .select(select_bands)
           .filterDate(start, end)
           .sort("system:time_start"))
    return ee.Image(col.first())


# =============================================================================
# IMAGE STACK BUILDERS
# =============================================================================

def make_era5_stack(timestamp: str):
    """
    Build ERA5-Land + ERA5 image stack for a timestamp.

    Returns an ee.Image with ~15 bands of atmospheric data.
    """
    ee = ensure_ee_initialized()

    # ERA5-Land hourly bands
    era5l_config = LAYERS["era5_land"]
    era5l_bands = era5l_config.bands
    era5l_img = _nearest_hourly_image(
        era5l_config.gee_collection,
        timestamp,
        era5l_bands
    ).rename(["era5l_" + b for b in era5l_bands])

    # ERA5 hourly bands
    era5_config = LAYERS["era5"]
    era5_bands = era5_config.bands
    era5_img = _nearest_hourly_image(
        era5_config.gee_collection,
        timestamp,
        era5_bands
    ).rename(["era5_" + b for b in era5_bands])

    return era5l_img.addBands(era5_img).toFloat()


def make_modis_stack(timestamp: str):
    """
    Build MODIS image stack for a timestamp.

    Includes: LST (8-day), NDVI/EVI (16-day), NDSI, BAI.
    Returns an ee.Image with ~6 bands.
    """
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)

    # MODIS LST (8-day composite)
    lst_config = LAYERS["modis_lst"]
    mod11 = _nearest_modis_collection(
        lst_config.gee_collection,
        ts,
        days_window=4,
        select_bands=lst_config.bands
    )
    lst_day = mod11.select("LST_Day_1km").multiply(lst_config.scale_factor)
    lst_night = mod11.select("LST_Night_1km").multiply(lst_config.scale_factor)
    lst_img = lst_day.addBands(lst_night).rename(
        ["mod11a2_LST_Day_K", "mod11a2_LST_Night_K"]
    )

    # MODIS VI (16-day composite)
    vi_config = LAYERS["modis_vi"]
    mod13 = _nearest_modis_collection(
        vi_config.gee_collection,
        ts,
        days_window=10,
        select_bands=vi_config.bands
    )
    ndvi = mod13.select("NDVI").multiply(vi_config.scale_factor)
    evi = mod13.select("EVI").multiply(vi_config.scale_factor)
    vi_img = ndvi.addBands(evi).rename(["mod13q1_NDVI", "mod13q1_EVI"])

    # MODIS NDSI
    ndsi_config = LAYERS["modis_ndsi"]
    ndsi_img = _nearest_modis_collection(
        ndsi_config.gee_collection,
        ts,
        days_window=10,
        select_bands=ndsi_config.bands
    ).select("NDSI").rename("mcd43a4_NDSI")

    # MODIS BAI
    bai_config = LAYERS["modis_bai"]
    bai_img = _nearest_modis_collection(
        bai_config.gee_collection,
        ts,
        days_window=10,
        select_bands=bai_config.bands
    ).select("BAI").rename("mcd43a4_BAI")

    return (lst_img
            .addBands(vi_img)
            .addBands(ndsi_img)
            .addBands(bai_img)
            .toFloat())


def make_dynamic_world_stack(timestamp: str, region):
    """
    Build Dynamic World probability stack for a timestamp.

    Uses a 30-day median composite to handle Sentinel-2 gaps from clouds.
    Falls back to 60-day window if needed.

    Args:
        timestamp: ISO format timestamp string
        region: ee.Geometry region to filter by

    Returns:
        ee.Image with 9 probability bands (dw_water, dw_trees, etc.)
    """
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)

    dw_config = LAYERS["dynamic_world"]
    dw_bands = dw_config.bands

    # Primary: 30-day window median composite
    dw_col_30 = (ee.ImageCollection(dw_config.gee_collection)
                 .filterDate(ts.advance(-15, "day"), ts.advance(15, "day"))
                 .filterBounds(region)
                 .select(dw_bands))

    # Fallback: 60-day window if 30-day is empty
    dw_col_60 = (ee.ImageCollection(dw_config.gee_collection)
                 .filterDate(ts.advance(-30, "day"), ts.advance(30, "day"))
                 .filterBounds(region)
                 .select(dw_bands))

    # Use conditional to pick appropriate collection
    dw_col = ee.ImageCollection(
        ee.Algorithms.If(
            dw_col_30.size().gt(0),
            dw_col_30,
            dw_col_60
        )
    )

    # Median composite (robust to clouds and outliers)
    dw_img = dw_col.median()

    # Rename bands for clarity
    dw_img = dw_img.rename(["dw_" + b for b in dw_bands])

    return dw_img.toFloat()


def make_dem_image():
    """
    Create DEM image from Copernicus GLO-30.

    Returns ee.Image with single 'dem_glo30' band.
    """
    ee = ensure_ee_initialized()
    dem_config = LAYERS["dem"]
    dem_ic = ee.ImageCollection(dem_config.gee_collection)
    dem_img = dem_ic.mosaic().select("DEM").rename("dem_glo30")
    return dem_img.toFloat()


def make_dist_water_image():
    """
    Create distance-to-water image from JRC Global Surface Water.

    Uses max_extent band to identify any pixel that has ever been water,
    then computes Euclidean distance transform. Result in meters.
    Max detectable distance: ~30km at 30m resolution (neighborhood=1024).

    Returns ee.Image with single 'dist_water' band.
    """
    ee = ensure_ee_initialized()
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
    water_mask = gsw.select("max_extent").gt(0)
    # fastDistanceTransform returns squared pixel distance
    dist_px2 = water_mask.fastDistanceTransform(neighborhood=1024).sqrt()
    # Convert pixel distance to meters (native 30m resolution)
    dist_m = dist_px2.multiply(30.0).rename("dist_water")
    return dist_m.toFloat()


def make_dist_coast_image():
    """
    Create distance-to-coast image from LSIB land boundaries.

    Rasterizes land polygons, inverts to get ocean mask, then computes
    distance transform. Result in meters.
    At 1km resolution, max detectable distance: ~2000km (neighborhood=2048).

    Returns ee.Image with single 'dist_coast' band.
    """
    ee = ensure_ee_initialized()
    land = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    # Rasterize land polygons to a binary mask
    land_img = land.reduceToImage(
        properties=[ee.Feature(land.first()).propertyNames().get(0)],
        reducer=ee.Reducer.count(),
    ).gt(0)
    # Ocean mask (invert land)
    ocean_mask = land_img.Not()
    # Distance from each pixel to nearest ocean pixel = distance to coast
    dist_px2 = ocean_mask.fastDistanceTransform(neighborhood=2048).sqrt()
    dist_m = dist_px2.multiply(1000.0).rename("dist_coast")
    return dist_m.toFloat()


def make_dist_road_image():
    """
    Create distance-to-road image from GRIP4 global roads dataset.

    Merges road FeatureCollections from all continents, rasterizes,
    then computes distance transform. Result in meters.
    Max detectable distance: ~15km at 30m resolution (neighborhood=512).

    Returns ee.Image with single 'dist_road' band.
    """
    ee = ensure_ee_initialized()
    continents = [
        "projects/sat-io/open-datasets/GRIP4/Africa",
        "projects/sat-io/open-datasets/GRIP4/Asia",
        "projects/sat-io/open-datasets/GRIP4/Europe",
        "projects/sat-io/open-datasets/GRIP4/MENA",
        "projects/sat-io/open-datasets/GRIP4/North-America",
        "projects/sat-io/open-datasets/GRIP4/Oceania",
        "projects/sat-io/open-datasets/GRIP4/South-America",
    ]
    # Merge all continents
    roads = ee.FeatureCollection(continents[0])
    for c in continents[1:]:
        roads = roads.merge(ee.FeatureCollection(c))
    # Rasterize roads (paint 1 on road pixels)
    road_img = ee.Image(0).paint(roads, 1).gt(0)
    dist_px2 = road_img.fastDistanceTransform(neighborhood=512).sqrt()
    dist_m = dist_px2.multiply(30.0).rename("dist_road")
    return dist_m.toFloat()


# Map tile layer names to their image builder functions
_TILE_IMAGE_BUILDERS = {
    "dem": make_dem_image,
    "dist_water": make_dist_water_image,
    "dist_coast": make_dist_coast_image,
    "dist_road": make_dist_road_image,
}


# =============================================================================
# MOVEMENT ANALYSIS (for buffer computation)
# =============================================================================

def get_buffer_m_for_study(df: pd.DataFrame, step_q: float = 0.95, step_multiplier: int = 2) -> float:
    """
    Compute appropriate buffer for a study based on movement data.

    Args:
        df: DataFrame with movement data
        step_q: Quantile for step length calculation (default 0.95)
        step_multiplier: Multiplier for final buffer size (default 2)

    Returns:
        Buffer distance in meters
    """
    if TIME_COL not in df.columns:
        raise RuntimeError(f"Expected a '{TIME_COL}' column in the data.")

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # Sort and compute time gaps
    df = df.sort_values(['individual-local-identifier', TIME_COL])
    df['time_since_last'] = df.groupby('individual-local-identifier')[TIME_COL].diff()

    # Filter by per-individual time gap quantile
    gap_q = 0.999
    per_indiv_cutoff = (
        df.groupby('individual-local-identifier')['time_since_last']
        .transform(lambda s: s.dropna().quantile(gap_q))
    )
    df_filtered = df[
        df['time_since_last'].notna() &
        (df['time_since_last'] <= per_indiv_cutoff)
    ].copy()

    # Compute step lengths using Haversine formula
    lat_rad = np.radians(df_filtered[LAT_COL])
    lon_rad = np.radians(df_filtered[LON_COL])

    lat_prev = lat_rad.groupby(df_filtered['individual-local-identifier']).shift()
    lon_prev = lon_rad.groupby(df_filtered['individual-local-identifier']).shift()

    dlat = lat_rad - lat_prev
    dlon = lon_rad - lon_prev

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # Earth radius in meters
    df_filtered['step_length'] = R * c
    df_filtered = df_filtered.dropna(subset=['step_length'])

    if len(df_filtered) == 0:
        raise ValueError("No valid step lengths computed")

    return step_multiplier * np.quantile(df_filtered['step_length'].values, q=step_q)


# =============================================================================
# CHIP DOWNLOAD FUNCTIONS
# =============================================================================

def _download_chip_worker(args: tuple) -> dict:
    """
    Worker function to download a single chip.

    Args:
        args: Tuple of (idx, request_dict)

    Returns:
        Dict with download results
    """
    idx, req = args

    point_id = req["point_id"]
    lon = req["lon"]
    lat = req["lat"]
    buffer_km = float(req["buffer_km"])
    timestamp = req["timestamp"]
    split = req["split"]
    study_dir = Path(req["study_dir"])
    groups = req["groups"]
    skip_existing = req["skip_existing"]

    results = {"point_id": point_id, "split": split, "files": {}}

    # Check if all files exist before initializing GEE
    all_exist = True
    for group_name in groups:
        out_path = study_dir / "chips" / split / f"{point_id:05d}_{group_name}.tif"
        if not (skip_existing and out_path.exists()):
            all_exist = False
            break

    if all_exist:
        for group_name in groups:
            out_path = study_dir / "chips" / split / f"{point_id:05d}_{group_name}.tif"
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
        return results

    # Only initialize GEE if we need to download something
    ensure_ee_initialized()

    region_mult = REGION_MULTIPLIER[split]
    region = square_region_for_point(lon, lat, region_mult * buffer_km)

    for group_name in groups:
        group_config = CHIP_GROUPS[group_name]
        target_scale_m = group_config["target_scale_m"]

        out_path = study_dir / "chips" / split / f"{point_id:05d}_{group_name}.tif"

        if skip_existing and out_path.exists():
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
            continue

        try:
            # Build image stack
            if group_name == "era5":
                img = make_era5_stack(timestamp)
            elif group_name == "modis":
                img = make_modis_stack(timestamp)
            elif group_name == "dynamic_world":
                img = make_dynamic_world_stack(timestamp, region)
            else:
                raise ValueError(f"Unknown chip group: {group_name}")

            # Compute download scale
            scale = compute_group_scale(buffer_km, target_scale_m)

            # Use tiled download (handles large regions automatically)
            download_tiled_image(img, region, scale, out_path, crs="EPSG:4326")

            results["files"][group_name] = {"path": str(out_path), "status": "downloaded"}
            print(f"[OK] point={point_id} group={group_name}")

        except Exception as e:
            results["files"][group_name] = {"path": None, "status": "failed", "error": str(e)}
            print(f"[FAIL] point={point_id} group={group_name}: {e}")

    return results


def download_chips(
    df: pd.DataFrame,
    study_dir: Path,
    split: str,
    buffer_km: float,
    groups: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict:
    """
    Download chip rasters for all points in a dataframe.

    Args:
        df: DataFrame with trajectory data (must have LON_COL, LAT_COL, TIME_COL, POINT_ID_COL)
        study_dir: Path to study directory
        split: "train" or "test"
        buffer_km: Buffer radius in kilometers (should be computed from training data)
        groups: List of chip groups to download (default: all in CHIP_GROUPS)
        n_workers: Number of parallel workers
        skip_existing: Skip chips that already exist

    Returns:
        Dict with download statistics
    """
    if groups is None:
        groups = list(CHIP_GROUPS.keys())

    # Validate columns
    required_cols = [LON_COL, LAT_COL, TIME_COL, POINT_ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    print(f"[download_chips] Buffer: {buffer_km:.3f} km, split: {split}, groups: {groups}")

    # Build request list
    requests = []
    for _, row in df.iterrows():
        ts = row[TIME_COL]
        if hasattr(ts, "isoformat"):
            ts_iso = ts.isoformat()
        else:
            ts_iso = str(ts)

        requests.append({
            "point_id": int(row[POINT_ID_COL]),
            "lon": float(row[LON_COL]),
            "lat": float(row[LAT_COL]),
            "buffer_km": buffer_km,
            "timestamp": ts_iso,
            "split": split,
            "study_dir": str(study_dir),
            "groups": groups,
            "skip_existing": skip_existing,
        })

    print(f"[download_chips] Starting download of {len(requests)} chips with {n_workers} workers")

    # Create output directory
    (study_dir / "chips" / split).mkdir(parents=True, exist_ok=True)

    # Download with multiprocessing (lazy GEE init in each worker)
    with mp.Pool(n_workers) as pool:
        results = pool.map(_download_chip_worker, enumerate(requests))

    # Aggregate statistics
    stats = {
        "total": len(results),
        "downloaded": 0,
        "existing": 0,
        "failed": 0,
        "failed_points": [],
    }

    for r in results:
        for group_name, file_info in r["files"].items():
            if file_info["status"] == "downloaded":
                stats["downloaded"] += 1
            elif file_info["status"] == "existing":
                stats["existing"] += 1
            else:
                stats["failed"] += 1
                stats["failed_points"].append({
                    "point_id": r["point_id"],
                    "group": group_name,
                    "error": file_info.get("error", "unknown"),
                })

    print(f"[download_chips] Complete: {stats['downloaded']} downloaded, "
          f"{stats['existing']} existing, {stats['failed']} failed")

    return stats


# =============================================================================
# DEM TILE DOWNLOAD FUNCTIONS
# =============================================================================

def download_tile_layer(
    df: pd.DataFrame,
    study_dir: Path,
    layer_name: str,
    make_image_fn,
    skip_existing: bool = True,
    px_per_side: int = 500,
    nodata_value: float = 0,
) -> dict:
    """
    Download tiles for a layer covering all points in a study.

    Creates a grid of tiles covering the union of all point regions,
    then downloads each tile.

    Args:
        df: DataFrame with trajectory data
        study_dir: Path to study directory
        layer_name: Name of the layer (used for directory naming)
        make_image_fn: Callable returning an ee.Image for this layer
        skip_existing: Skip tiles that already exist
        px_per_side: Target pixels per tile side
        nodata_value: Value to use for unmasking (default 0)

    Returns:
        Dict with download statistics and tile index
    """
    ee = ensure_ee_initialized()

    # Validate columns
    required_cols = [LON_COL, LAT_COL, SET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Compute buffer from training data only
    train_df = df[df[SET_COL] == "train"]
    buffer_km = get_buffer_m_for_study(train_df) / 1000.0

    layer_config = LAYERS[layer_name]
    layer_scale = compute_group_scale(buffer_km, layer_config.native_scale_m)

    print(f"[download_tile_layer:{layer_name}] Buffer: {buffer_km:.3f} km, scale: {layer_scale:.1f}m")

    # Build regions for all points
    regions = []
    for _, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        split = row[SET_COL]
        region_mult = REGION_MULTIPLIER.get(split, 1.0)
        region = square_region_for_point(lon, lat, region_mult * buffer_km)
        regions.append(region)

    print(f"[download_tile_layer:{layer_name}] Creating union of {len(regions)} regions...")
    union_region = union_regions(regions)

    # Create grid covering the union
    crs = "EPSG:4326"
    grid_size_m = layer_scale * px_per_side
    bounds = union_region.bounds(proj=crs, maxError=10)
    grid = bounds.coveringGrid(proj=crs, scale=grid_size_m)
    filtered_grid = grid.filter(ee.Filter.intersects('.geo', union_region))

    tile_ids = filtered_grid.aggregate_array('system:index').getInfo()
    download_regions = [
        ee.Feature(filtered_grid.toList(1, i).get(0)).geometry()
        for i, _ in enumerate(tile_ids)
    ]

    print(f"[download_tile_layer:{layer_name}] Created {len(download_regions)} tiles")

    # Create output directory
    tile_dir = study_dir / "tiles" / layer_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Build the image
    img = make_image_fn()
    img = img.unmask(value=nodata_value, sameFootprint=False)

    tile_index = []
    stats = {
        "total": len(download_regions),
        "downloaded": 0,
        "existing": 0,
        "failed": 0,
    }

    for i, region in enumerate(download_regions):
        tile_path = tile_dir / f"tile_{i:03d}.tif"

        # Get bounds for index
        info = region.getInfo()
        coords = info['coordinates'][0]
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)

        tile_meta = {
            "tile_id": i,
            "file": f"tile_{i:03d}.tif",
            "bounds": {
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat,
            },
        }

        if skip_existing and tile_path.exists():
            tile_meta["status"] = "existing"
            tile_index.append(tile_meta)
            stats["existing"] += 1
            continue

        try:
            print(f"[download_tile_layer:{layer_name}] Downloading tile {i+1}/{len(download_regions)}")
            params = {
                "scale": layer_scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            url = img.getDownloadURL(params)
            download_tiff(url, tile_path)

            tile_meta["status"] = "downloaded"
            stats["downloaded"] += 1

        except Exception as e:
            print(f"[download_tile_layer:{layer_name}] Failed tile {i}: {e}")
            tile_meta["status"] = "failed"
            tile_meta["error"] = str(e)
            stats["failed"] += 1

        tile_index.append(tile_meta)

    # Save tile index
    index_path = tile_dir / "index.json"
    index_data = {
        "layer_name": layer_name,
        "buffer_km": buffer_km,
        "scale": layer_scale,
        "num_tiles": len(tile_index),
        "tiles": tile_index,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"[download_tile_layer:{layer_name}] Complete: {stats['downloaded']} downloaded, "
          f"{stats['existing']} existing, {stats['failed']} failed")

    stats["index_path"] = str(index_path)
    return stats


def download_dem_tiles(
    df: pd.DataFrame,
    study_dir: Path,
    n_workers: int = 1,
    skip_existing: bool = True,
    px_per_side: int = 500,
) -> dict:
    """
    Download DEM tiles covering all points in a study.

    Thin wrapper around download_tile_layer for backwards compatibility.
    """
    return download_tile_layer(
        df, study_dir, "dem", make_dem_image,
        skip_existing=skip_existing, px_per_side=px_per_side,
        nodata_value=0,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_study_rasters(
    study_id: str,
    splits: list[str] | str | None = None,
    layers: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict:
    """
    Download all GEE rasters for a study.

    This is the main entry point for downloading environmental data.

    Args:
        study_id: Study identifier (UUID)
        splits: List of splits to download ("train", "test"), or single split.
                Default: ["train", "test"]
        layers: List of layers to download. Accepts chip group names
                ("era5", "modis", "dynamic_world") and tile layer names ("dem").
                Default: all GEE layers
        n_workers: Number of parallel download workers
        skip_existing: Skip files that already exist

    Returns:
        Dict with download results for each split and layer type

    Example:
        >>> result = download_study_rasters("abc123-def456")
        >>> print(result["chips"]["train"]["downloaded"])
    """
    # Normalize splits argument
    if splits is None:
        splits = ["train", "test"]
    elif isinstance(splits, str):
        splits = [splits]

    # Normalize layers - default to all GEE layers
    if layers is None:
        layers = GEE_LAYERS

    # Partition into chip groups vs tile layers
    chip_groups = [l for l in layers if l in CHIP_GROUPS]
    tile_layers = [l for l in layers if l not in CHIP_GROUPS]

    # Load study data
    study_dir = get_study_dir(study_id)
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")

    traj_path = study_dir / "trajectories.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {traj_path}")

    df = pd.read_csv(traj_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL])

    print(f"[download_study_rasters] Study: {study_id}")
    print(f"[download_study_rasters] Total points: {len(df)}")
    print(f"[download_study_rasters] Splits: {splits}, Chip groups: {chip_groups}, Tile layers: {tile_layers}")

    # Compute buffer from training data (used consistently for all splits)
    train_df = df[df[SET_COL] == "train"]
    if len(train_df) == 0:
        raise ValueError("No training data found - cannot compute buffer")
    buffer_km = get_buffer_m_for_study(train_df) / 1000.0

    results = {"chips": {}, "tiles": {}}

    # Download chips for each split (if any chip groups requested)
    if chip_groups:
        for split in splits:
            split_df = df[df[SET_COL] == split].copy()
            if len(split_df) == 0:
                print(f"[download_study_rasters] No data for split '{split}', skipping")
                results["chips"][split] = {"total": 0, "downloaded": 0, "existing": 0, "failed": 0}
                continue

            print(f"\n[download_study_rasters] Downloading {split} chips ({len(split_df)} points)...")
            results["chips"][split] = download_chips(
                split_df,
                study_dir,
                split,
                buffer_km=buffer_km,
                groups=chip_groups,
                n_workers=n_workers,
                skip_existing=skip_existing,
            )

    # Download tile layers (if any requested)
    for layer in tile_layers:
        if layer not in _TILE_IMAGE_BUILDERS:
            print(f"[download_study_rasters] Unknown tile layer '{layer}', skipping")
            continue
        print(f"\n[download_study_rasters] Downloading {layer} tiles...")
        results["tiles"][layer] = download_tile_layer(
            df,
            study_dir,
            layer,
            _TILE_IMAGE_BUILDERS[layer],
            skip_existing=skip_existing,
        )

    print(f"\n[download_study_rasters] All downloads complete for {study_id}")
    return results
