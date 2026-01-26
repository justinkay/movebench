"""
Google Earth Engine download functionality for MoveBench.

This module provides functions for downloading environmental raster data:
- ERA5 / ERA5-Land atmospheric data (hourly, ~11km)
- MODIS products: LST, VI, NDSI, BAI (~250m-1km)
- DEM tiles (static, 30m)

Downloads are organized per-study and use the naming convention from config.py.
"""

import json
import multiprocessing as mp
from pathlib import Path
from functools import wraps

import numpy as np
import pandas as pd
import requests
from retry import retry

from movebench.config import (
    PATHS,
    CHIP_GROUPS,
    LAYERS,
    REGION_MULTIPLIER,
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
            else:
                raise ValueError(f"Unknown chip group: {group_name}")

            # Compute download scale
            scale = compute_group_scale(buffer_km, target_scale_m)

            params = {
                "scale": scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            url = img.getDownloadURL(params)
            download_tiff(url, out_path)

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

    # Compute buffer from training data
    buffer_km = get_buffer_m_for_study(df) / 1000.0
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

def download_dem_tiles(
    df: pd.DataFrame,
    study_dir: Path,
    n_workers: int = 1,
    skip_existing: bool = True,
    px_per_side: int = 500,
) -> dict:
    """
    Download DEM tiles covering all points in a study.

    Creates a grid of tiles covering the union of all point regions,
    then downloads each tile.

    Args:
        df: DataFrame with trajectory data
        study_dir: Path to study directory
        n_workers: Number of parallel workers (currently sequential due to GEE limits)
        skip_existing: Skip tiles that already exist
        px_per_side: Target pixels per tile side

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

    dem_config = LAYERS["dem"]
    dem_scale = compute_group_scale(buffer_km, dem_config.native_scale_m)

    print(f"[download_dem_tiles] Buffer: {buffer_km:.3f} km, scale: {dem_scale:.1f}m")

    # Build regions for all points
    regions = []
    for _, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        split = row[SET_COL]
        region_mult = REGION_MULTIPLIER.get(split, 1.0)
        region = square_region_for_point(lon, lat, region_mult * buffer_km)
        regions.append(region)

    print(f"[download_dem_tiles] Creating union of {len(regions)} regions...")
    union_region = union_regions(regions)

    # Create grid covering the union
    crs = "EPSG:4326"
    grid_size_m = dem_scale * px_per_side
    bounds = union_region.bounds(proj=crs, maxError=10)
    grid = bounds.coveringGrid(proj=crs, scale=grid_size_m)
    filtered_grid = grid.filter(ee.Filter.intersects('.geo', union_region))

    tile_ids = filtered_grid.aggregate_array('system:index').getInfo()
    download_regions = [
        ee.Feature(filtered_grid.toList(1, i).get(0)).geometry()
        for i, _ in enumerate(tile_ids)
    ]

    print(f"[download_dem_tiles] Created {len(download_regions)} tiles")

    # Create output directory
    tile_dir = get_tile_dir(str(study_dir).replace(str(PATHS["studies"]) + "/", ""), "dem")
    # Actually use the study_dir directly
    tile_dir = study_dir / "tiles" / "dem"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Download DEM tiles
    dem_img = make_dem_image()

    # Unmask to ensure consistent dimensions
    nodata_value = 0
    dem_img = dem_img.unmask(value=nodata_value, sameFootprint=False)

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
            print(f"[download_dem_tiles] Downloading tile {i+1}/{len(download_regions)}")
            params = {
                "scale": dem_scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            url = dem_img.getDownloadURL(params)
            download_tiff(url, tile_path)

            tile_meta["status"] = "downloaded"
            stats["downloaded"] += 1

        except Exception as e:
            print(f"[download_dem_tiles] Failed tile {i}: {e}")
            tile_meta["status"] = "failed"
            tile_meta["error"] = str(e)
            stats["failed"] += 1

        tile_index.append(tile_meta)

    # Save tile index
    index_path = tile_dir / "index.json"
    index_data = {
        "buffer_km": buffer_km,
        "dem_scale": dem_scale,
        "num_tiles": len(tile_index),
        "tiles": tile_index,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"[download_dem_tiles] Complete: {stats['downloaded']} downloaded, "
          f"{stats['existing']} existing, {stats['failed']} failed")

    stats["index_path"] = str(index_path)
    return stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def download_study_rasters(
    study_id: str,
    splits: list[str] | str | None = None,
    groups: list[str] | None = None,
    include_dem: bool = True,
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
        groups: List of chip groups to download. Default: all groups in CHIP_GROUPS
        include_dem: Whether to download DEM tiles
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

    if groups is None:
        groups = list(CHIP_GROUPS.keys())

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
    print(f"[download_study_rasters] Splits: {splits}, Groups: {groups}, DEM: {include_dem}")

    results = {"chips": {}, "tiles": {}}

    # Download chips for each split
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
            groups=groups,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    # Download DEM tiles (covers all splits)
    if include_dem:
        print(f"\n[download_study_rasters] Downloading DEM tiles...")
        results["tiles"]["dem"] = download_dem_tiles(
            df,
            study_dir,
            n_workers=1,  # DEM download is sequential due to GEE rate limits
            skip_existing=skip_existing,
        )

    print(f"\n[download_study_rasters] All downloads complete for {study_id}")
    return results
