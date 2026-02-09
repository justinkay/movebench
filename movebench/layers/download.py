"""Earth Engine raster downloads for movebench.

This module writes rasters to the on-disk layout expected by `movebench.layers.io.LayerIO`:

- Chips (per-point): `study_dir/rasters/<rasters_hash>/chips/<chip_group>/<point_id:05d>.tif`
- Tiles (per-study): `study_dir/rasters/<rasters_hash>/tiles/<layer>/tile_XXX.tif` + `index.json`

All masked pixels are materialized as the numeric sentinel `-9999` in GeoTIFFs.
`LayerIO` converts this sentinel to NaN immediately after reading.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import requests
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from movebench.layers.plan import LayerPlan
from movebench.layers.registry import LayerRegistry, REGISTRY


# Movebank / trajectories.csv schema
LON_COL = "location-long"
LAT_COL = "location-lat"
TIME_COL = "timestamp"
INDIVIDUAL_COL = "individual-local-identifier"
SET_COL = "set"
POINT_ID_COL = "point_id"


# Train/test region multipliers used when constructing chip regions.
REGION_MULTIPLIER: dict[str, float] = {
    "train": 1.0,
    "test": 1.0,
}


# Chip-scale constraints 
MIN_PIXELS = 3
MAX_PIXELS = 100


GEE_NODATA_SENTINEL = -9999.0


_ee_initialized = False
_ee = None


def ensure_ee_initialized():
    """Lazily initialize the Earth Engine client.

    This keeps imports working on machines without Earth Engine installed.
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
    native_scale_m: float,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
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
        try:
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
            return
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(delay_s)
            delay_s *= 2.0


def square_region_for_point(lon: float, lat: float, buffer_km: float):
    ee = ensure_ee_initialized()
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(float(buffer_km) * 1000.0).bounds()


def union_regions(regions: list):
    ee = ensure_ee_initialized()
    return ee.Geometry.MultiPolygon(regions)


def _fully_masked_image(bands: list[str]):
    ee = ensure_ee_initialized()
    base = ee.Image.constant([0] * len(bands)).rename(bands).toFloat()
    return base.updateMask(ee.Image.constant(0))


def _parse_iso_timestamp(ts: str) -> datetime:
    # Accept "Z" suffix for UTC.
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _nearest_from_window(collection_id: str, timestamp: str, *, bands: list[str], start: int, end: int, unit: str):
    ee = ensure_ee_initialized()
    ts = ee.Date(timestamp)
    col = (
        ee.ImageCollection(collection_id)
        .filterDate(ts.advance(start, unit), ts.advance(end, unit))
        .select(bands)
        .sort("system:time_start")
    )
    fallback = _fully_masked_image(bands)
    return ee.Image(ee.Algorithms.If(col.size().gt(0), ee.Image(col.first()), fallback))


def _chip_layer_image(layer: str, *, spec, timestamp: str, region) -> Any:
    """Build an ee.Image for one gee_chip layer, with uniquely-named bands."""
    ee = ensure_ee_initialized()

    if layer == "landsat_ndwi":
        ts = ee.Date(timestamp)
        col = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(ts.advance(-32, "day"), ts.advance(32, "day"))
            .filterBounds(region)
            .select(spec.bands)
        )
        fallback = _fully_masked_image(spec.bands)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), fallback))
        img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
        return img

    if layer == "dynamic_world":
        DW_START = "2015-06-23"
        ts_dt = _parse_iso_timestamp(timestamp)
        dw_start_dt = datetime(2015, 6, 23, tzinfo=ts_dt.tzinfo)
        was_clamped = ts_dt < dw_start_dt

        if was_clamped:
            col = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(DW_START, "2016-06-23")
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

        fallback = _fully_masked_image(spec.bands)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), fallback))
        img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
        return img

    temporal = spec.temporal
    if temporal == "hourly":
        img = _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=-1, end=1, unit="hour")
    elif temporal == "daily":
        # MODIS "daily" products are more robust with a wider window.
        if layer in {"modis_ndsi", "modis_bai"}:
            img = _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=-10, end=10, unit="day")
        else:
            img = _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=-1, end=1, unit="day")
    elif temporal == "8day":
        img = _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=-4, end=4, unit="day")
    elif temporal == "16day":
        img = _nearest_from_window(spec.gee_collection, timestamp, bands=spec.bands, start=-10, end=10, unit="day")
    else:
        raise ValueError(f"Unsupported temporal={temporal!r} for gee_chip layer {layer!r}")

    if getattr(spec, "scale_factor", 1.0) not in (None, 1.0):
        img = img.multiply(float(spec.scale_factor))

    img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
    return img


def build_chip_group_image(
    plan: LayerPlan,
    *,
    registry: LayerRegistry,
    chip_group: str,
    timestamp: str,
    region,
) -> Any:
    """Build an ee.Image stack for a chip group, in plan order."""
    ee = ensure_ee_initialized()

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


def download_tiled_image(img, region, *, scale: float, out_path: Path, num_bands: int, crs: str = "EPSG:4326") -> None:
    """Download an ee.Image, tiling if needed, and write GeoTIFF to out_path."""
    ee = ensure_ee_initialized()

    # Get region bounds.
    bounds_info = region.bounds().getInfo()
    coords = bounds_info["coordinates"][0]
    min_lon = min(c[0] for c in coords)
    max_lon = max(c[0] for c in coords)
    min_lat = min(c[1] for c in coords)
    max_lat = max(c[1] for c in coords)

    # Estimate size (rough).
    center_lat = (min_lat + max_lat) / 2.0
    lon_deg_to_m = 111320.0 * math.cos(math.radians(center_lat))
    lat_deg_to_m = 110540.0

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

    import rasterio
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

        src_files = [rasterio.open(p) for p in tile_paths]
        try:
            mosaic, mosaic_transform = merge(src_files)
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


def _tile_image(layer: str, *, registry: LayerRegistry):
    """Build an ee.Image for a gee_tile layer."""
    ee = ensure_ee_initialized()
    spec = registry.get(layer)

    if layer == "dem":
        # COPERNICUS/DEM/GLO30 uses band "DEM"; we export one band.
        return ee.ImageCollection(spec.gee_collection).mosaic().select("DEM").toFloat()

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


def _chip_is_current(path: Path, expected_bands: int) -> bool:
    try:
        import rasterio

        with rasterio.open(path) as ds:
            return int(ds.count) == int(expected_bands)
    except Exception:
        return False


@dataclass(frozen=True)
class ChipRequest:
    point_id: int
    lon: float
    lat: float
    timestamp: str
    split: str


def _download_chip_worker(args: tuple[int, ChipRequest, dict[str, Any]]) -> dict[str, Any]:
    idx, req, ctx = args

    study_dir = Path(ctx["study_dir"])
    buffer_km = float(ctx["buffer_km"])
    skip_existing = bool(ctx["skip_existing"])
    plan: LayerPlan = ctx["plan"]
    registry: LayerRegistry = ctx["registry"]

    results: dict[str, Any] = {"point_id": int(req.point_id), "files": {}}

    # Avoid initializing EE unless we need to download something.
    all_exist = True
    if skip_existing:
        for group_name in plan.chip_groups.keys():
            out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
            expected = _expected_band_count_for_chip_group(plan, group_name, registry)
            if not (out_path.exists() and _chip_is_current(out_path, expected)):
                all_exist = False
                break
    else:
        all_exist = False

    if all_exist:
        for group_name in plan.chip_groups.keys():
            out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
        return results

    ensure_ee_initialized()

    region_mult = float(REGION_MULTIPLIER.get(req.split, 1.0))
    region = square_region_for_point(req.lon, req.lat, region_mult * buffer_km)

    for group_name, group in plan.chip_groups.items():
        out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
        expected = _expected_band_count_for_chip_group(plan, group_name, registry)
        if skip_existing and out_path.exists() and _chip_is_current(out_path, expected):
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
            continue

        target_scale_m = float(group["target_scale_m"])
        scale = compute_group_scale(region_mult * buffer_km, target_scale_m)

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
                crs="EPSG:4326",
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
    buffer_km: float,
    splits: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if splits is None:
        splits = sorted(df[SET_COL].dropna().unique().tolist())

    required_cols = [LON_COL, LAT_COL, TIME_COL, SET_COL, POINT_ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    sub = df[df[SET_COL].isin(splits)].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows matched splits={splits}")

    reqs: list[ChipRequest] = []
    for _, row in sub.iterrows():
        ts = row[TIME_COL]
        ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        reqs.append(
            ChipRequest(
                point_id=int(row[POINT_ID_COL]),
                lon=float(row[LON_COL]),
                lat=float(row[LAT_COL]),
                timestamp=ts_iso,
                split=str(row[SET_COL]),
            )
        )

    # Ensure output dirs exist up front (avoids races on first write).
    for group_name in plan.chip_groups.keys():
        (_rasters_root(study_dir, plan.rasters_hash) / "chips" / group_name).mkdir(parents=True, exist_ok=True)

    ctx = {
        "study_dir": str(study_dir),
        "buffer_km": float(buffer_km),
        "skip_existing": bool(skip_existing),
        "plan": plan,
        "registry": registry,
    }

    from tqdm import tqdm

    with mp.Pool(int(n_workers)) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_download_chip_worker, [(i, r, ctx) for i, r in enumerate(reqs)]),
                total=len(reqs),
                desc="Downloading chips",
                unit="point",
            )
        )

    stats = {"total_points": len(reqs), "downloaded": 0, "existing": 0, "failed": 0, "failures": []}
    for r in results:
        for group_name, info in r.get("files", {}).items():
            if info["status"] == "downloaded":
                stats["downloaded"] += 1
            elif info["status"] == "existing":
                stats["existing"] += 1
            else:
                stats["failed"] += 1
                stats["failures"].append(
                    {"point_id": r.get("point_id"), "group": group_name, "error": info.get("error", "unknown")}
                )
    return stats


def get_buffer_m_for_study(df: pd.DataFrame, *, step_q: float = 0.95, step_multiplier: float = 2.0, gap_q: float = 0.999) -> float:
    """Compute buffer size (meters) from movement data (train split only)."""
    import numpy as np

    if TIME_COL not in df.columns:
        raise RuntimeError(f"Expected a '{TIME_COL}' column in the data.")

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL])
    d = d.sort_values([INDIVIDUAL_COL, TIME_COL])

    d["time_since_last"] = d.groupby(INDIVIDUAL_COL)[TIME_COL].diff()
    per_indiv_cutoff = d.groupby(INDIVIDUAL_COL)["time_since_last"].transform(lambda s: s.dropna().quantile(gap_q))
    d = d[d["time_since_last"].notna() & (d["time_since_last"] <= per_indiv_cutoff)].copy()

    d["lat_rad"] = np.deg2rad(d[LAT_COL].astype(float))
    d["lon_rad"] = np.deg2rad(d[LON_COL].astype(float))

    g = d.groupby(INDIVIDUAL_COL)
    R = 6371000.0

    d["lat_prev"] = g["lat_rad"].shift()
    d["lon_prev"] = g["lon_rad"].shift()
    dlat = d["lat_rad"] - d["lat_prev"]
    dlon = d["lon_rad"] - d["lon_prev"]

    a = (np.sin(dlat / 2.0) ** 2) + (np.cos(d["lat_prev"]) * np.cos(d["lat_rad"]) * np.sin(dlon / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    d["step_length_m"] = R * c

    valid = d["step_length_m"].dropna()
    valid = valid[valid > 0]
    if len(valid) == 0:
        raise ValueError("No valid step lengths computed")

    return float(step_multiplier) * float(np.nanquantile(valid, q=step_q))


def download_tile_layer(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry,
    layer: str,
    buffer_km: float,
    px_per_side: int = 500,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if layer not in plan.layers_in_order:
        raise ValueError(f"Layer {layer!r} not present in plan.layers_in_order")

    spec = registry.get(layer)
    if spec.source != "gee_tile":
        raise ValueError(f"download_tile_layer only supports gee_tile layers, got {layer!r} source={spec.source!r}")
    if spec.native_scale_m is None:
        raise ValueError(f"gee_tile layer {layer!r} missing spec.native_scale_m")

    tile_dir = _tile_dir(study_dir, plan.rasters_hash, layer)
    index_path = tile_dir / "index.json"

    if skip_existing and index_path.exists():
        with open(index_path, "r") as f:
            existing = json.load(f)
        tiles = existing.get("tiles", [])
        if tiles and all((tile_dir / t["file"]).exists() for t in tiles):
            return {"total": len(tiles), "downloaded": 0, "existing": len(tiles), "failed": 0, "index_path": str(index_path)}

    required_cols = [LON_COL, LAT_COL, SET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    ensure_ee_initialized()

    layer_scale = compute_group_scale(buffer_km, float(spec.native_scale_m))

    regions = []
    for _, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        split = str(row[SET_COL])
        region_mult = float(REGION_MULTIPLIER.get(split, 1.0))
        regions.append(square_region_for_point(lon, lat, region_mult * buffer_km))

    union_region = union_regions(regions)

    ee = ensure_ee_initialized()
    crs = "EPSG:4326"
    grid_size_m = float(layer_scale) * float(px_per_side)
    bounds = union_region.bounds(proj=crs, maxError=10)
    grid = bounds.coveringGrid(proj=crs, scale=grid_size_m)
    filtered = grid.filter(ee.Filter.intersects(".geo", union_region))

    tile_ids = filtered.aggregate_array("system:index").getInfo()
    download_regions = [ee.Feature(filtered.toList(1, i).get(0)).geometry() for i, _ in enumerate(tile_ids)]

    tile_dir.mkdir(parents=True, exist_ok=True)

    img = _tile_image(layer, registry=registry).unmask(GEE_NODATA_SENTINEL, sameFootprint=False)

    tiles_out: list[dict[str, Any]] = []
    stats = {"total": len(download_regions), "downloaded": 0, "existing": 0, "failed": 0}

    for i, region in enumerate(download_regions):
        out_path = tile_dir / f"tile_{i:03d}.tif"

        info = region.getInfo()
        coords = info["coordinates"][0]
        min_lon = min(c[0] for c in coords)
        max_lon = max(c[0] for c in coords)
        min_lat = min(c[1] for c in coords)
        max_lat = max(c[1] for c in coords)

        tile_meta = {
            "file": out_path.name,
            "bounds": {"min_lon": min_lon, "max_lon": max_lon, "min_lat": min_lat, "max_lat": max_lat},
        }

        if skip_existing and out_path.exists():
            stats["existing"] += 1
            tiles_out.append(tile_meta)
            continue

        try:
            params = {"scale": float(layer_scale), "region": region, "format": "GEO_TIFF", "crs": crs}
            url = img.getDownloadURL(params)
            download_tiff(url, out_path)
            stats["downloaded"] += 1
            tiles_out.append(tile_meta)
        except Exception as e:
            stats["failed"] += 1
            tile_meta["error"] = str(e)
            tiles_out.append(tile_meta)

    with open(index_path, "w") as f:
        json.dump({"layer": layer, "scale": float(layer_scale), "tiles": tiles_out}, f, indent=2)

    stats["index_path"] = str(index_path)
    return stats


def download_tiles(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry = REGISTRY,
    buffer_km: float,
    px_per_side: int = 500,
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
            px_per_side=px_per_side,
            skip_existing=skip_existing,
        )
    return out
