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

import json
import hashlib
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
# These values are configured via `raster_params.region_multiplier` in the layers YAML.


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
    try:
        import rasterio
        from rasterio.enums import ColorInterp
    except Exception:
        return

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

    if layer == "era5_land":
        era5 = _nearest_from_window("ECMWF/ERA5/HOURLY", timestamp, bands=spec.bands, start=-1, end=1, unit="hour")
        img = img.unmask(era5, sameFootprint=False)

    if layer == "era5_land_daily":
        fallback = _fully_masked_image(spec.bands)
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
        fallback = fallback.addBands(era5_daily, overwrite=True).select(spec.bands)
        img = img.unmask(fallback, sameFootprint=False)

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


def download_tiled_image(
    img,
    region,
    *,
    scale: float,
    out_path: Path,
    num_bands: int,
    crs: str = "EPSG:4326",
    bounds_latlon: tuple[float, float, float, float] | None = None,
) -> None:
    """Download an ee.Image, tiling if needed, and write GeoTIFF to out_path."""
    ee = ensure_ee_initialized()

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

        log_path = str(Path(tmp_dir) / "gdal.log")
        with rasterio.Env(CPL_LOG=log_path):
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

        _fix_geotiff_colorinterp(out_path, num_bands=int(num_bands))


def _tile_image(layer: str, *, registry: LayerRegistry, year: int | None = None):
    """Build an ee.Image for a gee_tile layer."""
    ee = ensure_ee_initialized()
    spec = registry.get(layer)

    if layer == "dem":
        # COPERNICUS/DEM/GLO30 uses band "DEM"; we export one band.
        return ee.ImageCollection(spec.gee_collection).mosaic().select("DEM").toFloat()

    if layer == "alphaearth_embeddings":
        if year is None:
            raise ValueError("alphaearth_embeddings requires a year for gee_tile export")
        col = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(f"{int(year)}-01-01", f"{int(year) + 1}-01-01")
            .select(spec.bands)
        )
        fallback = _fully_masked_image(spec.bands)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.mosaic(), fallback))
        return img.toFloat()

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


def _chips_meta_path(study_dir: Path, rasters_hash: str) -> Path:
    return _rasters_root(study_dir, rasters_hash) / "chips_meta.json"


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
    split: str


@dataclass(frozen=True)
class TileRequest:
    tile_index: int
    bounds_latlon: tuple[float, float, float, float]


def _download_tile_worker(args: tuple[int, TileRequest, dict[str, Any]]) -> dict[str, Any]:
    _idx, req, ctx = args

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

    if year_int is None:
        out_path = tile_dir / f"tile_{tile_index:03d}.tif"
    else:
        out_path = tile_dir / f"tile_{tile_index:03d}_{year_int}.tif"

    if skip_existing and out_path.exists():
        return {"tile_index": tile_index, "status": "existing", "file": out_path.name}

    try:
        ee = ensure_ee_initialized()
        img = _tile_image(layer, registry=registry, year=year_int).unmask(
            GEE_NODATA_SENTINEL, sameFootprint=False
        )
        min_lon, min_lat, max_lon, max_lat = (float(x) for x in req.bounds_latlon)
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
            bounds_latlon=req.bounds_latlon,
        )
        return {"tile_index": tile_index, "status": "downloaded", "file": out_path.name}
    except Exception as e:
        return {
            "tile_index": tile_index,
            "status": "failed",
            "file": out_path.name,
            "error": str(e),
        }


def _download_chip_worker(args: tuple[int, ChipRequest, dict[str, Any]]) -> dict[str, Any]:
    idx, req, ctx = args

    study_dir = Path(ctx["study_dir"])
    buffer_km = float(ctx["buffer_km"])
    skip_existing = bool(ctx["skip_existing"])
    plan: LayerPlan = ctx["plan"]
    registry: LayerRegistry = ctx["registry"]
    region_multiplier: dict[str, float] = ctx["region_multiplier"]
    min_pixels = int(ctx["min_pixels"])
    max_pixels = int(ctx["max_pixels"])
    group_current: dict[str, bool] = ctx["group_current"]

    results: dict[str, Any] = {"point_id": int(req.point_id), "files": {}}

    # Avoid initializing EE unless we need to download something.
    all_exist = True
    if skip_existing:
        for group_name in plan.chip_groups.keys():
            out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
            if not (bool(group_current.get(group_name, False)) and out_path.exists()):
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

    region_mult = float(region_multiplier.get(req.split, 1.0))
    region = square_region_for_point(req.lon, req.lat, region_mult * buffer_km)

    for group_name, group in plan.chip_groups.items():
        out_path = _chip_path(study_dir, plan.rasters_hash, group_name, req.point_id)
        expected = _expected_band_count_for_chip_group(plan, group_name, registry)
        if skip_existing and bool(group_current.get(group_name, False)) and out_path.exists():
            results["files"][group_name] = {"path": str(out_path), "status": "existing"}
            continue

        target_scale_m = float(group["target_scale_m"])
        scale = compute_group_scale(
            region_mult * buffer_km,
            target_scale_m,
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
    region_multiplier: dict[str, float],
    min_pixels: int,
    max_pixels: int,
    splits: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    study_dir = Path(study_dir)
    region_multiplier = {str(k): float(v) for k, v in dict(region_multiplier).items()}
    min_pixels = int(min_pixels)
    max_pixels = int(max_pixels)
    if min_pixels <= 0:
        raise ValueError(f"min_pixels must be >= 1, got {min_pixels}")
    if max_pixels < min_pixels:
        raise ValueError(f"max_pixels must be >= min_pixels ({min_pixels}), got {max_pixels}")

    if splits is None:
        splits = sorted(df[SET_COL].dropna().unique().tolist())

    required_cols = [LON_COL, LAT_COL, TIME_COL, SET_COL, POINT_ID_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

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
        chips_root = _rasters_root(study_dir, plan.rasters_hash) / "chips"
        has_existing = chips_root.exists() and any(chips_root.glob("*/*.tif"))
        if skip_existing and has_existing:
            raise FileNotFoundError(
                "chips_meta.json not found for an existing chips directory. "
                "Run one_time_scripts/migrate_rasters_hash.py once to create it (and migrate rasters_hash). "
                f"Expected: {meta_path}"
            )
        meta = {"schema_version": "1", "chip_groups": {}}

    existing_groups = meta.get("chip_groups", {})
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
        group_dir.mkdir(parents=True, exist_ok=True)
        for p in group_dir.glob("*.tif"):
            p.unlink()

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
        "region_multiplier": region_multiplier,
        "min_pixels": int(min_pixels),
        "max_pixels": int(max_pixels),
        "group_current": group_current,
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

    meta["chip_groups"] = dict(existing_groups)
    for group_name, expected in expected_meta_by_group.items():
        meta["chip_groups"][group_name] = expected
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

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
    region_multiplier: dict[str, float],
    min_pixels: int,
    max_pixels: int,
    px_per_side: int,
    n_workers: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    if layer not in plan.layers_in_order:
        raise ValueError(f"Layer {layer!r} not present in plan.layers_in_order")

    region_multiplier = {str(k): float(v) for k, v in dict(region_multiplier).items()}
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
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    resolved_years: list[int] = []
    requested_years: list[int] = []

    if (not is_annual) and skip_existing and index_path.exists():
        with open(index_path, "r") as f:
            existing = json.load(f)
        tiles = existing.get("tiles", [])
        # If prior runs recorded errors, do not skip; allow reruns to retry failures.
        errors = existing.get("errors", [])
        if tiles and all((tile_dir / t["file"]).exists() for t in tiles) and not errors:
            return {
                "total": len(tiles),
                "downloaded": 0,
                "existing": len(tiles),
                "failed": 0,
                "index_path": str(index_path),
            }

    def _tile_index_is_complete(idx_path: Path) -> tuple[bool, int]:
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

    if is_annual:
        ts = pd.to_datetime(df[TIME_COL], errors="coerce")
        yrs = ts.dropna().dt.year.astype("int64")
        if len(yrs) == 0:
            raise ValueError(f"trajectories df has no valid {TIME_COL!r} values for annual layer {layer!r}")
        requested_years = sorted(set(int(y) for y in yrs.tolist()))

        ee = ensure_ee_initialized()

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
                ok, n = _tile_index_is_complete(idx_path_y)
                if not ok:
                    all_complete = False
                    break
                years_out[int(y)] = {
                    "total": int(n),
                    "downloaded": 0,
                    "existing": int(n),
                    "failed": 0,
                    "index_path": str(idx_path_y),
                }

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
        ee = ensure_ee_initialized()

    layer_scale = compute_group_scale(
        buffer_km,
        float(spec.native_scale_m),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    crs = "EPSG:4326"
    grid_size_m = float(layer_scale) * float(px_per_side)

    # Avoid building a huge Earth Engine union geometry from per-point buffers:
    # bucket points into coarse spatial bins and build one conservative rectangle per bucket.
    import numpy as np

    lon = df[LON_COL].astype(float)
    lat = df[LAT_COL].astype(float)
    if lon.isna().any() or lat.isna().any():
        raise ValueError("trajectories df contains NaN lon/lat values")

    ref_lat = float(lat.median())
    lon_deg_to_m_ref = 111320.0 * math.cos(math.radians(ref_lat))
    bucket_deg_lon = 360.0 if abs(lon_deg_to_m_ref) < 1e-6 else grid_size_m / lon_deg_to_m_ref
    bucket_deg_lat = grid_size_m / 110540.0

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
        split = str(r["split"])
        region_mult = float(region_multiplier.get(split, 1.0))
        buffer_m = float(buffer_km) * 1000.0 * region_mult

        buffer_deg_lat = buffer_m / 110540.0
        lon_deg_to_m = 111320.0 * math.cos(math.radians(float(r["max_abs_lat"])))
        buffer_deg_lon = 180.0 if abs(lon_deg_to_m) < 1e-6 else buffer_m / lon_deg_to_m

        min_lon_exp = float(r["min_lon"]) - buffer_deg_lon
        max_lon_exp = float(r["max_lon"]) + buffer_deg_lon
        min_lat_exp = float(r["min_lat"]) - buffer_deg_lat
        max_lat_exp = float(r["max_lat"]) + buffer_deg_lat

        min_lat_exp = max(min_lat_exp, -89.9999)
        max_lat_exp = min(max_lat_exp, 89.9999)

        rects: list[tuple[float, float, float, float]]
        if (max_lon_exp - min_lon_exp) >= 360.0:
            rects = [(-180.0, min_lat_exp, 180.0, max_lat_exp)]
        elif max_lon_exp > 180.0:
            rects = [
                (min_lon_exp, min_lat_exp, 180.0, max_lat_exp),
                (-180.0, min_lat_exp, max_lon_exp - 360.0, max_lat_exp),
            ]
        elif min_lon_exp < -180.0:
            rects = [
                (min_lon_exp + 360.0, min_lat_exp, 180.0, max_lat_exp),
                (-180.0, min_lat_exp, max_lon_exp, max_lat_exp),
            ]
        else:
            rects = [(min_lon_exp, min_lat_exp, max_lon_exp, max_lat_exp)]

        for a, b, c, d in rects:
            a = max(float(a), -180.0)
            c = min(float(c), 180.0)
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
    filtered = grid.filter(ee.Filter.intersects(".geo", union_region))

    n_tiles_reported = int(filtered.size().getInfo())

    tile_dir.mkdir(parents=True, exist_ok=True)

    tile_regions: list[Any] = []
    tile_bounds_latlon: list[tuple[float, float, float, float]] = []
    tile_indices: list[int] = []
    geometry_errors: list[dict[str, Any]] = []
    n_tiles_effective = int(n_tiles_reported)

    for i in range(int(n_tiles_reported)):
        try:
            region = ee.Feature(filtered.toList(1, i).get(0)).geometry()
            info = region.getInfo()
        except Exception as e:
            msg = str(e)
            is_list_index_error = ("List index must be between" in msg) or ("List.get: List is empty" in msg)

            # Earth Engine can occasionally report `size()` larger than the number
            # of items returned by `toList(1, offset)`. If that happens near the tail,
            # stop early and treat the effective tile count as the first missing index.
            if is_list_index_error and i >= max(int(n_tiles_reported) - 5, 0):
                n_tiles_effective = int(i)
                break

            geometry_errors.append(
                {
                    "tile_index": int(i),
                    "error": f"tile_geometry_failed: {type(e).__name__}: {e}",
                }
            )
            continue
        coords = info["coordinates"][0]
        min_lon = min(c[0] for c in coords)
        max_lon = max(c[0] for c in coords)
        min_lat = min(c[1] for c in coords)
        max_lat = max(c[1] for c in coords)

        tile_regions.append(region)
        tile_bounds_latlon.append((float(min_lon), float(min_lat), float(max_lon), float(max_lat)))
        tile_indices.append(int(i))

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

        from tqdm import tqdm

        pool = None
        try:
            for year in resolved_years:
                idx_path_y = tile_dir / f"index_{int(year)}.json"

                if skip_existing and idx_path_y.exists():
                    with open(idx_path_y, "r") as f:
                        existing = json.load(f)
                    tiles = existing.get("tiles", [])
                    errors = existing.get("errors", [])
                    if tiles and not errors and not any(("error" in t) for t in tiles) and all(
                        (tile_dir / t["file"]).exists() for t in tiles
                    ):
                        years_out[int(year)] = {
                            "total": len(tiles),
                            "downloaded": 0,
                            "existing": len(tiles),
                            "failed": 0,
                            "index_path": str(idx_path_y),
                        }
                        continue

                tiles_out: list[dict[str, Any]] = []
                errors_out: list[dict[str, Any]] = [
                    {
                        "tile_index": int(e["tile_index"]),
                        "file": f"tile_{int(e['tile_index']):03d}_{int(year)}.tif",
                        "error": str(e["error"]),
                    }
                    for e in geometry_errors
                ]
                stats = {
                    "total": int(n_tiles_reported),
                    "downloaded": 0,
                    "existing": 0,
                    "failed": int(len(errors_out)),
                }

                if reqs:
                    if pool is None:
                        pool = mp.Pool(int(n_workers))
                    ctx_y = {**ctx_base, "year": int(year)}
                    worker_args = [(i, r, ctx_y) for i, r in enumerate(reqs)]
                    results = list(
                        tqdm(
                            pool.imap_unordered(_download_tile_worker, worker_args),
                            total=len(reqs),
                            desc=f"Downloading tiles ({layer} {int(year)})",
                            unit="tile",
                        )
                    )
                    results_by_index = {int(r["tile_index"]): r for r in results}
                else:
                    results_by_index = {}

                for tile_index, bounds_latlon in zip(tile_indices, tile_bounds_latlon):
                    out_path = tile_dir / f"tile_{int(tile_index):03d}_{int(year)}.tif"
                    tile_meta = {
                        "file": out_path.name,
                        "tile_index": int(tile_index),
                        "bounds": {
                            "min_lon": float(bounds_latlon[0]),
                            "min_lat": float(bounds_latlon[1]),
                            "max_lon": float(bounds_latlon[2]),
                            "max_lat": float(bounds_latlon[3]),
                        },
                    }

                    info = results_by_index.get(int(tile_index))
                    if info is None:
                        stats["failed"] += 1
                        tile_meta["error"] = "tile_missing_result"
                    else:
                        status = str(info.get("status", "failed"))
                        if status == "downloaded":
                            stats["downloaded"] += 1
                        elif status == "existing":
                            stats["existing"] += 1
                        else:
                            stats["failed"] += 1
                            tile_meta["error"] = str(info.get("error", "unknown"))

                    tiles_out.append(tile_meta)

                with open(idx_path_y, "w") as f:
                    stats["total"] = int(n_tiles_effective)

                    payload: dict[str, Any] = {
                        "layer": layer,
                        "temporal": "annual",
                        "year": int(year),
                        "scale": float(layer_scale),
                        "num_bands": int(len(spec.bands)),
                        "tiles": tiles_out,
                    }
                    if errors_out:
                        payload["errors"] = errors_out
                        payload["n_tiles_reported"] = int(n_tiles_reported)
                    if int(n_tiles_effective) != int(n_tiles_reported):
                        payload["n_tiles_reported"] = int(n_tiles_reported)
                        payload["n_tiles_effective"] = int(n_tiles_effective)
                    json.dump(payload, f, indent=2)

                stats["index_path"] = str(idx_path_y)
                years_out[int(year)] = stats
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Compatibility index.json used for bounds-only lookups.
        if resolved_years and (not index_path.exists()):
            src = tile_dir / f"index_{int(resolved_years[0])}.json"
            if src.exists():
                with open(src, "r") as f:
                    payload = json.load(f)
                with open(index_path, "w") as f:
                    json.dump(payload, f, indent=2)

        return {
            "temporal": "annual",
            "years": years_out,
            "index_path": str(index_path),
        }

    tiles_out = []
    errors_out: list[dict[str, Any]] = [
        {"tile_index": int(e["tile_index"]), "file": f"tile_{int(e['tile_index']):03d}.tif", "error": str(e["error"])}
        for e in geometry_errors
    ]
    stats = {"total": int(n_tiles_reported), "downloaded": 0, "existing": 0, "failed": int(len(errors_out))}

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

    from tqdm import tqdm

    if reqs:
        worker_args = [(i, r, ctx) for i, r in enumerate(reqs)]
        with mp.Pool(int(n_workers)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_download_tile_worker, worker_args),
                    total=len(reqs),
                    desc=f"Downloading tiles ({layer})",
                    unit="tile",
                )
            )
        results_by_index = {int(r["tile_index"]): r for r in results}
    else:
        results_by_index = {}

    for tile_index, bounds_latlon in zip(tile_indices, tile_bounds_latlon):
        out_path = tile_dir / f"tile_{int(tile_index):03d}.tif"

        tile_meta = {
            "file": out_path.name,
            "tile_index": int(tile_index),
            "bounds": {
                "min_lon": float(bounds_latlon[0]),
                "min_lat": float(bounds_latlon[1]),
                "max_lon": float(bounds_latlon[2]),
                "max_lat": float(bounds_latlon[3]),
            },
        }

        info = results_by_index.get(int(tile_index))
        if info is None:
            stats["failed"] += 1
            tile_meta["error"] = "tile_missing_result"
        else:
            status = str(info.get("status", "failed"))
            if status == "downloaded":
                stats["downloaded"] += 1
            elif status == "existing":
                stats["existing"] += 1
            else:
                stats["failed"] += 1
                tile_meta["error"] = str(info.get("error", "unknown"))

        tiles_out.append(tile_meta)

    with open(index_path, "w") as f:
        stats["total"] = int(n_tiles_effective)

        payload = {
            "layer": layer,
            "scale": float(layer_scale),
            "num_bands": int(len(spec.bands)),
            "tiles": tiles_out,
        }
        if errors_out:
            payload["errors"] = errors_out
            payload["n_tiles_reported"] = int(n_tiles_reported)
        if int(n_tiles_effective) != int(n_tiles_reported):
            payload["n_tiles_reported"] = int(n_tiles_reported)
            payload["n_tiles_effective"] = int(n_tiles_effective)
        json.dump(payload, f, indent=2)

    stats["index_path"] = str(index_path)
    return stats


def download_tiles(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry = REGISTRY,
    buffer_km: float,
    region_multiplier: dict[str, float],
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
            region_multiplier=region_multiplier,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            px_per_side=px_per_side,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )
    return out
