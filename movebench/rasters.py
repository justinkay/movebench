from __future__ import annotations

import rasterio
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import from_bounds


@dataclass(frozen=True)
class Raster:
    """A raster with enough spatial metadata to sample in its native CRS.

    Notes:
    - `array` is expected to be a NumPy array with shape (C, H, W).
    - `bounds` are in the raster's CRS (same space as `transform` / `crs`).
    """

    array: Any
    transform: Any
    crs: Any
    nodata: Any
    bounds: tuple[float, float, float, float]
    band_ids: list[str]


def read_geotiff(path: str | Path) -> Raster:
    with rasterio.open(Path(path)) as ds:
        arr = ds.read()
        bounds = (float(ds.bounds.left), float(ds.bounds.bottom), float(ds.bounds.right), float(ds.bounds.top))
        band_ids = list(ds.descriptions) if ds.descriptions and any(ds.descriptions) else [f"band_{i+1}" for i in range(ds.count)]
        return Raster(
            array=arr,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            bounds=bounds,
            band_ids=band_ids,
        )


def read_geotiff_window(path: str | Path, bounds_latlon: tuple[float, float, float, float]) -> Raster:
    """Read a window from a GeoTIFF using an EPSG:4326 bounds box.

    If the dataset is not in EPSG:4326, the bounds are transformed into the
    dataset CRS before windowing.
    """
    min_lon, min_lat, max_lon, max_lat = bounds_latlon

    with rasterio.open(Path(path)) as ds:
        if ds.crs is not None and str(ds.crs) != "EPSG:4326":
            left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, min_lon, min_lat, max_lon, max_lat)
            bounds_in_ds = (left, bottom, right, top)
        else:
            bounds_in_ds = (min_lon, min_lat, max_lon, max_lat)

        window = from_bounds(*bounds_in_ds, transform=ds.transform)
        fill_val = ds.nodata if ds.nodata is not None else np.nan
        arr = ds.read(window=window, boundless=True, fill_value=fill_val)
        band_ids = list(ds.descriptions) if ds.descriptions and any(ds.descriptions) else [f"band_{i+1}" for i in range(ds.count)]

        return Raster(
            array=arr,
            transform=ds.window_transform(window),
            crs=ds.crs,
            nodata=fill_val,
            bounds=tuple(map(float, bounds_in_ds)),
            band_ids=band_ids,
        )


def merge_hdf_window(subdataset_paths: Iterable[str], bounds_latlon: tuple[float, float, float, float]) -> Raster:
    """Load and merge multiple HDF subdatasets for an EPSG:4326 bounds box."""
    paths = list(subdataset_paths)
    if not paths:
        raise ValueError("merge_hdf_window requires at least one subdataset path")

    datasets = [rasterio.open(p) for p in paths]
    try:
        crs = datasets[0].crs
        if crs is None:
            raise ValueError("HDF subdataset has no CRS; cannot window by lat/lon bounds")

        nodata_val = datasets[0].nodata
        if nodata_val is None:
            nodata_val = 255

        min_lon, min_lat, max_lon, max_lat = bounds_latlon
        left, bottom, right, top = transform_bounds("EPSG:4326", crs, min_lon, min_lat, max_lon, max_lat)

        mosaic, out_transform = merge(
            datasets,
            bounds=(left, bottom, right, top),
            nodata=nodata_val,
            resampling=Resampling.nearest,
        )

        descriptions = datasets[0].descriptions
        band_ids = list(descriptions) if descriptions and any(descriptions) else [f"band_{i+1}" for i in range(mosaic.shape[0])]

        # Ensure nodata uses a float NaN when appropriate for downstream math.
        if isinstance(nodata_val, (int, float)) and np.isfinite(nodata_val):
            nodata_out = nodata_val
        else:
            nodata_out = np.nan

        return Raster(
            array=mosaic,
            transform=out_transform,
            crs=crs,
            nodata=nodata_out,
            bounds=(float(left), float(bottom), float(right), float(top)),
            band_ids=band_ids,
        )
    finally:
        for ds in datasets:
            ds.close()


def reproject_to_meters(
    layers: list[Raster],
    dst_crs: str = "EPSG:3857",
    *,
    resampling: Resampling = Resampling.nearest,
) -> tuple[list[np.ndarray], Any, str]:
    """Reproject all layers to a metric CRS, aligned to the first layer's bounds.

    This mirrors the legacy behavior in movebench, but works with `Raster` objects.

    Args:
        layers: Input rasters to align. All outputs share a common grid.
        dst_crs: Destination CRS.
        resampling: Rasterio resampling strategy used for reprojection.
    """
    if not layers:
        raise ValueError("reproject_to_meters requires at least one Raster")

    ref = layers[0]
    if ref.crs is None:
        raise ValueError("Reference raster has no CRS; cannot reproject")

    _, H_ref, W_ref = ref.array.shape

    dst_transform, dst_width, dst_height = calculate_default_transform(
        ref.crs,
        dst_crs,
        W_ref,
        H_ref,
        *ref.bounds,
    )

    aligned: list[np.ndarray] = []
    for layer in layers:
        if layer.crs is None:
            raise ValueError("Raster has no CRS; cannot reproject")

        C = int(layer.array.shape[0])
        src_nodata = layer.nodata

        # Preserve categorical nodata values when possible; otherwise use NaN.
        if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            dst_nodata = src_nodata
        else:
            dst_nodata = np.nan

        dst = np.full((C, int(dst_height), int(dst_width)), dst_nodata, dtype=np.float32)

        reproject(
            source=layer.array,
            destination=dst,
            src_transform=layer.transform,
            src_crs=layer.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            src_nodata=src_nodata if src_nodata is not None else np.nan,
            dst_nodata=dst_nodata,
        )
        aligned.append(dst)

    return aligned, dst_transform, str(dst_crs)
