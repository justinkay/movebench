from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from movebench.layers.constants import WGS84_CRS


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
        bounds = (
            float(ds.bounds.left),
            float(ds.bounds.bottom),
            float(ds.bounds.right),
            float(ds.bounds.top),
        )
        band_ids = (
            list(ds.descriptions)
            if ds.descriptions and any(ds.descriptions)
            else [f"band_{i + 1}" for i in range(ds.count)]
        )
        return Raster(
            array=arr,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            bounds=bounds,
            band_ids=band_ids,
        )


def read_geotiff_window(
    path: str | Path,
    bounds_latlon: tuple[float, float, float, float],
    *,
    max_window_hw: tuple[int, int] | None = None,
    resampling: Resampling = Resampling.nearest,
) -> Raster:
    """Read a window from a GeoTIFF using an EPSG:4326 bounds box.

    If the dataset is not in EPSG:4326, the bounds are transformed into the
    dataset CRS before windowing.
    """
    min_lon, min_lat, max_lon, max_lat = bounds_latlon

    with rasterio.open(Path(path)) as ds:
        if ds.crs is not None and str(ds.crs) != WGS84_CRS:
            left, bottom, right, top = transform_bounds(
                WGS84_CRS, ds.crs, min_lon, min_lat, max_lon, max_lat
            )
            bounds_in_ds = (left, bottom, right, top)
        else:
            bounds_in_ds = (min_lon, min_lat, max_lon, max_lat)

        # Rasterio can produce empty windows for tiny bounds. Convert to an
        # integer window and force at least 2 pixels in each dimension so
        # downstream derived layers such as slope/aspect can compute gradients.
        try:
            win = from_bounds(*bounds_in_ds, transform=ds.transform)
        except Exception as exc:
            raise ValueError(
                "Failed to calculate raster window for "
                f"path={Path(path)!s} bounds_latlon={bounds_latlon!r} "
                f"bounds_in_ds={bounds_in_ds!r} crs={ds.crs!r} "
                f"transform={ds.transform!r}: {exc}"
            ) from exc
        col0 = int(math.floor(win.col_off))
        row0 = int(math.floor(win.row_off))
        col1 = int(math.ceil(win.col_off + win.width))
        row1 = int(math.ceil(win.row_off + win.height))
        if col1 <= col0 + 1:
            col1 = col0 + 2
        if row1 <= row0 + 1:
            row1 = row0 + 2
        window = Window(col0, row0, col1 - col0, row1 - row0)
        b = window_bounds(window, ds.transform)
        win_bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

        out_height = int(window.height)
        out_width = int(window.width)
        if max_window_hw is not None:
            max_height = max(2, int(max_window_hw[0]))
            max_width = max(2, int(max_window_hw[1]))
            scale = min(
                float(max_height) / float(out_height),
                float(max_width) / float(out_width),
                1.0,
            )
            if scale < 1.0:
                out_height = min(
                    max_height, max(2, int(math.ceil(float(out_height) * scale)))
                )
                out_width = min(
                    max_width, max(2, int(math.ceil(float(out_width) * scale)))
                )

        fill_val = ds.nodata if ds.nodata is not None else np.nan
        arr = ds.read(
            window=window,
            out_shape=(ds.count, out_height, out_width),
            resampling=resampling,
            boundless=True,
            fill_value=fill_val,
        )
        band_ids = (
            list(ds.descriptions)
            if ds.descriptions and any(ds.descriptions)
            else [f"band_{i + 1}" for i in range(ds.count)]
        )

        return Raster(
            array=arr,
            transform=transform_from_bounds(*win_bounds, out_width, out_height),
            crs=ds.crs,
            nodata=fill_val,
            bounds=win_bounds,
            band_ids=band_ids,
        )


def merge_hdf_window(
    subdataset_paths: Iterable[str], bounds_latlon: tuple[float, float, float, float]
) -> Raster:
    """Load and merge multiple HDF subdatasets for an EPSG:4326 bounds box."""
    paths = list(subdataset_paths)
    if not paths:
        raise ValueError("merge_hdf_window requires at least one subdataset path")

    datasets = [rasterio.open(p) for p in paths]
    try:
        crs = datasets[0].crs
        if crs is None:
            raise ValueError(
                "HDF subdataset has no CRS; cannot window by lat/lon bounds"
            )

        nodata_val = datasets[0].nodata
        if nodata_val is None:
            nodata_val = 255

        min_lon, min_lat, max_lon, max_lat = bounds_latlon
        left, bottom, right, top = transform_bounds(
            WGS84_CRS, crs, min_lon, min_lat, max_lon, max_lat
        )
        # Snap to >=2-pixel integer bounds in the dataset CRS so downstream
        # derived layers have enough spatial support for gradient calculations.
        win = from_bounds(
            float(left),
            float(bottom),
            float(right),
            float(top),
            transform=datasets[0].transform,
        )
        col0 = int(math.floor(win.col_off))
        row0 = int(math.floor(win.row_off))
        col1 = int(math.ceil(win.col_off + win.width))
        row1 = int(math.ceil(win.row_off + win.height))
        if col1 <= col0 + 1:
            col1 = col0 + 2
        if row1 <= row0 + 1:
            row1 = row0 + 2
        win_i = Window(col0, row0, col1 - col0, row1 - row0)
        b = window_bounds(win_i, datasets[0].transform)
        win_bounds = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

        mosaic, out_transform = merge(
            datasets,
            bounds=win_bounds,
            nodata=nodata_val,
            resampling=Resampling.nearest,
        )

        descriptions = datasets[0].descriptions
        band_ids = (
            list(descriptions)
            if descriptions and any(descriptions)
            else [f"band_{i + 1}" for i in range(mosaic.shape[0])]
        )

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
            bounds=win_bounds,
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
    dst_width: int | None = None,
    dst_height: int | None = None,
    dst_transform: Any | None = None,
) -> tuple[list[np.ndarray], Any, str]:
    """Reproject all layers to a metric CRS, aligned to the first layer's bounds.

    This mirrors the legacy behavior in movebench, but works with `Raster` objects.

    Args:
        layers: Input rasters to align. All outputs share a common grid.
        dst_crs: Destination CRS.
        resampling: Rasterio resampling strategy used for reprojection.
        dst_width: Optional fixed output width in pixels.
        dst_height: Optional fixed output height in pixels.
        dst_transform: Optional explicit destination transform (in `dst_crs`
            coordinates). If provided, `calculate_default_transform` is skipped
            and both `dst_width` and `dst_height` are required.
    """
    if not layers:
        raise ValueError("reproject_to_meters requires at least one Raster")

    ref = layers[0]
    if ref.crs is None:
        raise ValueError("Reference raster has no CRS; cannot reproject")

    if (dst_width is None) != (dst_height is None):
        raise ValueError("Pass both dst_width and dst_height, or neither")
    if dst_transform is not None and dst_width is None:
        raise ValueError("dst_transform requires dst_width and dst_height")
    if dst_width is not None:
        dst_width = int(dst_width)
        dst_height = int(dst_height)
        if dst_width <= 0 or dst_height <= 0:
            raise ValueError(
                f"dst_width/dst_height must be positive, got {(dst_width, dst_height)}"
            )

    _, H_ref, W_ref = ref.array.shape

    if dst_transform is None:
        if dst_width is None:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                ref.crs,
                dst_crs,
                W_ref,
                H_ref,
                *ref.bounds,
            )
        else:
            dst_transform, out_w, out_h = calculate_default_transform(
                ref.crs,
                dst_crs,
                W_ref,
                H_ref,
                *ref.bounds,
                dst_width=int(dst_width),
                dst_height=int(dst_height),
            )
            if int(out_w) != int(dst_width) or int(out_h) != int(dst_height):
                raise RuntimeError(
                    "rasterio.warp.calculate_default_transform did not respect requested output size "
                    f"(dst_width={dst_width}, dst_height={dst_height}); got (width={out_w}, height={out_h}). "
                    "Check rasterio version compatibility."
                )
            dst_width, dst_height = out_w, out_h

    aligned: list[np.ndarray] = []
    for layer in layers:
        if layer.crs is None:
            raise ValueError("Raster has no CRS; cannot reproject")

        C = int(layer.array.shape[0])
        src_nodata = layer.nodata

        # Preserve categorical nodata values when possible; otherwise use NaN.
        if src_nodata is not None and not (
            isinstance(src_nodata, float) and np.isnan(src_nodata)
        ):
            dst_nodata = src_nodata
        else:
            dst_nodata = np.nan

        dst = np.full(
            (C, int(dst_height), int(dst_width)), dst_nodata, dtype=np.float32
        )

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
