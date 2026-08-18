from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.transform import (
    from_bounds as transform_from_bounds,
    from_origin,
)
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rasterio.windows import bounds as window_bounds

from movebench.geometry import validate_local_bounds_m
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


def _nodata_key(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return "nan"
    return value


def aeqd_work_grid(
    rasters: list[Raster],
    dst_crs: Any,
    support_bounds_m: tuple[float, float, float, float],
    *,
    halo_pixels: int = 1,
) -> tuple[Any, int, int, float, tuple[float, float, float, float]]:
    """Return an origin-aligned square-pixel AEQD grid for fine derivation."""
    if not rasters:
        raise ValueError("aeqd_work_grid requires at least one Raster")

    resolutions = []
    for raster in rasters:
        if raster.crs is None:
            raise ValueError("Raster has no CRS; cannot calculate AEQD resolution")
        _, height, width = raster.array.shape
        transform, _width, _height = calculate_default_transform(
            raster.crs,
            dst_crs,
            width,
            height,
            *raster.bounds,
        )
        resolutions.append(max(abs(float(transform.a)), abs(float(transform.e))))

    resolution = max(resolutions)
    left, bottom, right, top = (float(value) for value in support_bounds_m)
    halo = max(0, int(halo_pixels)) * resolution

    def lower_edge(value: float) -> float:
        return (math.floor(value / resolution + 0.5) - 0.5) * resolution

    def upper_edge(value: float) -> float:
        return (math.ceil(value / resolution + 0.5) - 0.5) * resolution

    left = lower_edge(left - halo)
    bottom = lower_edge(bottom - halo)
    right = upper_edge(right + halo)
    top = upper_edge(top + halo)
    width = max(1, int(round((right - left) / resolution)))
    height = max(1, int(round((top - bottom) / resolution)))
    transform = from_origin(left, top, resolution, resolution)
    bounds = (left, bottom, right, top)
    validate_local_bounds_m(bounds)
    return transform, width, height, resolution, bounds


def reproject_multipart_to_meters(
    layers: list[list[Raster]],
    dst_crs: Any,
    *,
    resampling: list[Resampling] | Resampling = Resampling.nearest,
    dst_width: int | None = None,
    dst_height: int | None = None,
    dst_transform: Any | None = None,
) -> tuple[list[np.ndarray], Any, str]:
    """Group compatible rasters, reproject, and composite multipart inputs."""
    if not layers or any(not parts for parts in layers):
        raise ValueError("reproject_multipart_to_meters requires non-empty layers")

    if isinstance(resampling, Resampling):
        methods = [resampling] * len(layers)
    else:
        methods = list(resampling)
        if len(methods) != len(layers):
            raise ValueError("resampling must provide one method per layer")

    reference = layers[0][0]
    if reference.crs is None:
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

    _, ref_height, ref_width = reference.array.shape
    if dst_transform is None:
        dst_transform, dst_width, dst_height = calculate_default_transform(
            reference.crs,
            dst_crs,
            ref_width,
            ref_height,
            *reference.bounds,
            **(
                {
                    "dst_width": int(dst_width),
                    "dst_height": int(dst_height),
                }
                if dst_width is not None
                else {}
            ),
        )

    dst_width = int(dst_width)
    dst_height = int(dst_height)
    outputs = [
        np.full(
            (int(parts[0].array.shape[0]), dst_height, dst_width),
            np.nan,
            dtype=np.float32,
        )
        for parts in layers
    ]
    groups: dict[tuple[Any, ...], list[tuple[int, Raster]]] = {}
    for layer_index, (parts, method) in enumerate(zip(layers, methods)):
        band_count = int(parts[0].array.shape[0])
        for raster in parts:
            if raster.crs is None:
                raise ValueError("Raster has no CRS; cannot reproject")
            if int(raster.array.shape[0]) != band_count:
                raise ValueError("Multipart raster band counts must match")
            key = (
                int(method),
                str(raster.crs),
                tuple(raster.transform),
                tuple(raster.array.shape[1:]),
                _nodata_key(raster.nodata),
            )
            groups.setdefault(key, []).append((layer_index, raster))

    for key, entries in groups.items():
        source = np.concatenate([raster.array for _, raster in entries], axis=0)
        warped = np.full(
            (int(source.shape[0]), dst_height, dst_width),
            np.nan,
            dtype=np.float32,
        )
        raster = entries[0][1]
        reproject(
            source=source,
            destination=warped,
            src_transform=raster.transform,
            src_crs=raster.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling(key[0]),
            src_nodata=raster.nodata if raster.nodata is not None else np.nan,
            dst_nodata=np.nan,
        )

        offset = 0
        for layer_index, raster in entries:
            band_count = int(raster.array.shape[0])
            candidate = warped[offset : offset + band_count]
            current = outputs[layer_index]
            np.copyto(
                current,
                candidate,
                where=~np.isfinite(current) & np.isfinite(candidate),
            )
            offset += band_count

    return outputs, dst_transform, str(dst_crs)


def reproject_to_meters(
    layers: list[Raster],
    dst_crs: Any,
    *,
    resampling: Resampling = Resampling.nearest,
    dst_width: int | None = None,
    dst_height: int | None = None,
    dst_transform: Any | None = None,
) -> tuple[list[np.ndarray], Any, str]:
    """Reproject aligned rasters, grouping compatible inputs into multiband warps."""
    return reproject_multipart_to_meters(
        [[layer] for layer in layers],
        dst_crs,
        resampling=resampling,
        dst_width=dst_width,
        dst_height=dst_height,
        dst_transform=dst_transform,
    )
