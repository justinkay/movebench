from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class Raster:
    """In-memory raster with enough metadata for sampling and reprojection."""

    array: Any  # expected shape (C, H, W); typically a numpy array
    transform: Any
    crs: Any
    nodata: float | int | None
    bounds: tuple[float, float, float, float]
    band_ids: list[str]


def _band_ids_from_descriptions(descriptions: Sequence[str | None] | None, n: int) -> list[str]:
    if descriptions is None:
        return [f"band_{i+1}" for i in range(n)]
    out: list[str] = []
    for i in range(n):
        d = descriptions[i] if i < len(descriptions) else None
        out.append(d if d else f"band_{i+1}")
    return out


def read_geotiff(path: str | Path) -> Raster:
    import rasterio

    p = Path(path)
    with rasterio.open(p) as ds:
        arr = ds.read()
        band_ids = _band_ids_from_descriptions(ds.descriptions, arr.shape[0])
        b = ds.bounds
        return Raster(
            array=arr,
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            bounds=(float(b.left), float(b.bottom), float(b.right), float(b.top)),
            band_ids=band_ids,
        )


def read_geotiff_window(
    path: str | Path, bounds_latlon: tuple[float, float, float, float]
) -> Raster:
    """Read a window defined by EPSG:4326 bounds."""

    import numpy as np
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    p = Path(path)
    with rasterio.open(p) as ds:
        if ds.crs is not None and ds.crs.to_string() != "EPSG:4326":
            left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *bounds_latlon)
        else:
            left, bottom, right, top = bounds_latlon

        window = from_bounds(left, bottom, right, top, ds.transform)
        fill_val = ds.nodata if ds.nodata is not None else np.nan
        arr = ds.read(window=window, boundless=True, fill_value=fill_val)
        band_ids = _band_ids_from_descriptions(ds.descriptions, arr.shape[0])
        return Raster(
            array=arr,
            transform=ds.window_transform(window),
            crs=ds.crs,
            nodata=fill_val,
            bounds=(float(left), float(bottom), float(right), float(top)),
            band_ids=band_ids,
        )


def merge_hdf_window(
    paths: Sequence[str | Path], bounds_latlon: tuple[float, float, float, float]
) -> Raster:
    """Merge multiple HDF subdatasets over a window defined by EPSG:4326 bounds."""

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge
    from rasterio.warp import transform_bounds

    datasets = [rasterio.open(str(p)) for p in paths]
    try:
        crs = datasets[0].crs
        nodata_val = datasets[0].nodata if datasets[0].nodata is not None else 255

        left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bounds_latlon)
        mosaic, out_transform = merge(
            datasets,
            bounds=(left, bottom, right, top),
            nodata=nodata_val,
            resampling=Resampling.nearest,
        )
        band_ids = _band_ids_from_descriptions(datasets[0].descriptions, mosaic.shape[0])
        return Raster(
            array=mosaic,
            transform=out_transform,
            crs=crs,
            nodata=nodata_val,
            bounds=(float(left), float(bottom), float(right), float(top)),
            band_ids=band_ids,
        )
    finally:
        for ds in datasets:
            ds.close()

