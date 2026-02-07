from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from movebench2 import paths
from movebench2.layers.registry import LayerSpec
from movebench2.rasters import Raster, merge_hdf_window, read_geotiff, read_geotiff_window


def _glob_one(pattern: str) -> str:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    # Deterministic choice when multiple versions exist.
    return matches[-1]


def _format_path_pattern(pattern: str, **kwargs: Any) -> str:
    ctx = {**paths.paths_config(), **kwargs}
    try:
        return pattern.format(**ctx)
    except KeyError as e:
        raise KeyError(f"Missing key for path_pattern formatting: {e}. pattern={pattern!r}") from e


def _clamp_year(year: int, year_bounds: tuple[int, int] | None) -> int:
    if year_bounds is None:
        return year
    lo, hi = year_bounds
    return max(min(year, hi), lo)


@dataclass(frozen=True)
class ChipGroupLoader:
    """Loads a pre-downloaded chip GeoTIFF for a chip group."""

    study_dir: Path
    chip_group: str

    def path(self, point_id: int) -> Path:
        return self.study_dir / "rasters" / "chips" / self.chip_group / f"{point_id}.tif"

    def load(self, point_id: int) -> Raster:
        return read_geotiff(self.path(point_id))


@dataclass(frozen=True)
class TileLayerLoader:
    """Loads a study-wide mosaic GeoTIFF for a layer, windowed by bounds."""

    study_dir: Path
    layer: str

    def path(self) -> Path:
        return self.study_dir / "rasters" / "tiles" / f"{self.layer}.tif"

    def load(self, bounds_latlon: tuple[float, float, float, float]) -> Raster:
        return read_geotiff_window(self.path(), bounds_latlon)


@dataclass(frozen=True)
class LocalTiffLayerLoader:
    """Loads a window from a local GeoTIFF referenced by a LayerSpec path_pattern."""

    spec: LayerSpec

    def resolve_path(self, *, timestamp=None, band_num: int | None = None) -> str:
        pattern = self.spec.path_pattern
        if not pattern:
            raise ValueError(f"Layer {self.spec.name!r} missing path_pattern")

        fmt: dict[str, Any] = {}
        if "{year}" in pattern:
            if timestamp is None:
                raise ValueError(f"Layer {self.spec.name!r} requires timestamp to format {{year}}")
            fmt["year"] = _clamp_year(int(timestamp.year), self.spec.year_bounds)
        if "{band_num}" in pattern:
            if band_num is None:
                raise ValueError(f"Layer {self.spec.name!r} requires band_num to format {{band_num}}")
            fmt["band_num"] = int(band_num)

        return _format_path_pattern(pattern, **fmt)

    def load(
        self,
        bounds_latlon: tuple[float, float, float, float],
        *,
        timestamp=None,
        band_num: int | None = None,
    ) -> Raster:
        path = self.resolve_path(timestamp=timestamp, band_num=band_num)
        return read_geotiff_window(path, bounds_latlon)


@dataclass(frozen=True)
class LocalHdfLayerLoader:
    """Loads and merges MODIS HDF tiles for a region."""

    spec: LayerSpec

    def load(
        self,
        bounds_latlon: tuple[float, float, float, float],
        *,
        timestamp,
        hvs: list[tuple[int, int]],
    ) -> Raster:
        if not self.spec.path_pattern or not self.spec.hdf_path_suffix:
            raise ValueError(f"Layer {self.spec.name!r} missing path_pattern/hdf_path_suffix")

        year = _clamp_year(int(timestamp.year), self.spec.year_bounds)

        sd_paths: list[str] = []
        for (h, v) in hvs:
            pattern = _format_path_pattern(self.spec.path_pattern, year=year, h=int(h), v=int(v))
            hdf_path = _glob_one(pattern)
            sd_paths.append(f"HDF4_EOS:EOS_GRID:{hdf_path}{self.spec.hdf_path_suffix}")

        return merge_hdf_window(sd_paths, bounds_latlon)


# -----------------------------------------------------------------------------
# Derived layers (pure functions: parents -> Raster)
# -----------------------------------------------------------------------------


def _gradient_meters(elev, transform) -> tuple[Any, Any]:
    import numpy as np

    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)

    nrows = elev.shape[0]
    center_row = nrows // 2
    center_lat = transform.f + center_row * transform.e
    lat_rad = np.radians(center_lat)

    dy_m = dy_deg * 110540.0
    dx_m = dx_deg * 111320.0 * np.cos(lat_rad)

    gy, gx = np.gradient(elev, dy_m, dx_m)
    return gy, gx


def compute_slope(dem: Raster) -> Raster:
    import numpy as np

    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem.transform)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return Raster(
        array=slope_deg[np.newaxis],
        transform=dem.transform,
        crs=dem.crs,
        nodata=np.nan,
        bounds=dem.bounds,
        band_ids=["slope"],
    )


def compute_aspect(dem: Raster) -> Raster:
    import numpy as np

    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem.transform)
    aspect_rad = np.arctan2(-gx, gy)
    sin_aspect = np.sin(aspect_rad).astype(np.float32)
    cos_aspect = np.cos(aspect_rad).astype(np.float32)
    return Raster(
        array=np.stack([sin_aspect, cos_aspect], axis=0),
        transform=dem.transform,
        crs=dem.crs,
        nodata=np.nan,
        bounds=dem.bounds,
        band_ids=["aspect_sin", "aspect_cos"],
    )


def compute_rugosity(dem: Raster) -> Raster:
    import numpy as np

    elev = dem.array[0].astype(np.float64)
    h, w = elev.shape
    padded = np.pad(elev, 1, mode="edge")

    sq_sum = np.zeros((h, w), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            neighbor = padded[di : di + h, dj : dj + w]
            sq_sum += (neighbor - elev) ** 2

    tri = np.sqrt(sq_sum).astype(np.float32)
    return Raster(
        array=tri[np.newaxis],
        transform=dem.transform,
        crs=dem.crs,
        nodata=np.nan,
        bounds=dem.bounds,
        band_ids=["tri"],
    )


def compute_orographic_uplift(slope: Raster, aspect: Raster, era5_land: Raster) -> Raster:
    import numpy as np

    slope_rad = np.radians(slope.array[0])
    sin_asp = aspect.array[0]
    cos_asp = aspect.array[1]
    aspect_rad = np.arctan2(sin_asp, cos_asp)

    u10 = era5_land.array[4].astype(np.float64)
    v10 = era5_land.array[5].astype(np.float64)
    u10 = np.where(np.isfinite(u10), u10, np.nan)
    v10 = np.where(np.isfinite(v10), v10, np.nan)

    u10_mean = np.nanmean(u10)
    v10_mean = np.nanmean(v10)

    wind_speed = np.sqrt(u10_mean**2 + v10_mean**2)
    wind_dir = np.arctan2(u10_mean, v10_mean)

    with np.errstate(invalid="ignore"):
        wo = wind_speed * np.sin(slope_rad) * np.cos(wind_dir - aspect_rad)
    wo = np.nan_to_num(wo, nan=np.nan, posinf=np.nan, neginf=np.nan).astype(np.float32)

    return Raster(
        array=wo[np.newaxis],
        transform=slope.transform,
        crs=slope.crs,
        nodata=np.nan,
        bounds=slope.bounds,
        band_ids=["orographic_uplift"],
    )


def compute_thermal_uplift(era5_land: Raster, era5: Raster) -> Raster:
    import numpy as np

    t2m = era5_land.array[0].astype(np.float64)
    shf = era5_land.array[11].astype(np.float64)
    t2m = np.where(np.isfinite(t2m), t2m, np.nan)
    shf = np.where(np.isfinite(shf), shf, np.nan)

    blh = era5.array[0].astype(np.float64)
    blh = np.where(np.isfinite(blh), blh, np.nan)

    t2m_mean = np.nanmean(t2m)
    shf_mean = np.nanmean(shf)
    blh_mean = np.nanmean(blh)

    g = 9.81
    rho = 1.225
    cp = 1005.0

    shf_wm2 = shf_mean / 3600.0
    q0 = -shf_wm2 / (rho * cp)

    if (
        np.isfinite(q0)
        and q0 > 0
        and np.isfinite(blh_mean)
        and blh_mean > 0
        and np.isfinite(t2m_mean)
        and t2m_mean > 0
    ):
        w_star = (g * q0 * blh_mean / t2m_mean) ** (1.0 / 3.0)
    else:
        w_star = 0.0

    h, w = era5_land.array.shape[1:]
    result = np.full((1, h, w), w_star, dtype=np.float32)

    return Raster(
        array=result,
        transform=era5_land.transform,
        crs=era5_land.crs,
        nodata=np.nan,
        bounds=era5_land.bounds,
        band_ids=["thermal_uplift"],
    )

