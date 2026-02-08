from __future__ import annotations

import numpy as np

from movebench2.rasters import Raster


def _band_index(r: Raster, band_id: str) -> int:
    try:
        return r.band_ids.index(band_id)
    except ValueError as e:
        raise KeyError(f"Raster missing required band_id {band_id!r}. Has: {r.band_ids}") from e


def _gradient_meters(elev: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Compute elevation gradients in m/m, assuming transform is in degrees (EPSG:4326)."""

    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)

    # Use center latitude for longitude correction.
    nrows = elev.shape[0]
    center_row = nrows // 2
    center_lat = transform.f + center_row * transform.e
    lat_rad = np.radians(center_lat)

    dy_m = dy_deg * 110540.0
    dx_m = dx_deg * 111320.0 * np.cos(lat_rad)

    gy, gx = np.gradient(elev, dy_m, dx_m)
    return gy, gx


def compute_slope(*, dem: Raster) -> np.ndarray:
    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem.transform)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return slope_deg[np.newaxis]


def compute_aspect(*, dem: Raster) -> np.ndarray:
    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem.transform)

    # radians, north=0 clockwise
    aspect_rad = np.arctan2(-gx, gy)
    sin_aspect = np.sin(aspect_rad).astype(np.float32)
    cos_aspect = np.cos(aspect_rad).astype(np.float32)
    return np.stack([sin_aspect, cos_aspect], axis=0)


def compute_rugosity(*, dem: Raster) -> np.ndarray:
    elev = dem.array[0].astype(np.float64)
    H, W = elev.shape

    padded = np.pad(elev, 1, mode="edge")
    sq_sum = np.zeros((H, W), dtype=np.float64)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            neighbor = padded[di : di + H, dj : dj + W]
            sq_sum += (neighbor - elev) ** 2

    tri = np.sqrt(sq_sum).astype(np.float32)
    return tri[np.newaxis]


def compute_orographic_uplift(*, slope: Raster, aspect: Raster, era5_land: Raster) -> np.ndarray:
    slope_rad = np.radians(slope.array[0])

    sin_asp = aspect.array[0]
    cos_asp = aspect.array[1]
    aspect_rad = np.arctan2(sin_asp, cos_asp)

    u_i = _band_index(era5_land, "era5_land.u_component_of_wind_10m")
    v_i = _band_index(era5_land, "era5_land.v_component_of_wind_10m")
    u10 = era5_land.array[u_i].astype(np.float64)
    v10 = era5_land.array[v_i].astype(np.float64)
    u10 = np.where(np.isfinite(u10), u10, np.nan)
    v10 = np.where(np.isfinite(v10), v10, np.nan)

    u10_mean = np.nanmean(u10)
    v10_mean = np.nanmean(v10)

    wind_speed = np.sqrt(u10_mean**2 + v10_mean**2)
    wind_dir = np.arctan2(u10_mean, v10_mean)

    with np.errstate(invalid="ignore"):
        wo = wind_speed * np.sin(slope_rad) * np.cos(wind_dir - aspect_rad)
    wo = np.where(np.isfinite(wo), wo, np.nan).astype(np.float32)
    return wo[np.newaxis]


def compute_thermal_uplift(*, era5_land: Raster, era5: Raster) -> np.ndarray:
    t_i = _band_index(era5_land, "era5_land.temperature_2m")
    shf_i = _band_index(era5_land, "era5_land.surface_sensible_heat_flux")
    blh_i = _band_index(era5, "era5.boundary_layer_height")

    t2m = era5_land.array[t_i].astype(np.float64)
    shf = era5_land.array[shf_i].astype(np.float64)
    blh = era5.array[blh_i].astype(np.float64)

    t2m = np.where(np.isfinite(t2m), t2m, np.nan)
    shf = np.where(np.isfinite(shf), shf, np.nan)
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
        w_star = float((g * q0 * blh_mean / t2m_mean) ** (1.0 / 3.0))
    else:
        w_star = 0.0

    H, W = era5_land.array.shape[1:]
    return np.full((1, H, W), w_star, dtype=np.float32)

