from __future__ import annotations

import numpy as np
from pyproj import CRS
from rasterio.enums import Resampling

from movebench.rasters import Raster, aeqd_work_grid, reproject_multipart_to_meters


def _band_index(r: Raster, band_id: str) -> int:
    try:
        return r.band_ids.index(band_id)
    except ValueError as e:
        raise KeyError(
            f"Raster missing required band_id {band_id!r}. Has: {r.band_ids}"
        ) from e


def _band_array(r: Raster, band_id: str) -> np.ndarray:
    idx = _band_index(r, band_id)
    arr = r.array[idx].astype(np.float64)
    return np.where(np.isfinite(arr), arr, np.nan)


def _safe_divide(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def _safe_reciprocal(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 1.0 / x
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def _landsat_band(landsat: Raster, band: str) -> np.ndarray:
    return _band_array(landsat, f"landsat.{band}")


def _require_metric_grid(raster: Raster) -> None:
    crs = CRS.from_user_input(raster.crs)
    units = [axis.unit_conversion_factor for axis in crs.axis_info[:2]]
    operation = crs.coordinate_operation
    if (
        not crs.is_projected
        or any(abs(float(unit) - 1.0) > 1e-9 for unit in units)
        or operation is None
        or "Azimuthal Equidistant" not in operation.method_name
    ):
        raise ValueError("Metric terrain derivatives require an observation-centered AEQD meter grid")


def _gradient_meters(elev: np.ndarray, raster: Raster) -> tuple[np.ndarray, np.ndarray]:
    _require_metric_grid(raster)
    gy, gx = np.gradient(
        elev,
        abs(float(raster.transform.e)),
        abs(float(raster.transform.a)),
    )
    return gy, gx


def compute_slope(*, dem: Raster) -> np.ndarray:
    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    return slope_deg[np.newaxis]


def compute_aspect(*, dem: Raster) -> np.ndarray:
    elev = dem.array[0].astype(np.float64)
    gy, gx = _gradient_meters(elev, dem)

    # radians, north=0 clockwise
    aspect_rad = np.arctan2(-gx, gy)
    sin_aspect = np.sin(aspect_rad).astype(np.float32)
    cos_aspect = np.cos(aspect_rad).astype(np.float32)
    return np.stack([sin_aspect, cos_aspect], axis=0)


def compute_rugosity(*, dem: Raster) -> np.ndarray:
    _require_metric_grid(dem)
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


def compute_landsat_ndwi(*, landsat: Raster) -> np.ndarray:
    green = _landsat_band(landsat, "green")
    nir = _landsat_band(landsat, "nir")
    ndwi = _safe_divide(green - nir, green + nir)
    return ndwi[np.newaxis]


def compute_landsat_ndvi(*, landsat: Raster) -> np.ndarray:
    nir = _landsat_band(landsat, "nir")
    red = _landsat_band(landsat, "red")
    ndvi = _safe_divide(nir - red, nir + red)
    return ndvi[np.newaxis]


def compute_landsat_evi(*, landsat: Raster) -> np.ndarray:
    blue = _landsat_band(landsat, "blue")
    red = _landsat_band(landsat, "red")
    nir = _landsat_band(landsat, "nir")
    numer = nir - red
    denom = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = 2.5 * _safe_divide(numer, denom)
    return evi[np.newaxis]


def compute_landsat_ndsi(*, landsat: Raster) -> np.ndarray:
    green = _landsat_band(landsat, "green")
    swir1 = _landsat_band(landsat, "swir1")
    ndsi = _safe_divide(green - swir1, green + swir1)
    return ndsi[np.newaxis]


def compute_landsat_bai(*, landsat: Raster) -> np.ndarray:
    red = _landsat_band(landsat, "red")
    nir = _landsat_band(landsat, "nir")
    bai = _safe_reciprocal((0.1 - red) ** 2 + (0.06 - nir) ** 2)
    return bai[np.newaxis]


def compute_landsat_lst(*, landsat: Raster) -> np.ndarray:
    thermal = _landsat_band(landsat, "thermal")
    return thermal[np.newaxis]


def compute_orographic_uplift_hourly(
    *, slope: Raster, aspect: Raster, era5_land_hourly: Raster
) -> np.ndarray:
    _require_metric_grid(slope)
    _require_metric_grid(aspect)
    slope_rad = np.radians(slope.array[0])

    sin_asp = aspect.array[0]
    cos_asp = aspect.array[1]
    aspect_rad = np.arctan2(sin_asp, cos_asp)

    u_i = _band_index(era5_land_hourly, "era5_land_hourly.u_component_of_wind_10m")
    v_i = _band_index(era5_land_hourly, "era5_land_hourly.v_component_of_wind_10m")
    u10 = era5_land_hourly.array[u_i].astype(np.float64)
    v10 = era5_land_hourly.array[v_i].astype(np.float64)
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


def derive_metric_terrain_layers(
    *,
    dem_parts: list[Raster],
    dst_crs,
    support_bounds_m: tuple[float, float, float, float],
    requested_layers: set[str],
    era5_land_hourly: Raster | None = None,
) -> dict[str, Raster]:
    """Warp DEM once, then derive requested terrain on its fine AEQD grid."""
    transform, width, height, _resolution, bounds = aeqd_work_grid(
        dem_parts,
        dst_crs,
        support_bounds_m,
        halo_pixels=1,
    )
    [dem_array], _transform, _crs = reproject_multipart_to_meters(
        [dem_parts],
        dst_crs,
        resampling=Resampling.bilinear,
        dst_width=width,
        dst_height=height,
        dst_transform=transform,
    )
    dem = Raster(
        array=dem_array,
        transform=transform,
        crs=dst_crs,
        nodata=np.nan,
        bounds=bounds,
        band_ids=dem_parts[0].band_ids,
    )

    needed = set(requested_layers)
    if "orographic_uplift_hourly" in needed:
        needed.update(("slope", "aspect"))

    result: dict[str, Raster] = {}

    def store(layer: str, array: np.ndarray) -> None:
        result[layer] = Raster(
            array=array,
            transform=transform,
            crs=dst_crs,
            nodata=np.nan,
            bounds=bounds,
            band_ids={
                "slope": ["slope.slope"],
                "aspect": ["aspect.aspect_sin", "aspect.aspect_cos"],
                "rugosity": ["rugosity.tri"],
                "orographic_uplift_hourly": [
                    "orographic_uplift_hourly.orographic_uplift_hourly"
                ],
            }[layer],
        )

    if "slope" in needed:
        store("slope", compute_slope(dem=dem))
    if "aspect" in needed:
        store("aspect", compute_aspect(dem=dem))
    if "rugosity" in needed:
        store("rugosity", compute_rugosity(dem=dem))
    if "orographic_uplift_hourly" in needed:
        if era5_land_hourly is None:
            raise ValueError("orographic uplift requires era5_land_hourly")
        store(
            "orographic_uplift_hourly",
            compute_orographic_uplift_hourly(
                slope=result["slope"],
                aspect=result["aspect"],
                era5_land_hourly=era5_land_hourly,
            ),
        )
    return result


def compute_thermal_uplift_hourly(
    *, era5_land_hourly: Raster, era5_hourly: Raster
) -> np.ndarray:
    t_i = _band_index(era5_land_hourly, "era5_land_hourly.temperature_2m")
    shf_i = _band_index(era5_land_hourly, "era5_land_hourly.surface_sensible_heat_flux")
    blh_i = _band_index(era5_hourly, "era5_hourly.boundary_layer_height")

    t2m = era5_land_hourly.array[t_i].astype(np.float64)
    shf = era5_land_hourly.array[shf_i].astype(np.float64)
    blh = era5_hourly.array[blh_i].astype(np.float64)

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

    H, W = era5_land_hourly.array.shape[1:]
    return np.full((1, H, W), w_star, dtype=np.float32)
