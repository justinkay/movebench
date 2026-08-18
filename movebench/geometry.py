"""Per-observation metric geometry used after raster downloads."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pyproj import CRS, Geod

LOCAL_GEOMETRY = "aeqd_wgs84_per_observation_v2"
MAX_LOCAL_RADIUS_M = 10_000_000.0

_WGS84_GEOD = Geod(ellps="WGS84")


def local_aeqd_proj4(center_lon: float, center_lat: float) -> str:
    """Return the lightweight CRS metadata for one observation."""
    return (
        f"+proj=aeqd +lat_0={float(center_lat):.12g} "
        f"+lon_0={float(center_lon):.12g} +datum=WGS84 +units=m +no_defs"
    )


@lru_cache(maxsize=4096)
def local_aeqd_crs(center_lon: float, center_lat: float) -> CRS:
    """Return WGS84 AEQD centered on one observation."""
    return CRS.from_proj4(local_aeqd_proj4(center_lon, center_lat))


def validate_local_radius_m(radius_m) -> None:
    """Reject local AEQD coordinates beyond the supported radius."""
    radius = np.asarray(radius_m, dtype=np.float64)
    if np.any(radius > MAX_LOCAL_RADIUS_M):
        raise ValueError(
            f"Local AEQD radius exceeds {MAX_LOCAL_RADIUS_M:g} m: "
            f"maximum requested radius is {float(np.max(radius)):g} m"
        )


def validate_local_bounds_m(bounds: tuple[float, float, float, float]) -> None:
    """Validate the farthest outer corner of a local raster footprint."""
    left, bottom, right, top = (float(value) for value in bounds)
    validate_local_radius_m(
        max(
            np.hypot(x, y)
            for x in (left, right)
            for y in (bottom, top)
        )
    )


def local_xy_m_from_lonlat(
    center_lon,
    center_lat,
    lon,
    lat,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lon/lat to east/north meters in observation-centered AEQD."""
    center_lon, center_lat, lon, lat = np.broadcast_arrays(
        np.asarray(center_lon, dtype=np.float64),
        np.asarray(center_lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.asarray(lat, dtype=np.float64),
    )
    azimuth_deg, _back_azimuth_deg, distance_m = _WGS84_GEOD.inv(
        center_lon,
        center_lat,
        lon,
        lat,
    )
    azimuth_rad = np.deg2rad(azimuth_deg)
    return distance_m * np.sin(azimuth_rad), distance_m * np.cos(azimuth_rad)


def geodesic_distance_m(
    start_lat_deg,
    start_lon_deg,
    end_lat_deg,
    end_lon_deg,
) -> np.ndarray:
    """Return vectorized, uncapped WGS84 geodesic distances in meters."""
    start_lat_deg, start_lon_deg, end_lat_deg, end_lon_deg = np.broadcast_arrays(
        np.asarray(start_lat_deg, dtype=np.float64),
        np.asarray(start_lon_deg, dtype=np.float64),
        np.asarray(end_lat_deg, dtype=np.float64),
        np.asarray(end_lon_deg, dtype=np.float64),
    )
    _azimuth, _back_azimuth, distance_m = _WGS84_GEOD.inv(
        start_lon_deg,
        start_lat_deg,
        end_lon_deg,
        end_lat_deg,
    )
    return np.asarray(distance_m, dtype=np.float64)


def lonlat_from_local_xy_m(
    center_lon,
    center_lat,
    x_m,
    y_m,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert observation-centered AEQD east/north meters to lon/lat."""
    center_lon, center_lat, x_m, y_m = np.broadcast_arrays(
        np.asarray(center_lon, dtype=np.float64),
        np.asarray(center_lat, dtype=np.float64),
        np.asarray(x_m, dtype=np.float64),
        np.asarray(y_m, dtype=np.float64),
    )
    azimuth_deg = np.rad2deg(np.arctan2(x_m, y_m))
    distance_m = np.hypot(x_m, y_m)
    validate_local_radius_m(distance_m)
    lon, lat, _back_azimuth_deg = _WGS84_GEOD.fwd(
        center_lon,
        center_lat,
        azimuth_deg,
        distance_m,
    )
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def minimal_circular_longitude_extent(longitudes) -> tuple[float, float]:
    """Return the center and span of the smallest circular longitude interval."""
    values = np.asarray(longitudes, dtype=np.float64).ravel()
    if values.size == 0:
        raise ValueError("At least one longitude is required")
    values = np.mod(values, 360.0)
    values.sort()
    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    gap_index = int(np.argmax(gaps))
    start = float(values[(gap_index + 1) % values.size])
    span = float(360.0 - gaps[gap_index])
    center = (start + span / 2.0 + 180.0) % 360.0 - 180.0
    return center, span


def local_bounds_to_wgs84_parts(
    center_lon: float,
    center_lat: float,
    bounds_m: tuple[float, float, float, float],
    *,
    samples_per_edge: int = 181,
) -> tuple[tuple[float, float, float, float], ...]:
    """Convert a local footprint to one or two canonical WGS84 boxes."""
    validate_local_bounds_m(bounds_m)
    left, bottom, right, top = (float(value) for value in bounds_m)
    edge_samples = max(2, int(samples_per_edge))
    xs = np.linspace(left, right, edge_samples)
    ys = np.linspace(bottom, top, edge_samples)
    edge_x = np.concatenate(
        [xs, xs, np.full_like(ys, left), np.full_like(ys, right)]
    )
    edge_y = np.concatenate(
        [np.full_like(xs, bottom), np.full_like(xs, top), ys, ys]
    )
    lons, lats = lonlat_from_local_xy_m(
        center_lon,
        center_lat,
        edge_x,
        edge_y,
    )
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))

    for pole_lat in (-90.0, 90.0):
        pole_x, pole_y = local_xy_m_from_lonlat(
            center_lon,
            center_lat,
            center_lon,
            pole_lat,
        )
        if left <= float(pole_x) <= right and bottom <= float(pole_y) <= top:
            if pole_lat < 0:
                min_lat = -90.0
            else:
                max_lat = 90.0
            return ((-180.0, min_lat, 180.0, max_lat),)

    values = np.mod(np.asarray(lons, dtype=np.float64), 360.0)
    values.sort()
    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    gap_index = int(np.argmax(gaps))
    start = float(values[(gap_index + 1) % values.size])
    span = float(360.0 - gaps[gap_index])
    if span >= 360.0 - 1e-10:
        return ((-180.0, min_lat, 180.0, max_lat),)

    left_lon = (start + 180.0) % 360.0 - 180.0
    right_lon = left_lon + span
    if right_lon <= 180.0 + 1e-10:
        return ((left_lon, min_lat, min(180.0, right_lon), max_lat),)
    return (
        (left_lon, min_lat, 180.0, max_lat),
        (-180.0, min_lat, right_lon - 360.0, max_lat),
    )


def centered_grid_coordinates_m(
    output_hw: tuple[int, int],
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for a grid centered on local AEQD origin."""
    height, width = (int(output_hw[0]), int(output_hw[1]))
    x_m = (np.arange(width, dtype=np.float64) - (width - 1) / 2.0) * float(
        pixel_size_m
    )
    y_m = ((height - 1) / 2.0 - np.arange(height, dtype=np.float64)) * float(
        pixel_size_m
    )
    return x_m, y_m
