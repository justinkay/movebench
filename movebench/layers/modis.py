from __future__ import annotations

from functools import lru_cache

from movebench import paths


@lru_cache(maxsize=1)
def modis_grid_rows() -> list[tuple[int, int, float, float, float, float]]:
    """Rows from sn_bound_10deg.txt as (h, v, lon_min, lon_max, lat_min, lat_max)."""
    grid_path = paths.get_path("modis_grid")
    rows: list[tuple[int, int, float, float, float, float]] = []
    with open(grid_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            v = int(parts[0])
            h = int(parts[1])
            lon_min = float(parts[2])
            lon_max = float(parts[3])
            lat_min = float(parts[4])
            lat_max = float(parts[5])
            rows.append((h, v, lon_min, lon_max, lat_min, lat_max))
    return rows


def modis_tiles_for_bounds(bounds_latlon: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Return MODIS sinusoidal-grid tile indices (h, v) overlapping a lat/lon bounds box."""
    lon_min, lat_min, lon_max, lat_max = bounds_latlon
    out: list[tuple[int, int]] = []
    for h, v, lo0, lo1, la0, la1 in modis_grid_rows():
        if lo0 <= lon_max and lo1 >= lon_min and la0 <= lat_max and la1 >= lat_min:
            out.append((h, v))
    out.sort(key=lambda hv: (hv[1], hv[0]))
    return out

