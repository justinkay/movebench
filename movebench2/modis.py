from __future__ import annotations

"""MODIS sinusoidal grid helpers.

This is used to determine which MODIS HDF tiles intersect a lat/lon bounds.
The grid definition file path is provided by `configs/paths.yaml` via
`paths.get_path("modis_grid")`.
"""

import re
from io import StringIO
from typing import Any

from movebench2 import paths

_modis_grid: Any | None = None


def _load_modis_grid() -> Any:
    global _modis_grid
    if _modis_grid is not None:
        return _modis_grid

    grid_file = paths.get_path("modis_grid")
    with open(grid_file, "r") as f:
        raw = f.read()

    # Keep only numeric rows.
    lines = [ln for ln in raw.splitlines() if ln.strip() and re.match(r"^\\d", ln.strip())]
    buf = StringIO("\n".join(lines))

    import pandas as pd

    _modis_grid = pd.read_csv(
        buf,
        sep=r"\\s+",
        header=None,
        names=["iv", "ih", "lon_min", "lon_max", "lat_min", "lat_max"],
    )
    return _modis_grid


def modis_tiles_for_bounds(bounds_latlon: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    """Return intersecting MODIS tiles for bounds in EPSG:4326.

    Returns a list of (h, v) pairs sorted by (v, h).
    """
    grid = _load_modis_grid()
    lon_min, lat_min, lon_max, lat_max = bounds_latlon
    rows = grid[
        (grid.lon_min <= lon_max)
        & (grid.lon_max >= lon_min)
        & (grid.lat_min <= lat_max)
        & (grid.lat_max >= lat_min)
    ]
    rows = rows.sort_values(["iv", "ih"])
    return [(int(h), int(v)) for h, v in zip(rows.ih.values, rows.iv.values)]


def lat_lon_to_modis_tiles(lat: float, lon: float) -> list[tuple[int, int]]:
    """Return MODIS tiles (h, v) for a single point."""
    grid = _load_modis_grid()
    rows = grid[
        (grid.lon_min <= lon)
        & (lon <= grid.lon_max)
        & (grid.lat_min <= lat)
        & (lat <= grid.lat_max)
    ]
    return [(int(h), int(v)) for h, v in zip(rows.ih.values, rows.iv.values)]

