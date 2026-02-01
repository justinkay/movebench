"""Movement analysis and MODIS grid utilities."""

import re
from io import StringIO

import numpy as np
import pandas as pd

from movebench.config import PATHS, LON_COL, LAT_COL, TIME_COL, INDIVIDUAL_COL, LayerConfig

# Lazy-loaded MODIS grid
_modis_grid = None


def _load_modis_grid():
    """Load MODIS sinusoidal grid bounds (lazy)."""
    global _modis_grid
    if _modis_grid is None:
        grid_file = PATHS.get("modis_grid", "sn_bound_10deg.txt")
        with open(grid_file) as f:
            raw = f.read()
        lines = [ln for ln in raw.splitlines() if ln.strip() and re.match(r"^\d", ln.strip())]
        buf = StringIO("\n".join(lines))
        _modis_grid = pd.read_csv(buf, sep=r'\s+', header=None,
                                   names=["iv", "ih", "lon_min", "lon_max", "lat_min", "lat_max"])
    return _modis_grid


def lat_lon_to_modis_tiles(lat, lon):
    """Get MODIS tile (h, v) for a lat/lon coordinate."""
    grid = _load_modis_grid()
    rows = grid[(grid.lon_min <= lon) & (lon <= grid.lon_max) &
                (grid.lat_min <= lat) & (lat <= grid.lat_max)]
    return list(zip(rows.ih.values, rows.iv.values))


def modis_tiles_for_bounds(bounds):
    """Get MODIS tiles (h, v) intersecting a bounding box."""
    grid = _load_modis_grid()
    lon_min, lat_min, lon_max, lat_max = bounds
    rows = grid[(grid.lon_min <= lon_max) & (grid.lon_max >= lon_min) &
                (grid.lat_min <= lat_max) & (grid.lat_max >= lat_min)]
    rows = rows.sort_values(["iv", "ih"])
    return list(zip(rows.ih.values, rows.iv.values))


def make_placeholder(
    layer_config: LayerConfig,
    shape: tuple[int, int],
    transform,
    crs,
    bounds,
    nodata: float | None = np.nan,
    bands: list[str] | None = None,
) -> dict:
    """
    Build a NaN-filled placeholder raster for a layer.

    Args:
        layer_config: LayerConfig for the layer.
        shape: (height, width) for the placeholder array.
        transform: Rasterio transform to attach.
        crs: Coordinate reference system to attach.
        bounds: Bounds tuple to attach.
        nodata: Nodata value to use (defaults to NaN).
        bands: Optional band names override (defaults to layer_config.bands).

    Returns:
        Placeholder raster dict.
    """
    band_names = bands or layer_config.bands
    height, width = shape
    array = np.full((len(band_names), height, width), np.nan, dtype=np.float32)
    return {
        "array": array,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "bounds": bounds,
        "bands": band_names,
    }


# Alias for backward compatibility
lat_lon_to_modis = lat_lon_to_modis_tiles


def get_buffer_m_for_study(df, step_q=0.95, step_multiplier=2, gap_q=0.999):
    """
    Compute buffer size from movement data.

    The buffer is used to determine chip size around GPS points.

    Args:
        df: DataFrame with location columns
        step_q: Quantile of step lengths to use
        step_multiplier: Multiply quantile by this factor
        gap_q: Quantile of per-individual time gaps to retain

    Returns:
        Buffer size in meters
    """
    if TIME_COL not in df.columns:
        raise RuntimeError(f"Expected a '{TIME_COL}' column in the data.")

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL])
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL])

    df['time_since_last'] = df.groupby(INDIVIDUAL_COL)[TIME_COL].diff()
    per_indiv_cutoff = (
        df.groupby(INDIVIDUAL_COL)['time_since_last']
        .transform(lambda s: s.dropna().quantile(gap_q))
    )
    df = df[
        df['time_since_last'].notna()
        & (df['time_since_last'] <= per_indiv_cutoff)
    ].copy()

    df['lat_rad'] = np.deg2rad(df[LAT_COL])
    df['lon_rad'] = np.deg2rad(df[LON_COL])

    g = df.groupby(INDIVIDUAL_COL)

    R = 6371000.0  # Earth radius in meters
    df['lat_prev'] = g['lat_rad'].shift()
    df['lon_prev'] = g['lon_rad'].shift()
    dlat = df['lat_rad'] - df['lat_prev']
    dlon = df['lon_rad'] - df['lon_prev']

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(df['lat_prev']) * np.cos(df['lat_rad']) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df['step_length_m'] = R * c

    valid_steps = df['step_length_m'].dropna()
    valid_steps = valid_steps[valid_steps > 0]

    if len(valid_steps) == 0:
        raise ValueError("No valid step lengths computed")

    buffer_m = step_multiplier * np.nanquantile(valid_steps, q=step_q)
    return buffer_m


def setup_df_for_ssf(df, step_multiplier=2, step_q=0.95, gap_q=0.999):
    """
    Prepare DataFrame for Step Selection Function analysis.

    This function adds movement metrics (step length, turn angle, bearing)
    and identifies valid SSF start points.

    Args:
        df: DataFrame with location and timestamp columns
        step_multiplier: Multiply step quantile by this for max step
        step_q: Quantile for step length cutoff
        gap_q: Quantile for time gap cutoff

    Returns:
        DataFrame with additional columns for SSF analysis
    """
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL])

    # sort so diffs make sense; keep original index (chip ids)
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL])

    df['lat_rad'] = np.deg2rad(df[LAT_COL])
    df['lon_rad'] = np.deg2rad(df[LON_COL])

    g = df.groupby(INDIVIDUAL_COL)

    # ---- time gaps per individual ----
    df['time_since_last'] = g[TIME_COL].diff()
    per_indiv_cutoff = g['time_since_last'].transform(
        lambda s: s.dropna().quantile(gap_q)
    )
    df['within_time'] = df['time_since_last'] <= per_indiv_cutoff

    # ---- step length per individual ----
    R = 6371000.0  # Earth radius in meters
    df['lat_prev'] = g['lat_rad'].shift()
    df['lon_prev'] = g['lon_rad'].shift()
    dlat = df['lat_rad'] - df['lat_prev']
    dlon = df['lon_rad'] - df['lon_prev']

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(df['lat_prev']) * np.cos(df['lat_rad']) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df['step_length_m'] = R * c

    # initial valid step mask (time + positive distance)
    df['valid_step'] = (
        df['within_time']
        & df['step_length_m'].notna()
        & (df['step_length_m'] > 0)
    )

    # ---- step-length cutoff ----
    max_step_len = step_multiplier * np.nanquantile(
        df.loc[df['valid_step'], 'step_length_m'], q=step_q
    )
    df['max_step_len'] = max_step_len  # optional
    df['valid_step'] &= df['step_length_m'] <= max_step_len

    # ---- turn angles ----
    y = np.sin(dlon) * np.cos(df['lat_rad'])
    x = (
        np.cos(df['lat_prev']) * np.sin(df['lat_rad'])
        - np.sin(df['lat_prev']) * np.cos(df['lat_rad']) * np.cos(dlon)
    )
    df['bearing'] = np.arctan2(y, x)  # [-pi, pi], NaN at first row per individual

    df['bearing_prev'] = g['bearing'].shift()
    df['valid_step_prev'] = g['valid_step'].shift()

    turn = df['bearing'] - df['bearing_prev']
    turn = (turn + np.pi) % (2.0 * np.pi) - np.pi  # wrap

    df['turn_angle_rad'] = np.where(
        df['valid_step'] & df['valid_step_prev'],
        turn,
        np.nan,
    )

    # ---- bursts ----
    df['is_first_in_indiv'] = g.cumcount() == 0
    df['burst_start'] = df['is_first_in_indiv'] | (~df['valid_step'])
    df['burst_id'] = g['burst_start'].cumsum()

    # ---- SSF sample starts ----
    # index of the next row in time order
    df['next_index'] = df.index.to_series().shift(-1).astype('Int64')

    same_indiv_next = df[INDIVIDUAL_COL].eq(
        df[INDIVIDUAL_COL].shift(-1)
    )
    same_burst_next = df['burst_id'].eq(df['burst_id'].shift(-1))
    valid_step_next = df['valid_step'].shift(-1).fillna(False).astype(bool)

    df['can_be_ssf_start'] = (
        ~df['is_first_in_indiv'] &
        same_indiv_next &
        same_burst_next &
        valid_step_next
    )

    # optional - drop helpers you don't care about downstream
    df = df.drop(
        columns=[
            'lat_rad',
            'lon_rad',
            'lat_prev',
            'lon_prev',
            'bearing_prev',
            'burst_start',
            'is_first_in_indiv',
        ]
    )

    return df
