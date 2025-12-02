import time
from functools import wraps
import requests
import pandas as pd
import ee
import numpy as np
from retry import retry


# DataFrame column names
# TODO move to movebank specific module
LON_COL = "location-long"
LAT_COL = "location-lat"
TIME_COL = "timestamp"

# Pixel constraints for scale computation
MIN_PIXELS = 3
MAX_PIXELS = 100

# GEE download limits
MAX_DOWNLOAD_SIZE_MB = 32
MAX_GRID_DIMENSION = 10000


def compute_group_scale(buffer_km,
                        group_scale_m,
                        min_pixels=MIN_PIXELS,
                        max_pixels=MAX_PIXELS):
    """
    Choose a sampling scale (meters) for a group.

    Constraints:
      - chip side N = 2R / scale between [min_pixels, max_pixels]
      - scale as close as possible to group_scale_m
      - upsampling (scale < group_scale_m) allowed to reach min_pixels
      - downsampling (scale > group_scale_m) allowed to respect max_pixels

    Args:
        buffer_km: Buffer radius in kilometers
        group_scale_m: Target scale in meters
        min_pixels: Minimum allowed pixels per side
        max_pixels: Maximum allowed pixels per side

    Returns:
        Optimal scale in meters
    """
    R_m = buffer_km * 1000.0

    # Coarsest allowed scale → N = max_pixels
    scale_for_max = (2.0 * R_m) / max_pixels

    # Finest allowed scale → N = min_pixels
    scale_for_min = (2.0 * R_m) / min_pixels

    # Case 1: native is finer than we can afford → coarsen to scale_for_max
    if group_scale_m < scale_for_max:
        return scale_for_max

    # Case 2: native is coarser than allowed → upsample to scale_for_min
    if group_scale_m > scale_for_min:
        return scale_for_min

    # Case 3: native already in allowed [scale_for_max, scale_for_min]
    return group_scale_m


# -------------------------------------------------------------------------
# EARTH ENGINE HELPERS
# -------------------------------------------------------------------------

def square_region_for_point(lon, lat, buffer_km):
    pt = ee.Geometry.Point([lon, lat])
    return pt.buffer(buffer_km * 1000.0).bounds() # bounds() turns into bounding box


def union_regions(regions):
    return ee.Geometry.MultiPolygon(regions)


# -------------------------------------------------------------------------
# DOWNLOAD UTILITIES
# -------------------------------------------------------------------------

@retry((requests.RequestException,), tries=5, delay=2, backoff=2)
def download_tiff(url, out_path):
    """
    Download a TIFF file from URL with retry logic.

    Args:
        url: Download URL
        out_path: Output file path
    """
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(1 << 20):  # 1MB chunks
            if chunk:
                f.write(chunk)


# -------------------------------------------------------------------------
# MOVEMENT ANALYSIS
# -------------------------------------------------------------------------

def get_buffer_m_for_study(df,
                            step_q=0.95,
                            step_multiplier=2):
    """
    Compute appropriate buffer for a study based on movement data.

    Args:
        df: DataFrame with movement data
        step_q: Quantile for step length calculation
        step_multiplier: Multiplier for final buffer size

    Returns:
        Buffer distance in meters
    """
    if TIME_COL not in df.columns:
        raise RuntimeError(f"Expected a '{TIME_COL}' column in the data.")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # -------------------------------------------------------------------
    # 1) Compute and filter by time gap (per-individual quantile)
    # -------------------------------------------------------------------

    df = df.sort_values(['individual-local-identifier', TIME_COL])
    df['time_since_last'] = df.groupby('individual-local-identifier')[TIME_COL].diff()

    gap_q = 0.999  # very high quantile for time gap per individual
    per_indiv_cutoff = (
        df.groupby('individual-local-identifier')['time_since_last']
        .transform(lambda s: s.dropna().quantile(gap_q))
    )
    df_filtered = df[
        df['time_since_last'].notna() &
        (df['time_since_last'] <= per_indiv_cutoff)
    ].copy()

    # -------------------------------------------------------------------
    # 2) Compute and filter by step length (Haversine)
    # -------------------------------------------------------------------

    lat_rad = np.radians(df_filtered[LAT_COL])
    lon_rad = np.radians(df_filtered[LON_COL])

    lat_prev = lat_rad.groupby(df_filtered['individual-local-identifier']).shift()
    lon_prev = lon_rad.groupby(df_filtered['individual-local-identifier']).shift()

    dlat = lat_rad - lat_prev
    dlon = lon_rad - lon_prev

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # meters
    df_filtered['step_length'] = R * c
    df_filtered = df_filtered.dropna(subset='step_length')

    if len(df_filtered) == 0:
        raise ValueError("No valid step lengths computed")

    print(f"Step lengths computed: mean={df_filtered['step_length'].mean():.1f}m, "
          f"q95={np.quantile(df_filtered['step_length'], 0.95):.1f}m")

    return step_multiplier * np.quantile(df_filtered['step_length'].values, q=step_q)