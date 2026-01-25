"""Raster file loading utilities."""

import os
import glob
from functools import lru_cache

import rasterio
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.enums import Resampling


def load_geotiff(path):
    """
    Load a GeoTIFF file.

    Args:
        path: Path to the GeoTIFF file

    Returns:
        dict with keys: array, transform, crs, nodata, bounds, bands
    """
    with rasterio.open(path) as ds:
        arr = ds.read()
        return {
            "array": arr,
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
            "bounds": ds.bounds,
            "bands": ds.descriptions,
        }


@lru_cache(maxsize=16)
def load_geotiff_cached(path):
    """Load a GeoTIFF file with LRU caching."""
    return load_geotiff(path)


@lru_cache(maxsize=20)
def load_geotiff_window(path, bounds):
    """
    Load a windowed region from GeoTIFF.

    Args:
        path: Path to the GeoTIFF file
        bounds: Tuple (min_lon, min_lat, max_lon, max_lat)

    Returns:
        dict with keys: array, transform, crs, nodata, bounds
    """
    with rasterio.open(path) as ds:
        min_lon, min_lat, max_lon, max_lat = bounds
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, ds.transform)
        return {
            "array": ds.read(window=window),
            "transform": ds.window_transform(window),
            "crs": ds.crs,
            "nodata": ds.nodata,
            "bounds": bounds,
        }


@lru_cache(maxsize=20)
def load_merged_hdf_window(paths, bounds_latlon):
    """
    Load and merge multiple HDF files for a region.

    Applies CRS bounds transform (to accommodate MODIS sinusoidal projection).

    Args:
        paths: Tuple of HDF subdataset paths
        bounds_latlon: Tuple (min_lon, min_lat, max_lon, max_lat) in EPSG:4326

    Returns:
        dict with keys: array, transform, crs, nodata, bounds, bands
    """
    datasets = [rasterio.open(p) for p in paths]
    try:
        crs = datasets[0].crs
        descriptions = datasets[0].descriptions
        nodata_val = datasets[0].nodata or 255

        # transform to MODIS CRS
        left, bottom, right, top = transform_bounds(
            "EPSG:4326",  # bounds assumed to come from EPSG:4326
            crs,          # but we want to use them in the target (e.g. MODIS) CRS
            *bounds_latlon,
        )

        mosaic, out_transform = merge(
            datasets,
            bounds=(left, bottom, right, top),
            nodata=nodata_val,
            resampling=Resampling.nearest,
        )
    except Exception as e:
        print(f"Error in load_merged_hdf_window: {e}")
        raise
    finally:
        for ds in datasets:
            ds.close()

    return {
        "array": mosaic,           # (bands, H, W)
        "transform": out_transform,
        "crs": crs,
        "nodata": nodata_val,
        "bounds": (left, bottom, right, top),
        "bands": descriptions,
    }


class ModisFileIndex:
    """
    Index MODIS HDF files for fast (year, h, v) lookup.

    This class builds an index of MODIS HDF files at initialization
    time to avoid expensive glob operations during data loading.
    """

    def __init__(self, base_dir, product_prefix):
        """
        Build index of MODIS files.

        Args:
            base_dir: Directory containing HDF files
            product_prefix: e.g., "MCD12Q1" or "MOD44B"
        """
        self._index = {}
        self._base_dir = base_dir
        self._product_prefix = product_prefix

        pattern = os.path.join(base_dir, f"{product_prefix}.A*.h*v*.hdf")
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            try:
                year = int(fname.split(".A")[1][:4])
                h = int(fname.split(".h")[1][:2])
                v = int(fname.split("v")[1][:2])
                self._index[(year, h, v)] = path
            except (IndexError, ValueError):
                # Skip files that don't match expected format
                continue

    def get_path(self, year, h, v):
        """Get path for a specific (year, h, v) combination."""
        return self._index.get((year, h, v))

    def __contains__(self, key):
        """Check if (year, h, v) exists in index."""
        return key in self._index

    def __getitem__(self, key):
        """Get path by (year, h, v) key."""
        return self._index[key]

    def __len__(self):
        """Number of indexed files."""
        return len(self._index)
