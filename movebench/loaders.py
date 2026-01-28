"""Raster file loading utilities."""

from __future__ import annotations

import math
import os
import glob
import json
from functools import lru_cache
from typing import Protocol, TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.enums import Resampling

if TYPE_CHECKING:
    from movebench.config import LayerConfig


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

    Uses boundless read to ensure the returned array covers the exact
    requested bounds, filling out-of-bounds areas with nodata.

    Args:
        path: Path to the GeoTIFF file
        bounds: Tuple (min_lon, min_lat, max_lon, max_lat)

    Returns:
        dict with keys: array, transform, crs, nodata, bounds
    """
    with rasterio.open(path) as ds:
        min_lon, min_lat, max_lon, max_lat = bounds
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, ds.transform)
        # Use np.nan as fill when nodata is not defined
        fill_val = ds.nodata if ds.nodata is not None else np.nan
        arr = ds.read(window=window, boundless=True, fill_value=fill_val)
        return {
            "array": arr,
            "transform": ds.window_transform(window),
            "crs": ds.crs,
            "nodata": fill_val,
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

    @classmethod
    def from_layer_config(cls, layer_config: LayerConfig) -> "ModisFileIndex":
        """
        Create index from LayerConfig's path_pattern.

        Args:
            layer_config: LayerConfig with path_pattern containing {year}, {h}, {v}

        Returns:
            ModisFileIndex instance
        """
        # Extract base directory and product prefix from path_pattern
        # Pattern like: /path/to/MCD12Q1.A{year}001.h{h:02d}v{v:02d}.061.*.hdf
        pattern = layer_config.path_pattern
        base_dir = os.path.dirname(pattern)

        # Extract product prefix (e.g., "MCD12Q1" or "MOD44B")
        filename_pattern = os.path.basename(pattern)
        product_prefix = filename_pattern.split(".")[0]

        return cls(base_dir, product_prefix)


# =============================================================================
# LAYER LOADER PROTOCOL AND IMPLEMENTATIONS
# =============================================================================

class LayerLoader(Protocol):
    """Protocol for layer loaders."""

    def load(self, bounds: tuple, timestamp=None, **context) -> dict:
        """
        Load layer data for given bounds and context.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
            timestamp: Optional timestamp for temporal layers
            **context: Additional context (hvs, point_id, etc.)

        Returns:
            dict with keys: array, transform, crs, nodata, bounds, bands
        """
        ...


class LocalTiffLoader:
    """
    Loads windowed GeoTIFFs using path_pattern (for HII, bioclim).
    """

    def __init__(self, layer_config: LayerConfig):
        self.config = layer_config

    def load(self, bounds: tuple, timestamp=None, band_num: int = None, **kwargs) -> dict:
        """
        Load a windowed region from a local GeoTIFF.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            timestamp: Optional timestamp (used for {year} in path_pattern)
            band_num: Optional band number (for bioclim layers)

        Returns:
            dict with raster data
        """
        # Build path from pattern
        path = self.config.path_pattern
        if timestamp is not None and "{year}" in path:
            year_bounds = self.config.year_bounds or (1900, 2100)
            year = max(min(timestamp.year, year_bounds[1]), year_bounds[0])
            path = path.format(year=year)
        if band_num is not None and "{band_num}" in path:
            path = path.format(band_num=band_num)

        return load_geotiff_window(path, bounds)


class LocalHdfLoader:
    """
    Loads MODIS HDF products with tile merging (for modis_lc, modis_cvf_*).
    """

    def __init__(self, layer_config: LayerConfig):
        self.config = layer_config
        self._index = ModisFileIndex.from_layer_config(layer_config)

    def load(self, bounds: tuple, timestamp, hvs: list[tuple[int, int]], **kwargs) -> dict:
        """
        Load and merge MODIS HDF tiles for given bounds.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat)
            timestamp: Timestamp for year selection
            hvs: List of (h, v) MODIS tile coordinates

        Returns:
            dict with merged raster data
        """
        year_bounds = self.config.year_bounds or (1900, 2100)
        year = max(min(timestamp.year, year_bounds[1]), year_bounds[0])

        # Build subdataset paths using config.hdf_path_suffix
        sd_paths = []
        for (h, v) in hvs:
            hdf_path = self._index.get_path(year, int(h), int(v))
            if hdf_path is None:
                raise FileNotFoundError(f"No MODIS file for year={year}, h={h}, v={v}")
            sd_paths.append(f"HDF4_EOS:EOS_GRID:{hdf_path}{self.config.hdf_path_suffix}")

        return load_merged_hdf_window(tuple(sd_paths), bounds)


class ChipLoader:
    """
    Loads pre-downloaded GEE chips (era5, modis).
    """

    def __init__(self, chip_dir: str, chip_group: str):
        """
        Args:
            chip_dir: Directory containing chip files
            chip_group: Chip group name (e.g., "era5", "modis")
        """
        self.chip_dir = chip_dir
        self.chip_group = chip_group

    def load(self, point_id: int, **kwargs) -> dict:
        """
        Load a chip file for a given point.

        Args:
            point_id: Point identifier

        Returns:
            dict with raster data
        """
        chip_path = os.path.join(self.chip_dir, f"{point_id:05d}_{self.chip_group}.tif")
        return load_geotiff(chip_path)


class TileLoader:
    """
    Loads from tile manifest (dem).
    """

    def __init__(self, tile_dir: str, layer_name: str):
        """
        Args:
            tile_dir: Directory containing tiles and index.json
            layer_name: Layer name (e.g., "dem")
        """
        self.tile_dir = tile_dir
        self.layer_name = layer_name
        self._manifest = None

    @property
    def manifest(self) -> dict:
        """Lazy-load tile manifest."""
        if self._manifest is None:
            index_path = os.path.join(self.tile_dir, "index.json")
            with open(index_path, 'r') as f:
                self._manifest = json.load(f)
        return self._manifest

    def load(self, bounds: tuple, **kwargs) -> dict:
        """
        Load and merge all tiles that intersect the given bounds.

        Args:
            bounds: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326

        Returns:
            dict with raster data covering the requested bounds
        """
        min_lon, min_lat, max_lon, max_lat = bounds

        # Find all tiles that intersect with the requested bounds
        matching_tiles = []
        for tile in self.manifest.get("tiles", []):
            b = tile["bounds"]
            # Check if tile bounds intersect with requested bounds
            if (b["min_lon"] <= max_lon and b["max_lon"] >= min_lon and
                    b["min_lat"] <= max_lat and b["max_lat"] >= min_lat):
                tile_path = os.path.join(self.tile_dir, tile["file"])
                matching_tiles.append(tile_path)

        if not matching_tiles:
            center_lon = (min_lon + max_lon) / 2
            center_lat = (min_lat + max_lat) / 2
            raise ValueError(f"No tile found for bounds {bounds} (center: {center_lon}, {center_lat})")

        # If only one tile, read directly with window
        if len(matching_tiles) == 1:
            with rasterio.open(matching_tiles[0]) as ds:
                window = from_bounds(min_lon, min_lat, max_lon, max_lat, ds.transform)
                # Use boundless read to handle edge cases where bounds extend
                # slightly beyond tile extent. Use np.nan as fill when nodata
                # is not defined, since 0 is a valid elevation value.
                fill_val = ds.nodata if ds.nodata is not None else np.nan
                arr = ds.read(window=window, boundless=True, fill_value=fill_val)
                return {
                    "array": arr,
                    "transform": ds.window_transform(window),
                    "crs": ds.crs,
                    "nodata": fill_val,
                    "bounds": bounds,
                    "bands": ds.descriptions,
                }

        # Merge multiple tiles without bounds constraint, then window the result.
        # This avoids edge artifacts where merge() with bounds can produce zeros
        # at pixel edges that extend slightly beyond source data extent.
        datasets = [rasterio.open(p) for p in matching_tiles]
        try:
            crs = datasets[0].crs
            # Use np.nan as nodata when not defined, since 0 is valid for DEM
            nodata_val = datasets[0].nodata if datasets[0].nodata is not None else np.nan

            # Merge without bounds - combines all tiles
            mosaic, full_transform = merge(
                datasets,
                nodata=nodata_val,
                resampling=Resampling.bilinear,
            )

            # Window to requested bounds using floor/ceil to capture full extent
            window = from_bounds(min_lon, min_lat, max_lon, max_lat, full_transform)
            row_off = math.floor(window.row_off)
            col_off = math.floor(window.col_off)
            row_end = math.ceil(window.row_off + window.height)
            col_end = math.ceil(window.col_off + window.width)

            # Clamp to valid array range
            row_off = max(0, row_off)
            col_off = max(0, col_off)
            row_end = min(mosaic.shape[1], row_end)
            col_end = min(mosaic.shape[2], col_end)

            windowed = mosaic[:, row_off:row_end, col_off:col_end]

            # Compute transform for windowed region
            out_transform = full_transform * rasterio.Affine.translation(col_off, row_off)
        finally:
            for ds in datasets:
                ds.close()

        return {
            "array": windowed,
            "transform": out_transform,
            "crs": crs,
            "nodata": nodata_val,
            "bounds": bounds,
            "bands": None,
        }


# =============================================================================
# DERIVED LAYER LOADERS (computed from parent layers)
# =============================================================================

class DerivedLoader:
    """Base class for layers computed from other loaded layers."""

    def __init__(self, layer_config: "LayerConfig"):
        self.config = layer_config

    def compute(self, parent_data: dict[str, dict]) -> dict:
        """
        Compute derived layer from parent layer data.

        Args:
            parent_data: Dict mapping parent layer name to its loaded data dict
                         (each with keys: array, transform, crs, nodata, bounds)

        Returns:
            dict with keys: array, transform, crs, nodata, bounds, bands
        """
        raise NotImplementedError


def _gradient_meters(elev: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute elevation gradients in meters.

    Uses pixel spacing from the rasterio transform and corrects for
    latitude-dependent longitude spacing.

    Args:
        elev: 2D elevation array (H, W) in meters
        transform: rasterio Affine transform

    Returns:
        (gy, gx) gradients in m/m (rise/run)
    """
    # Pixel spacing in degrees
    dx_deg = abs(transform.a)
    dy_deg = abs(transform.e)

    # Convert to meters (approximate)
    # Use center latitude for longitude correction
    nrows = elev.shape[0]
    center_row = nrows // 2
    # Latitude at center of array
    center_lat = transform.f + center_row * transform.e
    lat_rad = np.radians(center_lat)

    dy_m = dy_deg * 110540.0  # meters per degree latitude
    dx_m = dx_deg * 111320.0 * np.cos(lat_rad)  # meters per degree longitude

    gy, gx = np.gradient(elev, dy_m, dx_m)
    return gy, gx


class SlopeLoader(DerivedLoader):
    """Compute slope in degrees from DEM."""

    def compute(self, parent_data: dict[str, dict]) -> dict:
        dem_data = parent_data["dem"]
        elev = dem_data["array"][0].astype(np.float64)  # (H, W)

        gy, gx = _gradient_meters(elev, dem_data["transform"])
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad).astype(np.float32)

        return {
            "array": slope_deg[np.newaxis],  # (1, H, W)
            "transform": dem_data["transform"],
            "crs": dem_data["crs"],
            "nodata": np.nan,
            "bounds": dem_data["bounds"],
            "bands": ["slope"],
        }


class AspectLoader(DerivedLoader):
    """Compute aspect as (sin, cos) encoding from DEM."""

    def compute(self, parent_data: dict[str, dict]) -> dict:
        dem_data = parent_data["dem"]
        elev = dem_data["array"][0].astype(np.float64)

        gy, gx = _gradient_meters(elev, dem_data["transform"])
        aspect_rad = np.arctan2(-gx, gy)  # radians, north=0 clockwise

        sin_aspect = np.sin(aspect_rad).astype(np.float32)
        cos_aspect = np.cos(aspect_rad).astype(np.float32)

        return {
            "array": np.stack([sin_aspect, cos_aspect], axis=0),  # (2, H, W)
            "transform": dem_data["transform"],
            "crs": dem_data["crs"],
            "nodata": np.nan,
            "bounds": dem_data["bounds"],
            "bands": ["aspect_sin", "aspect_cos"],
        }


class RugosityLoader(DerivedLoader):
    """
    Compute Terrain Ruggedness Index (TRI) from DEM.

    TRI = sqrt(sum((z_neighbor - z_center)^2)) over a 3x3 window.
    """

    def compute(self, parent_data: dict[str, dict]) -> dict:
        dem_data = parent_data["dem"]
        elev = dem_data["array"][0].astype(np.float64)
        H, W = elev.shape

        # Pad with edge values for border pixels
        padded = np.pad(elev, 1, mode="edge")

        # Sum of squared differences over 3x3 neighborhood
        sq_sum = np.zeros((H, W), dtype=np.float64)
        for di in range(3):
            for dj in range(3):
                if di == 1 and dj == 1:
                    continue  # skip center
                neighbor = padded[di:di + H, dj:dj + W]
                sq_sum += (neighbor - elev) ** 2

        tri = np.sqrt(sq_sum).astype(np.float32)

        return {
            "array": tri[np.newaxis],  # (1, H, W)
            "transform": dem_data["transform"],
            "crs": dem_data["crs"],
            "nodata": np.nan,
            "bounds": dem_data["bounds"],
            "bands": ["tri"],
        }


class OrographicUpliftLoader(DerivedLoader):
    """
    Compute orographic uplift: Wo = wind_speed * sin(slope) * cos(wind_dir - aspect).

    Based on Bohrer et al. 2012. Requires slope, aspect, and ERA5 wind components.
    Wind components are broadcast from coarse ERA5 grid to DEM resolution.
    """

    def compute(self, parent_data: dict[str, dict]) -> dict:
        slope_data = parent_data["slope"]
        aspect_data = parent_data["aspect"]
        era5l_data = parent_data["era5_land"]

        slope_rad = np.radians(slope_data["array"][0])  # slope in degrees -> radians

        # Aspect encoded as (sin, cos) -> recover angle
        sin_asp = aspect_data["array"][0]
        cos_asp = aspect_data["array"][1]
        aspect_rad = np.arctan2(sin_asp, cos_asp)

        # ERA5-Land wind components (u10 = index 4, v10 = index 5 in era5_land bands)
        # u_component_of_wind_10m, v_component_of_wind_10m
        u10 = era5l_data["array"][4].astype(np.float64)
        v10 = era5l_data["array"][5].astype(np.float64)

        # ERA5 is coarse - it may be a single pixel or small grid.
        # Take mean to get scalar wind, then broadcast.
        u10_mean = np.nanmean(u10)
        v10_mean = np.nanmean(v10)

        wind_speed = np.sqrt(u10_mean**2 + v10_mean**2)
        wind_dir = np.arctan2(u10_mean, v10_mean)  # meteorological convention

        # Orographic uplift formula
        wo = wind_speed * np.sin(slope_rad) * np.cos(wind_dir - aspect_rad)
        wo = wo.astype(np.float32)

        return {
            "array": wo[np.newaxis],  # (1, H, W)
            "transform": slope_data["transform"],
            "crs": slope_data["crs"],
            "nodata": np.nan,
            "bounds": slope_data["bounds"],
            "bands": ["orographic_uplift"],
        }


class ThermalUpliftLoader(DerivedLoader):
    """
    Compute Deardorff convective velocity scale.

    w* = (g * Q0 * zi / T0)^(1/3)

    Where:
        Q0 = surface_sensible_heat_flux / (rho * cp)
        zi = boundary_layer_height
        T0 = temperature_2m
        g = 9.81 m/s^2
        rho = 1.225 kg/m^3 (sea level air density)
        cp = 1005 J/(kg·K)

    Only computed where Q0 > 0 (convective conditions), else 0.
    """

    def compute(self, parent_data: dict[str, dict]) -> dict:
        era5l_data = parent_data["era5_land"]
        era5_data = parent_data["era5"]

        # ERA5-Land: temperature_2m (index 0), surface_sensible_heat_flux (index 11)
        t2m = era5l_data["array"][0].astype(np.float64)
        shf = era5l_data["array"][11].astype(np.float64)

        # ERA5: boundary_layer_height (index 0)
        blh = era5_data["array"][0].astype(np.float64)

        # Take spatial means (coarse grids, typically 1-few pixels)
        t2m_mean = np.nanmean(t2m)
        shf_mean = np.nanmean(shf)
        blh_mean = np.nanmean(blh)

        # Physical constants
        g = 9.81
        rho = 1.225
        cp = 1005.0

        # ERA5 sensible heat flux is in J/m^2 (accumulated), but hourly images
        # represent W/m^2. The sign convention: negative = upward (toward atmosphere).
        # For convective velocity we need upward flux, so negate.
        q0 = -shf_mean / (rho * cp)

        if q0 > 0 and blh_mean > 0 and t2m_mean > 0:
            w_star = (g * q0 * blh_mean / t2m_mean) ** (1.0 / 3.0)
        else:
            w_star = 0.0

        # Return as single-pixel array matching ERA5 grid
        H, W = era5l_data["array"].shape[1:]
        result = np.full((1, H, W), w_star, dtype=np.float32)

        return {
            "array": result,
            "transform": era5l_data["transform"],
            "crs": era5l_data["crs"],
            "nodata": np.nan,
            "bounds": era5l_data["bounds"],
            "bands": ["thermal_uplift"],
        }


# Map derived layer names to their loader classes
_DERIVED_LOADERS = {
    "slope": SlopeLoader,
    "aspect": AspectLoader,
    "rugosity": RugosityLoader,
    "orographic_uplift": OrographicUpliftLoader,
    "thermal_uplift": ThermalUpliftLoader,
}


def get_loader(layer_name: str, **init_kwargs) -> LayerLoader:
    """
    Factory that creates a loader based on layer config source type.

    Args:
        layer_name: Name of the layer in LAYERS config
        **init_kwargs: Additional arguments for loader initialization
            - chip_dir: Required for ChipLoader
            - tile_dir: Required for TileLoader
            - chip_group: Required for ChipLoader

    Returns:
        LayerLoader instance appropriate for the layer's source type
    """
    from movebench.config import LAYERS

    config = LAYERS[layer_name]

    if config.source == "local_tiff":
        return LocalTiffLoader(config)
    elif config.source == "local_hdf":
        return LocalHdfLoader(config)
    elif config.source == "gee_chip":
        chip_dir = init_kwargs.get("chip_dir")
        chip_group = init_kwargs.get("chip_group")
        if chip_dir is None or chip_group is None:
            raise ValueError("chip_dir and chip_group required for gee_chip layers")
        return ChipLoader(chip_dir, chip_group)
    elif config.source == "gee_tile":
        tile_dir = init_kwargs.get("tile_dir")
        if tile_dir is None:
            raise ValueError("tile_dir required for gee_tile layers")
        return TileLoader(tile_dir, layer_name)
    elif config.source == "derived":
        if layer_name not in _DERIVED_LOADERS:
            raise ValueError(f"No derived loader for layer: {layer_name}")
        return _DERIVED_LOADERS[layer_name](config)
    else:
        raise ValueError(f"Unknown source type: {config.source}")
