"""
Centralized configuration for MoveBench data pipeline.

This module defines:
- Layer configurations (bands, resolution, source type)
- Directory paths (with environment variable overrides)
- Constants for scale/resolution computation
- Chip groupings for GEE downloads
"""

from dataclasses import dataclass, field
from typing import Literal
from pathlib import Path
import os


# =============================================================================
# DIRECTORY STRUCTURE
# =============================================================================

# Root can be overridden via environment variable
DATA_ROOT = Path(os.environ.get("MOVEBENCH_DATA_ROOT", Path(__file__).parent.parent))

# Shared data (global layers used across all studies)
SHARED_ROOT = Path(os.environ.get(
    "MOVEBENCH_SHARED_ROOT",
    "/archive/vision/beery/movebench"
))

# Directory structure within DATA_ROOT
PATHS = {
    # Per-study data
    "studies": DATA_ROOT / "studies",

    # Shared/global layers
    "modis_lc": SHARED_ROOT / "modis_lc2015-2025",
    "modis_cvf": SHARED_ROOT / "modis_cvf2015-2025",
    "hii": SHARED_ROOT / "human_impact_index",
    "bioclim": SHARED_ROOT / "bioclim",

    # MODIS sinusoidal grid definition file
    "modis_grid": DATA_ROOT / "sn_bound_10deg.txt",
}

# File naming patterns
FILE_PATTERNS = {
    "trajectories": "trajectories.csv",
    "manifest": "manifest.json",
    "chip": "chips/{split}/{point_id:05d}_{layer}.tif",
    "tile": "tiles/{layer}/tile_{tile_id:03d}.tif",
    "tile_index": "tiles/{layer}/index.json",
}


# =============================================================================
# DATAFRAME COLUMN NAMES (Movebank convention)
# =============================================================================

LON_COL = "location-long"
LAT_COL = "location-lat"
TIME_COL = "timestamp"
INDIVIDUAL_COL = "individual-local-identifier"
SET_COL = "set"
POINT_ID_COL = "point_id"


# =============================================================================
# SCALE / RESOLUTION SETTINGS
# =============================================================================

# Constraints for chip pixel dimensions
MIN_PIXELS = 3
MAX_PIXELS = 100

# Target CRS for reprojection (metric, for consistent pixel sizes)
TARGET_CRS = "EPSG:3857"  # Web Mercator

# Region multipliers for train vs test
# Test regions are larger to allow simulation/forecasting
REGION_MULTIPLIER = {
    "train": 1.0,   # 2x 95th percentile step size (from buffer computation)
    "test": 15.0,   # 30x 95th percentile step size
}

# GEE download settings
GEE_MAX_DOWNLOAD_SIZE_MB = 32
GEE_MAX_GRID_DIMENSION = 10000

# Year bounds for temporal layers (clip timestamps to these ranges)
YEAR_BOUNDS = {
    "modis_lc": (2015, 2025),
    "modis_cvf": (2015, 2025),
    "hii": (2015, 2020),
}


# =============================================================================
# LAYER CONFIGURATION
# =============================================================================

@dataclass
class LayerConfig:
    """
    Configuration for a single raster layer or layer group.

    Attributes:
        name: Unique identifier for this layer
        source: Where the data comes from
            - "gee_chip": Downloaded from GEE per GPS point
            - "gee_tile": Downloaded from GEE as tiles covering study extent
            - "local_hdf": Local HDF files (e.g., MODIS products)
            - "local_tiff": Local GeoTIFF files
        bands: List of band names in this layer
        native_scale_m: Native resolution in meters
        temporal: Temporal resolution
            - "hourly", "daily", "8day", "16day", "annual", "static"
        gee_collection: Earth Engine collection ID (for GEE sources)
        path_pattern: File path pattern (for local sources)
        categorical: If True, values are class labels (apply one-hot encoding)
        num_classes: Number of classes (for categorical layers)
        normalize: Whether to include in z-score normalization
        valid_range: (min, max) tuple for valid data; values outside are masked
        scale_factor: Multiply raw values by this (e.g., 0.02 for MODIS LST)
    """
    name: str
    source: Literal["gee_chip", "gee_tile", "local_hdf", "local_tiff"]
    bands: list[str]
    native_scale_m: float
    temporal: Literal["hourly", "daily", "8day", "16day", "annual", "static"]
    gee_collection: str = ""
    path_pattern: str = ""
    categorical: bool = False
    num_classes: int = 0
    normalize: bool = True
    valid_range: tuple[float, float] | None = None  # (min, max) inclusive; values outside are masked
    scale_factor: float = 1.0


# -----------------------------------------------------------------------------
# GEE Chip Layers (downloaded per GPS point, high temporal resolution)
# -----------------------------------------------------------------------------

ERA5_LAND = LayerConfig(
    name="era5_land",
    source="gee_chip",
    bands=[
        "temperature_2m",
        "dewpoint_temperature_2m",
        "total_precipitation",
        "surface_pressure",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
    ],
    native_scale_m=11132.0,
    temporal="hourly",
    gee_collection="ECMWF/ERA5_LAND/HOURLY",
)

ERA5 = LayerConfig(
    name="era5",
    source="gee_chip",
    bands=[
        "boundary_layer_height",
        "total_cloud_cover",
        "mean_sea_level_pressure",
    ],
    native_scale_m=27830.0,  # ~0.25 degree
    temporal="hourly",
    gee_collection="ECMWF/ERA5/HOURLY",
)

MODIS_LST = LayerConfig(
    name="modis_lst",
    source="gee_chip",
    bands=[
        "LST_Day_1km",
        "LST_Night_1km",
    ],
    native_scale_m=1000.0,
    temporal="8day",
    gee_collection="MODIS/061/MOD11A2",
    scale_factor=0.02,  # Convert to Kelvin
)

MODIS_VI = LayerConfig(
    name="modis_vi",
    source="gee_chip",
    bands=[
        "NDVI",
        "EVI",
    ],
    native_scale_m=250.0,
    temporal="16day",
    gee_collection="MODIS/061/MOD13Q1",
    scale_factor=0.0001,  # Scale factor from MODIS docs
)

MODIS_NDSI = LayerConfig(
    name="modis_ndsi",
    source="gee_chip",
    bands=["NDSI"],
    native_scale_m=500.0,
    temporal="daily",  # Derived from daily MCD43A4
    gee_collection="MODIS/MCD43A4_006_NDSI",
)

MODIS_BAI = LayerConfig(
    name="modis_bai",
    source="gee_chip",
    bands=["BAI"],
    native_scale_m=500.0,
    temporal="daily",
    gee_collection="MODIS/MCD43A4_006_BAI",
)

# Dynamic World (currently disabled in original code)
DYNAMIC_WORLD = LayerConfig(
    name="dynamic_world",
    source="gee_chip",
    bands=[
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ],
    native_scale_m=10.0,
    temporal="daily",
    gee_collection="GOOGLE/DYNAMICWORLD/V1",
    normalize=False,  # Probabilities already 0-1
)


# -----------------------------------------------------------------------------
# GEE Tile Layers (downloaded as tiles covering study extent)
# -----------------------------------------------------------------------------

DEM = LayerConfig(
    name="dem",
    source="gee_tile",
    bands=["elevation"],
    native_scale_m=30.0,
    temporal="static",
    gee_collection="COPERNICUS/DEM/GLO30",
)


# -----------------------------------------------------------------------------
# Local HDF Layers (MODIS products from local archive)
# -----------------------------------------------------------------------------

MODIS_LC = LayerConfig(
    name="modis_lc",
    source="local_hdf",
    bands=["LC_Type1"],
    native_scale_m=500.0,
    temporal="annual",
    path_pattern=str(PATHS["modis_lc"] / "MCD12Q1.A{year}001.h{h:02d}v{v:02d}.061.*.hdf"),
    categorical=True,
    num_classes=18,  # 17 classes + 1 for invalid/masked
    normalize=False,
)

MODIS_CVF_TREE = LayerConfig(
    name="modis_cvf_tree",
    source="local_hdf",
    bands=["Percent_Tree_Cover"],
    native_scale_m=250.0,
    temporal="annual",
    path_pattern=str(PATHS["modis_cvf"] / "MOD44B.A{year}001.h{h:02d}v{v:02d}.061.*.hdf"),
    valid_range=(0, 100),
)

MODIS_CVF_NONTREE = LayerConfig(
    name="modis_cvf_nontree",
    source="local_hdf",
    bands=["Percent_NonTree_Vegetation"],
    native_scale_m=250.0,
    temporal="annual",
    path_pattern=str(PATHS["modis_cvf"] / "MOD44B.A{year}001.h{h:02d}v{v:02d}.061.*.hdf"),
    valid_range=(0, 100),
)

MODIS_CVF_NONVEG = LayerConfig(
    name="modis_cvf_nonveg",
    source="local_hdf",
    bands=["Percent_NonVegetated"],
    native_scale_m=250.0,
    temporal="annual",
    path_pattern=str(PATHS["modis_cvf"] / "MOD44B.A{year}001.h{h:02d}v{v:02d}.061.*.hdf"),
    valid_range=(0, 100),
)


# -----------------------------------------------------------------------------
# Local GeoTIFF Layers
# -----------------------------------------------------------------------------

HII = LayerConfig(
    name="hii",
    source="local_tiff",
    bands=["hii"],
    native_scale_m=1000.0,  # ~1km
    temporal="annual",
    path_pattern=str(PATHS["hii"] / "hii_{year}-01-01.tif"),
)

BIOCLIM = LayerConfig(
    name="bioclim",
    source="local_tiff",
    bands=[f"bio_{i}" for i in range(1, 20)],  # 19 bioclimatic variables
    native_scale_m=1000.0,  # 30 arc-seconds ~ 1km
    temporal="static",
    path_pattern=str(PATHS["bioclim"] / "wc2.1_30s_bio_{band_num}.tif"),
)


# =============================================================================
# LAYER REGISTRY
# =============================================================================

# All available layers
LAYERS: dict[str, LayerConfig] = {
    # GEE chips
    "era5_land": ERA5_LAND,
    "era5": ERA5,
    "modis_lst": MODIS_LST,
    "modis_vi": MODIS_VI,
    "modis_ndsi": MODIS_NDSI,
    "modis_bai": MODIS_BAI,
    "dynamic_world": DYNAMIC_WORLD,

    # GEE tiles
    "dem": DEM,

    # Local HDF
    "modis_lc": MODIS_LC,
    "modis_cvf_tree": MODIS_CVF_TREE,
    "modis_cvf_nontree": MODIS_CVF_NONTREE,
    "modis_cvf_nonveg": MODIS_CVF_NONVEG,

    # Local GeoTIFF
    "hii": HII,
    "bioclim": BIOCLIM,
}


# -----------------------------------------------------------------------------
# Chip Groupings for GEE Downloads
# -----------------------------------------------------------------------------
# These define which layers are downloaded together in a single GEE request.
# Grouping is based on similar native resolution to minimize resampling.

CHIP_GROUPS = {
    # "Big" group: coarse resolution atmospheric data (~11km)
    "era5": {
        "layers": ["era5_land", "era5"],
        "target_scale_m": 11132.0,
    },

    # "Medium" group: MODIS products (~250m-1km)
    "modis": {
        "layers": ["modis_lst", "modis_vi", "modis_ndsi", "modis_bai"],
        "target_scale_m": 250.0,
    },

    # "Small" group: high-res land cover (~10-30m) - currently disabled
    # "landcover": {
    #     "layers": ["dynamic_world"],
    #     "target_scale_m": 30.0,
    # },
}


# -----------------------------------------------------------------------------
# Default Layer Sets
# -----------------------------------------------------------------------------

# Layers to load by default (all currently working layers)
DEFAULT_LAYERS = [
    # GEE chips
    "era5_land",
    "era5",
    "modis_lst",
    "modis_vi",
    "modis_ndsi",
    "modis_bai",

    # GEE tiles
    "dem",

    # Local layers
    "modis_lc",
    "modis_cvf_tree",
    "modis_cvf_nontree",
    "modis_cvf_nonveg",
    "hii",
    "bioclim",
]

# -----------------------------------------------------------------------------
# Canonical Layer Loading Order
# -----------------------------------------------------------------------------
# This defines the canonical order for loading layers, which determines
# channel indices. Important for computing categorical indices deterministically.
# This order matches ChipDataset's actual loading order.

LAYER_ORDER = [
    # Medium chip (MODIS products at ~250m)
    "modis_lst",         # 2 bands  → channels 0-1
    "modis_vi",          # 2 bands  → channels 2-3
    "modis_ndsi",        # 1 band   → channel 4
    "modis_bai",         # 1 band   → channel 5
    # Big chip (ERA5 at ~11km)
    "era5_land",         # 12 bands → channels 6-17
    "era5",              # 3 bands  → channels 18-20
    # Other layers
    "dem",               # 1 band   → channel 21
    "modis_lc",          # 1→18 (one-hot) → channels 22-39
    "modis_cvf_tree",    # 1 band   → channel 40
    "modis_cvf_nontree", # 1 band   → channel 41
    "modis_cvf_nonveg",  # 1 band   → channel 42
    "hii",               # 1 band   → channel 43
    "bioclim",           # 19 bands → channels 44-62
]

# Minimal layer set for quick testing
MINIMAL_LAYERS = [
    "era5_land",
    "dem",
    "modis_lc",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_layer(name: str) -> LayerConfig:
    """Get a layer configuration by name."""
    if name not in LAYERS:
        raise ValueError(f"Unknown layer: {name}. Available: {list(LAYERS.keys())}")
    return LAYERS[name]


def get_layers(names: list[str]) -> list[LayerConfig]:
    """Get multiple layer configurations by name."""
    return [get_layer(name) for name in names]


def get_chip_group_layers(group_name: str) -> list[LayerConfig]:
    """Get all layers in a chip download group."""
    if group_name not in CHIP_GROUPS:
        raise ValueError(f"Unknown chip group: {group_name}. Available: {list(CHIP_GROUPS.keys())}")
    return get_layers(CHIP_GROUPS[group_name]["layers"])


def get_study_dir(study_id: str) -> Path:
    """Get the directory path for a study."""
    return PATHS["studies"] / study_id


def get_chip_path(study_id: str, split: str, point_id: int, layer_group: str) -> Path:
    """Get the path for a chip file."""
    return (
        get_study_dir(study_id)
        / "chips"
        / split
        / f"{point_id:05d}_{layer_group}.tif"
    )


def get_tile_dir(study_id: str, layer: str) -> Path:
    """Get the directory for tiles of a specific layer."""
    return get_study_dir(study_id) / "tiles" / layer


def compute_group_scale(
    buffer_km: float,
    native_scale_m: float,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> float:
    """
    Compute the download scale for a layer group.

    Ensures chip has between min_pixels and max_pixels per side while
    staying as close to native resolution as possible.

    Args:
        buffer_km: Buffer radius in kilometers (half the chip width)
        native_scale_m: Native resolution of the layer in meters
        min_pixels: Minimum pixels per side
        max_pixels: Maximum pixels per side

    Returns:
        Scale in meters per pixel
    """
    R_m = buffer_km * 1000.0

    # Coarsest allowed scale (fewest pixels)
    scale_for_max = (2.0 * R_m) / max_pixels

    # Finest allowed scale (most pixels)
    scale_for_min = (2.0 * R_m) / min_pixels

    # Clamp native scale to allowed range
    if native_scale_m < scale_for_max:
        # Native is finer than we can afford - coarsen
        return scale_for_max
    elif native_scale_m > scale_for_min:
        # Native is coarser than minimum - upsample
        return scale_for_min
    else:
        # Native is within range - use it
        return native_scale_m


def get_band_count(layer_names: list[str]) -> int:
    """Get total number of bands across layers (accounting for one-hot encoding)."""
    total = 0
    for name in layer_names:
        cfg = get_layer(name)
        if cfg.categorical:
            total += cfg.num_classes
        else:
            total += len(cfg.bands)
    return total


def get_band_names(layer_names: list[str]) -> list[str]:
    """Get list of all band names across layers."""
    names = []
    for layer_name in layer_names:
        cfg = get_layer(layer_name)
        if cfg.categorical:
            names.extend([f"{layer_name}_class_{i}" for i in range(cfg.num_classes)])
        else:
            names.extend([f"{layer_name}_{band}" for band in cfg.bands])
    return names


def get_channel_indices(layer_order: list[str] = None) -> dict[str, tuple[int, int]]:
    """
    Compute (start, end) channel indices for each layer.

    Args:
        layer_order: List of layer names in order. Defaults to LAYER_ORDER.

    Returns:
        Dict mapping layer name to (start_index, end_index) tuple.
    """
    if layer_order is None:
        layer_order = LAYER_ORDER

    indices = {}
    offset = 0
    for name in layer_order:
        layer = LAYERS[name]
        if layer.categorical:
            n_channels = layer.num_classes
        else:
            n_channels = len(layer.bands)
        indices[name] = (offset, offset + n_channels)
        offset += n_channels
    return indices


def get_categorical_indices(layer_order: list[str] = None) -> list[tuple[int, int]]:
    """
    Get (start, end) channel indices for all categorical layers.

    Args:
        layer_order: List of layer names in order. Defaults to LAYER_ORDER.

    Returns:
        List of (start_index, end_index) tuples for categorical layers.
    """
    if layer_order is None:
        layer_order = LAYER_ORDER

    channel_indices = get_channel_indices(layer_order)
    result = []
    for name in layer_order:
        layer = LAYERS[name]
        if layer.categorical:
            result.append(channel_indices[name])
    return result


def get_total_channels(layer_order: list[str] = None) -> int:
    """
    Get total number of channels for the given layer order.

    Args:
        layer_order: List of layer names in order. Defaults to LAYER_ORDER.

    Returns:
        Total number of channels.
    """
    if layer_order is None:
        layer_order = LAYER_ORDER
    return get_band_count(layer_order)
