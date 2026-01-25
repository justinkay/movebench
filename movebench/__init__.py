"""MoveBench: Animal movement benchmark with environmental context."""

from .config import (
    LAYERS,
    PATHS,
    YEAR_BOUNDS,
    LayerConfig,
    get_layer,
    get_layers,
    compute_group_scale,
    get_band_count,
    get_band_names,
    LON_COL,
    LAT_COL,
    TIME_COL,
    INDIVIDUAL_COL,
)
from .datasets import ChipDataset, SSFDataset, iSSFDataset, grouped_nll
from .transforms import Normalizer

__version__ = "0.2.0"

__all__ = [
    # Config
    "LAYERS",
    "PATHS",
    "YEAR_BOUNDS",
    "LayerConfig",
    "get_layer",
    "get_layers",
    "compute_group_scale",
    "get_band_count",
    "get_band_names",
    "LON_COL",
    "LAT_COL",
    "TIME_COL",
    "INDIVIDUAL_COL",
    # Datasets
    "ChipDataset",
    "SSFDataset",
    "iSSFDataset",
    "grouped_nll",
    # Transforms
    "Normalizer",
]
