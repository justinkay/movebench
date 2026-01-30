"""Raster transformation utilities."""

import os
from pathlib import Path

import numpy as np
import torch
from rasterio.warp import reproject, calculate_default_transform, Resampling


def reproject_to_meters(layers, dst_crs="EPSG:3857"):
    """
    Reproject all layers to a metric CRS, aligned to first layer's bounds.

    Args:
        layers: List of layer dicts with keys: array, transform, crs, bounds
        dst_crs: Target CRS (default: Web Mercator)

    Returns:
        Tuple of (aligned_arrays, dst_transform, dst_crs)
    """
    ref_raster = layers[0]
    _, H_ref, W_ref = ref_raster['array'].shape

    # compute transform/shape in metric CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        ref_raster["crs"],     # source CRS (lat/lon)
        dst_crs,               # target CRS (meters)
        W_ref, H_ref,
        *ref_raster['bounds'],
    )

    aligned = []
    for layer in layers:
        C = layer["array"].shape[0]

        # Get source nodata value
        src_nodata = layer.get("nodata")

        # Use layer's nodata as dst_nodata if defined (preserves categorical nodata),
        # otherwise use np.nan for continuous layers without defined nodata
        if src_nodata is not None and not (isinstance(src_nodata, float) and np.isnan(src_nodata)):
            dst_nodata = src_nodata
        else:
            dst_nodata = np.nan

        dst = np.full((C, dst_height, dst_width), dst_nodata, dtype=np.float32)

        reproject(
            source=layer["array"],
            destination=dst,
            src_transform=layer["transform"],
            src_crs=layer["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src_nodata if src_nodata is not None else np.nan,
            dst_nodata=dst_nodata,
        )
        aligned.append(dst)

    return aligned, dst_transform, dst_crs


def one_hot_encode(arr, num_classes):
    """
    One-hot encode categorical raster.

    Args:
        arr: Numpy array or torch tensor with integer class labels
        num_classes: Number of classes

    Returns:
        One-hot encoded tensor of shape (num_classes, H, W)
    """
    arr_t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    arr_t = arr_t.long()
    arr_t[arr_t >= num_classes] = 0
    return torch.nn.functional.one_hot(arr_t, num_classes).squeeze(0).permute(2, 0, 1)


class Normalizer:
    """
    Z-score normalization with selective channel mask.

    Assembles from per-layer/per-band JSON statistics.
    """

    def __init__(self, mean, std, normalize_mask):
        """
        Initialize normalizer.

        Args:
            mean: Tensor of per-channel means
            std: Tensor of per-channel standard deviations
            normalize_mask: Boolean tensor, True for channels to normalize
        """
        self.mean = mean
        self.std = std
        self.normalize_mask = normalize_mask

    @classmethod
    def from_per_layer_stats(cls, stats: dict, layer_order: list[str]):
        """
        Assemble normalizer from per-layer/per-band statistics.

        Args:
            stats: Dict mapping layer_name -> {"bands": {band_name: {"mean", "std"}}}
                   or None for categorical layers
            layer_order: List of layer names in order

        Returns:
            Normalizer instance
        """
        from movebench.config import LAYERS

        means = []
        stds = []
        mask = []

        for layer_name in layer_order:
            layer = LAYERS[layer_name]
            layer_stats = stats.get(layer_name)

            if layer.categorical:
                # Categorical layers: don't normalize
                n = layer.num_classes
                means.extend([0.0] * n)
                stds.extend([1.0] * n)
                mask.extend([False] * n)
            elif layer_stats is not None:
                # Assemble in band order from config
                for band in layer.bands:
                    band_stats = layer_stats.get("bands", {}).get(band)
                    if band_stats:
                        means.append(band_stats["mean"])
                        stds.append(band_stats["std"])
                    else:
                        # Missing band: identity transform
                        means.append(0.0)
                        stds.append(1.0)
                    mask.append(True)
            else:
                # Missing layer: identity transform for all bands
                n = len(layer.bands)
                means.extend([0.0] * n)
                stds.extend([1.0] * n)
                mask.extend([True] * n)

        return cls(
            torch.tensor(means, dtype=torch.float32),
            torch.tensor(stds, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    @staticmethod
    def load_per_layer_stats(path: str) -> dict:
        """
        Load per-layer stats from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Dict of per-layer statistics, or empty dict if file doesn't exist
        """
        import json
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    @staticmethod
    def save_per_layer_stats(stats: dict, path: str):
        """
        Save per-layer stats to JSON file.

        Args:
            stats: Dict of per-layer statistics
            path: Path to JSON file
        """
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)

    def __call__(self, x):
        """
        Apply normalization to tensor x.

        Args:
            x: Tensor of shape (C, H, W) or (C, N)

        Returns:
            Normalized tensor (categorical channels unchanged)
        """
        if x.dim() == 3:
            normed = (x - self.mean[:, None, None]) / self.std[:, None, None]
            return torch.where(self.normalize_mask[:, None, None], normed, x)
        elif x.dim() == 2:
            normed = (x - self.mean[:, None]) / self.std[:, None]
            return torch.where(self.normalize_mask[:, None], normed, x)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {x.dim()}D")
