"""Raster transformation utilities."""

import os

import numpy as np
import torch
from rasterio.warp import reproject, calculate_default_transform, Resampling


def reproject_all(layers):
    """
    Reproject all layers to match the first layer's CRS and extent.

    Args:
        layers: List of layer dicts with keys: array, transform, crs

    Returns:
        List of numpy arrays aligned to first layer
    """
    ref_raster = layers[0]
    _, H_ref, W_ref = ref_raster['array'].shape

    aligned = [ref_raster['array']]
    dst_crs = ref_raster['crs']
    dst_transform = ref_raster['transform']

    for layer in layers[1:]:
        # create new np array as destination - same channels as this layer but with target width/height
        dst = np.full((layer["array"].shape[0], H_ref, W_ref), np.nan, dtype=np.float32)
        reproject(
            source=layer["array"],
            destination=dst,
            src_transform=layer["transform"],
            src_crs=layer["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        aligned.append(dst)

    return aligned


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
        dst = np.full((C, dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=layer["array"],
            destination=dst,
            src_transform=layer["transform"],
            src_crs=layer["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
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


def compute_valid_mask(arrays, nodata_vals):
    """
    Compute valid data mask from list of arrays and nodata values.

    Args:
        arrays: List of torch tensors
        nodata_vals: List of nodata values (None treated as infinity)

    Returns:
        Concatenated boolean mask tensor
    """
    masks = []
    for arr, nodata in zip(arrays, nodata_vals):
        nodata_val = nodata if nodata is not None else float('inf')
        masks.append(arr != nodata_val)
    return torch.cat(masks)


class Normalizer:
    """
    Z-score normalization with selective channel mask.

    Supports two modes:
    1. Legacy: Load from {study}_mean.pt and {study}_std.pt files
    2. Per-layer: Assemble from per-layer/per-band JSON statistics
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

    @classmethod
    def load(cls, study_id, num_channels, categorical_start_idx, categorical_end_idx):
        """
        Load normalizer from existing mean/std files.

        Args:
            study_id: Study identifier
            num_channels: Total number of channels (e.g., 63)
            categorical_start_idx: Start index of categorical channels to skip
            categorical_end_idx: End index of categorical channels to skip

        Returns:
            Normalizer instance, or None if files don't exist
        """
        mean_path = f'{study_id}_mean.pt'
        std_path = f'{study_id}_std.pt'

        if not os.path.exists(mean_path):
            return None

        mean = torch.load(mean_path)
        std = torch.load(std_path)

        # Build mask: True = normalize, False = skip (categorical)
        normalize_mask = torch.ones(num_channels, dtype=torch.bool)
        normalize_mask[categorical_start_idx:categorical_end_idx] = False

        return cls(mean, std, normalize_mask)

    @classmethod
    def compute_from_dataset(cls, dataset, num_samples=1000, categorical_indices=None):
        """
        Compute normalization statistics from dataset samples.

        Args:
            dataset: ChipDataset instance (without normalization)
            num_samples: Number of samples to use
            categorical_indices: Tuple (start, end) of categorical channel indices to skip

        Returns:
            Normalizer instance
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

        layer_sum = None
        layer_sq_sum = None
        count = None

        for i_batch, (x, valid_mask, _, _) in tqdm(enumerate(dataloader), total=num_samples):
            x = x.squeeze(0)
            valid_mask = valid_mask.squeeze(0)
            num_valid = valid_mask.sum(dim=(-1, -2))
            x_clean = torch.where(valid_mask, x, torch.zeros_like(x))

            s = x_clean.sum(dim=(-1, -2))
            s2 = (x_clean ** 2).sum(dim=(-1, -2))

            if layer_sum is None:
                layer_sum = s
                layer_sq_sum = s2
                count = num_valid
            else:
                layer_sum += s
                layer_sq_sum += s2
                count += num_valid

            if i_batch >= num_samples:
                break

        mean = layer_sum / count
        var = torch.clamp(layer_sq_sum / count - mean**2, min=0.0)
        std = torch.sqrt(var)

        # Build normalize mask
        num_channels = mean.shape[0]
        normalize_mask = torch.ones(num_channels, dtype=torch.bool)
        if categorical_indices:
            normalize_mask[categorical_indices[0]:categorical_indices[1]] = False

        return cls(mean, std, normalize_mask)

    def save(self, study_id):
        """Save mean and std to files."""
        torch.save(self.mean, f'{study_id}_mean.pt')
        torch.save(self.std, f'{study_id}_std.pt')
