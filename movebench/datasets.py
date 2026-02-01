"""PyTorch datasets for movement ecology."""

import numpy as np
import torch
from torch.utils.data import Dataset

from movebench.config import PATHS, LAYERS, POINT_ID_COL
from movebench.transforms import reproject_to_meters, one_hot_encode
from movebench.layer_pipeline import LayerPipeline


class ChipDataset(Dataset):
    """
    Dataset loading multi-layer environmental chips at GPS points.

    Each sample returns:
        - tensor of shape (C, H, W) with all environmental layers
        - valid mask of shape (C, H, W) indicating non-nodata pixels
        - rasterio transform for the chip
        - CRS of the chip
    """

    def __init__(self, study_id, idx_to_timestamp, split, study_dir=None, normalizer=None):
        """
        Initialize ChipDataset.

        Args:
            study_id: Study identifier (UUID)
            idx_to_timestamp: DataFrame with columns ['index', 'timestamp', 'point_id']
            split: Data split ("train" or "test")
            study_dir: Study directory (default: PATHS["studies"] / study_id)
            normalizer: Optional pre-computed Normalizer instance. If provided,
                       skip auto-loading from files.
        """
        self.study_id = study_id
        self.idx_to_timestamp = idx_to_timestamp.reset_index(drop=True)
        self.split = split

        # Set up directory paths
        study_dir = study_dir or (PATHS["studies"] / study_id)
        self.study_dir = study_dir
        self.chip_dir = study_dir / "chips" / split
        self.dem_dir = study_dir / "tiles" / "dem"

        # Initialize loaders
        self._pipeline = LayerPipeline(self.study_dir, self.chip_dir)
        self._layer_order = self._pipeline.layer_order

        # Normalizer: use provided one or None (should be provided by Study)
        self.normalizer = normalizer
        self._normalizer_provided = normalizer is not None
        self._lc_nc = LAYERS["modis_lc"].num_classes
        self._categorical_start_idx = None

    def __getitem__(self, idx):
        """Load and process all layers for a single sample."""
        context = self._build_context(idx)
        all_layers, layer_names = self._load_all_layers(context)
        final_t, valid_masks, final_transform, final_crs = self._apply_transforms(all_layers, layer_names)

        if self.normalizer is not None:
            final_t = self.normalizer(final_t)

        return final_t, torch.cat(valid_masks), final_transform, final_crs

    def _build_context(self, idx) -> dict:
        """Build context dict with sample metadata."""
        row = self.idx_to_timestamp.iloc[idx]
        return {
            "point_id": int(row[POINT_ID_COL]),
            "timestamp": row['timestamp'],
        }

    def _load_all_layers(self, context: dict) -> tuple[list[dict], list[str]]:
        """Load all layers for a sample using configured loaders."""
        return self._pipeline.load_all(context)

    def _apply_transforms(self, all_layers: list[dict], layer_names: list[str]) -> tuple:
        """Reproject layers, encode categoricals, compute masks."""
        # Reproject all to same resolution/CRS/bounds
        reprojected, final_transform, final_crs = reproject_to_meters(all_layers)
        reprojected_t = [torch.from_numpy(layer) for layer in reprojected]

        def _compute_nodata_mask(layer: torch.Tensor, nodata_val):
            if nodata_val is None:
                return torch.isfinite(layer)
            if isinstance(nodata_val, float) and np.isnan(nodata_val):
                return ~torch.isnan(layer)
            return layer != nodata_val

        # Convert land cover to one-hot
        lc_idx = self._layer_order.index("modis_lc")
        lc_nc = LAYERS["modis_lc"].num_classes
        lc_nodata_val = all_layers[lc_idx]['nodata']
        lc_valid_mask = _compute_nodata_mask(reprojected_t[lc_idx], lc_nodata_val)
        reprojected_t[lc_idx] = one_hot_encode(reprojected_t[lc_idx], lc_nc)

        # Compute nodata masks (handle NaN nodata values specially)
        valid_masks = []
        for layer, layer_dict in zip(reprojected_t, all_layers):
            nodata_val = layer_dict['nodata']
            valid_masks.append(_compute_nodata_mask(layer, nodata_val))
        valid_masks[lc_idx] = lc_valid_mask.repeat(lc_nc, 1, 1)

        # Apply valid_range constraints from config
        for i, name in enumerate(layer_names):
            if name in LAYERS:
                valid_range = LAYERS[name].valid_range
                if valid_range is not None:
                    lo, hi = valid_range
                    valid_masks[i] = valid_masks[i] & (reprojected_t[i] >= lo) & (reprojected_t[i] <= hi)

        # Concatenate and set up normalization mask
        final_t = torch.cat(reprojected_t)
        lc_start_idx = sum(a['array'].shape[0] for a in all_layers[:lc_idx])

        if self._categorical_start_idx is None:
            num_channels = final_t.shape[0]
            self._categorical_start_idx = lc_start_idx

            if self.normalizer is not None:
                self.norm_mask = self.normalizer.normalize_mask
            else:
                self.norm_mask = torch.ones(num_channels, dtype=torch.bool)
                self.norm_mask[lc_start_idx:lc_start_idx + self._lc_nc] = False

        return final_t, valid_masks, final_transform, final_crs

    def __len__(self):
        return len(self.idx_to_timestamp)
