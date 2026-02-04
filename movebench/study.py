"""Study abstraction for movement ecology datasets."""

import json
from pathlib import Path

import pandas as pd

from movebench.config import (
    PATHS,
    TIME_COL,
    SET_COL,
    POINT_ID_COL,
    LAYER_ORDER,
    LAYERS,
    get_channel_indices,
)


class Study:
    """
    Represents one movement study with its data and metadata.

    A Study object encapsulates all data for one animal tracking study,
    providing a clean API for loading trajectories and creating datasets.

    Example:
        >>> study = Study.load("5a280544-06db-4c85-9713-842547972edd")
        >>> train_ds = study.chip_dataset(split="train")
        >>> ssf_ds = study.ssf_dataset(split="train")
    """

    def __init__(self, study_id: str):
        """
        Initialize a Study instance.

        Args:
            study_id: Study identifier (UUID)
        """
        self.study_id = study_id
        self._trajectories = None
        self._manifest = None
        self._normalizer = None

        # Resolve paths
        self._study_dir = self._find_study_dir()

    def _find_study_dir(self) -> Path:
        """Find study directory."""
        study_path = PATHS["studies"] / self.study_id
        if not study_path.exists():
            raise FileNotFoundError(
                f"Study directory not found: {study_path}. "
                f"Available studies: {', '.join(Study.list_available())}"
            )
        return study_path

    @classmethod
    def load(cls, study_id: str) -> "Study":
        """
        Load a study by ID.

        Args:
            study_id: Study identifier (UUID)

        Returns:
            Study instance
        """
        return cls(study_id)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available study IDs.

        Returns:
            Sorted list of study ID strings
        """
        studies = []
        if PATHS["studies"].exists():
            studies.extend([d.name for d in PATHS["studies"].iterdir() if d.is_dir()])
        return sorted(studies)

    @property
    def trajectories(self) -> pd.DataFrame:
        """
        Lazy-load trajectory data.

        Returns:
            DataFrame with parsed timestamps and NaN rows dropped
        """
        if self._trajectories is None:
            self._trajectories = self._load_trajectories_raw()
        return self._trajectories

    def _load_trajectories_raw(self) -> pd.DataFrame:
        """Load raw trajectory CSV with timestamp parsing."""
        csv_path = self._study_dir / "trajectories.csv"
        df = pd.read_csv(csv_path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
        df = df.dropna(subset=[TIME_COL])
        return df

    @property
    def manifest(self) -> dict | None:
        """
        Lazy-load manifest metadata.

        Returns:
            Manifest dict or None if not available
        """
        if self._manifest is None and self._study_dir:
            manifest_path = self._study_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    self._manifest = json.load(f)
        return self._manifest

    def get_split(self, split: str) -> pd.DataFrame:
        """
        Get trajectory data for a specific split.

        Args:
            split: "train" or "test"

        Returns:
            DataFrame filtered to the requested split
        """
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        return self.trajectories[self.trajectories[SET_COL] == split].copy()

    def chip_dataset(self, split: str = "train", auto_normalize: bool = True, force_normalize: bool = False, **kwargs):
        """
        Create ChipDataset for this study.

        Args:
            split: "train" or "test"
            auto_normalize: If True, ensure normalization exists and apply it.
            force_normalize: If True, recompute normalization statistics.
            **kwargs: Additional arguments passed to ChipDataset

        Returns:
            ChipDataset instance
        """
        from movebench.datasets import ChipDataset

        if auto_normalize:
            self._ensure_normalization(force=force_normalize)
            kwargs.setdefault("normalizer", self._normalizer) # can override with kwargs if desired

        df = self.get_split(split)

        # Build idx_to_timestamp DataFrame with index, timestamp, and point_id
        idx_to_timestamp = pd.DataFrame({
            'index': df.index,
            'timestamp': pd.to_datetime(df[TIME_COL]),
            POINT_ID_COL: df[POINT_ID_COL],
        })

        return ChipDataset(
            study_id=self.study_id,
            idx_to_timestamp=idx_to_timestamp,
            split=split,
            study_dir=self._study_dir,
            **kwargs
        )

    def _ensure_normalization(self, num_samples: int = 1000, force: bool = False):
        """
        Ensure normalization exists for all layers/bands, computing missing ones.

        Args:
            num_samples: Number of samples to use for computing statistics
            force: If True, recompute all layer statistics even if they exist
        """
        from movebench.transforms import Normalizer

        stats_path = self._study_dir / "normalization" / "stats.json"
        stats = {} if force else Normalizer.load_per_layer_stats(stats_path)

        # Find layers that need computation (missing or have missing bands)
        needs_compute = []
        for name in LAYER_ORDER:
            layer = LAYERS[name]
            if layer.categorical:
                continue  # Skip categorical layers

            if force:
                needs_compute.append(name)
            else:
                layer_stats = stats.get(name)
                if layer_stats is None:
                    needs_compute.append(name)
                else:
                    stored_bands = set(layer_stats.get("bands", {}).keys())
                    current_bands = set(layer.bands)
                    if not current_bands.issubset(stored_bands):
                        needs_compute.append(name)

        if needs_compute:
            print(f"[movebench] Computing normalization for {len(needs_compute)} layers: {needs_compute}")
            new_stats = self._compute_layer_stats(needs_compute, num_samples)
            stats.update(new_stats)
            Normalizer.save_per_layer_stats(stats, stats_path)

        self._normalizer = Normalizer.from_per_layer_stats(stats, LAYER_ORDER)

    def _compute_layer_stats(self, layer_names: list[str], num_samples: int) -> dict:
        """
        Compute normalization statistics for specific layers.

        Args:
            layer_names: List of layer names to compute statistics for
            num_samples: Number of samples to use

        Returns:
            Dict mapping layer_name -> {"bands": {band_name: {"mean", "std"}}}
        """
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        # Create dataset without normalization
        ds = self.chip_dataset(split="train", auto_normalize=False)

        # Sample and accumulate statistics
        indices = get_channel_indices()
        layer_sums = {name: None for name in layer_names}
        layer_sq_sums = {name: None for name in layer_names}
        layer_counts = {name: None for name in layer_names}

        dataloader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2)
        for i, (x, valid_mask, _, _) in tqdm(enumerate(dataloader), total=num_samples, desc="Computing stats"):
            if i >= num_samples:
                break

            x = x.squeeze(0)
            valid_mask = valid_mask.squeeze(0)

            for name in layer_names:
                start, end = indices[name]
                layer_x = x[start:end]
                layer_valid = valid_mask[start:end]

                # Masked sum
                x_clean = torch.where(layer_valid, layer_x, torch.zeros_like(layer_x))
                s = x_clean.sum(dim=(-1, -2))
                s2 = (x_clean ** 2).sum(dim=(-1, -2))
                count = layer_valid.sum(dim=(-1, -2))

                if layer_sums[name] is None:
                    layer_sums[name] = s
                    layer_sq_sums[name] = s2
                    layer_counts[name] = count
                else:
                    layer_sums[name] += s
                    layer_sq_sums[name] += s2
                    layer_counts[name] += count

        # Compute mean/std for each layer, storing per-band
        result = {}
        for name in layer_names:
            layer = LAYERS[name]
            mean = layer_sums[name] / layer_counts[name]
            var = torch.clamp(layer_sq_sums[name] / layer_counts[name] - mean**2, min=0)
            std = torch.sqrt(var)

            # Store per-band with band names
            result[name] = {
                "bands": {
                    band: {"mean": m.item(), "std": s.item()}
                    for band, m, s in zip(layer.bands, mean, std)
                }
            }

        return result

    def ssf_dataset(self, split: str = "train", auto_normalize: bool = True, force_normalize: bool = False, n_workers: int = 1, **kwargs):
        """
        Create SSFDataset for this study.

        Args:
            split: "train" or "test"
            auto_normalize: If True, ensure normalization exists and apply it.
            force_normalize: If True, recompute normalization statistics.
            n_workers: Number of parallel workers for feature extraction (default 1)
            **kwargs: Additional arguments passed to SSFDataset

        Returns:
            SSFDataset instance
        """
        from movebench.ssf import SSFDataset

        if auto_normalize:
            self._ensure_normalization(force=force_normalize)
            kwargs.setdefault("normalizer", self._normalizer)

        df = self.get_split(split)
        return SSFDataset(
            study_id=self.study_id,
            df=df,
            split=split,
            study_dir=self._study_dir,
            n_workers=n_workers,
            **kwargs
        )

    def issf_dataset(self, split: str = "train", auto_normalize: bool = True, force_normalize: bool = False, include_time: bool = False, include_individual_id: bool = False, n_workers: int = 1, **kwargs):
        """
        Create iSSFDataset for this study.

        Args:
            split: "train" or "test"
            auto_normalize: If True, ensure normalization exists and apply it.
            force_normalize: If True, recompute normalization statistics.
            include_time: If True, include cyclical time-of-year and time-of-day
                features and their interactions.
            include_individual_id: If True, include one-hot encoded individual ID
                features (length N_individuals).
            n_workers: Number of parallel workers for feature extraction (default 1)
            **kwargs: Additional arguments passed to iSSFDataset

        Returns:
            iSSFDataset instance
        """
        from movebench.ssf import iSSFDataset

        if auto_normalize:
            self._ensure_normalization(force=force_normalize)
            kwargs.setdefault("normalizer", self._normalizer)

        df = self.get_split(split)
        return iSSFDataset(
            study_id=self.study_id,
            df=df,
            split=split,
            study_dir=self._study_dir,
            include_time=include_time,
            include_individual_id=include_individual_id,
            n_workers=n_workers,
            **kwargs
        )

    def download_rasters(
        self,
        splits: list[str] | str | None = None,
        layers: list[str] | None = None,
        n_workers: int = 16,
        skip_existing: bool = True,
    ) -> dict:
        """
        Download GEE environmental rasters for this study.

        Downloads ERA5 atmospheric data, MODIS products (LST, VI, NDSI, BAI),
        Dynamic World land cover, and DEM tiles for the specified splits.

        Args:
            splits: List of splits to download ("train", "test"), or single split.
                    Default: ["train", "test"]
            layers: List of layers to download. Accepts chip group names
                    ("era5", "modis", "dynamic_world") and tile layer names ("dem").
                    Default: all GEE layers
            n_workers: Number of parallel download workers (default 16)
            skip_existing: Skip files that already exist (default True)

        Returns:
            Dict with download results:
            {
                "chips": {
                    "train": {"total": N, "downloaded": N, "existing": N, "failed": N},
                    "test": {...}
                },
                "tiles": {
                    "dem": {"total": N, "downloaded": N, ...}
                }
            }

        Example:
            >>> study = Study.load("abc123-def456")
            >>> result = study.download_rasters()
            >>> print(f"Downloaded: {result['chips']['train']['downloaded']} chips")

            # Download only ERA5 and DEM
            >>> result = study.download_rasters(layers=["era5", "dem"])

            # Download only train split chips (no DEM)
            >>> result = study.download_rasters(splits="train", layers=["era5", "modis"])
        """
        from movebench.download import download_study_rasters

        return download_study_rasters(
            study_id=self.study_id,
            splits=splits,
            layers=layers,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    def __repr__(self) -> str:
        return f"Study({self.study_id!r})"
