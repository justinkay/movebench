"""Optimized Step Selection Function datasets.

This module provides SSFDataset and iSSFDataset implementations that:
1. Pre-compute all candidate points before loading data
2. Load minimal bounding regions for each layer
3. Sample directly in native coordinates (no full reprojection)

This avoids the 15x spatial multiplier overhead for test data by loading only
the minimal region needed for the ~101 candidate points instead of full chips.
"""

import hashlib
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import rasterio
from tqdm import tqdm

from movebench.config import (
    PATHS,
    LAYERS,
    LON_COL,
    LAT_COL,
    POINT_ID_COL,
    INDIVIDUAL_COL,
    TARGET_CRS,
    LAYER_ORDER,
    get_channel_indices,
    get_total_channels,
)
from movebench.utils import modis_tiles_for_bounds, setup_df_for_ssf
from movebench.loaders import ChipLoader, TileLoader, get_loader
from movebench.transforms import Normalizer


def _create_loaders(study_dir: Path, split: str) -> dict:
    """Create fresh loaders for a study (used by worker processes).

    Args:
        study_dir: Path to study directory
        split: Data split ("train" or "test")

    Returns:
        Dict mapping loader names to loader instances
    """
    chip_dir = study_dir / "chips" / split
    dem_dir = study_dir / "tiles" / "dem"

    loaders = {}

    # GEE chip loaders
    loaders["era5"] = ChipLoader(str(chip_dir), "era5")
    loaders["modis"] = ChipLoader(str(chip_dir), "modis")
    loaders["high_res"] = ChipLoader(str(chip_dir), "high_res")

    # GEE tile loader (DEM)
    loaders["dem"] = TileLoader(str(dem_dir), "dem")

    # Local HDF loaders (MODIS products)
    loaders["modis_lc"] = get_loader("modis_lc")
    loaders["modis_cvf_tree"] = get_loader("modis_cvf_tree")
    loaders["modis_cvf_nontree"] = get_loader("modis_cvf_nontree")
    loaders["modis_cvf_nonveg"] = get_loader("modis_cvf_nonveg")

    # Local GeoTIFF loaders
    loaders["hii"] = get_loader("hii")
    loaders["bioclim"] = get_loader("bioclim")

    return loaders


def _extract_sample_worker(args: tuple) -> tuple:
    """Worker function for parallel feature extraction.

    Must be module-level to be picklable for multiprocessing.

    Args:
        args: Tuple of (idx, worker_args_dict) where worker_args_dict contains
            all data needed to extract features for sample idx

    Returns:
        Tuple of (idx, step_pixels_np, valid_pixels_np) - numpy arrays to avoid
        PyTorch tensor serialization issues across processes
    """
    idx, worker_args = args

    study_dir = Path(worker_args["study_dir"])
    split = worker_args["split"]
    point_id = worker_args["point_id"]
    timestamp = worker_args["timestamp"]
    lons = worker_args["lons"]
    lats = worker_args["lats"]
    bounds_4326 = worker_args["bounds_4326"]
    channel_indices = worker_args["channel_indices"]
    total_channels = worker_args["total_channels"]
    lc_num_classes = worker_args["lc_num_classes"]

    # Create fresh loaders in worker process
    loaders = _create_loaders(study_dir, split)

    # Extract features using the same logic as _load_and_extract_all_layers
    step_pixels, valid_pixels = _extract_all_layers_standalone(
        loaders=loaders,
        point_id=point_id,
        timestamp=timestamp,
        lons=lons,
        lats=lats,
        bounds_4326=bounds_4326,
        channel_indices=channel_indices,
        total_channels=total_channels,
        lc_num_classes=lc_num_classes,
    )

    # Convert to numpy to avoid PyTorch tensor serialization issues
    return idx, step_pixels.numpy(), valid_pixels.numpy()


def _extract_all_layers_standalone(
    loaders: dict,
    point_id: int,
    timestamp,
    lons: np.ndarray,
    lats: np.ndarray,
    bounds_4326: tuple,
    channel_indices: dict,
    total_channels: int,
    lc_num_classes: int,
) -> tuple:
    """Standalone feature extraction function for use in worker processes.

    This duplicates the logic from SSFDataset._load_and_extract_all_layers
    but works with passed-in loaders rather than instance attributes.

    Args:
        loaders: Dict of layer loaders
        point_id: Point ID for GEE chips
        timestamp: Timestamp for temporal layers
        lons, lats: Point coordinates in EPSG:4326
        bounds_4326: Bounding box in EPSG:4326
        channel_indices: Dict mapping layer names to (start, end) indices
        total_channels: Total number of channels
        lc_num_classes: Number of land cover classes

    Returns:
        Tuple of (step_pixels, valid_pixels) with shape (total_channels, N)
    """
    N = len(lons)

    # Allocate output tensors
    all_values = torch.zeros(total_channels, N)
    all_valid = torch.zeros(total_channels, N, dtype=torch.bool)

    # --- GEE Chips ---
    # MODIS chip (load first to get chip bounds for other layers)
    chip_bounds = None
    try:
        modis_chip = loaders["modis"].load(point_id=point_id)
        chip_bounds = modis_chip['bounds']
        vals, valid = _extract_from_chip(modis_chip, lons, lats, bilinear=True)
        chip_start = 0
        for layer_name in ["modis_lst", "modis_vi", "modis_ndsi", "modis_bai"]:
            start, end = channel_indices[layer_name]
            n_bands = end - start
            all_values[start:end] = vals[chip_start:chip_start + n_bands]
            all_valid[start:end] = valid[chip_start:chip_start + n_bands]
            chip_start += n_bands
    except (FileNotFoundError, OSError):
        pass

    # Use chip bounds for tile/local layers
    tile_bounds = chip_bounds if chip_bounds is not None else bounds_4326

    # Prepare MODIS tile coordinates
    hvs = modis_tiles_for_bounds(tile_bounds)

    # ERA5 chip
    try:
        era5_chip = loaders["era5"].load(point_id=point_id)
        vals, valid = _extract_from_chip(era5_chip, lons, lats, bilinear=True)
        chip_offset = 0
        for layer_name in ["era5_land", "era5", "era5_land_daily"]:
            start, end = channel_indices[layer_name]
            n_bands = end - start
            all_values[start:end] = vals[chip_offset:chip_offset + n_bands]
            all_valid[start:end] = valid[chip_offset:chip_offset + n_bands]
            chip_offset += n_bands
    except (FileNotFoundError, OSError):
        pass

    # --- DEM (tile) ---
    try:
        dem_start, dem_end = channel_indices["dem"]
        vals, valid = _extract_from_tiled(
            loaders["dem"], tile_bounds, lons, lats, bilinear=True
        )
        all_values[dem_start:dem_end] = vals
        all_valid[dem_start:dem_end] = valid
    except (FileNotFoundError, OSError, ValueError):
        pass

    # --- MODIS Land Cover (categorical) ---
    try:
        lc_start, lc_end = channel_indices["modis_lc"]
        vals, valid = _extract_categorical_standalone(
            loaders["modis_lc"], tile_bounds, lons, lats,
            lc_num_classes, timestamp=timestamp, hvs=hvs
        )
        all_values[lc_start:lc_end] = vals
        all_valid[lc_start:lc_end] = valid
    except (FileNotFoundError, OSError):
        pass

    # --- MODIS CVF (3 layers) ---
    for cvf_name in ["modis_cvf_tree", "modis_cvf_nontree", "modis_cvf_nonveg"]:
        try:
            start, end = channel_indices[cvf_name]
            vals, valid = _extract_from_tiled(
                loaders[cvf_name], tile_bounds, lons, lats,
                timestamp=timestamp, hvs=hvs, bilinear=True
            )
            all_values[start:end] = vals
            all_valid[start:end] = valid
        except (FileNotFoundError, OSError):
            pass

    # --- HII ---
    try:
        hii_start, hii_end = channel_indices["hii"]
        vals, valid = _extract_from_tiled(
            loaders["hii"], tile_bounds, lons, lats,
            timestamp=timestamp, bilinear=True
        )
        all_values[hii_start:hii_end] = vals
        all_valid[hii_start:hii_end] = valid
    except (FileNotFoundError, OSError):
        pass

    # --- BioClim (19 bands) ---
    bioclim_start, bioclim_end = channel_indices["bioclim"]
    for i in range(1, 20):
        try:
            vals, valid = _extract_from_tiled(
                loaders["bioclim"], tile_bounds, lons, lats,
                band_num=i, bilinear=True
            )
            ch = bioclim_start + (i - 1)
            all_values[ch:ch + 1] = vals
            all_valid[ch:ch + 1] = valid
        except (FileNotFoundError, OSError):
            pass

    # --- High-res chip (Dynamic World 9 bands + Landsat NDWI 1 band) ---
    try:
        hr_chip = loaders["high_res"].load(point_id=point_id)
        vals, valid = _extract_from_chip(hr_chip, lons, lats, bilinear=False)
        chip_offset = 0
        for layer_name in ["dynamic_world", "landsat_ndwi"]:
            start, end = channel_indices[layer_name]
            n_bands = end - start
            all_values[start:end] = vals[chip_offset:chip_offset + n_bands]
            all_valid[start:end] = valid[chip_offset:chip_offset + n_bands]
            chip_offset += n_bands
    except (FileNotFoundError, OSError):
        pass

    return all_values, all_valid


def _extract_from_chip(
    chip_dict: dict,
    lons: np.ndarray,
    lats: np.ndarray,
    bilinear: bool = True,
) -> tuple:
    """Sample points from GEE chip in native coords (standalone version).

    Args:
        chip_dict: Dict with array, transform, crs, nodata
        lons, lats: Point coordinates in EPSG:4326
        bilinear: Use bilinear interpolation (True) or nearest neighbor (False)

    Returns:
        Tuple of (values, valid_mask) with shape (C, N)
    """
    arr = chip_dict['array'].copy()
    transform = chip_dict['transform']
    src_crs = chip_dict['crs']
    nodata = chip_dict.get('nodata')

    C, H, W = arr.shape
    N = len(lons)

    # Transform points to chip CRS
    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    # Convert to pixel coordinates
    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    cols = np.asarray(cols)
    rows = np.asarray(rows)

    # Create nodata mask (True where data is valid)
    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            nodata_mask = ~np.isnan(arr)
        elif np.isinf(nodata):
            nodata_mask = ~np.isinf(arr)
        else:
            nodata_mask = arr != nodata
    else:
        nodata_mask = np.isfinite(arr)

    if bilinear:
        arr_clean = np.where(nodata_mask, arr, 0.0).astype(np.float32)
        arr_t = torch.from_numpy(arr_clean)

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_clamped = np.clip(rows, 0, H - 1)
        cols_clamped = np.clip(cols, 0, W - 1)

        x_norm = 2.0 * (cols_clamped / max(W - 1, 1)) - 1.0
        y_norm = 2.0 * (rows_clamped / max(H - 1, 1)) - 1.0

        grid = torch.stack([
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32),
        ], dim=-1).view(1, N, 1, 2)

        samples = torch.nn.functional.grid_sample(
            arr_t.unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',
        ).squeeze(0).squeeze(-1)

        mask_t = torch.from_numpy(nodata_mask.astype(np.float32))
        mask_interp = torch.nn.functional.grid_sample(
            mask_t.unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',
        ).squeeze(0).squeeze(-1)

        valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
        valid_t = valid_t & (mask_interp > 0.99)

        return samples, valid_t

    else:
        rows_int = np.clip(np.round(rows).astype(int), 0, H - 1)
        cols_int = np.clip(np.round(cols).astype(int), 0, W - 1)
        in_bounds = (rows >= -0.5) & (rows < H - 0.5) & (cols >= -0.5) & (cols < W - 0.5)

        samples = torch.from_numpy(arr[:, rows_int, cols_int]).float()
        valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
        valid_t = valid_t & torch.from_numpy(nodata_mask[:, rows_int, cols_int])

        return samples, valid_t


def _extract_from_tiled(
    loader,
    bounds: tuple,
    lons: np.ndarray,
    lats: np.ndarray,
    timestamp=None,
    hvs=None,
    band_num: int = None,
    bilinear: bool = True,
) -> tuple:
    """Load minimal window from tiled layer and extract points (standalone version).

    Args:
        loader: Layer loader instance
        bounds: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
        lons, lats: Point coordinates in EPSG:4326
        timestamp: Optional timestamp for temporal layers
        hvs: Optional MODIS tile coordinates
        band_num: Optional band number (for bioclim)
        bilinear: Use bilinear interpolation

    Returns:
        Tuple of (values, valid_mask) with shape (C, N)
    """
    kwargs = {}
    if timestamp is not None:
        kwargs['timestamp'] = timestamp
    if hvs is not None:
        kwargs['hvs'] = hvs
    if band_num is not None:
        kwargs['band_num'] = band_num

    raster = loader.load(bounds=bounds, **kwargs)
    return _extract_from_chip(raster, lons, lats, bilinear=bilinear)


def _extract_categorical_standalone(
    loader,
    bounds: tuple,
    lons: np.ndarray,
    lats: np.ndarray,
    num_classes: int,
    timestamp=None,
    hvs=None,
) -> tuple:
    """Extract and one-hot encode categorical layer (standalone version).

    Args:
        loader: Layer loader instance
        bounds: Bounding box in EPSG:4326
        lons, lats: Point coordinates
        num_classes: Number of classes for one-hot encoding
        timestamp: Optional timestamp
        hvs: Optional MODIS tile coordinates

    Returns:
        Tuple of (one_hot, valid_mask) with shape (num_classes, N)
    """
    kwargs = {'timestamp': timestamp}
    if hvs is not None:
        kwargs['hvs'] = hvs

    raster = loader.load(bounds=bounds, **kwargs)

    # Use nearest neighbor for categorical
    values, valid = _extract_from_chip(raster, lons, lats, bilinear=False)

    # values shape: (1, N), convert to one-hot (num_classes, N)
    class_vals = values.squeeze(0).long()
    class_vals = torch.clamp(class_vals, 0, num_classes - 1)

    one_hot = torch.nn.functional.one_hot(class_vals, num_classes).T.float()

    # Expand valid mask to all classes
    valid_expanded = valid.expand(num_classes, -1)

    return one_hot, valid_expanded


class SSFDataset(Dataset):
    """
    Optimized Step Selection Function dataset.

    Loads only minimal bounding regions around candidate points rather than
    full chips, providing significant speedups especially for test data with
    15x spatial multiplier.

    Each sample returns:
        - step_pixels: Tensor (72, 101) with environmental values at candidates
        - valid_pixels: Tensor (72, 101) validity mask
        - None: Placeholder for transform (not meaningful for point samples)
        - str: TARGET_CRS for reference
    """

    def __init__(
        self,
        study_id: str,
        df: pd.DataFrame,
        split: str,
        study_dir=None,
        random_seed: int = 0,
        num_candidates: int = 100,
        normalizer: Normalizer = None,
        force_recompute: bool = False,
        n_workers: int = 1,
    ):
        """
        Initialize SSFDataset.

        Args:
            study_id: Study identifier (UUID)
            df: DataFrame with location and timestamp columns (already filtered to train/test)
            split: Data split ("train" or "test")
            study_dir: Study directory (default: PATHS["studies"] / study_id)
            random_seed: Random seed for reproducibility
            num_candidates: Number of pseudo-absence candidates per step
            normalizer: Optional pre-computed Normalizer instance
            force_recompute: If True, recompute and re-cache features even if a
                cache file already exists (e.g. after re-downloading rasters)
            n_workers: Number of parallel workers for feature extraction (default 1)
        """
        self.study_id = study_id
        self.split = split
        self.random_seed = random_seed
        self.num_candidates = num_candidates
        self.normalizer = normalizer
        self.n_workers = n_workers

        # Set up directory paths
        self.study_dir = Path(study_dir or (PATHS["studies"] / study_id))
        self.chip_dir = self.study_dir / "chips" / split
        self.dem_dir = self.study_dir / "tiles" / "dem"

        # Prepare SSF DataFrame
        self.df = setup_df_for_ssf(df.copy()) # TODO this should probably be in this file
        self.sample_indices = self.df.index[self.df['can_be_ssf_start']].to_numpy()

        # Pre-compute all pseudo-step samples for reproducibility
        self._precomputed_samples = self._precompute_all_samples()

        # Lazy-initialize loaders
        self._loaders = None

        # Cache channel indices and layer info
        self._channel_indices = get_channel_indices()
        self._total_channels = get_total_channels()
        self._lc_num_classes = LAYERS["modis_lc"].num_classes

        # Transformer from EPSG:4326 to TARGET_CRS (cached)
        self._transformer_4326_to_target = Transformer.from_crs(
            "EPSG:4326", TARGET_CRS, always_xy=True
        )

        # Feature cache
        self._feature_cache = self._load_or_build_cache(force_recompute)

    def _cache_key(self) -> str:
        """Compute a hash key for the current dataset configuration."""
        key_data = (
            f"{self.study_id}|{self.split}|{self.random_seed}|{self.num_candidates}"
            f"|{len(self.sample_indices)}|{self._total_channels}"
            f"|{self.sample_indices.tobytes().hex()}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _cache_path(self) -> Path:
        """Return the path to the feature cache file."""
        cache_dir = self.study_dir / "cache"
        kind = type(self).__name__.lower()
        key = self._cache_key()
        return cache_dir / f"{kind}_{self.split}_{key}.pt"

    def _load_or_build_cache(self, force_recompute: bool) -> dict:
        """Load features from cache or build and save them."""
        path = self._cache_path()

        if not force_recompute and path.exists():
            print(f"Loading cached features from {path}")
            return torch.load(path, weights_only=True)

        print(f"Building feature cache ({len(self.sample_indices)} samples)...")
        cache = self._build_feature_cache()

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, path)
        print(f"Saved feature cache to {path}")

        return cache

    def _build_worker_args(self, idx: int) -> tuple:
        """Build arguments tuple for parallel worker.

        Args:
            idx: Sample index

        Returns:
            Tuple of (idx, worker_args_dict) for _extract_sample_worker
        """
        orig_idx = int(self.sample_indices[idx])
        row = self.df.loc[orig_idx]
        point_id = int(row[POINT_ID_COL])
        timestamp = row['timestamp']

        # Compute candidate points
        result = self._compute_candidate_points(idx)
        lons, lats, xs, ys, bounds_4326 = result[:5]

        worker_args = {
            "study_dir": str(self.study_dir),
            "split": self.split,
            "point_id": point_id,
            "timestamp": timestamp,
            "lons": lons,
            "lats": lats,
            "bounds_4326": bounds_4326,
            "channel_indices": self._channel_indices,
            "total_channels": self._total_channels,
            "lc_num_classes": self._lc_num_classes,
        }

        return (idx, worker_args)

    def _build_feature_cache(self) -> dict:
        """Compute env features for all samples and return as a dict of tensors."""
        N = self.num_candidates + 1
        n_samples = len(self.sample_indices)

        all_step_pixels = torch.zeros(n_samples, self._total_channels, N)
        all_valid_pixels = torch.zeros(n_samples, self._total_channels, N, dtype=torch.bool)

        if self.n_workers > 1:
            # Parallel processing with multiprocessing
            print(f"  Building feature cache with {self.n_workers} workers...")

            # Build args for all samples
            args_list = [self._build_worker_args(idx) for idx in range(n_samples)]

            # Use spawn context to avoid fork issues with torch
            # maxtasksperchild restarts workers periodically to prevent FD/memory leaks
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(self.n_workers, maxtasksperchild=200) as pool:
                results = list(tqdm(
                    pool.imap(_extract_sample_worker, args_list, chunksize=10),
                    total=n_samples,
                    desc="  Extracting features"
                ))

            # Reassemble into tensors (workers return numpy arrays)
            for idx, step_pixels_np, valid_pixels_np in results:
                all_step_pixels[idx] = torch.from_numpy(step_pixels_np)
                all_valid_pixels[idx] = torch.from_numpy(valid_pixels_np)
        else:
            # Sequential processing (original behavior)
            # Lazy-initialize loaders once for all samples
            if self._loaders is None:
                self._loaders = _create_loaders(self.study_dir, self.split)

            for idx in range(n_samples):
                if idx % 50 == 0:
                    print(f"  Processing sample {idx}/{n_samples}")

                orig_idx = int(self.sample_indices[idx])
                row = self.df.loc[orig_idx]
                point_id = int(row[POINT_ID_COL])
                timestamp = row['timestamp']

                # _compute_candidate_points returns at least (lons, lats, xs, ys, bounds_4326);
                # iSSFDataset adds (step_lengths, turn_angles) — only env features are cached.
                result = self._compute_candidate_points(idx)
                lons, lats, xs, ys, bounds_4326 = result[:5]

                step_pixels, valid_pixels = _extract_all_layers_standalone(
                    loaders=self._loaders,
                    point_id=point_id,
                    timestamp=timestamp,
                    lons=lons,
                    lats=lats,
                    bounds_4326=bounds_4326,
                    channel_indices=self._channel_indices,
                    total_channels=self._total_channels,
                    lc_num_classes=self._lc_num_classes,
                )

                all_step_pixels[idx] = step_pixels
                all_valid_pixels[idx] = valid_pixels

        return {
            'step_pixels': all_step_pixels,
            'valid_pixels': all_valid_pixels,
        }

    def _precompute_all_samples(self) -> dict:
        """Pre-sample step lengths and turn angles for all indices."""
        rng = np.random.default_rng(self.random_seed)

        # Get valid step lengths and turn angles from df
        valid_lengths = self.df.loc[self.df['valid_step'], 'step_length_m'].values
        valid_turns = self.df['turn_angle_rad'].dropna().values

        samples = {}
        for i, orig_idx in enumerate(self.sample_indices):
            pseudo_lens = rng.choice(valid_lengths, size=self.num_candidates, replace=True)
            pseudo_turns = rng.choice(valid_turns, size=self.num_candidates, replace=True)
            samples[i] = (pseudo_lens, pseudo_turns)

        return samples

    def _compute_candidate_points(self, idx: int) -> tuple:
        """
        Compute 101 candidate (lon, lat, x, y) coordinates.

        Returns:
            Tuple of (lons, lats, xs, ys, bounds_4326)
            - lons, lats: Arrays of shape (101,) in EPSG:4326
            - xs, ys: Arrays of shape (101,) in TARGET_CRS
            - bounds_4326: (min_lon, min_lat, max_lon, max_lat) bounding box
        """
        orig_idx = int(self.sample_indices[idx])
        row = self.df.loc[orig_idx]

        # Start location
        start_lat = row[LAT_COL]
        start_lon = row[LON_COL]
        start_bearing = row['bearing']

        # Get pre-sampled pseudo steps
        pseudo_lens, pseudo_turns = self._precomputed_samples[idx]

        # Compute candidate positions in target CRS
        x0, y0 = self._transformer_4326_to_target.transform(start_lon, start_lat)

        headings = start_bearing + pseudo_turns
        headings = (headings + np.pi) % (2 * np.pi) - np.pi
        dx = pseudo_lens * np.sin(headings)
        dy = pseudo_lens * np.cos(headings)
        xs = x0 + dx
        ys = y0 + dy

        # True endpoint
        next_idx = int(row['next_index'])
        next_row = self.df.loc[next_idx]
        end_lat = next_row[LAT_COL]
        end_lon = next_row[LON_COL]
        x1, y1 = self._transformer_4326_to_target.transform(end_lon, end_lat)

        # Append true endpoint (101 total)
        xs = np.append(xs, x1)
        ys = np.append(ys, y1)

        # Convert back to lat/lon for all points
        transformer_inv = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
        lons, lats = transformer_inv.transform(xs, ys)

        # Compute bounding box in EPSG:4326 with small buffer
        buffer_deg = 0.01  # ~1km buffer
        bounds_4326 = (
            float(np.min(lons)) - buffer_deg,
            float(np.min(lats)) - buffer_deg,
            float(np.max(lons)) + buffer_deg,
            float(np.max(lats)) + buffer_deg,
        )

        return lons, lats, xs, ys, bounds_4326

    def __getitem__(self, idx) -> tuple:
        """
        Load and process all layers for a single sample.

        Returns:
            Tuple of:
                - step_pixels: Tensor (72, 101) with environmental values
                - valid_pixels: Tensor (72, 101) validity mask
                - None: Placeholder for transform
                - str: TARGET_CRS
        """
        step_pixels = self._feature_cache['step_pixels'][idx].clone()
        valid_pixels = self._feature_cache['valid_pixels'][idx].clone()

        # Apply normalization
        if self.normalizer is not None:
            step_pixels = self.normalizer(step_pixels)

        # Return empty dict for transform (None can't be collated by DataLoader)
        return step_pixels, valid_pixels, {}, TARGET_CRS

    def __len__(self):
        return len(self.sample_indices)


MOVEMENT_FEATURES = ["sl_", "log_sl_", "cos_ta_"]
TIME_FEATURES = ["sin_doy", "cos_doy", "sin_tod", "cos_tod"]


def get_issf_channel_info(n_env_channels=None, include_time=False, include_individual_id=False, n_individuals=0):
    """Return dict with channel ranges for env, movement, interactions, and total.

    Args:
        n_env_channels: Number of environmental channels. Defaults to get_total_channels().
        include_time: If True, include cyclical time-of-year and time-of-day features
            and their interactions with env and movement features.
        include_individual_id: If True, include one-hot encoded individual ID features.
        n_individuals: Number of individuals (required if include_individual_id=True).

    Returns:
        Dict with keys: env, movement, interactions, total, and per-feature details.
    """
    if n_env_channels is None:
        n_env_channels = get_total_channels()

    n_move = len(MOVEMENT_FEATURES)
    n_time = len(TIME_FEATURES) if include_time else 0
    n_indiv = n_individuals if include_individual_id else 0

    # Channel layout: env | movement | time | individual_id | move×env | time×env | time×move
    n_move_env = n_move * n_env_channels
    n_time_env = n_time * n_env_channels
    n_time_move = n_time * n_move
    n_interact = n_move_env + n_time_env + n_time_move
    total = n_env_channels + n_move + n_time + n_indiv + n_interact

    offset = n_env_channels
    info = {
        "env": (0, n_env_channels),
        "sl_": (offset, offset + 1),
        "log_sl_": (offset + 1, offset + 2),
        "cos_ta_": (offset + 2, offset + 3),
        "movement": (offset, offset + n_move),
    }
    offset += n_move

    if include_time:
        info["sin_doy"] = (offset, offset + 1)
        info["cos_doy"] = (offset + 1, offset + 2)
        info["sin_tod"] = (offset + 2, offset + 3)
        info["cos_tod"] = (offset + 3, offset + 4)
        info["time"] = (offset, offset + n_time)
        offset += n_time

    if include_individual_id:
        info["individual_id"] = (offset, offset + n_indiv)
        offset += n_indiv

    # move × env interactions
    move_env_start = offset
    info["sl_interactions"] = (offset, offset + n_env_channels)
    info["log_sl_interactions"] = (offset + n_env_channels, offset + 2 * n_env_channels)
    info["cos_ta_interactions"] = (offset + 2 * n_env_channels, offset + 3 * n_env_channels)
    info["move_env_interactions"] = (offset, offset + n_move_env)
    offset += n_move_env

    if include_time:
        info["time_env_interactions"] = (offset, offset + n_time_env)
        offset += n_time_env
        info["time_move_interactions"] = (offset, offset + n_time_move)
        offset += n_time_move

    info["interactions"] = (move_env_start, total)
    info["total"] = total
    info["n_env"] = n_env_channels
    info["n_movement"] = n_move
    info["n_time"] = n_time
    info["n_individual_id"] = n_indiv
    info["n_interactions"] = n_interact

    return info


class iSSFDataset(SSFDataset):
    """
    Integrated SSF dataset with parametric movement distributions.

    Uses fitted gamma (step length) and von Mises (turn angle) distributions
    instead of empirical resampling. Output includes movement features
    (step length, log step length, cos turn angle) and their interactions
    with all environmental features.

    Args:
        normalize_movement: If True (default), z-score movement features using
            statistics computed from the training DataFrame. This puts them on
            the same scale as the normalized environmental features, which is
            important for stable optimization. Set to False to keep raw values
            (needed for amt-style coefficient interpretation).
        movement_stats: Optional pre-computed movement statistics dict. Use this
            to pass training statistics to a test dataset to avoid train/test
            leakage. If None and normalize_movement=True, stats are computed
            from this dataset's DataFrame.
    """

    def __init__(self, *args, normalize_movement: bool = True, include_time: bool = False,
                 include_individual_id: bool = False, movement_stats: dict = None, **kwargs):
        self.include_time = include_time
        self.include_individual_id = include_individual_id
        super().__init__(*args, **kwargs)
        self.normalize_movement = normalize_movement
        if normalize_movement:
            if movement_stats is not None:
                self._movement_stats = movement_stats
            else:
                self._movement_stats = self._compute_movement_stats()
        else:
            self._movement_stats = None

        # Build individual-to-index mapping for one-hot encoding
        if include_individual_id:
            unique_individuals = self.df[INDIVIDUAL_COL].unique()
            self._individual_to_idx = {ind: i for i, ind in enumerate(sorted(unique_individuals))}
            self._n_individuals = len(unique_individuals)
        else:
            self._individual_to_idx = {}
            self._n_individuals = 0

    def _compute_movement_stats(self) -> dict:
        """Compute mean/std for sl_, log_sl_, cos_ta_ from this dataset's DataFrame."""
        valid_lengths = self.df.loc[self.df['valid_step'], 'step_length_m'].values
        valid_turns = self.df['turn_angle_rad'].dropna().values

        sl = valid_lengths.astype(np.float64)
        log_sl = np.log(sl + 1e-8)
        cos_ta = np.cos(valid_turns.astype(np.float64))

        return {
            'sl_mean': float(np.mean(sl)),
            'sl_std': float(max(np.std(sl), 1e-8)),
            'log_sl_mean': float(np.mean(log_sl)),
            'log_sl_std': float(max(np.std(log_sl), 1e-8)),
            'cos_ta_mean': float(np.mean(cos_ta)),
            'cos_ta_std': float(max(np.std(cos_ta), 1e-8)),
        }

    def _cache_key(self) -> str:
        """Include include_time in the cache key to avoid collisions."""
        base = super()._cache_key()
        key_data = f"{base}|include_time={self.include_time}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _compute_time_features(self, idx: int) -> torch.Tensor:
        """Compute cyclical time-of-year and time-of-day features for a sample.

        Features are constant across all candidate points since they depend
        only on the step's start timestamp.

        Args:
            idx: Dataset index

        Returns:
            Tensor of shape (4, N) with [sin_doy, cos_doy, sin_tod, cos_tod]
        """
        orig_idx = int(self.sample_indices[idx])
        ts = self.df.loc[orig_idx, 'timestamp']
        N = self.num_candidates + 1

        doy = ts.timetuple().tm_yday
        hour_frac = ts.hour + ts.minute / 60 + ts.second / 3600

        sin_doy = np.sin(2 * np.pi * doy / 365.25)
        cos_doy = np.cos(2 * np.pi * doy / 365.25)
        sin_tod = np.sin(2 * np.pi * hour_frac / 24)
        cos_tod = np.cos(2 * np.pi * hour_frac / 24)

        feats = torch.tensor([sin_doy, cos_doy, sin_tod, cos_tod], dtype=torch.float32)
        return feats.unsqueeze(1).expand(4, N)

    def _compute_individual_id_features(self, idx: int) -> torch.Tensor:
        """Compute one-hot encoded individual ID features for a sample.

        Features are constant across all candidate points since they depend
        only on the individual that made this step.

        Args:
            idx: Dataset index

        Returns:
            Tensor of shape (n_individuals, N) with one-hot encoding
        """
        orig_idx = int(self.sample_indices[idx])
        individual = self.df.loc[orig_idx, INDIVIDUAL_COL]
        N = self.num_candidates + 1
        indiv_idx = self._individual_to_idx[individual]
        one_hot = torch.zeros(self._n_individuals, dtype=torch.float32)
        one_hot[indiv_idx] = 1.0
        return one_hot.unsqueeze(1).expand(-1, N)

    def _precompute_all_samples(self) -> dict:
        """Pre-sample from fitted gamma (step) and von Mises (turn) distributions."""
        from scipy import stats

        rng = np.random.default_rng(self.random_seed)

        # Get valid step lengths and turn angles from df
        valid_lengths = self.df.loc[self.df['valid_step'], 'step_length_m'].values
        valid_turns = self.df['turn_angle_rad'].dropna().values

        # Fit distributions
        try:
            gamma_params = stats.gamma.fit(valid_lengths, floc=0)
            use_gamma = True
        except Exception:
            print("Gamma fit failed, using empirical distribution for step lengths.")
            use_gamma = False

        try:
            vonmises_params = stats.vonmises.fit(valid_turns)
            use_vonmises = True
        except Exception:
            print("Von Mises fit failed, using empirical distribution for turn angles.")
            use_vonmises = False

        samples = {}
        for i, orig_idx in enumerate(self.sample_indices):
            if use_gamma:
                pseudo_lens = stats.gamma.rvs(*gamma_params, size=self.num_candidates, random_state=rng)
            else:
                pseudo_lens = rng.choice(valid_lengths, size=self.num_candidates, replace=True)

            if use_vonmises:
                pseudo_turns = stats.vonmises.rvs(*vonmises_params, size=self.num_candidates, random_state=rng)
            else:
                pseudo_turns = rng.choice(valid_turns, size=self.num_candidates, replace=True)

            samples[i] = (pseudo_lens, pseudo_turns)

        return samples

    def _compute_candidate_points(self, idx: int) -> tuple:
        """
        Compute 101 candidate coordinates plus step lengths and turn angles.

        Returns:
            Tuple of (lons, lats, xs, ys, bounds_4326, step_lengths, turn_angles)
            - lons, lats: Arrays of shape (101,) in EPSG:4326
            - xs, ys: Arrays of shape (101,) in TARGET_CRS
            - bounds_4326: (min_lon, min_lat, max_lon, max_lat) bounding box
            - step_lengths: Array of shape (101,) in meters (projected CRS)
            - turn_angles: Array of shape (101,) in radians
        """
        orig_idx = int(self.sample_indices[idx])
        row = self.df.loc[orig_idx]

        # Start location
        start_lat = row[LAT_COL]
        start_lon = row[LON_COL]
        start_bearing = row['bearing']

        # Get pre-sampled pseudo steps
        pseudo_lens, pseudo_turns = self._precomputed_samples[idx]

        # Compute candidate positions in target CRS
        x0, y0 = self._transformer_4326_to_target.transform(start_lon, start_lat)

        headings = start_bearing + pseudo_turns
        headings = (headings + np.pi) % (2 * np.pi) - np.pi
        dx = pseudo_lens * np.sin(headings)
        dy = pseudo_lens * np.cos(headings)
        xs = x0 + dx
        ys = y0 + dy

        # True endpoint
        next_idx = int(row['next_index'])
        next_row = self.df.loc[next_idx]
        end_lat = next_row[LAT_COL]
        end_lon = next_row[LON_COL]
        x1, y1 = self._transformer_4326_to_target.transform(end_lon, end_lat)

        # Compute true step length and turn angle in projected CRS
        true_sl = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        true_heading = np.arctan2(x1 - x0, y1 - y0)
        true_ta = true_heading - start_bearing
        # Wrap to [-pi, pi]
        true_ta = (true_ta + np.pi) % (2 * np.pi) - np.pi

        # Append true endpoint (101 total)
        xs = np.append(xs, x1)
        ys = np.append(ys, y1)
        step_lengths = np.append(pseudo_lens, true_sl)
        turn_angles = np.append(pseudo_turns, true_ta)

        # Convert back to lat/lon for all points
        transformer_inv = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
        lons, lats = transformer_inv.transform(xs, ys)

        # Compute bounding box in EPSG:4326 with small buffer
        buffer_deg = 0.01  # ~1km buffer
        bounds_4326 = (
            float(np.min(lons)) - buffer_deg,
            float(np.min(lats)) - buffer_deg,
            float(np.max(lons)) + buffer_deg,
            float(np.max(lats)) + buffer_deg,
        )

        return lons, lats, xs, ys, bounds_4326, step_lengths, turn_angles

    def _compute_movement_features(
        self, step_lengths: np.ndarray, turn_angles: np.ndarray
    ) -> torch.Tensor:
        """
        Compute movement feature tensor from step lengths and turn angles.

        If normalize_movement is True, features are z-scored using statistics
        computed from the training DataFrame. Otherwise raw values are returned
        (for amt-style coefficient interpretation).

        Args:
            step_lengths: Array of shape (N,) in meters
            turn_angles: Array of shape (N,) in radians

        Returns:
            Tensor of shape (3, N): [sl_, log_sl_, cos_ta_]
        """
        sl = torch.from_numpy(step_lengths.astype(np.float32))
        log_sl = torch.log(sl + 1e-8)
        cos_ta = torch.cos(torch.from_numpy(turn_angles.astype(np.float32)))

        if self._movement_stats is not None:
            s = self._movement_stats
            sl = (sl - s['sl_mean']) / s['sl_std']
            log_sl = (log_sl - s['log_sl_mean']) / s['log_sl_std']
            cos_ta = (cos_ta - s['cos_ta_mean']) / s['cos_ta_std']

        return torch.stack([sl, log_sl, cos_ta], dim=0)

    def _compute_interactions(
        self, features_a: torch.Tensor, features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interaction terms between two sets of features.

        Args:
            features_a: Tensor of shape (K, N)
            features_b: Tensor of shape (M, N)

        Returns:
            Tensor of shape (K * M, N)
        """
        K, N = features_a.shape
        M = features_b.shape[0]
        interactions = features_a.unsqueeze(1) * features_b.unsqueeze(0)
        return interactions.reshape(K * M, N)

    @property
    def individual_mapping(self) -> dict:
        """Return a copy of the individual-to-index mapping."""
        return self._individual_to_idx.copy()

    @property
    def n_individuals(self) -> int:
        """Return the number of unique individuals in the dataset."""
        return self._n_individuals

    def __getitem__(self, idx) -> tuple:
        """
        Load and process all layers for a single sample, including movement
        features, optional time features, and interaction terms.

        Returns:
            Tuple of:
                - step_pixels: Tensor (C, 101) with env + movement [+ time] + interactions
                - valid_pixels: Tensor (C, 101) validity mask
                - {}: Empty dict placeholder
                - str: TARGET_CRS
        """
        # Compute candidate points + movement data (always needed for movement features)
        lons, lats, xs, ys, bounds_4326, step_lengths, turn_angles = (
            self._compute_candidate_points(idx)
        )

        # Load env features from cache
        env_pixels = self._feature_cache['step_pixels'][idx].clone()
        env_valid = self._feature_cache['valid_pixels'][idx].clone()

        # Apply normalization to env features only
        if self.normalizer is not None:
            env_pixels = self.normalizer(env_pixels)

        N = len(lons)
        n_env = self._total_channels

        # Compute movement features (3, N)
        move_feats = self._compute_movement_features(step_lengths, turn_angles)
        move_valid = torch.ones(3, N, dtype=torch.bool)

        # move × env interactions
        move_env_inter = self._compute_interactions(move_feats, env_pixels)
        move_env_valid = env_valid.unsqueeze(0).expand(3, -1, -1).reshape(3 * n_env, N)

        # Optional time features
        if self.include_time:
            time_feats = self._compute_time_features(idx)
            time_valid = torch.ones(4, N, dtype=torch.bool)
        else:
            time_feats = None
            time_valid = None

        # Optional individual ID features
        if self.include_individual_id:
            indiv_feats = self._compute_individual_id_features(idx)
            indiv_valid = torch.ones(self._n_individuals, N, dtype=torch.bool)
        else:
            indiv_feats = None
            indiv_valid = None

        # Build feature list: env | movement | time | individual_id | interactions
        feat_parts = [env_pixels, move_feats]
        valid_parts = [env_valid, move_valid]

        if self.include_time:
            feat_parts.append(time_feats)
            valid_parts.append(time_valid)

        if self.include_individual_id:
            feat_parts.append(indiv_feats)
            valid_parts.append(indiv_valid)

        # move × env interactions
        feat_parts.append(move_env_inter)
        valid_parts.append(move_env_valid)

        if self.include_time:
            # time × env interactions (4 * n_env, N)
            time_env_inter = self._compute_interactions(time_feats, env_pixels)
            time_env_valid = env_valid.unsqueeze(0).expand(4, -1, -1).reshape(4 * n_env, N)
            feat_parts.append(time_env_inter)
            valid_parts.append(time_env_valid)

            # time × move interactions (4 * 3, N)
            time_move_inter = self._compute_interactions(time_feats, move_feats)
            time_move_valid = torch.ones(4 * 3, N, dtype=torch.bool)
            feat_parts.append(time_move_inter)
            valid_parts.append(time_move_valid)

        step_pixels = torch.cat(feat_parts, dim=0)
        valid_pixels = torch.cat(valid_parts, dim=0)

        return step_pixels, valid_pixels, {}, TARGET_CRS


def grouped_nll(beta_vec, X_t, y_t, starts_t, ends_t):
    """
    Compute grouped negative log-likelihood for conditional logistic regression.

    Args:
        beta_vec: Parameter vector
        X_t: Feature matrix
        y_t: Binary labels (1 for chosen, 0 for available)
        starts_t: Group start indices
        ends_t: Group end indices

    Returns:
        Mean NLL across groups
    """
    total = beta_vec.new_tensor(0.0)
    for s_t, e_t in zip(starts_t, ends_t):
        s, e = int(s_t), int(e_t)

        Xg = X_t[s:e]
        yg = y_t[s:e]
        logits = Xg @ beta_vec

        logZ = torch.logsumexp(logits, dim=0)
        jstar = torch.nonzero(yg, as_tuple=True)[0].item()
        total = total - (logits[jstar] - logZ)

    return total / max(len(starts_t), 1)


def grouped_nll_vectorized(beta_vec, X_t, K):
    """
    Vectorized grouped NLL for conditional logistic regression.

    Assumes data is structured as N groups of K candidates each, where the
    last candidate in each group (index K-1) is the true choice.

    Args:
        beta_vec: Parameter vector of shape (C,)
        X_t: Feature matrix of shape (N * K, C)
        K: Number of candidates per group (e.g., 101)

    Returns:
        Mean NLL across groups
    """
    # Compute all logits at once: (N * K,)
    logits = X_t @ beta_vec

    # Reshape to (N, K) where each row is a choice set
    N = logits.shape[0] // K
    logits = logits.view(N, K)

    # True choice is always the last column (index K-1)
    true_logits = logits[:, -1]  # (N,)

    # Log-sum-exp over candidates (dim=1)
    logZ = torch.logsumexp(logits, dim=1)  # (N,)

    # NLL = -sum(true_logit - logZ) / N
    return -(true_logits - logZ).mean()


def grouped_nll_mlp_vectorized(model, X_t, K):
    """
    Vectorized grouped NLL for MLP model.

    Assumes data is structured as N groups of K candidates each, where the
    last candidate in each group (index K-1) is the true choice.

    Args:
        model: MLP model that outputs scores
        X_t: Feature matrix of shape (N * K, C)
        K: Number of candidates per group (e.g., 101)

    Returns:
        Mean NLL across groups
    """
    # Compute all scores at once: (N * K,)
    scores = model(X_t)

    # Reshape to (N, K) where each row is a choice set
    N = scores.shape[0] // K
    logits = scores.view(N, K)

    # True choice is always the last column (index K-1)
    true_logits = logits[:, -1]  # (N,)

    # Log-sum-exp over candidates (dim=1)
    logZ = torch.logsumexp(logits, dim=1)  # (N,)

    # NLL = -sum(true_logit - logZ) / N
    return -(true_logits - logZ).mean()
