"""PyTorch datasets for movement ecology."""

import os
import json
import glob

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import rasterio
from pyproj import Transformer

from movebench.config import PATHS, LAYERS, YEAR_BOUNDS, LON_COL, LAT_COL, POINT_ID_COL
from movebench.utils import lat_lon_to_modis_tiles, setup_df_for_ssf
from movebench.loaders import (
    load_geotiff,
    load_geotiff_cached,
    load_geotiff_window,
    load_merged_hdf_window,
)
from movebench.transforms import reproject_to_meters, Normalizer


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
        self.chip_dir = study_dir / "chips" / split
        self.dem_dir = study_dir / "tiles" / "dem"

        # Load DEM tile index
        dem_index_path = self.dem_dir / "index.json"
        with open(dem_index_path, 'r') as f:
            self.dem_manifest = json.load(f)

        self.hii_dir = str(PATHS["hii"])
        self.cvf_dir = str(PATHS["modis_cvf"])
        self.lc_dir = str(PATHS["modis_lc"])
        self.bioclim_dir = str(PATHS["bioclim"])

        # Build MODIS filepath indices to avoid globs later
        self.modis_lc_index = {}
        for path in glob.glob(self.lc_dir + "/MCD12Q1.A*.h*v*.hdf"):
            fname = os.path.basename(path)
            year = int(fname.split(".A")[1][:4])
            h = int(fname.split(".h")[1][:2])
            v = int(fname.split("v")[1][:2])
            self.modis_lc_index[(int(year), int(h), int(v))] = path

        self.modis_cvf_index = {}
        for path in glob.glob(self.cvf_dir + "/MOD44B.A*.h*v*.hdf"):
            fname = os.path.basename(path)
            year = int(fname.split(".A")[1][:4])
            h = int(fname.split(".h")[1][:2])
            v = int(fname.split("v")[1][:2])
            self.modis_cvf_index[(int(year), int(h), int(v))] = path

        # Normalizer: use provided one or None (should be provided by Study)
        self.normalizer = normalizer
        self._normalizer_provided = normalizer is not None
        self._lc_nc = LAYERS["modis_lc"].num_classes
        self._categorical_start_idx = None

    def __getitem__(self, idx):
        row = self.idx_to_timestamp.iloc[idx]
        point_id = int(row[POINT_ID_COL])
        timestamp = row['timestamp']
        all_layers = []
        layer_names = []

        # ERA5 chips (new naming: {point_id:05d}_era5.tif)
        big_chip = load_geotiff(self.chip_dir / f"{point_id:05d}_era5.tif")
        # MODIS chips (new naming: {point_id:05d}_modis.tif)
        med_chip = load_geotiff(self.chip_dir / f"{point_id:05d}_modis.tif")
        all_layers.extend([med_chip, big_chip])
        layer_names.extend(["modis", "era5"])

        # Get location for other queries
        bounds = med_chip['bounds']
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

        # DEM - bigger tile (new structure: tiles key instead of chunk_info)
        for chunk in self.dem_manifest.get("tiles", []):
            b = chunk["bounds"]
            if b["min_lon"] <= center_lon <= b["max_lon"] and b["min_lat"] <= center_lat <= b["max_lat"]:
                demf = os.path.join(self.dem_dir, chunk["file"])
                dem_raster = load_geotiff_cached(demf)
                break
        all_layers.append(dem_raster)
        layer_names.append("DEM")

        # MODIS land cover -- annual, hdf files
        year = int(max(min(timestamp.year, YEAR_BOUNDS["modis_lc"][1]), YEAR_BOUNDS["modis_lc"][0]))
        hvs = []
        for lat in [bounds[1], bounds[3]]:
            for lon in [bounds[0], bounds[2]]:
                hvs.extend(lat_lon_to_modis_tiles(lat, lon))
        sd_paths = []
        for (h, v) in hvs:
            hdf_path_i = self.modis_lc_index[(year, int(h), int(v))]
            sd_paths.append("HDF4_EOS:EOS_GRID:" + hdf_path_i + ":MCD12Q1:LC_Type1")
        lc_raster = load_merged_hdf_window(tuple(sd_paths), bounds)
        all_layers.append(lc_raster)
        layer_names.append("modis_lc")

        # MODIS vegetation cover -- annual
        cvf_rasters = []
        for band in [':MOD44B_250m_GRID:Percent_Tree_Cover',
                     ':MOD44B_250m_GRID:Percent_NonTree_Vegetation',
                     ':MOD44B_250m_GRID:Percent_NonVegetated']:
            sd_paths = []
            for (h, v) in hvs:
                hdf_path_i = self.modis_cvf_index[(year, int(h), int(v))]
                sd_paths.append("HDF4_EOS:EOS_GRID:" + hdf_path_i + band)
            cvf_raster = load_merged_hdf_window(tuple(sd_paths), bounds)
            cvf_rasters.append(cvf_raster)
        all_layers.extend(cvf_rasters)
        layer_names.extend(["modis_treecover", "modis_nontreeveg", "modis_nonveg"])

        # Human impact -- annual
        year = max(min(timestamp.year, YEAR_BOUNDS["hii"][1]), YEAR_BOUNDS["hii"][0])
        hii_f = os.path.join(self.hii_dir, f'hii_{year}-01-01.tif')
        hii_raster = load_geotiff_window(hii_f, med_chip['bounds'])
        all_layers.append(hii_raster)
        layer_names.append("HII")

        # BioClim -- no time; whole world
        bioclim_fs = [os.path.join(self.bioclim_dir, f"wc2.1_30s_bio_{i}.tif") for i in range(1, 20)]
        bioclim_rasters = [load_geotiff_window(bioclim_f, med_chip['bounds']) for bioclim_f in bioclim_fs]
        all_layers.extend(bioclim_rasters)
        layer_names.extend([f"bioclim_{i}" for i in range(1, 20)])

        # Reproject all to same resolution/CRS/bounds
        reprojected, final_transform, final_crs = reproject_to_meters(all_layers)
        reprojected_t = [torch.from_numpy(l) for l in reprojected]

        # Convert classification data to one-hot
        lc_idx = layer_names.index("modis_lc")
        lc_nc = LAYERS["modis_lc"].num_classes
        reprojected_t[lc_idx][reprojected_t[lc_idx] > 17] = 0
        reprojected_t[lc_idx] = torch.nn.functional.one_hot(
            reprojected_t[lc_idx].long(),
            num_classes=lc_nc
        ).squeeze(0).permute(2, 0, 1)

        # Compute the nodata mask
        nodata_vals = [a['nodata'] or torch.inf for a in all_layers]
        valid_masks = [(layer != nodata_val) for layer, nodata_val in zip(reprojected_t, nodata_vals)]

        # Concat and standardize columns if appropriate
        final_t = torch.cat(reprojected_t)
        lc_start_idx = sum([a['array'].shape[0] for a in all_layers[:lc_idx]])

        # Set categorical start index on first call (for SSFDataset interpolation)
        if self._categorical_start_idx is None:
            num_channels = final_t.shape[0]
            self._categorical_start_idx = lc_start_idx

            if self.normalizer is not None:
                self.norm_mask = self.normalizer.normalize_mask
            else:
                # Create norm_mask even without normalizer (for SSFDataset interpolation)
                self.norm_mask = torch.ones(num_channels, dtype=torch.bool)
                self.norm_mask[lc_start_idx:lc_start_idx + self._lc_nc] = False

        # Apply normalization if available
        if self.normalizer is not None:
            final_t = self.normalizer(final_t)

        return final_t, torch.cat(valid_masks), final_transform, final_crs

    def __len__(self):
        return len(self.idx_to_timestamp)


class SSFDataset(ChipDataset):
    """
    Step Selection Function dataset.

    Extends ChipDataset to sample pseudo-absence points around each
    observed step, for use in conditional logistic regression.
    """

    def __init__(self,
                 study_id,
                 df,
                 split,
                 study_dir=None,
                 random_seed: int = 0,
                 num_candidates: int = 100,
                 filter_quantile: float = 0.98,
                 normalizer=None):
        """
        Initialize SSFDataset.

        Args:
            study_id: Study identifier
            df: DataFrame with location and timestamp columns (already filtered to train/test)
            split: Data split ("train" or "test")
            study_dir: Study directory (default: PATHS["studies"] / study_id)
            random_seed: Random seed for reproducibility
            num_candidates: Number of pseudo-absence candidates per step
            filter_quantile: Quantile for step length filtering
            normalizer: Optional pre-computed Normalizer instance
        """
        self.df = setup_df_for_ssf(df.copy())
        self.sample_indices = self.df.index[self.df['can_be_ssf_start']].to_numpy()

        # Build idx_to_timestamp DataFrame with index, timestamp, and point_id
        idx_to_timestamp = pd.DataFrame({
            'index': self.sample_indices,
            'timestamp': pd.to_datetime(self.df.loc[self.sample_indices, 'timestamp']),
            POINT_ID_COL: self.df.loc[self.sample_indices, POINT_ID_COL],
        })

        super().__init__(
            study_id,
            idx_to_timestamp=idx_to_timestamp,
            split=split,
            study_dir=study_dir,
            normalizer=normalizer,
        )

        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)
        self.num_candidates = num_candidates
        self.filter_quantile = filter_quantile
        self.valid_lengths = None
        self.valid_turns = None
        self._sl_scale = None
        self._log_sl_mean = None
        self._log_sl_std = None

    def __getitem__(self, idx):
        # Get chip and spatial info from parent
        chip, valid_mask, chip_transform, chip_crs = super().__getitem__(idx)
        C, H, W = chip.shape

        # Map dataset idx -> (original df index, timestamp)
        orig_idx, timestamp = self.idx_to_timestamp.iloc[idx].values
        orig_idx = int(orig_idx)

        # Start location
        row = self.df.loc[orig_idx]
        start_lat = row[LAT_COL]
        start_long = row[LON_COL]
        start_bearing = row['bearing']

        # Sample candidate steps (pseudo-negatives)
        steps_m, turns_rad = self._get_pseudo_steps_and_turns(
            self.num_candidates, self.rng
        )
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        x0, y0 = transformer.transform(start_long, start_lat)
        steps_m = np.asarray(steps_m, dtype=float)
        turns_rad = np.asarray(turns_rad, dtype=float)
        headings = start_bearing + turns_rad
        headings = (headings + np.pi) % (2 * np.pi) - np.pi
        dx = steps_m * np.sin(headings)
        dy = steps_m * np.cos(headings)
        xs = x0 + dx
        ys = y0 + dy

        # Append true endpoint
        next_idx = int(row['next_index'])
        next_row = self.df.loc[next_idx]
        end_lat = next_row[LAT_COL]
        end_long = next_row[LON_COL]
        x1, y1 = transformer.transform(end_long, end_lat)
        xs = np.append(xs, x1)
        ys = np.append(ys, y1)

        # Get features at (x,y) locations
        # Nearest neighbor method for mask and categorical layers
        rows, cols = rasterio.transform.rowcol(chip_transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        H, W = chip.shape[1:]
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_valid = rows[valid]
        cols_valid = cols[valid]
        assert len(rows_valid) == self.num_candidates + 1, f"only {len(rows_valid)} valid points"
        snapped_pixels = chip[:, rows_valid, cols_valid]
        valid_pixels = valid_mask[:, rows_valid, cols_valid]

        # Bilinear interpolation for continuous layers
        inv = ~chip_transform
        cols_f, rows_f = inv * (xs, ys)
        valid = (rows_f >= 0) & (rows_f < H) & (cols_f >= 0) & (cols_f < W)
        rows_valid = rows_f[valid]
        cols_valid = cols_f[valid]
        assert len(rows_valid) == self.num_candidates + 1, f"only {len(rows_valid)} valid points"
        cols_t = torch.as_tensor(cols_valid, dtype=torch.float32, device=chip.device)
        rows_t = torch.as_tensor(rows_valid, dtype=torch.float32, device=chip.device)
        x_norm = 2.0 * (cols_t / (W - 1)) - 1.0
        y_norm = 2.0 * (rows_t / (H - 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1)
        grid = grid.view(1, -1, 1, 2)
        raster_batched = chip.unsqueeze(0)
        samples = F.grid_sample(
            raster_batched,
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',
        )
        interp_pixels = samples.squeeze(0).squeeze(-1)

        step_pixels = torch.where(self.norm_mask[:, None], interp_pixels, snapped_pixels)

        return step_pixels, valid_pixels, chip_transform, chip_crs

    def _get_pseudo_steps_and_turns(self, total_random_samples: int, rng: np.random.Generator):
        """Sample pseudo step lengths and turn angles from empirical distribution."""
        valid_lengths = self.df.loc[self.df['valid_step'], 'step_length_m']
        valid_turns = self.df['turn_angle_rad'][~np.isnan(self.df['turn_angle_rad'])]
        pseudo_lens = rng.choice(valid_lengths, size=total_random_samples, replace=True) if total_random_samples > 0 else np.array([])
        pseudo_turns = rng.choice(valid_turns, size=total_random_samples, replace=True) if total_random_samples > 0 else np.array([])
        return pseudo_lens, pseudo_turns

    def __len__(self):
        return len(self.sample_indices)


class iSSFDataset(SSFDataset):
    """
    Integrated SSF dataset with parametric movement distributions.

    Uses fitted gamma (step length) and von Mises (turn angle) distributions
    instead of empirical resampling.
    """

    def _get_pseudo_steps_and_turns(self, total_random_samples: int, rng: np.random.Generator):
        """Sample from fitted gamma (step) and von Mises (turn) distributions."""
        from scipy import stats

        pseudo_lens, pseudo_turns = super()._get_pseudo_steps_and_turns(total_random_samples, rng)
        valid_lengths = self.df['step_length_m']
        valid_turns = self.df['turn_angle_rad'][~np.isnan(self.df['turn_angle_rad'])]

        try:
            self.gamma_params = stats.gamma.fit(valid_lengths, floc=0)
            pseudo_lens = stats.gamma.rvs(*self.gamma_params, size=total_random_samples, random_state=rng)
        except Exception:
            print("Gamma fit failed, using empirical distribution.")

        try:
            self.vonmises_params = stats.vonmises.fit(valid_turns)
            pseudo_turns = stats.vonmises.rvs(*self.vonmises_params, size=total_random_samples, random_state=rng)
        except Exception:
            print("Von Mises fit failed, using empirical distribution.")

        return pseudo_lens, pseudo_turns


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
