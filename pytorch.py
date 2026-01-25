"""
PyTorch datasets for MoveBench.

DEPRECATED: This module is deprecated. Please import from movebench.datasets instead:

    from movebench.datasets import ChipDataset, SSFDataset, iSSFDataset

This file is kept for backward compatibility during the migration period.

requires:
    conda install -c conda-forge gdal
    conda install -c conda-forge libgdal-hdf4
    conda install rasterio  # needs conda version for HDF support
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "Importing from pytorch.py is deprecated. "
    "Use 'from movebench.datasets import ChipDataset, SSFDataset, iSSFDataset' instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
from functools import lru_cache
import glob
import json
import numpy as np
import pandas as pd
from io import StringIO
import re

import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds, calculate_default_transform
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.transform import array_bounds
from rasterio.enums import Resampling
from pyproj import Transformer

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from movebench.config import (
    PATHS, LAYERS,
    LON_COL, LAT_COL, TIME_COL, INDIVIDUAL_COL,
    YEAR_BOUNDS,
)

# MODIS grid - lazy loaded
_modis_grid = None

def _load_modis_grid():
    """Load MODIS sinusoidal grid bounds (lazy)."""
    global _modis_grid
    if _modis_grid is None:
        grid_file = PATHS.get("modis_grid", "sn_bound_10deg.txt")
        with open(grid_file) as f:
            raw = f.read()
        lines = [ln for ln in raw.splitlines() if ln.strip() and re.match(r"^\d", ln.strip())]
        buf = StringIO("\n".join(lines))
        _modis_grid = pd.read_csv(buf, sep=r'\s+', header=None,
                                   names=["iv", "ih", "lon_min", "lon_max", "lat_min", "lat_max"])
    return _modis_grid

def lat_lon_to_modis(ctr_lat, ctr_long):
    """Get MODIS tile (h, v) for a lat/lon coordinate."""
    grid = _load_modis_grid()
    rows = grid[(grid.lon_min <= ctr_long) & (ctr_long <= grid.lon_max) &
                (grid.lat_min <= ctr_lat) & (ctr_lat <= grid.lat_max)]
    return list(zip(rows.ih.values, rows.iv.values))

### rasterio methods ###

def _get_rasterio(path):
    with rasterio.open(path) as ds:
        arr = ds.read()
        return {
                    "array": arr,
                    "transform": ds.transform,
                    "crs": ds.crs,
                    "nodata": ds.nodata,
                    "bounds": ds.bounds,
                    "bands": ds.descriptions
                }

@lru_cache(maxsize=16)
def _get_rasterio_cached(path):
    return _get_rasterio(path)

@lru_cache(maxsize=20)
def _get_rasterio_window_cached(path, bounds):
    with rasterio.open(path) as ds:
        min_lon, min_lat, max_lon, max_lat = bounds
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, ds.transform)
        return {
            "array": ds.read(window=window),
            "transform": ds.window_transform(window),
            "crs": ds.crs,
            "nodata": ds.nodata,
            "bounds": bounds, # rasterio.windows.bounds(window, ds.transform),
        }

@lru_cache(maxsize=20)
def _get_merged_rasterio_window_cached(paths, bounds_latlon):
    """Applies CRF bounds transform (to accomodate MODIS)
    """
    datasets = [rasterio.open(p) for p in paths]
    try:
        crs = datasets[0].crs
        descriptions = datasets[0].descriptions
        nodata_val = datasets[0].nodata or 255

        # transform to MODIS CRS
        left, bottom, right, top = transform_bounds(
            "EPSG:4326", # bounds assumed to come from EPSG:4326
            crs,         # but we want to use them in the target (e.g. MODIS) CRS
            *bounds_latlon,
        )

        mosaic, out_transform = merge(
            datasets,
            bounds=(left, bottom, right, top),
            nodata=nodata_val,
            resampling=rasterio.enums.Resampling.nearest,
        )
    except:
        print("Oh no! In _get_merged_rasterio_window_cached")
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

def _reproject_all(layers):
    """Return all layers reprojected as float32."""
    ref_raster = layers[0]
    _, H_ref, W_ref = ref_raster['array'].shape
    
    aligned = [ ref_raster['array'] ]
    dst_crs = ref_raster['crs']
    dst_transform = ref_raster['transform']
    for layer in layers[1:]:
        # create new np array as destination - same channels as this layer but with target width/height
        dst = np.full((layer["array"].shape[0], H_ref, W_ref), np.nan, dtype=np.float32)
        reproject(
            source=layer["array"], # (C, H_src, W_src)
            destination=dst,
            src_transform=layer["transform"],
            src_crs=layer["crs"],
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,   # nearest or bilinear
        )
        aligned.append(dst)

    return aligned

def _reproject_all_to_meters(layers, dst_crs="EPSG:3857"):
    ref_raster = layers[0]
    _, H_ref, W_ref = ref_raster['array'].shape

    # compute transform/shape in metric CRS
    dst_transform, dst_width, dst_height = calculate_default_transform(
        ref_raster["crs"],     # source CRS (lat/lon)
        dst_crs,               # target CRS (meters)
        W_ref, H_ref,
        *ref_raster['bounds'], # in ref_raster["crs"]
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


class Normalizer:
    """
    Z-score normalization with selective channel mask.

    Loads existing {study}_mean.pt and {study}_std.pt files.
    Skips normalization for categorical layers (e.g., one-hot MODIS LC).
    """
    def __init__(self, mean, std, normalize_mask):
        self.mean = mean
        self.std = std
        self.normalize_mask = normalize_mask

    def __call__(self, x):
        """Apply normalization to tensor x of shape (C, H, W) or (C, N)."""
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


class ChipDataset(Dataset):
    def __init__(self,  study_id, idx_to_timestamp, chip_base_dir=None, dem_base_dir=None):
        self.study_id = study_id
        self.idx_to_timestamp = idx_to_timestamp.reset_index() # reset index to allow easy iteration but retain old index

        chip_base_dir = chip_base_dir or str(PATHS["legacy_chips"])
        dem_base_dir = dem_base_dir or str(PATHS["legacy_dem"])

        self.chip_dir = os.path.join(chip_base_dir, study_id)
        self.dem_dir  = os.path.join(dem_base_dir, study_id)
        self.dem_manifest = json.load(open(os.path.join(self.dem_dir, 'dem_manifest.json'), 'r'))

        self.hii_dir = str(PATHS["hii"])
        self.cvf_dir = str(PATHS["modis_cvf"])
        self.lc_dir = str(PATHS["modis_lc"])
        self.bioclim_dir = str(PATHS["bioclim"])

        # build modis filepath indices to avoid globs later
        # self.lc_dir + f"/MCD12Q1.A{year}001.h{h:02d}v{v:02d}*.hdf"
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

        # Normalizer is loaded lazily on first __getitem__ call
        # (needs to know num_channels and categorical indices)
        self.normalizer = None
        self._lc_nc = LAYERS["modis_lc"].num_classes  # 18
        self._categorical_start_idx = None

        # Check if normalization files exist
        if not os.path.exists(f'{study_id}_mean.pt'):
            print("WARNING: Features will not be standardized for", study_id)

    def __getitem__(self, idx):
        idx, timestamp = self.idx_to_timestamp.iloc[idx].values # map to original df index
        idx = int(idx)
        all_layers = []
        layer_names = []
        
        # ERA5 chips
        big_chip   = _get_rasterio(os.path.join(self.chip_dir, f"point_{idx}_big.tif"))
        med_chip   = _get_rasterio(os.path.join(self.chip_dir, f"point_{idx}_medium.tif"))
        # print("big:", big_chip['array'].shape)
        # print("med:", med_chip['array'].shape)
        # right now, small is just dynamic world, and it's broken
        # small_chip = _get_rasterio(os.path.join(self.chip_dir, f"point_{idx}_small.tif"))
        all_layers.extend([med_chip, big_chip])
        layer_names.extend(["era5_med", "era5_big"]) # TODO: would need med_bands, big_bands to access these later for e.g. derived features

        # get location for other queries -- assume small will be the most accurate
        bounds = med_chip['bounds']
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2

        # DEM - bigger tile
        for chunk in self.dem_manifest.get("chunk_info", []):
            b = chunk["bounds"]
            if b["min_lon"] <= center_lon <= b["max_lon"] and b["min_lat"] <= center_lat <= b["max_lat"]:
                demf = os.path.join(self.dem_dir, chunk["filename"])
                dem_raster = _get_rasterio_cached(demf)
                # dem_raster = _get_rasterio_window_cached(demf, med_chip['bounds'])
                break
        # print("DEM", dem_raster['array'].shape)
        all_layers.append(dem_raster)
        layer_names.append("DEM")
        
        # MODIS land cover -- annual, hdf files
        # int-valued (should be one-hot as features)
        year = int(max(min(timestamp.year, YEAR_BOUNDS["modis_lc"][1]), YEAR_BOUNDS["modis_lc"][0]))
        # get all files that the bounds fall into - will need to merge them
        hvs = []
        for lat in [ bounds[1], bounds[3] ]:
            for lon in [ bounds[0], bounds[2] ]:
                hvs.extend(lat_lon_to_modis(lat, lon))
        sd_paths = []
        for (h, v) in hvs:
            hdf_path_i = self.modis_lc_index[(year, int(h), int(v))]
            sd_paths.append("HDF4_EOS:EOS_GRID:" + hdf_path_i + ":MCD12Q1:LC_Type1")
        lc_raster = _get_merged_rasterio_window_cached(tuple(sd_paths), bounds)
        # print("MODIS LC", lc_raster['array'].shape)
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
            # print("merging", sd_paths)
            cvf_raster = _get_merged_rasterio_window_cached(tuple(sd_paths), bounds)
            cvf_rasters.append(cvf_raster)
        # print("MODIS CVF", [cvf_raster['array'].shape for cvf_raster in cvf_rasters])
        all_layers.extend(cvf_rasters)
        layer_names.extend(["modis_treecover", "modis_nontreeveg", "modis_nonveg"])

        # Human impact -- annual
        year = max(min(timestamp.year, YEAR_BOUNDS["hii"][1]), YEAR_BOUNDS["hii"][0])
        hii_f = os.path.join(self.hii_dir, f'hii_{year}-01-01.tif')
        hii_raster = _get_rasterio_window_cached(hii_f, med_chip['bounds'])
        # print("HII", hii_raster['array'].shape)
        all_layers.append(hii_raster)
        layer_names.append("HII")

        # BioClim -- no time; whole world
        bioclim_fs = [ os.path.join(self.bioclim_dir, f"wc2.1_30s_bio_{i}.tif") for i in range(1,20) ] 
        bioclim_rasters = [ _get_rasterio_window_cached(bioclim_f, med_chip['bounds']) for bioclim_f in bioclim_fs ]
        # print("bioclim", [raster['array'].shape for raster in bioclim_rasters])
        all_layers.extend(bioclim_rasters)
        layer_names.extend([ f"bioclim_{i}" for i in range(1,20) ])

        # reproject all to same resolution/CRS/bounds
        # and convert to torch
        # TODO project to meters, not lat/long
        # reprojected = _reproject_all(all_layers)
        reprojected, final_transform, final_crs = _reproject_all_to_meters(all_layers)
        reprojected_t =[
            torch.from_numpy(l) for l in reprojected
        ]

        # TODO add derived types - e.g. uplift
        # ...later...

        # convert classification data to one-hot
        lc_idx = layer_names.index("modis_lc")
        lc_nc  = LAYERS["modis_lc"].num_classes  # 18: 17 classes + 1 for invalid/masked
        reprojected_t[lc_idx][reprojected_t[lc_idx] > 17] = 0 # make any invalid entries 0
        reprojected_t[lc_idx] = torch.nn.functional.one_hot(reprojected_t[lc_idx].long(), 
                                                            num_classes=lc_nc).squeeze(0).permute(2,0,1)
        
        # compute the nodata mask
        nodata_vals  = [ a['nodata'] or torch.inf for a in all_layers ] # need to 'or' to catch None values, which mess up the equality test
        valid_masks  = [ (layer != nodata_val) for layer, nodata_val in zip(reprojected_t, nodata_vals) ]

        # concat and standardize columns if appropriate
        final_t = torch.cat(reprojected_t)
        lc_start_idx = sum([ a['array'].shape[0] for a in all_layers[:lc_idx] ])

        # Load normalizer on first call (now we know num_channels)
        if self.normalizer is None and self._categorical_start_idx is None:
            num_channels = final_t.shape[0]
            self.normalizer = Normalizer.load(
                self.study_id,
                num_channels,
                lc_start_idx,
                lc_start_idx + self._lc_nc
            )
            self._categorical_start_idx = lc_start_idx
            # Store norm_mask for backward compatibility with SSFDataset
            if self.normalizer is not None:
                self.norm_mask = self.normalizer.normalize_mask

        # Apply normalization if available
        if self.normalizer is not None:
            final_t = self.normalizer(final_t)

        return final_t, torch.cat(valid_masks), final_transform, final_crs

    def __len__(self):
        return len(self.idx_to_timestamp)

def setup_df_for_ssf(df, step_multiplier=2, step_q=0.95, gap_q=0.999):
    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL])

    # sort so diffs make sense; keep original index (chip ids)
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL])

    df['lat_rad'] = np.deg2rad(df[LAT_COL])
    df['lon_rad'] = np.deg2rad(df[LON_COL])

    g = df.groupby(INDIVIDUAL_COL)

    # ---- time gaps per individual ----
    df['time_since_last'] = g[TIME_COL].diff()
    per_indiv_cutoff = g['time_since_last'].transform(
        lambda s: s.dropna().quantile(gap_q)
    )
    df['within_time'] = df['time_since_last'] <= per_indiv_cutoff

    # ---- step length per individual ----
    R = 6371000.0  # Earth radius in meters
    df['lat_prev'] = g['lat_rad'].shift()
    df['lon_prev'] = g['lon_rad'].shift()
    dlat = df['lat_rad'] - df['lat_prev']
    dlon = df['lon_rad'] - df['lon_prev']

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(df['lat_prev']) * np.cos(df['lat_rad']) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df['step_length_m'] = R * c

    # initial valid step mask (time + positive distance)
    df['valid_step'] = (
        df['within_time']
        & df['step_length_m'].notna()
        & (df['step_length_m'] > 0)
    )

    # ---- step-length cutoff ----
    max_step_len = step_multiplier * np.nanquantile(
        df.loc[df['valid_step'], 'step_length_m'], q=step_q
    )
    df['max_step_len'] = max_step_len  # optional
    df['valid_step'] &= df['step_length_m'] <= max_step_len

    # ---- turn angles ----
    y = np.sin(dlon) * np.cos(df['lat_rad'])
    x = (
        np.cos(df['lat_prev']) * np.sin(df['lat_rad'])
        - np.sin(df['lat_prev']) * np.cos(df['lat_rad']) * np.cos(dlon)
    )
    df['bearing'] = np.arctan2(y, x)  # [-pi, pi], NaN at first row per individual

    df['bearing_prev'] = g['bearing'].shift()
    df['valid_step_prev'] = g['valid_step'].shift()

    turn = df['bearing'] - df['bearing_prev']
    turn = (turn + np.pi) % (2.0 * np.pi) - np.pi  # wrap

    df['turn_angle_rad'] = np.where(
        df['valid_step'] & df['valid_step_prev'],
        turn,
        np.nan,
    )

    # ---- bursts ----
    df['is_first_in_indiv'] = g.cumcount() == 0
    df['burst_start'] = df['is_first_in_indiv'] | (~df['valid_step'])
    df['burst_id'] = g['burst_start'].cumsum()

    # ---- SSF sample starts ----
    # index of the next row in time order
    df['next_index'] = df.index.to_series().shift(-1).astype('Int64')

    same_indiv_next = df[INDIVIDUAL_COL].eq(
        df[INDIVIDUAL_COL].shift(-1)
    )
    same_burst_next = df['burst_id'].eq(df['burst_id'].shift(-1))
    valid_step_next = df['valid_step'].shift(-1).fillna(False)  # step from current -> next

    df['can_be_ssf_start'] = (
        ~df['is_first_in_indiv'] &
        same_indiv_next &
        same_burst_next &
        valid_step_next
    )

    # optional - drop helpers you don't care about downstream
    df = df.drop(
        columns=[
            'lat_rad',
            'lon_rad',
            'lat_prev',
            'lon_prev',
            'bearing_prev',
            'burst_start',
            'is_first_in_indiv',
        ]
    )

    return df


class SSFDataset(ChipDataset):
    def __init__(self,  
                 study_id, 
                 df, 
                 random_seed: int = 0,
                 num_candidates: int = 100,
                 filter_quantile: float = 0.98, # TODO
                 chip_base_dir='chips_out', 
                 dem_base_dir='dem_out'):
        """Assume df is already filtered to train or test"""

        self.df = setup_df_for_ssf(df.copy())
        # super().__init__(study_id, idx_to_timestamp=pd.to_datetime(self.df.timestamp), chip_base_dir=chip_base_dir, dem_base_dir=dem_base_dir)
        self.sample_indices = self.df.index[self.df['can_be_ssf_start']].to_numpy()
        timestamps = pd.to_datetime(self.df.loc[self.sample_indices, 'timestamp'])
        super().__init__(
            study_id,
            idx_to_timestamp=timestamps,
            chip_base_dir=chip_base_dir,
            dem_base_dir=dem_base_dir,
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
        # get chip and spatial info from parent
        chip, valid_mask, chip_transform, chip_crs = super().__getitem__(idx)
        C, H, W = chip.shape

        # map dataset idx -> (original df index, timestamp)
        orig_idx, timestamp = self.idx_to_timestamp.iloc[idx].values
        orig_idx = int(orig_idx)

        # start location
        row = self.df.loc[orig_idx]
        start_lat = row[LAT_COL]
        start_long = row[LON_COL]
        start_bearing = row['bearing']

        # sample candidate steps (pseudo-negatives)
        steps_m, turns_rad = self._get_pseudo_steps_and_turns(
            self.num_candidates, self.rng
        )
        transformer = Transformer.from_crs("EPSG:4326", chip_crs, always_xy=True)
        x0, y0 = transformer.transform(start_long, start_lat)
        steps_m = np.asarray(steps_m, dtype=float)
        turns_rad = np.asarray(turns_rad, dtype=float)
        headings = start_bearing + turns_rad
        headings = (headings + np.pi) % (2 * np.pi) - np.pi # wrap to [-pi, pi]
        dx = steps_m * np.sin(headings)
        dy = steps_m * np.cos(headings)
        xs = x0 + dx
        ys = y0 + dy

        # append true endpoint
        next_idx = int(row['next_index'])
        next_row = self.df.loc[next_idx]
        end_lat = next_row[LAT_COL]
        end_long = next_row[LON_COL]
        x1, y1 = transformer.transform(end_long, end_lat)
        xs = np.append(xs, x1)
        ys = np.append(ys, y1)

        ## get features at (x,y) locations ##

        # nearest neighbor method (many map to same pixel)
        # use this on the mask (and probably any int layers)
        rows, cols = rasterio.transform.rowcol(chip_transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        H, W = chip.shape[1:]
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_valid = rows[valid]
        cols_valid = cols[valid]
        assert(len(rows_valid) == self.num_candidates + 1), f"only {len(rows_valid)} valid points"
        snapped_pixels  = chip[:, rows_valid, cols_valid] # values on grid
        valid_pixels = valid_mask[:, rows_valid, cols_valid]

        # bilinear method
        inv = ~chip_transform
        cols_f, rows_f = inv * (xs, ys)
        valid = (rows_f >= 0) & (rows_f < H) & (cols_f >= 0) & (cols_f < W)
        rows_valid = rows_f[valid]
        cols_valid = cols_f[valid]
        assert(len(rows_valid) == self.num_candidates + 1), f"only {len(rows_valid)} valid points"
        cols_t = torch.as_tensor(cols_valid, dtype=torch.float32, device=chip.device)  # (N,)
        rows_t = torch.as_tensor(rows_valid, dtype=torch.float32, device=chip.device)  # (N,)
        x_norm = 2.0 * (cols_t / (W - 1)) - 1.0   # (N,)
        y_norm = 2.0 * (rows_t / (H - 1)) - 1.0   # (N,)
        # grid: (1, N, 1, 2)  -> sampling N points in a "1 x N" output
        grid = torch.stack([x_norm, y_norm], dim=-1)   # (N, 2)
        grid = grid.view(1, -1, 1, 2)                  # (1, N, 1, 2)
        raster_batched = chip.unsqueeze(0)             # (1, C, H, W)
        samples = F.grid_sample(
            raster_batched,   # (1, C, H, W)
            grid,             # (1, N, 1, 2)
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',  # outside -> zero
        )
        interp_pixels = samples.squeeze(0).squeeze(-1)  # (1, C, N, 1) -> squeeze to (C, N)

        step_pixels = torch.where(self.norm_mask[:, None], interp_pixels, snapped_pixels)

        return step_pixels, valid_pixels, chip_transform, chip_crs

    def _get_pseudo_steps_and_turns(self, total_random_samples: int, rng: np.random.Generator):
        """Sample pseudo step lengths and turn angles. Override in subclasses."""
        valid_lengths = self.df.loc[self.df['valid_step'], 'step_length_m']
        valid_turns   = self.df['turn_angle_rad'][~np.isnan(self.df['turn_angle_rad'])]
        pseudo_lens = rng.choice(valid_lengths, size=total_random_samples, replace=True) if total_random_samples > 0 else np.array([])
        pseudo_turns = rng.choice(valid_turns, size=total_random_samples, replace=True) if total_random_samples > 0 else np.array([])
        return pseudo_lens, pseudo_turns
    
    def __len__(self):
        return len(self.sample_indices)

from scipy import stats

class iSSFDataset(SSFDataset):
    """
    Integrated SSF dataset with parametric movement distributions
    and movement-environment interaction features.
    """
    def _get_pseudo_steps_and_turns(self, total_random_samples: int, rng: np.random.Generator):
        """Sample from fitted gamma (step) and von Mises (turn) distributions."""
        pseudo_lens, pseudo_turns = super()._get_pseudo_steps_and_turns(total_random_samples, rng)
        valid_lengths = self.df['step_length_m']
        valid_turns   = self.df['turn_angle_rad'][~np.isnan(self.df['turn_angle_rad'])]

        try:
            self.gamma_params = stats.gamma.fit(valid_lengths, floc=0)
            pseudo_lens = stats.gamma.rvs(*self.gamma_params, size=total_random_samples, random_state=rng)
        except:
            print("Gamma fit failed, using empirical distribution.")

        try:
            self.vonmises_params = stats.vonmises.fit(valid_turns)
            pseudo_turns = stats.vonmises.rvs(*self.vonmises_params, size=total_random_samples, random_state=rng)
        except:
            print("Von Mises fit failed, using empirical distribution.")

        return pseudo_lens, pseudo_turns

def grouped_nll(beta_vec, X_t, y_t, starts_t, ends_t):
    total = beta_vec.new_tensor(0.0)
    for s_t, e_t in zip(starts_t, ends_t):
        s, e = int(s_t), int(e_t)
        
        Xg = X_t[s:e]

        # print('xg',Xg)
        # print("Xg nan?", torch.isnan(Xg).any())
        # print("Xg inf?", torch.isinf(Xg).any())
        # print("beta has nan?", torch.isnan(beta_vec).any())
        # print("beta has inf?", torch.isinf(beta_vec).any())
        # print("beta all finite?", torch.isfinite(beta_vec).all())
        
        yg = y_t[s:e]
        logits = Xg @ beta_vec

        # print('yg',yg)
        # print('logiits',logits)
        # print("logits has nan?", torch.isnan(logits).any())
        # print("logits has inf?", torch.isinf(logits).any())
        
        m = logits.max()
        # logZ = m + torch.log(torch.exp(logits - m).sum())
        logZ = torch.logsumexp(logits, dim=0)
        jstar = torch.nonzero(yg, as_tuple=True)[0].item()
        total = total - (logits[jstar] - logZ)

        # print('m',m)
        # print('logz', logZ)
        # print('jstar', jstar)
        # print(total)
        
    return total / max(len(starts_t), 1)
