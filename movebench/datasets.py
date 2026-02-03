"""PyTorch datasets for movement ecology."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from movebench.config import PATHS, LAYERS, POINT_ID_COL
from movebench.utils import modis_tiles_for_bounds
from movebench.loaders import (
    ChipLoader, TileLoader, get_loader,
    SlopeLoader, AspectLoader, RugosityLoader,
    OrographicUpliftLoader, ThermalUpliftLoader,
)
from movebench.transforms import reproject_to_meters, one_hot_encode


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
        self._loaders = self._init_loaders()

        # Normalizer: use provided one or None (should be provided by Study)
        self.normalizer = normalizer
        self._normalizer_provided = normalizer is not None
        self._lc_nc = LAYERS["modis_lc"].num_classes
        self._categorical_start_idx = None

    def _init_loaders(self) -> dict:
        """Initialize layer loaders from config."""
        loaders = {}

        # GEE chip loaders
        loaders["era5"] = ChipLoader(str(self.chip_dir), "era5")
        loaders["modis"] = ChipLoader(str(self.chip_dir), "modis")
        loaders["high_res"] = ChipLoader(str(self.chip_dir), "high_res")

        # GEE tile loader (DEM)
        loaders["dem"] = TileLoader(str(self.dem_dir), "dem")

        # DEM-derived loaders
        loaders["slope"] = SlopeLoader(LAYERS["slope"])
        loaders["aspect"] = AspectLoader(LAYERS["aspect"])
        loaders["rugosity"] = RugosityLoader(LAYERS["rugosity"])

        # Distance tile loaders (gracefully handle missing tiles)
        for dist_layer in ["dist_water"]:
            tile_dir = self.study_dir / "tiles" / dist_layer
            index_path = tile_dir / "index.json"
            if index_path.exists():
                loaders[dist_layer] = TileLoader(str(tile_dir), dist_layer)

        # Uplift derived loaders
        loaders["orographic_uplift"] = OrographicUpliftLoader(LAYERS["orographic_uplift"])
        loaders["thermal_uplift"] = ThermalUpliftLoader(LAYERS["thermal_uplift"])

        # Local HDF loaders (MODIS products)
        loaders["modis_lc"] = get_loader("modis_lc")
        loaders["modis_cvf_tree"] = get_loader("modis_cvf_tree")
        loaders["modis_cvf_nontree"] = get_loader("modis_cvf_nontree")
        loaders["modis_cvf_nonveg"] = get_loader("modis_cvf_nonveg")

        # Local GeoTIFF loaders
        loaders["hii"] = get_loader("hii")
        loaders["bioclim"] = get_loader("bioclim")

        return loaders

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
        point_id = context["point_id"]
        timestamp = context["timestamp"]

        all_layers = []
        layer_names = []

        # GEE chips (MODIS and ERA5)
        med_chip = self._loaders["modis"].load(point_id=point_id)
        big_chip = self._loaders["era5"].load(point_id=point_id)
        all_layers.extend([med_chip, big_chip])
        layer_names.extend(["modis", "era5"])

        # DEM tile (load all tiles intersecting bounds and merge)
        bounds = med_chip['bounds']
        dem_raster = self._loaders["dem"].load(bounds=bounds)
        all_layers.append(dem_raster)
        layer_names.append("dem")

        # DEM-derived layers (slope, aspect, rugosity)
        dem_parent = {"dem": dem_raster}
        slope_raster = self._loaders["slope"].compute(dem_parent)
        all_layers.append(slope_raster)
        layer_names.append("slope")

        aspect_raster = self._loaders["aspect"].compute(dem_parent)
        all_layers.append(aspect_raster)
        layer_names.append("aspect")

        rugosity_raster = self._loaders["rugosity"].compute(dem_parent)
        all_layers.append(rugosity_raster)
        layer_names.append("rugosity")

        # Distance tile layers (NaN placeholder if not downloaded)
        for dist_layer in ["dist_water"]:
            if dist_layer in self._loaders:
                try:
                    dist_raster = self._loaders[dist_layer].load(bounds=bounds)
                except (ValueError, FileNotFoundError):
                    H, W = dem_raster['array'].shape[1:]
                    dist_raster = {
                        'array': np.full((1, H, W), np.nan, dtype=np.float32),
                        'transform': dem_raster['transform'],
                        'crs': dem_raster['crs'],
                        'nodata': np.nan,
                        'bounds': dem_raster['bounds'],
                        'bands': LAYERS[dist_layer].bands,
                    }
            else:
                H, W = dem_raster['array'].shape[1:]
                dist_raster = {
                    'array': np.full((1, H, W), np.nan, dtype=np.float32),
                    'transform': dem_raster['transform'],
                    'crs': dem_raster['crs'],
                    'nodata': np.nan,
                    'bounds': dem_raster['bounds'],
                    'bands': LAYERS[dist_layer].bands,
                }
            all_layers.append(dist_raster)
            layer_names.append(dist_layer)

        # Orographic uplift (requires slope, aspect, ERA5 wind)
        oro_parent = {
            "slope": slope_raster,
            "aspect": aspect_raster,
            "era5_land": big_chip,  # ERA5 chip contains era5_land bands (first 12)
        }
        oro_raster = self._loaders["orographic_uplift"].compute(oro_parent)
        all_layers.append(oro_raster)
        layer_names.append("orographic_uplift")

        # Thermal uplift (requires ERA5-Land + ERA5)
        # big_chip has era5_land (12) + era5 (3) + era5_land_daily (12)
        n_era5l = len(LAYERS["era5_land"].bands)
        n_era5 = len(LAYERS["era5"].bands)
        era5l_array = big_chip['array'][:n_era5l]
        era5_array = big_chip['array'][n_era5l:n_era5l + n_era5]
        era5l_data = {**big_chip, 'array': era5l_array}
        era5_data = {**big_chip, 'array': era5_array}
        therm_parent = {
            "era5_land": era5l_data,
            "era5": era5_data,
        }
        therm_raster = self._loaders["thermal_uplift"].compute(therm_parent)
        all_layers.append(therm_raster)
        layer_names.append("thermal_uplift")

        # Compute MODIS tile coordinates for bounds
        hvs = modis_tiles_for_bounds(bounds)

        # MODIS land cover
        lc_raster = self._loaders["modis_lc"].load(bounds=bounds, timestamp=timestamp, hvs=hvs)
        all_layers.append(lc_raster)
        layer_names.append("modis_lc")

        # MODIS vegetation cover (3 bands as separate layers)
        for cvf_name in ["modis_cvf_tree", "modis_cvf_nontree", "modis_cvf_nonveg"]:
            cvf_raster = self._loaders[cvf_name].load(bounds=bounds, timestamp=timestamp, hvs=hvs)
            all_layers.append(cvf_raster)
            layer_names.append(cvf_name)

        # Human Impact Index
        hii_raster = self._loaders["hii"].load(bounds=bounds, timestamp=timestamp)
        all_layers.append(hii_raster)
        layer_names.append("hii")

        # BioClim (19 bands)
        for i in range(1, 20):
            bioclim_raster = self._loaders["bioclim"].load(bounds=bounds, band_num=i)
            all_layers.append(bioclim_raster)
            layer_names.append(f"bioclim_{i}")

        # High-res chip (Dynamic World 9 bands + Landsat NDWI 1 band)
        try:
            hr_chip = self._loaders["high_res"].load(point_id=point_id)
            hr_array = hr_chip['array']
            n_dw = len(LAYERS["dynamic_world"].bands)
            n_ndwi = len(LAYERS["landsat_ndwi"].bands)
            expected_hr_bands = n_dw + n_ndwi

            # Guard against broken chips with wrong band count
            if hr_array.shape[0] < expected_hr_bands:
                raise ValueError(
                    f"high_res chip has {hr_array.shape[0]} bands, expected {expected_hr_bands}"
                )

            # Split into DW and NDWI sub-chips
            dw_chip = {**hr_chip, 'array': hr_array[:n_dw], 'bands': LAYERS["dynamic_world"].bands}
            ndwi_chip = {**hr_chip, 'array': hr_array[n_dw:n_dw + n_ndwi], 'bands': LAYERS["landsat_ndwi"].bands}
            all_layers.extend([dw_chip, ndwi_chip])
            layer_names.extend(["dynamic_world", "landsat_ndwi"])
        except (FileNotFoundError, ValueError):
            H, W = med_chip['array'].shape[1:]
            dw_placeholder = {
                'array': np.full((len(LAYERS["dynamic_world"].bands), H, W), np.nan, dtype=np.float32),
                'transform': med_chip['transform'],
                'crs': med_chip['crs'],
                'nodata': np.nan,
                'bounds': med_chip['bounds'],
                'bands': LAYERS["dynamic_world"].bands,
            }
            ndwi_placeholder = {
                'array': np.full((len(LAYERS["landsat_ndwi"].bands), H, W), np.nan, dtype=np.float32),
                'transform': med_chip['transform'],
                'crs': med_chip['crs'],
                'nodata': np.nan,
                'bounds': med_chip['bounds'],
                'bands': LAYERS["landsat_ndwi"].bands,
            }
            all_layers.extend([dw_placeholder, ndwi_placeholder])
            layer_names.extend(["dynamic_world", "landsat_ndwi"])

        return all_layers, layer_names

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
        lc_idx = layer_names.index("modis_lc")
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
        lc_start_idx = sum([a['array'].shape[0] for a in all_layers[:lc_idx]])

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
