"""ChipDataset: raster stacks around each trajectory point.

This is a refactor of the legacy `movebench.datasets.ChipDataset`, rewritten to
use the movebench layer registry + LayerPlan + LayerIO.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_bounds, rowcol
from rasterio.warp import calculate_default_transform
from torch.utils.data import Dataset

import movebench.layers.derived as derived
from movebench.features.config import SamplerConfig
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.geometry import (
    local_aeqd_crs,
    local_bounds_to_wgs84_parts,
    local_xy_m_from_lonlat,
    validate_local_bounds_m,
)
from movebench.layers.constants import WGS84_CRS
from movebench.layers.io import LayerIO
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import REGISTRY, LayerRegistry
from movebench.rasters import Raster, reproject_multipart_to_meters
from movebench.trajectories import (
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)


def _compute_valid_mask(values: torch.Tensor, nodata: Any) -> torch.Tensor:
    """Return per-pixel validity mask for a (C,H,W) tensor."""
    if nodata is None:
        return torch.isfinite(values)
    if isinstance(nodata, float) and np.isnan(nodata):
        return torch.isfinite(values)
    return torch.isfinite(values) & (values != float(nodata))


@dataclass(frozen=True)
class _Row:
    point_id: int
    timestamp: Any
    lon: float
    lat: float


@dataclass(frozen=True)
class _StepRow:
    point_id: int
    timestamp: Any
    lon: float
    lat: float
    end_lon: float
    end_lat: float


class ChipDataset(Dataset):
    """Load model-ready raster stacks for points or step starts.

    Modes:
      - mode="points" (default): iterate over every point in the split and return
        `(env, valid, transform, crs)`.
      - mode="steps": iterate only over valid step starts and return
        `(env, valid, y, transform, crs)`, where `y` is a one-hot (H,W) mask for
        the observed next-step endpoint.
    """

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        split: str,
        plan: LayerPlan,
        sampler_config: SamplerConfig | None = None,
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
        mode: Literal["points", "steps"] = "points",
        selected_splits: list[str] | None = None,
        registry: LayerRegistry = REGISTRY,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.trajectories_path = Path(trajectories_path)
        self.split = str(split)
        self.plan = plan
        self.registry = registry
        try:
            oh, ow = output_hw
        except Exception as e:
            raise ValueError(
                f"output_hw must be a (H,W) tuple, got {output_hw!r}"
            ) from e
        oh_i, ow_i = int(oh), int(ow)
        if oh_i <= 0 or ow_i <= 0:
            raise ValueError(f"output_hw must be positive, got {(oh_i, ow_i)}")
        self.output_hw = (oh_i, ow_i)
        self.normalize = bool(normalize)
        self.mode = str(mode)
        self.selected_splits = (
            None if selected_splits is None else [str(split_key) for split_key in selected_splits]
        )
        self.sampler_config = sampler_config
        if self.mode not in {"points", "steps"}:
            raise ValueError(f"Unknown mode={mode!r} (expected 'points' or 'steps')")

        self.io = LayerIO(
            study_dir=self.study_dir, plan=self.plan, registry=self.registry
        )
        if self.normalize:
            if self.sampler_config is None:
                raise ValueError(
                    "ChipDataset(normalize=True) requires sampler_config so normalization stats can be resolved explicitly."
                )
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_config=self.sampler_config,
            )
        else:
            self._mean = None
            self._std = None

        df_all = load_trajectories_csv(str(self.trajectories_path))
        split_df = df_all[df_all[SET_COL] == self.split].copy()
        if len(split_df) == 0:
            raise ValueError(
                f"No rows found for split={split!r} in {self.trajectories_path}"
            )
        self.df = split_df

        if POINT_ID_COL not in split_df.columns:
            raise ValueError(
                f"trajectories.csv missing required column: {POINT_ID_COL!r}"
            )
        if LON_COL not in split_df.columns or LAT_COL not in split_df.columns:
            raise ValueError(
                f"trajectories.csv missing required columns: {LON_COL!r}, {LAT_COL!r}"
            )
        if TIME_COL not in split_df.columns:
            raise ValueError(f"trajectories.csv missing required column: {TIME_COL!r}")

        self._step_starts_df: pd.DataFrame | None = None
        self._warned_oob = False
        self._step_buffer_m: float | None = None
        self._warned_fixed_grid_coverage = False

        if self.mode == "steps":
            if self.output_hw[0] != self.output_hw[1]:
                raise ValueError(
                    "ChipDataset(mode='steps') requires square output_hw for deepSSF. "
                    f"Got output_hw={self.output_hw}."
                )
            if self.output_hw[0] < 2:
                raise ValueError(
                    "ChipDataset(mode='steps') requires output_hw >= (2,2) to define a pixel size. "
                    f"Got output_hw={self.output_hw}."
                )
            df_train = df_all[df_all[SET_COL] == "train"].copy()
            if len(df_train) == 0:
                raise ValueError(
                    "ChipDataset(mode='steps') requires a non-empty train split in trajectories.csv to define a "
                    "deterministic chip extent. "
                    f"Found no rows for set='train' in {self.trajectories_path}."
                )
            support = fit_interval_step_support(
                df_train,
                step_time_s=int(self.plan.step_spec.step_time_s),
                gap_q=float(self.plan.step_spec.gap_q),
                step_q=float(self.plan.step_spec.step_q),
                max_step_q=float(self.plan.step_spec.max_step_q),
                step_multiplier=float(self.plan.step_spec.step_multiplier),
            )

            steps_source_df = split_df
            if self.selected_splits:
                steps_source_df = df_all[
                    df_all[SET_COL].isin(self.selected_splits)
                ].copy()
            steps = select_step_starts_with_support(
                steps_source_df,
                step_time_s=int(self.plan.step_spec.step_time_s),
                gap_q=float(self.plan.step_spec.gap_q),
                support=support,
            )
            steps = steps[steps[SET_COL] == self.split].reset_index(drop=True).copy()
            if len(steps) == 0:
                raise ValueError(
                    f"No valid step starts found for split={split!r} in {self.trajectories_path}"
                )
            self._step_starts_df = steps.copy()
            self._rows: list[_Row | _StepRow] = [
                _StepRow(
                    point_id=int(r.point_id_start),
                    timestamp=pd.Timestamp(r.timestamp),
                    lon=float(r.start_lon),
                    lat=float(r.start_lat),
                    end_lon=float(r.end_lon),
                    end_lat=float(r.end_lat),
                )
                for r in steps.itertuples(index=False)
            ]

            try:
                buffer_m = float(support.max_step_length_m)
            except Exception as exc:
                raise ValueError(
                    "Could not compute step_buffer_m from the train split. "
                    "This usually means there are no valid steps after filtering."
                ) from exc
            if not np.isfinite(buffer_m) or buffer_m <= 0:
                raise ValueError(
                    f"Computed invalid step_buffer_m={buffer_m} from train split."
                )
            self._step_buffer_m = buffer_m
        else:
            self._rows = [
                _Row(
                    point_id=int(row[POINT_ID_COL]),
                    timestamp=row[TIME_COL],
                    lon=float(row[LON_COL]),
                    lat=float(row[LAT_COL]),
                )
                for _, row in split_df.iterrows()
            ]

        self._ref_layer = self._select_reference_layer()
        self._chip_group_layer_slices = self._build_chip_group_layer_slices()

    @property
    def step_starts_df(self) -> pd.DataFrame:
        """Step starts dataframe used by this dataset when mode='steps'."""
        if self.mode != "steps":
            raise AttributeError("step_starts_df is only available when mode='steps'")
        assert self._step_starts_df is not None
        return self._step_starts_df

    @property
    def step_buffer_m(self) -> float:
        """Half-width of the step chip extent (meters), defined from the train split."""
        if self.mode != "steps":
            raise AttributeError("step_buffer_m is only available when mode='steps'")
        assert self._step_buffer_m is not None
        return float(self._step_buffer_m)

    @property
    def pixel_size_m(self) -> float:
        """Meters/pixel for the deterministic local AEQD step grid."""
        if self.mode != "steps":
            raise AttributeError("pixel_size_m is only available when mode='steps'")
        H, W = self.output_hw
        if H != W:
            raise ValueError(
                f"pixel_size_m requires square output_hw; got {self.output_hw}."
            )
        if W < 2:
            raise ValueError(
                f"pixel_size_m requires W>=2; got output_hw={self.output_hw}."
            )
        return (2.0 * float(self.step_buffer_m)) / (float(W) - 1.0)

    def _select_reference_layer(self) -> str:
        layers = list(self.plan.layers_in_order)
        for parents in self.plan.derived_dependencies.values():
            layers.extend(parent for parent in parents if parent not in layers)
        for layer in layers:
            if self.registry.get(layer).source == "gee_chip":
                return layer
        tile_layers = [
            layer
            for layer in layers
            if self.registry.get(layer).source == "gee_tile"
        ]
        if not tile_layers:
            raise ValueError(
                "ChipDataset requires at least one gee_chip or gee_tile layer in the LayerPlan to define bounds."
            )
        if "dem" in tile_layers:
            return "dem"
        return tile_layers[0]

    def _tile_bounds_for_point(
        self, layer: str, *, lon: float, lat: float
    ) -> tuple[float, float, float, float]:
        """Return the downloaded tile bounds (EPSG:4326) that contain a point."""

        tile_dir = (
            self.study_dir
            / "rasters"
            / str(self.plan.rasters_hash)
            / "tiles"
            / str(layer)
        )
        index_path = tile_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Tile index.json not found for layer={layer!r}: {index_path}"
            )

        with open(index_path, "r") as f:
            d = json.load(f)
        tiles = d.get("tiles", [])
        for t in tiles:
            b = t.get("bounds", {})
            min_lon = float(b["min_lon"])
            max_lon = float(b["max_lon"])
            min_lat = float(b["min_lat"])
            max_lat = float(b["max_lat"])
            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
                return (min_lon, min_lat, max_lon, max_lat)

        raise FileNotFoundError(
            f"No downloaded tile for layer={layer!r} contains lon/lat=({lon},{lat})"
        )

    def _build_chip_group_layer_slices(self) -> dict[str, dict[str, tuple[int, int]]]:
        out: dict[str, dict[str, tuple[int, int]]] = {}
        for group_name, g in self.plan.chip_groups.items():
            offset = 0
            layer_slices: dict[str, tuple[int, int]] = {}
            for layer in g["layers"]:
                spec = self.registry.get(layer)
                n = len(spec.bands)
                layer_slices[layer] = (offset, offset + n)
                offset += n
            out[group_name] = layer_slices
        return out

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[Any, ...]:
        row = self._rows[int(idx)]
        point_id = int(row.point_id)
        timestamp = row.timestamp
        target_crs = local_aeqd_crs(float(row.lon), float(row.lat))
        height, width = self.output_hw
        fixed_dst_transform = None
        support_bounds_m = None
        bounds_parts = None

        if self.mode == "steps":
            if not isinstance(row, _StepRow):
                raise RuntimeError(
                    "Internal error: expected _StepRow rows when mode='steps'."
                )
            if self._step_buffer_m is None:
                raise RuntimeError(
                    "Internal error: missing buffer for mode='steps'."
                )
            buffer_m = float(self._step_buffer_m)
            pixel_size_m = (2.0 * buffer_m) / (float(width) - 1.0)
            support_bounds_m = (
                -buffer_m - 0.5 * pixel_size_m,
                -buffer_m - 0.5 * pixel_size_m,
                buffer_m + 0.5 * pixel_size_m,
                buffer_m + 0.5 * pixel_size_m,
            )
            validate_local_bounds_m(support_bounds_m)
            fixed_dst_transform = from_bounds(
                *support_bounds_m,
                int(width),
                int(height),
            )
            bounds_parts = local_bounds_to_wgs84_parts(
                float(row.lon),
                float(row.lat),
                support_bounds_m,
            )

        ref_spec = self.registry.get(self._ref_layer)
        chip_groups: dict[str, Raster] = {}
        source_cache: dict[str, list[Raster]] = {}

        if ref_spec.source == "gee_chip":
            ref_group = self.plan.chip_group_for_layer(self._ref_layer)
            if ref_group is None:
                raise ValueError(
                    f"Reference layer {self._ref_layer!r} is not in any chip group"
                )
            chip_groups[ref_group] = self.io.load_chip_group(
                ref_group,
                point_id=point_id,
            )
            reference = chip_groups[ref_group]
            if bounds_parts is None:
                bounds_parts = (reference.bounds,)
        else:
            if bounds_parts is None:
                bounds_parts = (
                    self._tile_bounds_for_point(
                        self._ref_layer,
                        lon=float(row.lon),
                        lat=float(row.lat),
                    ),
                )
            reference_parts = [
                self.io.load_layer(
                    self._ref_layer,
                    bounds_latlon=bounds,
                    timestamp=timestamp,
                )
                for bounds in bounds_parts
            ]
            source_cache[self._ref_layer] = reference_parts
            reference = reference_parts[0]

        if fixed_dst_transform is None:
            _, source_height, source_width = reference.array.shape
            fixed_dst_transform, output_width, output_height = (
                calculate_default_transform(
                    reference.crs,
                    target_crs,
                    source_width,
                    source_height,
                    *reference.bounds,
                    dst_width=int(width),
                    dst_height=int(height),
                )
            )
            if (int(output_height), int(output_width)) != (height, width):
                raise RuntimeError("Rasterio did not preserve the requested output size")
            support_bounds_m = tuple(
                float(value)
                for value in array_bounds(height, width, fixed_dst_transform)
            )
            validate_local_bounds_m(support_bounds_m)

        assert bounds_parts is not None
        assert support_bounds_m is not None
        metric_cache: dict[str, Raster] = {}

        def get_raster_parts(layer: str) -> list[Raster]:
            cached = source_cache.get(layer)
            if cached is not None:
                return cached

            spec = self.registry.get(layer)
            if spec.metric_grid_derived:
                if layer not in metric_cache:
                    requested = {
                        name
                        for name in self.plan.layers_in_order
                        if self.registry.get(name).metric_grid_derived
                    }
                    era5_parts = (
                        get_raster_parts("era5_land_hourly")
                        if "orographic_uplift_hourly" in requested
                        else None
                    )
                    metric_cache.update(
                        derived.derive_metric_terrain_layers(
                            dem_parts=get_raster_parts("dem"),
                            dst_crs=target_crs,
                            support_bounds_m=support_bounds_m,
                            requested_layers=requested,
                            era5_land_hourly=era5_parts[0] if era5_parts else None,
                        )
                    )
                return [metric_cache[layer]]

            if spec.source == "gee_chip":
                group = self.plan.chip_group_for_layer(layer)
                if group is None:
                    raise ValueError(
                        f"gee_chip layer {layer!r} is not listed in any chip group"
                    )
                group_raster = chip_groups.get(group)
                if group_raster is None:
                    group_raster = self.io.load_chip_group(group, point_id=point_id)
                    chip_groups[group] = group_raster
                start, end = self._chip_group_layer_slices[group][layer]
                rasters = [
                    Raster(
                        array=group_raster.array[start:end],
                        transform=group_raster.transform,
                        crs=group_raster.crs,
                        nodata=group_raster.nodata,
                        bounds=group_raster.bounds,
                        band_ids=[f"{layer}.{band}" for band in spec.bands],
                    )
                ]
            elif spec.source == "derived":
                if not spec.derived_from:
                    raise ValueError(
                        f"Derived layer {layer!r} must define derived_from"
                    )
                parent_parts = {
                    parent: get_raster_parts(parent)
                    for parent in spec.derived_from
                }
                part_count = max(len(parts) for parts in parent_parts.values())
                rasters = []
                for part_index in range(part_count):
                    parents = {
                        parent: parts[part_index] if len(parts) > 1 else parts[0]
                        for parent, parts in parent_parts.items()
                    }
                    fn = getattr(derived, f"compute_{layer}", None)
                    if fn is None:
                        raise ValueError(
                            f"Missing derived compute function compute_{layer}"
                        )
                    base = parents[spec.derived_from[0]]
                    rasters.append(
                        Raster(
                            array=fn(**parents),
                            transform=base.transform,
                            crs=base.crs,
                            nodata=np.nan,
                            bounds=base.bounds,
                            band_ids=[
                                f"{layer}.{band}" for band in spec.bands
                            ],
                        )
                    )
            else:
                rasters = [
                    self.io.load_layer(
                        layer,
                        bounds_latlon=bounds,
                        timestamp=timestamp,
                    )
                    for bounds in bounds_parts
                ]

            source_cache[layer] = rasters
            return rasters

        layer_parts = [
            get_raster_parts(layer) for layer in self.plan.layers_in_order
        ]
        methods = []
        for layer, parts in zip(self.plan.layers_in_order, layer_parts):
            spec = self.registry.get(layer)
            if spec.categorical:
                methods.append(Resampling.nearest)
            elif spec.metric_grid_derived:
                source_resolution = max(
                    abs(float(parts[0].transform.a)),
                    abs(float(parts[0].transform.e)),
                )
                destination_resolution = max(
                    abs(float(fixed_dst_transform.a)),
                    abs(float(fixed_dst_transform.e)),
                )
                methods.append(
                    Resampling.average
                    if destination_resolution >= source_resolution
                    else Resampling.bilinear
                )
            else:
                methods.append(Resampling.bilinear)

        aligned_arrays, final_transform, final_crs = (
            reproject_multipart_to_meters(
                layer_parts,
                dst_crs=target_crs,
                resampling=methods,
                dst_height=height,
                dst_width=width,
                dst_transform=fixed_dst_transform,
            )
        )
        aligned_layers = dict(zip(self.plan.layers_in_order, aligned_arrays))
        nodata_by_layer = {
            layer: parts[0].nodata
            for layer, parts in zip(self.plan.layers_in_order, layer_parts)
        }

        if not self._warned_fixed_grid_coverage:
            first_layer = self.plan.layers_in_order[0]
            ref_arr = aligned_layers[first_layer]
            ref_valid = _compute_valid_mask(
                torch.from_numpy(ref_arr.astype(np.float32, copy=False)),
                nodata_by_layer[first_layer],
            )
            if not bool(ref_valid.all()):
                self._warned_fixed_grid_coverage = True

        H, W = aligned_arrays[0].shape[1:]
        C_total = int(self.plan.num_channels())
        env = torch.zeros((C_total, H, W), dtype=torch.float32)
        valid = torch.zeros((C_total, H, W), dtype=torch.bool)

        for layer in self.plan.layers_in_order:
            start, end = self.plan.layer_slice(layer)
            spec = self.registry.get(layer)
            arr = aligned_layers[layer]
            nodata = nodata_by_layer[layer]
            vals = torch.from_numpy(arr.astype(np.float32, copy=False))
            vmask = _compute_valid_mask(vals, nodata)

            if spec.categorical:
                if vals.shape[0] != 1:
                    raise ValueError(
                        f"Categorical layer {layer!r} must have 1 band, got {vals.shape[0]}"
                    )
                class_vals = torch.clamp(
                    vals[0].to(torch.int64),
                    0,
                    int(spec.num_classes) - 1,
                )
                one_hot = (
                    F.one_hot(class_vals, int(spec.num_classes))
                    .permute(2, 0, 1)
                    .to(torch.float32)
                )
                vexp = vmask[0].unsqueeze(0).expand(int(spec.num_classes), -1, -1)
                env[start:end] = one_hot * vexp.to(one_hot.dtype)
                valid[start:end] = vexp
                continue

            if spec.valid_range is not None:
                lo, hi = float(spec.valid_range[0]), float(spec.valid_range[1])
                vmask = vmask & (vals >= lo) & (vals <= hi)

            env[start:end] = torch.where(vmask, vals, torch.zeros_like(vals))
            valid[start:end] = vmask

        if self.normalize:
            assert self._mean is not None
            assert self._std is not None
            mean = self._mean.view(-1, 1, 1)
            std = self._std.view(-1, 1, 1)
            env = torch.where(valid, (env - mean) / std, torch.zeros_like(env))

        if self.mode == "steps":
            if not isinstance(row, _StepRow):
                raise RuntimeError(
                    "Internal error: expected _StepRow rows when mode='steps'"
                )
            x_m, y_m = local_xy_m_from_lonlat(
                row.lon,
                row.lat,
                row.end_lon,
                row.end_lat,
            )
            rr, cc = rowcol(final_transform, x_m, y_m)
            y = torch.zeros((int(H), int(W)), dtype=torch.float32)
            if 0 <= int(rr) < int(H) and 0 <= int(cc) < int(W):
                y[int(rr), int(cc)] = 1.0
            elif not self._warned_oob:
                self._warned_oob = True
            return env, valid, y, final_transform, str(final_crs)

        return env, valid, final_transform, str(final_crs)
