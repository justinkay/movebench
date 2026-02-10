"""ChipDataset: raster stacks around each trajectory point.

This is a refactor of the legacy `movebench.datasets.ChipDataset`, rewritten to
use the movebench layer registry + LayerPlan + LayerIO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from torch.utils.data import Dataset

from movebench.layers.io import LayerIO
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import LayerRegistry, REGISTRY
from movebench.rasters import Raster, reproject_to_meters
from movebench.trajectories import POINT_ID_COL, SET_COL, TIME_COL, load_trajectories_csv


TARGET_CRS = "EPSG:3857" # meters


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


class ChipDataset(Dataset):
    """Load a model-ready raster stack (C,H,W) for each point in a split."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        split: str,
        plan: LayerPlan,
        registry: LayerRegistry = REGISTRY,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.trajectories_path = Path(trajectories_path)
        self.split = str(split)
        self.plan = plan
        self.registry = registry

        self.io = LayerIO(study_dir=self.study_dir, plan=self.plan, registry=self.registry)

        df = load_trajectories_csv(str(self.trajectories_path))
        df = df[df[SET_COL] == self.split].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {self.trajectories_path}")

        if POINT_ID_COL not in df.columns:
            raise ValueError(f"trajectories.csv missing required column: {POINT_ID_COL!r}")

        self._rows: list[_Row] = [
            _Row(point_id=int(row[POINT_ID_COL]), timestamp=row[TIME_COL]) for _, row in df.iterrows()
        ]

        self._ref_layer = self._select_reference_layer()
        self._chip_group_layer_slices = self._build_chip_group_layer_slices()

    def _select_reference_layer(self) -> str:
        for layer in self.plan.layers_in_order:
            if self.registry.get(layer).source == "gee_chip":
                return layer
        raise ValueError("ChipDataset requires at least one gee_chip layer in the LayerPlan to define bounds.")

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, Any, str]:
        row = self._rows[int(idx)]
        point_id = int(row.point_id)
        timestamp = row.timestamp

        ref_group = self.plan.chip_group_for_layer(self._ref_layer)
        if ref_group is None:
            raise ValueError(f"Reference layer {self._ref_layer!r} is not in any chip group")

        # Reference chip group defines bounds for tile/local layers and the
        # output grid for metric reprojection.
        ref_chip = self.io.load_chip_group(ref_group, point_id=point_id)
        bounds_latlon = ref_chip.bounds

        cache: dict[str, Raster] = {}

        def get_raster(layer: str) -> Raster:
            return self.io.load_layer_cached(
                layer,
                cache=cache,
                point_id=point_id,
                bounds_latlon=bounds_latlon,
                timestamp=timestamp,
            )

        # Load and reproject chip-group stacks once each, then slice into layers.
        needed_chip_groups: list[str] = []
        for layer in self.plan.layers_in_order:
            if self.registry.get(layer).source != "gee_chip":
                continue
            g = self.plan.chip_group_for_layer(layer)
            if g is None:
                raise ValueError(f"gee_chip layer {layer!r} is not listed in any chip group")
            if g not in needed_chip_groups:
                needed_chip_groups.append(g)

        chip_groups: dict[str, Raster] = {g: self.io.load_chip_group(g, point_id=point_id) for g in needed_chip_groups}

        # Build the set of non-chip output layers to load/compute once.
        non_chip_layers = [l for l in self.plan.layers_in_order if self.registry.get(l).source != "gee_chip"]
        non_chip_rasters = [get_raster(l) for l in non_chip_layers]

        # Warp: reference chip group first, then other chip groups, then non-chip layers.
        warp_kinds: list[str] = []
        warp_names: list[str] = []
        warp_rasters: list[Raster] = []

        warp_kinds.append("chip_group")
        warp_names.append(ref_group)
        warp_rasters.append(chip_groups[ref_group])

        for g in needed_chip_groups:
            if g == ref_group:
                continue
            warp_kinds.append("chip_group")
            warp_names.append(g)
            warp_rasters.append(chip_groups[g])

        for layer, r in zip(non_chip_layers, non_chip_rasters):
            warp_kinds.append("layer")
            warp_names.append(layer)
            warp_rasters.append(r)

        # Default to bilinear resampling for continuous layers; categorical layers
        # are overridden below with nearest neighbor to preserve class IDs.
        aligned_arrays, final_transform, final_crs = reproject_to_meters(
            warp_rasters,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
        )

        aligned_chip_groups: dict[str, np.ndarray] = {}
        aligned_layers: dict[str, np.ndarray] = {}
        nodata_by_name: dict[tuple[str, str], Any] = {}
        for i, arr in enumerate(aligned_arrays):
            kind = warp_kinds[i]
            name = warp_names[i]
            if kind == "chip_group":
                aligned_chip_groups[name] = arr
            else:
                aligned_layers[name] = arr
            nodata_by_name[(kind, name)] = warp_rasters[i].nodata

        # Categorical layers (one-hot encoded later) must use nearest-neighbor
        # resampling to avoid creating fractional class IDs during reprojection.
        ref_raster = chip_groups[ref_group]
        for layer, r in zip(non_chip_layers, non_chip_rasters):
            if not self.registry.get(layer).categorical:
                continue
            cat_aligned, _, _ = reproject_to_meters(
                [ref_raster, r],
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
            )
            aligned_layers[layer] = cat_aligned[1]

        # Build final (C,H,W) stack in LayerPlan channel order.
        H, W = aligned_arrays[0].shape[1:]
        C_total = int(self.plan.num_channels())
        env = torch.zeros((C_total, H, W), dtype=torch.float32)
        valid = torch.zeros((C_total, H, W), dtype=torch.bool)

        for layer in self.plan.layers_in_order:
            start, end = self.plan.layer_slice(layer)
            spec = self.registry.get(layer)

            if spec.source == "gee_chip":
                group = self.plan.chip_group_for_layer(layer)
                if group is None:
                    raise ValueError(f"gee_chip layer {layer!r} is not listed in any chip group")
                gs = self._chip_group_layer_slices.get(group)
                if gs is None or layer not in gs:
                    raise ValueError(f"Layer {layer!r} missing chip-group slice mapping for group {group!r}")
                gstart, gend = gs[layer]
                arr = aligned_chip_groups[group][gstart:gend]
                nodata = nodata_by_name[("chip_group", group)]
            else:
                arr = aligned_layers[layer]
                nodata = nodata_by_name[("layer", layer)]

            vals = torch.from_numpy(arr.astype(np.float32, copy=False))
            vmask = _compute_valid_mask(vals, nodata)

            if spec.categorical:
                # Expect a single band with integer class labels.
                if vals.shape[0] != 1:
                    raise ValueError(f"Categorical layer {layer!r} must have 1 band, got {vals.shape[0]}")

                class_vals = vals[0].to(torch.int64)
                class_vals = torch.clamp(class_vals, 0, int(spec.num_classes) - 1)
                one_hot = F.one_hot(class_vals, int(spec.num_classes)).permute(2, 0, 1).to(torch.float32)

                vexp = vmask[0].unsqueeze(0).expand(int(spec.num_classes), -1, -1)
                env[start:end] = one_hot * vexp.to(one_hot.dtype)
                valid[start:end] = vexp
                continue

            if spec.valid_range is not None:
                lo, hi = float(spec.valid_range[0]), float(spec.valid_range[1])
                vmask = vmask & (vals >= lo) & (vals <= hi)

            env[start:end] = torch.where(vmask, vals, torch.zeros_like(vals))
            valid[start:end] = vmask

        return env, valid, final_transform, str(final_crs)
