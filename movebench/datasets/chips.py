"""ChipDataset: raster stacks around each trajectory point.

This is a refactor of the legacy `movebench.datasets.ChipDataset`, rewritten to
use the movebench layer registry + LayerPlan + LayerIO.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from torch.utils.data import Dataset

from movebench.features.normalization import load_points_normalization_mean_std
from movebench.layers.io import LayerIO
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import LayerRegistry, REGISTRY
from movebench.rasters import Raster, reproject_to_meters
from movebench.trajectories import (
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    load_trajectories_csv,
    select_step_starts,
)


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
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
        normalization_sampler_hash: str | None = None,
        mode: Literal["points", "steps"] = "points",
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
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
            raise ValueError(f"output_hw must be a (H,W) tuple, got {output_hw!r}") from e
        oh_i, ow_i = int(oh), int(ow)
        if oh_i <= 0 or ow_i <= 0:
            raise ValueError(f"output_hw must be positive, got {(oh_i, ow_i)}")
        self.output_hw = (oh_i, ow_i)
        self.normalize = bool(normalize)
        self.mode = str(mode)
        if self.mode not in {"points", "steps"}:
            raise ValueError(f"Unknown mode={mode!r} (expected 'points' or 'steps')")

        self.io = LayerIO(study_dir=self.study_dir, plan=self.plan, registry=self.registry)
        if self.normalize:
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_hash=normalization_sampler_hash,
            )
        else:
            self._mean = None
            self._std = None

        df = load_trajectories_csv(str(self.trajectories_path))
        df = df[df[SET_COL] == self.split].copy()
        if len(df) == 0:
            raise ValueError(f"No rows found for split={split!r} in {self.trajectories_path}")
        self.df = df

        if POINT_ID_COL not in df.columns:
            raise ValueError(f"trajectories.csv missing required column: {POINT_ID_COL!r}")
        if LON_COL not in df.columns or LAT_COL not in df.columns:
            raise ValueError(f"trajectories.csv missing required columns: {LON_COL!r}, {LAT_COL!r}")
        if TIME_COL not in df.columns:
            raise ValueError(f"trajectories.csv missing required column: {TIME_COL!r}")

        self._to_target = None

        if self.mode == "steps":
            steps = select_step_starts(df, step_multiplier=step_multiplier, step_q=step_q, gap_q=gap_q)
            if len(steps) == 0:
                raise ValueError(f"No valid step starts found for split={split!r} in {self.trajectories_path}")
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
            from pyproj import Transformer

            self._to_target = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        else:
            self._rows = [
                _Row(
                    point_id=int(row[POINT_ID_COL]),
                    timestamp=row[TIME_COL],
                    lon=float(row[LON_COL]),
                    lat=float(row[LAT_COL]),
                )
                for _, row in df.iterrows()
            ]

        self._ref_layer = self._select_reference_layer()
        self._chip_group_layer_slices = self._build_chip_group_layer_slices()

    def _select_reference_layer(self) -> str:
        # Prefer a per-point chip group when available.
        for layer in self.plan.layers_in_order:
            if self.registry.get(layer).source == "gee_chip":
                return layer
        # Otherwise fall back to a tile layer (e.g. dem/dist_water). Prefer dem when present.
        tile_layers = [l for l in self.plan.layers_in_order if self.registry.get(l).source == "gee_tile"]
        if not tile_layers:
            raise ValueError(
                "ChipDataset requires at least one gee_chip or gee_tile layer in the LayerPlan to define bounds."
            )
        if "dem" in tile_layers:
            return "dem"
        return tile_layers[0]

    def _tile_bounds_for_point(self, layer: str, *, lon: float, lat: float) -> tuple[float, float, float, float]:
        """Return the downloaded tile bounds (EPSG:4326) that contain a point."""
        import json

        tile_dir = self.study_dir / "rasters" / str(self.plan.rasters_hash) / "tiles" / str(layer)
        index_path = tile_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Tile index.json not found for layer={layer!r}: {index_path}")

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

        raise FileNotFoundError(f"No downloaded tile for layer={layer!r} contains lon/lat=({lon},{lat})")

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

        ref_spec = self.registry.get(self._ref_layer)
        if ref_spec.source == "gee_chip":
            ref_group = self.plan.chip_group_for_layer(self._ref_layer)
            if ref_group is None:
                raise ValueError(f"Reference layer {self._ref_layer!r} is not in any chip group")

            # Reference chip group defines bounds for tile/local layers and the
            # output grid for metric reprojection.
            ref_raster = self.io.load_chip_group(ref_group, point_id=point_id)
            bounds_latlon = ref_raster.bounds
        else:
            # No per-point chip groups: use the downloaded tile that contains the point.
            bounds_latlon = self._tile_bounds_for_point(self._ref_layer, lon=float(row.lon), lat=float(row.lat))
            ref_group = None
            ref_raster = self.io.load_layer(self._ref_layer, bounds_latlon=bounds_latlon, timestamp=timestamp)

        cache: dict[str, Raster] = {}
        if ref_group is None:
            cache[self._ref_layer] = ref_raster

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

        # Warp: reference raster first, then other chip groups, then non-chip layers.
        warp_kinds: list[str] = []
        warp_names: list[str] = []
        warp_rasters: list[Raster] = []

        if ref_group is not None:
            warp_kinds.append("chip_group")
            warp_names.append(ref_group)
            warp_rasters.append(ref_raster)

            for g in needed_chip_groups:
                if g == ref_group:
                    continue
                warp_kinds.append("chip_group")
                warp_names.append(g)
                warp_rasters.append(chip_groups[g])
        else:
            warp_kinds.append("layer")
            warp_names.append(self._ref_layer)
            warp_rasters.append(ref_raster)

        for layer, r in zip(non_chip_layers, non_chip_rasters):
            if ref_group is None and layer == self._ref_layer:
                continue
            warp_kinds.append("layer")
            warp_names.append(layer)
            warp_rasters.append(r)

        # Default to bilinear resampling for continuous layers; categorical layers
        # are overridden below with nearest neighbor to preserve class IDs.
        aligned_arrays, final_transform, final_crs = reproject_to_meters(
            warp_rasters,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            dst_height=self.output_hw[0],
            dst_width=self.output_hw[1],
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
        for layer, r in zip(non_chip_layers, non_chip_rasters):
            if not self.registry.get(layer).categorical:
                continue
            cat_aligned, _, _ = reproject_to_meters(
                [ref_raster, r],
                dst_crs=TARGET_CRS,
                resampling=Resampling.nearest,
                dst_height=self.output_hw[0],
                dst_width=self.output_hw[1],
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

        if self.normalize:
            assert self._mean is not None
            assert self._std is not None
            mean = self._mean.view(-1, 1, 1)
            std = self._std.view(-1, 1, 1)
            env = torch.where(valid, (env - mean) / std, torch.zeros_like(env))

        if self.mode == "steps":
            if not isinstance(row, _StepRow):
                raise RuntimeError("Internal error: expected _StepRow rows when mode='steps'")
            if self._to_target is None:
                raise RuntimeError("Internal error: missing coordinate transformer for mode='steps'")
            x_m, y_m = self._to_target.transform(float(row.end_lon), float(row.end_lat))
            rr, cc = rowcol(final_transform, x_m, y_m)
            if not (0 <= int(rr) < int(H) and 0 <= int(cc) < int(W)):
                print("WARNING: Observed endpoint falls outside the chip grid for point_id_start="
                    f"{point_id}. (row,col)=({rr},{cc}) chip(H,W)=({H},{W})"
                )
            
            # TODO HACKY CLAMP
            # FIX THIS LATER
            y = torch.zeros((int(H), int(W)), dtype=torch.float32)
            y[min(H-1, max(0, int(rr))), min(W-1, max(0, int(cc)))] = 1.0
            return env, valid, y, final_transform, str(final_crs)

        return env, valid, final_transform, str(final_crs)
