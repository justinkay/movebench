from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from movebench2.layers import REGISTRY
from movebench2.layers.loaders import (
    ChipGroupLoader,
    LocalHdfLayerLoader,
    LocalTiffLayerLoader,
    TileLayerLoader,
    compute_aspect,
    compute_orographic_uplift,
    compute_rugosity,
    compute_slope,
    compute_thermal_uplift,
)
from movebench2.layers.plan import LayerPlan
from movebench2.modis import modis_tiles_for_bounds
from movebench2.rasters import Raster


def sampler_hash(sampler: dict[str, Any]) -> str:
    payload = json.dumps(sampler, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _wrap_pi(angle_rad):
    import numpy as np

    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _extract_from_raster(
    raster: Raster,
    lons: Any,
    lats: Any,
    *,
    bilinear: bool,
) -> tuple[Any, Any]:
    """Sample points from a Raster in its native CRS.

    Returns (values, valid) with shape (C, N) as torch tensors.
    """
    import numpy as np
    import torch
    from pyproj import Transformer

    arr = raster.array
    transform = raster.transform
    src_crs = raster.crs
    nodata = raster.nodata

    C, H, W = arr.shape
    N = len(lons)

    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    cols = np.asarray(cols)
    rows = np.asarray(rows)

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

        grid = torch.stack(
            [torch.tensor(x_norm, dtype=torch.float32), torch.tensor(y_norm, dtype=torch.float32)],
            dim=-1,
        ).view(1, N, 1, 2)

        samples = torch.nn.functional.grid_sample(
            arr_t.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        ).squeeze(0).squeeze(-1)

        mask_t = torch.from_numpy(nodata_mask.astype(np.float32))
        mask_interp = torch.nn.functional.grid_sample(
            mask_t.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        ).squeeze(0).squeeze(-1)

        valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
        valid_t = valid_t & (mask_interp > 0.99)

        return samples, valid_t

    rows_int = np.clip(np.round(rows).astype(int), 0, H - 1)
    cols_int = np.clip(np.round(cols).astype(int), 0, W - 1)
    in_bounds = (rows >= -0.5) & (rows < H - 0.5) & (cols >= -0.5) & (cols < W - 0.5)

    samples = torch.from_numpy(arr[:, rows_int, cols_int]).float()
    valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
    valid_t = valid_t & torch.from_numpy(nodata_mask[:, rows_int, cols_int])
    return samples, valid_t


def _one_hot_encode(class_vals, num_classes: int):
    import torch

    class_vals = class_vals.long()
    class_vals = torch.clamp(class_vals, 0, num_classes - 1)
    return torch.nn.functional.one_hot(class_vals, num_classes).T.float()


def _apply_valid_range(values_t, valid_t, valid_range: tuple[float, float] | None):
    if valid_range is None:
        return values_t, valid_t
    lo, hi = valid_range
    in_range = (values_t >= lo) & (values_t <= hi)
    valid_t = valid_t & in_range
    values_t = values_t.masked_fill(~in_range, 0.0)
    return values_t, valid_t


def _apply_scale_factor(values_t, scale_factor: float):
    if scale_factor == 1.0:
        return values_t
    return values_t * float(scale_factor)


def _chip_layer_offsets(plan: LayerPlan) -> dict[str, dict[str, tuple[int, int]]]:
    """chip_group -> layer_name -> (start,end) band offsets within the chip file."""
    offsets: dict[str, dict[str, tuple[int, int]]] = {}
    for group_name, group in plan.chip_groups.items():
        start = 0
        offsets[group_name] = {}
        for layer in group["layers"]:
            spec = REGISTRY.get(layer)
            n = len(spec.bands)
            offsets[group_name][layer] = (start, start + n)
            start += n
    return offsets


def _maybe_bilinear_for_chip_group(plan: LayerPlan, chip_group: str) -> bool:
    # Match legacy behavior: high-res chips are sampled with nearest-neighbor.
    try:
        scale = float(plan.chip_groups[chip_group]["target_scale_m"])
    except Exception:
        return True
    return scale > 30.0


def _compute_derived(layer_name: str, parents: dict[str, Raster]) -> Raster:
    if layer_name == "slope":
        return compute_slope(parents["dem"])
    if layer_name == "aspect":
        return compute_aspect(parents["dem"])
    if layer_name == "rugosity":
        return compute_rugosity(parents["dem"])
    if layer_name == "orographic_uplift":
        return compute_orographic_uplift(parents["slope"], parents["aspect"], parents["era5_land"])
    if layer_name == "thermal_uplift":
        return compute_thermal_uplift(parents["era5_land"], parents["era5"])
    raise ValueError(f"No derived compute function for layer: {layer_name!r}")


@dataclass(frozen=True)
class _WorkerCtx:
    study_dir: Path
    plan: LayerPlan
    chip_offsets: dict[str, dict[str, tuple[int, int]]]


_CTX: _WorkerCtx | None = None


def _init_worker(study_dir: str, plan_dict: dict[str, Any]) -> None:
    global _CTX
    plan = LayerPlan.from_dict(plan_dict)
    _CTX = _WorkerCtx(study_dir=Path(study_dir), plan=plan, chip_offsets=_chip_layer_offsets(plan))


def _extract_env_for_sample(
    *,
    point_id: int,
    timestamp: Any,
    lons: Any,
    lats: Any,
    bounds_4326: tuple[float, float, float, float],
) -> tuple[Any, Any]:
    import torch

    assert _CTX is not None
    plan = _CTX.plan

    C = plan.num_channels()
    N = len(lons)
    all_values = torch.zeros(C, N, dtype=torch.float32)
    all_valid = torch.zeros(C, N, dtype=torch.bool)

    chip_group_rasters: dict[str, Raster] = {}
    raster_cache: dict[str, Raster] = {}

    def get_chip_group_raster(group_name: str) -> Raster:
        if group_name not in chip_group_rasters:
            loader = ChipGroupLoader(_CTX.study_dir, group_name)
            chip_group_rasters[group_name] = loader.load(point_id)
        return chip_group_rasters[group_name]

    def get_layer_raster(layer_name: str) -> Raster:
        if layer_name in raster_cache:
            return raster_cache[layer_name]

        spec = REGISTRY.get(layer_name)

        if spec.source == "gee_chip":
            group_name = plan.chip_group_for_layer(layer_name)
            if group_name is None:
                raise ValueError(f"gee_chip layer {layer_name!r} not present in any chip group")
            chip = get_chip_group_raster(group_name)
            start, end = _CTX.chip_offsets[group_name][layer_name]
            sub = Raster(
                array=chip.array[start:end],
                transform=chip.transform,
                crs=chip.crs,
                nodata=chip.nodata,
                bounds=chip.bounds,
                band_ids=[f"{layer_name}.{b}" for b in spec.bands],
            )
            raster_cache[layer_name] = sub
            return sub

        if spec.source == "gee_tile":
            loader = TileLayerLoader(_CTX.study_dir, layer_name)
            r = loader.load(bounds_4326)
            raster_cache[layer_name] = r
            return r

        if spec.source == "local_tiff":
            loader = LocalTiffLayerLoader(spec)
            r = loader.load(bounds_4326, timestamp=timestamp)
            raster_cache[layer_name] = r
            return r

        if spec.source == "local_hdf":
            hvs = modis_tiles_for_bounds(bounds_4326)
            loader = LocalHdfLayerLoader(spec)
            r = loader.load(bounds_4326, timestamp=timestamp, hvs=hvs)
            raster_cache[layer_name] = r
            return r

        if spec.source == "derived":
            if not spec.derived_from:
                raise ValueError(f"Derived layer {layer_name!r} must define derived_from")
            parents: dict[str, Raster] = {p: get_layer_raster(p) for p in spec.derived_from}
            r = _compute_derived(layer_name, parents)
            raster_cache[layer_name] = r
            return r

        raise ValueError(f"Unknown layer source: {spec.source!r}")

    # --- chip groups: sample once per group then slice into output ---
    for group_name, group in plan.chip_groups.items():
        chip = get_chip_group_raster(group_name)

        bilinear = _maybe_bilinear_for_chip_group(plan, group_name)
        chip_vals, chip_valid = _extract_from_raster(chip, lons, lats, bilinear=bilinear)

        offset = 0
        for layer_name in group["layers"]:
            spec = REGISTRY.get(layer_name)
            n = len(spec.bands)
            if layer_name in plan.layers_in_order:
                start, end = plan.layer_slice(layer_name)
                vals = chip_vals[offset : offset + n]
                valid = chip_valid[offset : offset + n]
                vals = _apply_scale_factor(vals, spec.scale_factor)
                vals, valid = _apply_valid_range(vals, valid, spec.valid_range)
                all_values[start:end] = vals
                all_valid[start:end] = valid
            offset += n

    # --- non-chip output layers ---
    for layer_name in plan.layers_in_order:
        spec = REGISTRY.get(layer_name)
        if spec.source == "gee_chip":
            continue

        start, end = plan.layer_slice(layer_name)

        if spec.categorical:
            r = get_layer_raster(layer_name)
            values, valid = _extract_from_raster(r, lons, lats, bilinear=False)
            class_vals = values.squeeze(0)
            one_hot = _one_hot_encode(class_vals, spec.num_classes)
            valid_expanded = valid.expand(spec.num_classes, -1)
            all_values[start:end] = one_hot
            all_valid[start:end] = valid_expanded
            continue

        if layer_name == "bioclim":
            # Stored as 19 separate GeoTIFFs addressed by {band_num}.
            loader = LocalTiffLayerLoader(spec)
            ch = start
            for band_num in range(1, 20):
                r = loader.load(bounds_4326, timestamp=timestamp, band_num=band_num)
                values, valid = _extract_from_raster(r, lons, lats, bilinear=True)
                values = _apply_scale_factor(values, spec.scale_factor)
                values, valid = _apply_valid_range(values, valid, spec.valid_range)
                all_values[ch : ch + 1] = values
                all_valid[ch : ch + 1] = valid
                ch += 1
            continue

        r = get_layer_raster(layer_name)
        values, valid = _extract_from_raster(r, lons, lats, bilinear=True)
        values = _apply_scale_factor(values, spec.scale_factor)
        values, valid = _apply_valid_range(values, valid, spec.valid_range)
        all_values[start:end] = values
        all_valid[start:end] = valid

    return all_values.numpy(), all_valid.numpy()


def _extract_env_worker(args: tuple[int, dict[str, Any]]) -> tuple[int, Any, Any]:
    idx, sample = args
    env, valid = _extract_env_for_sample(
        point_id=int(sample["point_id"]),
        timestamp=sample["timestamp"],
        lons=sample["lons"],
        lats=sample["lats"],
        bounds_4326=tuple(sample["bounds_4326"]),
    )
    return idx, env, valid


def extract_env_features_parallel(
    *,
    study_dir: Path,
    plan: LayerPlan,
    samples: list[dict[str, Any]],
    n_workers: int,
) -> tuple[Any, Any]:
    """Extract env and valid tensors for a list of samples.

    Returns:
      env: (n_steps, C, P) float32 numpy
      valid: (n_steps, C, P) bool numpy
    """
    import numpy as np

    n_steps = len(samples)
    C = plan.num_channels()
    P = len(samples[0]["lons"]) if samples else 0

    env_out = np.zeros((n_steps, C, P), dtype=np.float32)
    valid_out = np.zeros((n_steps, C, P), dtype=np.bool_)

    if n_steps == 0:
        return env_out, valid_out

    plan_dict = plan.to_dict()

    if n_workers <= 1:
        _init_worker(str(study_dir), plan_dict)
        for i, sample in enumerate(samples):
            _, env, valid = _extract_env_worker((i, sample))
            env_out[i] = env
            valid_out[i] = valid
        return env_out, valid_out

    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(str(study_dir), plan_dict)) as pool:
        for i, env, valid in pool.imap_unordered(_extract_env_worker, enumerate(samples)):
            env_out[i] = env
            valid_out[i] = valid

    return env_out, valid_out
