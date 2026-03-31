"""Point feature extraction for SSF / iSSF workflows."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from movebench.features.config import SamplerConfig
from movebench.features.manifest import (
    PointFeatureManifest,
    read_manifest,
    utc_now_iso,
    write_manifest,
)
from movebench.layers.config import FixedStepSpec, _truncate_key
from movebench.layers.constants import WGS84_CRS
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import REGISTRY
from movebench.trajectories import (
    SET_COL,
    canonicalize_split_key,
    compute_step_length_extent_m,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)
from scipy import stats

TARGET_CRS = "EPSG:3857"  # meters
SamplerType = Literal["empirical", "parametric"]


@lru_cache(maxsize=1)
def _transformers():
    from pyproj import Transformer

    to_target = Transformer.from_crs(WGS84_CRS, TARGET_CRS, always_xy=True)
    to_4326 = Transformer.from_crs(TARGET_CRS, WGS84_CRS, always_xy=True)
    return to_target, to_4326


@lru_cache(maxsize=1)
def _extraction_runtime():
    from movebench.layers.io import LayerIO
    from movebench.rasters import Raster
    from pyproj import Transformer

    return LayerIO, Raster, Transformer


@lru_cache(maxsize=1)
def _derived_module():
    import movebench.layers.derived as derived

    return derived


def _stable_payload(d: dict[str, Any]) -> bytes:
    return json.dumps(
        d, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _step_seed(random_seed: int, point_id_start: int) -> int:
    s = f"{int(random_seed)}|{int(point_id_start)}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def _sampler_payload_dict(sampler: SamplerConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(sampler, SamplerConfig):
        return {
            "type": str(sampler.type),
            "num_candidates": int(sampler.num_candidates),
            "random_seed": int(sampler.random_seed),
        }
    return {
        "type": str(sampler["type"]),
        "num_candidates": int(sampler["num_candidates"]),
        "random_seed": int(sampler["random_seed"]),
    }


def sampler_hash(sampler: SamplerConfig | dict[str, Any]) -> str:
    payload_dict = _sampler_payload_dict(sampler)
    payload = _stable_payload(payload_dict)
    parts = [
        f"tp{'par' if payload_dict['type'] == 'parametric' else 'emp'}",
        f"nc{payload_dict['num_candidates']}",
        f"rs{payload_dict['random_seed']}",
    ]
    return _truncate_key("-".join(parts), payload=payload)


def resolve_sampler_hash(sampler: SamplerConfig | dict[str, Any]) -> str:
    return sampler_hash(sampler)


def point_stores_dir(*, study_dir: Path, split: str, rasters_hash: str) -> Path:
    return (
        Path(study_dir)
        / "features"
        / "points"
        / canonicalize_split_key(split)
        / str(rasters_hash)
    )


def point_store_dir(
    *, study_dir: Path, split: str, rasters_hash: str, sampler_hash: str
) -> Path:
    return point_stores_dir(
        study_dir=study_dir, split=split, rasters_hash=rasters_hash
    ) / str(sampler_hash)


def point_store_manifest_path(
    *, study_dir: Path, split: str, rasters_hash: str, sampler_hash: str
) -> Path:
    return (
        point_store_dir(
            study_dir=study_dir,
            split=split,
            rasters_hash=rasters_hash,
            sampler_hash=sampler_hash,
        )
        / "manifest.json"
    )


def point_store_train_stats_path(
    *, study_dir: Path, rasters_hash: str, sampler_hash: str
) -> Path:
    return (
        point_store_dir(
            study_dir=study_dir,
            split="train",
            rasters_hash=rasters_hash,
            sampler_hash=sampler_hash,
        )
        / "train_stats.json"
    )


def load_point_store_manifest(
    *,
    study_dir: Path,
    split: str,
    rasters_hash: str,
    sampler_hash: str,
) -> tuple[Path, PointFeatureManifest]:
    split = canonicalize_split_key(split)
    store_dir = point_store_dir(
        study_dir=study_dir,
        split=split,
        rasters_hash=rasters_hash,
        sampler_hash=sampler_hash,
    )
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No extracted point feature store found for "
            f"split={split!r}, rasters_hash={rasters_hash!r}, sampler_hash={sampler_hash!r}. "
            "Run Study.extract_point_features(...) first."
    )
    return store_dir, read_manifest(manifest_path)


def _row_idxs_numpy(row_idxs: np.ndarray | torch.Tensor | list[int]) -> np.ndarray:
    if torch.is_tensor(row_idxs):
        arr = row_idxs.detach().cpu().numpy()
    else:
        arr = np.asarray(row_idxs)
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def _numeric_shared_field(field: np.ndarray) -> torch.Tensor | np.ndarray:
    arr = np.asarray(field)
    if arr.dtype.kind in {"U", "S", "O"}:
        return arr.copy()
    return torch.from_numpy(np.array(arr, copy=True))


class PointStoreReader:
    def __init__(self, *, store_dir: Path, manifest: PointFeatureManifest) -> None:
        self.store_dir = Path(store_dir)
        self.manifest = manifest
        self._shared_memmap: np.ndarray | None = None
        self._layer_memmaps: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def n_rows(self) -> int:
        return int(self.manifest.n_rows)

    @property
    def points_per_step(self) -> int:
        return int(self.manifest.points_per_step)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_shared_memmap"] = None
        state["_layer_memmaps"] = {}
        return state

    def _shared_array(self) -> np.ndarray:
        if self._shared_memmap is None:
            self._shared_memmap = np.load(
                self.store_dir / str(self.manifest.shared["path"]),
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._shared_memmap

    def _layer_arrays(self, layer: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._layer_memmaps.get(layer)
        if cached is not None:
            return cached
        meta = self.manifest.layers[layer]
        env = np.load(
            self.store_dir / str(meta["env_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        valid = np.load(
            self.store_dir / str(meta["valid_path"]),
            mmap_mode="r",
            allow_pickle=False,
        )
        self._layer_memmaps[layer] = (env, valid)
        return env, valid

    def load_env_valid_rows(
        self,
        *,
        band_ids: list[str],
        row_idxs: np.ndarray | torch.Tensor | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = _row_idxs_numpy(row_idxs)
        batch_size = int(rows.shape[0])
        if not band_ids:
            return (
                torch.zeros(
                    (batch_size, 0, self.points_per_step),
                    dtype=torch.float32,
                ),
                torch.zeros(
                    (batch_size, 0, self.points_per_step),
                    dtype=torch.bool,
                ),
            )

        env_out = np.zeros(
            (batch_size, len(band_ids), self.points_per_step),
            dtype=np.float32,
        )
        valid_out = np.zeros_like(env_out, dtype=np.bool_)
        by_layer: dict[str, list[tuple[int, int]]] = {}
        for out_idx, band_id in enumerate(band_ids):
            meta = self.manifest.bands.get(str(band_id))
            if meta is None:
                raise ValueError(
                    f"Requested band_id {band_id!r} not present in point store"
                )
            by_layer.setdefault(str(meta["layer"]), []).append(
                (int(out_idx), int(meta["channel_idx"]))
            )

        for layer, band_refs in by_layer.items():
            layer_env, layer_valid = self._layer_arrays(layer)
            slab_env = np.asarray(layer_env[rows])
            slab_valid = np.asarray(layer_valid[rows])
            out_positions = [ref[0] for ref in band_refs]
            channel_idxs = [ref[1] for ref in band_refs]
            env_out[:, out_positions, :] = slab_env[:, channel_idxs, :]
            valid_out[:, out_positions, :] = slab_valid[:, channel_idxs, :]

        return torch.from_numpy(env_out), torch.from_numpy(valid_out)

    def load_shared_rows(
        self,
        *,
        keys: list[str],
        row_idxs: np.ndarray | torch.Tensor | list[int],
    ) -> dict[str, torch.Tensor | np.ndarray]:
        rows = _row_idxs_numpy(row_idxs)
        shared = self._shared_array()
        out: dict[str, torch.Tensor | np.ndarray] = {}
        for key in keys:
            out[str(key)] = _numeric_shared_field(shared[str(key)][rows])
        return out

    def iter_row_blocks(
        self,
        *,
        band_ids: list[str] | None,
        shared_keys: list[str],
        block_size: int,
    ) -> Iterator[dict[str, torch.Tensor | np.ndarray]]:
        if int(block_size) <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size!r}")
        for start in range(0, self.n_rows, int(block_size)):
            stop = min(self.n_rows, start + int(block_size))
            rows = np.arange(start, stop, dtype=np.int64)
            block: dict[str, torch.Tensor | np.ndarray] = {
                "row_idxs": torch.from_numpy(rows.copy())
            }
            if band_ids is not None:
                env, valid = self.load_env_valid_rows(band_ids=band_ids, row_idxs=rows)
                block["env"] = env
                block["valid"] = valid
            block.update(self.load_shared_rows(keys=shared_keys, row_idxs=rows))
            yield block


@dataclass(frozen=True)
class _StepRequest:
    point_id_start: int
    point_id_end: int
    timestamp: Any
    timestamp_end: Any
    start_lon: float
    start_lat: float
    start_bearing: float
    end_lon: float
    end_lat: float
    actual_step_time_s: float
    step_time_error_s: float
    individual_id: str
    burst_id: int


def _fit_sampler_state(
    *,
    step_lengths_m: np.ndarray,
    turn_angles_rad: np.ndarray,
    sampler_config: SamplerConfig,
    step_spec: FixedStepSpec,
    max_step_len_m_train: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    valid_lengths = np.asarray(step_lengths_m, dtype=np.float64)
    valid_lengths = valid_lengths[np.isfinite(valid_lengths) & (valid_lengths > 0.0)]
    valid_turns = np.asarray(turn_angles_rad, dtype=np.float64)
    valid_turns = valid_turns[np.isfinite(valid_turns)]

    if len(valid_lengths) == 0:
        raise ValueError(
            "No valid interval step lengths in train split; cannot build candidate sampler"
        )
    if len(valid_turns) == 0:
        raise ValueError(
            "No valid interval turn angles in train split; cannot build candidate sampler"
        )

    if max_step_len_m_train is None:
        max_step_len_m_train = compute_step_length_extent_m(
            valid_lengths,
            step_q=float(step_spec.step_q),
            max_step_q=float(step_spec.max_step_q),
            step_multiplier=float(step_spec.step_multiplier),
        )
    else:
        max_step_len_m_train = float(max_step_len_m_train)
        if not np.isfinite(max_step_len_m_train) or max_step_len_m_train <= 0.0:
            raise ValueError(
                "max_step_len_m_train must be finite and > 0, got "
                f"{max_step_len_m_train}"
            )

    sampler = {
        "type": str(sampler_config.type),
        "fit_source_split": "train",
        "num_candidates": int(sampler_config.num_candidates),
        "random_seed": int(sampler_config.random_seed),
        "n_valid_lengths_train": int(len(valid_lengths)),
        "n_valid_turns_train": int(len(valid_turns)),
        "max_step_len_m_train": float(max_step_len_m_train),
    }
    state: dict[str, Any] = {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(sampler_config.num_candidates),
        "random_seed": int(sampler_config.random_seed),
        "gamma_used": False,
        "gamma_params": None,
        "vonmises_used": False,
        "vonmises_params": None,
    }

    if str(sampler_config.type) == "parametric":
        gamma_used = False
        gamma_params = None
        vonmises_used = False
        vonmises_params = None
        try:
            gamma_params = stats.gamma.fit(valid_lengths, floc=0)
            gamma_used = True
        except Exception:
            gamma_params = None
            gamma_used = False
        try:
            vonmises_params = stats.vonmises.fit(valid_turns)
            vonmises_used = True
        except Exception:
            vonmises_params = None
            vonmises_used = False
        sampler["gamma"] = {
            "used": bool(gamma_used),
            "params": [float(x) for x in gamma_params]
            if gamma_used and gamma_params is not None
            else None,
        }
        sampler["vonmises"] = {
            "used": bool(vonmises_used),
            "params": [float(x) for x in vonmises_params]
            if vonmises_used and vonmises_params is not None
            else None,
        }
        state["gamma_used"] = bool(gamma_used)
        state["gamma_params"] = gamma_params
        state["vonmises_used"] = bool(vonmises_used)
        state["vonmises_params"] = vonmises_params

    return sampler, state, sampler_hash(sampler_config)


def build_train_sampler(
    train_steps: pd.DataFrame,
    *,
    sampler_config: SamplerConfig,
    step_spec: FixedStepSpec,
    max_step_len_m_train: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    return _fit_sampler_state(
        step_lengths_m=train_steps["step_length_m"].to_numpy(dtype=np.float64),
        turn_angles_rad=train_steps["turn_angle_rad"].to_numpy(dtype=np.float64),
        sampler_config=sampler_config,
        step_spec=step_spec,
        max_step_len_m_train=max_step_len_m_train,
    )


@torch.no_grad()
def _sample_from_raster(
    raster: Raster,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    bilinear: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    arr = raster.array
    transform = raster.transform
    src_crs = raster.crs
    nodata = raster.nodata

    c_channels, height, width = arr.shape
    n_points = int(len(lons))

    _LayerIO, _Raster, Transformer = _extraction_runtime()
    transformer = Transformer.from_crs(WGS84_CRS, src_crs, always_xy=True)
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
        arr_clean = np.where(nodata_mask, arr, 0.0).astype(np.float32, copy=False)
        arr_t = torch.from_numpy(arr_clean)

        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows_clamped = np.clip(rows, 0, height - 1)
        cols_clamped = np.clip(cols, 0, width - 1)

        x_norm = 2.0 * (cols_clamped / max(width - 1, 1)) - 1.0
        y_norm = 2.0 * (rows_clamped / max(height - 1, 1)) - 1.0
        grid = torch.stack(
            [
                torch.tensor(x_norm, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32),
            ],
            dim=-1,
        ).view(1, n_points, 1, 2)

        samples = (
            F.grid_sample(
                arr_t.unsqueeze(0),
                grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            )
            .squeeze(0)
            .squeeze(-1)
        )

        mask_t = torch.from_numpy(nodata_mask.astype(np.float32, copy=False))
        mask_interp = (
            F.grid_sample(
                mask_t.unsqueeze(0),
                grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="zeros",
            )
            .squeeze(0)
            .squeeze(-1)
        )

        valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(
            c_channels, -1
        )
        valid_t = valid_t & (mask_interp > 0.99)
        return samples, valid_t

    rows_int = np.clip(np.round(rows).astype(int), 0, height - 1)
    cols_int = np.clip(np.round(cols).astype(int), 0, width - 1)
    in_bounds = (rows >= -0.5) & (rows < height - 0.5) & (cols >= -0.5) & (
        cols < width - 0.5
    )

    samples = torch.from_numpy(arr[:, rows_int, cols_int]).float()
    valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(
        c_channels, -1
    )
    valid_t = valid_t & torch.from_numpy(nodata_mask[:, rows_int, cols_int])
    return samples, valid_t


def _apply_valid_range(
    values: torch.Tensor,
    valid: torch.Tensor,
    valid_range: tuple[float, float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_range is None:
        return values, valid
    lo, hi = float(valid_range[0]), float(valid_range[1])
    in_range = (values >= lo) & (values <= hi)
    valid2 = valid & in_range
    values2 = torch.where(valid2, values, torch.zeros_like(values))
    return values2, valid2


def _sample_pseudo_steps(
    state: dict[str, Any], point_id_start: int
) -> tuple[np.ndarray, np.ndarray]:
    seed = _step_seed(int(state["random_seed"]), int(point_id_start))
    rng = np.random.default_rng(seed)
    valid_lengths = state["valid_lengths"]
    valid_turns = state["valid_turns"]
    n = int(state["num_candidates"])

    if state.get("gamma_used", False) and state.get("gamma_params") is not None:
        pseudo_lens = stats.gamma.rvs(
            *state["gamma_params"], size=n, random_state=rng
        ).astype(np.float64, copy=False)
    else:
        pseudo_lens = rng.choice(valid_lengths, size=n, replace=True).astype(
            np.float64, copy=False
        )

    if state.get("vonmises_used", False) and state.get("vonmises_params") is not None:
        pseudo_turns = stats.vonmises.rvs(
            *state["vonmises_params"], size=n, random_state=rng
        ).astype(np.float64, copy=False)
    else:
        pseudo_turns = rng.choice(valid_turns, size=n, replace=True).astype(
            np.float64, copy=False
        )
    return pseudo_lens, pseudo_turns


@dataclass(frozen=True)
class _CandidateSet:
    lons: np.ndarray
    lats: np.ndarray
    bounds_4326: tuple[float, float, float, float]
    step_lengths_m: np.ndarray
    turn_angles_rad: np.ndarray


def _compute_candidates(
    req: _StepRequest,
    *,
    sampler_state: dict[str, Any],
) -> _CandidateSet:
    pseudo_lens, pseudo_turns = _sample_pseudo_steps(sampler_state, req.point_id_start)

    to_target, to_4326 = _transformers()
    x0, y0 = to_target.transform(float(req.start_lon), float(req.start_lat))
    x1, y1 = to_target.transform(float(req.end_lon), float(req.end_lat))

    headings = float(req.start_bearing) + pseudo_turns
    headings = (headings + np.pi) % (2.0 * np.pi) - np.pi

    dx = pseudo_lens * np.sin(headings)
    dy = pseudo_lens * np.cos(headings)
    xs = np.append(x0 + dx, x1)
    ys = np.append(y0 + dy, y1)

    true_sl = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    true_heading = float(np.arctan2(x1 - x0, y1 - y0))
    true_ta = true_heading - float(req.start_bearing)
    true_ta = float((true_ta + np.pi) % (2.0 * np.pi) - np.pi)

    step_lengths = np.append(pseudo_lens, true_sl)
    turn_angles = np.append(pseudo_turns, true_ta)

    lons, lats = to_4326.transform(xs, ys)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)
    bounds_4326 = (
        float(np.min(lons)),
        float(np.min(lats)),
        float(np.max(lons)),
        float(np.max(lats)),
    )

    return _CandidateSet(
        lons=lons,
        lats=lats,
        bounds_4326=bounds_4326,
        step_lengths_m=step_lengths.astype(np.float32, copy=False),
        turn_angles_rad=turn_angles.astype(np.float32, copy=False),
    )


def _ensure_timestamp_ns(ts: Any) -> int:
    if hasattr(ts, "value"):
        return int(ts.value)
    if hasattr(ts, "timestamp"):
        return int(float(ts.timestamp()) * 1e9)
    raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")


def _extract_one_step(
    req: _StepRequest,
    *,
    io: LayerIO,
    plan: LayerPlan,
    layer_slices: dict[str, tuple[int, int]],
    sampler_state: dict[str, Any],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    cand = _compute_candidates(req, sampler_state=sampler_state)
    points_per_step = int(len(cand.lons))
    n_channels = int(plan.num_channels())

    env = torch.zeros((n_channels, points_per_step), dtype=torch.float32)
    valid = torch.zeros((n_channels, points_per_step), dtype=torch.bool)
    cache: dict[str, Raster] = {}

    def get_raster(layer: str) -> Raster:
        cached = cache.get(layer)
        if cached is not None:
            return cached

        _LayerIO, Raster, _Transformer = _extraction_runtime()
        spec = REGISTRY.get(layer)
        if spec.source == "derived":
            if not spec.derived_from:
                raise ValueError(f"Derived layer {layer!r} must define derived_from")
            parents = {p: get_raster(p) for p in spec.derived_from}
            fn = getattr(_derived_module(), f"compute_{layer}", None)
            if fn is None:
                raise ValueError(
                    f"Missing derived compute function compute_{layer} in movebench.layers.derived"
                )
            arr = fn(**parents)
            base = parents[spec.derived_from[0]]
            r = Raster(
                array=arr,
                transform=base.transform,
                crs=base.crs,
                nodata=float("nan"),
                bounds=base.bounds,
                band_ids=[f"{layer}.{b}" for b in spec.bands],
            )
            cache[layer] = r
            return r

        r = io.load_layer(
            layer,
            point_id=req.point_id_start,
            bounds_latlon=cand.bounds_4326,
            timestamp=req.timestamp,
        )
        cache[layer] = r
        return r

    for layer in plan.layers_in_order:
        start, end = layer_slices[layer]
        spec = REGISTRY.get(layer)
        bilinear = not spec.categorical
        r = get_raster(layer)
        vals, vmask = _sample_from_raster(r, cand.lons, cand.lats, bilinear=bilinear)

        if spec.categorical:
            class_vals = vals.squeeze(0).to(torch.int64)
            class_vals = torch.clamp(class_vals, 0, int(spec.num_classes) - 1)
            one_hot = F.one_hot(class_vals, int(spec.num_classes)).T.to(torch.float32)
            vexp = vmask.expand(int(spec.num_classes), -1)
            env[start:end] = one_hot * vexp.to(one_hot.dtype)
            valid[start:end] = vexp
            continue

        vals, vmask = _apply_valid_range(vals, vmask, spec.valid_range)
        env[start:end] = vals
        valid[start:end] = vmask

    env = torch.where(valid, env, torch.zeros_like(env))
    return (
        env.numpy(),
        valid.numpy(),
        cand.lons.astype(np.float32, copy=False),
        cand.lats.astype(np.float32, copy=False),
        cand.step_lengths_m,
        cand.turn_angles_rad,
        int(req.point_id_start),
        int(req.point_id_end),
        _ensure_timestamp_ns(req.timestamp),
        _ensure_timestamp_ns(req.timestamp_end),
        float(req.actual_step_time_s),
        float(req.step_time_error_s),
    )


_W_IO = None
_W_PLAN = None
_W_LAYER_SLICES = None
_W_SAMPLER_STATE = None


def _worker_init(ctx: dict[str, Any]) -> None:
    global _W_IO, _W_PLAN, _W_LAYER_SLICES, _W_SAMPLER_STATE
    LayerIO, _Raster, _Transformer = _extraction_runtime()
    _W_PLAN = LayerPlan.from_dict(ctx["plan_dict"])
    _W_LAYER_SLICES = {k: tuple(v) for k, v in ctx["layer_slices"].items()}
    _W_SAMPLER_STATE = ctx["sampler_state"]
    _W_IO = LayerIO(study_dir=Path(ctx["study_dir"]), plan=_W_PLAN)


def _worker_run(
    args: tuple[int, _StepRequest],
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
    int,
    float,
    float,
]:
    idx, req = args
    out = _extract_one_step(
        req,
        io=_W_IO,
        plan=_W_PLAN,
        layer_slices=_W_LAYER_SLICES,
        sampler_state=_W_SAMPLER_STATE,
    )
    return (idx,) + out


def _subset_plan_for_band_ids(plan: LayerPlan, band_ids: list[str]) -> LayerPlan:
    wanted = set(band_ids)
    channels = [ch for ch in plan.channels if ch.band_id in wanted]
    layers_in_order: list[str] = []
    seen: set[str] = set()
    for ch in channels:
        if ch.layer in seen:
            continue
        seen.add(ch.layer)
        layers_in_order.append(ch.layer)
    return LayerPlan(
        rasters_hash=str(plan.rasters_hash),
        step=plan.step,
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups={k: dict(v) for k, v in plan.chip_groups.items()},
        derived_dependencies={k: list(v) for k, v in plan.derived_dependencies.items()},
    )


def _atomic_save_npy(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)


def _build_step_requests(steps: pd.DataFrame) -> list[_StepRequest]:
    reqs: list[_StepRequest] = []
    for row in steps.to_dict(orient="records"):
        reqs.append(
            _StepRequest(
                point_id_start=int(row["point_id_start"]),
                point_id_end=int(row["point_id_end"]),
                timestamp=pd.Timestamp(row["timestamp"]),
                timestamp_end=pd.Timestamp(row["timestamp_end"]),
                start_lon=float(row["start_lon"]),
                start_lat=float(row["start_lat"]),
                start_bearing=float(row["start_bearing"]),
                end_lon=float(row["end_lon"]),
                end_lat=float(row["end_lat"]),
                actual_step_time_s=float(row["actual_step_time_s"]),
                step_time_error_s=float(row["step_time_error_s"]),
                individual_id=str(row["individual-local-identifier"]),
                burst_id=int(row["burst_id"]),
            )
        )
    return reqs


def _extract_env_block(
    *,
    study_id: str,
    study_dir: Path,
    plan: LayerPlan,
    reqs: list[_StepRequest],
    sampler_state: dict[str, Any],
    n_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    n_steps = int(len(reqs))
    points_per_step = int(sampler_state["num_candidates"]) + 1
    n_channels = int(plan.num_channels())
    env = torch.zeros((n_steps, n_channels, points_per_step), dtype=torch.float32)
    valid = torch.zeros((n_steps, n_channels, points_per_step), dtype=torch.bool)
    shared = {
        "candidate_lon": torch.zeros((n_steps, points_per_step), dtype=torch.float32),
        "candidate_lat": torch.zeros((n_steps, points_per_step), dtype=torch.float32),
        "step_length_m": torch.zeros((n_steps, points_per_step), dtype=torch.float32),
        "turn_angle_rad": torch.zeros((n_steps, points_per_step), dtype=torch.float32),
        "start_lon": torch.zeros((n_steps,), dtype=torch.float32),
        "start_lat": torch.zeros((n_steps,), dtype=torch.float32),
        "burst_id": torch.zeros((n_steps,), dtype=torch.int64),
        "individual_id": ["" for _ in range(n_steps)],
        "point_id_start": torch.zeros((n_steps,), dtype=torch.int64),
        "point_id_end": torch.zeros((n_steps,), dtype=torch.int64),
        "timestamp_ns": torch.zeros((n_steps,), dtype=torch.int64),
        "timestamp_end_ns": torch.zeros((n_steps,), dtype=torch.int64),
        "actual_step_time_s": torch.zeros((n_steps,), dtype=torch.float32),
        "step_time_error_s": torch.zeros((n_steps,), dtype=torch.float32),
    }
    layer_slices = {layer: plan.layer_slice(layer) for layer in plan.layers_in_order}

    if n_workers <= 1:
        LayerIO, _Raster, _Transformer = _extraction_runtime()
        io = LayerIO(study_dir=Path(study_dir), plan=plan)
        iterator = enumerate(reqs)
        for i, req in tqdm(
            iterator, total=n_steps, desc=f"{study_id}:{plan.rasters_hash} steps"
        ):
            (
                e,
                v,
                lons,
                lats,
                sl,
                ta,
                pid_start,
                pid_end,
                ts_start,
                ts_end,
                actual_step_time_s,
                step_time_error_s,
            ) = _extract_one_step(
                req,
                io=io,
                plan=plan,
                layer_slices=layer_slices,
                sampler_state=sampler_state,
            )
            env[i] = torch.from_numpy(e)
            valid[i] = torch.from_numpy(v)
            shared["candidate_lon"][i] = torch.from_numpy(lons)
            shared["candidate_lat"][i] = torch.from_numpy(lats)
            shared["step_length_m"][i] = torch.from_numpy(sl)
            shared["turn_angle_rad"][i] = torch.from_numpy(ta)
            shared["start_lon"][i] = float(req.start_lon)
            shared["start_lat"][i] = float(req.start_lat)
            shared["burst_id"][i] = int(req.burst_id)
            shared["individual_id"][i] = str(req.individual_id)
            shared["point_id_start"][i] = int(pid_start)
            shared["point_id_end"][i] = int(pid_end)
            shared["timestamp_ns"][i] = int(ts_start)
            shared["timestamp_end_ns"][i] = int(ts_end)
            shared["actual_step_time_s"][i] = float(actual_step_time_s)
            shared["step_time_error_s"][i] = float(step_time_error_s)
    else:
        ctx = mp.get_context("spawn")
        init_ctx = {
            "study_dir": str(study_dir),
            "plan_dict": plan.to_dict(),
            "layer_slices": {k: list(v) for k, v in layer_slices.items()},
            "sampler_state": sampler_state,
        }
        with ctx.Pool(
            processes=int(n_workers),
            initializer=_worker_init,
            initargs=(init_ctx,),
            maxtasksperchild=200,
        ) as pool:
            for (
                idx,
                e,
                v,
                lons,
                lats,
                sl,
                ta,
                pid_start,
                pid_end,
                ts_start,
                ts_end,
                actual_step_time_s,
                step_time_error_s,
            ) in tqdm(
                pool.imap_unordered(_worker_run, list(enumerate(reqs)), chunksize=10),
                total=n_steps,
                desc=f"{study_id}:{plan.rasters_hash} steps",
            ):
                env[idx] = torch.from_numpy(e)
                valid[idx] = torch.from_numpy(v)
                shared["candidate_lon"][idx] = torch.from_numpy(lons)
                shared["candidate_lat"][idx] = torch.from_numpy(lats)
                shared["step_length_m"][idx] = torch.from_numpy(sl)
                shared["turn_angle_rad"][idx] = torch.from_numpy(ta)
                req = reqs[idx]
                shared["start_lon"][idx] = float(req.start_lon)
                shared["start_lat"][idx] = float(req.start_lat)
                shared["burst_id"][idx] = int(req.burst_id)
                shared["individual_id"][idx] = str(req.individual_id)
                shared["point_id_start"][idx] = int(pid_start)
                shared["point_id_end"][idx] = int(pid_end)
                shared["timestamp_ns"][idx] = int(ts_start)
                shared["timestamp_end_ns"][idx] = int(ts_end)
                shared["actual_step_time_s"][idx] = float(actual_step_time_s)
                shared["step_time_error_s"][idx] = float(step_time_error_s)

    return env, valid, shared


def _manifest_matches_request(
    manifest: PointFeatureManifest,
    *,
    plan: LayerPlan,
    sampler_hash_value: str,
    points_per_step: int,
    selected_splits: list[str],
) -> bool:
    return (
        manifest.schema_version == "4"
        and manifest.rasters_hash == str(plan.rasters_hash)
        and manifest.sampler_hash == str(sampler_hash_value)
        and manifest.extraction.get("step_selector") == "interval"
        and int(manifest.extraction.get("step_time_s", -1))
        == int(plan.step_spec.step_time_s)
        and float(manifest.extraction.get("gap_q", float("nan")))
        == float(plan.step_spec.gap_q)
        and manifest.extraction.get("gap_cutoff_source") == "train"
        and int(manifest.extraction.get("points_per_step", -1)) == int(points_per_step)
        and list(manifest.extraction.get("selected_splits", [])) == list(selected_splits)
        and manifest.shared.get("path") == "shared/shared.npy"
        and set(manifest.bands.keys()) == set(plan.band_ids_in_order())
    )


def _shared_structured_array(
    shared: dict[str, Any],
    *,
    points_per_step: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_rows = int(len(shared["individual_id"]))
    max_chars = (
        max(1, max(len(str(x)) for x in shared["individual_id"]))
        if n_rows > 0
        else 1
    )
    dtype = np.dtype(
        [
            ("candidate_lon", np.float32, (points_per_step,)),
            ("candidate_lat", np.float32, (points_per_step,)),
            ("step_length_m", np.float32, (points_per_step,)),
            ("turn_angle_rad", np.float32, (points_per_step,)),
            ("start_lon", np.float32),
            ("start_lat", np.float32),
            ("burst_id", np.int64),
            ("individual_id", f"U{max_chars}"),
            ("point_id_start", np.int64),
            ("point_id_end", np.int64),
            ("timestamp_ns", np.int64),
            ("timestamp_end_ns", np.int64),
            ("actual_step_time_s", np.float32),
            ("step_time_error_s", np.float32),
        ]
    )
    out = np.zeros((n_rows,), dtype=dtype)
    fields_meta: dict[str, Any] = {}
    for key, value in shared.items():
        if key == "individual_id":
            arr = np.asarray([str(x) for x in value], dtype=f"U{max_chars}")
            fields_meta[key] = {
                "dtype": "unicode",
                "shape": [],
                "max_chars": int(max_chars),
            }
        else:
            arr = (
                value.detach().cpu().numpy()
                if torch.is_tensor(value)
                else np.asarray(value)
            )
            arr = np.array(arr, copy=False)
            fields_meta[key] = {
                "dtype": "float32"
                if arr.dtype == np.float32
                else "int64"
                if arr.dtype == np.int64
                else str(arr.dtype),
                "shape": [points_per_step] if arr.ndim == 2 else [],
            }
        out[key] = arr
    return out, {
        "path": "shared/shared.npy",
        "shape": [n_rows],
        "dtype": "structured",
        "fields": fields_meta,
    }


def _layer_store_meta(
    *,
    plan: LayerPlan,
    env: torch.Tensor,
    valid: torch.Tensor,
    out_dir: Path,
    n_rows: int,
    points_per_step: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    layers_meta: dict[str, dict[str, Any]] = {}
    bands_meta: dict[str, dict[str, Any]] = {}
    for layer in plan.layers_in_order:
        start, end = plan.layer_slice(layer)
        layer_env = env[:, start:end, :].contiguous().cpu().numpy()
        layer_valid = valid[:, start:end, :].contiguous().cpu().numpy()
        env_path = out_dir / "layers" / f"{layer}.env.npy"
        valid_path = out_dir / "layers" / f"{layer}.valid.npy"
        _atomic_save_npy(layer_env, env_path)
        _atomic_save_npy(layer_valid, valid_path)
        layer_channels = plan.channels[start:end]
        layer_band_ids = [channel.band_id for channel in layer_channels]
        layers_meta[str(layer)] = {
            "env_path": f"layers/{layer}.env.npy",
            "valid_path": f"layers/{layer}.valid.npy",
            "shape": [n_rows, len(layer_band_ids), points_per_step],
            "env_dtype": "float32",
            "valid_dtype": "bool",
            "band_ids": layer_band_ids,
        }
        for channel_idx, channel in enumerate(layer_channels):
            bands_meta[channel.band_id] = {
                "layer": str(layer),
                "channel_idx": int(channel_idx),
                "categorical": bool(channel.categorical),
                "normalize": bool(channel.normalize),
            }
    return layers_meta, bands_meta


def extract_split_feature_store(
    *,
    study_id: str,
    study_dir: Path,
    split: str,
    steps: pd.DataFrame,
    plan: LayerPlan,
    sampler: dict[str, Any],
    sampler_state: dict[str, Any],
    sampler_hash: str,
    selected_splits: list[str],
    n_workers: int,
    skip_existing: bool,
) -> dict[str, Any]:
    out_dir = point_store_dir(
        study_dir=Path(study_dir),
        split=str(split),
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )
    manifest_path = out_dir / "manifest.json"

    reqs = _build_step_requests(steps)
    n_steps = int(len(reqs))
    points_per_step = int(sampler["num_candidates"]) + 1

    if manifest_path.exists():
        try:
            existing_manifest = read_manifest(manifest_path)
        except Exception:
            existing_manifest = None
        else:
            if (
                skip_existing
                and _manifest_matches_request(
                    existing_manifest,
                    plan=plan,
                    sampler_hash_value=sampler_hash,
                    points_per_step=points_per_step,
                    selected_splits=selected_splits,
                )
            ):
                return {
                    "split": split,
                    "status": "existing",
                    "out_dir": str(out_dir),
                    "n_steps": int(existing_manifest.n_rows),
                }

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, valid, shared_data = _extract_env_block(
        study_id=study_id,
        study_dir=study_dir,
        plan=plan,
        reqs=reqs,
        sampler_state=sampler_state,
        n_workers=n_workers,
    )
    shared_array, shared_meta = _shared_structured_array(
        shared_data,
        points_per_step=points_per_step,
    )
    _atomic_save_npy(shared_array, out_dir / "shared" / "shared.npy")
    layers_meta, bands_meta = _layer_store_meta(
        plan=plan,
        env=env,
        valid=valid,
        out_dir=out_dir,
        n_rows=n_steps,
        points_per_step=points_per_step,
    )

    manifest = PointFeatureManifest(
        schema_version="4",
        study_id=str(study_id),
        split=str(split),
        created_at=utc_now_iso(),
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
        sampler=dict(sampler),
        extraction={
            "step_selector": "interval",
            "step_time_s": int(plan.step_spec.step_time_s),
            "gap_q": float(plan.step_spec.gap_q),
            "gap_cutoff_source": "train",
            "selected_splits": list(selected_splits),
            "points_per_step": int(points_per_step),
            "choice_policy": "last_is_true",
            "target_crs": TARGET_CRS,
        },
        shared=shared_meta,
        layers=layers_meta,
        bands=bands_meta,
    )
    write_manifest(manifest_path, manifest)
    return {
        "split": split,
        "status": "written",
        "out_dir": str(out_dir),
        "n_steps": n_steps,
    }


def extract_point_feature_stores(
    *,
    study_id: str,
    study_dir: Path,
    trajectories_path: Path,
    plan: LayerPlan,
    sampler_config: SamplerConfig,
    splits: list[str] | None,
    n_workers: int,
    skip_existing: bool,
) -> dict[str, Any]:
    df = load_trajectories_csv(Path(trajectories_path))
    if splits is None:
        splits = sorted(df[SET_COL].dropna().unique().tolist())
    else:
        splits = [canonicalize_split_key(split) for split in splits]

    # Match download_rasters(): select interval starts once over the combined
    # requested splits, then partition by the start row's split.
    selected_df = df[df[SET_COL].isin(splits)].copy()
    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("No training data found (set=='train'); cannot fit sampler")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        step_q=float(plan.step_spec.step_q),
        max_step_q=float(plan.step_spec.max_step_q),
        step_multiplier=float(plan.step_spec.step_multiplier),
    )
    requested_steps = select_step_starts_with_support(
        selected_df,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        support=support,
    )
    if len(splits) == 1 and str(splits[0]) == "train":
        train_steps = requested_steps.copy()
    else:
        train_steps = select_step_starts_with_support(
            train_df,
            step_time_s=int(plan.step_spec.step_time_s),
            gap_q=float(plan.step_spec.gap_q),
            support=support,
        )
    sampler, sampler_state, sampler_hash_value = build_train_sampler(
        train_steps,
        sampler_config=sampler_config,
        step_spec=plan.step_spec,
        max_step_len_m_train=float(support.max_step_length_m),
    )

    results: dict[str, Any] = {}
    for split in splits:
        split_steps = requested_steps[requested_steps[SET_COL] == split].copy()
        results[split] = extract_split_feature_store(
            study_id=study_id,
            study_dir=study_dir,
            split=split,
            steps=split_steps,
            plan=plan,
            sampler=sampler,
            sampler_state=sampler_state,
            sampler_hash=sampler_hash_value,
            selected_splits=list(splits),
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    return {
        "study_id": str(study_id),
        "rasters_hash": plan.rasters_hash,
        "sampler_hash": sampler_hash_value,
        "splits": results,
    }
