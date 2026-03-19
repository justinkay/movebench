"""Point feature extraction for SSF / iSSF workflows.

This module writes a disk-backed "point feature store" used by downstream
datasets/models. Each "step" corresponds to one start location with:
- P candidate endpoints (num_candidates pseudo + 1 true endpoint),
- environmental values sampled at each candidate,
- a validity mask per (channel, candidate),
- movement values (step length, turn angle) for each candidate.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from movebench.features.manifest import (
    PointFeatureManifest,
    read_manifest,
    utc_now_iso,
    write_manifest,
)
from movebench.layers.constants import WGS84_CRS
from movebench.layers.config import _format_slug_number, _truncate_key
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import REGISTRY
from movebench.rasters import Raster
from movebench.trajectories import (
    SET_COL,
    compute_movement_features,
    select_step_starts,
)

TARGET_CRS = "EPSG:3857"  # meters
SamplerType = Literal["empirical", "parametric"]
DEFAULT_NUM_CANDIDATES = 100
DEFAULT_RANDOM_SEED = 0
DEFAULT_STEP_MULTIPLIER = 2.0
DEFAULT_STEP_Q = 0.95
DEFAULT_MAX_STEP_Q = 0.99
DEFAULT_GAP_Q = 0.999
DEFAULT_BOUNDS_BUFFER_DEG = 0.01


@lru_cache(maxsize=1)
def _transformers():
    """Construct cached CRS transformers lazily (pyproj is an optional dependency for readers)."""
    from pyproj import (
        Transformer,
    )  # imported lazily to avoid requiring pyproj for point-store readers

    to_target = Transformer.from_crs(WGS84_CRS, TARGET_CRS, always_xy=True)
    to_4326 = Transformer.from_crs(TARGET_CRS, WGS84_CRS, always_xy=True)
    return to_target, to_4326


def _stable_payload(d: dict[str, Any]) -> bytes:
    return json.dumps(
        d, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _step_seed(random_seed: int, point_id_start: int) -> int:
    s = f"{int(random_seed)}|{int(point_id_start)}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def sampler_hash(
    *,
    parametric: bool,
    num_candidates: int,
    random_seed: int,
    step_multiplier: float,
    step_q: float,
    max_step_q: float,
    gap_q: float,
    bounds_buffer_deg: float,
) -> str:
    payload_dict = {
        "type": "parametric" if parametric else "empirical",
        "num_candidates": int(num_candidates),
        "random_seed": int(random_seed),
        "step_multiplier": float(step_multiplier),
        "step_q": float(step_q),
        "max_step_q": float(max_step_q),
        "gap_q": float(gap_q),
        "bounds_buffer_deg": float(bounds_buffer_deg),
    }
    payload = _stable_payload(payload_dict)
    parts = [
        f"tp{'par' if parametric else 'emp'}",
        f"nc{int(num_candidates)}",
        f"rs{int(random_seed)}",
        f"sm{_format_slug_number(float(step_multiplier))}",
        f"sq{_format_slug_number(float(step_q))}",
        f"msq{_format_slug_number(float(max_step_q))}",
        f"gq{_format_slug_number(float(gap_q))}",
        f"bb{_format_slug_number(float(bounds_buffer_deg))}",
    ]
    return _truncate_key("-".join(parts), payload=payload)


def resolve_sampler_hash(
    *,
    sampler_type: SamplerType,
    num_candidates: int | None = None,
    random_seed: int | None = None,
    step_multiplier: float | None = None,
    step_q: float | None = None,
    max_step_q: float | None = None,
    gap_q: float | None = None,
    bounds_buffer_deg: float | None = None,
) -> str:
    return sampler_hash(
        parametric=str(sampler_type) == "parametric",
        num_candidates=DEFAULT_NUM_CANDIDATES
        if num_candidates is None
        else int(num_candidates),
        random_seed=DEFAULT_RANDOM_SEED if random_seed is None else int(random_seed),
        step_multiplier=DEFAULT_STEP_MULTIPLIER
        if step_multiplier is None
        else float(step_multiplier),
        step_q=DEFAULT_STEP_Q if step_q is None else float(step_q),
        max_step_q=DEFAULT_MAX_STEP_Q if max_step_q is None else float(max_step_q),
        gap_q=DEFAULT_GAP_Q if gap_q is None else float(gap_q),
        bounds_buffer_deg=DEFAULT_BOUNDS_BUFFER_DEG
        if bounds_buffer_deg is None
        else float(bounds_buffer_deg),
    )


def point_stores_dir(*, study_dir: Path, split: str, rasters_hash: str) -> Path:
    return Path(study_dir) / "features" / "points" / str(split) / str(rasters_hash)


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


def load_point_store_shared(store_dir: Path) -> dict[str, torch.Tensor]:
    shared_path = Path(store_dir) / "shared.pt"
    if not shared_path.exists():
        raise FileNotFoundError(f"shared.pt not found under: {store_dir}")
    return torch.load(shared_path, map_location="cpu")


def load_point_store_bands(
    *,
    store_dir: Path,
    manifest: PointFeatureManifest,
    band_ids: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not band_ids:
        shared_shape = list(manifest.shared["shared.pt"]["shape"])
        return (
            torch.zeros((shared_shape[0], 0, shared_shape[1]), dtype=torch.float32),
            torch.zeros((shared_shape[0], 0, shared_shape[1]), dtype=torch.bool),
        )

    env_parts: list[torch.Tensor] = []
    valid_parts: list[torch.Tensor] = []
    for band_id in band_ids:
        meta = manifest.bands.get(band_id)
        if meta is None:
            raise ValueError(
                f"Requested band_id {band_id!r} not present in point store"
            )
        d = torch.load(Path(store_dir) / str(meta["path"]), map_location="cpu")
        env = d["env"]
        valid = d["valid"]
        if env.ndim != 2 or valid.ndim != 2:
            raise ValueError(f"Band {band_id!r} must have rank-2 env/valid tensors")
        env_parts.append(env.unsqueeze(1))
        valid_parts.append(valid.unsqueeze(1))
    return torch.cat(env_parts, dim=1), torch.cat(valid_parts, dim=1)


@dataclass(frozen=True)
class _StepRequest:
    point_id_start: int
    timestamp: Any
    start_lon: float
    start_lat: float
    start_bearing: float
    end_lon: float
    end_lat: float


def build_train_sampler(
    train_df: pd.DataFrame,
    *,
    parametric: bool,
    num_candidates: int,
    random_seed: int,
    step_multiplier: float,
    step_q: float,
    max_step_q: float,
    gap_q: float,
    bounds_buffer_deg: float,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Fit/train-only movement sampler for candidate generation."""
    valid_lengths = train_df.loc[train_df["valid_step"], "step_length_m"].to_numpy(
        dtype=np.float64
    )
    valid_turns = train_df["turn_angle_rad"].dropna().to_numpy(dtype=np.float64)

    if len(valid_lengths) == 0:
        raise ValueError(
            "No valid step lengths in train split; cannot build candidate sampler"
        )
    if len(valid_turns) == 0:
        raise ValueError(
            "No valid turn angles in train split; cannot build candidate sampler"
        )

    max_step_len = (
        float(train_df["max_step_len"].iloc[0]) if len(train_df) else float("nan")
    )

    sampler: dict[str, Any] = {
        "type": "parametric" if parametric else "empirical",
        "fit_source_split": "train",
        "num_candidates": int(num_candidates),
        "random_seed": int(random_seed),
        "step_multiplier": float(step_multiplier),
        "step_q": float(step_q),
        "max_step_q": float(max_step_q),
        "gap_q": float(gap_q),
        "bounds_buffer_deg": float(bounds_buffer_deg),
        "n_valid_lengths_train": int(len(valid_lengths)),
        "n_valid_turns_train": int(len(valid_turns)),
        "max_step_len_m_train": float(max_step_len),
    }

    state: dict[str, Any] = {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(num_candidates),
        "random_seed": int(random_seed),
        "gamma_used": False,
        "gamma_params": None,
        "vonmises_used": False,
        "vonmises_params": None,
    }

    if parametric:
        gamma_used = False
        vonmises_used = False
        gamma_params = None
        vonmises_params = None

        try:
            from scipy import stats  # type: ignore

            try:
                gamma_params = stats.gamma.fit(valid_lengths, floc=0)
                gamma_used = True
            except Exception:
                gamma_used = False
                gamma_params = None

            try:
                vonmises_params = stats.vonmises.fit(valid_turns)
                vonmises_used = True
            except Exception:
                vonmises_used = False
                vonmises_params = None

        except Exception:
            gamma_used = False
            vonmises_used = False
            gamma_params = None
            vonmises_params = None

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

    return (
        sampler,
        state,
        sampler_hash(
            parametric=parametric,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            max_step_q=max_step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        ),
    )


@torch.no_grad()
def _sample_from_raster(
    raster: Raster,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    bilinear: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample raster at point locations (EPSG:4326) in raster's native CRS."""
    from pyproj import (
        Transformer,
    )  # imported lazily to avoid requiring pyproj for point-store readers

    arr = raster.array
    transform = raster.transform
    src_crs = raster.crs
    nodata = raster.nodata

    C, H, W = arr.shape
    N = int(len(lons))

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

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_clamped = np.clip(rows, 0, H - 1)
        cols_clamped = np.clip(cols, 0, W - 1)

        x_norm = 2.0 * (cols_clamped / max(W - 1, 1)) - 1.0
        y_norm = 2.0 * (rows_clamped / max(H - 1, 1)) - 1.0

        grid = torch.stack(
            [
                torch.tensor(x_norm, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32),
            ],
            dim=-1,
        ).view(1, N, 1, 2)

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

    pseudo_lens: np.ndarray
    pseudo_turns: np.ndarray

    if state.get("gamma_used", False) and state.get("gamma_params") is not None:
        from scipy import stats  # type: ignore

        pseudo_lens = stats.gamma.rvs(
            *state["gamma_params"], size=n, random_state=rng
        ).astype(np.float64, copy=False)
    else:
        pseudo_lens = rng.choice(valid_lengths, size=n, replace=True).astype(
            np.float64, copy=False
        )

    if state.get("vonmises_used", False) and state.get("vonmises_params") is not None:
        from scipy import stats  # type: ignore

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
    bounds_buffer_deg: float,
) -> _CandidateSet:
    pseudo_lens, pseudo_turns = _sample_pseudo_steps(sampler_state, req.point_id_start)

    to_target, to_4326 = _transformers()
    x0, y0 = to_target.transform(float(req.start_lon), float(req.start_lat))
    x1, y1 = to_target.transform(float(req.end_lon), float(req.end_lat))

    headings = float(req.start_bearing) + pseudo_turns
    headings = (headings + np.pi) % (2.0 * np.pi) - np.pi

    dx = pseudo_lens * np.sin(headings)
    dy = pseudo_lens * np.cos(headings)
    xs = x0 + dx
    ys = y0 + dy

    # True endpoint appended last.
    xs = np.append(xs, x1)
    ys = np.append(ys, y1)

    # Step lengths / turn angles for all candidates.
    true_sl = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    true_heading = float(np.arctan2(x1 - x0, y1 - y0))
    true_ta = true_heading - float(req.start_bearing)
    true_ta = float((true_ta + np.pi) % (2.0 * np.pi) - np.pi)

    step_lengths = np.append(pseudo_lens, true_sl)
    turn_angles = np.append(pseudo_turns, true_ta)

    lons, lats = to_4326.transform(xs, ys)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    b = float(bounds_buffer_deg)
    bounds_4326 = (
        float(np.min(lons) - b),
        float(np.min(lats) - b),
        float(np.max(lons) + b),
        float(np.max(lats) + b),
    )

    return _CandidateSet(
        lons=lons,
        lats=lats,
        bounds_4326=bounds_4326,
        step_lengths_m=step_lengths.astype(np.float32, copy=False),
        turn_angles_rad=turn_angles.astype(np.float32, copy=False),
    )


def _ensure_timestamp_ns(ts: Any) -> int:
    if hasattr(ts, "value"):  # pandas Timestamp
        return int(ts.value)
    if hasattr(ts, "timestamp"):
        return int(float(ts.timestamp()) * 1e9)
    raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")


def _extract_one_step(
    req: _StepRequest,
    *,
    io,
    plan: LayerPlan,
    layer_slices: dict[str, tuple[int, int]],
    sampler_state: dict[str, Any],
    bounds_buffer_deg: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int
]:
    cand = _compute_candidates(
        req, sampler_state=sampler_state, bounds_buffer_deg=bounds_buffer_deg
    )
    P = int(len(cand.lons))
    C = int(plan.num_channels())

    env = torch.zeros((C, P), dtype=torch.float32)
    valid = torch.zeros((C, P), dtype=torch.bool)

    cache: dict[str, Raster] = {}

    def get_raster(layer: str) -> Raster:
        cached = cache.get(layer)
        if cached is not None:
            return cached

        spec = REGISTRY.get(layer)
        if spec.source == "derived":
            if not spec.derived_from:
                raise ValueError(f"Derived layer {layer!r} must define derived_from")
            parents = {p: get_raster(p) for p in spec.derived_from}
            import movebench.layers.derived as derived

            fn = getattr(derived, f"compute_{layer}", None)
            if fn is None:
                raise ValueError(
                    f"Missing derived compute function compute_{layer} in movebench.layers.derived"
                )
            arr = fn(**parents)
            base = parents[spec.derived_from[0]]
            from movebench.rasters import Raster

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
            # Expect a single band with integer class labels.
            class_vals = vals.squeeze(0).to(torch.int64)
            class_vals = torch.clamp(class_vals, 0, int(spec.num_classes) - 1)
            one_hot = F.one_hot(class_vals, int(spec.num_classes)).T.to(torch.float32)
            vexp = vmask.expand(int(spec.num_classes), -1)
            one_hot = one_hot * vexp.to(one_hot.dtype)

            env[start:end] = one_hot
            valid[start:end] = vexp
            continue

        vals, vmask = _apply_valid_range(vals, vmask, spec.valid_range)
        env[start:end] = vals
        valid[start:end] = vmask

    # Invalid pixels must be materialized as 0.0 in env.
    env = torch.where(valid, env, torch.zeros_like(env))

    return (
        env.numpy(),
        valid.numpy(),
        cand.lons.astype(np.float32, copy=False),
        cand.lats.astype(np.float32, copy=False),
        cand.step_lengths_m,
        cand.turn_angles_rad,
        int(req.point_id_start),
        _ensure_timestamp_ns(req.timestamp),
    )


# ---- multiprocessing wiring ----

_W_IO = None
_W_PLAN = None
_W_LAYER_SLICES = None
_W_SAMPLER_STATE = None
_W_BOUNDS_BUFFER_DEG = None


def _worker_init(ctx: dict[str, Any]) -> None:
    global _W_IO, _W_PLAN, _W_LAYER_SLICES, _W_SAMPLER_STATE, _W_BOUNDS_BUFFER_DEG

    from movebench.layers.io import LayerIO

    _W_PLAN = LayerPlan.from_dict(ctx["plan_dict"])
    _W_LAYER_SLICES = {k: tuple(v) for k, v in ctx["layer_slices"].items()}
    _W_SAMPLER_STATE = ctx["sampler_state"]
    _W_BOUNDS_BUFFER_DEG = float(ctx["bounds_buffer_deg"])
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
]:
    idx, req = args
    out = _extract_one_step(
        req,
        io=_W_IO,
        plan=_W_PLAN,
        layer_slices=_W_LAYER_SLICES,
        sampler_state=_W_SAMPLER_STATE,
        bounds_buffer_deg=_W_BOUNDS_BUFFER_DEG,
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
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups={k: dict(v) for k, v in plan.chip_groups.items()},
        derived_dependencies={k: list(v) for k, v in plan.derived_dependencies.items()},
    )


def _torch_atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _build_step_requests(steps: pd.DataFrame) -> list[_StepRequest]:
    reqs: list[_StepRequest] = []
    for row in steps.itertuples(index=False):
        reqs.append(
            _StepRequest(
                point_id_start=int(row.point_id_start),
                timestamp=pd.Timestamp(row.timestamp),
                start_lon=float(row.start_lon),
                start_lat=float(row.start_lat),
                start_bearing=float(row.start_bearing),
                end_lon=float(row.end_lon),
                end_lat=float(row.end_lat),
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
    bounds_buffer_deg: float,
    n_workers: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
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
        "point_id_start": torch.zeros((n_steps,), dtype=torch.int64),
        "timestamp_ns": torch.zeros((n_steps,), dtype=torch.int64),
    }
    layer_slices = {layer: plan.layer_slice(layer) for layer in plan.layers_in_order}

    if n_workers <= 1:
        from movebench.layers.io import LayerIO

        io = LayerIO(study_dir=Path(study_dir), plan=plan)
        for i, req in enumerate(
            tqdm(reqs, total=n_steps, desc=f"{study_id}:{plan.rasters_hash} steps")
        ):
            e, v, lons, lats, sl, ta, pid, tsn = _extract_one_step(
                req,
                io=io,
                plan=plan,
                layer_slices=layer_slices,
                sampler_state=sampler_state,
                bounds_buffer_deg=bounds_buffer_deg,
            )
            env[i] = torch.from_numpy(e)
            valid[i] = torch.from_numpy(v)
            shared["candidate_lon"][i] = torch.from_numpy(lons)
            shared["candidate_lat"][i] = torch.from_numpy(lats)
            shared["step_length_m"][i] = torch.from_numpy(sl)
            shared["turn_angle_rad"][i] = torch.from_numpy(ta)
            shared["point_id_start"][i] = int(pid)
            shared["timestamp_ns"][i] = int(tsn)
    else:
        ctx = mp.get_context("spawn")
        init_ctx = {
            "study_dir": str(study_dir),
            "plan_dict": plan.to_dict(),
            "layer_slices": {k: list(v) for k, v in layer_slices.items()},
            "sampler_state": sampler_state,
            "bounds_buffer_deg": float(bounds_buffer_deg),
        }
        with ctx.Pool(
            processes=int(n_workers),
            initializer=_worker_init,
            initargs=(init_ctx,),
            maxtasksperchild=200,
        ) as pool:
            for idx, e, v, lons, lats, sl, ta, pid, tsn in tqdm(
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
                shared["point_id_start"][idx] = int(pid)
                shared["timestamp_ns"][idx] = int(tsn)

    return env, valid, shared


def _shared_meta(
    shared: dict[str, torch.Tensor], *, points_per_step: int
) -> dict[str, dict[str, Any]]:
    n_steps = int(shared["point_id_start"].shape[0])
    return {"shared.pt": {"shape": [n_steps, int(points_per_step)], "dtype": "mixed"}}


def _band_meta(
    *, band_id: str, channel, n_steps: int, points_per_step: int
) -> dict[str, Any]:
    return {
        "path": f"bands/{band_id}.pt",
        "shape": [int(n_steps), int(points_per_step)],
        "dtype": "float32/bool",
        "layer": str(channel.layer),
        "categorical": bool(channel.categorical),
        "normalize": bool(channel.normalize),
    }


def extract_split_feature_store(
    *,
    study_id: str,
    study_dir: Path,
    split: str,
    df_split: pd.DataFrame,
    plan: LayerPlan,
    sampler: dict[str, Any],
    sampler_state: dict[str, Any],
    sampler_hash: str,
    num_candidates: int,
    bounds_buffer_deg: float,
    n_workers: int,
    skip_existing: bool,
) -> dict[str, Any]:
    """Extract one split into a point feature store on disk."""
    out_dir = point_store_dir(
        study_dir=Path(study_dir),
        split=str(split),
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )
    manifest_path = out_dir / "manifest.json"
    shared_path = out_dir / "shared.pt"

    existing_manifest: PointFeatureManifest | None = None
    if manifest_path.exists():
        existing_manifest = read_manifest(manifest_path)
        if existing_manifest.rasters_hash != str(plan.rasters_hash):
            raise ValueError(
                "Existing manifest rasters_hash does not match requested plan"
            )
        if existing_manifest.sampler_hash != str(sampler_hash):
            raise ValueError(
                "Existing manifest sampler_hash does not match requested sampler_hash"
            )

    existing_band_ids = (
        set(existing_manifest.bands.keys()) if existing_manifest is not None else set()
    )
    requested_band_ids = plan.band_ids_in_order()
    missing_band_ids = [
        band_id for band_id in requested_band_ids if band_id not in existing_band_ids
    ]

    if (
        skip_existing
        and existing_manifest is not None
        and not missing_band_ids
        and shared_path.exists()
    ):
        return {
            "split": split,
            "status": "existing",
            "out_dir": str(out_dir),
            "n_steps": int(existing_manifest.shared["shared.pt"]["shape"][0]),
        }

    steps = select_step_starts(
        df_split,
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        max_step_q=float(sampler["max_step_q"]),
        gap_q=float(sampler["gap_q"]),
    )
    reqs = _build_step_requests(steps)
    n_steps = int(len(reqs))
    points_per_step = int(num_candidates) + 1

    if existing_manifest is None:
        write_band_ids = requested_band_ids
        write_plan = plan
        missing_band_ids = requested_band_ids
    else:
        write_band_ids = missing_band_ids
        write_plan = (
            _subset_plan_for_band_ids(plan, write_band_ids) if write_band_ids else plan
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    shared_data: dict[str, torch.Tensor]
    if existing_manifest is None:
        env, valid, shared_data = _extract_env_block(
            study_id=study_id,
            study_dir=study_dir,
            plan=write_plan,
            reqs=reqs,
            sampler_state=sampler_state,
            bounds_buffer_deg=bounds_buffer_deg,
            n_workers=n_workers,
        )
        _torch_atomic_save(shared_data, shared_path)
    else:
        shared_data = torch.load(shared_path, map_location="cpu")
        shape = list(existing_manifest.shared["shared.pt"]["shape"])
        if shape != [n_steps, points_per_step]:
            raise ValueError(
                f"Existing shared.pt shape {shape} does not match requested [{n_steps}, {points_per_step}]"
            )
        if write_band_ids:
            env, valid, _unused_shared = _extract_env_block(
                study_id=study_id,
                study_dir=study_dir,
                plan=write_plan,
                reqs=reqs,
                sampler_state=sampler_state,
                bounds_buffer_deg=bounds_buffer_deg,
                n_workers=n_workers,
            )

    bands_meta = (
        {}
        if existing_manifest is None
        else {k: dict(v) for k, v in existing_manifest.bands.items()}
    )
    if write_band_ids:
        n_write_channels = int(write_plan.num_channels())
        for i, channel in enumerate(write_plan.channels):
            band_id = channel.band_id
            band_path = out_dir / "bands" / f"{band_id}.pt"
            if band_path.exists() and band_id not in existing_band_ids:
                raise ValueError(
                    f"Band file already exists but is not tracked in manifest: {band_path}"
                )
            _torch_atomic_save(
                {"env": env[:, i, :], "valid": valid[:, i, :]}, band_path
            )
            bands_meta[band_id] = _band_meta(
                band_id=band_id,
                channel=channel,
                n_steps=n_steps,
                points_per_step=points_per_step,
            )
        assert n_write_channels == len(write_band_ids)

    manifest = PointFeatureManifest(
        schema_version="2",
        study_id=str(study_id),
        split=str(split),
        created_at=utc_now_iso(),
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
        sampler=dict(sampler),
        extraction={
            "points_per_step": int(points_per_step),
            "bounds_buffer_deg": float(bounds_buffer_deg),
            "choice_policy": "last_is_true",
            "target_crs": TARGET_CRS,
        },
        shared=_shared_meta(shared_data, points_per_step=points_per_step),
        bands=bands_meta,
    )
    write_manifest(manifest_path, manifest)
    return {
        "split": split,
        "status": "written"
        if write_band_ids or existing_manifest is None
        else "existing",
        "out_dir": str(out_dir),
        "n_steps": n_steps,
        "n_bands_written": int(len(write_band_ids)),
    }


def extract_point_feature_stores(
    *,
    study_id: str,
    study_dir: Path,
    trajectories_path: Path,
    plan: LayerPlan,
    splits: list[str] | None,
    parametric: bool,
    num_candidates: int,
    random_seed: int,
    n_workers: int,
    skip_existing: bool,
    step_multiplier: float,
    step_q: float,
    max_step_q: float,
    gap_q: float,
    bounds_buffer_deg: float,
) -> dict[str, Any]:
    """Extract point feature stores for one study."""
    df = pd.read_csv(Path(trajectories_path))

    if splits is None:
        splits = sorted(df[SET_COL].dropna().unique().tolist())

    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("No training data found (set=='train'); cannot fit sampler")
    train_df = compute_movement_features(
        train_df,
        step_multiplier=step_multiplier,
        step_q=step_q,
        max_step_q=max_step_q,
        gap_q=gap_q,
    )

    sampler, sampler_state, sampler_hash = build_train_sampler(
        train_df,
        parametric=parametric,
        num_candidates=num_candidates,
        random_seed=random_seed,
        step_multiplier=step_multiplier,
        step_q=step_q,
        max_step_q=max_step_q,
        gap_q=gap_q,
        bounds_buffer_deg=bounds_buffer_deg,
    )

    results: dict[str, Any] = {}
    for split in splits:
        split_df = df[df[SET_COL] == split].copy()
        results[split] = extract_split_feature_store(
            study_id=study_id,
            study_dir=study_dir,
            split=split,
            df_split=split_df,
            plan=plan,
            sampler=sampler,
            sampler_state=sampler_state,
            sampler_hash=sampler_hash,
            num_candidates=num_candidates,
            bounds_buffer_deg=bounds_buffer_deg,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    return {
        "study_id": str(study_id),
        "rasters_hash": plan.rasters_hash,
        "sampler_hash": sampler_hash,
        "splits": results,
    }
