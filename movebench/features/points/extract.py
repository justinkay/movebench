from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from movebench.features.points.candidates import StepRequest, compute_candidates
from movebench.features.points.config import PointStoreConfig
from movebench.features.points.manifest import PointStoreManifest, utc_now_iso
from movebench.features.points.raster_sampling import apply_valid_range, sample_from_raster
from movebench.features.points.store import point_store_dir, save_point_store
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import REGISTRY
from movebench.trajectories import (
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    compute_movement_features,
    select_step_starts,
)


def _ensure_timestamp_ns(ts: Any) -> int:
    if hasattr(ts, "value"):  # pandas Timestamp
        return int(ts.value)
    if hasattr(ts, "timestamp"):
        return int(float(ts.timestamp()) * 1e9)
    raise TypeError(f"Unsupported timestamp type: {type(ts)!r}")


def _should_bilinear(*, layer: str) -> bool:
    spec = REGISTRY.get(layer)
    # Historical exceptions kept for parity with prior extraction.
    return (not spec.categorical) and (layer not in {"dynamic_world", "landsat_ndwi"})


def _extract_one_step(
    req: StepRequest,
    *,
    io,
    plan: LayerPlan,
    layer_slices: dict[str, tuple[int, int]],
    sampler_state: dict[str, Any],
    bounds_buffer_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cand = compute_candidates(req, sampler_state=sampler_state, bounds_buffer_deg=bounds_buffer_deg)
    P = int(len(cand.lons))
    C = int(plan.num_channels())

    env = torch.zeros((C, P), dtype=torch.float32)
    valid = torch.zeros((C, P), dtype=torch.bool)

    cache: dict[str, Any] = {}

    for layer in plan.layers_in_order:
        start, end = layer_slices[layer]
        spec = REGISTRY.get(layer)

        r = io.load_layer_cached(
            layer,
            cache=cache,
            point_id=req.point_id_start,
            bounds_latlon=cand.bounds_4326,
            timestamp=req.timestamp_start,
        )

        vals, vmask = sample_from_raster(r, cand.lons, cand.lats, bilinear=_should_bilinear(layer=layer))

        if spec.categorical:
            class_vals = vals.squeeze(0).to(torch.int64)
            class_vals = torch.clamp(class_vals, 0, int(spec.num_classes) - 1)
            one_hot = F.one_hot(class_vals, int(spec.num_classes)).T.to(torch.float32)
            vexp = vmask.expand(int(spec.num_classes), -1)
            one_hot = one_hot * vexp.to(one_hot.dtype)

            env[start:end] = one_hot
            valid[start:end] = vexp
            continue

        vals, vmask = apply_valid_range(vals, vmask, spec.valid_range)
        env[start:end] = vals
        valid[start:end] = vmask

    env = torch.where(valid, env, torch.zeros_like(env))
    return (
        env.numpy(),
        valid.numpy(),
        cand.step_lengths_m,
        cand.turn_angles_rad,
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


def _worker_run(args: tuple[int, StepRequest]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx, req = args
    e, v, sl, ta = _extract_one_step(
        req,
        io=_W_IO,
        plan=_W_PLAN,
        layer_slices=_W_LAYER_SLICES,
        sampler_state=_W_SAMPLER_STATE,
        bounds_buffer_deg=_W_BOUNDS_BUFFER_DEG,
    )
    return idx, e, v, sl, ta


def extract_point_store_split(
    *,
    study_id: str,
    study_dir: Path,
    split: str,
    df_split: pd.DataFrame,
    plan: LayerPlan,
    config_path: str | None,
    store_cfg: PointStoreConfig,
    sampler_summary: dict[str, Any],
    sampler_state: dict[str, Any],
    n_workers: int,
    skip_existing: bool,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract one split into a point feature store on disk."""
    if not store_cfg.is_fully_specified():
        raise ValueError("extract_point_store_split requires a fully specified PointStoreConfig.")

    store_id = store_cfg.store_id()
    out_dir = point_store_dir(
        study_dir=Path(study_dir),
        split=str(split),
        config_hash=str(plan.config_hash),
        store_id=str(store_id),
    )
    out_pt = out_dir / "store.pt"
    out_manifest = out_dir / "manifest.json"

    if skip_existing and out_pt.exists() and out_manifest.exists():
        return {"split": split, "status": "existing", "out_dir": str(out_dir), "store_id": str(store_id)}

    steps = select_step_starts(
        df_split,
        step_multiplier=float(store_cfg.step_multiplier),
        step_q=float(store_cfg.step_q),
        gap_q=float(store_cfg.gap_q),
    )
    n_steps = int(len(steps))
    if n_steps == 0:
        raise ValueError(f"No valid step starts found for split={split!r}.")

    P = int(store_cfg.num_candidates) + 1
    C = int(plan.num_channels())

    layer_slices = {layer: plan.layer_slice(layer) for layer in plan.layers_in_order}

    reqs: list[StepRequest] = []
    for row in steps.itertuples(index=False):
        reqs.append(
            StepRequest(
                point_id_start=int(row.point_id_start),
                point_id_end=int(row.point_id_end),
                timestamp_start=pd.Timestamp(row.timestamp),
                timestamp_end=pd.Timestamp(row.end_timestamp),
                start_lon=float(row.start_lon),
                start_lat=float(row.start_lat),
                start_bearing=float(row.start_bearing),
                end_lon=float(row.end_lon),
                end_lat=float(row.end_lat),
            )
        )

    env = torch.zeros((n_steps, C, P), dtype=torch.float32)
    valid = torch.zeros((n_steps, C, P), dtype=torch.bool)
    step_length_m = torch.zeros((n_steps, P), dtype=torch.float32)
    turn_angle_rad = torch.zeros((n_steps, P), dtype=torch.float32)

    point_id_start = torch.zeros((n_steps,), dtype=torch.int64)
    point_id_end = torch.zeros((n_steps,), dtype=torch.int64)
    timestamp_ns = torch.zeros((n_steps,), dtype=torch.int64)
    end_timestamp_ns = torch.zeros((n_steps,), dtype=torch.int64)

    start_lon = torch.zeros((n_steps,), dtype=torch.float32)
    start_lat = torch.zeros((n_steps,), dtype=torch.float32)
    start_bearing = torch.zeros((n_steps,), dtype=torch.float32)
    end_lon = torch.zeros((n_steps,), dtype=torch.float32)
    end_lat = torch.zeros((n_steps,), dtype=torch.float32)

    if n_workers <= 1:
        from movebench.layers.io import LayerIO

        io = LayerIO(study_dir=Path(study_dir), plan=plan)
        for i, req in enumerate(tqdm(reqs, total=n_steps, desc=f"{study_id}:{split} steps")):
            e, v, sl, ta = _extract_one_step(
                req,
                io=io,
                plan=plan,
                layer_slices=layer_slices,
                sampler_state=sampler_state,
                bounds_buffer_deg=float(store_cfg.bounds_buffer_deg),
            )
            env[i] = torch.from_numpy(e)
            valid[i] = torch.from_numpy(v)
            step_length_m[i] = torch.from_numpy(sl)
            turn_angle_rad[i] = torch.from_numpy(ta)
            point_id_start[i] = int(req.point_id_start)
            point_id_end[i] = int(req.point_id_end)
            timestamp_ns[i] = int(_ensure_timestamp_ns(req.timestamp_start))
            end_timestamp_ns[i] = int(_ensure_timestamp_ns(req.timestamp_end))
            start_lon[i] = float(req.start_lon)
            start_lat[i] = float(req.start_lat)
            start_bearing[i] = float(req.start_bearing)
            end_lon[i] = float(req.end_lon)
            end_lat[i] = float(req.end_lat)
    else:
        ctx = mp.get_context("spawn")
        init_ctx = {
            "study_dir": str(study_dir),
            "plan_dict": plan.to_dict(),
            "layer_slices": {k: list(v) for k, v in layer_slices.items()},
            "sampler_state": sampler_state,
            "bounds_buffer_deg": float(store_cfg.bounds_buffer_deg),
        }

        with ctx.Pool(processes=int(n_workers), initializer=_worker_init, initargs=(init_ctx,), maxtasksperchild=200) as pool:
            for idx, e, v, sl, ta in tqdm(
                pool.imap_unordered(_worker_run, list(enumerate(reqs)), chunksize=10),
                total=n_steps,
                desc=f"{study_id}:{split} steps",
            ):
                req = reqs[int(idx)]
                env[idx] = torch.from_numpy(e)
                valid[idx] = torch.from_numpy(v)
                step_length_m[idx] = torch.from_numpy(sl)
                turn_angle_rad[idx] = torch.from_numpy(ta)
                point_id_start[idx] = int(req.point_id_start)
                point_id_end[idx] = int(req.point_id_end)
                timestamp_ns[idx] = int(_ensure_timestamp_ns(req.timestamp_start))
                end_timestamp_ns[idx] = int(_ensure_timestamp_ns(req.timestamp_end))
                start_lon[idx] = float(req.start_lon)
                start_lat[idx] = float(req.start_lat)
                start_bearing[idx] = float(req.start_bearing)
                end_lon[idx] = float(req.end_lon)
                end_lat[idx] = float(req.end_lat)

    extraction = {
        "points_per_step": int(P),
        "bounds_buffer_deg": float(store_cfg.bounds_buffer_deg),
        "choice_policy": "last_is_true",
        "target_crs": "EPSG:3857",
    }
    tensors_meta = {
        "env": {"shape": list(env.shape), "dtype": str(env.dtype)},
        "valid": {"shape": list(valid.shape), "dtype": str(valid.dtype)},
        "step_length_m": {"shape": list(step_length_m.shape), "dtype": str(step_length_m.dtype)},
        "turn_angle_rad": {"shape": list(turn_angle_rad.shape), "dtype": str(turn_angle_rad.dtype)},
        "point_id_start": {"shape": list(point_id_start.shape), "dtype": str(point_id_start.dtype)},
        "point_id_end": {"shape": list(point_id_end.shape), "dtype": str(point_id_end.dtype)},
        "timestamp_ns": {"shape": list(timestamp_ns.shape), "dtype": str(timestamp_ns.dtype)},
        "end_timestamp_ns": {"shape": list(end_timestamp_ns.shape), "dtype": str(end_timestamp_ns.dtype)},
        "start_lon": {"shape": list(start_lon.shape), "dtype": str(start_lon.dtype)},
        "start_lat": {"shape": list(start_lat.shape), "dtype": str(start_lat.dtype)},
        "start_bearing": {"shape": list(start_bearing.shape), "dtype": str(start_bearing.dtype)},
        "end_lon": {"shape": list(end_lon.shape), "dtype": str(end_lon.dtype)},
        "end_lat": {"shape": list(end_lat.shape), "dtype": str(end_lat.dtype)},
    }

    manifest = PointStoreManifest(
        schema_version="1",
        study_id=str(study_id),
        split=str(split),
        created_at=utc_now_iso(),
        config_path=str(config_path) if config_path is not None else None,
        config_hash=str(plan.config_hash),
        store_id=str(store_id),
        layer_plan=plan,
        store_config=store_cfg.to_dict(),
        sampler=sampler_summary,
        extraction=extraction,
        tensors=tensors_meta,
    )

    tensors = {
        "env": env,
        "valid": valid,
        "step_length_m": step_length_m,
        "turn_angle_rad": turn_angle_rad,
        "point_id_start": point_id_start,
        "point_id_end": point_id_end,
        "timestamp_ns": timestamp_ns,
        "end_timestamp_ns": end_timestamp_ns,
        "start_lon": start_lon,
        "start_lat": start_lat,
        "start_bearing": start_bearing,
        "end_lon": end_lon,
        "end_lat": end_lat,
    }

    save_point_store(store_dir=out_dir, tensors=tensors, manifest=manifest, overwrite=overwrite)
    return {"split": split, "status": "written", "out_dir": str(out_dir), "store_id": str(store_id), "n_steps": n_steps}


def extract_point_feature_stores(
    *,
    study_id: str,
    study_dir: Path,
    trajectories_path: Path,
    plan: LayerPlan,
    config_path: str | None,
    splits: list[str] | None,
    parametric: bool,
    num_candidates: int,
    random_seed: int,
    n_workers: int,
    skip_existing: bool,
    step_multiplier: float,
    step_q: float,
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
    train_df = compute_movement_features(train_df, step_multiplier=step_multiplier, step_q=step_q, gap_q=gap_q)

    store_cfg = PointStoreConfig(
        sampler_type="parametric" if parametric else "empirical",
        num_candidates=int(num_candidates),
        random_seed=int(random_seed),
        step_multiplier=float(step_multiplier),
        step_q=float(step_q),
        gap_q=float(gap_q),
        bounds_buffer_deg=float(bounds_buffer_deg),
    )

    from movebench.features.points.sampler import fit_sampler

    sampler_summary, sampler_state = fit_sampler(train_df, cfg=store_cfg)

    results: dict[str, Any] = {}
    for split in splits:
        split_df = df[df[SET_COL] == split].copy()
        results[split] = extract_point_store_split(
            study_id=study_id,
            study_dir=study_dir,
            split=split,
            df_split=split_df,
            plan=plan,
            config_path=config_path,
            store_cfg=store_cfg,
            sampler_summary=sampler_summary,
            sampler_state=sampler_state,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    return {"study_id": str(study_id), "config_hash": plan.config_hash, "store_id": store_cfg.store_id(), "splits": results}
