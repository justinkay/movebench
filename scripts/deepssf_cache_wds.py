#!/usr/bin/env python
"""Build and benchmark WebDataset deepSSF caches.

This script is a convenience wrapper around `movebench.datasets.deepssf_wds_cache`.
It can:
  - build caches for one or more studies/splits, and
  - benchmark baseline (on-the-fly rasters) vs cached throughput.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from movebench import Study
from movebench.datasets.deepssf_wds_cache import (
    ensure_deepssf_wds_cache_split,
    make_deepssf_wds_loader,
)
from movebench.models.deepssf import DeepSSF
from movebench.train import collate_deepssf


# Match configs/benchmarks/default.yaml iSSF sampler settings (used for normalization stats).
NUM_CANDIDATES = 100
SAMPLER_RANDOM_SEED = 0
BOUNDS_BUFFER_DEG = 0.01


def _parse_hw(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("output_hw must be like '101,101'")
    return int(parts[0]), int(parts[1])


def _parse_splits(s: str) -> list[str]:
    out = [p.strip() for p in str(s).split(",") if p.strip()]
    if not out:
        raise ValueError("splits must be a comma-separated list like 'train,test'")
    return out


def _parse_studies(val: str) -> list[str]:
    v = str(val).strip()
    if not v or v.lower() == "all":
        return Study.list_available()
    return [p.strip() for p in v.split(",") if p.strip()]


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += int((Path(root) / f).stat().st_size)
            except FileNotFoundError:
                continue
    return int(total)


def _ensure_ready(*, study: Study, layers_config: str, setup_workers: int, step_multiplier: float, step_q: float, gap_q: float) -> str:
    """Ensure rasters, parametric point store, and normalization stats exist; return sampler_hash."""
    from movebench.features.points import select_point_store_dir

    plan = study.layer_plan(layers_config)
    try:
        _train_dir, train_m = select_point_store_dir(
            study_dir=study.study_dir,
            split="train",
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=int(NUM_CANDIDATES),
            random_seed=int(SAMPLER_RANDOM_SEED),
            step_multiplier=float(step_multiplier),
            step_q=float(step_q),
            gap_q=float(gap_q),
            bounds_buffer_deg=float(BOUNDS_BUFFER_DEG),
        )
        sampler_hash = str(train_m.sampler_hash)
        # Also require the test split point store (benchmarks evaluate on test).
        select_point_store_dir(
            study_dir=study.study_dir,
            split="test",
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=int(NUM_CANDIDATES),
            random_seed=int(SAMPLER_RANDOM_SEED),
            step_multiplier=float(step_multiplier),
            step_q=float(step_q),
            gap_q=float(gap_q),
            bounds_buffer_deg=float(BOUNDS_BUFFER_DEG),
        )
    except FileNotFoundError:
        try:
            rep = study.extract_point_features(
                config=layers_config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(NUM_CANDIDATES),
                random_seed=int(SAMPLER_RANDOM_SEED),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(step_multiplier),
                step_q=float(step_q),
                gap_q=float(gap_q),
                bounds_buffer_deg=float(BOUNDS_BUFFER_DEG),
            )
        except FileNotFoundError as e:
            if "rasters" not in str(e):
                raise
            study.download_rasters(
                config=layers_config,
                splits=["train", "test"],
                n_workers=int(setup_workers),
                skip_existing=True,
            )
            rep = study.extract_point_features(
                config=layers_config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(NUM_CANDIDATES),
                random_seed=int(SAMPLER_RANDOM_SEED),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(step_multiplier),
                step_q=float(step_q),
                gap_q=float(gap_q),
                bounds_buffer_deg=float(BOUNDS_BUFFER_DEG),
            )
        sampler_hash = str(rep["sampler_hash"])

    study.compute_normalization_vals(
        from_="points",
        config=layers_config,
        sampler_hash=sampler_hash,
        split="train",
        skip_existing=True,
    )
    return sampler_hash


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _bench_loader(loader: Any, *, device: torch.device, batches: int, warmup_batches: int, move_to_device: bool) -> dict[str, Any]:
    it = iter(loader)
    for _ in range(int(warmup_batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if move_to_device:
            _ = tuple(t.to(device, non_blocking=True) for t in batch)

    _sync(device)
    t0 = time.perf_counter()
    total_samples = 0
    for _ in range(int(batches)):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if move_to_device:
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
        total_samples += int(batch[0].shape[0])
    _sync(device)
    t1 = time.perf_counter()

    dt = float(t1 - t0)
    return {
        "batches": int(batches),
        "samples": int(total_samples),
        "seconds": float(dt),
        "samples_per_sec": float(total_samples) / float(dt) if dt > 0 else float("nan"),
        "batches_per_sec": float(batches) / float(dt) if dt > 0 else float("nan"),
        "move_to_device": bool(move_to_device),
    }


def _make_deepssf_optimizer(model: DeepSSF, *, lr_movement: float, lr_habitat: float, weight_decay: float):
    groups = model.parameter_groups()
    return torch.optim.Adam(
        [
            {"params": list(groups["movement"]), "lr": float(lr_movement)},
            {"params": list(groups["habitat"]), "lr": float(lr_habitat)},
        ],
        weight_decay=float(weight_decay),
    )


def _bench_train(
    loader: Any,
    model: DeepSSF,
    *,
    device: torch.device,
    updates: int,
    warmup_updates: int,
    lr_movement: float,
    lr_habitat: float,
    weight_decay: float,
) -> dict[str, Any]:
    model.to(device)
    model.train()
    opt = _make_deepssf_optimizer(
        model, lr_movement=float(lr_movement), lr_habitat=float(lr_habitat), weight_decay=float(weight_decay)
    )

    def _one_update(batch) -> tuple[int, int]:
        x1, x2, x3, y = batch
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        x3 = x3.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model((x1, x2, x3))
        if out.ndim != 4 or int(out.shape[-1]) != 2:
            raise ValueError(f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}")

        scores = out[..., 0] + out[..., 1]
        B, H, W = scores.shape
        logits = scores.view(int(B), int(H) * int(W))
        y_flat = y.view(int(B), int(H) * int(W))

        t_sum = y_flat.sum(dim=1)
        valid = (t_sum > 0.5) & (t_sum < 1.5)
        if not valid.any().item():
            return int(B), 0

        logits = logits[valid]
        true_idx = torch.argmax(y_flat[valid], dim=1)

        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, true_idx, reduction="mean")
        loss.backward()
        opt.step()
        return int(B), int(valid.sum().item())

    it = iter(loader)
    seen_total = 0
    used_total = 0

    warm = 0
    while warm < int(warmup_updates):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        seen, used = _one_update(batch)
        if used == 0:
            continue
        warm += 1
        seen_total += int(seen)
        used_total += int(used)

    _sync(device)
    t0 = time.perf_counter()

    done = 0
    while done < int(updates):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        seen, used = _one_update(batch)
        if used == 0:
            continue
        done += 1
        seen_total += int(seen)
        used_total += int(used)

    _sync(device)
    t1 = time.perf_counter()

    dt = float(t1 - t0)
    return {
        "updates": int(updates),
        "seconds": float(dt),
        "updates_per_sec": float(updates) / float(dt) if dt > 0 else float("nan"),
        "seen_examples": int(seen_total),
        "used_examples": int(used_total),
        "used_per_sec": float(used_total) / float(dt) if dt > 0 else float("nan"),
    }


def _cmd_build(args: argparse.Namespace) -> None:
    studies = _parse_studies(str(args.studies))
    if not studies:
        raise ValueError("No studies resolved.")

    for study_id in studies:
        study = Study(str(study_id))

        sampler_hash = _ensure_ready(
            study=study,
            layers_config=str(args.config),
            setup_workers=int(args.setup_workers),
            step_multiplier=float(args.step_multiplier),
            step_q=float(args.step_q),
            gap_q=float(args.gap_q),
        )

        normalize = bool(args.normalize)
        norm_hash = str(sampler_hash) if normalize else None

        for split in list(args.splits):
            ds_kwargs = {
                "split": str(split),
                "config": str(args.config),
                "output_hw": tuple(args.output_hw),
                "normalize": bool(normalize),
                "normalization_sampler_hash": norm_hash,
                "step_multiplier": float(args.step_multiplier),
                "step_q": float(args.step_q),
                "gap_q": float(args.gap_q),
            }

            out_dir, _m = ensure_deepssf_wds_cache_split(
                study=study,
                split=str(split),
                layers_config=str(args.config),
                ds_kwargs=ds_kwargs,
                samples_per_shard=int(args.samples_per_shard),
                spatial_dtype=str(args.spatial_dtype),  # type: ignore[arg-type]
                target_dtype=str(args.target_dtype),  # type: ignore[arg-type]
                compression=str(args.compression),  # type: ignore[arg-type]
                seed=int(args.seed),
                build_workers=int(args.build_workers),
                build_if_missing=True,
                overwrite=bool(args.overwrite),
            )
            print(f"[{study_id}::{split}] ready: {out_dir}")


def _cmd_bench(args: argparse.Namespace) -> None:
    device = torch.device(str(args.device))
    study = Study(str(args.study))
    plan = study.layer_plan(str(args.config))

    sampler_hash = _ensure_ready(
        study=study,
        layers_config=str(args.config),
        setup_workers=int(args.setup_workers),
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
    )

    normalize = bool(args.normalize)
    norm_hash = str(sampler_hash) if normalize else None

    ds_kwargs = {
        "split": str(args.split),
        "config": str(args.config),
        "output_hw": tuple(args.output_hw),
        "normalize": normalize,
        "normalization_sampler_hash": norm_hash,
        "step_multiplier": float(args.step_multiplier),
        "step_q": float(args.step_q),
        "gap_q": float(args.gap_q),
    }
    cache_dir, manifest = ensure_deepssf_wds_cache_split(
        study=study,
        split=str(args.split),
        layers_config=str(args.config),
        ds_kwargs=ds_kwargs,
        samples_per_shard=int(args.samples_per_shard),
        spatial_dtype=str(args.spatial_dtype),  # type: ignore[arg-type]
        target_dtype=str(args.target_dtype),  # type: ignore[arg-type]
        compression=str(args.compression),  # type: ignore[arg-type]
        seed=int(args.seed),
        build_workers=int(args.build_workers),
        build_if_missing=True,
    )

    baseline_ds = study.deepssf_dataset(
        split=str(args.split),
        config=str(args.config),
        output_hw=tuple(args.output_hw),
        normalize=normalize,
        normalization_sampler_hash=norm_hash,
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
    )
    baseline_loader = DataLoader(
        baseline_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_deepssf,
        drop_last=False,
        pin_memory=bool(args.pin_memory),
    )

    cached_loader = make_deepssf_wds_loader(
        cache_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        shardshuffle=int(args.shardshuffle),
        shuffle_buffer=int(args.shuffle_buffer),
        seed=int(args.seed),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
    )

    # Create a model (same weights for both paths).
    x1, _x2, _x3, _y = next(iter(baseline_loader))
    _b, c_env, h, w = x1.shape
    if int(h) != int(w):
        raise ValueError(f"deepSSF expects square chips. Got H={h}, W={w}.")
    if int(h) != int(manifest["output_hw"][0]) or int(w) != int(manifest["output_hw"][1]):
        raise ValueError(f"Cache output_hw mismatch. cache={manifest['output_hw']} loader={(h,w)}")

    init_model = DeepSSF(int(c_env), image_dim=int(h), pixel_size_m=float(manifest["pixel_size_m"]), device=device)
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    def _clone_model() -> DeepSSF:
        m = DeepSSF(int(c_env), image_dim=int(h), pixel_size_m=float(manifest["pixel_size_m"]), device=device)
        m.load_state_dict({k: v.to(device) for k, v in init_state.items()})
        return m

    cache_bytes = _dir_size_bytes(cache_dir)
    rasters_dir = study.study_dir / "rasters" / str(plan.rasters_hash)
    rasters_bytes = _dir_size_bytes(rasters_dir)

    out: dict[str, Any] = {
        "study_id": str(args.study),
        "split": str(args.split),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "config": str(args.config),
        "config_hash": str(plan.config_hash),
        "rasters_hash": str(plan.rasters_hash),
        "output_hw": [int(h), int(w)],
        "env_channels": int(c_env),
        "cache_dir": str(cache_dir),
        "cache_hash": str(manifest["cache_hash"]),
        "cache_bytes": int(cache_bytes),
        "cache_bytes_per_sample": float(cache_bytes) / float(manifest["num_samples"]) if manifest["num_samples"] > 0 else None,
        "rasters_dir": str(rasters_dir),
        "rasters_bytes": int(rasters_bytes),
        "cache_vs_rasters_ratio": float(cache_bytes) / float(rasters_bytes) if rasters_bytes > 0 else None,
        "bench": {},
    }

    out["bench"]["baseline_loader_cpu"] = _bench_loader(
        baseline_loader,
        device=device,
        batches=int(args.data_batches),
        warmup_batches=int(args.data_warmup),
        move_to_device=False,
    )
    out["bench"]["baseline_loader_to_device"] = _bench_loader(
        baseline_loader,
        device=device,
        batches=int(args.data_batches),
        warmup_batches=int(args.data_warmup),
        move_to_device=True,
    )
    out["bench"]["cached_loader_cpu"] = _bench_loader(
        cached_loader,
        device=device,
        batches=int(args.data_batches),
        warmup_batches=int(args.data_warmup),
        move_to_device=False,
    )
    out["bench"]["cached_loader_to_device"] = _bench_loader(
        cached_loader,
        device=device,
        batches=int(args.data_batches),
        warmup_batches=int(args.data_warmup),
        move_to_device=True,
    )

    out["bench"]["baseline_train"] = _bench_train(
        baseline_loader,
        _clone_model(),
        device=device,
        updates=int(args.train_updates),
        warmup_updates=int(args.train_warmup),
        lr_movement=float(args.lr_movement),
        lr_habitat=float(args.lr_habitat),
        weight_decay=float(args.weight_decay),
    )
    out["bench"]["cached_train"] = _bench_train(
        cached_loader,
        _clone_model(),
        device=device,
        updates=int(args.train_updates),
        warmup_updates=int(args.train_warmup),
        lr_movement=float(args.lr_movement),
        lr_habitat=float(args.lr_habitat),
        weight_decay=float(args.weight_decay),
    )

    def _speedup(a: float, b: float) -> float | None:
        if not (math.isfinite(a) and math.isfinite(b) and a > 0):
            return None
        return float(b) / float(a)

    out["bench"]["speedup"] = {
        "loader_cpu_samples_per_sec": _speedup(
            float(out["bench"]["baseline_loader_cpu"]["samples_per_sec"]),
            float(out["bench"]["cached_loader_cpu"]["samples_per_sec"]),
        ),
        "loader_to_device_samples_per_sec": _speedup(
            float(out["bench"]["baseline_loader_to_device"]["samples_per_sec"]),
            float(out["bench"]["cached_loader_to_device"]["samples_per_sec"]),
        ),
        "train_updates_per_sec": _speedup(
            float(out["bench"]["baseline_train"]["updates_per_sec"]),
            float(out["bench"]["cached_train"]["updates_per_sec"]),
        ),
        "train_used_examples_per_sec": _speedup(
            float(out["bench"]["baseline_train"]["used_per_sec"]),
            float(out["bench"]["cached_train"]["used_per_sec"]),
        ),
    }

    print(json.dumps(out, indent=2, sort_keys=True))


def _add_common_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default="configs/layers_all.yaml", help="Layer config YAML.")
    p.add_argument("--output-hw", type=_parse_hw, default=(101, 101), help="Chip (H,W), e.g. 101,101.")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", action="store_false", dest="normalize")
    p.add_argument("--step-multiplier", type=float, default=2.0)
    p.add_argument("--step-q", type=float, default=0.95)
    p.add_argument("--gap-q", type=float, default=0.999)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/benchmark WebDataset caches for deepSSF.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build deepSSF WebDataset caches.")
    b.add_argument("--studies", type=str, default="all", help="Comma-separated study ids, or 'all' (default).")
    b.add_argument("--splits", type=_parse_splits, default=["train", "test"], help="Comma list, e.g. train,test")
    _add_common_dataset_args(b)
    b.add_argument("--setup-workers", type=int, default=16, help="Workers for raster download / point extraction.")
    b.add_argument("--samples-per-shard", type=int, default=256)
    b.add_argument("--spatial-dtype", type=str, default="float16", choices=("float16", "float32"))
    b.add_argument("--target-dtype", type=str, default="uint8", choices=("uint8", "float16", "float32"))
    b.add_argument("--compression", type=str, default="none", choices=("none", "gz"))
    b.add_argument("--build-workers", type=int, default=0, help="DataLoader workers used while building the cache.")
    b.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=_cmd_build)

    q = sub.add_parser("bench", help="Benchmark baseline vs cached throughput.")
    q.add_argument("--study", type=str, required=True)
    q.add_argument("--split", type=str, default="train")
    _add_common_dataset_args(q)
    q.add_argument("--setup-workers", type=int, default=16)
    q.add_argument("--samples-per-shard", type=int, default=256)
    q.add_argument("--spatial-dtype", type=str, default="float16", choices=("float16", "float32"))
    q.add_argument("--target-dtype", type=str, default="uint8", choices=("uint8", "float16", "float32"))
    q.add_argument("--compression", type=str, default="none", choices=("none", "gz"))
    q.add_argument("--build-workers", type=int, default=0)
    q.add_argument("--device", type=str, default="cpu")
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--num-workers", type=int, default=0)
    q.add_argument("--pin-memory", action="store_true")
    q.add_argument("--prefetch-factor", type=int, default=None)
    q.add_argument("--persistent-workers", action="store_true")
    q.add_argument("--shardshuffle", type=int, default=100)
    q.add_argument("--shuffle-buffer", type=int, default=0)
    q.add_argument("--data-batches", type=int, default=200)
    q.add_argument("--data-warmup", type=int, default=10)
    q.add_argument("--train-updates", type=int, default=100)
    q.add_argument("--train-warmup", type=int, default=10)
    q.add_argument("--lr-movement", type=float, default=1e-4)
    q.add_argument("--lr-habitat", type=float, default=1e-4)
    q.add_argument("--weight-decay", type=float, default=0.0)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(func=_cmd_bench)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
