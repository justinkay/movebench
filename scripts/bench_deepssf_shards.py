#!/usr/bin/env python
"""Benchmark deepSSF training throughput with/without a tar-sharded cache.

This is intended to quantify:
  - additional storage required for a cached dataset representation, and
  - training/dataloader speedup vs on-the-fly raster assembly (ChipDataset).
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
from movebench.datasets.deepssf_shards import (
    DeepSSFShardDataset,
    build_deepssf_shards,
    deepssf_cache_hash,
    read_deepssf_manifest,
)
from movebench.models.deepssf import DeepSSF
from movebench.train import collate_deepssf


def _parse_hw(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("output_hw must be like '101,101'")
    return int(parts[0]), int(parts[1])


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
                # Racy if something deletes while walking.
                continue
    return int(total)


def _fmt_bytes(n: int) -> str:
    n = int(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} TB"


def _make_baseline_loader(
    ds,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
    shuffle: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_deepssf,
        "drop_last": False,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(ds, **kwargs)


def _make_cached_loader(
    cache_dir: Path,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
    shuffle_buffer: int,
    seed: int,
) -> DataLoader:
    ds = DeepSSFShardDataset(
        cache_dir=str(cache_dir),
        shuffle_shards=True,
        shuffle_buffer=int(shuffle_buffer),
        seed=int(seed),
    )
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "drop_last": False,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(ds, **kwargs)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _bench_loader(
    loader: DataLoader,
    *,
    device: torch.device,
    batches: int,
    warmup_batches: int,
    move_to_device: bool,
) -> dict[str, Any]:
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
    loader: DataLoader,
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


def _default_cache_dir(
    *,
    study: Study,
    split: str,
    plan_config_hash: str,
    cache_hash: str,
) -> Path:
    return study.study_dir / "cache" / "deepssf_shards" / str(split) / str(plan_config_hash) / str(cache_hash)


def _add_common_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--study", type=str, required=True, help="Study id.")
    p.add_argument("--split", type=str, default="train", help="Dataset split (train/test).")
    p.add_argument("--config", type=str, default="configs/layers_all.yaml", help="Layer config YAML.")
    p.add_argument("--output-hw", type=_parse_hw, default=(101, 101), help="Chip (H,W), e.g. 101,101.")
    p.add_argument("--normalize", action="store_true", default=True, help="Normalize env channels (default).")
    p.add_argument(
        "--no-normalize", action="store_false", dest="normalize", help="Disable env normalization."
    )
    p.add_argument(
        "--normalization-sampler-hash",
        type=str,
        default=None,
        help="Sampler hash for normalization stats (omit if unambiguous).",
    )
    p.add_argument("--step-multiplier", type=float, default=2.0, help="Step filtering multiplier.")
    p.add_argument("--step-q", type=float, default=0.95, help="Step length quantile filter.")
    p.add_argument("--gap-q", type=float, default=0.999, help="Gap quantile filter.")


def _cmd_build(args: argparse.Namespace) -> None:
    study = Study(str(args.study))
    plan = study.layer_plan(str(args.config))

    cache_hash = deepssf_cache_hash(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        split=str(args.split),
        output_hw=tuple(args.output_hw),
        normalize=bool(args.normalize),
        normalization_sampler_hash=args.normalization_sampler_hash,
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
        samples_per_shard=int(args.samples_per_shard),
        spatial_dtype=str(args.spatial_dtype),
        target_dtype=str(args.target_dtype),
        compression=str(args.compression),
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir(
        study=study, split=str(args.split), plan_config_hash=str(plan.config_hash), cache_hash=str(cache_hash)
    )

    ds = study.deepssf_dataset(
        split=str(args.split),
        config=str(args.config),
        output_hw=tuple(args.output_hw),
        normalize=bool(args.normalize),
        normalization_sampler_hash=args.normalization_sampler_hash,
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
    )

    manifest = build_deepssf_shards(
        dataset=ds,
        out_dir=str(cache_dir),
        config_path=str(args.config),
        normalization_sampler_hash=args.normalization_sampler_hash,
        samples_per_shard=int(args.samples_per_shard),
        spatial_dtype=str(args.spatial_dtype),
        target_dtype=str(args.target_dtype),
        compression=str(args.compression),
        num_workers=int(args.num_workers),
        overwrite=bool(args.overwrite),
        progress=not bool(args.no_progress),
    )
    print(f"Wrote cache: {cache_dir}")
    print(f"Manifest: {cache_dir / 'manifest.json'}")
    print(f"cache_hash={manifest.cache_hash}")


def _cmd_bench(args: argparse.Namespace) -> None:
    device = torch.device(str(args.device))

    study = Study(str(args.study))
    plan = study.layer_plan(str(args.config))

    cache_hash = deepssf_cache_hash(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        split=str(args.split),
        output_hw=tuple(args.output_hw),
        normalize=bool(args.normalize),
        normalization_sampler_hash=args.normalization_sampler_hash,
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
        samples_per_shard=int(args.samples_per_shard),
        spatial_dtype=str(args.spatial_dtype),
        target_dtype=str(args.target_dtype),
        compression=str(args.compression),
    )
    cache_dir = Path(args.cache_dir) if args.cache_dir else _default_cache_dir(
        study=study, split=str(args.split), plan_config_hash=str(plan.config_hash), cache_hash=str(cache_hash)
    )

    if not (cache_dir / "manifest.json").exists():
        raise FileNotFoundError(
            f"Cache manifest not found: {cache_dir / 'manifest.json'}. "
            "Run the 'build' subcommand first or pass --cache-dir."
        )
    manifest = read_deepssf_manifest(cache_dir / "manifest.json")

    ds = study.deepssf_dataset(
        split=str(args.split),
        config=str(args.config),
        output_hw=tuple(args.output_hw),
        normalize=bool(args.normalize),
        normalization_sampler_hash=args.normalization_sampler_hash,
        step_multiplier=float(args.step_multiplier),
        step_q=float(args.step_q),
        gap_q=float(args.gap_q),
    )

    baseline_loader = _make_baseline_loader(
        ds,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        shuffle=bool(args.shuffle),
    )
    cached_loader = _make_cached_loader(
        cache_dir,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        prefetch_factor=args.prefetch_factor,
        persistent_workers=bool(args.persistent_workers),
        shuffle_buffer=int(args.shuffle_buffer),
        seed=int(args.seed),
    )

    # Determine model dimensions.
    x1, x2, x3, y = next(iter(baseline_loader))
    B, C, H, W = x1.shape
    if int(H) != int(W):
        raise ValueError(f"deepSSF expects square chips. Got H={H}, W={W}.")
    if int(H) != int(args.output_hw[0]) or int(W) != int(args.output_hw[1]):
        raise ValueError(f"Dataset output_hw mismatch. Expected {args.output_hw}, got {(H, W)}.")
    if not math.isclose(float(ds.chips.pixel_size_m), float(manifest.pixel_size_m), rel_tol=1e-6, abs_tol=0.0):
        raise ValueError(
            "Cache pixel_size_m does not match on-the-fly dataset. "
            f"cache={manifest.pixel_size_m} dataset={ds.chips.pixel_size_m}"
        )

    init_model = DeepSSF(
        int(C),
        image_dim=int(H),
        pixel_size_m=float(ds.chips.pixel_size_m),
        device=device,
    )
    init_state = {k: v.detach().cpu().clone() for k, v in init_model.state_dict().items()}

    def _clone_model() -> DeepSSF:
        m = DeepSSF(int(C), image_dim=int(H), pixel_size_m=float(ds.chips.pixel_size_m), device=device)
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
        "output_hw": [int(H), int(W)],
        "env_channels": int(C),
        "cache_dir": str(cache_dir),
        "cache_hash": str(manifest.cache_hash),
        "cache_bytes": int(cache_bytes),
        "rasters_dir": str(rasters_dir),
        "rasters_bytes": int(rasters_bytes),
        "cache_vs_rasters_ratio": float(cache_bytes) / float(rasters_bytes) if rasters_bytes > 0 else None,
        "cache_bytes_per_sample": float(cache_bytes) / float(manifest.num_samples) if manifest.num_samples > 0 else None,
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
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"Wrote JSON: {out_path}")

    bps = out.get("cache_bytes_per_sample")
    if isinstance(bps, (int, float)) and math.isfinite(float(bps)):
        bps_str = f"{float(bps):.1f} bytes/sample"
    else:
        bps_str = "n/a"

    su = out["bench"]["speedup"].get("train_updates_per_sec")
    if isinstance(su, (int, float)) and math.isfinite(float(su)):
        su_str = f"x{float(su):.2f}"
    else:
        su_str = "n/a"

    print(
        "\nSummary\n"
        f"- Cache size:  {_fmt_bytes(cache_bytes)} ({bps_str})\n"
        f"- Rasters:     {_fmt_bytes(rasters_bytes)}\n"
        f"- Train speed: {out['bench']['baseline_train']['updates_per_sec']:.2f} -> "
        f"{out['bench']['cached_train']['updates_per_sec']:.2f} updates/sec "
        f"({su_str})\n"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark deepSSF shard cache vs runtime raster assembly.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a deepSSF tar-shard cache.")
    _add_common_dataset_args(b)
    b.add_argument("--cache-dir", type=str, default=None, help="Cache output dir (default: under study_dir/cache/...).")
    b.add_argument("--samples-per-shard", type=int, default=256, help="Samples per tar shard.")
    b.add_argument("--spatial-dtype", type=str, default="float16", choices=("float16", "float32"))
    b.add_argument("--target-dtype", type=str, default="uint8", choices=("uint8", "float16", "float32"))
    b.add_argument("--compression", type=str, default="none", choices=("none", "gz"))
    b.add_argument("--num-workers", type=int, default=0, help="Workers to build the cache (DataLoader workers).")
    b.add_argument("--overwrite", action="store_true", help="Overwrite cache_dir if it exists.")
    b.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    b.set_defaults(func=_cmd_build)

    q = sub.add_parser("bench", help="Benchmark baseline vs cached dataloading/training.")
    _add_common_dataset_args(q)
    q.add_argument("--cache-dir", type=str, default=None, help="Cache dir (default: derived from args).")
    q.add_argument("--samples-per-shard", type=int, default=256, help="Used only for deriving default cache dir.")
    q.add_argument("--spatial-dtype", type=str, default="float16", choices=("float16", "float32"))
    q.add_argument("--target-dtype", type=str, default="uint8", choices=("uint8", "float16", "float32"))
    q.add_argument("--compression", type=str, default="none", choices=("none", "gz"))

    q.add_argument("--device", type=str, default="cpu", help="cpu or cuda (or cuda:0, etc).")
    q.add_argument("--batch-size", type=int, default=8)
    q.add_argument("--num-workers", type=int, default=0)
    q.add_argument("--pin-memory", action="store_true", help="Enable pin_memory in DataLoader.")
    q.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (num_workers>0 only).")
    q.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive (num_workers>0 only).")
    q.add_argument("--no-shuffle", action="store_false", dest="shuffle", help="Disable baseline DataLoader shuffle.")
    q.add_argument("--shuffle-buffer", type=int, default=0, help="Cached dataset shuffle buffer (0 disables).")
    q.add_argument("--seed", type=int, default=0, help="Seed for cached shuffle.")

    q.add_argument("--data-batches", type=int, default=200, help="Batches to time for loader-only benchmark.")
    q.add_argument("--data-warmup", type=int, default=10, help="Warmup batches (not timed).")

    q.add_argument("--train-updates", type=int, default=100, help="Optimizer updates to time.")
    q.add_argument("--train-warmup", type=int, default=10, help="Warmup updates (not timed).")
    q.add_argument("--lr-movement", type=float, default=1e-4)
    q.add_argument("--lr-habitat", type=float, default=1e-4)
    q.add_argument("--weight-decay", type=float, default=0.0)

    q.add_argument("--json-out", type=str, default=None, help="Optional path to write results JSON.")
    q.set_defaults(func=_cmd_bench)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
