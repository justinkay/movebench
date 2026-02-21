#!/usr/bin/env python
"""Launch and run deepSSF WebDataset cache builds on Slurm.

Single-entry-point script with two modes:
  - Launcher mode (default): discovers studies needing WebDataset cache shards
    and launches one `srun` job per study with throttling.
  - Worker mode (`--study STUDY_ID`): builds the cache for one study.

This script NEVER downloads rasters and NEVER runs point extraction. It expects:
  - rasters cache to already exist for studies that use GEE layers, and
  - parametric point-store train + points normalization stats to exist (to get
    normalization_sampler_hash).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _SamplerParams:
    num_candidates: int
    random_seed: int
    step_multiplier: float
    step_q: float
    gap_q: float
    bounds_buffer_deg: float


@dataclass(frozen=True)
class _CacheParams:
    samples_per_shard: int
    spatial_dtype: str
    target_dtype: str
    compression: str
    seed: int
    build_workers: int
    overwrite: bool
    resume: bool


def _parse_studies_override(val: str | None) -> tuple[str, list[str] | None]:
    if val is None:
        return "config_default", None
    v = str(val).strip()
    if not v:
        return "config_default", None
    if v.lower() == "all":
        return "all", None
    out = [p.strip() for p in v.split(",") if p.strip()]
    return "explicit", out


def _parse_splits(val: str) -> list[str]:
    out = [p.strip() for p in str(val).split(",") if p.strip()]
    if not out:
        raise ValueError("--splits must be a comma-separated list like 'train,test'")
    bad = [s for s in out if s not in {"train", "test"}]
    if bad:
        raise ValueError(f"Unknown split(s) {bad}. Expected 'train' and/or 'test'.")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = _REPO_ROOT / "configs" / "benchmarks" / "all.yaml"

    p = argparse.ArgumentParser(description="Launch Slurm jobs to build deepSSF WebDataset caches per study.")
    p.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Benchmark YAML config path (must include datasets.deepssf and datasets.issf.sampler).",
    )
    p.add_argument("--study", type=str, default=None, help="Run worker mode for a single study id.")
    p.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Launcher override: 'all' or comma-separated study ids. If omitted, use config's `studies` or all available.",
    )
    p.add_argument("--splits", type=_parse_splits, default="train,test", help="Comma-separated splits: train,test")

    p.add_argument("--max-concurrent-jobs", type=int, default=32, help="Maximum concurrent `srun` jobs.")
    p.add_argument("--polling-interval", type=int, default=30, help="Seconds between job status checks.")
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for Slurm stdout/stderr logs (default: results/slurm/deepssf_wds/<config_stem>/).",
    )

    # Slurm defaults match scripts/slurm/extract.py except GPUs default to 0.
    p.add_argument("--partition", type=str, default="vision-beery", help="Slurm partition (-p).")
    p.add_argument("--qos", type=str, default="vision-beery-main", help="Slurm QoS (-q).")
    p.add_argument("--account", type=str, default="vision-beery", help="Slurm account (-A).")
    p.add_argument("--time", type=str, default="7-00:00:00", help="Slurm time limit (-t), e.g. '7-00:00:00'.")
    p.add_argument("--mem", type=str, default="64GB", help="Slurm memory request, e.g. '64GB'.")
    p.add_argument("--cpus-per-task", type=int, default=16, help="Slurm CPUs per task (--cpus-per-task).")
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=0,
        help="Slurm GPUs per node (--gpus-per-node). Only passed if > 0.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions/commands but do not submit jobs.")

    # Cache build params (defaults filled after config load).
    p.add_argument("--samples-per-shard", type=int, default=None)
    p.add_argument("--spatial-dtype", type=str, default=None, choices=("float16", "float32"))
    p.add_argument("--target-dtype", type=str, default=None, choices=("uint8", "float16", "float32"))
    p.add_argument("--compression", type=str, default=None, choices=("none", "gz"))
    p.add_argument("--build-workers", type=int, default=None, help="DataLoader workers used while building the cache.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    p.add_argument("--resume", action="store_true", help="Resume partially-built caches (missing manifest).")

    return p.parse_args(argv)


def _load_cfg(config_path: str) -> dict[str, Any]:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.benchmark import load_config

    return load_config(config_path)


def _resolve_layer_plan(layers_config: str):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.layers.config import load_layers_config
    from movebench.layers.registry import REGISTRY

    cfg = load_layers_config(layers_config)
    return REGISTRY.resolve(cfg)


def _issf_sampler_params(cfg: dict[str, Any]) -> _SamplerParams:
    sampler = cfg["datasets"]["issf"]["sampler"]
    return _SamplerParams(
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )


def _uses_gee_layers(plan) -> bool:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.layers.registry import REGISTRY

    required = set(str(x) for x in plan.layers_in_order)
    for parents in (plan.derived_dependencies or {}).values():
        required.update(str(p) for p in parents)

    for layer in sorted(required):
        src = REGISTRY.get(str(layer)).source
        if src in {"gee_chip", "gee_tile"}:
            return True
    return False


def _rasters_root_for(plan, *, study_dir: Path) -> Path:
    return Path(study_dir) / "rasters" / str(plan.rasters_hash)


def _select_parametric_train_store(*, study_dir: Path, plan, sampler: _SamplerParams):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.features.points import select_point_store_dir

    return select_point_store_dir(
        study_dir=Path(study_dir),
        split="train",
        config_hash=str(plan.config_hash),
        expected_type="parametric",
        num_candidates=int(sampler.num_candidates),
        random_seed=int(sampler.random_seed),
        step_multiplier=float(sampler.step_multiplier),
        step_q=float(sampler.step_q),
        gap_q=float(sampler.gap_q),
        bounds_buffer_deg=float(sampler.bounds_buffer_deg),
    )


def _norm_stats_path(*, study_dir: Path, config_hash: str, sampler_hash: str) -> Path:
    return Path(study_dir) / "normalization" / "points" / str(config_hash) / str(sampler_hash) / "stats.json"


def _resolve_cache_params(args: argparse.Namespace, cfg: dict[str, Any]) -> _CacheParams:
    dcfg = cfg.get("datasets", {}).get("deepssf", {})
    if not isinstance(dcfg, dict):
        raise ValueError("Benchmark config datasets.deepssf must be a mapping/dict.")
    cache_cfg = dcfg.get("cache", {}) if isinstance(dcfg.get("cache", {}), dict) else {}

    def pick_int(name: str, default: int) -> int:
        v = getattr(args, name)
        if v is not None:
            return int(v)
        if name in cache_cfg and cache_cfg[name] is not None:
            return int(cache_cfg[name])
        return int(default)

    def pick_str(name: str, default: str) -> str:
        v = getattr(args, name)
        if v is not None:
            return str(v)
        if name in cache_cfg and cache_cfg[name] is not None:
            return str(cache_cfg[name])
        return str(default)

    runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
    runtime_seed = int(runtime.get("seed", 0))

    return _CacheParams(
        samples_per_shard=pick_int("samples_per_shard", 256),
        spatial_dtype=pick_str("spatial_dtype", "float16"),
        target_dtype=pick_str("target_dtype", "uint8"),
        compression=pick_str("compression", "none"),
        seed=pick_int("seed", runtime_seed),
        build_workers=pick_int("build_workers", 0),
        overwrite=bool(getattr(args, "overwrite", False)),
        resume=bool(getattr(args, "resume", False)),
    )


def _deepssf_ds_kwargs(cfg: dict[str, Any], *, sampler_hash: str) -> dict[str, Any]:
    dcfg = cfg.get("datasets", {}).get("deepssf", None)
    if not isinstance(dcfg, dict):
        raise ValueError("Benchmark config missing required key: datasets.deepssf")

    ds_kwargs: dict[str, Any] = {
        "normalization_sampler_hash": str(sampler_hash),
    }
    for k in ("output_hw", "normalize", "step_multiplier", "step_q", "gap_q"):
        if k not in dcfg:
            continue
        v = dcfg[k]
        if k == "output_hw":
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError("datasets.deepssf.output_hw must be a 2-item sequence like [101,101].")
            v = (int(v[0]), int(v[1]))
        ds_kwargs[k] = v
    return ds_kwargs


def _expected_cache_manifest(
    *,
    study_dir: Path,
    plan,
    split: str,
    ds_kwargs: dict[str, Any],
    cache_params: _CacheParams,
) -> Path:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.datasets.deepssf_wds_cache import deepssf_wds_cache_hash

    output_hw = ds_kwargs.get("output_hw", (101, 101))
    if not (isinstance(output_hw, (list, tuple)) and len(output_hw) == 2):
        raise ValueError("datasets.deepssf.output_hw must be a 2-item sequence like [101,101].")
    output_hw = (int(output_hw[0]), int(output_hw[1]))

    normalize = bool(ds_kwargs.get("normalize", True))
    norm_hash_raw = ds_kwargs.get("normalization_sampler_hash", None)
    norm_hash = str(norm_hash_raw) if (normalize and norm_hash_raw is not None) else None

    step_multiplier = float(ds_kwargs.get("step_multiplier", 2.0))
    step_q = float(ds_kwargs.get("step_q", 0.95))
    gap_q = float(ds_kwargs.get("gap_q", 0.999))

    cache_hash = deepssf_wds_cache_hash(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        split=str(split),
        output_hw=output_hw,
        normalize=bool(normalize),
        normalization_sampler_hash=norm_hash,
        step_multiplier=float(step_multiplier),
        step_q=float(step_q),
        gap_q=float(gap_q),
        samples_per_shard=int(cache_params.samples_per_shard),
        spatial_dtype=str(cache_params.spatial_dtype),  # type: ignore[arg-type]
        target_dtype=str(cache_params.target_dtype),  # type: ignore[arg-type]
        compression=str(cache_params.compression),  # type: ignore[arg-type]
    )

    cache_dir = Path(study_dir) / "cache" / "deepssf_wds" / str(split) / str(plan.config_hash) / str(cache_hash)
    return cache_dir / "manifest.json"


def _check_prereqs(
    *,
    study_id: str,
    cfg: dict[str, Any],
    plan,
    issf_sampler: _SamplerParams,
) -> tuple[bool, str | None, str]:
    """Return (ok, sampler_hash, reason)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import Study, paths

    try:
        study = Study(str(study_id))
    except FileNotFoundError as e:
        return False, None, f"missing_study: {e}"

    if _uses_gee_layers(plan):
        rasters_root = _rasters_root_for(plan, study_dir=paths.study_dir(str(study_id)))
        if not rasters_root.exists():
            return False, None, f"missing_rasters_root: {rasters_root}"

    try:
        _train_dir, train_m = _select_parametric_train_store(study_dir=study.study_dir, plan=plan, sampler=issf_sampler)
    except (FileNotFoundError, ValueError) as e:
        return False, None, f"missing_parametric_point_store_train: {e}"

    sampler_hash = str(train_m.sampler_hash)
    stats_path = _norm_stats_path(study_dir=study.study_dir, config_hash=str(plan.config_hash), sampler_hash=sampler_hash)
    if not stats_path.exists():
        return False, None, f"missing_points_normalization_stats: {stats_path}"

    return True, sampler_hash, "ok"


def _build_srun_cmd(
    *,
    args: argparse.Namespace,
    study_id: str,
    config_path: str,
    splits: list[str],
    log_dir: Path,
    cache_params: _CacheParams,
) -> list[str]:
    script_path = Path(__file__).resolve()

    out_base = str(study_id).replace(os.sep, "_")
    out_path = log_dir / f"{out_base}.%j.out"
    err_path = log_dir / f"{out_base}.%j.err"

    cmd: list[str] = ["srun", "--export=ALL"]
    if args.partition:
        cmd.extend(["-p", str(args.partition)])
    if args.qos:
        cmd.extend(["-q", str(args.qos)])
    if args.account:
        cmd.extend(["-A", str(args.account)])
    if args.time:
        cmd.extend(["-t", str(args.time)])
    if args.mem:
        cmd.append(f"--mem={args.mem}")
    cmd.extend(["--cpus-per-task", str(int(args.cpus_per_task))])
    if int(getattr(args, "gpus_per_node", 0)) > 0:
        cmd.extend(["--gpus-per-node", str(int(args.gpus_per_node))])
    cmd.extend(["--output", str(out_path), "--error", str(err_path)])

    cmd.extend(
        [
            sys.executable,
            "-u",
            str(script_path),
            "--config",
            str(config_path),
            "--study",
            str(study_id),
            "--splits",
            ",".join(splits),
        ]
    )

    # Pass cache params explicitly so workers match launcher skip logic.
    cmd.extend(
        [
            "--samples-per-shard",
            str(int(cache_params.samples_per_shard)),
            "--spatial-dtype",
            str(cache_params.spatial_dtype),
            "--target-dtype",
            str(cache_params.target_dtype),
            "--compression",
            str(cache_params.compression),
            "--build-workers",
            str(int(cache_params.build_workers)),
            "--seed",
            str(int(cache_params.seed)),
        ]
    )
    if cache_params.overwrite:
        cmd.append("--overwrite")
    if cache_params.resume:
        cmd.append("--resume")
    return cmd


def _worker(args: argparse.Namespace) -> int:
    cfg = _load_cfg(str(args.config))
    layers_config = str(cfg["layers_config"])
    plan = _resolve_layer_plan(layers_config)
    issf_sampler = _issf_sampler_params(cfg)
    cache_params = _resolve_cache_params(args, cfg)

    assert args.study is not None
    ok, sampler_hash, reason = _check_prereqs(
        study_id=str(args.study),
        cfg=cfg,
        plan=plan,
        issf_sampler=issf_sampler,
    )
    if not ok or sampler_hash is None:
        print(
            f"[wds] ERROR: prerequisites not met for study={args.study!r}: {reason}\n"
            f"[wds] Hint: run /Users/justinkay/movebench/scripts/slurm/extract.py first (point stores + normalization).",
            file=sys.stderr,
        )
        return 2

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import Study
    from movebench.datasets.deepssf_wds_cache import ensure_deepssf_wds_cache_split

    ds_kwargs = _deepssf_ds_kwargs(cfg, sampler_hash=str(sampler_hash))
    study = Study(str(args.study))

    cache_dirs: list[str] = []
    for split in list(args.splits):
        cache_dir, _m = ensure_deepssf_wds_cache_split(
            study=study,
            split=str(split),
            layers_config=str(layers_config),
            ds_kwargs=dict(ds_kwargs),
            samples_per_shard=int(cache_params.samples_per_shard),
            spatial_dtype=str(cache_params.spatial_dtype),  # type: ignore[arg-type]
            target_dtype=str(cache_params.target_dtype),  # type: ignore[arg-type]
            compression=str(cache_params.compression),  # type: ignore[arg-type]
            seed=int(cache_params.seed),
            build_workers=int(cache_params.build_workers),
            build_if_missing=True,
            overwrite=bool(cache_params.overwrite),
            resume=bool(cache_params.resume),
        )
        cache_dirs.append(str(cache_dir))

    print(
        f"[wds] OK study_id={args.study} config_hash={plan.config_hash} "
        f"sampler_hash={sampler_hash} splits={args.splits} cache_dirs={cache_dirs}",
        flush=True,
    )
    return 0


def _launcher(args: argparse.Namespace) -> int:
    cfg = _load_cfg(str(args.config))
    layers_config = str(cfg["layers_config"])
    plan = _resolve_layer_plan(layers_config)
    issf_sampler = _issf_sampler_params(cfg)
    cache_params = _resolve_cache_params(args, cfg)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import paths
    from movebench.benchmark import resolve_studies

    override_kind, override_list = _parse_studies_override(args.studies)
    if override_kind == "config_default":
        studies = resolve_studies(cfg)
    elif override_kind == "all":
        cfg2 = dict(cfg)
        cfg2["studies"] = "all"
        studies = resolve_studies(cfg2)
    else:
        assert override_list is not None
        studies = sorted(dict.fromkeys([str(s) for s in override_list]))

    config_stem = Path(str(args.config)).stem
    log_dir = Path(args.log_dir) if args.log_dir else (_REPO_ROOT / "results" / "slurm" / "deepssf_wds" / config_stem)
    if not log_dir.is_absolute():
        log_dir = (_REPO_ROOT / log_dir).resolve()

    blocked: list[tuple[str, str]] = []
    skipped: list[str] = []
    queue: list[str] = []

    for study_id in studies:
        ok, sampler_hash, reason = _check_prereqs(study_id=str(study_id), cfg=cfg, plan=plan, issf_sampler=issf_sampler)
        if not ok or sampler_hash is None:
            blocked.append((str(study_id), reason))
            continue

        ds_kwargs = _deepssf_ds_kwargs(cfg, sampler_hash=str(sampler_hash))
        need = False
        if cache_params.overwrite:
            need = True
        else:
            for split in list(args.splits):
                mpth = _expected_cache_manifest(
                    study_dir=paths.study_dir(str(study_id)),
                    plan=plan,
                    split=split,
                    ds_kwargs=ds_kwargs,
                    cache_params=cache_params,
                )
                if not mpth.exists():
                    need = True
                    break

        if need:
            queue.append(str(study_id))
        else:
            skipped.append(str(study_id))

    print(
        f"[wds] studies total={len(studies)} queued={len(queue)} skipped={len(skipped)} blocked={len(blocked)}",
        flush=True,
    )
    if blocked:
        for sid, reason in blocked:
            print(f"[wds] BLOCKED study={sid} reason={reason}", file=sys.stderr)

    if args.dry_run:
        if queue:
            print(f"[wds] Dry-run: would launch {len(queue)} jobs (max_concurrent={args.max_concurrent_jobs}).", flush=True)
            for sid in queue[:10]:
                cmd = _build_srun_cmd(
                    args=args,
                    study_id=sid,
                    config_path=str(args.config),
                    splits=list(args.splits),
                    log_dir=log_dir,
                    cache_params=cache_params,
                )
                print("[wds] ", shlex.join(cmd), flush=True)
            if len(queue) > 10:
                print(f"[wds] ... ({len(queue)-10} more)", flush=True)
        return 2 if blocked else 0

    if not queue:
        return 2 if blocked else 0

    if int(args.max_concurrent_jobs) <= 0:
        raise ValueError("--max-concurrent-jobs must be > 0")
    if int(args.polling_interval) <= 0:
        raise ValueError("--polling-interval must be > 0")

    log_dir.mkdir(parents=True, exist_ok=True)

    running: dict[str, subprocess.Popen] = {}
    failures: list[str] = []
    completed = 0
    total = len(queue)
    queue_idx = 0

    def _launch_one(study_id: str) -> None:
        cmd = _build_srun_cmd(
            args=args,
            study_id=study_id,
            config_path=str(args.config),
            splits=list(args.splits),
            log_dir=log_dir,
            cache_params=cache_params,
        )
        print(f"[wds] Launching study={study_id}: {shlex.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
        running[study_id] = proc

    try:
        while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
            sid = queue[queue_idx]
            queue_idx += 1
            _launch_one(sid)

        while running or queue_idx < len(queue):
            time.sleep(int(args.polling_interval))

            for sid, proc in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                running.pop(sid, None)
                completed += 1
                if int(rc) != 0:
                    failures.append(sid)
                    print(f"[wds] FAILED study={sid} exit_code={rc}", file=sys.stderr, flush=True)
                else:
                    print(f"[wds] DONE study={sid}", flush=True)

            while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
                sid = queue[queue_idx]
                queue_idx += 1
                _launch_one(sid)

            pending = total - (completed + len(running))
            print(
                f"[wds] Progress completed={completed}/{total} running={len(running)} pending={pending} failed={len(failures)}",
                flush=True,
            )
            if running:
                print(f"[wds] Running: {sorted(running.keys())}", flush=True)
    except KeyboardInterrupt:
        print("[wds] Interrupted. Not launching new jobs.", file=sys.stderr, flush=True)
        if running:
            print(f"[wds] Still running: {sorted(running.keys())}", file=sys.stderr, flush=True)
        return 130

    if failures or blocked:
        print(f"[wds] Finished with failures={len(failures)} blocked={len(blocked)}", file=sys.stderr, flush=True)
        if failures:
            print(f"[wds] Failed studies: {sorted(failures)}", file=sys.stderr, flush=True)
        return 2

    print("[wds] All jobs completed successfully.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    if args.study:
        return _worker(args)
    return _launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
