#!/usr/bin/env python
"""Launch and run MoveBench point feature extraction jobs on Slurm.

Single-entry-point script with two modes:
  - Launcher mode (default): discovers studies needing extraction + normalization,
    then launches one `srun` job per study with throttling.
  - Worker mode (`--study STUDY_ID`): performs extraction + normalization for one study.

This script NEVER downloads rasters (no GEE calls). If rasters are missing, the
worker fails with an actionable error.
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = _REPO_ROOT / "configs" / "benchmarks" / "issf.yaml"

    p = argparse.ArgumentParser(
        description="Launch Slurm jobs to extract MoveBench parametric point features per study (and compute normalization stats)."
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Benchmark YAML config path (used for layers + iSSF sampler params).",
    )
    p.add_argument(
        "--study",
        type=str,
        default=None,
        help="Run worker mode for a single study id (no Slurm submission).",
    )
    p.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Launcher override: 'all' or comma-separated study ids. If omitted, use config's `studies` or all available.",
    )
    p.add_argument("--max-concurrent-jobs", type=int, default=32, help="Maximum concurrent `srun` jobs.")
    p.add_argument("--polling-interval", type=int, default=30, help="Seconds between job status checks.")
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for Slurm stdout/stderr logs (default: results/slurm/extract_points/<config_stem>/).",
    )
    p.add_argument("--partition", type=str, default="vision-beery", help="Slurm partition (-p).")
    p.add_argument("--qos", type=str, default="vision-beery-main", help="Slurm QoS (-q).")
    p.add_argument("--account", type=str, default="vision-beery", help="Slurm account (-A).")
    p.add_argument("--time", type=str, default="7-00:00:00", help="Slurm time limit (-t), e.g. '7-00:00:00'.")
    p.add_argument("--mem", type=str, default="64GB", help="Slurm memory request, e.g. '64GB'.")
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="Slurm CPUs per task (--cpus-per-task).",
    )
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Slurm GPUs per node (--gpus-per-node). Only passed if > 0.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions/commands but do not submit jobs.")
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

    for layer in plan.layers_in_order:
        src = REGISTRY.get(str(layer)).source
        if src in {"gee_chip", "gee_tile"}:
            return True
    return False


def _rasters_root_for(plan, *, study_dir: Path) -> Path:
    return Path(study_dir) / "rasters" / str(plan.rasters_hash)


def _select_parametric_store(
    *,
    study_dir: Path,
    plan,
    split: str,
    sampler: _SamplerParams,
):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.features.points import select_point_store_dir

    return select_point_store_dir(
        study_dir=Path(study_dir),
        split=str(split),
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


def _is_complete(*, study_dir: Path, plan, sampler: _SamplerParams) -> tuple[bool, str | None]:
    try:
        _train_dir, train_m = _select_parametric_store(study_dir=study_dir, plan=plan, split="train", sampler=sampler)
        _select_parametric_store(study_dir=study_dir, plan=plan, split="test", sampler=sampler)
    except (FileNotFoundError, ValueError):
        return False, None

    sampler_hash = str(train_m.sampler_hash)
    stats_path = _norm_stats_path(
        study_dir=study_dir,
        config_hash=str(plan.config_hash),
        sampler_hash=sampler_hash,
    )
    if not stats_path.exists():
        return False, sampler_hash
    return True, sampler_hash


def _build_srun_cmd(
    *,
    args: argparse.Namespace,
    study_id: str,
    config_path: str,
    log_dir: Path,
    cpus_per_task: int,
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
    cmd.extend(["--cpus-per-task", str(int(cpus_per_task))])
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
        ]
    )
    return cmd


def _worker(*, study_id: str, cfg: dict[str, Any], plan, sampler: _SamplerParams) -> int:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import Study, paths

    layers_config = str(cfg["layers_config"])
    setup_workers = int(cfg["runtime"]["setup_workers"])

    # Best-effort fast failure if rasters are clearly missing (and would require GEE downloads).
    if _uses_gee_layers(plan):
        rasters_root = _rasters_root_for(plan, study_dir=paths.study_dir(str(study_id)))
        if not rasters_root.exists():
            print(
                f"[extract] ERROR: rasters cache missing for study={study_id!r}. "
                f"Expected: {rasters_root}. Prepare rasters on a login node first (no downloads in this script).",
                file=sys.stderr,
            )
            return 2

    study = Study(str(study_id))

    sampler_hash: str | None = None
    try:
        _train_dir, train_m = _select_parametric_store(study_dir=study.study_dir, plan=plan, split="train", sampler=sampler)
        _select_parametric_store(study_dir=study.study_dir, plan=plan, split="test", sampler=sampler)
        sampler_hash = str(train_m.sampler_hash)
        print(f"[extract] study={study_id} point stores already exist (sampler_hash={sampler_hash})", flush=True)
    except (FileNotFoundError, ValueError):
        try:
            rep = study.extract_point_features(
                config=layers_config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(sampler.num_candidates),
                random_seed=int(sampler.random_seed),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(sampler.step_multiplier),
                step_q=float(sampler.step_q),
                gap_q=float(sampler.gap_q),
                bounds_buffer_deg=float(sampler.bounds_buffer_deg),
            )
        except FileNotFoundError as e:
            msg = str(e)
            if "rasters" in msg:
                print(
                    f"[extract] ERROR: missing rasters while extracting for study={study_id!r}. "
                    "Prepare rasters on a login node first (no downloads in this script).",
                    file=sys.stderr,
                )
            else:
                print(f"[extract] ERROR: extraction failed for study={study_id!r}: {e}", file=sys.stderr)
            return 2
        sampler_hash = str(rep["sampler_hash"])
        print(f"[extract] study={study_id} extracted point stores (sampler_hash={sampler_hash})", flush=True)

    assert sampler_hash is not None
    _ = study.compute_normalization_vals(
        from_="points",
        config=layers_config,
        sampler_hash=sampler_hash,
        split="train",
        skip_existing=True,
    )

    print(
        f"[extract] OK study_id={study_id} config_hash={plan.config_hash} sampler_hash={sampler_hash}",
        flush=True,
    )
    return 0


def _launcher(args: argparse.Namespace) -> int:
    cfg = _load_cfg(str(args.config))
    plan = _resolve_layer_plan(str(cfg["layers_config"]))
    sampler = _issf_sampler_params(cfg)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import Study, paths
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
    log_dir = Path(args.log_dir) if args.log_dir else (_REPO_ROOT / "results" / "slurm" / "extract_points" / config_stem)
    if not log_dir.is_absolute():
        log_dir = (_REPO_ROOT / log_dir).resolve()

    uses_gee = _uses_gee_layers(plan)

    blocked: list[tuple[str, Path]] = []
    skipped: list[str] = []
    queue: list[str] = []

    for study_id in studies:
        # Validate the study exists early for cleaner launcher errors.
        try:
            _ = Study(str(study_id))
        except FileNotFoundError as e:
            print(f"[extract] WARNING: skipping missing study={study_id!r}: {e}", file=sys.stderr)
            continue

        study_dir = paths.study_dir(str(study_id))
        if uses_gee:
            rasters_root = _rasters_root_for(plan, study_dir=study_dir)
            if not rasters_root.exists():
                blocked.append((str(study_id), rasters_root))
                continue

        done, _sampler_hash = _is_complete(study_dir=study_dir, plan=plan, sampler=sampler)
        if done:
            skipped.append(str(study_id))
        else:
            queue.append(str(study_id))

    print(
        f"[extract] studies total={len(studies)} queued={len(queue)} skipped={len(skipped)} blocked={len(blocked)}",
        flush=True,
    )
    if blocked:
        for sid, expected in blocked:
            print(f"[extract] BLOCKED study={sid} missing_rasters_root={expected}", file=sys.stderr)

    cpus_per_task = int(args.cpus_per_task) if args.cpus_per_task is not None else int(cfg["runtime"]["setup_workers"])

    if args.dry_run:
        if queue:
            print(f"[extract] Dry-run: would launch {len(queue)} jobs (max_concurrent={args.max_concurrent_jobs}).", flush=True)
            for sid in queue[:10]:
                cmd = _build_srun_cmd(args=args, study_id=sid, config_path=str(args.config), log_dir=log_dir, cpus_per_task=cpus_per_task)
                print("[extract] ", shlex.join(cmd), flush=True)
            if len(queue) > 10:
                print(f"[extract] ... ({len(queue)-10} more)", flush=True)
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
        cmd = _build_srun_cmd(args=args, study_id=study_id, config_path=str(args.config), log_dir=log_dir, cpus_per_task=cpus_per_task)
        print(f"[extract] Launching study={study_id}: {shlex.join(cmd)}", flush=True)
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
                    print(f"[extract] FAILED study={sid} exit_code={rc}", file=sys.stderr, flush=True)
                else:
                    print(f"[extract] DONE study={sid}", flush=True)

            while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
                sid = queue[queue_idx]
                queue_idx += 1
                _launch_one(sid)

            pending = total - (completed + len(running))
            print(
                f"[extract] Progress completed={completed}/{total} running={len(running)} pending={pending} failed={len(failures)}",
                flush=True,
            )
            if running:
                print(f"[extract] Running: {sorted(running.keys())}", flush=True)
    except KeyboardInterrupt:
        print("[extract] Interrupted. Not launching new jobs.", file=sys.stderr, flush=True)
        if running:
            print(f"[extract] Still running: {sorted(running.keys())}", file=sys.stderr, flush=True)
        return 130

    if failures or blocked:
        print(f"[extract] Finished with failures={len(failures)} blocked={len(blocked)}", file=sys.stderr, flush=True)
        if failures:
            print(f"[extract] Failed studies: {sorted(failures)}", file=sys.stderr, flush=True)
        return 2

    print("[extract] All jobs completed successfully.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Ensure the repo root is on sys.path before any MoveBench imports.
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    if args.study:
        cfg = _load_cfg(str(args.config))
        plan = _resolve_layer_plan(str(cfg["layers_config"]))
        sampler = _issf_sampler_params(cfg)
        return _worker(study_id=str(args.study), cfg=cfg, plan=plan, sampler=sampler)
    return _launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
