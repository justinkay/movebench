#!/usr/bin/env python
"""Launch and run MoveBench point feature extraction jobs on Slurm.

Single-entry-point script with two modes:
  - Launcher mode (default): discovers studies/layer-config/sampler-config
    combinations needing extraction + normalization, then launches one `srun`
    job per study/combo pair with throttling.
  - Worker mode (`--study STUDY_ID`): performs extraction + normalization for one
    study across one or more requested layer/sampler combinations.

This script NEVER downloads rasters (no GEE calls). If rasters are missing, the
worker fails with an actionable error.
"""

from __future__ import annotations

import argparse
import _bootstrap as bootstrap
import hashlib
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from movebench import paths
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.layers.config import load_layers_config
from movebench.layers.registry import REGISTRY
from movebench.study import Study

_REPO_ROOT = bootstrap.REPO_ROOT


@dataclass(frozen=True)
class _ExtractionRun:
    layers_config_path: str
    sampler_config_path: str
    plan: Any
    sampler: SamplerConfig
    combo_stem: str
    combo_label: str
    log_dir_name: str


def _parse_multi_path_arg(val: str, *, flag_name: str) -> list[str]:
    out = [p.strip() for p in str(val).split(",") if p.strip()]
    if not out:
        raise ValueError(
            f"{flag_name} must include at least one path or a comma-separated list of paths."
        )
    return out


def _normalize_path(path: str) -> str:
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    else:
        p = p.resolve()
    return str(p)


def _resolve_layer_plan(layers_config: str):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    cfg = load_layers_config(layers_config)
    return REGISTRY.resolve(cfg)


def _combo_stem(layers_config_path: str, sampler_config_path: str) -> str:
    return f"{Path(str(layers_config_path)).stem}__{Path(str(sampler_config_path)).stem}"


def _combo_label(layers_config_path: str, sampler_config_path: str) -> str:
    stem = _combo_stem(layers_config_path, sampler_config_path)
    digest = hashlib.sha1(
        f"{layers_config_path}|{sampler_config_path}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{stem}-{digest}"


def _load_extraction_run(
    *, layers_config_path: str, sampler_config_path: str
) -> _ExtractionRun:
    normalized_layers = _normalize_path(layers_config_path)
    normalized_sampler = _normalize_path(sampler_config_path)
    combo_stem = _combo_stem(normalized_layers, normalized_sampler)
    combo_label = _combo_label(normalized_layers, normalized_sampler)
    return _ExtractionRun(
        layers_config_path=normalized_layers,
        sampler_config_path=normalized_sampler,
        plan=_resolve_layer_plan(normalized_layers),
        sampler=load_sampler_config(normalized_sampler),
        combo_stem=combo_stem,
        combo_label=combo_label,
        log_dir_name=combo_stem,
    )


def _finalize_log_dir_names(runs: list[_ExtractionRun]) -> list[_ExtractionRun]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.combo_stem] = counts.get(run.combo_stem, 0) + 1

    out: list[_ExtractionRun] = []
    for run in runs:
        log_dir_name = run.combo_stem
        if counts[run.combo_stem] > 1:
            log_dir_name = run.combo_label
        out.append(replace(run, log_dir_name=log_dir_name))
    return out


def _resolve_requested_studies(studies_arg: str | None) -> list[str]:
    if studies_arg is None or not str(studies_arg).strip():
        studies = Study.list_available()
    else:
        raw = str(studies_arg).strip()
        if raw.lower() == "all":
            studies = Study.list_available()
        else:
            studies = sorted(
                dict.fromkeys([part.strip() for part in raw.split(",") if part.strip()])
            )

    if not studies:
        raise ValueError(
            "No studies resolved. "
            f"Expected studies under: {paths.studies_root()} "
            "(override with env MOVEBENCH_DATA_ROOT or config.env)."
        )
    return studies


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_layers_config = _REPO_ROOT / "configs" / "layers" / "all.yaml"
    default_sampler_config = _REPO_ROOT / "configs" / "samplers" / "issf_defaults.yaml"

    p = argparse.ArgumentParser(
        description="Launch Slurm jobs to extract MoveBench point features per study (and compute normalization stats)."
    )
    p.add_argument(
        "--layers-config",
        dest="layers_configs",
        action="append",
        type=lambda val: _parse_multi_path_arg(val, flag_name="--layers-config"),
        default=None,
        help=(
            "Layers YAML config path. May be passed multiple times or as a "
            "comma-separated list; extraction runs for every requested layer config."
        ),
    )
    p.add_argument(
        "--sampler-config",
        dest="sampler_configs",
        action="append",
        type=lambda val: _parse_multi_path_arg(val, flag_name="--sampler-config"),
        default=None,
        help=(
            "Sampler YAML config path. May be passed multiple times or as a "
            "comma-separated list; extraction runs for every requested sampler config."
        ),
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
        help="Launcher override: 'all' or comma-separated study ids. If omitted, use all available studies.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers for feature extraction.",
    )
    p.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=32,
        help="Maximum concurrent `srun` jobs.",
    )
    p.add_argument(
        "--polling-interval",
        type=int,
        default=30,
        help="Seconds between job status checks.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help=(
            "Directory for Slurm stdout/stderr logs "
            "(default: results/slurm/extract_points/<combo_stem>/)."
        ),
    )
    p.add_argument(
        "--partition", type=str, default="vision-beery", help="Slurm partition (-p)."
    )
    p.add_argument(
        "--qos", type=str, default="vision-beery-main", help="Slurm QoS (-q)."
    )
    p.add_argument(
        "--account", type=str, default="vision-beery", help="Slurm account (-A)."
    )
    p.add_argument(
        "--time",
        type=str,
        default="7-00:00:00",
        help="Slurm time limit (-t), e.g. '7-00:00:00'.",
    )
    p.add_argument(
        "--mem", type=str, default="64GB", help="Slurm memory request, e.g. '64GB'."
    )
    p.add_argument(
        "--cpus-per-task",
        type=int,
        default=16,
        help="Slurm CPUs per task (--cpus-per-task).",
    )
    p.add_argument(
        "--gpus-per-node",
        type=int,
        default=0,
        help="Slurm GPUs per node (--gpus-per-node). Only passed if > 0.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions/commands but do not submit jobs.",
    )
    args = p.parse_args(argv)

    raw_layers = args.layers_configs or [[str(default_layers_config)]]
    raw_samplers = args.sampler_configs or [[str(default_sampler_config)]]

    layers_configs: list[str] = []
    seen_layers: set[str] = set()
    for group in raw_layers:
        for config_path in group:
            normalized = _normalize_path(config_path)
            if normalized in seen_layers:
                continue
            seen_layers.add(normalized)
            layers_configs.append(normalized)

    sampler_configs: list[str] = []
    seen_samplers: set[str] = set()
    for group in raw_samplers:
        for config_path in group:
            normalized = _normalize_path(config_path)
            if normalized in seen_samplers:
                continue
            seen_samplers.add(normalized)
            sampler_configs.append(normalized)

    args.layers_configs = layers_configs
    args.sampler_configs = sampler_configs
    return args


def _uses_gee_layers(plan) -> bool:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    for layer in plan.layers_in_order:
        src = REGISTRY.get(str(layer)).source
        if src in {"gee_chip", "gee_tile"}:
            return True
    return False


def _rasters_root_for(plan, *, study_dir: Path) -> Path:
    return Path(study_dir) / "rasters" / str(plan.rasters_hash)


def _resolve_run_sampler_hash(run: _ExtractionRun) -> str:
    from movebench.features.points import resolve_sampler_hash

    return resolve_sampler_hash(run.sampler)


def _default_splits(study: Study) -> list[str]:
    split_keys = [str(split) for split in study.split_inventory()["split_keys"]]
    return split_keys or ["train"]


def _is_complete(
    *,
    study_dir: Path,
    run: _ExtractionRun,
    splits: list[str],
) -> tuple[bool, str]:
    from movebench.features.points import (
        point_store_manifest_path,
        point_store_train_stats_path,
    )

    sampler_hash = _resolve_run_sampler_hash(run)
    for split in list(splits):
        manifest_path = point_store_manifest_path(
            study_dir=study_dir,
            split=split,
            rasters_hash=str(run.plan.rasters_hash),
            sampler_hash=sampler_hash,
        )
        if not manifest_path.exists():
            return False, sampler_hash

    stats_path = point_store_train_stats_path(
        study_dir=study_dir,
        rasters_hash=str(run.plan.rasters_hash),
        sampler_hash=sampler_hash,
    )
    if not stats_path.exists():
        return False, sampler_hash
    return True, sampler_hash


def _build_srun_cmd(
    *,
    args: argparse.Namespace,
    study_id: str,
    run: _ExtractionRun,
    log_dir: Path,
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
            "--layers-config",
            str(run.layers_config_path),
            "--sampler-config",
            str(run.sampler_config_path),
            "--study",
            str(study_id),
            "--n-workers",
            str(int(args.n_workers)),
        ]
    )
    return cmd


def _worker_one(*, study_id: str, run: _ExtractionRun, n_workers: int) -> int:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    plan = run.plan
    sampler = run.sampler
    layers_config = str(run.layers_config_path)

    if _uses_gee_layers(plan):
        rasters_root = _rasters_root_for(plan, study_dir=paths.study_dir(str(study_id)))
        if not rasters_root.exists():
            print(
                f"[extract] ERROR: rasters cache missing for study={study_id!r} layers={run.layers_config_path!r} sampler={run.sampler_config_path!r}. "
                f"Expected: {rasters_root}. Prepare rasters on a login node first (no downloads in this script).",
                file=sys.stderr,
            )
            return 2

    study = Study(str(study_id))
    splits = _default_splits(study)

    done, sampler_hash = _is_complete(
        study_dir=study.study_dir,
        run=run,
        splits=splits,
    )
    if done:
        print(
            f"[extract] study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path} "
            f"point stores already exist (sampler_hash={sampler_hash})",
            flush=True,
        )
    else:
        try:
            rep = study.extract_point_features(
                layers_config=layers_config,
                sampler_config=run.sampler,
                splits=splits,
                n_workers=int(n_workers),
                skip_existing=True,
            )
        except FileNotFoundError as e:
            msg = str(e)
            if "rasters" in msg:
                print(
                    f"[extract] ERROR: missing rasters while extracting for study={study_id!r} layers={run.layers_config_path!r} sampler={run.sampler_config_path!r}. "
                    "Prepare rasters on a login node first (no downloads in this script).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[extract] ERROR: extraction failed for study={study_id!r} layers={run.layers_config_path!r} sampler={run.sampler_config_path!r}: {e}",
                    file=sys.stderr,
                )
            return 2
        sampler_hash = str(rep["sampler_hash"])
        print(
            f"[extract] study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path} "
            f"extracted point stores (sampler_hash={sampler_hash})",
            flush=True,
        )

    _ = study.compute_normalization_vals(
        from_="points",
        layers_config=layers_config,
        sampler_config=run.sampler,
        split="train",
        skip_existing=True,
    )

    print(
        f"[extract] OK study_id={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path} "
        f"rasters_hash={plan.rasters_hash} sampler_hash={sampler_hash}",
        flush=True,
    )
    return 0


def _worker(*, study_id: str, runs: list[_ExtractionRun], n_workers: int) -> int:
    failures: list[str] = []
    for run in runs:
        rc = _worker_one(study_id=study_id, run=run, n_workers=n_workers)
        if rc != 0:
            failures.append(
                f"layers={run.layers_config_path} sampler={run.sampler_config_path}"
            )

    if failures:
        print(
            f"[extract] Worker finished with failures for study={study_id}: {failures}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


def _item_log_dir(
    *, args: argparse.Namespace, base_log_dir: Path, run: _ExtractionRun, run_count: int
) -> Path:
    if args.log_dir and run_count == 1:
        return base_log_dir
    return base_log_dir / run.log_dir_name


def _build_runs(args: argparse.Namespace) -> list[_ExtractionRun]:
    return _finalize_log_dir_names(
        [
            _load_extraction_run(
                layers_config_path=layers_config_path,
                sampler_config_path=sampler_config_path,
            )
            for layers_config_path in args.layers_configs
            for sampler_config_path in args.sampler_configs
        ]
    )


def _launcher(args: argparse.Namespace, runs: list[_ExtractionRun]) -> int:
    studies = _resolve_requested_studies(args.studies)

    requested: list[tuple[str, _ExtractionRun]] = []
    for run in runs:
        for study_id in studies:
            requested.append((str(study_id), run))

    base_log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else (_REPO_ROOT / "results" / "slurm" / "extract_points")
    )
    if not base_log_dir.is_absolute():
        base_log_dir = (_REPO_ROOT / base_log_dir).resolve()

    blocked: list[tuple[str, Path]] = []
    skipped: list[str] = []
    queue: list[tuple[str, _ExtractionRun]] = []
    missing_studies: set[str] = set()
    validated_studies: set[str] = set()

    for study_id, run in requested:
        if study_id in missing_studies:
            continue
        try:
            if study_id not in validated_studies:
                _ = Study(str(study_id))
                validated_studies.add(study_id)
        except FileNotFoundError as e:
            missing_studies.add(str(study_id))
            print(
                f"[extract] WARNING: skipping missing study={study_id!r}: {e}",
                file=sys.stderr,
            )
            continue

        study = Study(str(study_id))
        study_dir = paths.study_dir(str(study_id))
        splits = _default_splits(study)
        if _uses_gee_layers(run.plan):
            rasters_root = _rasters_root_for(run.plan, study_dir=study_dir)
            if not rasters_root.exists():
                blocked.append(
                    (
                        f"study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path}",
                        rasters_root,
                    )
                )
                continue

        done, _sampler_hash = _is_complete(
            study_dir=study_dir,
            run=run,
            splits=splits,
        )
        if done:
            skipped.append(
                f"study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path}"
            )
        else:
            queue.append((str(study_id), run))

    unique_studies = sorted({study_id for study_id, _ in requested})

    print(
        f"[extract] combos={len(runs)} studies={len(unique_studies)} "
        f"requested_pairs={len(requested)} queued={len(queue)} "
        f"skipped={len(skipped)} blocked={len(blocked)}",
        flush=True,
    )
    if blocked:
        for item_label, expected in blocked:
            print(
                f"[extract] BLOCKED {item_label} missing_rasters_root={expected}",
                file=sys.stderr,
            )

    if args.dry_run:
        if queue:
            print(
                f"[extract] Dry-run: would launch {len(queue)} jobs (max_concurrent={args.max_concurrent_jobs}).",
                flush=True,
            )
            for sid, run in queue[:10]:
                item_log_dir = _item_log_dir(
                    args=args,
                    base_log_dir=base_log_dir,
                    run=run,
                    run_count=len(runs),
                )
                cmd = _build_srun_cmd(
                    args=args,
                    study_id=sid,
                    run=run,
                    log_dir=item_log_dir,
                )
                print("[extract] ", shlex.join(cmd), flush=True)
            if len(queue) > 10:
                print(f"[extract] ... ({len(queue) - 10} more)", flush=True)
        return 2 if blocked else 0

    if not queue:
        return 2 if blocked else 0

    if int(args.max_concurrent_jobs) <= 0:
        raise ValueError("--max-concurrent-jobs must be > 0")
    if int(args.polling_interval) <= 0:
        raise ValueError("--polling-interval must be > 0")

    base_log_dir.mkdir(parents=True, exist_ok=True)

    running: dict[str, subprocess.Popen] = {}
    failures: list[str] = []
    completed = 0
    total = len(queue)
    queue_idx = 0

    def _launch_one(study_id: str, run: _ExtractionRun) -> None:
        item_label = (
            f"study={study_id} layers={run.layers_config_path} "
            f"sampler={run.sampler_config_path}"
        )
        item_log_dir = _item_log_dir(
            args=args,
            base_log_dir=base_log_dir,
            run=run,
            run_count=len(runs),
        )
        item_log_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_srun_cmd(
            args=args,
            study_id=study_id,
            run=run,
            log_dir=item_log_dir,
        )
        print(f"[extract] Launching {item_label}: {shlex.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
        running[item_label] = proc

    try:
        while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
            sid, run = queue[queue_idx]
            queue_idx += 1
            _launch_one(sid, run)

        while running or queue_idx < len(queue):
            time.sleep(int(args.polling_interval))

            for item_label, proc in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                running.pop(item_label, None)
                completed += 1
                if int(rc) != 0:
                    failures.append(item_label)
                    print(
                        f"[extract] FAILED {item_label} exit_code={rc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[extract] DONE {item_label}", flush=True)

            while queue_idx < len(queue) and len(running) < int(
                args.max_concurrent_jobs
            ):
                sid, run = queue[queue_idx]
                queue_idx += 1
                _launch_one(sid, run)

            pending = total - (completed + len(running))
            print(
                f"[extract] Progress completed={completed}/{total} running={len(running)} pending={pending} failed={len(failures)}",
                flush=True,
            )
            if running:
                print(f"[extract] Running: {sorted(running.keys())}", flush=True)
    except KeyboardInterrupt:
        print(
            "[extract] Interrupted. Not launching new jobs.",
            file=sys.stderr,
            flush=True,
        )
        if running:
            print(
                f"[extract] Still running: {sorted(running.keys())}",
                file=sys.stderr,
                flush=True,
            )
        return 130

    if failures or blocked:
        print(
            f"[extract] Finished with failures={len(failures)} blocked={len(blocked)}",
            file=sys.stderr,
            flush=True,
        )
        if failures:
            print(
                f"[extract] Failed items: {sorted(failures)}",
                file=sys.stderr,
                flush=True,
            )
        return 2

    print("[extract] All jobs completed successfully.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    runs = _build_runs(args)

    if args.study:
        return _worker(
            study_id=str(args.study),
            runs=runs,
            n_workers=int(args.n_workers),
        )
    return _launcher(args, runs)


if __name__ == "__main__":
    raise SystemExit(main())
