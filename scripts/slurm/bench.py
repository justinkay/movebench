#!/usr/bin/env python
"""Launch and run MoveBench benchmarks on Slurm (one study per job).

Single-entry-point script with two modes:
  - Launcher mode (default): resolves studies from a benchmark config, skips
    studies already complete, and launches one `srun` job per study with
    throttling.
  - Worker mode (`--study STUDY_ID`): runs all benchmark methods for one study,
    writing a per-study results JSON file under the benchmark output_dir.

This script NEVER downloads rasters and NEVER extracts point features. It
expects rasters + point stores + (when required) normalization stats to already
exist. Use:
  - scripts/slurm/extract.py to build point stores + normalization stats
  - scripts/slurm/wds.py to build deepSSF WebDataset caches (optional)
"""

from __future__ import annotations

import argparse
import _bootstrap as bootstrap
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from movebench.benchmark import load_config
from movebench.benchmark import resolve_default_eval_split
from movebench.benchmark import run_method
from movebench.benchmark import write_outputs
from movebench import Study, paths
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.features.points import (
    load_point_store_manifest,
    point_store_train_stats_path,
    resolve_sampler_hash,
)
from movebench.layers.config import load_layers_config
from movebench.layers.registry import REGISTRY
from movebench.study import resolve_study_selection
import torch

_REPO_ROOT = bootstrap.REPO_ROOT


def _resolve_launcher_studies(cfg: dict[str, Any], override: str | None) -> list[str]:
    if override is not None and str(override).strip():
        return resolve_study_selection(str(override))
    return resolve_study_selection(cfg.get("studies", "all"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = _REPO_ROOT / "configs" / "benchmarks" / "all.yaml"

    p = argparse.ArgumentParser(
        description="Launch Slurm jobs to run MoveBench benchmarks per study."
    )
    p.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Benchmark YAML config path.",
    )
    p.add_argument(
        "--study", type=str, default=None, help="Run worker mode for a single study id."
    )
    p.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Launcher override: 'all', 'mini', or comma-separated study ids. If omitted, use config's studies value.",
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
        help="Directory for Slurm stdout/stderr logs (default: results/slurm/benchmark/<config_stem>/).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing per-study results and rerun all methods for queued studies.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions/commands but do not submit jobs.",
    )

    # Slurm defaults match scripts/slurm/extract.py.
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
        default="1-00:00:00",
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
        default=1,
        help="Slurm GPUs per node (--gpus-per-node). Only passed if > 0.",
    )

    return p.parse_args(argv)


def _load_cfg(config_path: str) -> dict[str, Any]:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    return load_config(config_path)


def _resolve_layer_plan(layers_config: str):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    cfg = load_layers_config(layers_config)
    return REGISTRY.resolve(cfg)


def _uses_gee_layers(plan) -> bool:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

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


def _issf_sampler_config(cfg: dict[str, Any]) -> SamplerConfig:
    return load_sampler_config(str(cfg["sampler_config"]))


def _select_parametric_store(
    *,
    study_dir: Path,
    plan,
    split: str,
    sampler: SamplerConfig,
):
    return load_point_store_manifest(
        study_dir=Path(study_dir),
        split=str(split),
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=resolve_sampler_hash(sampler),
    )


def _norm_stats_path(*, study_dir: Path, rasters_hash: str, sampler_hash: str) -> Path:
    return point_store_train_stats_path(
        study_dir=Path(study_dir),
        rasters_hash=str(rasters_hash),
        sampler_hash=str(sampler_hash),
    )


def _needs_points_normalization(cfg: dict[str, Any]) -> bool:
    methods = cfg.get("methods", [])
    if not isinstance(methods, list):
        return False
    kinds = {str(m.get("kind", "")) for m in methods if isinstance(m, dict)}

    need = False
    if any(k in {"issf_linear", "issf_mlp"} for k in kinds):
        issf = cfg.get("datasets", {}).get("issf", {})
        if isinstance(issf, dict):
            need |= bool(issf.get("normalize", True))

    if "deepssf" in kinds:
        dcfg = cfg.get("datasets", {}).get("deepssf", {})
        if isinstance(dcfg, dict):
            need |= bool(dcfg.get("normalize", True))

    if "moveformer" in kinds:
        mcfg = cfg.get("datasets", {}).get("moveformer", {})
        if isinstance(mcfg, dict):
            normalize_env = bool(mcfg.get("normalize_env", True))
            require = bool(mcfg.get("require_normalization_stats", False))
            need |= bool(normalize_env and require)

    return bool(need)


def _method_complete(val: Any) -> bool:
    return isinstance(val, dict) and "fit" in val and "test" in val


def _study_results_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "slurm_studies"


def _study_results_path(out_dir: Path, study_id: str) -> Path:
    safe = str(study_id).replace(os.sep, "_")
    return _study_results_dir(out_dir) / f"{safe}.json"


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _check_prereqs(
    *,
    study_id: str,
    cfg: dict[str, Any],
    plan,
    sampler: SamplerConfig,
    require_norm_stats: bool,
) -> tuple[bool, str | None, str]:
    """Return (ok, sampler_hash, reason)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        study = Study(str(study_id))
    except FileNotFoundError as e:
        return False, None, f"missing_study: {e}"

    if _uses_gee_layers(plan):
        rasters_root = _rasters_root_for(plan, study_dir=paths.study_dir(str(study_id)))
        if not rasters_root.exists():
            return False, None, f"missing_rasters_root: {rasters_root}"

    try:
        _train_dir, train_m = _select_parametric_store(
            study_dir=study.study_dir, plan=plan, split="train", sampler=sampler
        )
        eval_split = resolve_default_eval_split(study)
        _select_parametric_store(
            study_dir=study.study_dir,
            plan=plan,
            split=eval_split,
            sampler=sampler,
        )
    except (FileNotFoundError, ValueError) as e:
        return False, None, f"missing_parametric_point_store: {e}"

    sampler_hash = str(train_m.sampler_hash)
    if require_norm_stats:
        stats_path = _norm_stats_path(
            study_dir=study.study_dir,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=sampler_hash,
        )
        if not stats_path.exists():
            return False, None, f"missing_points_normalization_stats: {stats_path}"

    return True, sampler_hash, "ok"


def _build_srun_cmd(
    *,
    args: argparse.Namespace,
    study_id: str,
    config_path: str,
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
            "--config",
            str(config_path),
            "--study",
            str(study_id),
        ]
    )
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _merge_outputs(*, cfg: dict[str, Any], out_dir: Path) -> None:
    """Merge per-study results into raw_results.json + CSV summaries."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    raw_path = Path(out_dir) / "raw_results.json"
    results: dict[str, Any] = {}
    if raw_path.exists():
        try:
            loaded = _load_json(raw_path)
            if isinstance(loaded, dict):
                results = dict(loaded)
        except Exception:
            results = {}

    sdir = _study_results_dir(out_dir)
    if sdir.exists():
        for p in sorted(sdir.glob("*.json")):
            try:
                d = _load_json(p)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            study_id = p.stem
            # If the filename was sanitized, the true study_id might differ, but for
            # typical MoveBench study ids this is fine.
            meta = d.get("__meta__")
            if isinstance(meta, dict) and isinstance(meta.get("study_id"), str):
                study_id = str(meta["study_id"])
            if "__meta__" in d:
                d = dict(d)
                d.pop("__meta__", None)
            results[str(study_id)] = dict(d)

    write_outputs(results=results, out_dir=Path(out_dir), cfg=cfg)


def _worker(args: argparse.Namespace) -> int:
    assert args.study is not None

    cfg = _load_cfg(str(args.config))
    out_dir = Path(str(cfg["output_dir"]))
    config_stem = Path(str(args.config)).stem

    plan = _resolve_layer_plan(str(cfg["layers_config"]))
    sampler = _issf_sampler_config(cfg)
    need_norm = _needs_points_normalization(cfg)

    ok, sampler_hash, reason = _check_prereqs(
        study_id=str(args.study),
        cfg=cfg,
        plan=plan,
        sampler=sampler,
        require_norm_stats=bool(need_norm),
    )
    if not ok or sampler_hash is None:
        print(
            f"[bench] ERROR: prerequisites not met for study={args.study!r}: {reason}\n"
            f"[bench] Hint: run /Users/justinkay/movebench/scripts/slurm/extract.py first (rasters/points/normalization).",
            file=sys.stderr,
        )
        return 2

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    device = torch.device(str(cfg["runtime"]["device"]))
    study = Study(str(args.study))

    per_study_path = _study_results_path(out_dir, str(args.study))
    study_results: dict[str, Any] = {}
    if per_study_path.exists() and not bool(args.overwrite):
        try:
            loaded = _load_json(per_study_path)
            if isinstance(loaded, dict):
                study_results = dict(loaded)
        except Exception:
            study_results = {}

    if bool(args.overwrite):
        study_results = {}

    for method in cfg["methods"]:
        if not isinstance(method, dict):
            continue
        method_name = str(method["name"])
        kind = str(method["kind"])

        if _method_complete(study_results.get(method_name)) and not bool(
            args.overwrite
        ):
            print(f"[{study.study_id}] {method_name} exists; skipping", flush=True)
            continue

        print(f"Running {study.study_id} {method_name}", flush=True)
        study_results[method_name] = run_method(
            kind=kind,
            study=study,
            sampler_hash=str(sampler_hash),
            cfg=cfg,
            method=method,
            device=device,
        )

        study_results.setdefault("__meta__", {})
        if isinstance(study_results["__meta__"], dict):
            study_results["__meta__"].update(
                {
                    "study_id": str(study.study_id),
                    "benchmark": str(config_stem),
                    "config_path": str(args.config),
                    "output_dir": str(out_dir),
                    "layers_config": str(cfg["layers_config"]),
                    "rasters_hash": str(plan.rasters_hash),
                    "sampler_hash": str(sampler_hash),
                }
            )
        _write_json_atomic(per_study_path, study_results)
        print(f"[{study.study_id}] {method_name} done", flush=True)

    print(
        f"[bench] OK study_id={study.study_id} benchmark={config_stem} output_dir={out_dir} per_study={per_study_path}",
        flush=True,
    )
    return 0


def _launcher(args: argparse.Namespace) -> int:
    cfg = _load_cfg(str(args.config))
    out_dir = Path(str(cfg["output_dir"]))
    plan = _resolve_layer_plan(str(cfg["layers_config"]))
    sampler = _issf_sampler_config(cfg)
    need_norm = _needs_points_normalization(cfg)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    studies = _resolve_launcher_studies(cfg, args.studies)

    config_stem = Path(str(args.config)).stem
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else (_REPO_ROOT / "results" / "slurm" / "benchmark" / str(config_stem))
    )
    if not log_dir.is_absolute():
        log_dir = (_REPO_ROOT / log_dir).resolve()

    blocked: list[tuple[str, str]] = []
    skipped: list[str] = []
    queue: list[str] = []

    # Determine completion based on per-study JSON files.
    want_methods = [
        str(m["name"]) for m in cfg["methods"] if isinstance(m, dict) and "name" in m
    ]
    raw_results: dict[str, Any] = {}
    raw_results_path = out_dir / "raw_results.json"
    if not bool(args.overwrite) and raw_results_path.exists():
        try:
            loaded = _load_json(raw_results_path)
            if isinstance(loaded, dict):
                raw_results = dict(loaded)
        except Exception:
            raw_results = {}
    for study_id in studies:
        ok, _sampler_hash, reason = _check_prereqs(
            study_id=str(study_id),
            cfg=cfg,
            plan=plan,
            sampler=sampler,
            require_norm_stats=bool(need_norm),
        )
        if not ok:
            blocked.append((str(study_id), reason))
            continue

        if bool(args.overwrite):
            queue.append(str(study_id))
            continue

        pth = _study_results_path(out_dir, str(study_id))
        if not pth.exists():
            rr = raw_results.get(str(study_id))
            if isinstance(rr, dict) and all(
                _method_complete(rr.get(m)) for m in want_methods
            ):
                skipped.append(str(study_id))
                continue
            queue.append(str(study_id))
            continue
        try:
            d = _load_json(pth)
        except Exception:
            rr = raw_results.get(str(study_id))
            if isinstance(rr, dict) and all(
                _method_complete(rr.get(m)) for m in want_methods
            ):
                skipped.append(str(study_id))
                continue
            queue.append(str(study_id))
            continue
        if not isinstance(d, dict):
            rr = raw_results.get(str(study_id))
            if isinstance(rr, dict) and all(
                _method_complete(rr.get(m)) for m in want_methods
            ):
                skipped.append(str(study_id))
                continue
            queue.append(str(study_id))
            continue
        if all(_method_complete(d.get(m)) for m in want_methods):
            skipped.append(str(study_id))
        else:
            queue.append(str(study_id))

    print(
        f"[bench] studies total={len(studies)} queued={len(queue)} skipped={len(skipped)} blocked={len(blocked)}",
        flush=True,
    )
    if blocked:
        for sid, reason in blocked:
            print(f"[bench] BLOCKED study={sid} reason={reason}", file=sys.stderr)

    if args.dry_run:
        if queue:
            print(
                f"[bench] Dry-run: would launch {len(queue)} jobs (max_concurrent={args.max_concurrent_jobs}).",
                flush=True,
            )
            for sid in queue[:10]:
                cmd = _build_srun_cmd(
                    args=args,
                    study_id=sid,
                    config_path=str(args.config),
                    log_dir=log_dir,
                )
                print("[bench] ", shlex.join(cmd), flush=True)
            if len(queue) > 10:
                print(f"[bench] ... ({len(queue) - 10} more)", flush=True)
        return 2 if blocked else 0

    if int(args.max_concurrent_jobs) <= 0:
        raise ValueError("--max-concurrent-jobs must be > 0")
    if int(args.polling_interval) <= 0:
        raise ValueError("--polling-interval must be > 0")

    log_dir.mkdir(parents=True, exist_ok=True)

    if not queue:
        # Still merge existing per-study outputs into raw_results + CSVs.
        try:
            _merge_outputs(cfg=cfg, out_dir=out_dir)
        except Exception as e:
            print(f"[bench] WARNING: failed to merge outputs: {e}", file=sys.stderr)
        return 2 if blocked else 0

    running: dict[str, subprocess.Popen] = {}
    failures: list[str] = []
    completed = 0
    total = len(queue)
    queue_idx = 0

    def _launch_one(study_id: str) -> None:
        cmd = _build_srun_cmd(
            args=args, study_id=study_id, config_path=str(args.config), log_dir=log_dir
        )
        print(f"[bench] Launching study={study_id}: {shlex.join(cmd)}", flush=True)
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
                    print(
                        f"[bench] FAILED study={sid} exit_code={rc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(f"[bench] DONE study={sid}", flush=True)

            while queue_idx < len(queue) and len(running) < int(
                args.max_concurrent_jobs
            ):
                sid = queue[queue_idx]
                queue_idx += 1
                _launch_one(sid)

            pending = total - (completed + len(running))
            print(
                f"[bench] Progress completed={completed}/{total} running={len(running)} pending={pending} failed={len(failures)}",
                flush=True,
            )
            if running:
                print(f"[bench] Running: {sorted(running.keys())}", flush=True)
    except KeyboardInterrupt:
        print(
            "[bench] Interrupted. Not launching new jobs.", file=sys.stderr, flush=True
        )
        if running:
            print(
                f"[bench] Still running: {sorted(running.keys())}",
                file=sys.stderr,
                flush=True,
            )
        return 130

    # Merge per-study results into raw_results.json + CSV summaries.
    try:
        _merge_outputs(cfg=cfg, out_dir=out_dir)
    except Exception as e:
        print(f"[bench] WARNING: failed to merge outputs: {e}", file=sys.stderr)

    if failures or blocked:
        print(
            f"[bench] Finished with failures={len(failures)} blocked={len(blocked)}",
            file=sys.stderr,
            flush=True,
        )
        if failures:
            print(
                f"[bench] Failed studies: {sorted(failures)}",
                file=sys.stderr,
                flush=True,
            )
        return 2

    print("[bench] All jobs completed successfully.", flush=True)
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
