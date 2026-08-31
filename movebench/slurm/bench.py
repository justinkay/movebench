from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from movebench.slurm.launcher import (
    ArrayRequest,
    SlurmResources,
    apply_slurm_config,
    register_common_slurm_arguments,
    run_sbatch_queue,
)

_SLURM_DEFAULTS = {
    "cpus_per_task": 1,
    "max_cpu_jobs": 32,
    "max_gpu_jobs": 32,
    "polling_interval": 5.0,
    "log_dir": Path("results/slurm/bench"),
}

_GPU_DEFAULTS = {
    "baseline_nomove": 0,
    "baseline_constvel": 0,
    "baseline_randomwalk": 0,
    "issf_linear": 0,
    "issf_mlp": 0,
    "deepssf": 1,
    "moveformer": 1,
    "moveformer-full-trainonly": 1,
    "moveformer-full": 1,
}


def _bench_run_module():
    from movebench.bench import run as bench_run

    return bench_run


def _default_gpus_for_method(method: str) -> int:
    return int(_GPU_DEFAULTS.get(str(method), 0))


def _resources_for_run(
    args: argparse.Namespace,
    *,
    run_payload: dict[str, object],
) -> SlurmResources:
    requested_gpus = args.gpus_per_node
    if requested_gpus is None:
        requested_gpus = _default_gpus_for_method(str(run_payload["method"]))
    return SlurmResources(
        partition=args.partition,
        exclude=args.exclude,
        qos=args.qos,
        account=args.account,
        time=args.time,
        mem=args.mem,
        cpus_per_task=int(args.cpus_per_task),
        gpus_per_node=int(requested_gpus),
    )


def _worker_args(
    manifest_path: Path,
    run_key: str,
    *,
    rerun: bool,
) -> list[str]:
    worker_args = [
        "slurm",
        "bench",
        "run",
        "--manifest-path",
        str(manifest_path),
        "--run-key",
        str(run_key),
    ]
    if bool(rerun):
        worker_args.append("--rerun")
    return worker_args


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _bucket_pending_runs(
    args: argparse.Namespace,
    pending_runs: list[dict[str, object]],
) -> list[tuple[SlurmResources, list[dict[str, object]]]]:
    buckets: dict[SlurmResources, list[dict[str, object]]] = {}
    for run_payload in pending_runs:
        resources = _resources_for_run(args, run_payload=run_payload)
        buckets.setdefault(resources, []).append(run_payload)
    return list(buckets.items())


def _run_worker(args: argparse.Namespace) -> int:
    if not args.manifest_path or not args.run_key:
        raise ValueError("Worker mode requires --manifest-path and --run-key")
    bench_run = _bench_run_module()
    run_payload = bench_run.load_resolved_run(Path(args.manifest_path), str(args.run_key))
    summary = bench_run.execute_run_payload(
        run_payload,
        resume=not bool(args.rerun),
    )
    return 0 if str(summary.get("status", "completed")) == "completed" else 2


def _run_launcher(args: argparse.Namespace) -> int:
    if not args.experiment_module:
        raise ValueError("Launcher mode requires an experiment module")
    for name in ("max_cpu_jobs", "max_gpu_jobs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")

    bench_run = _bench_run_module()
    resolved = bench_run.resolve_pending_runs(
        args.experiment_module,
        studies_override=args.studies,
        methods=args.methods,
        max_runs=args.max_runs,
        rerun=bool(args.rerun),
    )
    counts = dict(resolved["counts"])
    pending_runs = list(resolved["pending_runs"])
    manifest_path = Path(resolved["manifest_path"])

    print(
        f"[bench] planned={counts['planned_runs']} pending={counts['pending_runs']} "
        f"skipped={counts['skipped_runs']} blocked={counts['blocked_runs']}"
    )
    if not pending_runs:
        print("[bench] No pending runs.")
        return 0

    if not args.dry_run:
        _write_manifest(manifest_path, dict(resolved["manifest_payload"]))

    buckets = _bucket_pending_runs(args, pending_runs)
    requests: list[ArrayRequest] = []
    for bucket_index, (resources, bucket_runs) in enumerate(buckets):
        max_jobs = args.max_gpu_jobs if resources.gpus_per_node else args.max_cpu_jobs
        requests.append(
            ArrayRequest(
                resources=resources,
                array_concurrency=min(len(bucket_runs), int(max_jobs)),
                log_path=Path(args.log_dir) / f"bucket-{bucket_index}.%A_%a.log",
                worker_args_list=tuple(
                    tuple(
                        _worker_args(
                            manifest_path,
                            str(run_payload["run_key"]),
                            rerun=bool(args.rerun),
                        )
                    )
                    for run_payload in bucket_runs
                ),
                manifest_dir=Path(args.log_dir) / "manifests",
                manifest_stem=f"bucket{bucket_index}",
                name=f"bucket={bucket_index}",
            )
        )

    return run_sbatch_queue(
        label="bench",
        requests=requests,
        polling_interval=float(args.polling_interval),
        dry_run=bool(args.dry_run),
    )


def _run(args: argparse.Namespace) -> int:
    apply_slurm_config(args, command_name="bench", defaults=_SLURM_DEFAULTS)
    if args.manifest_path or args.run_key:
        return _run_worker(args)
    return _run_launcher(args)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "bench",
        help="Submit benchmark runs through Slurm.",
        description="Submit benchmark runs through Slurm.",
    )
    bench_subparsers = parser.add_subparsers(dest="slurm_bench_command", required=True)
    run_parser = bench_subparsers.add_parser(
        "run",
        help="Submit benchmark runs through Slurm.",
        description="Submit benchmark runs through Slurm.",
    )
    run_parser.add_argument("experiment_module", nargs="?")
    run_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    run_parser.add_argument("--methods", nargs="+", default=None)
    run_parser.add_argument("--max-runs", type=int, default=None)
    run_parser.add_argument("--max-cpu-jobs", type=int, default=None)
    run_parser.add_argument("--max-gpu-jobs", type=int, default=None)
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.add_argument("--manifest-path", type=Path, default=None, help=argparse.SUPPRESS)
    run_parser.add_argument("--run-key", default=None, help=argparse.SUPPRESS)
    register_common_slurm_arguments(run_parser)
    run_parser.set_defaults(
        func=_run,
    )
