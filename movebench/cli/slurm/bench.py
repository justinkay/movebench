from __future__ import annotations
import argparse
from pathlib import Path

import movebench.bench.config as bench_config
import movebench.bench.run as bench_run
from movebench.cli.bench import add_run_arguments
from movebench.cli.slurm.common import (
    apply_slurm_config,
    register_common_slurm_arguments,
)
from movebench.slurm.launcher import (
    ArrayRequest,
    SlurmResources,
    run_sbatch_queue,
)

_SLURM_DEFAULTS = {
    "cpus_per_task": 1,
    "max_cpu_jobs": 32,
    "max_gpu_jobs": 32,
    "polling_interval": 5.0,
    "log_dir": Path("results/slurm/bench"),
}

def _default_gpus_for_method(method: str) -> int:
    return int(bench_config.METHODS.get(str(method), {}).get("gpus", 0))


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


def _bucket_pending_runs(
    args: argparse.Namespace,
    pending_runs: list[dict[str, object]],
) -> list[tuple[SlurmResources, list[dict[str, object]]]]:
    buckets: dict[SlurmResources, list[dict[str, object]]] = {}
    for run_payload in pending_runs:
        resources = _resources_for_run(args, run_payload=run_payload)
        buckets.setdefault(resources, []).append(run_payload)
    return list(buckets.items())


def _run_launcher(args: argparse.Namespace) -> int:
    apply_slurm_config(args, command_name="bench", defaults=_SLURM_DEFAULTS)
    for name in ("max_cpu_jobs", "max_gpu_jobs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")

    resolved = bench_run.resolve_pending_runs(
        args.experiment_module,
        studies_override=args.studies,
        methods=args.methods,
        max_runs=args.max_runs,
        rerun=bool(args.rerun),
    )
    counts = dict(resolved["counts"])
    pending_runs = list(resolved["pending_runs"])

    print(
        f"[bench] planned={counts['planned_runs']} pending={counts['pending_runs']} "
        f"skipped={counts['skipped_runs']} blocked={counts['blocked_runs']}"
    )
    if not pending_runs:
        print("[bench] No pending runs.")
        return 0

    buckets = _bucket_pending_runs(args, pending_runs)
    requests: list[ArrayRequest] = []
    for bucket_index, (resources, bucket_runs) in enumerate(buckets):
        max_jobs = args.max_gpu_jobs if resources.gpus_per_node else args.max_cpu_jobs
        requests.append(
            ArrayRequest(
                resources=resources,
                array_concurrency=min(len(bucket_runs), int(max_jobs)),
                log_path=Path(args.log_dir) / f"bucket-{bucket_index}.%A_%a.log",
                tasks=tuple(
                    {"run_payload": run_payload, "resume": not bool(args.rerun)}
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
    add_run_arguments(run_parser)
    run_parser.add_argument("--max-cpu-jobs", type=int, default=None)
    run_parser.add_argument("--max-gpu-jobs", type=int, default=None)
    register_common_slurm_arguments(run_parser)
    run_parser.set_defaults(
        func=_run_launcher,
    )
