from __future__ import annotations

import argparse
from pathlib import Path

from movebench import pack as pack_runtime
from movebench.slurm.launcher import (
    ArrayRequest,
    SlurmResources,
    apply_slurm_config,
    register_common_slurm_arguments,
    run_sbatch_queue,
)

_SLURM_DEFAULTS = {
    "cpus_per_task": 1,
    "gpus_per_node": 0,
    "max_concurrent_jobs": 32,
    "polling_interval": 5.0,
    "log_dir": Path("results/slurm/pack"),
}


def _make_pack_kwargs(
    *,
    studies: list[str],
    timescales: list[str],
    splits: list[str] | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "studies": studies,
        "timescales": timescales,
        "splits": splits,
    }
    if args.samples_per_shard is not None:
        kwargs["samples_per_shard"] = args.samples_per_shard
    if args.spatial_dtype is not None:
        kwargs["spatial_dtype"] = args.spatial_dtype
    if args.target_dtype is not None:
        kwargs["target_dtype"] = args.target_dtype
    if args.compression is not None:
        kwargs["compression"] = args.compression
    if int(args.build_workers) != 16:
        kwargs["build_workers"] = int(args.build_workers)
    if bool(args.overwrite):
        kwargs["overwrite"] = True
    kwargs["resume"] = bool(args.resume)
    if int(args.seed):
        kwargs["seed"] = int(args.seed)
    return kwargs


def _worker_args_for_job(args: argparse.Namespace, study_id: str, timescale_key: str) -> list[str]:
    worker_args = [
        "slurm",
        "pack",
        "--worker-study",
        str(study_id),
        "--worker-timescale",
        str(timescale_key),
    ]
    if args.splits is not None:
        worker_args.extend(["--splits", *list(args.splits)])
    if args.samples_per_shard is not None:
        worker_args.extend(["--samples-per-shard", str(args.samples_per_shard)])
    if args.spatial_dtype is not None:
        worker_args.extend(["--spatial-dtype", str(args.spatial_dtype)])
    if args.target_dtype is not None:
        worker_args.extend(["--target-dtype", str(args.target_dtype)])
    if args.compression is not None:
        worker_args.extend(["--compression", str(args.compression)])
    if int(args.build_workers) != 16:
        worker_args.extend(["--build-workers", str(int(args.build_workers))])
    if bool(args.overwrite):
        worker_args.append("--overwrite")
    worker_args.append("--resume" if bool(args.resume) else "--no-resume")
    if int(args.seed):
        worker_args.extend(["--seed", str(int(args.seed))])
    return worker_args


def _run_worker(args: argparse.Namespace) -> int:
    if not args.worker_study or not args.worker_timescale:
        raise ValueError("Worker mode requires --worker-study and --worker-timescale")

    jobs = pack_runtime.resolve_pack_jobs(
        **_make_pack_kwargs(
            studies=[str(args.worker_study)],
            timescales=[str(args.worker_timescale)],
            splits=args.splits,
            args=args,
        )
    )
    if not jobs:
        print("[pack] No pending study/timescale jobs.")
        return 0

    for job in jobs:
        print(
            f"[pack] study={job.study_id} timescale={job.timescale_key} "
            f"splits={','.join(job.split_keys)}"
        )
        pack_runtime.execute_pack_job(job)
    return 0


def _run_launcher(args: argparse.Namespace) -> int:
    if not args.studies or not args.timescales:
        raise ValueError("--studies and --timescales are required unless worker mode is active")

    jobs = pack_runtime.resolve_pack_jobs(
        **_make_pack_kwargs(
            studies=list(args.studies),
            timescales=list(args.timescales),
            splits=args.splits,
            args=args,
        )
    )
    resources = SlurmResources.from_args(args)
    requests = []
    if jobs:
        requests.append(
            ArrayRequest(
                resources=resources,
                array_concurrency=int(args.max_concurrent_jobs),
                log_path=Path(args.log_dir) / "pack.%A_%a.log",
                worker_args_list=tuple(
                    tuple(_worker_args_for_job(args, job.study_id, job.timescale_key))
                    for job in jobs
                ),
                manifest_dir=Path(args.log_dir) / "manifests",
                manifest_stem="pack",
            )
        )
    if not jobs:
        print("[pack] No pending study/timescale jobs.")
    return run_sbatch_queue(
        label="pack",
        requests=requests,
        polling_interval=float(args.polling_interval),
        dry_run=bool(args.dry_run),
    )


def _run(args: argparse.Namespace) -> int:
    pack_runtime.normalize_cli_values(args)
    apply_slurm_config(args, command_name="pack", defaults=_SLURM_DEFAULTS)
    if args.worker_study:
        return _run_worker(args)
    return _run_launcher(args)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Submit pack jobs through Slurm.",
        description="Submit pack jobs through Slurm.",
    )
    pack_runtime.register_pack_arguments(parser)
    for action in parser._actions:
        if action.dest in {"studies", "timescales"}:
            action.required = False
    parser.add_argument("--worker-study", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-timescale", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=_run)
