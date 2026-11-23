from __future__ import annotations

import argparse
from pathlib import Path

from movebench import pack as pack_runtime
from movebench.cli.pack import add_arguments, pack_options
from movebench.cli.slurm.common import (
    apply_slurm_config,
    register_common_slurm_arguments,
    resources_from_args,
)
from movebench.slurm.launcher import (
    ArrayRequest,
    run_sbatch_queue,
)

_SLURM_DEFAULTS = {
    "cpus_per_task": 1,
    "gpus_per_node": 0,
    "max_concurrent_jobs": 32,
    "polling_interval": 5.0,
    "log_dir": Path("results/slurm/pack"),
}


def _worker_args_for_job(
    args: argparse.Namespace, study_id: str, timescale_key: str
) -> list[str]:
    worker_args = ["pack", "--studies", study_id, "--timescales", timescale_key]
    for name, value in pack_options(args).items():
        flag = "--" + name.replace("_", "-")
        if name == "resume":
            worker_args.append("--resume" if value else "--no-resume")
        elif isinstance(value, bool):
            if value:
                worker_args.append(flag)
        elif value is not None:
            worker_args.append(flag)
            values = value if isinstance(value, list) else [value]
            worker_args.extend(str(item) for item in values)
    return worker_args


def run(args: argparse.Namespace) -> int:
    apply_slurm_config(args, command_name="pack", defaults=_SLURM_DEFAULTS)

    jobs = pack_runtime.resolve_pack_jobs(
        studies=args.studies, timescales=args.timescales, **pack_options(args)
    )
    resources = resources_from_args(args)
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


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Submit pack jobs through Slurm.",
        description="Submit pack jobs through Slurm.",
    )
    add_arguments(parser)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=run)
