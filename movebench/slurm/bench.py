from __future__ import annotations

import argparse

from movebench.slurm.launcher import register_common_slurm_arguments


def _run(_args: argparse.Namespace) -> int:
    print("movebench slurm bench run is not implemented yet.")
    return 2


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
    register_common_slurm_arguments(run_parser)
    run_parser.set_defaults(func=_run)
