from __future__ import annotations

import argparse

from movebench.slurm.launcher import register_common_slurm_arguments


def _run(_args: argparse.Namespace) -> int:
    print("movebench slurm extract is not implemented yet.")
    return 2


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Submit extraction jobs through Slurm.",
        description="Submit extraction jobs through Slurm.",
    )
    parser.add_argument("--studies", nargs="+", required=True)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=_run)
