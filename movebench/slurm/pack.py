from __future__ import annotations

import argparse

from movebench import pack as pack_runtime
from movebench.slurm.launcher import register_common_slurm_arguments


def _run(_args: argparse.Namespace) -> int:
    print("movebench slurm pack is not implemented yet.")
    return 2


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Submit pack jobs through Slurm.",
        description="Submit pack jobs through Slurm.",
    )
    pack_runtime.register_pack_arguments(parser)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=_run)
