from __future__ import annotations

import argparse

from movebench import pack as pack_runtime


def _run(args: argparse.Namespace) -> int:
    return int(pack_runtime.run_local_pack_from_args(args))


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "pack",
        help="Build deepSSF cache shards locally.",
        description="Build deepSSF cache shards locally.",
    )
    pack_runtime.register_pack_arguments(parser)
    parser.set_defaults(func=_run)
