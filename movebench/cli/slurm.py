from __future__ import annotations

from movebench.slurm import bench, extract, pack


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "slurm",
        help="Submit MoveBench workflows through Slurm.",
        description="Submit MoveBench workflows through Slurm.",
    )
    slurm_subparsers = parser.add_subparsers(dest="slurm_command", required=True)
    extract.register(slurm_subparsers)
    pack.register(slurm_subparsers)
    bench.register(slurm_subparsers)
