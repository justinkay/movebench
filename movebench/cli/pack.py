from __future__ import annotations

import argparse

from movebench import pack as pack_runtime


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--studies",
        nargs="+",
        required=True,
        help="Study ids or named sets like 'mini'. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--timescales",
        nargs="+",
        required=True,
        help="Timescale keys such as 1d. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split keys. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument("--layers-config", type=str, default=None)
    parser.add_argument("--samples-per-shard", type=int, default=None)
    parser.add_argument(
        "--spatial-dtype",
        type=str,
        default=None,
        choices=("float16", "float32"),
    )
    parser.add_argument(
        "--target-dtype",
        type=str,
        default=None,
        choices=("uint8", "float16", "float32"),
    )
    parser.add_argument("--compression", type=str, default=None, choices=("none", "gz"))
    parser.add_argument("--build-workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=0)


def pack_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "splits": args.splits,
        "layers_config": args.layers_config,
        "samples_per_shard": args.samples_per_shard,
        "spatial_dtype": args.spatial_dtype,
        "target_dtype": args.target_dtype,
        "compression": args.compression,
        "build_workers": args.build_workers,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "seed": args.seed,
    }


def run(args: argparse.Namespace) -> int:
    return pack_runtime.run_local_pack(
        studies=args.studies, timescales=args.timescales, **pack_options(args)
    )


def register(subparsers) -> None:
    parser = subparsers.add_parser("pack", help="Build deepSSF cache shards locally.")
    add_arguments(parser)
    parser.set_defaults(func=run)
