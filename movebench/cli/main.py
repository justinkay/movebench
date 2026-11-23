from __future__ import annotations

import argparse

from movebench.cli import bench, download, extract, ingest, pack, slurm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movebench", description="MoveBench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download.register(subparsers)
    extract.register(subparsers)
    ingest.register(subparsers)
    pack.register(subparsers)
    slurm.register(subparsers)
    bench.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
