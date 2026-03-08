from __future__ import annotations

import argparse
import sys

from movebench.cli import download
from movebench.cli import extract
from movebench.cli import ingest
from movebench.cli import sample_points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movebench", description="MoveBench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download.register(subparsers)
    extract.register(subparsers)
    ingest.register(subparsers)
    sample_points.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
