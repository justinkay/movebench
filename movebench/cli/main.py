from __future__ import annotations

import argparse
import sys

from movebench.cli import download, extract, ingest, pack, slurm

_BENCH_OPTIONAL_DEPENDENCIES = {"matplotlib"}


def _missing_bench_dependency_root(exc: ModuleNotFoundError) -> str:
    return str(getattr(exc, "name", "") or "").split(".", 1)[0]


def _missing_bench_command(args: argparse.Namespace) -> int:
    missing_root = str(getattr(args, "_missing_bench_dependency", "")).strip()
    if missing_root:
        print(
            f"Benchmark commands require the optional bench dependencies; missing {missing_root!r}. "
            "Install them with: python -m pip install -e .[bench]"
        )
    else:
        print(
            "Benchmark commands require the optional bench dependencies. "
            "Install them with: python -m pip install -e .[bench]"
        )
    return 2


def _register_missing_bench_parser(subparsers, *, missing_root: str) -> None:
    parser = subparsers.add_parser(
        "bench",
        help="Benchmark commands require the optional bench dependencies.",
        description=(
            "Benchmark commands require the optional bench dependencies. "
            "Install them with: python -m pip install -e .[bench]"
        ),
    )
    parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    parser.set_defaults(
        func=_missing_bench_command,
        _missing_bench_dependency=str(missing_root),
    )


def _register_bench_commands(subparsers) -> None:
    try:
        from movebench.bench import run as bench_run
    except ModuleNotFoundError as exc:
        missing_root = _missing_bench_dependency_root(exc)
        if missing_root not in _BENCH_OPTIONAL_DEPENDENCIES:
            raise
        _register_missing_bench_parser(subparsers, missing_root=missing_root)
        return
    bench_run.register(subparsers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="movebench", description="MoveBench CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download.register(subparsers)
    extract.register(subparsers)
    ingest.register(subparsers)
    pack.register(subparsers)
    slurm.register(subparsers)
    _register_bench_commands(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
