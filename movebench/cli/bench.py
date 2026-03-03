from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "bench",
        help="Run MoveBench benchmarks from a YAML config.",
        description="Run MoveBench benchmarks from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmarks/all.yaml",
        help="Benchmark YAML config path.",
    )
    g = parser.add_mutually_exclusive_group(required=False)
    g.add_argument("--study", type=str, default=None, help="Run a single study id.")
    g.add_argument("--studies", type=str, default=None, help="Comma-separated study ids, or 'all'.")
    parser.add_argument("--resume", action="store_true", help="Skip methods already present in per-study outputs.")
    parser.add_argument(
        "--no-setup",
        action="store_true",
        help="Fail if rasters/point stores/normalization are missing (do not download/extract).",
    )
    parser.add_argument("--print-studies", action="store_true", help="Print resolved studies (one per line) and exit.")
    parser.set_defaults(func=run)


def _parse_studies_arg(val: str) -> list[str] | None:
    v = str(val).strip()
    if not v:
        return None
    if v.lower() == "all":
        return None
    return [s.strip() for s in v.split(",") if s.strip()]


def run(args: argparse.Namespace) -> int:
    from movebench.benchmark import load_config, main, resolve_studies

    studies_override: list[str] | None = None
    if args.study:
        studies_override = [str(args.study)]
    elif args.studies:
        override = _parse_studies_arg(str(args.studies))
        if override is None:
            studies_override = resolve_studies({**load_config(str(args.config)), "studies": "all"})
        else:
            studies_override = override

    if bool(args.print_studies):
        if studies_override is not None:
            studies = studies_override
        else:
            cfg = load_config(str(args.config))
            studies = resolve_studies(cfg)
        for s in studies:
            print(s)
        return 0

    main(
        config_path=str(args.config),
        resume=bool(args.resume),
        allow_setup=not bool(args.no_setup),
        studies=studies_override,
    )
    return 0
