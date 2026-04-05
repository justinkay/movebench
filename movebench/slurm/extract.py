from __future__ import annotations

import argparse
import sys
from pathlib import Path

from movebench.download_reports import load_chip_request_status_for_extract
from movebench.extract import (
    ExtractionRun,
    build_extraction_runs,
    default_splits,
    describe_known_download_failure,
    is_extract_complete,
    normalize_split_values,
    parse_multi_path_arg,
    rasters_root_for,
    resolve_requested_studies,
    run_extract_job,
    split_cli_values,
    uses_gee_layers,
)
from movebench.slurm.launcher import (
    SlurmResources,
    apply_slurm_config,
    build_srun_command,
    register_common_slurm_arguments,
    run_srun_queue,
)
from movebench.study import Study

_SLURM_DEFAULTS = {
    "cpus_per_task": 16,
    "gpus_per_node": 0,
    "max_concurrent_jobs": 32,
    "polling_interval": 5.0,
    "log_dir": Path("results/slurm/extract_points"),
}


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Submit extraction jobs through Slurm.",
        description="Submit extraction jobs through Slurm.",
    )
    _register_arguments(parser)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Submit extraction jobs through Slurm.",
    )
    _register_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def _run(args: argparse.Namespace) -> int:
    args.splits = normalize_split_values(args.splits)
    apply_slurm_config(args, command_name="extract", defaults=_SLURM_DEFAULTS)
    if args.worker_study:
        return _run_worker(args)
    return _run_launcher(args)


def _run_worker(args: argparse.Namespace) -> int:
    if not args.worker_study or not args.worker_layers_config or not args.worker_sampler_config:
        raise ValueError(
            "Worker mode requires --worker-study, --worker-layers-config, and --worker-sampler-config"
        )
    runs = build_extraction_runs(
        layers_configs=[[str(args.worker_layers_config)]],
        sampler_configs=[[str(args.worker_sampler_config)]],
    )
    if len(runs) != 1:
        raise ValueError("Worker mode must resolve exactly one extraction run")
    return run_extract_job(
        study_id=str(args.worker_study),
        run=runs[0],
        n_workers=int(args.n_workers),
        splits=args.splits,
        skip_existing=not bool(args.no_skip_existing),
        allow_known_download_failures=bool(args.allow_known_download_failures),
    )


def _run_launcher(args: argparse.Namespace) -> int:
    studies = resolve_requested_studies(_collapse_cli_values(args.studies))
    runs = build_extraction_runs(
        layers_configs=args.layers_configs,
        sampler_configs=args.sampler_configs,
    )
    requested: list[tuple[str, ExtractionRun]] = [
        (str(study_id), run) for study_id in studies for run in runs
    ]

    queued_commands: list[list[str]] = []
    blocked: list[tuple[str, str]] = []
    skipped = 0
    resources = SlurmResources.from_args(args)

    for study_id, run in requested:
        study = Study(str(study_id))
        resolved_splits = args.splits or default_splits(study)
        if not args.no_skip_existing:
            complete, _ = is_extract_complete(
                study_dir=Path(study.study_dir),
                run=run,
                splits=resolved_splits,
            )
            if complete:
                skipped += 1
                continue
        if uses_gee_layers(run.plan):
            rasters_root = rasters_root_for(run.plan, study_dir=Path(study.study_dir))
            if not rasters_root.exists():
                blocked.append(
                    (
                        f"study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path}",
                        f"missing_rasters_root={rasters_root}",
                    )
                )
                continue
            request_status = load_chip_request_status_for_extract(
                study=study,
                plan=run.plan,
                splits=resolved_splits,
            )
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and not bool(args.allow_known_download_failures)
            ):
                blocked.append(
                    (
                        f"study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path}",
                        describe_known_download_failure(request_status),
                    )
                )
                continue
            if (
                isinstance(request_status, dict)
                and str(request_status.get("status")) == "failed"
                and bool(args.allow_known_download_failures)
            ):
                print(
                    "[extract] WARNING: proceeding despite known download failures for "
                    f"study={study_id} layers={run.layers_config_path} "
                    f"sampler={run.sampler_config_path} "
                    f"{describe_known_download_failure(request_status)}",
                    file=sys.stderr,
                )

        log_path = _log_path_for_run(args=args, study_id=study_id, run=run)
        if not args.dry_run:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        queued_commands.append(
            build_srun_command(
                resources=resources,
                worker_args=_worker_args_for_run(args=args, study_id=study_id, run=run),
                log_path=log_path,
            )
        )

    print(
        f"[extract] combos={len(runs)} studies={len(studies)} "
        f"requested_pairs={len(requested)} queued={len(queued_commands)} "
        f"skipped={skipped} blocked={len(blocked)}"
    )
    for label, reason in blocked:
        print(f"[extract] BLOCKED {label} {reason}", file=sys.stderr)

    rc = run_srun_queue(
        label="extract",
        commands=queued_commands,
        max_concurrent_jobs=int(args.max_concurrent_jobs),
        polling_interval=float(args.polling_interval),
        dry_run=bool(args.dry_run),
    )
    return 2 if blocked else rc


def _worker_args_for_run(
    *,
    args: argparse.Namespace,
    study_id: str,
    run: ExtractionRun,
) -> list[str]:
    worker_args = [
        "slurm",
        "extract",
        "--worker-study",
        str(study_id),
        "--worker-layers-config",
        str(run.layers_config_path),
        "--worker-sampler-config",
        str(run.sampler_config_path),
        "--n-workers",
        str(int(args.n_workers)),
    ]
    if args.splits:
        worker_args.extend(["--splits", *list(args.splits)])
    if bool(args.no_skip_existing):
        worker_args.append("--no-skip-existing")
    if bool(args.allow_known_download_failures):
        worker_args.append("--allow-known-download-failures")
    return worker_args


def _log_path_for_run(*, args: argparse.Namespace, study_id: str, run: ExtractionRun) -> Path:
    return Path(args.log_dir) / run.log_dir_name / f"{study_id}.%j.log"


def _collapse_cli_values(values: list[str] | None) -> str | None:
    resolved = split_cli_values(values)
    if not resolved:
        return None
    return ",".join(resolved)


def _register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Study ids or named sets like 'mini'. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--layers-config",
        dest="layers_configs",
        action="append",
        type=lambda value: parse_multi_path_arg(value, flag_name="--layers-config"),
        default=None,
        help="Layer config path(s). Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--sampler-config",
        dest="sampler_configs",
        action="append",
        type=lambda value: parse_multi_path_arg(value, flag_name="--sampler-config"),
        default=None,
        help="Sampler config path(s). Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split keys. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing point stores on disk.",
    )
    parser.add_argument(
        "--allow-known-download-failures",
        action="store_true",
        help="Proceed even if the latest matching chip download report is marked failed.",
    )
    parser.add_argument("--worker-study", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-layers-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sampler-config", default=None, help=argparse.SUPPRESS)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=_run)
