from __future__ import annotations
import argparse
from pathlib import Path
import sys

from movebench.cli.extract import add_arguments, normalize_split_values
from movebench.cli.slurm.common import (
    apply_slurm_config,
    register_common_slurm_arguments,
    resources_from_args,
)
from movebench.extract import (
    ExtractionRun,
    build_extraction_runs,
    check_extraction_inputs,
    default_splits,
)
from movebench.slurm.launcher import (
    ArrayRequest,
    run_sbatch_queue,
)
from movebench.study import Study, resolve_study_selection

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
    add_arguments(parser)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    register_common_slurm_arguments(parser)
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    splits = normalize_split_values(args.splits)
    apply_slurm_config(args, command_name="extract", defaults=_SLURM_DEFAULTS)
    studies = resolve_study_selection(args.studies)
    runs = build_extraction_runs(
        layers_configs=args.layers_configs,
        sampler_configs=args.sampler_configs,
    )
    requested: list[tuple[str, ExtractionRun]] = [
        (str(study_id), run) for study_id in studies for run in runs
    ]

    queued_worker_args: list[tuple[str, ...]] = []
    blocked: list[tuple[str, str]] = []
    skipped = 0
    resources = resources_from_args(args)

    for study_id, run in requested:
        study = Study(str(study_id))
        resolved_splits = default_splits(study) if splits is None else splits
        try:
            complete, _, _ = check_extraction_inputs(
                study=study,
                run=run,
                splits=resolved_splits,
                skip_existing=not args.no_skip_existing,
                allow_known_download_failures=args.allow_known_download_failures,
            )
        except ValueError as exc:
            blocked.append(
                (
                    f"study={study_id} layers={run.layers_config_path} "
                    f"sampler={run.sampler_config_path}",
                    str(exc),
                )
            )
            continue
        if complete:
            skipped += 1
            continue
        queued_worker_args.append(
            tuple(_worker_args_for_run(args=args, study_id=study_id, run=run))
        )

    print(
        f"[extract] combos={len(runs)} studies={len(studies)} "
        f"requested_pairs={len(requested)} queued={len(queued_worker_args)} "
        f"skipped={skipped} blocked={len(blocked)}"
    )
    for label, reason in blocked:
        print(f"[extract] BLOCKED {label} {reason}", file=sys.stderr)

    requests = []
    if queued_worker_args:
        requests.append(
            ArrayRequest(
                resources=resources,
                array_concurrency=int(args.max_concurrent_jobs),
                log_path=Path(args.log_dir) / "extract.%A_%a.log",
                worker_args_list=tuple(queued_worker_args),
                manifest_dir=Path(args.log_dir) / "manifests",
                manifest_stem="extract",
            )
        )

    rc = run_sbatch_queue(
        label="extract",
        requests=requests,
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
        "extract",
        "--studies",
        str(study_id),
        "--layers-config",
        str(run.layers_config_path),
        "--sampler-config",
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
