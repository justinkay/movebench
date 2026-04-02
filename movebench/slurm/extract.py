from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from movebench import Study
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.features.points import (
    point_store_manifest_path,
    point_store_train_stats_path,
    resolve_sampler_hash,
)
from movebench.layers.config import load_layers_config
from movebench.layers.registry import REGISTRY
from movebench.slurm.launcher import (
    SlurmResources,
    build_srun_command,
    register_common_slurm_arguments,
    run_srun_queue,
)
from movebench.study import normalize_split_keys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LAYERS_CONFIG = _REPO_ROOT / "configs" / "layers" / "all.yaml"
_DEFAULT_SAMPLER_CONFIG = _REPO_ROOT / "configs" / "samplers" / "issf_defaults.yaml"


@dataclass(frozen=True)
class ExtractionRun:
    layers_config_path: str
    sampler_config_path: str
    plan: Any
    sampler: SamplerConfig | dict[str, Any]
    combo_stem: str
    combo_label: str
    log_dir_name: str


def _split_arg_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None

    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            text = item.strip()
            if not text or text in seen:
                continue
            out.append(text)
            seen.add(text)
    return out


def _normalize_split_values(values: list[str] | None) -> list[str] | None:
    resolved = _split_arg_values(values)
    if resolved is None:
        return None
    return normalize_split_keys(resolved)


def _normalize_path(path: str) -> str:
    resolved = Path(str(path)).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return str(resolved)


def _parse_multi_path_arg(value: str, *, flag_name: str) -> list[str]:
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError(
            f"{flag_name} must include at least one path or a comma-separated list of paths."
        )
    return out


def _resolve_layer_plan(layers_config: str):
    return REGISTRY.resolve(load_layers_config(layers_config))


def _combo_stem(layers_config_path: str, sampler_config_path: str) -> str:
    return f"{Path(str(layers_config_path)).stem}__{Path(str(sampler_config_path)).stem}"


def _combo_label(layers_config_path: str, sampler_config_path: str) -> str:
    digest = hashlib.sha1(
        f"{layers_config_path}|{sampler_config_path}".encode("utf-8")
    ).hexdigest()[:8]
    return f"{_combo_stem(layers_config_path, sampler_config_path)}-{digest}"


def _load_extraction_run(
    *, layers_config_path: str, sampler_config_path: str
) -> ExtractionRun:
    normalized_layers = _normalize_path(layers_config_path)
    normalized_sampler = _normalize_path(sampler_config_path)
    combo_stem = _combo_stem(normalized_layers, normalized_sampler)
    return ExtractionRun(
        layers_config_path=normalized_layers,
        sampler_config_path=normalized_sampler,
        plan=_resolve_layer_plan(normalized_layers),
        sampler=load_sampler_config(normalized_sampler),
        combo_stem=combo_stem,
        combo_label=_combo_label(normalized_layers, normalized_sampler),
        log_dir_name=combo_stem,
    )


def _finalize_log_dir_names(runs: list[ExtractionRun]) -> list[ExtractionRun]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.combo_stem] = counts.get(run.combo_stem, 0) + 1

    out: list[ExtractionRun] = []
    for run in runs:
        log_dir_name = run.combo_stem
        if counts[run.combo_stem] > 1:
            log_dir_name = run.combo_label
        out.append(replace(run, log_dir_name=log_dir_name))
    return out


def _resolved_studies(values: list[str] | None) -> list[str]:
    studies = _split_arg_values(values)
    if not studies:
        raise ValueError("--studies is required unless worker mode is active")
    return studies


def _default_splits(study: Study) -> list[str]:
    return [str(split) for split in study.split_inventory()["split_keys"]]


def _uses_gee_layers(plan: Any) -> bool:
    for layer in getattr(plan, "layers_in_order", []):
        if getattr(REGISTRY.get(str(layer)), "source", None) in {"gee_chip", "gee_tile"}:
            return True
    return False


def _rasters_root_for(plan: Any, *, study_dir: Path) -> Path:
    return Path(study_dir) / "rasters" / str(plan.rasters_hash)


def _is_complete(
    *,
    study_dir: Path,
    run: ExtractionRun,
    splits: list[str],
) -> tuple[bool, str]:
    sampler_hash = resolve_sampler_hash(run.sampler)
    for split in splits:
        manifest_path = point_store_manifest_path(
            study_dir=Path(study_dir),
            split=str(split),
            rasters_hash=str(run.plan.rasters_hash),
            sampler_hash=sampler_hash,
        )
        if not manifest_path.exists():
            return False, sampler_hash
    stats_path = point_store_train_stats_path(
        study_dir=Path(study_dir),
        rasters_hash=str(run.plan.rasters_hash),
        sampler_hash=sampler_hash,
    )
    return stats_path.exists(), sampler_hash


def _build_runs(args: argparse.Namespace) -> list[ExtractionRun]:
    raw_layers = args.layers_configs or [[str(_DEFAULT_LAYERS_CONFIG)]]
    raw_samplers = args.sampler_configs or [[str(_DEFAULT_SAMPLER_CONFIG)]]

    layers_configs: list[str] = []
    seen_layers: set[str] = set()
    for group in raw_layers:
        for config_path in group:
            normalized = _normalize_path(config_path)
            if normalized in seen_layers:
                continue
            seen_layers.add(normalized)
            layers_configs.append(normalized)

    sampler_configs: list[str] = []
    seen_samplers: set[str] = set()
    for group in raw_samplers:
        for config_path in group:
            normalized = _normalize_path(config_path)
            if normalized in seen_samplers:
                continue
            seen_samplers.add(normalized)
            sampler_configs.append(normalized)

    return _finalize_log_dir_names(
        [
            _load_extraction_run(
                layers_config_path=layers_config_path,
                sampler_config_path=sampler_config_path,
            )
            for layers_config_path in layers_configs
            for sampler_config_path in sampler_configs
        ]
    )


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
    return worker_args


def _log_path_for_run(*, args: argparse.Namespace, study_id: str, run: ExtractionRun) -> Path:
    base_log_dir = Path(args.log_dir)
    return base_log_dir / run.log_dir_name / f"{study_id}.%j.log"


def _run_worker(args: argparse.Namespace) -> int:
    if not args.worker_study or not args.worker_layers_config or not args.worker_sampler_config:
        raise ValueError(
            "Worker mode requires --worker-study, --worker-layers-config, and --worker-sampler-config"
        )

    run = _load_extraction_run(
        layers_config_path=str(args.worker_layers_config),
        sampler_config_path=str(args.worker_sampler_config),
    )
    study = Study(str(args.worker_study))
    if _uses_gee_layers(run.plan):
        rasters_root = _rasters_root_for(run.plan, study_dir=Path(study.study_dir))
        if not rasters_root.exists():
            print(
                f"[extract] ERROR: missing raster cache root for study={args.worker_study}: {rasters_root}",
                file=sys.stderr,
            )
            return 2

    requested_splits = list(args.splits) if args.splits else None
    splits_for_status = (
        requested_splits
        if requested_splits is not None
        else _default_splits(study)
    )
    done, sampler_hash = _is_complete(
        study_dir=Path(study.study_dir),
        run=run,
        splits=splits_for_status,
    )
    if not done:
        result = study.extract_point_features(
            layers_config=str(run.layers_config_path),
            sampler_config=run.sampler,
            splits=requested_splits,
            n_workers=int(args.n_workers),
            skip_existing=True,
        )
        sampler_hash = str(result["sampler_hash"])
    study.compute_normalization_vals(
        from_="points",
        layers_config=str(run.layers_config_path),
        sampler_config=run.sampler,
        split="train",
        skip_existing=True,
    )
    print(
        f"[extract] OK study_id={args.worker_study} "
        f"layers={run.layers_config_path} sampler={run.sampler_config_path} "
        f"sampler_hash={sampler_hash}"
    )
    return 0


def _run_launcher(args: argparse.Namespace) -> int:
    studies = _resolved_studies(args.studies)
    runs = _build_runs(args)

    requested: list[tuple[str, ExtractionRun]] = []
    for study_id in studies:
        for run in runs:
            requested.append((str(study_id), run))

    queued_commands: list[list[str]] = []
    blocked: list[tuple[str, str]] = []
    skipped = 0
    resources = SlurmResources.from_args(args)

    for study_id, run in requested:
        study = Study(str(study_id))
        resolved_splits = (
            list(args.splits)
            if args.splits is not None
            else _default_splits(study)
        )
        complete, _ = _is_complete(
            study_dir=Path(study.study_dir),
            run=run,
            splits=resolved_splits,
        )
        if complete:
            skipped += 1
            continue
        if _uses_gee_layers(run.plan):
            rasters_root = _rasters_root_for(run.plan, study_dir=Path(study.study_dir))
            if not rasters_root.exists():
                blocked.append(
                    (
                        f"study={study_id} layers={run.layers_config_path} sampler={run.sampler_config_path}",
                        f"missing_rasters_root={rasters_root}",
                    )
                )
                continue

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


def _run(args: argparse.Namespace) -> int:
    args.studies = _split_arg_values(args.studies)
    args.splits = _normalize_split_values(args.splits)
    if args.worker_study:
        return _run_worker(args)
    return _run_launcher(args)


def register(subparsers) -> None:
    parser = subparsers.add_parser(
        "extract",
        help="Submit extraction jobs through Slurm.",
        description="Submit extraction jobs through Slurm.",
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Study ids. Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--layers-config",
        dest="layers_configs",
        action="append",
        type=lambda value: _parse_multi_path_arg(value, flag_name="--layers-config"),
        default=None,
        help="Layer config path(s). Accepts repeated values or comma-separated lists.",
    )
    parser.add_argument(
        "--sampler-config",
        dest="sampler_configs",
        action="append",
        type=lambda value: _parse_multi_path_arg(value, flag_name="--sampler-config"),
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
    parser.add_argument("--worker-study", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-layers-config", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-sampler-config", default=None, help=argparse.SUPPRESS)
    register_common_slurm_arguments(parser)
    parser.set_defaults(
        func=_run,
        log_dir=Path("results/slurm/extract_points"),
        cpus_per_task=16,
    )
