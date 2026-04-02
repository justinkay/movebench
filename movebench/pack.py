from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from movebench import Study
from movebench.bench.config import deepssf_cache_hash_for_split, resolve_timescale
from movebench.datasets.deepssf_wds_cache import (
    cache_dir_for_study,
    deepssf_wds_cache_hash,
    ensure_deepssf_wds_cache_split,
    read_deepssf_wds_manifest_if_complete,
)
from movebench.features.points import (
    load_point_store_manifest,
    point_store_train_stats_path,
)
from movebench.study import normalize_split_keys


@dataclass(frozen=True)
class PackJob:
    study_id: str
    study: Any
    timescale_key: str
    timescale: dict[str, object]
    split_keys: tuple[str, ...]
    options: dict[str, object] | None = None


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
    normalized = _split_arg_values(values)
    if normalized is None:
        return None
    return normalize_split_keys(normalized)


def register_pack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--studies",
        nargs="+",
        required=True,
        help="Study ids. Accepts repeated values or comma-separated lists.",
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
    parser.add_argument("--build-workers", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)


def normalize_cli_values(args: argparse.Namespace) -> None:
    args.studies = _split_arg_values(args.studies) or []
    args.timescales = _split_arg_values(args.timescales) or []
    args.splits = _normalize_split_values(args.splits)


def _resolve_study_ids(studies: list[str]) -> list[str]:
    study_ids = _split_arg_values(studies) or []
    if study_ids == ["all"]:
        return Study.list_available()
    return study_ids


def _build_options(
    *,
    samples_per_shard: int | None,
    spatial_dtype: str | None,
    target_dtype: str | None,
    compression: str | None,
    build_workers: int,
    overwrite: bool,
    resume: bool,
    seed: int,
) -> dict[str, object]:
    return {
        "samples_per_shard": samples_per_shard,
        "spatial_dtype": spatial_dtype,
        "target_dtype": target_dtype,
        "compression": compression,
        "build_workers": int(build_workers),
        "overwrite": bool(overwrite),
        "resume": bool(resume),
        "seed": int(seed),
    }


def _job_option(job: PackJob, key: str, fallback: object) -> object:
    if not job.options:
        return fallback
    value = job.options.get(key)
    return fallback if value is None else value


def _expected_cache_hash(
    *,
    timescale_key: str,
    timescale: dict[str, object],
    split: str,
    options: dict[str, object],
) -> str:
    samples_per_shard = options.get("samples_per_shard")
    spatial_dtype = options.get("spatial_dtype")
    target_dtype = options.get("target_dtype")
    compression = options.get("compression")
    if (
        samples_per_shard is None
        and spatial_dtype is None
        and target_dtype is None
        and compression is None
    ):
        return deepssf_cache_hash_for_split(timescale_key, split)
    return deepssf_wds_cache_hash(
        rasters_hash=str(timescale["rasters_hash"]),
        band_ids=list(timescale["all_band_ids"]),
        split=str(split),
        output_hw=tuple(timescale["output_hw"]),
        normalize=bool(timescale["normalize"]),
        normalization_sampler_hash=(
            str(timescale["sampler_hash"]) if bool(timescale["normalize"]) else None
        ),
        samples_per_shard=int(samples_per_shard or int(timescale["samples_per_shard"])),
        spatial_dtype=str(spatial_dtype or str(timescale["spatial_dtype"])),
        target_dtype=str(target_dtype or str(timescale["target_dtype"])),
        compression=str(compression or str(timescale["compression"])),
    )


def _cache_split_complete(cache_dir: Path) -> bool:
    return read_deepssf_wds_manifest_if_complete(cache_dir) is not None


def resolve_pack_jobs(
    *,
    studies: list[str],
    timescales: list[str],
    splits: list[str] | None,
    samples_per_shard: int | None = None,
    spatial_dtype: str | None = None,
    target_dtype: str | None = None,
    compression: str | None = None,
    build_workers: int = 0,
    overwrite: bool = False,
    resume: bool = False,
    seed: int = 0,
) -> list[PackJob]:
    normalized_splits = _normalize_split_values(splits)
    options = _build_options(
        samples_per_shard=samples_per_shard,
        spatial_dtype=spatial_dtype,
        target_dtype=target_dtype,
        compression=compression,
        build_workers=build_workers,
        overwrite=overwrite,
        resume=resume,
        seed=seed,
    )
    jobs: list[PackJob] = []

    for study_id in _resolve_study_ids(studies):
        study = Study(str(study_id))
        study_splits = normalized_splits
        if study_splits is None:
            study_splits = _normalize_split_values(
                [str(split) for split in study.split_inventory()["split_keys"]]
            ) or []
        for timescale_key in _split_arg_values(timescales) or []:
            timescale = resolve_timescale(str(timescale_key))
            pending_splits: list[str] = []
            for split in study_splits:
                cache_hash = _expected_cache_hash(
                    timescale_key=str(timescale_key),
                    timescale=timescale,
                    split=str(split),
                    options=options,
                )
                cache_dir = cache_dir_for_study(
                    study_dir=Path(study.study_dir),
                    split=str(split),
                    cache_hash=cache_hash,
                )
                if bool(overwrite) or not _cache_split_complete(Path(cache_dir)):
                    pending_splits.append(str(split))
            if pending_splits:
                jobs.append(
                    PackJob(
                        study_id=str(study_id),
                        study=study,
                        timescale_key=str(timescale_key),
                        timescale=timescale,
                        split_keys=tuple(pending_splits),
                        options=options,
                    )
                )
    return jobs


def _require_pack_inputs(job: PackJob) -> None:
    study_dir = Path(job.study.study_dir)
    rasters_hash = str(job.timescale["rasters_hash"])
    sampler_hash = str(job.timescale["sampler_hash"])
    for split in job.split_keys:
        try:
            load_point_store_manifest(
                study_dir=study_dir,
                split=str(split),
                rasters_hash=rasters_hash,
                sampler_hash=sampler_hash,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Pack prerequisites missing for study={job.study_id} "
                f"timescale={job.timescale_key} split={split}: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"Pack prerequisites invalid for study={job.study_id} "
                f"timescale={job.timescale_key} split={split}: {exc}"
            ) from exc
    if bool(job.timescale["normalize"]):
        stats_path = point_store_train_stats_path(
            study_dir=study_dir,
            rasters_hash=rasters_hash,
            sampler_hash=sampler_hash,
        )
        if not Path(stats_path).exists():
            raise FileNotFoundError(
                f"Missing train normalization stats for study={job.study_id} timescale={job.timescale_key}: {stats_path}"
            )


def execute_pack_job(job: PackJob) -> list[tuple[Path, dict[str, Any]]]:
    _require_pack_inputs(job)
    layers_config = str(job.timescale["layers_config"])
    normalize = bool(job.timescale["normalize"])
    results: list[tuple[Path, dict[str, Any]]] = []
    for split in job.split_keys:
        ds_kwargs = {
            "split": str(split),
            "layers_config": layers_config,
            "sampler_config": job.timescale["sampler"],
            "output_hw": tuple(job.timescale["output_hw"]),
            "normalize": normalize,
        }
        result = ensure_deepssf_wds_cache_split(
            study=job.study,
            split=str(split),
            layers_config=layers_config,
            ds_kwargs=ds_kwargs,
            samples_per_shard=int(
                _job_option(job, "samples_per_shard", int(job.timescale["samples_per_shard"]))
            ),
            spatial_dtype=str(
                _job_option(job, "spatial_dtype", str(job.timescale["spatial_dtype"]))
            ),
            target_dtype=str(
                _job_option(job, "target_dtype", str(job.timescale["target_dtype"]))
            ),
            compression=str(
                _job_option(job, "compression", str(job.timescale["compression"]))
            ),
            seed=int(_job_option(job, "seed", 0)),
            build_workers=int(_job_option(job, "build_workers", 0)),
            build_if_missing=True,
            overwrite=bool(_job_option(job, "overwrite", False)),
            resume=bool(_job_option(job, "resume", False)),
        )
        results.append(result)
    return results


def run_local_pack(
    *,
    studies: list[str],
    timescales: list[str],
    splits: list[str] | None,
    samples_per_shard: int | None = None,
    spatial_dtype: str | None = None,
    target_dtype: str | None = None,
    compression: str | None = None,
    build_workers: int = 0,
    overwrite: bool = False,
    resume: bool = False,
    seed: int = 0,
) -> int:
    normalized_splits = _normalize_split_values(splits)
    resolve_kwargs: dict[str, object] = {
        "studies": studies,
        "timescales": timescales,
        "splits": normalized_splits,
    }
    if samples_per_shard is not None:
        resolve_kwargs["samples_per_shard"] = samples_per_shard
    if spatial_dtype is not None:
        resolve_kwargs["spatial_dtype"] = spatial_dtype
    if target_dtype is not None:
        resolve_kwargs["target_dtype"] = target_dtype
    if compression is not None:
        resolve_kwargs["compression"] = compression
    if build_workers:
        resolve_kwargs["build_workers"] = build_workers
    if overwrite:
        resolve_kwargs["overwrite"] = overwrite
    if resume:
        resolve_kwargs["resume"] = resume
    if seed:
        resolve_kwargs["seed"] = seed

    jobs = resolve_pack_jobs(**resolve_kwargs)
    if not jobs:
        print("[pack] No pending study/timescale jobs.")
        return 0

    print(f"[pack] Processing {len(jobs)} study/timescale jobs locally.")
    for job in jobs:
        print(
            f"[pack] study={job.study_id} timescale={job.timescale_key} "
            f"splits={','.join(job.split_keys)}"
        )
        execute_pack_job(job)
    return 0


def run_local_pack_from_args(args: argparse.Namespace) -> int:
    normalize_cli_values(args)
    return run_local_pack(
        studies=list(args.studies),
        timescales=list(args.timescales),
        splits=args.splits,
        samples_per_shard=args.samples_per_shard,
        spatial_dtype=args.spatial_dtype,
        target_dtype=args.target_dtype,
        compression=args.compression,
        build_workers=int(args.build_workers),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
        seed=int(args.seed),
    )
