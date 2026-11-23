from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from movebench import Study
from movebench.bench.config import resolve_timescale
from movebench.datasets.deepssf_wds_cache import (
    build_deepssf_wds_cache,
    cache_dir_for_study,
    deepssf_wds_cache_hash,
    read_deepssf_wds_manifest_if_complete,
)
from movebench.features.points import load_point_store_manifest, point_store_train_stats_path
from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.study import normalize_split_keys, resolve_study_selection, selection_tokens


@dataclass(frozen=True)
class PackJob:
    study_id: str
    study: Any
    timescale_key: str
    timescale: dict[str, object]
    split_keys: tuple[str, ...]
    options: dict[str, object]


def _expected_cache_hash(
    *,
    timescale: dict[str, object],
    split: str,
    options: dict[str, object],
) -> str:
    return deepssf_wds_cache_hash(
        rasters_hash=str(timescale["rasters_hash"]),
        band_ids=list(timescale["all_band_ids"]),
        split=str(split),
        output_hw=tuple(timescale["output_hw"]),
        normalize=bool(timescale["normalize"]),
        normalization_sampler_hash=(
            str(timescale["sampler_hash"]) if bool(timescale["normalize"]) else None
        ),
        samples_per_shard=int(options["samples_per_shard"]),
        spatial_dtype=str(options["spatial_dtype"]),
        target_dtype=str(options["target_dtype"]),
        compression=str(options["compression"]),
    )


def _resolve_pack_timescale(
    timescale_key: str,
    *,
    study_id: str,
    layers_config: str | None,
) -> dict[str, object]:
    timescale = dict(resolve_timescale(str(timescale_key), study_id=str(study_id)))
    if layers_config is None:
        return timescale
    plan = REGISTRY.resolve(load_layers_config(str(layers_config)))
    timescale["layers_config"] = str(layers_config)
    timescale["plan"] = plan
    timescale["rasters_hash"] = str(plan.rasters_hash)
    timescale["all_band_ids"] = plan.band_ids_in_order()
    return timescale


def resolve_pack_jobs(
    *,
    studies: list[str],
    timescales: list[str],
    splits: list[str] | None,
    layers_config: str | None = None,
    samples_per_shard: int | None = None,
    spatial_dtype: str | None = None,
    target_dtype: str | None = None,
    compression: str | None = None,
    build_workers: int = 16,
    overwrite: bool = False,
    resume: bool = True,
    seed: int = 0,
) -> list[PackJob]:
    normalized_splits = (
        None if splits is None else normalize_split_keys(selection_tokens(splits))
    )
    cache_options = {
        "samples_per_shard": samples_per_shard,
        "spatial_dtype": spatial_dtype,
        "target_dtype": target_dtype,
        "compression": compression,
    }
    jobs: list[PackJob] = []

    for study_id in resolve_study_selection(studies):
        study = Study(str(study_id))
        study_splits = normalized_splits
        if study_splits is None:
            study_splits = normalize_split_keys(
                [str(split) for split in study.split_inventory()["split_keys"]]
            ) or []
        for timescale_key in selection_tokens(timescales):
            timescale = _resolve_pack_timescale(
                str(timescale_key),
                study_id=str(study_id),
                layers_config=layers_config,
            )
            options = {
                key: timescale[key] if value is None else value
                for key, value in cache_options.items()
            }
            options.update(
                build_workers=int(build_workers),
                overwrite=bool(overwrite),
                resume=bool(resume),
                seed=int(seed),
            )
            pending_splits: list[str] = []
            for split in study_splits:
                cache_hash = _expected_cache_hash(
                    timescale=timescale,
                    split=str(split),
                    options=options,
                )
                cache_dir = cache_dir_for_study(
                    study_dir=Path(study.study_dir),
                    split=str(split),
                    cache_hash=cache_hash,
                )
                if (
                    overwrite
                    or read_deepssf_wds_manifest_if_complete(cache_dir) is None
                ):
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
    options = job.options
    results: list[tuple[Path, dict[str, Any]]] = []
    for split in job.split_keys:
        cache_dir = cache_dir_for_study(
            study_dir=job.study.study_dir,
            split=split,
            cache_hash=_expected_cache_hash(
                timescale=job.timescale, split=split, options=options
            ),
        )
        manifest = read_deepssf_wds_manifest_if_complete(cache_dir)
        if manifest is not None and not options["overwrite"]:
            results.append((cache_dir, manifest))
            continue
        overwrite = bool(options["overwrite"]) or (cache_dir / "manifest.json").exists()
        ds_kwargs = {
            "split": str(split),
            "layers_config": layers_config,
            "sampler_config": job.timescale["sampler"],
            "output_hw": tuple(job.timescale["output_hw"]),
            "normalize": normalize,
        }
        try:
            _, point_store_manifest = load_point_store_manifest(
                study_dir=Path(job.study.study_dir),
                split=str(split),
                rasters_hash=str(job.timescale["rasters_hash"]),
                sampler_hash=str(job.timescale["sampler_hash"]),
            )
        except FileNotFoundError:
            point_store_manifest = None
        if point_store_manifest is not None:
            selected_splits = point_store_manifest.extraction.get("selected_splits")
            if isinstance(selected_splits, list) and selected_splits:
                ds_kwargs["selected_splits"] = [
                    str(split_key) for split_key in selected_splits
                ]
        manifest = build_deepssf_wds_cache(
            job.study.deepssf_dataset(**ds_kwargs),
            cache_dir,
            samples_per_shard=int(options["samples_per_shard"]),
            spatial_dtype=str(options["spatial_dtype"]),
            target_dtype=str(options["target_dtype"]),
            compression=str(options["compression"]),
            seed=int(options["seed"]),
            num_workers=int(options["build_workers"]),
            overwrite=overwrite,
            resume=bool(options["resume"]) and not overwrite,
        )
        results.append((cache_dir, manifest))
    return results


def run_local_pack(
    *,
    studies: list[str],
    timescales: list[str],
    splits: list[str] | None,
    layers_config: str | None = None,
    samples_per_shard: int | None = None,
    spatial_dtype: str | None = None,
    target_dtype: str | None = None,
    compression: str | None = None,
    build_workers: int = 16,
    overwrite: bool = False,
    resume: bool = True,
    seed: int = 0,
) -> int:
    jobs = resolve_pack_jobs(
        studies=studies,
        timescales=timescales,
        splits=splits,
        layers_config=layers_config,
        samples_per_shard=samples_per_shard,
        spatial_dtype=spatial_dtype,
        target_dtype=target_dtype,
        compression=compression,
        build_workers=build_workers,
        overwrite=overwrite,
        resume=resume,
        seed=seed,
    )
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
