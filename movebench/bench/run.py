from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import shutil

from movebench.datasets.deepssf_gpu import (
    ALPHAEARTH_LAYER,
    AlphaEarthCatalog,
)
from movebench.datasets.deepssf_wds_cache import (
    list_deepssf_wds_shards,
    read_deepssf_wds_manifest,
)
from movebench.features.manifest import read_manifest
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import (
    point_store_dir,
    point_store_manifest_path,
    point_store_train_stats_path,
)
from movebench.layers.plan import LayerPlan
import movebench.paths as paths
from movebench.study import resolve_study_selection, _list_available_studies
from movebench.trajectories import canonicalize_split_key

import movebench.bench.config as bench_config
import movebench.bench.run_state as bench_run_state

PREFERRED_EARLY_STOP_SPLITS = ["test_temporal", "test_individual"]
_SUPPORTED_METHODS = {
    str(method_key)
    for method_key, method_cfg in bench_config.METHODS.items()
    if bool(method_cfg.get("executor"))
}


class _RunInterrupted(BaseException):
    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _log_progress(message: str) -> None:
    print(str(message), flush=True)


def _load_module_from_path(module_path: str | Path):
    path = Path(module_path)
    if not path.exists():
        candidate = paths.repo_root() / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Experiment module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"movebench_bench_{path.stem}",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.resolve()

def _bench_report_module():
    import movebench.bench.report as bench_report

    return bench_report


def _bench_summary_module():
    import movebench.bench.summary as bench_summary

    return bench_summary


def _evaluation_runner_module():
    import movebench.bench.evaluation.runner as evaluation_runner

    return evaluation_runner


def _resolve_studies(
    experiment: dict[str, object],
    studies_override: list[str] | None = None,
) -> list[str]:
    if studies_override:
        return resolve_study_selection(studies_override)
    return resolve_study_selection(experiment.get("studies", "all"))


def _study_split_keys(study_id: str) -> dict[str, list[str] | bool]:
    from movebench.study import Study

    inventory = Study(study_id).split_inventory()
    split_keys = [str(split) for split in inventory["split_keys"]]
    extras = [
        split
        for split in split_keys
        if split not in {"train", *PREFERRED_EARLY_STOP_SPLITS}
    ]
    ordered = [
        split for split in PREFERRED_EARLY_STOP_SPLITS if split in split_keys
    ] + extras
    return {
        "has_train": bool(inventory["has_train"]),
        "held_out_splits": ordered,
    }


def _point_store_state(study_id: str, timescale_key: str, split: str) -> dict[str, object]:
    split = canonicalize_split_key(split)
    timescale = bench_config.resolve_timescale(timescale_key, study_id=study_id)
    manifest_path = point_store_manifest_path(
        study_dir=paths.study_dir(study_id),
        split=str(split),
        rasters_hash=str(timescale["rasters_hash"]),
        sampler_hash=str(timescale["sampler_hash"]),
    )
    if not manifest_path.exists():
        return {
            "available": False,
            "reason": f"missing point store manifest for {split}",
            "manifest_path": manifest_path,
        }
    try:
        manifest = read_manifest(manifest_path)
        if manifest.study_id != study_id or manifest.split != split:
            raise ValueError("manifest metadata mismatch")
    except Exception as exc:
        return {
            "available": False,
            "reason": f"invalid point store manifest for {split}: {exc}",
            "manifest_path": manifest_path,
        }
    if manifest.n_rows == 0:
        return {
            "available": False,
            "reason": f"point store for {split} has zero valid steps",
            "manifest_path": manifest_path,
            "n_rows": 0,
        }
    return {
        "available": True,
        "manifest_path": manifest_path,
        "n_rows": manifest.n_rows,
    }


def _deepssf_cache_state(study_id: str, timescale_key: str, split: str) -> dict[str, object]:
    split = canonicalize_split_key(split)
    cache_hash = bench_config.deepssf_cache_hash_for_split(
        timescale_key,
        split,
        study_id=study_id,
    )
    cache_dir = paths.study_dir(study_id) / "cache" / "deepssf_wds" / str(split) / str(cache_hash)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "available": False,
            "reason": f"missing deepSSF cache manifest for {split}",
            "manifest_path": manifest_path,
        }
    if not list_deepssf_wds_shards(cache_dir):
        return {
            "available": False,
            "reason": f"missing deepSSF shards for {split}",
            "manifest_path": manifest_path,
        }
    try:
        read_deepssf_wds_manifest(manifest_path)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"invalid deepSSF cache manifest for {split}: {exc}",
            "manifest_path": manifest_path,
        }
    return {
        "available": True,
        "manifest_path": manifest_path,
    }


def _alphaearth_tile_state(
    study_id: str,
    timescale_key: str,
) -> dict[str, object]:
    timescale = bench_config.resolve_timescale(timescale_key, study_id=study_id)
    tile_dir = (
        paths.study_dir(study_id)
        / "rasters"
        / str(timescale["rasters_hash"])
        / "tiles"
        / ALPHAEARTH_LAYER
    )
    try:
        catalog = AlphaEarthCatalog.from_tile_directories([(study_id, tile_dir)])
    except Exception as exc:
        return {
            "available": False,
            "reason": f"invalid or missing AlphaEarth tiles: {exc}",
            "tile_dir": tile_dir,
        }
    return {
        "available": True,
        "tile_dir": tile_dir,
        "tile_count": catalog.tile_count,
    }


def _load_deepssf_cache_entry(
    cache_dir: Path,
    *,
    split: str,
) -> tuple[dict[str, object] | None, str | None]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None, f"missing deepSSF cache manifest for {split}"
    if not list_deepssf_wds_shards(cache_dir):
        return None, f"missing deepSSF shards for {split}"
    try:
        manifest = read_deepssf_wds_manifest(manifest_path)
    except Exception as exc:
        return None, f"invalid deepSSF cache manifest for {split}: {exc}"
    if str(manifest.get("sample_identity", "")) != "point_id_start":
        return None, (
            f"deepSSF cache for {split} is missing sample_identity=point_id_start; "
            "rebuild the cache"
        )
    return {
        "cache_dir": cache_dir,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }, None


def _compatible_deepssf_cache_candidates(
    *,
    study_id: str,
    split: str,
    expected_output_hw: tuple[int, int],
    expected_rasters_hash: str,
    requested_band_ids: list[str],
    required_cache_band_ids: list[str] | None = None,
) -> list[dict[str, object]]:
    split = canonicalize_split_key(split)
    split_dir = paths.study_dir(str(study_id)) / "cache" / "deepssf_wds" / str(split)
    if not split_dir.exists():
        return []
    candidates: list[dict[str, object]] = []
    for cache_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        entry, _ = _load_deepssf_cache_entry(cache_dir, split=str(split))
        if entry is None:
            continue
        manifest = dict(entry["manifest"])
        if str(manifest.get("rasters_hash", "")) != str(expected_rasters_hash):
            continue
        manifest_band_ids = manifest.get("band_ids")
        if not isinstance(manifest_band_ids, list) or not manifest_band_ids:
            continue
        cache_band_ids = [str(band_id) for band_id in manifest_band_ids]
        if int(manifest["env_channels"]) != len(cache_band_ids):
            continue
        split_output_hw = tuple(int(v) for v in manifest["output_hw"])
        if split_output_hw != expected_output_hw:
            continue
        if required_cache_band_ids is not None:
            if cache_band_ids != required_cache_band_ids:
                continue
        elif any(band_id not in cache_band_ids for band_id in requested_band_ids):
            continue
        candidates.append(
            {
                **entry,
                "cache_band_ids": cache_band_ids,
                "channel_indices": [
                    cache_band_ids.index(band_id) for band_id in requested_band_ids
                ],
            }
        )
    candidates.sort(
        key=lambda candidate: (
            len(candidate["cache_band_ids"]),
            str(candidate["manifest_path"]),
        )
    )
    return candidates


def _preferred_split(splits: list[str]) -> str | None:
    if not splits:
        return None
    for split in PREFERRED_EARLY_STOP_SPLITS:
        if split in splits:
            return split
    return splits[0]


def _dedupe_reasons(*reason_groups: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for reasons in reason_groups:
        for reason in reasons:
            if reason in seen:
                continue
            out.append(reason)
            seen.add(reason)
    return out


def _study_timescale_readiness(
    study_id: str,
    timescale_key: str,
    *,
    deepssf_data: str = "wds",
    needs_points: bool = True,
) -> dict[str, object]:
    split_info = _study_split_keys(study_id)
    held_out_splits = list(split_info["held_out_splits"])
    if needs_points:
        train_points = _point_store_state(study_id, timescale_key, "train")
        held_out_point_states = {
            split: _point_store_state(study_id, timescale_key, split)
            for split in held_out_splits
        }
    else:
        train_points = {"available": True}
        held_out_point_states = {
            split: {"available": True} for split in held_out_splits
        }
    point_splits_available = [
        split for split, state in held_out_point_states.items() if state["available"]
    ]
    if str(deepssf_data) == "gpu":
        train_cache = _alphaearth_tile_state(study_id, timescale_key)
        held_out_raster_states = {
            split: train_cache for split in held_out_splits
        }
    else:
        train_cache = _deepssf_cache_state(study_id, timescale_key, "train")
        held_out_raster_states = {
            split: _deepssf_cache_state(study_id, timescale_key, split)
            for split in held_out_splits
        }
    raster_splits_available = [
        split
        for split, state in held_out_raster_states.items()
        if state["available"]
    ]

    point_ready = bool(
        split_info["has_train"]
        and train_points["available"]
        and held_out_splits
        and len(point_splits_available) == len(held_out_splits)
    )
    raster_ready = bool(train_cache["available"]) and bool(raster_splits_available)
    fully_runnable = point_ready and raster_ready

    point_blocked_reasons: list[str] = []
    raster_blocked_reasons: list[str] = []
    if not split_info["has_train"]:
        point_blocked_reasons.append("study has no train split")
        raster_blocked_reasons.append("study has no train split")
    if not train_points["available"]:
        point_blocked_reasons.append(str(train_points["reason"]))
    if len(point_splits_available) != len(held_out_splits):
        point_blocked_reasons.append(
            next(
                (
                    str(state["reason"])
                    for state in held_out_point_states.values()
                    if not state["available"]
                ),
                "no held-out point stores available",
            )
        )
    if not train_cache["available"]:
        raster_blocked_reasons.append(str(train_cache["reason"]))
    if not raster_splits_available:
        raster_blocked_reasons.append(
            next(
                (
                    str(state["reason"])
                    for state in held_out_raster_states.values()
                    if not state["available"]
                ),
                (
                    "no held-out AlphaEarth data available"
                    if str(deepssf_data) == "gpu"
                    else "no held-out deepSSF caches available"
                ),
            )
        )

    return {
        "study_id": study_id,
        "timescale": timescale_key,
        "deepssf_data": str(deepssf_data),
        "has_train": bool(split_info["has_train"]),
        "held_out_splits": held_out_splits,
        "available_point_splits": point_splits_available,
        "available_raster_splits": raster_splits_available,
        "early_stop_point_split": _preferred_split(point_splits_available),
        "early_stop_raster_split": _preferred_split(raster_splits_available),
        "point_ready": point_ready,
        "raster_ready": raster_ready,
        "fully_runnable": fully_runnable,
        "point_blocked_reasons": point_blocked_reasons,
        "raster_blocked_reasons": raster_blocked_reasons,
        "blocked_reasons": _dedupe_reasons(
            point_blocked_reasons,
            raster_blocked_reasons,
        ),
    }


def _blocked_reason(
    readiness: dict[str, object],
    *,
    method_key: str,
    artifact_requirement: str | None = None,
) -> str:
    requirement = str(
        artifact_requirement
        or bench_config.METHODS[method_key]["artifact_requirement"]
    )
    if requirement in {"deepssf_cache", "deepssf_gpu"}:
        reasons = _dedupe_reasons(
            list(readiness.get("point_blocked_reasons", [])),
            list(readiness.get("raster_blocked_reasons", [])),
        )
    elif requirement == "points":
        reasons = list(readiness.get("point_blocked_reasons", []))
    else:
        raise ValueError(f"Unsupported artifact requirement: {requirement!r}")
    return reasons[0] if reasons else "missing prerequisites"


def _run_key(run_payload: dict[str, object]) -> str:
    payload = bench_config.stable_payload(_canonical_run_payload(run_payload))
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    variant = run_payload.get("method_variant")
    variant_key = f"__variant={variant}" if variant is not None else ""
    return (
        f"study={run_payload['study_id']}"
        f"__ts={run_payload['timescale']}"
        f"__cov={run_payload['covariate_set']}"
        f"__method={run_payload['method']}"
        f"{variant_key}"
        f"__seed={run_payload['seed']}"
        f"__k={payload_hash}"
    )


def _canonical_run_payload(run_payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in run_payload.items()
        if key not in {"experiment_name", "run_key"}
    }


def load_experiment(module_path: str) -> dict[str, object]:
    module, path = _load_module_from_path(module_path)
    if not hasattr(module, "EXPERIMENT"):
        raise ValueError(f"Experiment module must export EXPERIMENT: {path}")
    return bench_config.normalize_experiment(module.EXPERIMENT, source_path=path)


def checked_in_experiment_module_paths() -> list[str]:
    experiments_dir = Path(__file__).resolve().parents[2] / "experiments"
    if not experiments_dir.exists():
        return []
    return [
        str(path)
        for path in sorted(experiments_dir.glob("*.py"))
        if path.name != "__init__.py" and not path.name.startswith("_")
    ]


def _expand_method_variants(
    experiment: dict[str, object],
    method_key: str,
    method: dict[str, object],
) -> list[dict[str, object]]:
    configured = dict(experiment.get("method_variants", {})).get(method_key, [{}])
    expanded: list[dict[str, object]] = []
    for configured_variant in configured:
        variant = dict(configured_variant)
        model = dict(method.get("model", {}))
        model.update(dict(variant.get("model", {})))
        hparams = dict(method.get("hparams", {}))
        hparams.update(dict(variant.get("hparams", {})))
        expanded.append(
            {
                "key": variant.get("key"),
                "model": model,
                "hparams": hparams,
            }
        )
    return expanded


def expand_experiment(
    experiment: dict[str, object],
    studies_override: list[str] | None = None,
    *,
    readiness_map: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    normalized = bench_config.normalize_experiment(
        experiment,
        source_path=experiment.get("source_path"),
    )
    experiment.clear()
    experiment.update(normalized)

    studies = _resolve_studies(experiment, studies_override)
    blocked: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    local_readiness_map: dict[str, dict[str, object]] = {}
    deepssf_data = str(experiment["deepssf_data"])
    needs_points = True

    for study_id in studies:
        for timescale_key in experiment["timescales"]:
            readiness_key = f"{study_id}:{timescale_key}"
            if deepssf_data == "gpu":
                readiness_key += ":gpu"
                if needs_points:
                    readiness_key += ":points"
            readiness = None if readiness_map is None else readiness_map.get(readiness_key)
            if readiness is None:
                readiness = _study_timescale_readiness(
                    study_id,
                    str(timescale_key),
                    deepssf_data=deepssf_data,
                    needs_points=needs_points,
                )
                if readiness_map is not None:
                    readiness_map[readiness_key] = readiness
            local_readiness_map[readiness_key] = readiness
            for covariate_set_key in experiment["covariate_sets"]:
                band_ids = bench_config.resolve_covariate_band_ids(
                    str(timescale_key),
                    str(covariate_set_key),
                    study_id=study_id,
                )
                timescale = bench_config.resolve_timescale(
                    str(timescale_key),
                    study_id=study_id,
                )
                for method_key in experiment["methods"]:
                    method = bench_config.METHODS[str(method_key)]
                    method_variants = _expand_method_variants(
                        experiment,
                        str(method_key),
                        method,
                    )
                    requirement = str(method["artifact_requirement"])
                    if str(method_key) == "deepssf" and deepssf_data == "gpu":
                        requirement = "deepssf_gpu"

                    if requirement == "deepssf_gpu":
                        band_to_layer = {
                            channel.band_id: channel.layer
                            for channel in timescale["plan"].channels
                        }
                        incompatible = [
                            band_id
                            for band_id in band_ids
                            if band_to_layer.get(band_id) != ALPHAEARTH_LAYER
                        ]
                        if incompatible:
                            for variant in method_variants:
                                blocked_run = {
                                    "study_id": study_id,
                                    "timescale": str(timescale_key),
                                    "covariate_set": str(covariate_set_key),
                                    "method": str(method_key),
                                    "reason": (
                                        "deepssf_data='gpu' only supports AlphaEarth "
                                        "covariates"
                                    ),
                                }
                                if variant["key"] is not None:
                                    blocked_run["method_variant"] = str(variant["key"])
                                blocked.append(blocked_run)
                            continue

                    if requirement in {"deepssf_cache", "deepssf_gpu"}:
                        evaluation_splits = list(readiness["held_out_splits"])
                        early_stop_split_key = _preferred_split(evaluation_splits)
                        runnable = bool(
                            readiness["point_ready"] and evaluation_splits
                        )
                    elif requirement == "points":
                        evaluation_splits = list(readiness["available_point_splits"])
                        early_stop_split_key = readiness["early_stop_point_split"]
                        runnable = bool(readiness["point_ready"])
                    else:
                        raise ValueError(
                            f"Unsupported artifact requirement for method {method_key!r}: {requirement!r}"
                        )

                    if not runnable or early_stop_split_key is None:
                        for variant in method_variants:
                            blocked_run = {
                                "study_id": study_id,
                                "timescale": str(timescale_key),
                                "covariate_set": str(covariate_set_key),
                                "method": str(method_key),
                                "reason": _blocked_reason(
                                    readiness,
                                    method_key=str(method_key),
                                    artifact_requirement=requirement,
                                ),
                            }
                            if variant["key"] is not None:
                                blocked_run["method_variant"] = str(variant["key"])
                            blocked.append(blocked_run)
                        continue

                    base_run_payload = {
                        "study_id": study_id,
                        "timescale": str(timescale_key),
                        "step_time_s": int(timescale["step_time_s"]),
                        "layers_config": str(
                            timescale["alphaearth_layers_config"]
                            if requirement == "deepssf_gpu"
                            else timescale["layers_config"]
                        ),
                        "rasters_hash": str(timescale["rasters_hash"]),
                        "sampler_config": str(timescale["sampler_config"]),
                        "sampler_hash": str(timescale["sampler_hash"]),
                        "geometry": str(timescale["geometry"]),
                        "covariate_set": str(covariate_set_key),
                        "band_ids": band_ids,
                        "method": str(method_key),
                        "artifact_requirement": requirement,
                        "executor": str(method["executor"]),
                        "seed": int(experiment["seed"]),
                        "early_stop_split_key": str(early_stop_split_key),
                        "evaluation_split_keys": evaluation_splits,
                    }
                    if requirement == "deepssf_gpu":
                        base_run_payload["deepssf_data"] = "gpu"
                        base_run_payload["selected_splits"] = [
                            "train",
                            *evaluation_splits,
                        ]

                    for variant in method_variants:
                        run_payload = {
                            **base_run_payload,
                            "method_model": dict(variant["model"]),
                            "method_hparams": dict(variant["hparams"]),
                        }
                        if variant["key"] is not None:
                            run_payload["method_variant"] = str(variant["key"])

                        resolved = _resolve_artifacts_for_run(run_payload)
                        if resolved["status"] != "resolved":
                            blocked_run = {
                                "study_id": study_id,
                                "timescale": str(timescale_key),
                                "covariate_set": str(covariate_set_key),
                                "method": str(method_key),
                                "reason": str(resolved["reason"]),
                            }
                            if "method_variant" in run_payload:
                                blocked_run["method_variant"] = run_payload[
                                    "method_variant"
                                ]
                            blocked.append(blocked_run)
                            continue
                        run_payload["run_key"] = _run_key(run_payload)
                        runs.append(run_payload)

    experiment["resolved_studies"] = studies
    experiment["blocked"] = blocked
    experiment["study_readiness"] = local_readiness_map
    experiment["planning_summary"] = {
        "planned_runs": len(runs),
        "blocked_runs": len(blocked),
        "studies": len(studies),
        "study_timescale_pairs": len(studies) * len(experiment["timescales"]),
    }
    return runs


def expanded_checked_in_experiments(
    *,
    readiness_map: dict[str, dict[str, object]] | None = None,
    studies_override: list[str] | None = None,
) -> list[dict[str, object]]:
    expanded: list[dict[str, object]] = []
    for module_path in checked_in_experiment_module_paths():
        experiment = load_experiment(module_path)
        runs = expand_experiment(
            experiment,
            studies_override=studies_override,
            readiness_map=readiness_map,
        )
        expanded.append(
            {
                "module_path": str(module_path),
                "experiment": experiment,
                "runs": runs,
            }
        )
    return expanded


def _resolve_point_artifacts(
    run_payload: dict[str, object],
    *,
    method: dict[str, object],
    timescale: dict[str, object],
) -> dict[str, object]:
    study_dir = paths.study_dir(str(run_payload["study_id"]))
    all_band_ids = list(timescale["all_band_ids"])
    band_ids = [str(band_id) for band_id in run_payload["band_ids"]]
    missing_band_ids = [band_id for band_id in band_ids if band_id not in all_band_ids]
    if missing_band_ids:
        return {
            "status": "blocked",
            "reason": f"unknown point-store band ids: {missing_band_ids}",
        }
    band_indices = [all_band_ids.index(band_id) for band_id in band_ids]

    point_stores: dict[str, dict[str, object]] = {}
    for split in ["train", *list(run_payload["evaluation_split_keys"])]:
        store_dir = point_store_dir(
            study_dir=study_dir,
            split=str(split),
            rasters_hash=str(run_payload["rasters_hash"]),
            sampler_hash=str(run_payload["sampler_hash"]),
        )
        manifest_path = point_store_manifest_path(
            study_dir=study_dir,
            split=str(split),
            rasters_hash=str(run_payload["rasters_hash"]),
            sampler_hash=str(run_payload["sampler_hash"]),
        )
        if not manifest_path.exists():
            return {
                "status": "blocked",
                "reason": f"missing point store manifest for {split}",
            }
        try:
            manifest = read_manifest(manifest_path)
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"invalid point store manifest for {split}: {exc}",
            }
        missing_manifest_bands = [
            band_id for band_id in band_ids if band_id not in manifest.bands
        ]
        if missing_manifest_bands:
            return {
                "status": "blocked",
                "reason": (
                    f"point store for {split} is missing bands {missing_manifest_bands}"
                ),
            }
        point_stores[str(split)] = {
            "store_dir": store_dir,
            "manifest_path": manifest_path,
            "manifest": manifest,
        }

    if str(run_payload["method"]) == "moveformer":
        expected_selected_splits: list[str] | None = None
        for split, store in point_stores.items():
            selected_splits = store["manifest"].extraction.get("selected_splits")
            if not isinstance(selected_splits, list) or not selected_splits:
                return {
                    "status": "blocked",
                    "reason": (
                        f"point store for {split} is missing extraction.selected_splits "
                        "required for MoveFormer context resolution"
                    ),
                }
            selected_splits = [str(split_key) for split_key in selected_splits]
            if expected_selected_splits is None:
                expected_selected_splits = selected_splits
                continue
            if selected_splits != expected_selected_splits:
                return {
                    "status": "blocked",
                    "reason": (
                        "MoveFormer requires train/eval point stores extracted from "
                        "the same selected_splits representation"
                    ),
                }

    normalization_stats_path: Path | None = None
    if bool(method.get("requires_normalization_stats", False)):
        normalization_stats_path = point_store_train_stats_path(
            study_dir=study_dir,
            rasters_hash=str(run_payload["rasters_hash"]),
            sampler_hash=str(run_payload["sampler_hash"]),
        )
        if not normalization_stats_path.exists():
            return {
                "status": "blocked",
                "reason": "missing normalization stats for train",
            }
        try:
            load_points_normalization_mean_std(
                study_dir=study_dir,
                plan=_subset_layer_plan(timescale["plan"], band_ids),
                sampler_config=timescale["sampler"],
            )
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"invalid normalization stats for train: {exc}",
            }

    return {
        "status": "resolved",
        "artifact_requirement": "points",
        "study_dir": study_dir,
        "trajectories_path": study_dir / "trajectories.csv",
        "plan": timescale["plan"],
        "sampler": timescale["sampler"],
        "all_band_ids": all_band_ids,
        "band_ids": band_ids,
        "band_indices": band_indices,
        "normalization_stats_path": normalization_stats_path,
        "point_stores": point_stores,
    }


def _resolve_deepssf_cache_artifacts(
    run_payload: dict[str, object],
    *,
    timescale: dict[str, object],
) -> dict[str, object]:
    all_band_ids = list(timescale["all_band_ids"])
    expected_env_channels = len(all_band_ids)
    expected_output_hw = tuple(int(v) for v in timescale["output_hw"])
    band_ids = [str(band_id) for band_id in run_payload["band_ids"]]
    missing_band_ids = [band_id for band_id in band_ids if band_id not in all_band_ids]
    if missing_band_ids:
        return {
            "status": "blocked",
            "reason": f"unknown deepSSF cache band ids: {missing_band_ids}",
        }
    channel_indices = [all_band_ids.index(band_id) for band_id in band_ids]

    requested_splits = ["train", *list(run_payload["evaluation_split_keys"])]
    cache_manifests: dict[str, dict[str, object]] = {}
    reference_output_hw: tuple[int, int] | None = None
    reference_pixel_size_m: float | None = None
    exact_failure_reason: str | None = None
    exact_cache_manifests: dict[str, dict[str, object]] = {}
    for split in requested_splits:
        cache_hash = bench_config.deepssf_cache_hash_for_split(
            str(run_payload["timescale"]),
            str(split),
            study_id=str(run_payload["study_id"]),
        )
        cache_dir = (
            paths.study_dir(str(run_payload["study_id"]))
            / "cache"
            / "deepssf_wds"
            / str(split)
            / str(cache_hash)
        )
        entry, reason = _load_deepssf_cache_entry(cache_dir, split=str(split))
        if entry is None:
            exact_failure_reason = reason
            exact_cache_manifests = {}
            break
        manifest = dict(entry["manifest"])
        env_channels = int(manifest["env_channels"])
        if env_channels != expected_env_channels:
            exact_failure_reason = (
                f"deepSSF cache for {split} exposes env_channels={env_channels} "
                f"but timescale {run_payload['timescale']} expects the full-cache "
                f"env_channels={expected_env_channels}"
            )
            exact_cache_manifests = {}
            break
        split_output_hw = tuple(int(v) for v in manifest["output_hw"])
        if split_output_hw != expected_output_hw:
            exact_failure_reason = (
                f"deepSSF cache for {split} exposes output_hw={list(split_output_hw)} "
                f"but timescale {run_payload['timescale']} expects output_hw="
                f"{list(expected_output_hw)}"
            )
            exact_cache_manifests = {}
            break
        split_pixel_size_m = float(manifest["pixel_size_m"])
        if reference_output_hw is None:
            reference_output_hw = split_output_hw
            reference_pixel_size_m = split_pixel_size_m
        elif split_output_hw != reference_output_hw:
            exact_failure_reason = (
                f"deepSSF cache metadata mismatch for {split}: "
                f"output_hw train={list(reference_output_hw)} {split}={list(split_output_hw)}"
            )
            exact_cache_manifests = {}
            break
        elif reference_pixel_size_m is not None and not math.isclose(
            split_pixel_size_m,
            reference_pixel_size_m,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            exact_failure_reason = (
                f"deepSSF cache metadata mismatch for {split}: "
                f"pixel_size_m train={reference_pixel_size_m} {split}={split_pixel_size_m}"
            )
            exact_cache_manifests = {}
            break
        exact_cache_manifests[str(split)] = entry
    if exact_cache_manifests:
        cache_manifests = exact_cache_manifests
    else:
        reference_output_hw = None
        reference_pixel_size_m = None
        train_candidates = _compatible_deepssf_cache_candidates(
            study_id=str(run_payload["study_id"]),
            split="train",
            expected_output_hw=expected_output_hw,
            expected_rasters_hash=str(run_payload["rasters_hash"]),
            requested_band_ids=band_ids,
        )
        for train_candidate in train_candidates:
            candidate_cache_manifests = {"train": train_candidate}
            candidate_band_ids = list(train_candidate["cache_band_ids"])
            candidate_output_hw = tuple(
                int(v) for v in train_candidate["manifest"]["output_hw"]
            )
            candidate_pixel_size_m = float(train_candidate["manifest"]["pixel_size_m"])
            for split in requested_splits[1:]:
                split_candidates = _compatible_deepssf_cache_candidates(
                    study_id=str(run_payload["study_id"]),
                    split=str(split),
                    expected_output_hw=expected_output_hw,
                    expected_rasters_hash=str(run_payload["rasters_hash"]),
                    requested_band_ids=band_ids,
                    required_cache_band_ids=candidate_band_ids,
                )
                if not split_candidates:
                    candidate_cache_manifests = {}
                    break
                split_candidate = split_candidates[0]
                split_output_hw = tuple(
                    int(v) for v in split_candidate["manifest"]["output_hw"]
                )
                split_pixel_size_m = float(split_candidate["manifest"]["pixel_size_m"])
                if split_output_hw != candidate_output_hw:
                    candidate_cache_manifests = {}
                    break
                if not math.isclose(
                    split_pixel_size_m,
                    candidate_pixel_size_m,
                    rel_tol=1e-6,
                    abs_tol=0.0,
                ):
                    candidate_cache_manifests = {}
                    break
                candidate_cache_manifests[str(split)] = split_candidate
            if candidate_cache_manifests:
                cache_manifests = candidate_cache_manifests
                reference_output_hw = candidate_output_hw
                reference_pixel_size_m = candidate_pixel_size_m
                channel_indices = list(train_candidate["channel_indices"])
                break
        if not cache_manifests:
            if exact_failure_reason is None:
                exact_failure_reason = "missing compatible deepSSF caches"
            return {
                "status": "blocked",
                "reason": exact_failure_reason,
            }

    if reference_output_hw is None:
        return {
            "status": "blocked",
            "reason": "missing deepSSF cache metadata",
        }
    if len(reference_output_hw) != 2 or int(reference_output_hw[0]) != int(
        reference_output_hw[1]
    ):
        return {
            "status": "blocked",
            "reason": (
                f"deepSSF expects square chips; cache output_hw={list(reference_output_hw)}"
            ),
        }

    image_dim = int(
        run_payload.get("method_model", {}).get("image_dim", reference_output_hw[0])
    )
    if image_dim != int(reference_output_hw[0]):
        return {
            "status": "blocked",
            "reason": (
                f"deepSSF model.image_dim={image_dim} is incompatible with "
                f"cache output_hw={list(reference_output_hw)}"
            ),
        }

    return {
        "status": "resolved",
        "artifact_requirement": "deepssf_cache",
        "study_dir": paths.study_dir(str(run_payload["study_id"])),
        "all_band_ids": all_band_ids,
        "band_ids": band_ids,
        "channel_indices": channel_indices,
        "cache_manifests": cache_manifests,
    }


def _resolve_deepssf_gpu_artifacts(
    run_payload: dict[str, object],
    *,
    timescale: dict[str, object],
) -> dict[str, object]:
    alphaearth_band_ids = [
        channel.band_id
        for channel in timescale["plan"].channels
        if channel.layer == ALPHAEARTH_LAYER
    ]
    band_ids = [str(band_id) for band_id in run_payload["band_ids"]]
    incompatible = [
        band_id for band_id in band_ids if band_id not in alphaearth_band_ids
    ]
    if incompatible:
        return {
            "status": "blocked",
            "reason": "deepssf_data='gpu' only supports AlphaEarth covariates",
        }

    output_hw = tuple(int(value) for value in timescale["output_hw"])
    if len(output_hw) != 2 or output_hw[0] != output_hw[1]:
        return {
            "status": "blocked",
            "reason": f"deepSSF expects square chips; output_hw={list(output_hw)}",
        }
    image_dim = int(
        run_payload.get("method_model", {}).get("image_dim", output_hw[0])
    )
    if image_dim != output_hw[0]:
        return {
            "status": "blocked",
            "reason": (
                f"deepSSF model.image_dim={image_dim} is incompatible with "
                f"output_hw={list(output_hw)}"
            ),
        }

    study_id = str(run_payload["study_id"])
    study_dir = paths.study_dir(study_id)
    trajectories_path = study_dir / "trajectories.csv"
    if not trajectories_path.is_file():
        return {
            "status": "blocked",
            "reason": f"missing trajectories: {trajectories_path}",
        }
    tile_dir = (
        study_dir
        / "rasters"
        / str(timescale["rasters_hash"])
        / "tiles"
        / ALPHAEARTH_LAYER
    )
    try:
        catalog = AlphaEarthCatalog.from_tile_directories([(study_id, tile_dir)])
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"invalid or missing AlphaEarth tiles: {exc}",
        }

    return {
        "status": "resolved",
        "artifact_requirement": "deepssf_gpu",
        "study_dir": study_dir,
        "trajectories_path": trajectories_path,
        "plan": _subset_layer_plan(timescale["plan"], alphaearth_band_ids),
        "catalog": catalog,
        "all_band_ids": alphaearth_band_ids,
        "band_ids": band_ids,
        "channel_indices": [
            alphaearth_band_ids.index(band_id) for band_id in band_ids
        ],
        "output_hw": output_hw,
    }


def _resolve_artifacts_for_run(run_payload: dict[str, object]) -> dict[str, object]:
    method = bench_config.METHODS[str(run_payload["method"])]
    timescale = bench_config.resolve_timescale(
        str(run_payload["timescale"]),
        study_id=str(run_payload["study_id"]),
    )
    requirement = str(
        run_payload.get("artifact_requirement", method["artifact_requirement"])
    )
    if requirement == "points":
        return _resolve_point_artifacts(run_payload, method=method, timescale=timescale)
    if requirement == "deepssf_cache":
        return _resolve_deepssf_cache_artifacts(run_payload, timescale=timescale)
    if requirement == "deepssf_gpu":
        return _resolve_deepssf_gpu_artifacts(run_payload, timescale=timescale)
    raise ValueError(f"Unsupported artifact requirement: {requirement!r}")


def load_saved_run_bundle(
    run_dir: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    run_dir = Path(run_dir)
    run_payload = json.loads((run_dir / "resolved_run.json").read_text(encoding="utf-8"))
    if not isinstance(run_payload, dict):
        raise ValueError(f"resolved_run.json must contain an object: {run_dir}")
    return dict(run_payload), _resolve_artifacts_for_run(run_payload)


def _load_executor(executor_ref: str):
    module_name, func_name = str(executor_ref).split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _subset_layer_plan(plan: LayerPlan, band_ids: list[str]) -> LayerPlan:
    wanted = set(band_ids)
    channels = [channel for channel in plan.channels if channel.band_id in wanted]
    layers_in_order: list[str] = []
    seen_layers: set[str] = set()
    for channel in channels:
        if channel.layer in seen_layers:
            continue
        seen_layers.add(channel.layer)
        layers_in_order.append(channel.layer)
    return LayerPlan(
        rasters_hash=str(plan.rasters_hash),
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups={key: dict(value) for key, value in plan.chip_groups.items()},
        derived_dependencies={
            key: list(value) for key, value in plan.derived_dependencies.items()
        },
        step=plan.step,
    )


class _RunTracker:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        del metrics, step

    def write_history(self, history: list[dict[str, object]]) -> None:
        history_path = self.run_dir / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _run_dir(run_key: str) -> Path:
    return paths.repo_root() / "results" / "bench" / "runs" / str(run_key)


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _history_steps_completed(run_dir: Path) -> int:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return 0
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    steps = [
        int(float(entry["step"]))
        for entry in history_payload
        if isinstance(entry, dict) and "step" in entry
    ]
    return max(steps) if steps else 0


def _interrupted_summary(
    run_payload: dict[str, object],
    *,
    run_dir: Path,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "interrupted",
        "reason": str(reason),
        "train_steps": _history_steps_completed(run_dir),
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
    }


def _interrupt_reason(exc: BaseException) -> str:
    if isinstance(exc, _RunInterrupted):
        return str(exc.reason)
    return exc.__class__.__name__


def _install_interrupt_handlers() -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handle_signal(signum, frame) -> None:
        del frame
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{int(signum)}"
        raise _RunInterrupted(name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError, RuntimeError):
            continue
    return previous


def _restore_interrupt_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError, RuntimeError):
            continue


def _manifest_slug(name: str) -> str:
    slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(name).strip()
    ).strip("_")
    return slug or "experiment"


def _resolved_manifest_payload(
    *,
    experiment: dict[str, object],
    planned_runs: list[dict[str, object]],
    pending_runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "experiment": dict(experiment),
        "planned_runs": [dict(run_payload) for run_payload in planned_runs],
        "planner_blocked_runs": [dict(run) for run in experiment.get("blocked", [])],
        "pending_run_keys": [str(run["run_key"]) for run in pending_runs],
    }


def _resolved_manifest_path(
    *,
    experiment: dict[str, object],
    manifest_payload: dict[str, object],
) -> Path:
    payload_hash = hashlib.sha256(
        bench_config.stable_payload(manifest_payload).encode("utf-8")
    ).hexdigest()[:16]
    return (
        paths.repo_root()
        / "results"
        / "bench"
        / "manifests"
        / f"{_manifest_slug(str(experiment['name']))}__{payload_hash}.json"
    )


def _clear_runtime_outputs(run_dir: Path) -> None:
    bench_run_state.remove_active_run(run_dir)
    for path in (
        run_dir / "training_summary.json",
        run_dir / "history.json",
        run_dir / "checkpoint_best.pt",
    ):
        if path.exists():
            path.unlink()
    predictions_dir = run_dir / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
    final_metrics_dir = run_dir / "final_metrics"
    if final_metrics_dir.exists():
        shutil.rmtree(final_metrics_dir)


def _write_blocked_run(
    run_payload: dict[str, object],
    *,
    run_dir: Path,
    reason: str,
) -> dict[str, object]:
    _log_progress(f"{run_payload['run_key']}: blocked: {reason}")
    summary = {
        "status": "blocked",
        "reason": reason,
        "train_steps": 0,
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
    }
    _write_json(run_dir / "training_summary.json", summary)
    return summary


def _execute_run(
    run_payload: dict[str, object],
    *,
    resume: bool = True,
) -> dict[str, object]:
    """
    Note this deletes existing artifacts! It should only be called when you *really* want to execute the run;
    that's why it is a private method.
    """
    run_dir = _run_dir(str(run_payload["run_key"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    canonical_payload = _canonical_run_payload(run_payload)
    resume_run = False
    resolved_run_path = run_dir / "resolved_run.json"
    if bool(resume) and resolved_run_path.exists():
        try:
            resume_run = (
                bench_config.stable_payload(_canonical_run_payload(_load_json(resolved_run_path)))
                == bench_config.stable_payload(canonical_payload)
            )
        except Exception:
            resume_run = False

    if resume_run:
        training_summary_path = run_dir / "training_summary.json"
        if training_summary_path.exists():
            training_summary_path.unlink()
        final_metrics_dir = run_dir / "final_metrics"
        if final_metrics_dir.exists():
            shutil.rmtree(final_metrics_dir)
    else:
        _clear_runtime_outputs(run_dir)
    _write_json(resolved_run_path, canonical_payload)

    if str(run_payload["method"]) not in _SUPPORTED_METHODS:
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=f"executor not implemented for method {run_payload['method']}",
        )

    resolved = _resolve_artifacts_for_run(run_payload)
    if resolved["status"] != "resolved":
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=str(resolved["reason"]),
        )

    _log_progress(f"{run_payload['run_key']}: starting executor {run_payload['method']}")
    tracker = _RunTracker(run_dir)
    bench_run_state.write_active_run(run_dir)
    previous_handlers = _install_interrupt_handlers()
    try:
        try:
            executor = _load_executor(str(run_payload["executor"]))
            summary = dict(
                executor(
                    run_payload=run_payload,
                    resolved_artifacts=resolved,
                    run_dir=run_dir,
                    tracker=tracker,
                )
            )
            summary.setdefault("status", "completed")
            if str(summary["status"]) == "completed":
                _write_json(run_dir / "training_summary.json", summary)
                _log_progress(f"{run_payload['run_key']}: starting evaluation")
                _evaluation_runner_module().evaluate_run_dir(run_dir)
                _log_progress(f"{run_payload['run_key']}: finished evaluation")
            _write_json(run_dir / "training_summary.json", summary)
            return summary
        except (KeyboardInterrupt, _RunInterrupted) as exc:
            reason = _interrupt_reason(exc)
            _log_progress(f"{run_payload['run_key']}: interrupted: {reason}")
            summary = _interrupted_summary(run_payload, run_dir=run_dir, reason=reason)
            _write_json(run_dir / "training_summary.json", summary)
            raise
        except Exception as exc:
            _log_progress(f"{run_payload['run_key']}: failed: {exc}")
            summary = {
                "status": "failed",
                "reason": str(exc),
                "train_steps": 0,
                "early_stop_split_key": str(run_payload["early_stop_split_key"]),
            }
            _write_json(run_dir / "training_summary.json", summary)
            return summary
    finally:
        _restore_interrupt_handlers(previous_handlers)
        bench_run_state.remove_active_run(run_dir)


def execute_run_payload(
    run_payload: dict[str, object],
    *,
    resume: bool = True,
) -> dict[str, object]:
    return _execute_run(dict(run_payload), resume=resume)


def resolve_pending_runs(
    experiment_module: str,
    *,
    studies_override: list[str] | None = None,
    methods: list[str] | None = None,
    max_runs: int | None = None,
    rerun: bool = False,
) -> dict[str, object]:
    print("Loading experiment")
    experiment = load_experiment(experiment_module)
    if methods is not None:
        experiment["methods"] = [str(method) for method in methods]

    print("Expanding experiment")
    planned_runs = expand_experiment(experiment, studies_override=studies_override)
    planner_blocked_runs = [dict(run) for run in experiment.get("blocked", [])]

    print("Loading latest runs")
    latest_runs = bench_run_state.search_latest_run_records_by_key(
        [str(run["run_key"]) for run in planned_runs]
    )

    print("Prepping pending runs")
    pending_runs: list[dict[str, object]] = []
    skipped_runs = 0
    for run_payload in planned_runs:
        run_key = str(run_payload["run_key"])
        latest_run = dict(latest_runs.get(run_key, {}))
        latest_status = str(latest_run.get("status", ""))
        if not bool(rerun):
            if latest_status == "completed":
                skipped_runs += 1
                continue
            if latest_status == "running":
                skipped_runs += 1
                continue
            failure_reason = str(dict(latest_run.get("tags", {})).get("failure_reason", ""))
            if failure_reason == "stale interrupted run":
                bench_run_state.remove_active_run(_run_dir(run_key))
        pending_runs.append(dict(run_payload))

    if max_runs is not None:
        pending_runs = pending_runs[: int(max_runs)]

    manifest_payload = _resolved_manifest_payload(
        experiment=experiment,
        planned_runs=planned_runs,
        pending_runs=pending_runs,
    )
    manifest_path = _resolved_manifest_path(
        experiment=experiment,
        manifest_payload=manifest_payload,
    )

    return {
        "experiment": experiment,
        "planned_runs": [dict(run_payload) for run_payload in planned_runs],
        "pending_runs": pending_runs,
        "planner_blocked_runs": planner_blocked_runs,
        "latest_runs": latest_runs,
        "manifest_payload": manifest_payload,
        "manifest_path": manifest_path,
        "counts": {
            "planned_runs": len(planned_runs) + len(planner_blocked_runs),
            "pending_runs": len(pending_runs),
            "skipped_runs": skipped_runs,
            "blocked_runs": len(planner_blocked_runs),
        },
    }


def load_resolved_run(manifest_path: str | Path, run_key: str) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    for run_payload in manifest.get("planned_runs", []):
        if str(run_payload.get("run_key")) == str(run_key):
            return dict(run_payload)
    raise KeyError(f"Run key {run_key!r} not found in manifest {manifest_path}")


def run_experiment(
    experiment_module: str,
    *,
    studies_override: list[str] | None = None,
    methods: list[str] | None = None,
    max_runs: int | None = None,
    rerun: bool = False,
) -> dict[str, object]:
    resolved = resolve_pending_runs(
        experiment_module,
        studies_override=studies_override,
        methods=methods,
        max_runs=max_runs,
        rerun=rerun,
    )

    executed_runs = 0
    executor_blocked_runs = 0
    failed_runs = 0
    pending_runs = list(resolved["pending_runs"])
    total_pending_runs = len(pending_runs)
    skipped_runs = int(resolved["counts"]["skipped_runs"])
    planner_blocked_runs = int(resolved["counts"]["blocked_runs"])
    if total_pending_runs:
        _log_progress(
            "Executing "
            f"{total_pending_runs} run(s); skipped={skipped_runs}, planner_blocked={planner_blocked_runs}"
        )
    for run_index, run_payload in enumerate(pending_runs, start=1):
        _log_progress(
            f"[{run_index}/{total_pending_runs}] starting {run_payload['run_key']}"
        )
        summary = execute_run_payload(run_payload, resume=not bool(rerun))
        executed_runs += 1
        status = str(summary.get("status", ""))
        if status == "blocked":
            executor_blocked_runs += 1
        elif status == "failed":
            failed_runs += 1
        _log_progress(
            f"[{run_index}/{total_pending_runs}] finished {run_payload['run_key']} status={status or 'completed'}"
        )

    return {
        "planned_runs": int(resolved["counts"]["planned_runs"]),
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "blocked_runs": planner_blocked_runs + executor_blocked_runs,
        "failed_runs": failed_runs,
    }


def _status_context(
    studies_override: list[str] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    studies = (
        resolve_study_selection(studies_override)
        if studies_override is not None
        else _list_available_studies()
    )
    study_entries: list[dict[str, object]] = []
    readiness_index: dict[str, dict[str, object]] = {}

    ssf_ready_pairs = 0
    raster_ready_pairs = 0
    fully_runnable_pairs = 0
    blocked_pairs = 0
    blocked_studies = 0

    for study_id in studies:
        timescale_entries: list[dict[str, object]] = []
        study_has_blocked_pair = False
        for timescale_key in bench_config.TIMESCALES:
            readiness = _study_timescale_readiness(study_id, timescale_key)
            readiness_index[f"{study_id}:{timescale_key}"] = readiness
            timescale_entries.append(readiness)
            if readiness["point_ready"]:
                ssf_ready_pairs += 1
            if readiness["raster_ready"]:
                raster_ready_pairs += 1
            if readiness["fully_runnable"]:
                fully_runnable_pairs += 1
            else:
                blocked_pairs += 1
                study_has_blocked_pair = True
        if study_has_blocked_pair:
            blocked_studies += 1
        study_entries.append({"study_id": study_id, "timescales": timescale_entries})

    expanded_experiments = expanded_checked_in_experiments(
        readiness_map=readiness_index,
        studies_override=studies_override,
    )
    planned_keys: list[str] = []
    seen_planned_keys: set[str] = set()
    for expanded in expanded_experiments:
        for run in expanded.get("runs", []):
            run_key = str(run["run_key"])
            if run_key in seen_planned_keys:
                continue
            seen_planned_keys.add(run_key)
            planned_keys.append(run_key)
    latest_runs = bench_run_state.search_latest_run_records_by_key(planned_keys)
    completed_runs = [
        run_key
        for run_key, run in latest_runs.items()
        if str(dict(run).get("status", "")) == "completed"
    ]

    snapshot = {
        "counts": {
            "studies": len(studies),
            "study_timescale_pairs": len(studies) * len(bench_config.TIMESCALES),
            "blocked_studies": blocked_studies,
            "ssf_ready_pairs": ssf_ready_pairs,
            "raster_ready_pairs": raster_ready_pairs,
            "fully_runnable_pairs": fully_runnable_pairs,
            "blocked_pairs": blocked_pairs,
            "planned_runs": len(planned_keys),
            "completed_runs": len(completed_runs),
        },
        "studies": study_entries,
        "planned_run_keys": planned_keys,
        "completed_run_keys": completed_runs,
        "checked_in_experiment_count": len(expanded_experiments),
    }
    return snapshot, expanded_experiments


def status_snapshot(studies_override: list[str] | None = None) -> dict[str, object]:
    snapshot, _expanded_experiments = _status_context(studies_override=studies_override)
    return snapshot


def _render_status_text(snapshot: dict[str, object]) -> str:
    counts = snapshot["counts"]
    lines = [
        f"Studies: {counts['studies']}",
        f"Blocked studies: {counts['blocked_studies']}",
        f"SSF-ready pairs: {counts['ssf_ready_pairs']}",
        f"deepSSF-cache-ready pairs: {counts['raster_ready_pairs']}",
        f"Fully runnable pairs: {counts['fully_runnable_pairs']}",
        f"Blocked pairs: {counts['blocked_pairs']}",
        f"Planned runs: {counts['planned_runs']}",
        f"Completed runs: {counts['completed_runs']}",
    ]
    for study in snapshot["studies"]:
        for readiness in study["timescales"]:
            parts: list[str] = []
            if readiness["point_ready"]:
                parts.append("ssf-ready")
            if readiness["raster_ready"]:
                parts.append("deepSSF-cache-ready")
            if readiness["fully_runnable"]:
                parts.append("fully-runnable")
            else:
                parts.append("blocked")
            if readiness["blocked_reasons"]:
                parts.append(f"reasons={'; '.join(readiness['blocked_reasons'])}")
            lines.append(
                f"{study['study_id']} / {readiness['timescale']}: {', '.join(parts)}"
            )
    return "\n".join(lines)


def _status_selection_label(studies_override: list[str] | None) -> str:
    if not studies_override:
        return "all"
    return " ".join(str(value) for value in studies_override)


def _command_plan(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment_module)
    runs = expand_experiment(experiment, studies_override=args.studies)
    planned_runs = int(len(runs) + len(experiment["blocked"]))
    existing_runs = len(
        bench_run_state.search_latest_run_records_by_key(
            [str(run["run_key"]) for run in runs]
        )
    )
    print(f"Planned runs: {planned_runs}")
    print(f"Existing runs: {existing_runs}")
    print(f"Blocked runs: {len(experiment['blocked'])}")
    print(f"Runnable runs: {len(runs)}")
    return 0


def _command_status(args: argparse.Namespace) -> int:
    if args.format == "text":
        selections = [list(args.studies)] if args.studies else [["all"], ["mini"]]
        sections = []
        for studies_override in selections:
            snapshot = status_snapshot(studies_override=studies_override)
            sections.append(
                "\n".join(
                    [
                        f"Study selection: {_status_selection_label(studies_override)}",
                        _render_status_text(snapshot),
                    ]
                )
            )
        print("\n\n".join(sections))
        return 0
    if args.format == "html":
        bench_report = _bench_report_module()
        result = bench_report.write_report(
            status_snapshot=status_snapshot(studies_override=args.studies),
            refresh_metrics=False,
            allow_metric_repair=False,
        )
        print(result["index_path"])
        return 0
    raise ValueError(f"Unsupported status format: {args.format}")


def _command_report(args: argparse.Namespace) -> int:
    bench_report = _bench_report_module()
    result = bench_report.write_report(
        status_snapshot=status_snapshot(),
        refresh_metrics=bool(args.refresh_metrics),
        allow_metric_repair=True,
    )
    print(result["index_path"])
    return 0


def _command_summary(args: argparse.Namespace) -> int:
    bench_summary = _bench_summary_module()
    experiments_override = bench_summary.normalize_experiment_selection(args.experiments)
    selections = [list(args.studies)] if args.studies else [["all"], ["mini"]]
    sections = []
    for studies_override in selections:
        lines = [f"Study selection: {_status_selection_label(studies_override)}"]
        if experiments_override:
            lines.append(
                "Experiment selection: " + ", ".join(experiments_override)
            )
        lines.append(
            bench_summary.render_current_summary(
                studies_override=studies_override,
                experiments_override=experiments_override,
            )
        )
        sections.append("\n".join(lines))
    print("\n\n".join(sections))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    summary = run_experiment(
        args.experiment_module,
        studies_override=args.studies,
        methods=args.methods,
        max_runs=args.max_runs,
        rerun=bool(args.rerun),
    )
    print(f"Planned runs: {summary['planned_runs']}")
    print(f"Executed runs: {summary['executed_runs']}")
    print(f"Skipped runs: {summary['skipped_runs']}")
    print(f"Blocked runs: {summary['blocked_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("bench", help="Plan and inspect the benchmark runtime")
    bench_subparsers = parser.add_subparsers(dest="bench_command", required=True)

    plan_parser = bench_subparsers.add_parser("plan", help="Plan a benchmark experiment")
    plan_parser.add_argument("experiment_module")
    plan_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    plan_parser.set_defaults(func=_command_plan)

    status_parser = bench_subparsers.add_parser("status", help="Inspect artifact and run readiness")
    status_parser.add_argument("--format", choices=["text", "html"], default="text")
    status_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default text output prints both 'all' and 'mini'.",
    )
    status_parser.set_defaults(func=_command_status)

    report_parser = bench_subparsers.add_parser("report", help="Generate the benchmark report")
    report_parser.add_argument("--refresh-metrics", action="store_true")
    report_parser.set_defaults(func=_command_report)

    summary_parser = bench_subparsers.add_parser(
        "summary",
        help="Print high-level benchmark progress and blocker summaries",
    )
    summary_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default text output prints both 'all' and 'mini'.",
    )
    summary_parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional experiment names like 'main_results', 'covariate_comparison', "
            "'baselines', or 'smoke_one_study'. Aliases: 'main', 'covariates', 'smoke'."
        ),
    )
    summary_parser.set_defaults(func=_command_summary)

    run_parser = bench_subparsers.add_parser("run", help="Execute runnable benchmark runs")
    run_parser.add_argument("experiment_module")
    run_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    run_parser.add_argument("--methods", nargs="+", default=None)
    run_parser.add_argument("--max-runs", type=int, default=None)
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.set_defaults(func=_command_run)
