from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import shutil

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
from movebench.study import Study

import movebench.bench_v2.mlflow as bench_mlflow
import movebench.bench_v2.config as bench_config
from movebench.bench_v2.mlflow import search_latest_runs_by_key

PREFERRED_EARLY_STOP_SPLITS = ["test-temporal", "test-individual", "test"]
_SUPPORTED_METHODS = {
    "baseline_nomove",
    "baseline_constvel",
    "baseline_randomwalk",
    "issf_linear",
    "issf_mlp",
    "deepssf",
}


def _load_module_from_path(module_path: str | Path):
    path = Path(module_path)
    if not path.exists():
        candidate = paths.repo_root() / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        raise FileNotFoundError(f"Experiment module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        f"movebench_bench_v2_{path.stem}",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.resolve()


def _resolve_studies(
    experiment: dict[str, object],
    studies_override: list[str] | None = None,
) -> list[str]:
    if studies_override:
        studies = [str(study_id).strip() for study_id in studies_override if str(study_id).strip()]
        if len(studies) == 1 and studies[0].lower() == "all":
            return Study.list_available()
        if not studies:
            raise ValueError("No studies resolved from --studies")
        return studies

    studies = experiment.get("studies", "all")
    if studies == "all":
        resolved = Study.list_available()
        if not resolved:
            raise ValueError(
                "No studies resolved. Set MOVEBENCH_DATA_ROOT or pass --studies."
            )
        return resolved

    if not isinstance(studies, list) or not studies:
        raise ValueError("Experiment 'studies' must be 'all' or a non-empty list")
    return [str(study_id) for study_id in studies]


def _study_split_keys(study_id: str) -> dict[str, list[str] | bool]:
    study_dir = paths.study_dir(study_id)
    path = study_dir / "trajectories.csv"
    if not path.exists():
        raise FileNotFoundError(f"Study trajectories.csv not found: {path}")

    seen: list[str] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "set" not in reader.fieldnames:
            raise ValueError(f"Study trajectories.csv missing 'set' column: {path}")
        for row in reader:
            split = str(row.get("set", "")).strip()
            if split and split not in seen:
                seen.append(split)

    extras = [
        split
        for split in seen
        if split not in {"train", *PREFERRED_EARLY_STOP_SPLITS}
    ]
    ordered = [
        split for split in PREFERRED_EARLY_STOP_SPLITS if split in seen
    ] + extras
    return {
        "has_train": "train" in seen,
        "held_out_splits": ordered,
    }


def _point_store_state(study_id: str, timescale_key: str, split: str) -> dict[str, object]:
    timescale = bench_config.resolve_timescale(timescale_key)
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
    return {
        "available": True,
        "manifest_path": manifest_path,
    }


def _deepssf_cache_state(study_id: str, timescale_key: str, split: str) -> dict[str, object]:
    cache_hash = bench_config.deepssf_cache_hash_for_split(timescale_key, split)
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


def _study_timescale_readiness(study_id: str, timescale_key: str) -> dict[str, object]:
    split_info = _study_split_keys(study_id)
    held_out_splits = list(split_info["held_out_splits"])
    train_points = _point_store_state(study_id, timescale_key, "train")
    held_out_point_states = {
        split: _point_store_state(study_id, timescale_key, split)
        for split in held_out_splits
    }
    point_splits_available = [
        split
        for split, state in held_out_point_states.items()
        if state["available"]
    ]
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

    point_ready = bool(train_points["available"]) and bool(point_splits_available)
    raster_ready = bool(train_cache["available"]) and bool(raster_splits_available)
    fully_runnable = point_ready and raster_ready

    point_blocked_reasons: list[str] = []
    raster_blocked_reasons: list[str] = []
    if not split_info["has_train"]:
        point_blocked_reasons.append("study has no train split")
        raster_blocked_reasons.append("study has no train split")
    if not train_points["available"]:
        point_blocked_reasons.append(str(train_points["reason"]))
    if not point_splits_available:
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
                "no held-out deepSSF caches available",
            )
        )

    return {
        "study_id": study_id,
        "timescale": timescale_key,
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
) -> str:
    requirement = str(bench_config.METHODS[method_key]["artifact_requirement"])
    if requirement == "deepssf_cache":
        reasons = list(readiness.get("raster_blocked_reasons", []))
    elif requirement == "points":
        reasons = list(readiness.get("point_blocked_reasons", []))
    else:
        raise ValueError(f"Unsupported artifact requirement: {requirement!r}")
    return reasons[0] if reasons else "missing prerequisites"


def _run_key(run_payload: dict[str, object]) -> str:
    payload = bench_config.stable_payload(_canonical_run_payload(run_payload))
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return (
        f"study={run_payload['study_id']}"
        f"__ts={run_payload['timescale']}"
        f"__cov={run_payload['covariate_set']}"
        f"__method={run_payload['method']}"
        f"__seed={run_payload['seed']}"
        f"__k={payload_hash}"
    )


def _canonical_run_payload(run_payload: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in run_payload.items()
        if key not in {"experiment_name", "experiment_kind", "run_key"}
    }


def load_experiment(module_path: str) -> dict[str, object]:
    module, path = _load_module_from_path(module_path)
    if not hasattr(module, "EXPERIMENT"):
        raise ValueError(f"Experiment module must export EXPERIMENT: {path}")
    return bench_config.normalize_experiment(module.EXPERIMENT, source_path=path)


def expand_experiment(
    experiment: dict[str, object],
    studies_override: list[str] | None = None,
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
    readiness_map: dict[str, dict[str, object]] = {}

    for study_id in studies:
        for timescale_key in experiment["timescales"]:
            readiness = _study_timescale_readiness(study_id, str(timescale_key))
            readiness_map[f"{study_id}:{timescale_key}"] = readiness
            for covariate_set_key in experiment["covariate_sets"]:
                band_ids = bench_config.resolve_covariate_band_ids(
                    str(timescale_key),
                    str(covariate_set_key),
                )
                timescale = bench_config.resolve_timescale(str(timescale_key))
                for method_key in experiment["methods"]:
                    method = bench_config.METHODS[str(method_key)]
                    requirement = str(method["artifact_requirement"])
                    if requirement == "deepssf_cache":
                        evaluation_splits = list(readiness["available_raster_splits"])
                        early_stop_split_key = readiness["early_stop_raster_split"]
                        runnable = bool(readiness["raster_ready"])
                    elif requirement == "points":
                        evaluation_splits = list(readiness["available_point_splits"])
                        early_stop_split_key = readiness["early_stop_point_split"]
                        runnable = bool(readiness["point_ready"])
                    else:
                        raise ValueError(
                            f"Unsupported artifact requirement for method {method_key!r}: {requirement!r}"
                        )

                    if not runnable or early_stop_split_key is None:
                        blocked.append(
                            {
                                "study_id": study_id,
                                "timescale": str(timescale_key),
                                "covariate_set": str(covariate_set_key),
                                "method": str(method_key),
                                "reason": _blocked_reason(readiness, method_key=str(method_key)),
                            }
                        )
                        continue

                    run_payload = {
                        "study_id": study_id,
                        "timescale": str(timescale_key),
                        "step_time_s": int(timescale["step_time_s"]),
                        "layers_config": str(timescale["layers_config"]),
                        "rasters_hash": str(timescale["rasters_hash"]),
                        "sampler_config": str(timescale["sampler_config"]),
                        "sampler_hash": str(timescale["sampler_hash"]),
                        "covariate_set": str(covariate_set_key),
                        "band_ids": band_ids,
                        "method": str(method_key),
                        "artifact_requirement": str(method["artifact_requirement"]),
                        "executor": str(method["executor"]),
                        "method_model": dict(method.get("model", {})),
                        "method_hparams": dict(method.get("hparams", {})),
                        "seed": int(experiment["seed"]),
                        "early_stop_split_key": str(early_stop_split_key),
                        "evaluation_split_keys": evaluation_splits,
                    }
                    run_payload["run_key"] = _run_key(run_payload)
                    runs.append(run_payload)

    experiment["resolved_studies"] = studies
    experiment["blocked"] = blocked
    experiment["study_readiness"] = readiness_map
    experiment["planning_summary"] = {
        "planned_runs": len(runs),
        "blocked_runs": len(blocked),
        "studies": len(studies),
        "study_timescale_pairs": len(studies) * len(experiment["timescales"]),
    }
    return runs


def _manifest_payload(
    experiment: dict[str, object],
    runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": "1",
        "experiment": {
            key: value
            for key, value in experiment.items()
            if key
            not in {"blocked", "planning_summary", "study_readiness"}
        },
        "summary": dict(experiment.get("planning_summary", {})),
        "blocked": list(experiment.get("blocked", [])),
        "runs": runs,
    }


def write_resolved_manifest(
    experiment: dict[str, object],
    runs: list[dict[str, object]],
) -> Path:
    out_dir = paths.repo_root() / "results" / "bench_v2" / "resolved_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment['name']}.json"
    manifest = _manifest_payload(experiment, runs)

    existing_runs: dict[str, str] = {}
    for existing_manifest in _load_resolved_manifests():
        for existing_run in existing_manifest.get("runs", []):
            run_key = str(existing_run["run_key"])
            payload = bench_config.stable_payload(_canonical_run_payload(existing_run))
            if run_key in existing_runs and existing_runs[run_key] != payload:
                raise ValueError(
                    f"Existing resolved manifests contain conflicting payloads for run_key {run_key!r}"
                )
            existing_runs[run_key] = payload

    for run in runs:
        run_key = str(run["run_key"])
        payload = bench_config.stable_payload(_canonical_run_payload(run))
        if run_key in existing_runs and existing_runs[run_key] != payload:
            raise ValueError(
                f"Existing run_key {run_key!r} resolves to a different payload"
            )

    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def _load_resolved_manifests() -> list[dict[str, object]]:
    manifest_dir = paths.repo_root() / "results" / "bench_v2" / "resolved_experiments"
    if not manifest_dir.exists():
        return []
    manifests: list[dict[str, object]] = []
    for path in sorted(manifest_dir.glob("*.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


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
        shared_path = store_dir / "shared.pt"
        if not shared_path.exists():
            return {
                "status": "blocked",
                "reason": f"missing shared.pt for {split}",
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
    band_ids = [str(band_id) for band_id in run_payload["band_ids"]]
    missing_band_ids = [band_id for band_id in band_ids if band_id not in all_band_ids]
    if missing_band_ids:
        return {
            "status": "blocked",
            "reason": f"unknown deepSSF cache band ids: {missing_band_ids}",
        }
    channel_indices = [all_band_ids.index(band_id) for band_id in band_ids]

    cache_manifests: dict[str, dict[str, object]] = {}
    reference_output_hw: tuple[int, int] | None = None
    reference_pixel_size_m: float | None = None
    for split in ["train", *list(run_payload["evaluation_split_keys"])]:
        cache_hash = bench_config.deepssf_cache_hash_for_split(
            str(run_payload["timescale"]),
            str(split),
        )
        cache_dir = (
            paths.study_dir(str(run_payload["study_id"]))
            / "cache"
            / "deepssf_wds"
            / str(split)
            / str(cache_hash)
        )
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            return {
                "status": "blocked",
                "reason": f"missing deepSSF cache manifest for {split}",
            }
        if not list_deepssf_wds_shards(cache_dir):
            return {
                "status": "blocked",
                "reason": f"missing deepSSF shards for {split}",
            }
        try:
            manifest = read_deepssf_wds_manifest(manifest_path)
        except Exception as exc:
            return {
                "status": "blocked",
                "reason": f"invalid deepSSF cache manifest for {split}: {exc}",
            }
        env_channels = int(manifest["env_channels"])
        if channel_indices and max(channel_indices) >= env_channels:
            return {
                "status": "blocked",
                "reason": (
                    f"deepSSF cache for {split} exposes env_channels={env_channels} "
                    f"but the requested covariate view needs index {max(channel_indices)}"
                ),
            }
        split_output_hw = tuple(int(v) for v in manifest["output_hw"])
        split_pixel_size_m = float(manifest["pixel_size_m"])
        if reference_output_hw is None:
            reference_output_hw = split_output_hw
            reference_pixel_size_m = split_pixel_size_m
        elif split_output_hw != reference_output_hw:
            return {
                "status": "blocked",
                "reason": (
                    f"deepSSF cache metadata mismatch for {split}: "
                    f"output_hw train={list(reference_output_hw)} {split}={list(split_output_hw)}"
                ),
            }
        elif reference_pixel_size_m is not None and not math.isclose(
            split_pixel_size_m,
            reference_pixel_size_m,
            rel_tol=1e-6,
            abs_tol=0.0,
        ):
            return {
                "status": "blocked",
                "reason": (
                    f"deepSSF cache metadata mismatch for {split}: "
                    f"pixel_size_m train={reference_pixel_size_m} {split}={split_pixel_size_m}"
                ),
            }
        cache_manifests[str(split)] = {
            "cache_dir": cache_dir,
            "manifest_path": manifest_path,
            "manifest": manifest,
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


def _resolve_artifacts_for_run(run_payload: dict[str, object]) -> dict[str, object]:
    method = bench_config.METHODS[str(run_payload["method"])]
    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    requirement = str(method["artifact_requirement"])
    if requirement == "points":
        return _resolve_point_artifacts(run_payload, method=method, timescale=timescale)
    if requirement == "deepssf_cache":
        return _resolve_deepssf_cache_artifacts(run_payload, timescale=timescale)
    raise ValueError(f"Unsupported artifact requirement: {requirement!r}")


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
        log_fn = getattr(bench_mlflow.mlflow, "log_metrics", None)
        if callable(log_fn):
            log_fn(dict(metrics), step=int(step))

    def write_history(self, history: list[dict[str, float]]) -> None:
        history_path = self.run_dir / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _set_mlflow_tag(key: str, value: object) -> None:
    set_tag = getattr(bench_mlflow.mlflow, "set_tag", None)
    if callable(set_tag):
        set_tag(str(key), value)


def _end_mlflow_run(status: str | None) -> None:
    end_run = getattr(bench_mlflow.mlflow, "end_run", None)
    if callable(end_run):
        end_run(status=status)


def _latest_run_status(latest_run: object | None) -> str:
    if latest_run is None:
        return ""
    return str(getattr(getattr(latest_run, "data", None), "tags", {}).get("status", ""))


def _run_dir(run_key: str) -> Path:
    return paths.repo_root() / "results" / "bench_v2" / "runs" / str(run_key)


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _clear_runtime_outputs(run_dir: Path) -> None:
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


def _write_blocked_run(
    run_payload: dict[str, object],
    *,
    run_dir: Path,
    reason: str,
) -> dict[str, object]:
    bench_mlflow.start_benchmark_run({**run_payload, "status": "blocked"})
    _set_mlflow_tag("status", "blocked")
    _set_mlflow_tag("blocked_reason", reason)
    summary = {
        "status": "blocked",
        "reason": reason,
        "train_steps": 0,
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
    }
    _write_json(run_dir / "training_summary.json", summary)
    _end_mlflow_run("FAILED")
    return summary


def _execute_run(run_payload: dict[str, object]) -> dict[str, object]:
    """
    Note this deletes existing artifacts! It should only be called when you *really* want to execute the run;
    that's why it is a private method.
    """
    run_dir = _run_dir(str(run_payload["run_key"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    _clear_runtime_outputs(run_dir)
    _write_json(run_dir / "resolved_run.json", _canonical_run_payload(run_payload))

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

    bench_mlflow.start_benchmark_run({**run_payload, "status": "running"})
    tracker = _RunTracker(run_dir)
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
        _write_json(run_dir / "training_summary.json", summary)
        _set_mlflow_tag("status", summary["status"])
        _end_mlflow_run("FINISHED")
        return summary
    except Exception as exc:
        summary = {
            "status": "failed",
            "reason": str(exc),
            "train_steps": 0,
            "early_stop_split_key": str(run_payload["early_stop_split_key"]),
        }
        _write_json(run_dir / "training_summary.json", summary)
        _set_mlflow_tag("status", "failed")
        _set_mlflow_tag("failure_reason", str(exc))
        _end_mlflow_run("FAILED")
        return summary


def run_experiment(
    experiment_module: str,
    *,
    studies_override: list[str] | None = None,
    methods: list[str] | None = None,
    max_runs: int | None = None,
) -> dict[str, object]:
    experiment = load_experiment(experiment_module)
    if methods is not None:
        experiment["methods"] = [str(method) for method in methods]

    runs = expand_experiment(experiment, studies_override=studies_override)
    planner_blocked_runs = len(experiment.get("blocked", []))
    manifest_path = write_resolved_manifest(experiment, runs)
    latest_runs = bench_mlflow.search_latest_runs_by_key(
        [str(run["run_key"]) for run in runs]
    )

    pending_runs: list[dict[str, object]] = []
    skipped_runs = 0
    for run_payload in runs:
        latest_status = _latest_run_status(
            latest_runs.get(str(run_payload["run_key"]))
        )
        if latest_status == "completed":
            skipped_runs += 1
            continue
        pending_runs.append(run_payload)

    if max_runs is not None:
        pending_runs = pending_runs[: int(max_runs)]

    executed_runs = 0
    executor_blocked_runs = 0
    failed_runs = 0
    for run_payload in pending_runs:
        summary = _execute_run(run_payload)
        executed_runs += 1
        status = str(summary.get("status", ""))
        if status == "blocked":
            executor_blocked_runs += 1
        elif status == "failed":
            failed_runs += 1

    return {
        "manifest_path": manifest_path,
        "planned_runs": len(runs) + planner_blocked_runs,
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "blocked_runs": planner_blocked_runs + executor_blocked_runs,
        "failed_runs": failed_runs,
    }


def status_snapshot() -> dict[str, object]:
    studies = Study.list_available()
    study_entries: list[dict[str, object]] = []

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

    manifests = _load_resolved_manifests()
    planned_keys: list[str] = []
    seen_planned_keys: set[str] = set()
    for manifest in manifests:
        for run in manifest.get("runs", []):
            run_key = str(run["run_key"])
            if run_key in seen_planned_keys:
                continue
            seen_planned_keys.add(run_key)
            planned_keys.append(run_key)
    latest_runs = search_latest_runs_by_key(planned_keys)
    completed_runs = [
        run_key
        for run_key, run in latest_runs.items()
        if str(getattr(run.data, "tags", {}).get("status", "")) == "completed"
    ]

    return {
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
        "resolved_experiment_count": len(manifests),
    }


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


def _command_plan(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment_module)
    runs = expand_experiment(experiment, studies_override=args.studies)
    manifest_path = write_resolved_manifest(experiment, runs)
    print(f"Planned runs: {len(runs)}")
    print(f"Blocked runs: {len(experiment['blocked'])}")
    print(f"Resolved manifest: {manifest_path}")
    return 0


def _command_status(args: argparse.Namespace) -> int:
    if args.format != "text":
        raise ValueError(f"Unsupported status format: {args.format}")
    print(_render_status_text(status_snapshot()))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    summary = run_experiment(
        args.experiment_module,
        studies_override=args.studies,
        methods=args.methods,
        max_runs=args.max_runs,
    )
    print(f"Planned runs: {summary['planned_runs']}")
    print(f"Executed runs: {summary['executed_runs']}")
    print(f"Skipped runs: {summary['skipped_runs']}")
    print(f"Blocked runs: {summary['blocked_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    print(f"Resolved manifest: {summary['manifest_path']}")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("bench", help="Plan and inspect the v2 benchmark runtime")
    bench_subparsers = parser.add_subparsers(dest="bench_command", required=True)

    plan_parser = bench_subparsers.add_parser("plan", help="Plan a benchmark experiment")
    plan_parser.add_argument("experiment_module")
    plan_parser.add_argument("--studies", nargs="+", default=None)
    plan_parser.set_defaults(func=_command_plan)

    status_parser = bench_subparsers.add_parser("status", help="Inspect artifact and run readiness")
    status_parser.add_argument("--format", choices=["text"], default="text")
    status_parser.set_defaults(func=_command_status)

    run_parser = bench_subparsers.add_parser("run", help="Execute runnable benchmark runs")
    run_parser.add_argument("experiment_module")
    run_parser.add_argument("--studies", nargs="+", default=None)
    run_parser.add_argument("--methods", nargs="+", default=None)
    run_parser.add_argument("--max-runs", type=int, default=None)
    run_parser.set_defaults(func=_command_run)
