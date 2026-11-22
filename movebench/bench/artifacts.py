from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import movebench.bench.config as bench_config
import movebench.bench.run_state as bench_run_state
import movebench.paths as paths
from movebench.datasets import (
    DataSplit,
    GPUInputs,
    PointInputs,
    PointStoreInput,
    SharedTrainingData,
    SplitMetadata,
    StudyTrainingData,
    WDSCacheInput,
    WDSInputs,
)
from movebench.datasets.deepssf_gpu import ALPHAEARTH_LAYER, AlphaEarthCatalog
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
from movebench.geometry import LOCAL_GEOMETRY
from movebench.layers.plan import LayerPlan
from movebench.study import Study
from movebench.trajectories import (
    SET_COL,
    canonicalize_split_key,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)

PREFERRED_EARLY_STOP_SPLITS = ["test_temporal", "test_individual"]


def _study_split_keys(study_id: str) -> dict[str, list[str] | bool]:
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


@lru_cache(maxsize=None)
def _train_step_count(study_id: str, timescale_key: str) -> int:
    timescale = bench_config.resolve_timescale(timescale_key, study_id=study_id)
    step_spec = timescale["plan"].step_spec
    trajectories = load_trajectories_csv(
        paths.study_dir(study_id) / "trajectories.csv"
    )
    train = trajectories[trajectories[SET_COL] == "train"].copy()
    try:
        support = fit_interval_step_support(
            train,
            step_time_s=int(step_spec.step_time_s),
            gap_q=float(step_spec.gap_q),
            step_q=float(step_spec.step_q),
            max_step_q=float(step_spec.max_step_q),
            step_multiplier=float(step_spec.step_multiplier),
        )
    except ValueError as exc:
        if str(exc) != "No valid interval step lengths computed":
            raise
        return 0
    return len(
        select_step_starts_with_support(
            train,
            step_time_s=int(step_spec.step_time_s),
            gap_q=float(step_spec.gap_q),
            support=support,
        )
    )


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
    train_steps_empty = bool(
        needs_points
        and split_info["has_train"]
        and not train_points["available"]
        and (
            train_points.get("n_rows") == 0
            or (
                str(train_points["reason"])
                == "missing point store manifest for train"
                and _train_step_count(study_id, timescale_key) == 0
            )
        )
    )
    if train_steps_empty:
        train_points = {
            **train_points,
            "reason": f"train split has zero usable {timescale_key} steps",
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

    train_point_ready = bool(
        split_info["has_train"] and train_points["available"]
    )
    point_ready = train_point_ready and bool(point_splits_available)
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
        "train_steps_empty": train_steps_empty,
        "train_point_ready": train_point_ready,
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


def _resolved_splits(
    run_payload: dict[str, object],
    inputs: PointInputs | WDSInputs | GPUInputs,
    output_hw: tuple[int, int],
) -> dict[str, object]:
    metadata: SplitMetadata = {
        "study_id": str(run_payload["study_id"]),
        "timescale": str(run_payload["timescale"]),
        "band_ids": tuple(str(band_id) for band_id in run_payload["band_ids"]),
        "geometry": LOCAL_GEOMETRY,
        "output_hw": tuple(int(value) for value in output_hw),
    }
    return {
        "status": "resolved",
        "splits": {
            str(split): DataSplit(str(split), metadata, inputs)
            for split in ["train", *run_payload["evaluation_split_keys"]]
        },
    }


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

    point_stores: dict[str, PointStoreInput] = {}
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

    if bool(method.get("sequence_context", False)):
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

    inputs: PointInputs = {
        "kind": "points",
        "study_dir": study_dir,
        "trajectories_path": study_dir / "trajectories.csv",
        "plan": timescale["plan"],
        "sampler": timescale["sampler"],
        "all_band_ids": all_band_ids,
        "band_indices": band_indices,
        "normalization_stats_path": normalization_stats_path,
        "point_stores": point_stores,
    }
    return _resolved_splits(run_payload, inputs, timescale["output_hw"])


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
    cache_manifests: dict[str, WDSCacheInput] = {}
    reference_output_hw: tuple[int, int] | None = None
    reference_pixel_size_m: float | None = None
    exact_failure_reason: str | None = None
    exact_cache_manifests: dict[str, WDSCacheInput] = {}
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

    inputs: WDSInputs = {
        "kind": "wds",
        "study_dir": paths.study_dir(str(run_payload["study_id"])),
        "all_band_ids": all_band_ids,
        "channel_indices": channel_indices,
        "cache_manifests": cache_manifests,
    }
    return _resolved_splits(run_payload, inputs, reference_output_hw)


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

    inputs: GPUInputs = {
        "kind": "gpu",
        "study_dir": study_dir,
        "trajectories_path": trajectories_path,
        "plan": _subset_layer_plan(timescale["plan"], alphaearth_band_ids),
        "catalog": catalog,
        "all_band_ids": alphaearth_band_ids,
        "channel_indices": [
            alphaearth_band_ids.index(band_id) for band_id in band_ids
        ],
        "selected_splits": tuple(str(split) for split in run_payload["selected_splits"]),
    }
    return _resolved_splits(run_payload, inputs, output_hw)


def _resolve_shared_point_artifacts(
    run_payload: dict[str, object],
    *,
    method: dict[str, object],
) -> dict[str, object]:
    studies: list[StudyTrainingData] = []
    for descriptor in run_payload["shared_studies"]:
        study_run = {**run_payload, **dict(descriptor)}
        study_run["shared_training_run"] = False
        timescale = bench_config.resolve_timescale(
            str(study_run["timescale"]),
            study_id=str(study_run["study_id"]),
        )
        resolved = _resolve_point_artifacts(
            study_run,
            method=method,
            timescale=timescale,
        )
        if resolved["status"] != "resolved":
            return {
                "status": "blocked",
                "reason": f"{study_run['study_id']}: {resolved['reason']}",
            }
        splits = resolved["splits"]
        validation_keys = dict.fromkeys(
            [
                str(study_run["early_stop_split_key"]),
                *[str(split) for split in study_run["evaluation_split_keys"]],
                "test_temporal",
                "test_individual",
            ]
        )
        studies.append(
            StudyTrainingData(
                train=splits["train"],
                validation=tuple(splits[key] for key in validation_keys if key in splits),
            )
        )
    return {
        "status": "resolved",
        "train": SharedTrainingData(
            studies=tuple(studies),
            requested_study_ids=tuple(run_payload["shared_study_ids"]),
            zero_step_studies=tuple(run_payload.get("shared_zero_step_studies", ())),
            training_signature=str(run_payload["shared_training_signature"]),
        ),
    }


def _resolve_shared_checkpoint(
    run_payload: dict[str, object],
) -> dict[str, object]:
    run_key = str(run_payload["shared_checkpoint_run_key"])
    run_dir = bench_run_state.run_dir(run_key)
    payload_path = run_dir / "resolved_run.json"
    summary_path = run_dir / "training_summary.json"
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if not payload_path.exists() or not summary_path.exists():
        return {
            "status": "blocked",
            "reason": f"shared training run {run_key} is not completed",
        }
    try:
        training_payload = json.loads(payload_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "blocked",
            "reason": f"invalid shared training run {run_key}: {exc}",
        }
    if training_payload.get("shared_training_signature") != run_payload.get(
        "shared_training_signature"
    ):
        return {
            "status": "blocked",
            "reason": f"shared training run {run_key} has an incompatible signature",
        }
    if (
        bench_run_state.active_run_is_live(run_dir)
        or summary.get("status") != "completed"
    ):
        return {
            "status": "blocked",
            "reason": f"shared training run {run_key} is not completed",
        }
    if not checkpoint_path.exists():
        return {
            "status": "blocked",
            "reason": f"shared training run {run_key} is missing checkpoint_best.pt",
        }
    error = bench_run_state.completion_error(run_dir, training_payload, summary)
    if error:
        return {"status": "blocked", "reason": f"shared training run {run_key}: {error}"}
    return {
        "status": "resolved",
        "shared_checkpoint_path": checkpoint_path,
        "shared_training_run_dir": run_dir,
        "shared_training_payload": training_payload,
        "shared_training_summary": summary,
    }


def _resolve_artifacts_for_run(run_payload: dict[str, object]) -> dict[str, object]:
    method = bench_config.METHODS[str(run_payload["method"])]
    if bool(run_payload.get("shared_training_run", False)):
        return _resolve_shared_point_artifacts(run_payload, method=method)

    timescale = bench_config.resolve_timescale(
        str(run_payload["timescale"]),
        study_id=str(run_payload["study_id"]),
    )
    requirement = str(
        run_payload.get("artifact_requirement", method["artifact_requirement"])
    )
    if requirement == "points":
        resolved = _resolve_point_artifacts(
            run_payload,
            method=method,
            timescale=timescale,
        )
    elif requirement == "deepssf_cache":
        resolved = _resolve_deepssf_cache_artifacts(
            run_payload,
            timescale=timescale,
        )
    elif requirement == "deepssf_gpu":
        resolved = _resolve_deepssf_gpu_artifacts(
            run_payload,
            timescale=timescale,
        )
    else:
        raise ValueError(f"Unsupported artifact requirement: {requirement!r}")

    if resolved["status"] != "resolved":
        return resolved
    if method.get("shared_training_method") and run_payload.get(
        "shared_checkpoint_run_key"
    ):
        checkpoint = _resolve_shared_checkpoint(run_payload)
        if checkpoint["status"] != "resolved":
            return checkpoint
        resolved.update(checkpoint)
    return resolved


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
