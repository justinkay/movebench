from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import movebench.bench.artifacts as bench_artifacts
import movebench.bench.config as bench_config
import movebench.bench.run_state as bench_run_state
import movebench.paths as paths
from movebench.datasets.deepssf_gpu import ALPHAEARTH_LAYER
from movebench.study import _list_available_studies, resolve_study_selection


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


def _resolve_studies(
    experiment: dict[str, object],
    studies_override: list[str] | None = None,
) -> list[str]:
    if studies_override:
        return resolve_study_selection(studies_override)
    return resolve_study_selection(experiment.get("studies", "all"))


def _run_key(run_payload: dict[str, object]) -> str:
    payload = bench_config.stable_payload(_canonical_run_payload(run_payload))
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    variant = run_payload.get("method_variant")
    variant_key = f"__variant={variant}" if variant is not None else ""
    study_label = (
        "shared" if run_payload.get("shared_training_run") else run_payload["study_id"]
    )
    return (
        f"study={study_label}"
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


_SHARED_STUDY_FIELDS = (
    "study_id",
    "timescale",
    "step_time_s",
    "layers_config",
    "rasters_hash",
    "sampler_config",
    "sampler_hash",
    "geometry",
    "covariate_set",
    "band_ids",
    "artifact_requirement",
    "early_stop_split_key",
    "evaluation_split_keys",
)


def _shared_group_signature(run_payload: dict[str, object]) -> str:
    payload = {
        key: run_payload.get(key)
        for key in (
            "timescale",
            "step_time_s",
            "covariate_set",
            "band_ids",
            "sampler_config",
            "sampler_hash",
            "geometry",
            "seed",
            "method_variant",
            "method_model",
            "method_hparams",
        )
    }
    return hashlib.sha256(
        bench_config.stable_payload(payload).encode("utf-8")
    ).hexdigest()


def _collapse_shared_runs(
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        groups.setdefault(_shared_group_signature(candidate), []).append(candidate)

    runs: list[dict[str, object]] = []
    for group_signature in sorted(groups):
        group = sorted(groups[group_signature], key=lambda run: str(run["study_id"]))
        data_group = [run for run in group if not run.get("shared_zero_train_steps")]
        if not data_group:
            continue
        requested_studies = [
            {key: run[key] for key in _SHARED_STUDY_FIELDS}
            for run in group
        ]
        studies = [
            {key: run[key] for key in _SHARED_STUDY_FIELDS}
            for run in data_group
        ]
        zero_step_studies = [
            {
                "study_id": str(run["study_id"]),
                "reason": str(run["shared_zero_train_steps_reason"]),
            }
            for run in group
            if run.get("shared_zero_train_steps")
        ]
        signature_payload = {
            "group_signature": group_signature,
            "studies": requested_studies,
        }
        if zero_step_studies:
            signature_payload["zero_step_study_ids"] = [
                study["study_id"] for study in zero_step_studies
            ]
        training_signature = hashlib.sha256(
            bench_config.stable_payload(signature_payload).encode("utf-8")
        ).hexdigest()
        run_payload = {
            **data_group[0],
            "shared_training_run": True,
            "shared_group_signature": group_signature,
            "shared_training_signature": training_signature,
            "shared_studies": studies,
            "shared_study_ids": [study["study_id"] for study in requested_studies],
            "early_stop_split_key": "shared_eval",
            "evaluation_split_keys": [],
        }
        if zero_step_studies:
            run_payload["shared_data_study_ids"] = [
                study["study_id"] for study in studies
            ]
            run_payload["shared_zero_step_studies"] = zero_step_studies
        run_payload["run_key"] = _run_key(run_payload)
        runs.append(run_payload)
    return runs


def _plan_shared_training_runs(
    *,
    experiment: dict[str, object],
    training_method: str,
    score_method: str,
    study_ids: list[str],
    readiness_map: dict[str, dict[str, object]] | None,
) -> list[dict[str, object]]:
    companion: dict[str, object] = {
        "version": 1,
        "name": f"{experiment['name']}__shared_training",
        "seed": experiment["seed"],
        "studies": list(study_ids),
        "timescales": list(experiment["timescales"]),
        "covariate_sets": list(experiment["covariate_sets"]),
        "methods": [str(training_method)],
        "deepssf_data": experiment["deepssf_data"],
    }
    variants = dict(experiment.get("method_variants", {})).get(str(score_method))
    if variants:
        companion["method_variants"] = {str(training_method): list(variants)}
    return expand_experiment(
        companion,
        studies_override=list(study_ids),
        readiness_map=readiness_map,
    )


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
    candidate_runs: list[dict[str, object]] = []
    blocked_shared_groups: set[tuple[str, str]] = set()
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
                readiness = bench_artifacts._study_timescale_readiness(
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
                        evaluation_splits = list(readiness["available_point_splits"])
                        early_stop_split_key = bench_artifacts._preferred_split(evaluation_splits)
                        runnable = bool(
                            readiness["point_ready"] and evaluation_splits
                        )
                    elif requirement == "points":
                        evaluation_splits = list(readiness["available_point_splits"])
                        early_stop_split_key = readiness["early_stop_point_split"]
                        if str(method.get("run_scope", "study")) == "shared":
                            runnable = bool(
                                readiness.get(
                                    "train_point_ready",
                                    readiness["point_ready"],
                                )
                                or readiness.get("train_steps_empty", False)
                            )
                        else:
                            runnable = bool(readiness["point_ready"])
                    else:
                        raise ValueError(
                            f"Unsupported artifact requirement for method {method_key!r}: {requirement!r}"
                        )

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
                        "implementation": str(method["implementation"]),
                        "seed": int(experiment["seed"]),
                        "early_stop_split_key": (
                            ""
                            if early_stop_split_key is None
                            else str(early_stop_split_key)
                        ),
                        "evaluation_split_keys": evaluation_splits,
                    }
                    if not bool(method.get("evaluates", True)):
                        base_run_payload["evaluates"] = False
                    if requirement == "deepssf_gpu":
                        base_run_payload["deepssf_data"] = "gpu"
                        base_run_payload["selected_splits"] = [
                            "train",
                            *evaluation_splits,
                        ]

                    requires_study_eval = (
                        str(method.get("run_scope", "study")) != "shared"
                    )
                    if not runnable or (
                        requires_study_eval and early_stop_split_key is None
                    ):
                        for variant in method_variants:
                            blocked_payload = {
                                **base_run_payload,
                                "method_model": dict(variant["model"]),
                                "method_hparams": dict(variant["hparams"]),
                            }
                            if variant["key"] is not None:
                                blocked_payload["method_variant"] = str(variant["key"])
                            if str(method.get("run_scope", "study")) == "shared":
                                blocked_shared_groups.add(
                                    (
                                        str(method_key),
                                        _shared_group_signature(blocked_payload),
                                    )
                                )
                            blocked_run = {
                                "study_id": study_id,
                                "timescale": str(timescale_key),
                                "covariate_set": str(covariate_set_key),
                                "method": str(method_key),
                                "reason": bench_artifacts._blocked_reason(
                                    readiness,
                                    method_key=str(method_key),
                                    artifact_requirement=requirement,
                                ),
                            }
                            if variant["key"] is not None:
                                blocked_run["method_variant"] = str(variant["key"])
                            blocked.append(blocked_run)
                        continue

                    for variant in method_variants:
                        run_payload = {
                            **base_run_payload,
                            "method_model": dict(variant["model"]),
                            "method_hparams": dict(variant["hparams"]),
                        }
                        if variant["key"] is not None:
                            run_payload["method_variant"] = str(variant["key"])

                        if (
                            str(method.get("run_scope", "study")) == "shared"
                            and readiness.get("train_steps_empty", False)
                        ):
                            reason = bench_artifacts._blocked_reason(
                                readiness,
                                method_key=str(method_key),
                                artifact_requirement=requirement,
                            )
                            run_payload["shared_zero_train_steps"] = True
                            run_payload["shared_zero_train_steps_reason"] = reason
                            candidate_runs.append(run_payload)
                            blocked_run = {
                                "study_id": study_id,
                                "timescale": str(timescale_key),
                                "covariate_set": str(covariate_set_key),
                                "method": str(method_key),
                                "reason": reason,
                            }
                            if variant["key"] is not None:
                                blocked_run["method_variant"] = str(variant["key"])
                            blocked.append(blocked_run)
                            continue

                        resolved = bench_artifacts._resolve_artifacts_for_run(run_payload)
                        if resolved["status"] != "resolved":
                            if str(method.get("run_scope", "study")) == "shared":
                                blocked_shared_groups.add(
                                    (
                                        str(method_key),
                                        _shared_group_signature(run_payload),
                                    )
                                )
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
                        candidate_runs.append(run_payload)

    runs: list[dict[str, object]] = []
    shared_candidates: list[dict[str, object]] = []
    score_candidates: list[dict[str, object]] = []
    for candidate in candidate_runs:
        method = bench_config.METHODS[str(candidate["method"])]
        if str(method.get("run_scope", "study")) == "shared":
            group_key = (
                str(candidate["method"]),
                _shared_group_signature(candidate),
            )
            if group_key not in blocked_shared_groups:
                shared_candidates.append(candidate)
        elif method.get("shared_training_method"):
            score_candidates.append(candidate)
        else:
            candidate["run_key"] = _run_key(candidate)
            runs.append(candidate)
    runs.extend(_collapse_shared_runs(shared_candidates))

    if score_candidates:
        training_studies = (
            resolve_study_selection(experiment["shared_training_studies"])
            if "shared_training_studies" in experiment
            else list(studies)
        )
        experiment["resolved_shared_training_studies"] = training_studies
        companion_runs: dict[tuple[str, str], list[dict[str, object]]] = {}
        for candidate in score_candidates:
            score_method = str(candidate["method"])
            training_method = str(
                bench_config.METHODS[score_method]["shared_training_method"]
            )
            companion_key = (training_method, score_method)
            if companion_key not in companion_runs:
                companion_runs[companion_key] = _plan_shared_training_runs(
                    experiment=experiment,
                    training_method=training_method,
                    score_method=score_method,
                    study_ids=training_studies,
                    readiness_map=readiness_map,
                )
            group_signature = _shared_group_signature(candidate)
            training_run = next(
                (
                    run
                    for run in companion_runs[companion_key]
                    if run["shared_group_signature"] == group_signature
                    and candidate["study_id"] in run["shared_study_ids"]
                ),
                None,
            )
            if training_run is None:
                blocked.append(
                    {
                        "study_id": candidate["study_id"],
                        "timescale": candidate["timescale"],
                        "covariate_set": candidate["covariate_set"],
                        "method": score_method,
                        "reason": "no compatible shared training run for selected cohort",
                    }
                )
                continue
            candidate.update(
                {
                    "shared_checkpoint_run_key": training_run["run_key"],
                    "shared_training_signature": training_run[
                        "shared_training_signature"
                    ],
                    "shared_training_study_ids": training_run["shared_study_ids"],
                }
            )
            resolved = bench_artifacts._resolve_artifacts_for_run(candidate)
            if resolved["status"] != "resolved":
                blocked.append(
                    {
                        "study_id": candidate["study_id"],
                        "timescale": candidate["timescale"],
                        "covariate_set": candidate["covariate_set"],
                        "method": score_method,
                        "reason": str(resolved["reason"]),
                    }
                )
                continue
            candidate["run_key"] = _run_key(candidate)
            runs.append(candidate)

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
            readiness = bench_artifacts._study_timescale_readiness(study_id, timescale_key)
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
