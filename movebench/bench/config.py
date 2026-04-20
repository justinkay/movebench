from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import movebench.paths as paths
from movebench.datasets.deepssf_wds_cache import deepssf_wds_cache_hash
from movebench.features.config import load_sampler_config
from movebench.features.points import resolve_sampler_hash
from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.study import MARINE_STUDIES, is_named_study_set

EXPERIMENT_DEFAULTS: dict[str, object] = {
    "version": 1,
    "seed": 0,
    "studies": "all",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["all"],
    "methods": ["issf_linear", "issf_mlp", "deepssf", "moveformer"],
}

TIMESCALES: dict[str, dict[str, object]] = {
    "1h": {
        "key": "1h",
        "step_time_s": 3600,
        "layers_config": "configs/layers/all_ae_1h_1h.yaml",
        "marine_layers_config": "configs/layers/all_ae_plus_marine_1h_1h.yaml",
        "sampler_config": "configs/samplers/issf_defaults.yaml",
        "output_hw": (101, 101),
        "normalize": True,
        "samples_per_shard": 256,
        "spatial_dtype": "float16",
        "target_dtype": "uint8",
        "compression": "none",
    },
    "12h": {
        "key": "12h",
        "step_time_s": 43200,
        "layers_config": "configs/layers/all_ae_12h_12h.yaml",
        "marine_layers_config": "configs/layers/all_ae_plus_marine_12h_12h.yaml",
        "sampler_config": "configs/samplers/issf_defaults.yaml",
        "output_hw": (101, 101),
        "normalize": True,
        "samples_per_shard": 256,
        "spatial_dtype": "float16",
        "target_dtype": "uint8",
        "compression": "none",
    },
    "1d": {
        "key": "1d",
        "step_time_s": 86400,
        "layers_config": "configs/layers/all_ae_1d_1d.yaml",
        "marine_layers_config": "configs/layers/all_ae_plus_marine_1d_1d.yaml",
        "sampler_config": "configs/samplers/issf_defaults.yaml",
        "output_hw": (101, 101),
        "normalize": True,
        "samples_per_shard": 256,
        "spatial_dtype": "float16",
        "target_dtype": "uint8",
        "compression": "none",
    },
    "7d": {
        "key": "7d",
        "step_time_s": 604800,
        "layers_config": "configs/layers/all_ae_7d_7d.yaml",
        "marine_layers_config": "configs/layers/all_ae_plus_marine_7d_7d.yaml",
        "sampler_config": "configs/samplers/issf_defaults.yaml",
        "output_hw": (101, 101),
        "normalize": True,
        "samples_per_shard": 256,
        "spatial_dtype": "float16",
        "target_dtype": "uint8",
        "compression": "none",
    },
}

COVARIATE_SETS: dict[str, dict[str, object]] = {
    "all": {
        "key": "all",
        "selection": "all",
    },
    "all_environmental": {
        "key": "all_environmental",
        "selection": "exclude_layers",
        "exclude_layers": ["alphaearth_embeddings"],
    },
    "alphaearth_only": {
        "key": "alphaearth_only",
        "selection": "include_layers",
        "include_layers": ["alphaearth_embeddings"],
    },
    "lite": {
        "key": "lite",
        "selection": "mixed",
        "include_layers": ["dynamic_world"],
        "include_band_ids": ["dem.elevation", "landsat_ndvi.NDVI"],
    },
}

METHODS: dict[str, dict[str, object]] = {
    "baseline_nomove": {
        "key": "baseline_nomove",
        "trains": False,
        "artifact_requirement": "points",
        "requires_normalization_stats": False,
        "executor": "movebench.bench.executors.baselines:run_nomove",
        "model": {},
        "hparams": {},
    },
    "baseline_constvel": {
        "key": "baseline_constvel",
        "trains": False,
        "artifact_requirement": "points",
        "requires_normalization_stats": False,
        "executor": "movebench.bench.executors.baselines:run_constvel",
        "model": {},
        "hparams": {},
    },
    "baseline_randomwalk": {
        "key": "baseline_randomwalk",
        "trains": False,
        "artifact_requirement": "points",
        "requires_normalization_stats": False,
        "executor": "movebench.bench.executors.baselines:run_randomwalk",
        "model": {},
        "hparams": {},
    },
    "issf_linear": {
        "key": "issf_linear",
        "trains": True,
        "artifact_requirement": "points",
        "requires_normalization_stats": True,
        "executor": "movebench.bench.executors.issf:run_linear",
        "model": {},
        "hparams": {
            "lr": 0.01,
            "weight_decay": 0.001,
            "batch_size": 1048,
            "num_workers": 0,
            "prediction_num_workers": 4,
            "max_steps": 2000,
            "eval_every_steps": 10,
            "patience_evals": 10,
            "max_eval_items": 2500,
        },
    },
    "issf_mlp": {
        "key": "issf_mlp",
        "trains": True,
        "artifact_requirement": "points",
        "requires_normalization_stats": True,
        "executor": "movebench.bench.executors.issf:run_mlp",
        "model": {
            "hidden_dims": [128],
            "dropout": 0.6,
        },
        "hparams": {
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 1048,
            "num_workers": 0,
            "prediction_num_workers": 4,
            "max_steps": 2000,
            "eval_every_steps": 10,
            "patience_evals": 10,
            "max_eval_items": 2500,
        },
    },
    "deepssf": {
        "key": "deepssf",
        "trains": True,
        "artifact_requirement": "deepssf_cache",
        "requires_normalization_stats": False,
        "executor": "movebench.bench.executors.deepssf:run",
        "model": {
            "image_dim": 101,
        },
        "hparams": {
            "lr_movement": 0.00001,
            "lr_habitat": 0.0001,
            "weight_decay": 0.0,
            "batch_size": 32,
            "max_steps": 70000,
            "eval_every_steps": 350,
            "patience_evals": 10,
            "max_eval_items": 2500, # to keep validation fast during training loop
        },
    },
    "moveformer": {
        "key": "moveformer",
        "trains": True,
        "artifact_requirement": "points",
        "requires_normalization_stats": True,
        "executor": "movebench.bench.executors.moveformer:run",
        "model": {},
        "hparams": {
            "lr": 0.00005,
            "weight_decay": 0.0,
            "warmup_steps": 50,
            "lr_gamma": 1.0, # don't decay LR - causes issues with small studies
            "batch_size": 32,
            "max_len": 500,
            "min_len": 2,
            "max_steps": 2000,
            "eval_every_steps": 10, # effectively batch size is ~ batch_size*max_len
            "patience_evals": 10,
        },
    },
}


def _dedupe_strings(values: list[object], *, label: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} entries must be non-empty strings")
        if text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _validate_registry_keys(
    values: list[str],
    *,
    registry: dict[str, dict[str, object]],
    label: str,
) -> None:
    unknown = [value for value in values if value not in registry]
    if unknown:
        raise ValueError(
            f"Unknown {label}: {unknown}. Allowed: {sorted(registry.keys())}"
        )


def normalize_experiment(
    experiment: dict[str, object],
    *,
    source_path: str | Path | None = None,
) -> dict[str, object]:
    if not isinstance(experiment, dict):
        raise ValueError(
            f"EXPERIMENT must be a dict, got {type(experiment).__name__}"
        )

    merged = copy.deepcopy(EXPERIMENT_DEFAULTS)
    merged.update(copy.deepcopy(experiment))

    if int(merged.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported experiment version={merged.get('version')!r} (expected 1)"
        )

    name = str(merged.get("name", "")).strip()
    if not name:
        raise ValueError("Experiment must define a non-empty 'name'")

    studies = merged.get("studies", "all")
    if isinstance(studies, str):
        studies = studies.strip()
        if studies.lower() == "all":
            studies = "all"
        elif is_named_study_set(studies):
            studies = studies.lower()
        else:
            raise ValueError(
                "'studies' must be 'all', a named study set, or a list of study ids"
            )
    else:
        if not isinstance(studies, list):
            raise ValueError(
                "'studies' must be 'all', a named study set, or a list of study ids"
            )
        studies = _dedupe_strings(studies, label="studies")

    timescales = merged.get("timescales")
    if not isinstance(timescales, list) or not timescales:
        raise ValueError("'timescales' must be a non-empty list")
    timescales = _dedupe_strings(timescales, label="timescales")
    _validate_registry_keys(timescales, registry=TIMESCALES, label="timescales")

    covariate_sets = merged.get("covariate_sets")
    if not isinstance(covariate_sets, list) or not covariate_sets:
        raise ValueError("'covariate_sets' must be a non-empty list")
    covariate_sets = _dedupe_strings(covariate_sets, label="covariate_sets")
    _validate_registry_keys(
        covariate_sets,
        registry=COVARIATE_SETS,
        label="covariate_sets",
    )

    methods = merged.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("'methods' must be a non-empty list")
    methods = _dedupe_strings(methods, label="methods")
    _validate_registry_keys(methods, registry=METHODS, label="methods")

    normalized: dict[str, object] = {
        "version": 1,
        "name": name,
        "seed": int(merged.get("seed", 0)),
        "studies": studies,
        "timescales": timescales,
        "covariate_sets": covariate_sets,
        "methods": methods,
    }
    if source_path is not None:
        normalized["source_path"] = str(Path(source_path).resolve())
    if "description" in merged:
        normalized["description"] = str(merged["description"])
    return normalized


@lru_cache(maxsize=None)
def resolve_timescale(
    timescale_key: str,
    study_id: str | None = None,
) -> dict[str, object]:
    try:
        base = TIMESCALES[str(timescale_key)]
    except KeyError as e:
        raise KeyError(f"Unknown timescale: {timescale_key!r}") from e

    resolved = copy.deepcopy(base)
    if study_id is not None and str(study_id) in MARINE_STUDIES:
        resolved["layers_config"] = str(resolved["marine_layers_config"])
    layers_config = load_layers_config(_repo_relative_path(resolved["layers_config"]))
    plan = REGISTRY.resolve(layers_config)
    sampler_config = load_sampler_config(_repo_relative_path(resolved["sampler_config"]))
    sampler_hash = resolve_sampler_hash(sampler_config)
    resolved["plan"] = plan
    resolved["sampler"] = sampler_config
    resolved["sampler_hash"] = sampler_hash
    resolved["rasters_hash"] = str(plan.rasters_hash)
    resolved["all_band_ids"] = plan.band_ids_in_order()
    return resolved


def _repo_relative_path(path: object) -> str:
    p = Path(str(path))
    if p.is_absolute():
        return str(p)
    return str(paths.repo_root() / p)


def deepssf_cache_hash_for_split(
    timescale_key: str,
    split: str,
    study_id: str | None = None,
) -> str:
    timescale = resolve_timescale(timescale_key, study_id=study_id)
    return deepssf_wds_cache_hash(
        rasters_hash=str(timescale["rasters_hash"]),
        band_ids=list(timescale["all_band_ids"]),
        split=str(split),
        output_hw=tuple(timescale["output_hw"]),
        normalize=bool(timescale["normalize"]),
        normalization_sampler_hash=str(timescale["sampler_hash"]),
        samples_per_shard=int(timescale["samples_per_shard"]),
        spatial_dtype=str(timescale["spatial_dtype"]),
        target_dtype=str(timescale["target_dtype"]),
        compression=str(timescale["compression"]),
    )


def resolve_covariate_band_ids(
    timescale_key: str,
    covariate_set_key: str,
    study_id: str | None = None,
) -> list[str]:
    try:
        covariate_set = COVARIATE_SETS[str(covariate_set_key)]
    except KeyError as e:
        raise KeyError(f"Unknown covariate set: {covariate_set_key!r}") from e

    timescale = resolve_timescale(timescale_key, study_id=study_id)
    all_band_ids = list(timescale["all_band_ids"])
    band_to_layer = {
        channel.band_id: channel.layer for channel in timescale["plan"].channels
    }

    selection = str(covariate_set["selection"])
    if selection == "all":
        selected = all_band_ids
    elif selection == "exclude_layers":
        excluded = {str(layer) for layer in covariate_set.get("exclude_layers", [])}
        selected = [
            band_id
            for band_id in all_band_ids
            if band_to_layer[band_id] not in excluded
        ]
    elif selection == "include_layers":
        included = {str(layer) for layer in covariate_set.get("include_layers", [])}
        selected = [
            band_id
            for band_id in all_band_ids
            if band_to_layer[band_id] in included
        ]
    elif selection == "mixed":
        included_layers = {
            str(layer) for layer in covariate_set.get("include_layers", [])
        }
        included_band_ids = {
            str(band_id) for band_id in covariate_set.get("include_band_ids", [])
        }
        selected = [
            band_id
            for band_id in all_band_ids
            if band_id in included_band_ids or band_to_layer[band_id] in included_layers
        ]
    else:
        raise ValueError(
            f"Unsupported covariate selection mode={selection!r} for {covariate_set_key!r}"
        )

    if not selected:
        raise ValueError(
            f"Covariate set {covariate_set_key!r} resolved to zero bands for {timescale_key!r}"
        )
    return selected


def stable_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
