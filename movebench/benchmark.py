"""YAML-driven benchmark runner for MoveBench baselines.

This module intentionally keeps the benchmarking "framework" minimal. It is a
single file with a small set of functions for:
- config loading/validation,
- per-study setup (rasters/feature stores/normalization),
- per-method training/evaluation,
- writing JSON + CSV summaries.
"""

from __future__ import annotations

import functools
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from movebench.train import collate_deepssf
import torch
from movebench import Study, paths
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.features.points import resolve_sampler_hash
from torch.utils.data import DataLoader
from movebench.models.ssf import (
    LinearStepSelect,
    MLPStepSelect,
    random_baseline_metrics,
)
from movebench.train import fit_step_select_tensor
from movebench.models.deepssf import DeepSSF
from movebench.train import (
    eval_deepssf,
    fit_deepssf,
    random_grid_baseline_metrics,
)
from movebench.datasets.deepssf_wds_cache import (
    make_deepssf_wds_train_test_loaders,
)
from movebench.models.moveformer import build_moveformer_selection_model
from movebench.train import collate_moveformer, eval_moveformer, fit_moveformer

TOP_K_VALUES = (1, 5, 10, 25, 50)


def _collate_deepssf_subset(batch, *, subset_idxs: Sequence[int]):

    x1, x2, x3, y = collate_deepssf(batch)

    idx = torch.as_tensor(list(subset_idxs), dtype=torch.long)
    return x1.index_select(1, idx), x2, x3, y


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark config not found: {p}")
    with open(p, "r") as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(
            f"Benchmark config must be a mapping/dict at the top level; got {type(user).__name__}"
        )

    cfg = user
    if int(cfg.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported benchmark config version={cfg.get('version')!r} (expected 1)"
        )

    for key in (
        "layers_config",
        "sampler_config",
        "output_dir",
        "runtime",
        "datasets",
        "methods",
    ):
        if key not in cfg:
            raise ValueError(f"Benchmark config missing required key: {key!r}")

    if not isinstance(cfg["runtime"], dict):
        raise ValueError("Benchmark config `runtime` must be a mapping/dict.")
    for key in ("device", "seed", "setup_workers"):
        if key not in cfg["runtime"]:
            raise ValueError(f"Benchmark config missing required key: runtime.{key!r}")

    if not isinstance(cfg["datasets"], dict):
        raise ValueError("Benchmark config `datasets` must be a mapping/dict.")
    if "issf" not in cfg["datasets"] or not isinstance(cfg["datasets"]["issf"], dict):
        raise ValueError("Benchmark config missing required key: datasets.issf")
    issf = cfg["datasets"]["issf"]
    if "sampler" in issf:
        raise ValueError(
            "datasets.issf.sampler has been removed; use top-level sampler_config"
        )

    if not isinstance(cfg["methods"], list) or not cfg["methods"]:
        raise ValueError("Benchmark config must include a non-empty `methods:` list.")

    # Apply env overrides early (can affect paths/study discovery).
    env = cfg.get("env")
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError(
                "`env` must be a mapping of environment variable overrides."
            )
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    def _validate_schedule_hparams(
        *, method_name: str, hparams: dict[str, Any]
    ) -> None:
        use_epochs = any(
            k in hparams for k in ("max_epochs", "eval_every_epochs", "patience_epochs")
        )
        if use_epochs:
            for key in ("max_epochs", "patience_epochs"):
                if key not in hparams:
                    raise ValueError(
                        f"Method {method_name!r} missing required hparam: {key!r} (epoch schedule)"
                    )
        else:
            for key in ("max_steps", "eval_interval", "patience"):
                if key not in hparams:
                    raise ValueError(
                        f"Method {method_name!r} missing required hparam: {key!r} (step schedule)"
                    )

    # Validate methods (no default filling; rely on model/dataset defaults where applicable).
    allowed_kinds = {"issf_linear", "issf_mlp", "deepssf", "moveformer"}
    seen_names: set[str] = set()
    methods_out: list[dict[str, Any]] = []
    for m in cfg["methods"]:
        if not isinstance(m, dict):
            raise ValueError("Each entry in `methods` must be a mapping/dict.")
        name = str(m.get("name", "")).strip()
        if not name:
            raise ValueError("Each method must have a non-empty `name`.")
        if name in seen_names:
            raise ValueError(f"Duplicate method name: {name!r}")
        seen_names.add(name)

        kind = str(m.get("kind", "")).strip()
        if kind not in allowed_kinds:
            allowed = sorted(allowed_kinds)
            raise ValueError(f"Unknown method kind={kind!r}. Allowed: {allowed}")

        model_cfg = m.get("model") or {}
        hparams = m.get("hparams") or {}
        if not isinstance(model_cfg, dict) or not isinstance(hparams, dict):
            raise ValueError(f"Method {name!r}: `model` and `hparams` must be dicts.")

        if kind in {"issf_linear", "issf_mlp"}:
            for key in ("lr", "weight_decay"):
                if key not in hparams:
                    raise ValueError(
                        f"Method {name!r} missing required hparam: {key!r}"
                    )
            _validate_schedule_hparams(method_name=name, hparams=hparams)
        elif kind == "deepssf":
            if "deepssf" not in cfg["datasets"] or not isinstance(
                cfg["datasets"]["deepssf"], dict
            ):
                raise ValueError(
                    "Benchmark config missing required key: datasets.deepssf (required for kind='deepssf')"
                )
            for key in ("lr_movement", "lr_habitat", "weight_decay"):
                if key not in hparams:
                    raise ValueError(
                        f"Method {name!r} missing required hparam: {key!r}"
                    )
            _validate_schedule_hparams(method_name=name, hparams=hparams)
        elif kind == "moveformer":
            if "moveformer" not in cfg["datasets"] or not isinstance(
                cfg["datasets"]["moveformer"], dict
            ):
                raise ValueError(
                    "Benchmark config missing required key: datasets.moveformer (required for kind='moveformer')"
                )
            dcfg = cfg["datasets"]["moveformer"]
            if "normalize_env" in dcfg and not isinstance(dcfg["normalize_env"], bool):
                raise ValueError(
                    "datasets.moveformer.normalize_env must be a boolean if provided"
                )
            if "require_normalization_stats" in dcfg and not isinstance(
                dcfg["require_normalization_stats"], bool
            ):
                raise ValueError(
                    "datasets.moveformer.require_normalization_stats must be a boolean if provided"
                )
            _validate_schedule_hparams(method_name=name, hparams=hparams)
        else:
            raise RuntimeError(f"Internal error: unhandled kind={kind!r}")

        methods_out.append(
            {"name": name, "kind": kind, "model": model_cfg, "hparams": hparams}
        )

    cfg["methods"] = methods_out
    return cfg


def load_sampler(cfg: dict[str, Any]) -> SamplerConfig:
    return load_sampler_config(str(cfg["sampler_config"]))


def resolve_studies(cfg: dict[str, Any]) -> list[str]:

    studies_val = cfg.get("studies", None)
    if studies_val is None or (
        isinstance(studies_val, str) and studies_val.strip().lower() == "all"
    ):
        out = Study.list_available()
    elif isinstance(studies_val, list):
        out = [str(s).strip() for s in studies_val if str(s).strip()]
    else:
        raise ValueError("`studies` must be omitted/null/'all' or a list of study ids.")

    out = sorted(dict.fromkeys(out))
    if not out:
        raise ValueError(
            "No studies resolved. "
            f"Expected studies under: {paths.studies_root()} "
            "(override with env MOVEBENCH_DATA_ROOT or config.env)."
        )
    return out


def resolve_default_eval_split(study) -> str:
    inventory = study.split_inventory()
    held_out_splits = [str(split) for split in inventory["held_out_splits"]]
    if not held_out_splits:
        raise ValueError(f"Study {study.study_id!r} has no held-out splits.")
    for split in ("test_temporal", "test_individual", "test"):
        if split in held_out_splits:
            return split
    return held_out_splits[0]


def ensure_study_ready(*, study, cfg: dict[str, Any]) -> str:
    """Ensure rasters, parametric point store, and normalization stats exist; return sampler_hash."""

    layers_config = str(cfg["layers_config"])
    setup_workers = int(cfg["runtime"]["setup_workers"])
    sampler = load_sampler(cfg)
    sampler_hash = resolve_sampler_hash(sampler)

    try:
        study.extract_point_features(
            layers_config=layers_config,
            sampler_config=sampler,
            splits=None,
            n_workers=int(setup_workers),
            skip_existing=True,
        )
    except FileNotFoundError as e:
        if "rasters" not in str(e):
            raise
        study.download_rasters(
            layers_config=layers_config,
            splits=None,
            n_workers=int(setup_workers),
            skip_existing=True,
        )
        study.extract_point_features(
            layers_config=layers_config,
            sampler_config=sampler,
            splits=None,
            n_workers=int(setup_workers),
            skip_existing=True,
        )

    study.compute_normalization_vals(
        from_="points",
        layers_config=layers_config,
        sampler_config=sampler,
        split="train",
        skip_existing=True,
    )
    return sampler_hash


def _make_loaders(
    *, ds_train, ds_test, collate_fn, batch_size: int | None, num_workers: int | None
):

    kwargs: dict[str, Any] = {"collate_fn": collate_fn, "drop_last": False}
    if batch_size is not None:
        kwargs["batch_size"] = int(batch_size)
    if num_workers is not None:
        kwargs["num_workers"] = int(num_workers)

    train_loader = DataLoader(
        ds_train,
        shuffle=True,
        **kwargs,
    )
    test_loader = DataLoader(
        ds_test,
        shuffle=False,
        **kwargs,
    )
    return train_loader, test_loader


def _resolve_fit_schedule(
    *, hparams: dict[str, Any], steps_per_epoch: int
) -> dict[str, int]:
    steps_per_epoch = int(steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0; got {steps_per_epoch}")

    use_epochs = any(
        k in hparams for k in ("max_epochs", "eval_every_epochs", "patience_epochs")
    )
    if use_epochs:
        if "max_epochs" not in hparams or "patience_epochs" not in hparams:
            raise ValueError(
                "Epoch-based schedule requires hparams.max_epochs and hparams.patience_epochs."
            )
        max_epochs = int(hparams["max_epochs"])
        eval_every_epochs = int(hparams.get("eval_every_epochs", 1))
        patience_epochs = int(hparams["patience_epochs"])
        if max_epochs <= 0:
            raise ValueError(f"hparams.max_epochs must be > 0; got {max_epochs}")
        if eval_every_epochs <= 0:
            raise ValueError(
                f"hparams.eval_every_epochs must be > 0; got {eval_every_epochs}"
            )
        if patience_epochs <= 0:
            raise ValueError(
                f"hparams.patience_epochs must be > 0; got {patience_epochs}"
            )

        max_steps = int(max_epochs) * int(steps_per_epoch)
        eval_interval = int(eval_every_epochs) * int(steps_per_epoch)
        patience = int(math.ceil(float(patience_epochs) / float(eval_every_epochs)))
    else:
        max_steps = int(hparams["max_steps"])
        eval_interval = int(hparams["eval_interval"])
        patience = int(hparams["patience"])

    if max_steps <= 0:
        raise ValueError(f"max_steps must be > 0; got {max_steps}")
    if eval_interval <= 0:
        raise ValueError(f"eval_interval must be > 0; got {eval_interval}")
    if patience <= 0:
        raise ValueError(f"patience must be > 0; got {patience}")

    return {
        "steps_per_epoch": int(steps_per_epoch),
        "max_steps": int(max_steps),
        "eval_interval": int(eval_interval),
        "patience": int(patience),
    }


def _resolve_max_eval_steps(cfg: dict[str, Any]) -> int | None:
    runtime = cfg.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("cfg.runtime must be a dict")
    val = runtime.get("max_eval_steps", None)
    if val is None:
        return None
    max_eval = int(val)
    if max_eval <= 0:
        raise ValueError(f"runtime.max_eval_steps must be > 0 or null; got {val!r}")
    return int(max_eval)


def _run_issf(
    *,
    study,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    model_kind: str,
    log_callback: Any | None = None,
) -> dict[str, Any]:

    issf = cfg["datasets"]["issf"]
    sampler = load_sampler(cfg)

    ds_kwargs: dict[str, Any] = {
        "layers_config": str(cfg["layers_config"]),
        "sampler_config": sampler,
    }
    for k in (
        "include_time",
        "normalize",
        "subset_env_layers",
        "subset_band_ids",
        "include_individual_id",
        "normalize_movement",
    ):
        if k in issf:
            ds_kwargs[k] = issf[k]

    test_split = resolve_default_eval_split(study)
    train_ds = study.issf_dataset(split="train", **ds_kwargs)
    test_ds = study.issf_dataset(split=test_split, **ds_kwargs)

    batch_size = int(issf.get("batch_size", None) or 1)
    steps_per_epoch = int(math.ceil(float(len(train_ds)) / float(batch_size)))
    schedule = _resolve_fit_schedule(
        hparams=dict(method["hparams"]), steps_per_epoch=steps_per_epoch
    )

    # Cache expanded features once. On CUDA we store bf16 on GPU when possible,
    # otherwise fall back to pinned CPU and stream batches to the GPU.
    feat_dtype = torch.bfloat16 if str(device.type) == "cuda" else torch.float32
    x_train = train_ds.build_expanded_x(
        cache_device=device, dtype=feat_dtype, fallback_to_pinned_cpu=True
    )
    x_test = test_ds.build_expanded_x(
        cache_device=device, dtype=feat_dtype, fallback_to_pinned_cpu=True
    )

    if x_train.ndim != 3:
        raise ValueError(
            f"iSSF dataset must produce x with shape (N,C,P); got {tuple(x_train.shape)}"
        )
    _n, c, p = x_train.shape
    random_metrics = random_baseline_metrics(int(p), TOP_K_VALUES)

    if model_kind == "linear":
        model = LinearStepSelect(int(c))
    elif model_kind == "mlp":
        model_cfg = method.get("model") or {}
        kwargs: dict[str, Any] = {}
        if "hidden_dims" in model_cfg:
            kwargs["hidden_dims"] = tuple(int(x) for x in model_cfg["hidden_dims"])
        if "dropout" in model_cfg:
            kwargs["dropout"] = float(model_cfg["dropout"])
        model = MLPStepSelect(int(c), **kwargs)
    else:
        raise ValueError(f"Unknown model_kind={model_kind!r}")

    max_eval_steps = _resolve_max_eval_steps(cfg)
    fit = fit_step_select_tensor(
        model,
        x_train,
        x_test,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr=float(method["hparams"]["lr"]),
        weight_decay=float(method["hparams"]["weight_decay"]),
        max_steps=int(schedule["max_steps"]),
        eval_interval=int(schedule["eval_interval"]),
        patience=int(schedule["patience"]),
        batch_size=int(batch_size),
        max_eval_items=max_eval_steps,
        log_callback=log_callback,
    )

    # Save the best checkpoint (fit_step_select restores the best weights before returning).
    out_dir = Path(str(cfg["output_dir"]))
    ckpt_dir = out_dir / "checkpoints" / str(study.study_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    method_name = str(method["name"])
    ckpt_path = ckpt_dir / f"{method_name}.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {
                "study_id": str(study.study_id),
                "method_name": method_name,
                "kind": str(method["kind"]),
                "in_channels": int(c),
                "num_candidates": int(p),
                "best_step": int(fit["best_step"]),
                "best_test_nll": float(fit["val"]["nll"]),
            },
        },
        ckpt_path,
    )
    train_metrics = dict(fit["train"])
    test_metrics = dict(fit["val"])

    return {
        "P": int(p),
        "random": random_metrics,
        "schedule": schedule,
        "fit": fit,
        "train": train_metrics,
        "test": test_metrics,
        "test_split_key": str(test_split),
        "checkpoint_path": str(ckpt_path),
    }


def run_issf_linear(
    *,
    study,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    log_callback: Any | None = None,
) -> dict[str, Any]:
    return _run_issf(
        study=study,
        cfg=cfg,
        method=method,
        device=device,
        model_kind="linear",
        log_callback=log_callback,
    )


def run_issf_mlp(
    *,
    study,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    log_callback: Any | None = None,
) -> dict[str, Any]:
    return _run_issf(
        study=study,
        cfg=cfg,
        method=method,
        device=device,
        model_kind="mlp",
        log_callback=log_callback,
    )


def run_deepssf(
    *,
    study,
    sampler_hash: str,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    log_callback: Any | None = None,
) -> dict[str, Any]:

    dcfg = cfg["datasets"]["deepssf"]
    subset_idxs = None
    subset_band_ids = dcfg.get("subset_band_ids")
    if subset_band_ids:
        plan0 = study.layer_plan(str(cfg["layers_config"]))
        band_ids = plan0.band_ids_in_order()
        want = {str(x) for x in subset_band_ids}
        missing = sorted(want - set(band_ids))
        if missing:
            raise ValueError(
                f"datasets.deepssf.subset_band_ids contains unknown band_id(s): {missing}"
            )
        subset_idxs = [i for i, bid in enumerate(band_ids) if bid in want]

    ds_kwargs_train: dict[str, Any] = {
        "split": "train",
        "layers_config": str(cfg["layers_config"]),
        "sampler_config": load_sampler(cfg),
    }
    ds_kwargs_test = dict(ds_kwargs_train)
    test_split = resolve_default_eval_split(study)
    ds_kwargs_test["split"] = test_split

    for k in ("output_hw", "normalize"):
        if k not in dcfg:
            continue
        v = dcfg[k]
        if k == "output_hw":
            if not (isinstance(v, (list, tuple)) and len(v) == 2):
                raise ValueError(
                    "datasets.deepssf.output_hw must be a 2-item sequence like [101,101]."
                )
            v = (int(v[0]), int(v[1]))
        ds_kwargs_train[k] = v
        ds_kwargs_test[k] = v

    cache_cfg = dcfg.get("cache", None)
    use_cache = False
    if cache_cfg is not None:
        if not isinstance(cache_cfg, dict):
            raise ValueError(
                "datasets.deepssf.cache must be a mapping/dict when provided"
            )
        use_cache = bool(cache_cfg.get("enabled", False))

    if use_cache:
        assert isinstance(cache_cfg, dict)
        batch_size = dcfg.get("batch_size", None)
        bs = int(batch_size) if batch_size is not None else 1
        num_workers = int(dcfg.get("num_workers", 0) or 0)

        train_loader, test_loader, m_train, m_test, steps_per_epoch = (
            make_deepssf_wds_train_test_loaders(
                study=study,
                layers_config=str(cfg["layers_config"]),
                ds_kwargs_train=ds_kwargs_train,
                ds_kwargs_test=ds_kwargs_test,
                cache_cfg=cache_cfg,
                batch_size=int(bs),
                num_workers=int(num_workers),
                runtime_seed=int(cfg["runtime"]["seed"]),
            )
        )

        schedule = _resolve_fit_schedule(
            hparams=dict(method["hparams"]), steps_per_epoch=int(steps_per_epoch)
        )
        pixel_size_test = float(m_test["pixel_size_m"])
        pixel_size_train = float(m_train["pixel_size_m"])

    else:
        train_ds = study.deepssf_dataset(
            **ds_kwargs_train,
        )
        test_ds = study.deepssf_dataset(
            **ds_kwargs_test,
        )

        collate_fn = collate_deepssf
        if subset_idxs is not None:
            collate_fn = functools.partial(
                _collate_deepssf_subset, subset_idxs=subset_idxs
            )
        train_loader, test_loader = _make_loaders(
            ds_train=train_ds,
            ds_test=test_ds,
            collate_fn=collate_fn,
            batch_size=dcfg.get("batch_size", None),
            num_workers=dcfg.get("num_workers", None),
        )

        schedule = _resolve_fit_schedule(
            hparams=dict(method["hparams"]), steps_per_epoch=len(train_loader)
        )

    x1 = next(iter(train_loader))[0]
    b, c_env, h, w = x1.shape
    if int(h) != int(w):
        raise ValueError(f"deepSSF expects square chips. Got H={h}, W={w}.")

    model_cfg = method["model"]
    image_dim = int(model_cfg.get("image_dim", int(h)))
    if int(image_dim) != int(h):
        raise ValueError(
            f"deepSSF model.image_dim ({image_dim}) must match dataset chip size ({h})."
        )

    model = DeepSSF(
        int(c_env),
        image_dim=int(image_dim),
        pixel_size_m=float(
            pixel_size_train if use_cache else train_ds.chips.pixel_size_m
        ),
        device=device,
    )

    if not use_cache:
        pixel_size_test = float(test_ds.chips.pixel_size_m)
        pixel_size_train = float(model.pixel_size_m)
    else:
        pixel_size_train = float(model.pixel_size_m)
    if not math.isclose(pixel_size_test, pixel_size_train, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError(
            "deepSSF requires consistent chip grid scale across splits. "
            f"Got pixel_size_m train={pixel_size_train} test={pixel_size_test}."
        )

    max_eval_steps = _resolve_max_eval_steps(cfg)
    fit = fit_deepssf(
        model,
        train_loader,
        test_loader,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr_movement=float(method["hparams"]["lr_movement"]),
        lr_habitat=float(method["hparams"]["lr_habitat"]),
        weight_decay=float(method["hparams"]["weight_decay"]),
        max_steps=int(schedule["max_steps"]),
        eval_interval=int(schedule["eval_interval"]),
        patience=int(schedule["patience"]),
        max_eval_steps=max_eval_steps,
        log_callback=log_callback,
    )

    # Save the best checkpoint (fit_deepssf restores the best weights before returning).

    out_dir = Path(str(cfg["output_dir"]))
    ckpt_dir = out_dir / "checkpoints" / str(study.study_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    method_name = str(method["name"])
    ckpt_path = ckpt_dir / f"{method_name}.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {
                "study_id": str(study.study_id),
                "method_name": method_name,
                "env_channels": int(c_env),
                "image_dim": int(image_dim),
                "pixel_size_m": float(model.pixel_size_m),
                "best_step": int(fit["best_step"]),
                "best_test_nll": float(fit["val"]["nll"]),
            },
        },
        ckpt_path,
    )

    train_metrics = eval_deepssf(model, train_loader, device, topk=TOP_K_VALUES)
    test_metrics = eval_deepssf(model, test_loader, device, topk=TOP_K_VALUES)
    random_grid = random_grid_baseline_metrics(int(h) * int(w), TOP_K_VALUES)

    return {
        "grid": {
            "H": int(h),
            "W": int(w),
            "num_pixels": int(h) * int(w),
            "pixel_size_m": float(model.pixel_size_m),
        },
        "random_grid": random_grid,
        "schedule": schedule,
        "fit": fit,
        "train": train_metrics,
        "test": test_metrics,
        "test_split_key": str(test_split),
        "checkpoint_path": str(ckpt_path),
    }


def run_moveformer(
    *,
    study,
    sampler_hash: str,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    log_callback: Any | None = None,
) -> dict[str, Any]:

    dcfg = cfg["datasets"]["moveformer"]
    sampler = load_sampler(cfg)

    ds_kwargs: dict[str, Any] = {
        "layers_config": str(cfg["layers_config"]),
        "sampler_config": sampler,
    }
    if "max_len" in dcfg:
        ds_kwargs["max_len"] = int(dcfg["max_len"])
    if "min_len" in dcfg:
        ds_kwargs["min_len"] = int(dcfg["min_len"])
    if "normalize_env" in dcfg:
        ds_kwargs["normalize_env"] = bool(dcfg["normalize_env"])
    if "require_normalization_stats" in dcfg:
        ds_kwargs["require_normalization_stats"] = bool(
            dcfg["require_normalization_stats"]
        )
    if "subset_band_ids" in dcfg:
        ds_kwargs["subset_band_ids"] = dcfg["subset_band_ids"]

    test_split = resolve_default_eval_split(study)
    train_ds = study.moveformer_dataset(split="train", **ds_kwargs)
    test_ds = study.moveformer_dataset(split=test_split, **ds_kwargs)

    train_loader, test_loader = _make_loaders(
        ds_train=train_ds,
        ds_test=test_ds,
        collate_fn=collate_moveformer,
        batch_size=dcfg.get("batch_size", None),
        num_workers=dcfg.get("num_workers", None),
    )

    schedule = _resolve_fit_schedule(
        hparams=dict(method["hparams"]), steps_per_epoch=len(train_loader)
    )

    P = int(getattr(train_ds, "num_candidates_total", 0))
    if P <= 0:
        raise ValueError("MoveFormer dataset must expose num_candidates_total > 0")
    random_metrics = random_baseline_metrics(P, TOP_K_VALUES)

    cov_dim = int(getattr(train_ds, "cov_dim", 0))
    if cov_dim <= 0:
        raise ValueError("MoveFormer dataset must expose cov_dim > 0")
    model = build_moveformer_selection_model(cov_dim=cov_dim)

    max_eval_steps = _resolve_max_eval_steps(cfg)
    hparams = dict(method["hparams"])
    fit = fit_moveformer(
        model,
        train_loader,
        test_loader,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr=float(hparams.get("lr", 5e-5)),
        weight_decay=float(hparams.get("weight_decay", 0.0)),
        warmup_steps=int(hparams.get("warmup_steps", 1000)),
        lr_gamma=float(hparams.get("lr_gamma", 0.98)),
        max_steps=int(schedule["max_steps"]),
        eval_interval=int(schedule["eval_interval"]),
        patience=int(schedule["patience"]),
        max_eval_steps=max_eval_steps,
        log_callback=log_callback,
    )

    # Save the best checkpoint (fit_moveformer restores the best weights before returning).

    out_dir = Path(str(cfg["output_dir"]))
    ckpt_dir = out_dir / "checkpoints" / str(study.study_id)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    method_name = str(method["name"])
    ckpt_path = ckpt_dir / f"{method_name}.pt"
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "meta": {
                "study_id": str(study.study_id),
                "method_name": method_name,
                "kind": str(method["kind"]),
                "num_candidates_total": int(P),
                "max_len": int(ds_kwargs.get("max_len", 500)),
                "best_step": int(fit["best_step"]),
                "best_test_nll": float(fit["val"]["nll"]),
            },
        },
        ckpt_path,
    )

    train_metrics = eval_moveformer(model, train_loader, device, topk=TOP_K_VALUES)
    test_metrics = eval_moveformer(model, test_loader, device, topk=TOP_K_VALUES)

    return {
        "P": int(P),
        "random": random_metrics,
        "schedule": schedule,
        "fit": fit,
        "train": train_metrics,
        "test": test_metrics,
        "test_split_key": str(test_split),
        "checkpoint_path": str(ckpt_path),
    }


def run_method(
    *,
    kind: str,
    study,
    sampler_hash: str,
    cfg: dict[str, Any],
    method: dict[str, Any],
    device,
    log_callback: Any | None = None,
):
    if kind == "issf_linear":
        return run_issf_linear(
            study=study,
            cfg=cfg,
            method=method,
            device=device,
            log_callback=log_callback,
        )
    if kind == "issf_mlp":
        return run_issf_mlp(
            study=study,
            cfg=cfg,
            method=method,
            device=device,
            log_callback=log_callback,
        )
    if kind == "deepssf":
        return run_deepssf(
            study=study,
            sampler_hash=sampler_hash,
            cfg=cfg,
            method=method,
            device=device,
            log_callback=log_callback,
        )
    if kind == "moveformer":
        return run_moveformer(
            study=study,
            sampler_hash=sampler_hash,
            cfg=cfg,
            method=method,
            device=device,
            log_callback=log_callback,
        )
    raise ValueError(f"Unknown kind={kind!r}")


# def write_outputs(*, results: dict[str, dict[str, Any]], out_dir: Path, cfg: dict[str, Any]) -> None:
#     import pandas as pd
#
#     out_dir.mkdir(parents=True, exist_ok=True)
#     (out_dir / "raw_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
#
#     issf_methods = [m["name"] for m in cfg["methods"] if m["kind"] in {"issf_linear", "issf_mlp", "moveformer"}]
#     deepssf_methods = [m["name"] for m in cfg["methods"] if m["kind"] == "deepssf"]
#
#     def _write_nll_csv(*, split: str, methods: list[str], prefix: str, random_key: str, random_label: str) -> None:
#         if not methods:
#             return
#         rows = []
#         for study_id, sr in results.items():
#             row: dict[str, Any] = {"study": study_id}
#
#             first = sr.get(methods[0], {})
#             r = first.get(random_key, {})
#             if isinstance(r, dict) and "nll" in r:
#                 row[random_label] = float(r["nll"])
#
#             for m in methods:
#                 if m in sr and split in sr[m]:
#                     row[m] = float(sr[m][split]["nll"])
#             rows.append(row)
#         df = pd.DataFrame(rows).set_index("study")
#         df.to_csv(out_dir / f"{prefix}nll_{split}.csv")
#
#     def _write_topk_csvs(
#         *,
#         split: str,
#         methods: list[str],
#         prefix: str,
#         random_key: str,
#         random_label: str,
#         topk: Sequence[int] = TOP_K_VALUES,
#     ) -> None:
#         if not methods:
#             return
#         for k in topk:
#             rows = []
#             for study_id, sr in results.items():
#                 row: dict[str, Any] = {"study": study_id}
#
#                 first = sr.get(methods[0], {})
#                 r = first.get(random_key, {})
#                 if isinstance(r, dict) and "topk" in r and int(k) in r["topk"]:
#                     row[random_label] = float(r["topk"][int(k)])
#
#                 for m in methods:
#                     if m in sr and split in sr[m]:
#                         row[m] = float(sr[m][split]["topk"][int(k)])
#                 rows.append(row)
#             df = pd.DataFrame(rows).set_index("study")
#             df.to_csv(out_dir / f"{prefix}top{int(k)}_{split}.csv")
#
#     _write_nll_csv(split="train", methods=issf_methods, prefix="", random_key="random", random_label="Random")
#     _write_nll_csv(split="test", methods=issf_methods, prefix="", random_key="random", random_label="Random")
#     _write_topk_csvs(split="train", methods=issf_methods, prefix="", random_key="random", random_label="Random")
#     _write_topk_csvs(split="test", methods=issf_methods, prefix="", random_key="random", random_label="Random")
#
#     _write_nll_csv(split="train", methods=deepssf_methods, prefix="deepssf_", random_key="random_grid", random_label="RandomGrid")
#     _write_nll_csv(split="test", methods=deepssf_methods, prefix="deepssf_", random_key="random_grid", random_label="RandomGrid")
#     _write_topk_csvs(
#         split="train",
#         methods=deepssf_methods,
#         prefix="deepssf_",
#         random_key="random_grid",
#         random_label="RandomGrid",
#     )
#     _write_topk_csvs(
#         split="test",
#         methods=deepssf_methods,
#         prefix="deepssf_",
#         random_key="random_grid",
#         random_label="RandomGrid",
#     )
#


def main(
    *,
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
    resume: bool = False,
) -> None:

    cfg = load_config(config_path)
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a dict when provided")
        cfg = _deep_update(cfg, overrides)

    studies = resolve_studies(cfg)
    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = out_dir / "raw_results.json"

    device = torch.device(str(cfg["runtime"]["device"]))

    results: dict[str, dict[str, Any]] = {}
    if resume and raw_results_path.exists():
        loaded = json.loads(raw_results_path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError(
                f"Expected raw_results.json to contain a dict; got {type(loaded).__name__}"
            )
        results = loaded  # type: ignore[assignment]

    skipped_studies: list[tuple[str, str]] = []

    for study_id in studies:
        existing_study = results.get(study_id, {})
        if not isinstance(existing_study, dict):
            existing_study = {}

        def _is_complete(x: Any) -> bool:
            return isinstance(x, dict) and "fit" in x and "test" in x

        if resume:
            needs_any = any(
                not _is_complete(existing_study.get(str(m["name"])))
                for m in cfg["methods"]
            )
            if not needs_any:
                print(f"[{study_id}] all methods already present; skipping (--resume)")
                continue

        try:
            study = Study(study_id)
        except FileNotFoundError as e:
            msg = f"missing study files: {e}"
            skipped_studies.append((str(study_id), msg))
            print(f"[{study_id}] WARNING: {msg} (skipping)", flush=True)
            continue

        try:
            sampler_hash = ensure_study_ready(study=study, cfg=cfg)
        except (FileNotFoundError, ValueError) as e:
            msg = f"study setup failed: {type(e).__name__}: {e}"
            skipped_studies.append((str(study_id), msg))
            print(f"[{study_id}] WARNING: {msg} (skipping)", flush=True)
            continue

        study_results: dict[str, Any] = dict(existing_study)
        for method in cfg["methods"]:
            method_name = str(method["name"])
            if resume and _is_complete(study_results.get(method_name)):
                print(f"[{study_id}] {method_name} exists; skipping (--resume)")
                continue
            kind = str(method["kind"])

            print(f"Running {study_id} {method_name}")

            try:
                study_results[method_name] = run_method(
                    kind=kind,
                    study=study,
                    sampler_hash=str(sampler_hash),
                    cfg=cfg,
                    method=method,
                    device=device,
                )
            except (FileNotFoundError, ValueError) as e:
                msg = f"{method_name} failed: {type(e).__name__}: {e}"
                skipped_studies.append((str(study_id), msg))
                print(f"[{study_id}] WARNING: {msg} (skipping method)", flush=True)
                continue
            results[study_id] = study_results
            raw_results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(f"[{study_id}] {method_name} done")

        results[study_id] = study_results
        # write_outputs(results=results, out_dir=out_dir, cfg=cfg)

    # write_outputs(results=results, out_dir=out_dir, cfg=cfg)
