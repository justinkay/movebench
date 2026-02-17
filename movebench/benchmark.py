"""YAML-driven benchmark runner for MoveBench baselines.

This module intentionally keeps the benchmarking "framework" minimal. It is a
single file with a small set of functions for:
- config loading/validation,
- per-study setup (rasters/feature stores/normalization),
- per-method training/evaluation,
- writing JSON + CSV summaries.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


TOP_K_VALUES = (1, 5, 10, 25, 50)


_DEFAULT_CFG: dict[str, Any] = {
    "version": 1,
    "layers_config": "configs/layers_all.yaml",
    "output_dir": "results/benchmarks",
    "runtime": {"device": "cpu", "seed": 0, "setup_workers": 16},
    "datasets": {
        "issf": {
            "batch_size": 32,
            "num_workers": 0,
            "normalize": True,
            "include_time": False,
            "subset_env_layers": None,
            "subset_band_ids": None,
            "sampler": {
                "num_candidates": 100,
                "random_seed": 0,
                "step_multiplier": 2.0,
                "step_q": 0.95,
                "gap_q": 0.999,
                "bounds_buffer_deg": 0.01,
            },
        },
        "deepssf": {
            "batch_size": 16,
            "num_workers": 0,
            "output_hw": (101, 101),
            "normalize": True,
            "step_multiplier": 2.0,
            "step_q": 0.95,
            "gap_q": 0.999,
        },
    },
}


_DEFAULT_METHODS: dict[str, dict[str, Any]] = {
    "issf_linear": {
        "model": {},
        "hparams": {"lr": 1e-2, "weight_decay": 1e-3, "max_steps": 1000, "eval_interval": 10, "patience": 50},
    },
    "issf_mlp": {
        "model": {"hidden_dims": (128,), "dropout": 0.6},
        "hparams": {"lr": 1e-3, "weight_decay": 1e-4, "max_steps": 2000, "eval_interval": 10, "patience": 50},
    },
    "deepssf": {
        "model": {"image_dim": 101},
        "hparams": {
            "lr_movement": 1e-5,
            "lr_habitat": 1e-4,
            "weight_decay": 0.0,
            "max_steps": 2000,
            "eval_interval": 10,
            "patience": 50,
        },
    },
}


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
        raise ValueError(f"Benchmark config must be a mapping/dict at the top level; got {type(user).__name__}")

    cfg = _deep_update(_DEFAULT_CFG, user)

    if int(cfg.get("version", -1)) != 1:
        raise ValueError(f"Unsupported benchmark config version={cfg.get('version')!r} (expected 1)")

    for key in ("layers_config", "output_dir", "runtime", "datasets", "methods"):
        if key not in cfg:
            raise ValueError(f"Benchmark config missing required key: {key!r}")

    if not isinstance(cfg["methods"], list) or not cfg["methods"]:
        raise ValueError("Benchmark config must include a non-empty `methods:` list.")

    # Apply env overrides early (can affect paths/study discovery).
    env = cfg.get("env")
    if env is not None:
        if not isinstance(env, dict):
            raise ValueError("`env` must be a mapping of environment variable overrides.")
        for k, v in env.items():
            os.environ[str(k)] = str(v)

    # Validate + fill method defaults.
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
        if kind not in _DEFAULT_METHODS:
            allowed = sorted(_DEFAULT_METHODS.keys())
            raise ValueError(f"Unknown method kind={kind!r}. Allowed: {allowed}")

        model_cfg = m.get("model") or {}
        hparams = m.get("hparams") or {}
        if not isinstance(model_cfg, dict) or not isinstance(hparams, dict):
            raise ValueError(f"Method {name!r}: `model` and `hparams` must be dicts.")

        defaults = _DEFAULT_METHODS[kind]
        model_cfg = _deep_update(defaults["model"], model_cfg)
        hparams = _deep_update(defaults["hparams"], hparams)

        methods_out.append({"name": name, "kind": kind, "model": model_cfg, "hparams": hparams})

    cfg["methods"] = methods_out
    return cfg


def resolve_studies(cfg: dict[str, Any]) -> list[str]:
    from movebench import Study
    from movebench import paths

    studies_val = cfg.get("studies", None)
    if studies_val is None or (isinstance(studies_val, str) and studies_val.strip().lower() == "all"):
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


def ensure_study_ready(*, study, cfg: dict[str, Any]) -> str:
    """Ensure rasters, parametric point store, and normalization stats exist; return sampler_hash."""
    from movebench.features.points import select_point_store_dir

    layers_config = str(cfg["layers_config"])
    setup_workers = int(cfg["runtime"]["setup_workers"])
    sampler = dict(cfg["datasets"]["issf"]["sampler"])

    plan = study.layer_plan(layers_config)

    try:
        _train_dir, train_m = select_point_store_dir(
            study_dir=study.study_dir,
            split="train",
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=int(sampler["num_candidates"]),
            random_seed=int(sampler["random_seed"]),
            step_multiplier=float(sampler["step_multiplier"]),
            step_q=float(sampler["step_q"]),
            gap_q=float(sampler["gap_q"]),
            bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
        )
        sampler_hash = str(train_m.sampler_hash)

        # Also require the test split point store (benchmark evaluates on test).
        select_point_store_dir(
            study_dir=study.study_dir,
            split="test",
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=int(sampler["num_candidates"]),
            random_seed=int(sampler["random_seed"]),
            step_multiplier=float(sampler["step_multiplier"]),
            step_q=float(sampler["step_q"]),
            gap_q=float(sampler["gap_q"]),
            bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
        )
    except FileNotFoundError:
        try:
            rep = study.extract_point_features(
                config=layers_config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(sampler["num_candidates"]),
                random_seed=int(sampler["random_seed"]),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(sampler["step_multiplier"]),
                step_q=float(sampler["step_q"]),
                gap_q=float(sampler["gap_q"]),
                bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
            )
        except FileNotFoundError as e:
            if "rasters" not in str(e):
                raise
            study.download_rasters(
                config=layers_config,
                splits=["train", "test"],
                n_workers=int(setup_workers),
                skip_existing=True,
            )
            rep = study.extract_point_features(
                config=layers_config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(sampler["num_candidates"]),
                random_seed=int(sampler["random_seed"]),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(sampler["step_multiplier"]),
                step_q=float(sampler["step_q"]),
                gap_q=float(sampler["gap_q"]),
                bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
            )
        sampler_hash = str(rep["sampler_hash"])

    study.compute_normalization_vals(
        from_="points",
        config=layers_config,
        sampler_hash=sampler_hash,
        split="train",
        skip_existing=True,
    )
    return sampler_hash


def _make_issf_loaders(*, ds_train, ds_test, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    from movebench.train import collate_step_select

    train_loader = DataLoader(
        ds_train,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_step_select,
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_step_select,
        drop_last=False,
    )

    # Verify we can draw at least one batch (helps catch empty splits early).
    x0, _valid0 = next(iter(train_loader))
    if x0.ndim != 3:
        raise ValueError(f"iSSF dataset must produce x with shape (B,C,P); got {tuple(x0.shape)}")
    return train_loader, test_loader


def _make_deepssf_loaders(*, ds_train, ds_test, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    from movebench.train import collate_deepssf

    train_loader = DataLoader(
        ds_train,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_deepssf,
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_deepssf,
        drop_last=False,
    )

    x1, x2, x3, y = next(iter(train_loader))
    if x1.ndim != 4 or y.ndim != 3:
        raise ValueError("DeepSSFDataset must produce (spatial, scalars, bearing, target) tensors.")
    return train_loader, test_loader


def run_issf_linear(*, study, cfg: dict[str, Any], method: dict[str, Any], device) -> dict[str, Any]:
    from movebench.models.ssf import LinearStepSelect, random_baseline_metrics
    from movebench.train import eval_step_select, fit_step_select

    issf = cfg["datasets"]["issf"]
    sampler = issf["sampler"]

    train_ds = study.issf_dataset(
        split="train",
        config=str(cfg["layers_config"]),
        include_time=bool(issf.get("include_time", False)),
        normalize=bool(issf.get("normalize", True)),
        subset_env_layers=issf.get("subset_env_layers", None),
        subset_band_ids=issf.get("subset_band_ids", None),
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )
    test_ds = study.issf_dataset(
        split="test",
        config=str(cfg["layers_config"]),
        include_time=bool(issf.get("include_time", False)),
        normalize=bool(issf.get("normalize", True)),
        subset_env_layers=issf.get("subset_env_layers", None),
        subset_band_ids=issf.get("subset_band_ids", None),
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )

    train_loader, test_loader = _make_issf_loaders(
        ds_train=train_ds,
        ds_test=test_ds,
        batch_size=int(issf["batch_size"]),
        num_workers=int(issf["num_workers"]),
    )

    x0, _valid0 = next(iter(train_loader))
    _b, c, p = x0.shape
    random_metrics = random_baseline_metrics(int(p), TOP_K_VALUES)

    model = LinearStepSelect(int(c))
    fit = fit_step_select(
        model,
        train_loader,
        test_loader,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr=float(method["hparams"]["lr"]),
        weight_decay=float(method["hparams"]["weight_decay"]),
        max_steps=int(method["hparams"]["max_steps"]),
        eval_interval=int(method["hparams"]["eval_interval"]),
        patience=int(method["hparams"]["patience"]),
    )
    train_metrics = eval_step_select(model, train_loader, device, topk=TOP_K_VALUES)
    test_metrics = eval_step_select(model, test_loader, device, topk=TOP_K_VALUES)

    return {"P": int(p), "random": random_metrics, "fit": fit, "train": train_metrics, "test": test_metrics}


def run_issf_mlp(*, study, cfg: dict[str, Any], method: dict[str, Any], device) -> dict[str, Any]:
    from movebench.models.ssf import MLPStepSelect, random_baseline_metrics
    from movebench.train import eval_step_select, fit_step_select

    issf = cfg["datasets"]["issf"]
    sampler = issf["sampler"]

    train_ds = study.issf_dataset(
        split="train",
        config=str(cfg["layers_config"]),
        include_time=bool(issf.get("include_time", False)),
        normalize=bool(issf.get("normalize", True)),
        subset_env_layers=issf.get("subset_env_layers", None),
        subset_band_ids=issf.get("subset_band_ids", None),
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )
    test_ds = study.issf_dataset(
        split="test",
        config=str(cfg["layers_config"]),
        include_time=bool(issf.get("include_time", False)),
        normalize=bool(issf.get("normalize", True)),
        subset_env_layers=issf.get("subset_env_layers", None),
        subset_band_ids=issf.get("subset_band_ids", None),
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )

    train_loader, test_loader = _make_issf_loaders(
        ds_train=train_ds,
        ds_test=test_ds,
        batch_size=int(issf["batch_size"]),
        num_workers=int(issf["num_workers"]),
    )

    x0, _valid0 = next(iter(train_loader))
    _b, c, p = x0.shape
    random_metrics = random_baseline_metrics(int(p), TOP_K_VALUES)

    model_cfg = method["model"]
    hidden_dims = tuple(int(x) for x in model_cfg.get("hidden_dims", (128,)))
    dropout = float(model_cfg.get("dropout", 0.6))

    model = MLPStepSelect(int(c), hidden_dims=hidden_dims, dropout=dropout)
    fit = fit_step_select(
        model,
        train_loader,
        test_loader,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr=float(method["hparams"]["lr"]),
        weight_decay=float(method["hparams"]["weight_decay"]),
        max_steps=int(method["hparams"]["max_steps"]),
        eval_interval=int(method["hparams"]["eval_interval"]),
        patience=int(method["hparams"]["patience"]),
    )
    train_metrics = eval_step_select(model, train_loader, device, topk=TOP_K_VALUES)
    test_metrics = eval_step_select(model, test_loader, device, topk=TOP_K_VALUES)

    return {"P": int(p), "random": random_metrics, "fit": fit, "train": train_metrics, "test": test_metrics}


def run_deepssf(*, study, sampler_hash: str, cfg: dict[str, Any], method: dict[str, Any], device) -> dict[str, Any]:
    from movebench.models.deepssf import DeepSSF
    from movebench.train import eval_deepssf, fit_deepssf, random_grid_baseline_metrics

    dcfg = cfg["datasets"]["deepssf"]

    output_hw_raw = dcfg.get("output_hw", (101, 101))
    if isinstance(output_hw_raw, (list, tuple)) and len(output_hw_raw) == 2:
        output_hw = (int(output_hw_raw[0]), int(output_hw_raw[1]))
    else:
        raise ValueError("datasets.deepssf.output_hw must be a 2-item sequence like [101,101].")

    train_ds = study.deepssf_dataset(
        split="train",
        config=str(cfg["layers_config"]),
        output_hw=output_hw,
        normalize=bool(dcfg.get("normalize", True)),
        normalization_sampler_hash=str(sampler_hash),
        step_multiplier=float(dcfg["step_multiplier"]),
        step_q=float(dcfg["step_q"]),
        gap_q=float(dcfg["gap_q"]),
    )
    test_ds = study.deepssf_dataset(
        split="test",
        config=str(cfg["layers_config"]),
        output_hw=output_hw,
        normalize=bool(dcfg.get("normalize", True)),
        normalization_sampler_hash=str(sampler_hash),
        step_multiplier=float(dcfg["step_multiplier"]),
        step_q=float(dcfg["step_q"]),
        gap_q=float(dcfg["gap_q"]),
    )

    train_loader, test_loader = _make_deepssf_loaders(
        ds_train=train_ds,
        ds_test=test_ds,
        batch_size=int(dcfg["batch_size"]),
        num_workers=int(dcfg["num_workers"]),
    )

    x1, _x2, _x3, _y = next(iter(train_loader))
    b, c_env, h, w = x1.shape
    if int(h) != int(w):
        raise ValueError(f"deepSSF expects square chips. Got H={h}, W={w}.")

    model_cfg = method["model"]
    image_dim = int(model_cfg.get("image_dim", int(h)))
    if int(image_dim) != int(h):
        raise ValueError(f"deepSSF model.image_dim ({image_dim}) must match dataset chip size ({h}).")

    model = DeepSSF(
        int(c_env),
        image_dim=int(image_dim),
        pixel_size_m=float(train_ds.pixel_size_m_estimate),
        device=device,
    )

    pixel_size_test = float(test_ds.pixel_size_m_estimate)
    pixel_size_train = float(model.pixel_size_m)
    if not math.isclose(pixel_size_test, pixel_size_train, rel_tol=1e-6, abs_tol=0.0):
        raise ValueError(
            "deepSSF requires consistent chip grid scale across splits. "
            f"Got pixel_size_m train={pixel_size_train} test={pixel_size_test}."
        )

    fit = fit_deepssf(
        model,
        train_loader,
        test_loader,
        device,
        seed=int(cfg["runtime"]["seed"]),
        lr_movement=float(method["hparams"]["lr_movement"]),
        lr_habitat=float(method["hparams"]["lr_habitat"]),
        weight_decay=float(method["hparams"]["weight_decay"]),
        max_steps=int(method["hparams"]["max_steps"]),
        eval_interval=int(method["hparams"]["eval_interval"]),
        patience=int(method["hparams"]["patience"]),
    )

    train_metrics = eval_deepssf(model, train_loader, device, topk=TOP_K_VALUES)
    test_metrics = eval_deepssf(model, test_loader, device, topk=TOP_K_VALUES)
    random_grid = random_grid_baseline_metrics(int(h) * int(w), TOP_K_VALUES)

    return {
        "grid": {"H": int(h), "W": int(w), "num_pixels": int(h) * int(w), "pixel_size_m": float(model.pixel_size_m)},
        "random_grid": random_grid,
        "fit": fit,
        "train": train_metrics,
        "test": test_metrics,
    }


def run_method(*, kind: str, study, sampler_hash: str, cfg: dict[str, Any], method: dict[str, Any], device):
    if kind == "issf_linear":
        return run_issf_linear(study=study, cfg=cfg, method=method, device=device)
    if kind == "issf_mlp":
        return run_issf_mlp(study=study, cfg=cfg, method=method, device=device)
    if kind == "deepssf":
        return run_deepssf(study=study, sampler_hash=sampler_hash, cfg=cfg, method=method, device=device)
    raise ValueError(f"Unknown kind={kind!r}")


def write_outputs(*, results: dict[str, dict[str, Any]], out_dir: Path, cfg: dict[str, Any]) -> None:
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    issf_methods = [m["name"] for m in cfg["methods"] if m["kind"] in {"issf_linear", "issf_mlp"}]
    deepssf_methods = [m["name"] for m in cfg["methods"] if m["kind"] == "deepssf"]

    def _write_nll_csv(*, split: str, methods: list[str], prefix: str, random_key: str, random_label: str) -> None:
        if not methods:
            return
        rows = []
        for study_id, sr in results.items():
            row: dict[str, Any] = {"study": study_id}

            first = sr.get(methods[0], {})
            r = first.get(random_key, {})
            if isinstance(r, dict) and "nll" in r:
                row[random_label] = float(r["nll"])

            for m in methods:
                if m in sr and split in sr[m]:
                    row[m] = float(sr[m][split]["nll"])
            rows.append(row)
        df = pd.DataFrame(rows).set_index("study")
        df.to_csv(out_dir / f"{prefix}nll_{split}.csv")

    def _write_topk_csvs(
        *,
        split: str,
        methods: list[str],
        prefix: str,
        random_key: str,
        random_label: str,
        topk: Sequence[int] = TOP_K_VALUES,
    ) -> None:
        if not methods:
            return
        for k in topk:
            rows = []
            for study_id, sr in results.items():
                row: dict[str, Any] = {"study": study_id}

                first = sr.get(methods[0], {})
                r = first.get(random_key, {})
                if isinstance(r, dict) and "topk" in r and int(k) in r["topk"]:
                    row[random_label] = float(r["topk"][int(k)])

                for m in methods:
                    if m in sr and split in sr[m]:
                        row[m] = float(sr[m][split]["topk"][int(k)])
                rows.append(row)
            df = pd.DataFrame(rows).set_index("study")
            df.to_csv(out_dir / f"{prefix}top{int(k)}_{split}.csv")

    _write_nll_csv(split="train", methods=issf_methods, prefix="", random_key="random", random_label="Random")
    _write_nll_csv(split="test", methods=issf_methods, prefix="", random_key="random", random_label="Random")
    _write_topk_csvs(split="train", methods=issf_methods, prefix="", random_key="random", random_label="Random")
    _write_topk_csvs(split="test", methods=issf_methods, prefix="", random_key="random", random_label="Random")

    _write_nll_csv(split="train", methods=deepssf_methods, prefix="deepssf_", random_key="random_grid", random_label="RandomGrid")
    _write_nll_csv(split="test", methods=deepssf_methods, prefix="deepssf_", random_key="random_grid", random_label="RandomGrid")
    _write_topk_csvs(
        split="train",
        methods=deepssf_methods,
        prefix="deepssf_",
        random_key="random_grid",
        random_label="RandomGrid",
    )
    _write_topk_csvs(
        split="test",
        methods=deepssf_methods,
        prefix="deepssf_",
        random_key="random_grid",
        random_label="RandomGrid",
    )


def main(*, config_path: str | Path, overrides: dict[str, Any] | None = None) -> None:
    import torch

    from movebench import Study

    cfg = load_config(config_path)
    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError("overrides must be a dict when provided")
        cfg = _deep_update(cfg, overrides)

    studies = resolve_studies(cfg)
    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg["runtime"]["device"]))

    results: dict[str, dict[str, Any]] = {}

    for study_id in studies:
        study = Study(study_id)
        sampler_hash = ensure_study_ready(study=study, cfg=cfg)

        study_results: dict[str, Any] = {}
        for method in cfg["methods"]:
            name = str(method["name"])
            kind = str(method["kind"])
            study_results[name] = run_method(
                kind=kind,
                study=study,
                sampler_hash=str(sampler_hash),
                cfg=cfg,
                method=method,
                device=device,
            )
            print(f"[{study_id}] {name} done")

        results[study_id] = study_results

    write_outputs(results=results, out_dir=out_dir, cfg=cfg)
