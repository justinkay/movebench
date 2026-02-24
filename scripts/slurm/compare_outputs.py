#!/usr/bin/env python
"""Evaluate SSF surface sampling outputs on Slurm (one study per job).

This script mirrors the logic in `notebooks/compare_ssf_outputs_fast.ipynb`,
but runs as a per-study Slurm worker.

Modes:
  - Launcher mode (default): resolves studies from a benchmark config and
    launches one `srun` worker per study with throttling.
  - Worker mode (`--study STUDY_ID`): runs surface sampling + metrics for one
    study and writes per-study outputs under `<output_dir>/compare_outputs/`.

Notes:
  - Assumes rasters, parametric point stores, and normalization stats already
    exist. Run `scripts/slurm/extract.py` first if needed.
  - Checkpoints are expected under `<output_dir>/checkpoints/<study_id>/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _SamplerParams:
    num_candidates: int
    random_seed: int
    step_multiplier: float
    step_q: float
    gap_q: float
    bounds_buffer_deg: float


def _parse_studies_override(val: str | None) -> tuple[str, list[str] | None]:
    if val is None:
        return "config_default", None
    v = str(val).strip()
    if not v:
        return "config_default", None
    if v.lower() == "all":
        return "all", None
    out = [p.strip() for p in v.split(",") if p.strip()]
    return "explicit", out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_config = _REPO_ROOT / "configs" / "benchmarks" / "all.yaml"

    p = argparse.ArgumentParser(description="Evaluate SSF surface samples per study on Slurm.")
    p.add_argument("--config", type=str, default=str(default_config), help="Benchmark YAML config path.")
    p.add_argument("--study", type=str, default=None, help="Worker mode: run a single study id.")
    p.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Launcher override: 'all' or comma-separated study ids. If omitted, use config's `studies` or all available.",
    )
    p.add_argument(
        "--checkpoint-root",
        type=str,
        default=None,
        help="Optional override for checkpoint directory (default: <output_dir>/checkpoints).",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for Slurm stdout/stderr logs (default: results/slurm/compare_outputs/<config_stem>/).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite per-study outputs if they exist.")
    p.add_argument("--dry-run", action="store_true", help="Print actions/commands but do not submit jobs.")

    # Sampling params (defaults match the notebook).
    p.add_argument("--issf-surface-num-candidates", type=int, default=10_000)
    p.add_argument("--issf-surface-num-samples", type=int, default=500)
    p.add_argument("--deepssf-surface-num-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=8, help="Max workers for threaded iSSF extraction/scoring.")
    p.add_argument("--max-steps", type=int, default=2500, help="Optional cap on number of steps (debug).")
    p.add_argument("--device", type=str, default=None, help="Override torch device (e.g. 'cuda' or 'cpu').")
    p.add_argument("--skip-energy", action="store_true", help="Skip energy-score computation (faster).")

    # Slurm defaults match other scripts in scripts/slurm.
    p.add_argument("--max-concurrent-jobs", type=int, default=32, help="Launcher: maximum concurrent `srun` jobs.")
    p.add_argument("--polling-interval", type=int, default=30, help="Launcher: seconds between job status checks.")
    p.add_argument("--partition", type=str, default="vision-beery", help="Slurm partition (-p).")
    p.add_argument("--qos", type=str, default="vision-beery-main", help="Slurm QoS (-q).")
    p.add_argument("--account", type=str, default="vision-beery", help="Slurm account (-A).")
    p.add_argument("--time", type=str, default="1-00:00:00", help="Slurm time limit (-t), e.g. '7-00:00:00'.")
    p.add_argument("--mem", type=str, default="64GB", help="Slurm memory request, e.g. '64GB'.")
    p.add_argument("--cpus-per-task", type=int, default=16, help="Slurm CPUs per task (--cpus-per-task).")
    p.add_argument("--gpus-per-node", type=int, default=1, help="Slurm GPUs per node (--gpus-per-node).")
    return p.parse_args(argv)


def _seed(*parts: object) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def _load_cfg(config_path: str) -> dict[str, Any]:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.benchmark import load_config

    p = Path(config_path).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return load_config(str(p))


def _resolve_layer_plan(layers_config: str):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.layers.config import load_layers_config
    from movebench.layers.registry import REGISTRY

    cfg = load_layers_config(layers_config)
    return REGISTRY.resolve(cfg)


def _uses_gee_layers(plan) -> bool:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.layers.registry import REGISTRY

    required = set(str(x) for x in plan.layers_in_order)
    for parents in (plan.derived_dependencies or {}).values():
        required.update(str(p) for p in parents)

    for layer in sorted(required):
        src = REGISTRY.get(str(layer)).source
        if src in {"gee_chip", "gee_tile"}:
            return True
    return False


def _issf_sampler_params(cfg: dict[str, Any]) -> _SamplerParams:
    sampler = cfg["datasets"]["issf"]["sampler"]
    return _SamplerParams(
        num_candidates=int(sampler["num_candidates"]),
        random_seed=int(sampler["random_seed"]),
        step_multiplier=float(sampler["step_multiplier"]),
        step_q=float(sampler["step_q"]),
        gap_q=float(sampler["gap_q"]),
        bounds_buffer_deg=float(sampler["bounds_buffer_deg"]),
    )


def _select_parametric_store(*, study_dir: Path, plan, split: str, sampler: _SamplerParams):
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.features.points import select_point_store_dir

    return select_point_store_dir(
        study_dir=Path(study_dir),
        split=str(split),
        config_hash=str(plan.config_hash),
        expected_type="parametric",
        num_candidates=int(sampler.num_candidates),
        random_seed=int(sampler.random_seed),
        step_multiplier=float(sampler.step_multiplier),
        step_q=float(sampler.step_q),
        gap_q=float(sampler.gap_q),
        bounds_buffer_deg=float(sampler.bounds_buffer_deg),
    )


def _norm_stats_path(*, study_dir: Path, config_hash: str, sampler_hash: str) -> Path:
    return Path(study_dir) / "normalization" / "points" / str(config_hash) / str(sampler_hash) / "stats.json"


def _compare_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "compare_outputs"


def _study_results_dir(out_dir: Path) -> Path:
    return _compare_dir(out_dir) / "slurm_studies"


def _study_results_path(out_dir: Path, study_id: str) -> Path:
    safe = str(study_id).replace(os.sep, "_")
    return _study_results_dir(out_dir) / f"{safe}.json"


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def _check_prereqs(
    *,
    study_id: str,
    cfg: dict[str, Any],
    plan,
    sampler: _SamplerParams,
    ckpt_root: Path,
) -> tuple[bool, str | None, str]:
    """Return (ok, sampler_hash, reason)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench import Study

    try:
        study = Study(str(study_id))
    except FileNotFoundError as e:
        return False, None, f"missing_study: {e}"

    if _uses_gee_layers(plan):
        rasters_root = study.study_dir / "rasters" / str(plan.rasters_hash)
        if not rasters_root.exists():
            return False, None, f"missing_rasters_root: {rasters_root}"

    try:
        _train_dir, train_m = _select_parametric_store(study_dir=study.study_dir, plan=plan, split="train", sampler=sampler)
        _select_parametric_store(study_dir=study.study_dir, plan=plan, split="test", sampler=sampler)
    except (FileNotFoundError, ValueError) as e:
        return False, None, f"missing_parametric_point_store: {e}"

    sampler_hash = str(train_m.sampler_hash)
    stats_path = _norm_stats_path(
        study_dir=study.study_dir,
        config_hash=str(plan.config_hash),
        sampler_hash=sampler_hash,
    )
    if not stats_path.exists():
        return False, None, f"missing_points_normalization_stats: {stats_path}"

    # Need train point store to compute movement stats used by iSSF features.
    train_features = Path(_train_dir) / "features.pt"
    if not train_features.exists():
        return False, None, f"missing_train_features_pt: {train_features}"

    # Require at least one checkpoint to exist; otherwise skip.
    if not ckpt_root.exists():
        return False, sampler_hash, f"missing_checkpoint_root: {ckpt_root}"

    return True, sampler_hash, "ok"


def _build_srun_cmd(
    *,
    args: argparse.Namespace,
    study_id: str,
    config_path: str,
    log_dir: Path,
) -> list[str]:
    script_path = Path(__file__).resolve()

    out_base = str(study_id).replace(os.sep, "_")
    out_path = log_dir / f"{out_base}.%j.out"
    err_path = log_dir / f"{out_base}.%j.err"

    cmd: list[str] = ["srun", "--export=ALL"]
    if args.partition:
        cmd.extend(["-p", str(args.partition)])
    if args.qos:
        cmd.extend(["-q", str(args.qos)])
    if args.account:
        cmd.extend(["-A", str(args.account)])
    if args.time:
        cmd.extend(["-t", str(args.time)])
    if args.mem:
        cmd.append(f"--mem={args.mem}")
    cmd.extend(["--cpus-per-task", str(int(args.cpus_per_task))])
    if int(getattr(args, "gpus_per_node", 0)) > 0:
        cmd.extend(["--gpus-per-node", str(int(args.gpus_per_node))])
    cmd.extend(["--output", str(out_path), "--error", str(err_path)])

    cmd.extend(
        [
            sys.executable,
            "-u",
            str(script_path),
            "--config",
            str(config_path),
            "--study",
            str(study_id),
            "--issf-surface-num-candidates",
            str(int(args.issf_surface_num_candidates)),
            "--issf-surface-num-samples",
            str(int(args.issf_surface_num_samples)),
            "--deepssf-surface-num-samples",
            str(int(args.deepssf_surface_num_samples)),
            "--seed",
            str(int(args.seed)),
            "--max-workers",
            str(int(args.max_workers)),
        ]
    )
    if args.max_steps is not None:
        cmd.extend(["--max-steps", str(int(args.max_steps))])
    if args.checkpoint_root:
        cmd.extend(["--checkpoint-root", str(args.checkpoint_root)])
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.skip_energy:
        cmd.append("--skip-energy")
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _movement_stats_from_train_store(*, study_dir: Path, plan_config_hash: str, sampler_hash: str) -> dict[str, float]:
    import torch

    pt = Path(study_dir) / "features" / "points" / "train" / str(plan_config_hash) / str(sampler_hash) / "features.pt"
    d = torch.load(pt, map_location="cpu")
    sl = d["step_length_m"][:, -1].to(dtype=torch.float64)
    ta = d["turn_angle_rad"][:, -1].to(dtype=torch.float64)
    log_sl, cos_ta = torch.log(sl + 1e-8), torch.cos(ta)

    def ms(x: torch.Tensor) -> tuple[float, float]:
        m, s = float(x.mean().item()), float(x.std(unbiased=False).item())
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        return m, s

    sl_m, sl_s = ms(sl)
    ls_m, ls_s = ms(log_sl)
    ct_m, ct_s = ms(cos_ta)
    return {
        "sl_mean": sl_m,
        "sl_std": sl_s,
        "log_sl_mean": ls_m,
        "log_sl_std": ls_s,
        "cos_ta_mean": ct_m,
        "cos_ta_std": ct_s,
    }


def _issf_step_x(
    *,
    env: Any,
    valid: Any,
    sl_m: Any,
    ta: Any,
    timestamp_ns: int,
    mv: dict[str, float],
    mean: Any,
    std: Any,
    issf_cfg: dict[str, Any],
):
    import pandas as pd
    import torch

    if bool(issf_cfg.get("include_individual_id", False)):
        raise ValueError("include_individual_id is not supported in this evaluation script")

    P, Cenv = int(env.shape[1]), int(env.shape[0])
    if bool(issf_cfg.get("normalize", True)):
        env = torch.where(valid, (env - mean.view(-1, 1)) / std.view(-1, 1), torch.zeros_like(env))

    move = torch.stack([sl_m, torch.log(sl_m + 1e-8), torch.cos(ta)], dim=0)
    if bool(issf_cfg.get("normalize_movement", True)):
        move[0] = (move[0] - float(mv["sl_mean"])) / float(mv["sl_std"])
        move[1] = (move[1] - float(mv["log_sl_mean"])) / float(mv["log_sl_std"])
        move[2] = (move[2] - float(mv["cos_ta_mean"])) / float(mv["cos_ta_std"])

    parts: list[torch.Tensor] = [env, move]
    valids: list[torch.Tensor] = [valid, torch.ones((3, P), dtype=torch.bool)]

    time = None
    if bool(issf_cfg.get("include_time", False)):
        ts = pd.to_datetime(int(timestamp_ns), unit="ns", utc=True)
        doy = int(ts.dayofyear)
        hf = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
        time = torch.tensor(
            [
                math.sin(2.0 * math.pi * doy / 365.25),
                math.cos(2.0 * math.pi * doy / 365.25),
                math.sin(2.0 * math.pi * hf / 24.0),
                math.cos(2.0 * math.pi * hf / 24.0),
            ],
            dtype=torch.float32,
        ).view(4, 1).expand(4, P)
        parts.append(time)
        valids.append(torch.ones((4, P), dtype=torch.bool))

    def inter(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        Ka, P = a.shape
        Kb = b.shape[0]
        return (a.unsqueeze(1) * b.unsqueeze(0)).reshape(Ka * Kb, P)

    parts.append(inter(move, env))
    valids.append(valid.unsqueeze(0).expand(3, -1, -1).reshape(3 * Cenv, P))

    if time is not None:
        parts.append(inter(time, env))
        valids.append(valid.unsqueeze(0).expand(4, -1, -1).reshape(4 * Cenv, P))
        parts.append(inter(time, move))
        valids.append(torch.ones((12, P), dtype=torch.bool))

    x, v = torch.cat(parts, dim=0), torch.cat(valids, dim=0)
    return torch.where(v, x, torch.zeros_like(x))


def _worker(args: argparse.Namespace) -> int:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    try:
        import numpy as np
        import pandas as pd
        import torch
        from pyproj import Transformer
        from tqdm.auto import tqdm
        from torch.utils.data import DataLoader, Subset
    except Exception as e:
        print(f"[compare_outputs] ERROR importing dependencies: {e}", file=sys.stderr)
        return 2

    from movebench import Study
    from movebench.datasets.deepssf import DeepSSFDataset
    from movebench.features.normalization import load_points_normalization_mean_std
    from movebench.models.deepssf import DeepSSF
    from movebench.models.ssf import LinearStepSelect, MLPStepSelect
    from movebench.trajectories import SET_COL, TIME_COL, POINT_ID_COL, compute_movement_features, select_step_starts
    from movebench.train import collate_deepssf, collate_moveformer
    import movebench.features.points as mb_points

    cfg = _load_cfg(str(args.config))
    out_dir = Path(str(cfg["output_dir"]))
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_dir).resolve()
    cmp_dir = _compare_dir(out_dir)
    cmp_dir.mkdir(parents=True, exist_ok=True)

    ckpt_root = Path(str(args.checkpoint_root)) if args.checkpoint_root else (out_dir / "checkpoints")
    methods = {str(m["kind"]): dict(m) for m in cfg.get("methods", []) if isinstance(m, dict) and "kind" in m and "name" in m}

    study_id = str(args.study)
    per_study_path = _study_results_path(out_dir, study_id)
    if per_study_path.exists() and not bool(args.overwrite):
        print(f"[compare_outputs] exists; skipping study={study_id}", flush=True)
        return 0
    study = Study(study_id)
    layers_config = Path(str(cfg["layers_config"]))
    if not layers_config.is_absolute():
        layers_config = (_REPO_ROOT / layers_config).resolve()
    plan = study.layer_plan(str(layers_config))

    # Resolve sampler hash from extracted point store manifest.
    sampler = _issf_sampler_params(cfg)
    ok, sampler_hash, reason = _check_prereqs(
        study_id=study_id,
        cfg=cfg,
        plan=plan,
        sampler=sampler,
        ckpt_root=ckpt_root,
    )
    if not ok or sampler_hash is None:
        print(
            f"[compare_outputs] ERROR: prerequisites not met for study={study_id!r}: {reason}\n"
            f"[compare_outputs] Hint: run /Users/justinkay/movebench/scripts/slurm/extract.py first (rasters/points/normalization).",
            file=sys.stderr,
        )
        return 2

    # Device.
    if args.device:
        device = torch.device(str(args.device))
    else:
        runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), dict) else {}
        device = torch.device(str(runtime.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    print("[compare_outputs] study:", study_id, "device:", device, flush=True)

    # Helper: checkpoint path based on benchmark method name.
    def ckpt_path(kind: str) -> Path:
        m = methods.get(kind)
        name = str(m["name"]) if isinstance(m, dict) and "name" in m else str(kind)
        return ckpt_root / study_id / f"{name}.pt"

    def load_model(kind: str, *, cov_dim: int | None = None) -> torch.nn.Module | None:
        p = ckpt_path(kind)
        if not p.exists():
            return None
        d = torch.load(p, map_location="cpu")
        meta = dict(d.get("meta", {}) or {})
        state = dict(d.get("state_dict", {}) or {})
        if kind == "issf_linear":
            m = LinearStepSelect(int(meta["in_channels"]))
        elif kind == "issf_mlp":
            mc = dict((methods.get(kind, {}) or {}).get("model", {}) or {})
            hd = tuple(int(x) for x in mc.get("hidden_dims", (128,)))
            dr = float(mc.get("dropout", 0.6))
            m = MLPStepSelect(int(meta["in_channels"]), hidden_dims=hd, dropout=dr)
        elif kind == "deepssf":
            m = DeepSSF(
                int(meta["env_channels"]),
                image_dim=int(meta["image_dim"]),
                pixel_size_m=float(meta["pixel_size_m"]),
                device=device,
            )
        elif kind == "moveformer":
            from movebench.models.moveformer import build_moveformer_selection_model

            cd = int(cov_dim or meta.get("cov_dim", 0) or 0)
            if cd <= 0:
                raise ValueError("Need cov_dim to load moveformer")
            m = build_moveformer_selection_model(cov_dim=cd)
        else:
            raise ValueError(kind)
        m.load_state_dict(state)
        return m.to(device).eval()

    # Load normalization + movement stats (from extracted point store).
    mean, std = load_points_normalization_mean_std(study_dir=study.study_dir, plan=plan, sampler_hash=str(sampler_hash))
    mv = _movement_stats_from_train_store(study_dir=study.study_dir, plan_config_hash=str(plan.config_hash), sampler_hash=str(sampler_hash))

    # Load available models.
    models: dict[str, torch.nn.Module] = {}
    for kind in ("issf_linear", "issf_mlp", "deepssf"):
        m = load_model(kind)
        if m is not None:
            models[kind] = m

    mf_test = None
    if ckpt_path("moveformer").exists():
        mfcfg = dict((cfg.get("datasets", {}) or {}).get("moveformer", {}) or {})
        s_cfg = cfg["datasets"]["issf"]["sampler"]
        mf_test = study.moveformer_dataset(
            split="test",
            config=str(cfg["layers_config"]),
            max_len=int(mfcfg.get("max_len", 500)),
            min_len=int(mfcfg.get("min_len", 2)),
            normalize_env=bool(mfcfg.get("normalize_env", True)),
            require_normalization_stats=bool(mfcfg.get("require_normalization_stats", False)),
            num_candidates=int(s_cfg["num_candidates"]),
            random_seed=int(s_cfg["random_seed"]),
            step_multiplier=float(s_cfg["step_multiplier"]),
            step_q=float(s_cfg["step_q"]),
            gap_q=float(s_cfg["gap_q"]),
            bounds_buffer_deg=float(s_cfg["bounds_buffer_deg"]),
        )
        m = load_model("moveformer", cov_dim=int(getattr(mf_test, "cov_dim", 0)))
        if m is not None:
            models["moveformer"] = m

    if not models:
        print(f"[compare_outputs] WARNING: no checkpoints found for study={study_id!r}; skipping", flush=True)
        return 0

    issf_cfg = cfg["datasets"]["issf"]
    s_cfg = issf_cfg["sampler"]

    # Load trajectories + step starts.
    df = pd.read_csv(study.trajectories_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])
    df_test = df[df[SET_COL] == "test"].copy()
    steps = select_step_starts(
        df_test.copy(),
        step_multiplier=float(s_cfg["step_multiplier"]),
        step_q=float(s_cfg["step_q"]),
        gap_q=float(s_cfg["gap_q"]),
    )
    if len(steps) == 0:
        print(f"[compare_outputs] WARNING: no test steps for study={study_id!r}; skipping", flush=True)
        return 0

    n_steps = int(len(steps))
    idxs = list(range(n_steps))
    if args.max_steps is not None:
        idxs = idxs[: max(0, int(args.max_steps))]
    else:
        k = min(int(args.max_steps), n)
        rng = np.random.default_rng(0)
        idxs = sorted(int(x) for x in rng.choice(n, size=k, replace=False).tolist())

    # Truth dx/dy in meters.
    to_m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = to_m.transform(steps["start_lon"].to_numpy(), steps["start_lat"].to_numpy())
    x1, y1 = to_m.transform(steps["end_lon"].to_numpy(), steps["end_lat"].to_numpy())
    truth = pd.DataFrame(
        {
            "study_id": study_id,
            "step_idx": np.arange(n_steps, dtype=int),
            "dx_true_m": np.asarray(x1) - np.asarray(x0),
            "dy_true_m": np.asarray(y1) - np.asarray(y0),
        }
    )

    # Build candidate sampler state (10k candidates) from train split movement features.
    df_tr = compute_movement_features(
        df[df[SET_COL] == "train"].copy(),
        step_multiplier=float(s_cfg["step_multiplier"]),
        step_q=float(s_cfg["step_q"]),
        gap_q=float(s_cfg["gap_q"]),
    )
    _sampler, sampler_state, _ = mb_points.build_train_sampler(
        df_tr,
        parametric=True,
        num_candidates=int(args.issf_surface_num_candidates),
        random_seed=int(s_cfg["random_seed"]),
        step_multiplier=float(s_cfg["step_multiplier"]),
        step_q=float(s_cfg["step_q"]),
        gap_q=float(s_cfg["gap_q"]),
    )

    # deepSSF dataset (for surface sampling).
    dcfg = cfg["datasets"]["deepssf"]
    chips_test = study.chip_dataset(
        split="test",
        config=str(layers_config),
        mode="steps",
        output_hw=tuple(int(x) for x in dcfg.get("output_hw", (101, 101))),
        normalize=bool(dcfg.get("normalize", True)),
        normalization_sampler_hash=str(sampler_hash),
        step_multiplier=float(dcfg.get("step_multiplier", 2.0)),
        step_q=float(dcfg.get("step_q", 0.95)),
        gap_q=float(dcfg.get("gap_q", 0.999)),
    )
    deepssf_test = DeepSSFDataset(chips_test)
    px = float(getattr(chips_test, "pixel_size_m", 1.0))

    surface_rows: list[dict[str, Any]] = []

    # --- iSSF surface sampling (threaded extract + score + sample) ---
    if ("issf_linear" in models) or ("issf_mlp" in models):
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tls = threading.local()
        layer_slices = {layer: plan.layer_slice(layer) for layer in plan.layers_in_order}

        def get_io():
            if not hasattr(tls, "io"):
                tls.io = study.layer_io(str(layers_config))
            return tls.io

        reqs = [
            mb_points._StepRequest(
                point_id_start=int(r.point_id_start),
                timestamp=pd.Timestamp(r.timestamp),
                start_lon=float(r.start_lon),
                start_lat=float(r.start_lat),
                start_bearing=float(r.start_bearing),
                end_lon=float(r.end_lon),
                end_lat=float(r.end_lat),
            )
            for r in steps.iloc[idxs].itertuples(index=False)
        ]

        def work(step_idx: int, req) -> list[dict[str, Any]]:
            env_np, valid_np, sl_np, ta_np, _pid, tsn = mb_points._extract_one_step(
                req,
                io=get_io(),
                plan=plan,
                layer_slices=layer_slices,
                sampler_state=sampler_state,
                bounds_buffer_deg=float(s_cfg["bounds_buffer_deg"]),
            )
            # Drop true endpoint (last candidate).
            env = torch.from_numpy(env_np).float()[:, :-1]
            valid = torch.from_numpy(valid_np).bool()[:, :-1]
            sl = np.asarray(sl_np[:-1], dtype=np.float32)
            ta = np.asarray(ta_np[:-1], dtype=np.float32)

            x_step = (
                _issf_step_x(
                    env=env,
                    valid=valid,
                    sl_m=torch.from_numpy(sl),
                    ta=torch.from_numpy(ta),
                    timestamp_ns=int(tsn),
                    mv=mv,
                    mean=mean,
                    std=std,
                    issf_cfg=issf_cfg,
                )
                .unsqueeze(0)
                .to(device)
            )

            headings = float(req.start_bearing) + ta
            dx_all = sl * np.sin(headings)
            dy_all = sl * np.cos(headings)

            out: list[dict[str, Any]] = []
            with torch.no_grad():
                for kind in ("issf_linear", "issf_mlp"):
                    if kind not in models:
                        continue
                    logits = models[kind](x_step).squeeze(0).detach().cpu()
                    prob = torch.softmax(logits, dim=0)
                    g = torch.Generator(device="cpu")
                    g.manual_seed(_seed(int(args.seed), study_id, int(step_idx), kind))
                    samp = torch.multinomial(
                        prob,
                        int(args.issf_surface_num_samples),
                        replacement=True,
                        generator=g,
                    ).numpy()
                    out.extend(
                        {
                            "study_id": study_id,
                            "step_idx": int(step_idx),
                            "kind": kind,
                            "dx_m": float(dx),
                            "dy_m": float(dy),
                        }
                        for dx, dy in zip(dx_all[samp], dy_all[samp])
                    )
            return out

        items = list(zip(idxs, reqs))
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
            futs = [ex.submit(work, int(step_idx), req) for step_idx, req in items]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="issf surface"):
                surface_rows.extend(fut.result())

    # --- deepSSF surface sampling (batched) ---
    if "deepssf" in models:
        nw_cfg = int(dcfg.get("num_workers", 0) or 0)
        nw = min(8, os.cpu_count() or 1) if nw_cfg <= 0 else min(8, os.cpu_count() or 1, nw_cfg)
        loader = DataLoader(
            Subset(deepssf_test, idxs),
            batch_size=int(dcfg.get("batch_size", 32)),
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_deepssf,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
        pos = 0
        with torch.no_grad():
            for x1b, x2b, x3b, _yb in tqdm(loader, desc="deepssf surface"):
                step_batch = idxs[pos : pos + int(x1b.shape[0])]
                pos += int(x1b.shape[0])
                out = models["deepssf"]((x1b.to(device), x2b.to(device), x3b.to(device)))
                scores = out[..., 0] + out[..., 1]
                B, H, W = scores.shape
                prob = torch.softmax(scores.reshape(int(B), int(H) * int(W)).detach().cpu(), dim=1)
                for bi, step_idx in enumerate(step_batch):
                    g = torch.Generator(device="cpu")
                    g.manual_seed(_seed(int(args.seed), study_id, int(step_idx), "deepssf"))
                    pix = torch.multinomial(
                        prob[int(bi)],
                        int(args.deepssf_surface_num_samples),
                        replacement=True,
                        generator=g,
                    ).numpy()
                    rr = (pix // int(W)).astype(int)
                    cc = (pix % int(W)).astype(int)
                    dx = (cc - (int(W) // 2)) * px
                    dy = ((int(H) // 2) - rr) * px
                    surface_rows.extend(
                        {"study_id": study_id, "step_idx": int(step_idx), "kind": "deepssf", "dx_m": float(ddx), "dy_m": float(ddy)}
                        for ddx, ddy in zip(dx, dy)
                    )

    # --- MoveFormer surface sampling ---
    if ("moveformer" in models) and (mf_test is not None):
        perm = mf_test._cand_perm.numpy()
        sl = mf_test.step_length_m.numpy().astype(np.float32, copy=False)[:, perm]
        ta = mf_test.turn_angle_rad.numpy().astype(np.float32, copy=False)[:, perm]
        sb = np.asarray(mf_test._starts.start_bearing, dtype=np.float32)
        heading = sb[:, None] + ta
        dx_cand = sl * np.sin(heading)
        dy_cand = sl * np.cos(heading)

        P = int(dx_cand.shape[1])
        logits_steps = np.full((n_steps, P), np.nan, dtype=np.float32)
        mask_steps = np.zeros((n_steps,), dtype=np.bool_)

        class _MoveFormerIdx:
            def __init__(self, base):
                self.base = base

            def __len__(self):
                return len(self.base)

            def __getitem__(self, i):
                return int(i), self.base[int(i)]

        def _collate_moveformer_idx(batch):
            seg_ids = [b[0] for b in batch]
            feats = [b[1] for b in batch]
            return seg_ids, collate_moveformer(feats)

        mfcfg = dict((cfg.get("datasets", {}) or {}).get("moveformer", {}) or {})
        nw_cfg = int(mfcfg.get("num_workers", 0) or 0)
        nw = min(8, os.cpu_count() or 1) if nw_cfg <= 0 else min(8, os.cpu_count() or 1, nw_cfg)
        loader = DataLoader(
            _MoveFormerIdx(mf_test),
            batch_size=int(mfcfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=nw,
            collate_fn=_collate_moveformer_idx,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=nw > 0,
        )
        with torch.no_grad():
            for seg_ids, batch in tqdm(loader, desc="moveformer logits"):
                features = {k: v.to(device) for k, v in batch.items()}
                logits, _loss, _metrics = models["moveformer"](features)
                mask = features.get("loss_mask", features["mask"]).to(torch.bool).detach().cpu().numpy()
                logits = logits.detach().cpu().numpy()
                for bi, seg_id in enumerate(seg_ids):
                    start, end = mf_test._segments[int(seg_id)]
                    L = int(end - start)
                    logits_steps[start:end] = logits[int(bi), :L]
                    mask_steps[start:end] = mask[int(bi), :L]

        for step_idx in tqdm(idxs, desc="moveformer surface"):
            if not bool(mask_steps[int(step_idx)]):
                continue
            prob = torch.softmax(torch.from_numpy(logits_steps[int(step_idx)]), dim=0)
            if int(prob.numel()) > 1:
                prob = prob.clone()
                prob[0] = 0.0
                sprob = prob.sum()
                if float(sprob) <= 0:
                    prob[1:] = 1.0
                    prob = prob / prob.sum()
                else:
                    prob = prob / sprob
            g = torch.Generator(device="cpu")
            g.manual_seed(_seed(int(args.seed), study_id, int(step_idx), "moveformer"))
            samp = torch.multinomial(prob, int(args.issf_surface_num_samples), replacement=True, generator=g).numpy()
            surface_rows.extend(
                {"study_id": study_id, "step_idx": int(step_idx), "kind": "moveformer", "dx_m": float(dx), "dy_m": float(dy)}
                for dx, dy in zip(dx_cand[int(step_idx), samp], dy_cand[int(step_idx), samp])
            )

    # --- Baselines ---
    d_test = compute_movement_features(
        df_test.copy(),
        step_multiplier=float(s_cfg["step_multiplier"]),
        step_q=float(s_cfg["step_q"]),
        gap_q=float(s_cfg["gap_q"]),
    )
    sl_prev = d_test.set_index(POINT_ID_COL)["step_length_m"]

    rw_state = dict(sampler_state)
    rw_state["num_candidates"] = int(args.issf_surface_num_samples)

    for step_idx in idxs:
        r = steps.iloc[int(step_idx)]
        b = float(r.start_bearing)

        surface_rows.append({"study_id": study_id, "step_idx": int(step_idx), "kind": "baseline_nomove", "dx_m": 0.0, "dy_m": 0.0})

        sl0 = float(sl_prev.get(int(r.point_id_start), np.nan))
        surface_rows.append(
            {"study_id": study_id, "step_idx": int(step_idx), "kind": "baseline_constvel", "dx_m": sl0 * math.sin(b), "dy_m": sl0 * math.cos(b)}
        )

        lens, turns = mb_points._sample_pseudo_steps(rw_state, int(r.point_id_start))
        headings = b + turns
        surface_rows.extend(
            {"study_id": study_id, "step_idx": int(step_idx), "kind": "baseline_randomwalk", "dx_m": float(dx), "dy_m": float(dy)}
            for dx, dy in zip(lens * np.sin(headings), lens * np.cos(headings))
        )

    # --- Write surface samples ---
    df_surf = pd.DataFrame(surface_rows)
    surf_path = cmp_dir / f"{study_id}_surface_samples.csv"
    df_surf.to_csv(surf_path, index=False)
    print("[compare_outputs] wrote:", surf_path, flush=True)

    # --- L2 summary ---
    df_surf_l2 = df_surf.merge(truth, on=["study_id", "step_idx"], how="left")
    df_surf_l2["l2_m"] = np.hypot(df_surf_l2["dx_m"] - df_surf_l2["dx_true_m"], df_surf_l2["dy_m"] - df_surf_l2["dy_true_m"])
    l2_summary = (
        df_surf_l2.groupby("kind", sort=False)["l2_m"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("mean", ascending=True)
        .reset_index(drop=True)
    )
    l2_path = cmp_dir / f"{study_id}_l2_summary.csv"
    l2_summary.to_csv(l2_path, index=False)
    print("[compare_outputs] wrote:", l2_path, flush=True)

    # --- Energy score (optional) ---
    energy_path = None
    energy_summary_path = None
    energy_summary = None

    if not bool(args.skip_energy):
        rows: list[dict[str, Any]] = []
        for (step_idx, kind), g in df_surf_l2.groupby(["step_idx", "kind"], sort=False):
            g = g.dropna(subset=["dx_m", "dy_m", "dx_true_m", "dy_true_m"])
            n = int(len(g))
            if n == 0:
                continue
            y = torch.tensor([g["dx_true_m"].iloc[0], g["dy_true_m"].iloc[0]], dtype=torch.float32)
            x = torch.tensor(g[["dx_m", "dy_m"]].to_numpy(), dtype=torch.float32)
            term1 = torch.linalg.norm(x - y, dim=1).mean()
            term2 = torch.tensor(0.0) if n == 1 else torch.cdist(x, x).mean()
            rows.append({"study_id": study_id, "step_idx": int(step_idx), "kind": str(kind), "n": n, "energy_l2": float(term1 - 0.5 * term2)})

        df_energy = pd.DataFrame(rows)
        energy_path = cmp_dir / f"{study_id}_energy_by_step.csv"
        df_energy.to_csv(energy_path, index=False)
        print("[compare_outputs] wrote:", energy_path, flush=True)

        energy_summary = (
            df_energy.groupby("kind", sort=False)["energy_l2"]
            .agg(["count", "mean", "median", "std"])
            .reset_index()
            .sort_values("mean", ascending=True)
            .reset_index(drop=True)
        )
        energy_summary_path = cmp_dir / f"{study_id}_energy_summary.csv"
        energy_summary.to_csv(energy_summary_path, index=False)
        print("[compare_outputs] wrote:", energy_summary_path, flush=True)

    # --- Per-study JSON record ---
    payload: dict[str, Any] = {
        "__meta__": {
            "study_id": study_id,
            "config_path": str(args.config),
            "output_dir": str(out_dir),
            "layers_config": str(cfg["layers_config"]),
            "config_hash": str(plan.config_hash),
            "sampler_hash": str(sampler_hash),
            "checkpoint_root": str(ckpt_root),
            "device": str(device),
            "issf_surface_num_candidates": int(args.issf_surface_num_candidates),
            "issf_surface_num_samples": int(args.issf_surface_num_samples),
            "deepssf_surface_num_samples": int(args.deepssf_surface_num_samples),
            "seed": int(args.seed),
            "max_workers": int(args.max_workers),
            "n_steps_total": int(n_steps),
            "n_steps_eval": int(len(idxs)),
        },
        "surface_samples_csv": str(surf_path),
        "l2_summary_csv": str(l2_path),
    }
    if energy_path is not None:
        payload["energy_by_step_csv"] = str(energy_path)
    if energy_summary_path is not None:
        payload["energy_summary_csv"] = str(energy_summary_path)

    _write_json_atomic(per_study_path, payload)
    print("[compare_outputs] wrote:", per_study_path, flush=True)

    return 0


def _merge_outputs(*, out_dir: Path) -> None:
    import pandas as pd

    sdir = _study_results_dir(out_dir)
    if not sdir.exists():
        return
    rows: list[dict[str, Any]] = []
    for p in sorted(sdir.glob("*.json")):
        try:
            d = _load_json(p)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        meta = d.get("__meta__", {}) if isinstance(d.get("__meta__", {}), dict) else {}
        sid = str(meta.get("study_id", p.stem))

        l2_csv = d.get("l2_summary_csv")
        if isinstance(l2_csv, str) and Path(l2_csv).exists():
            try:
                l2 = pd.read_csv(l2_csv)
                l2["study_id"] = sid
                l2["metric"] = "l2_m"
                rows.extend(l2.to_dict("records"))
            except Exception:
                pass

        e_csv = d.get("energy_summary_csv")
        if isinstance(e_csv, str) and Path(e_csv).exists():
            try:
                e = pd.read_csv(e_csv)
                e["study_id"] = sid
                e["metric"] = "energy_l2"
                rows.extend(e.to_dict("records"))
            except Exception:
                pass

    if not rows:
        return
    df = pd.DataFrame(rows)
    out = _compare_dir(out_dir) / "slurm_summary.csv"
    df.to_csv(out, index=False)
    print("[compare_outputs] wrote:", out, flush=True)


def _launcher(args: argparse.Namespace) -> int:
    cfg = _load_cfg(str(args.config))
    out_dir = Path(str(cfg["output_dir"]))
    if not out_dir.is_absolute():
        out_dir = (_REPO_ROOT / out_dir).resolve()

    config_stem = Path(str(args.config)).stem
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else (_REPO_ROOT / "results" / "slurm" / "compare_outputs" / str(config_stem))
    )
    if not log_dir.is_absolute():
        log_dir = (_REPO_ROOT / log_dir).resolve()

    ckpt_root = Path(str(args.checkpoint_root)) if args.checkpoint_root else (out_dir / "checkpoints")
    layers_config = Path(str(cfg["layers_config"]))
    if not layers_config.is_absolute():
        layers_config = (_REPO_ROOT / layers_config).resolve()
    plan = _resolve_layer_plan(str(layers_config))
    sampler = _issf_sampler_params(cfg)

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from movebench.benchmark import resolve_studies

    override_kind, override_list = _parse_studies_override(args.studies)
    if override_kind == "config_default":
        studies = resolve_studies(cfg)
    elif override_kind == "all":
        cfg2 = dict(cfg)
        cfg2["studies"] = "all"
        studies = resolve_studies(cfg2)
    else:
        assert override_list is not None
        studies = sorted(dict.fromkeys([str(s) for s in override_list]))

    blocked: list[tuple[str, str]] = []
    skipped: list[str] = []
    queue: list[str] = []

    for sid in studies:
        ok, _sh, reason = _check_prereqs(
            study_id=str(sid),
            cfg=cfg,
            plan=plan,
            sampler=sampler,
            ckpt_root=ckpt_root,
        )
        if not ok:
            blocked.append((str(sid), reason))
            continue

        if bool(args.overwrite):
            queue.append(str(sid))
            continue

        per_study_path = _study_results_path(out_dir, str(sid))
        if per_study_path.exists():
            skipped.append(str(sid))
        else:
            queue.append(str(sid))

    print(
        f"[compare_outputs] studies total={len(studies)} queued={len(queue)} skipped={len(skipped)} blocked={len(blocked)}",
        flush=True,
    )
    if blocked:
        for sid, reason in blocked:
            print(f"[compare_outputs] BLOCKED study={sid} reason={reason}", file=sys.stderr)

    if args.dry_run:
        if queue:
            print(
                f"[compare_outputs] Dry-run: would launch {len(queue)} jobs (max_concurrent={args.max_concurrent_jobs}).",
                flush=True,
            )
            for sid in queue[:10]:
                cmd = _build_srun_cmd(args=args, study_id=sid, config_path=str(args.config), log_dir=log_dir)
                print("[compare_outputs] ", shlex.join(cmd), flush=True)
            if len(queue) > 10:
                print(f"[compare_outputs] ... ({len(queue)-10} more)", flush=True)
        return 2 if blocked else 0

    if int(args.max_concurrent_jobs) <= 0:
        raise ValueError("--max-concurrent-jobs must be > 0")
    if int(args.polling_interval) <= 0:
        raise ValueError("--polling-interval must be > 0")

    log_dir.mkdir(parents=True, exist_ok=True)

    if not queue:
        try:
            _merge_outputs(out_dir=out_dir)
        except Exception as e:
            print(f"[compare_outputs] WARNING: failed to merge outputs: {e}", file=sys.stderr)
        return 2 if blocked else 0

    running: dict[str, subprocess.Popen] = {}
    failures: list[str] = []
    completed = 0
    total = len(queue)
    queue_idx = 0

    def _launch_one(study_id: str) -> None:
        cmd = _build_srun_cmd(args=args, study_id=study_id, config_path=str(args.config), log_dir=log_dir)
        print(f"[compare_outputs] Launching study={study_id}: {shlex.join(cmd)}", flush=True)
        proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
        running[study_id] = proc

    try:
        while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
            sid = queue[queue_idx]
            queue_idx += 1
            _launch_one(sid)

        while running or queue_idx < len(queue):
            time.sleep(int(args.polling_interval))

            for sid, proc in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                running.pop(sid, None)
                completed += 1
                if int(rc) != 0:
                    failures.append(sid)
                    print(f"[compare_outputs] FAILED study={sid} exit_code={rc}", file=sys.stderr, flush=True)
                else:
                    print(f"[compare_outputs] DONE study={sid}", flush=True)

            while queue_idx < len(queue) and len(running) < int(args.max_concurrent_jobs):
                sid = queue[queue_idx]
                queue_idx += 1
                _launch_one(sid)

            pending = total - (completed + len(running))
            print(
                f"[compare_outputs] Progress completed={completed}/{total} running={len(running)} pending={pending} failed={len(failures)}",
                flush=True,
            )
            if running:
                print(f"[compare_outputs] Running: {sorted(running.keys())}", flush=True)
    except KeyboardInterrupt:
        print("[compare_outputs] Interrupted. Not launching new jobs.", file=sys.stderr, flush=True)
        if running:
            print(f"[compare_outputs] Still running: {sorted(running.keys())}", file=sys.stderr, flush=True)
        return 130

    try:
        _merge_outputs(out_dir=out_dir)
    except Exception as e:
        print(f"[compare_outputs] WARNING: failed to merge outputs: {e}", file=sys.stderr)

    if failures or blocked:
        print(f"[compare_outputs] Finished with failures={len(failures)} blocked={len(blocked)}", file=sys.stderr, flush=True)
        if failures:
            print(f"[compare_outputs] Failed studies: {sorted(failures)}", file=sys.stderr, flush=True)
        return 2

    print("[compare_outputs] All jobs completed successfully.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.study:
        return _worker(args)
    return _launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
