from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from movebench.features.manifest import read_manifest, utc_now_iso
from movebench.layers.plan import LayerPlan


def load_points_normalization_mean_std(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_hash: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load per-channel mean/std vectors for env normalization from cached stats.json."""
    stats_path = (
        Path(study_dir)
        / "normalization"
        / "points"
        / str(plan.config_hash)
        / str(sampler_hash)
        / "stats.json"
    )
    if not stats_path.exists():
        raise FileNotFoundError(
            "Normalization stats not found. Run "
            "Study.compute_normalization_vals(from_='points', ...) first. "
            f"Expected: {stats_path}"
        )

    with open(stats_path, "r") as f:
        d = json.load(f)
    band_stats: dict[str, Any] = dict(d.get("band_stats", {}))

    band_ids = plan.band_ids_in_order()
    C = int(plan.num_channels())
    mean = torch.zeros(C, dtype=torch.float32)
    std = torch.ones(C, dtype=torch.float32)

    for i, ch in enumerate(plan.channels):
        if not ch.normalize:
            continue
        band_id = band_ids[i]
        st = band_stats.get(band_id)
        if st is None:
            raise ValueError(f"Missing normalization stats for band_id={band_id!r} in {stats_path}")
        mean[i] = float(st["mean"])
        s = float(st["std"])
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        std[i] = s

    return mean, std


def _select_sampler_dir(base_dir: Path, sampler_hash: str | None) -> tuple[str, Path]:
    if sampler_hash is not None:
        return str(sampler_hash), base_dir / str(sampler_hash)

    if not base_dir.exists():
        raise FileNotFoundError(f"Feature store directory not found: {base_dir}")

    candidates = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No sampler directories found under: {base_dir}")
    if len(candidates) == 1:
        sh = candidates[0]
        return sh, base_dir / sh
    # Caller can handle this case by iterating the returned candidates.
    raise ValueError(f"Multiple sampler directories found under {base_dir}: {candidates}")


def compute_points_normalization_stats(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_hash: str | None = None,
    split: str = "train",
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Compute per-band mean/std from an extracted point feature store on disk."""
    study_dir = Path(study_dir)
    base_dir = study_dir / "features" / "points" / str(split) / str(plan.config_hash)
    if sampler_hash is None:
        if not base_dir.exists():
            raise FileNotFoundError(f"Feature store directory not found: {base_dir}")
        candidates = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
        if not candidates:
            raise FileNotFoundError(f"No sampler directories found under: {base_dir}")
        if len(candidates) == 1:
            sampler_hash, in_dir = candidates[0], base_dir / candidates[0]
        else:
            # Compute stats for every sampler store present (empirical + parametric, etc).
            out = []
            for sh in candidates:
                out.append(
                    compute_points_normalization_stats(
                        study_dir=study_dir,
                        plan=plan,
                        sampler_hash=sh,
                        split=split,
                        skip_existing=skip_existing,
                    )
                )
            return {
                "status": "multiple",
                "config_hash": str(plan.config_hash),
                "split": str(split),
                "samplers": out,
            }
    else:
        sampler_hash, in_dir = _select_sampler_dir(base_dir, sampler_hash)

    in_pt = in_dir / "features.pt"
    in_manifest = in_dir / "manifest.json"
    if not in_pt.exists():
        raise FileNotFoundError(f"features.pt not found: {in_pt}")
    if not in_manifest.exists():
        raise FileNotFoundError(f"manifest.json not found: {in_manifest}")

    m = read_manifest(in_manifest)
    if m.config_hash != plan.config_hash:
        raise ValueError("Input manifest config_hash does not match LayerPlan.config_hash")
    if m.sampler_hash != sampler_hash:
        raise ValueError("Input manifest sampler_hash does not match the selected sampler_hash directory")

    out_dir = study_dir / "normalization" / "points" / str(plan.config_hash) / str(sampler_hash)
    out_path = out_dir / "stats.json"
    if skip_existing and out_path.exists():
        return {
            "status": "existing",
            "stats_path": str(out_path),
            "config_hash": str(plan.config_hash),
            "sampler_hash": str(sampler_hash),
        }

    d = torch.load(in_pt, map_location="cpu")
    if "env" not in d or "valid" not in d:
        raise ValueError(f"features.pt missing required keys 'env' and 'valid': {in_pt}")
    env = d["env"]
    valid = d["valid"]

    if env.ndim != 3 or valid.ndim != 3:
        raise ValueError(f"Expected env/valid rank-3 [N,C,P]; got env={tuple(env.shape)} valid={tuple(valid.shape)}")
    if env.shape != valid.shape:
        raise ValueError(f"env and valid shapes must match; got env={tuple(env.shape)} valid={tuple(valid.shape)}")
    if int(env.shape[1]) != int(plan.num_channels()):
        raise ValueError(f"env.shape[1] ({env.shape[1]}) must match plan.num_channels() ({plan.num_channels()})")

    normalize_idxs = [i for i, ch in enumerate(plan.channels) if ch.normalize]
    band_ids = plan.band_ids_in_order()

    if not normalize_idxs:
        stats = {
            "schema_version": "1",
            "created_at": utc_now_iso(),
            "source": "points",
            "source_split": str(split),
            "scope": "all_candidates",
            "ddof": 0,
            "config_hash": str(plan.config_hash),
            "sampler_hash": str(sampler_hash),
            "n_steps": int(env.shape[0]),
            "points_per_step": int(env.shape[2]),
            "band_stats": {},
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        return {
            "status": "written",
            "stats_path": str(out_path),
            "config_hash": str(plan.config_hash),
            "sampler_hash": str(sampler_hash),
            "n_bands_normalized": 0,
            "n_steps": int(env.shape[0]),
            "points_per_step": int(env.shape[2]),
        }

    env_sel = env[:, normalize_idxs, :]
    valid_sel = valid[:, normalize_idxs, :]

    count = valid_sel.sum(dim=(0, 2), dtype=torch.int64)  # (Cnorm,)
    count_f = count.to(dtype=torch.float64)

    valid_f = valid_sel.to(dtype=env_sel.dtype)
    sum_ = (env_sel * valid_f).sum(dim=(0, 2), dtype=torch.float64)
    sumsq = (env_sel.square() * valid_f).sum(dim=(0, 2), dtype=torch.float64)

    zero = (count == 0).nonzero(as_tuple=False).flatten().tolist()
    if zero:
        bad_band_ids = [band_ids[normalize_idxs[i]] for i in zero]
        raise ValueError(f"No valid points for normalized channel(s): {bad_band_ids}")

    mean = sum_ / count_f
    var = sumsq / count_f - mean.square()
    var = torch.clamp(var, min=0.0)
    std = torch.sqrt(var)

    band_stats: dict[str, dict[str, Any]] = {}
    for j, idx in enumerate(normalize_idxs):
        band_id = band_ids[idx]
        band_stats[band_id] = {
            "mean": float(mean[j].item()),
            "std": float(std[j].item()),
            "count": int(count[j].item()),
        }

    stats = {
        "schema_version": "1",
        "created_at": utc_now_iso(),
        "source": "points",
        "source_split": str(split),
        "scope": "all_candidates",
        "ddof": 0,
        "config_hash": str(plan.config_hash),
        "sampler_hash": str(sampler_hash),
        "n_steps": int(env.shape[0]),
        "points_per_step": int(env.shape[2]),
        "band_stats": band_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    return {
        "status": "written",
        "stats_path": str(out_path),
        "config_hash": str(plan.config_hash),
        "sampler_hash": str(sampler_hash),
        "n_bands_normalized": int(len(band_stats)),
        "n_steps": int(env.shape[0]),
        "points_per_step": int(env.shape[2]),
    }
