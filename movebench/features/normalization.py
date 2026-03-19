from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from movebench.features.manifest import read_manifest, utc_now_iso
from movebench.features.points import (
    point_store_dir,
    point_stores_dir,
    point_store_train_stats_path,
)
from movebench.layers.plan import LayerPlan


def _resolve_sampler_hash(
    *, study_dir: Path, rasters_hash: str, sampler_hash: str | None
) -> str:
    if sampler_hash is not None:
        return str(sampler_hash)

    base_dir = point_stores_dir(
        study_dir=study_dir, split="train", rasters_hash=rasters_hash
    )
    if not base_dir.exists():
        raise FileNotFoundError(
            "Normalization stats not found. Run "
            "Study.compute_normalization_vals(from_='points', ...) first. "
            f"Expected at least one sampler directory under: {base_dir}"
        )
    candidates = sorted([p.name for p in base_dir.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(
            "Normalization stats not found. Run "
            "Study.compute_normalization_vals(from_='points', ...) first. "
            f"Expected at least one sampler directory under: {base_dir}"
        )
    if len(candidates) > 1:
        raise ValueError(
            "Multiple normalization sampler directories found under "
            f"{base_dir}: {candidates}. Pass sampler_hash=... to disambiguate."
        )
    return candidates[0]


def load_points_normalization_mean_std(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_hash: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    study_dir = Path(study_dir)
    sampler_hash = _resolve_sampler_hash(
        study_dir=study_dir,
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=sampler_hash,
    )
    stats_path = point_store_train_stats_path(
        study_dir=study_dir,
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
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
    mean = torch.zeros(int(plan.num_channels()), dtype=torch.float32)
    std = torch.ones(int(plan.num_channels()), dtype=torch.float32)
    for i, ch in enumerate(plan.channels):
        if not ch.normalize:
            continue
        band_id = band_ids[i]
        st = band_stats.get(band_id)
        if st is None:
            raise ValueError(
                f"Missing normalization stats for band_id={band_id!r} in {stats_path}"
            )
        mean[i] = float(st["mean"])
        s = float(st["std"])
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        std[i] = s
    return mean, std


def compute_points_normalization_stats(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_hash: str | None = None,
    split: str = "train",
    skip_existing: bool = True,
) -> dict[str, Any]:
    study_dir = Path(study_dir)
    split = str(split)
    if split != "train":
        raise ValueError("Point normalization stats are stored only for split='train'")

    sampler_hash = _resolve_sampler_hash(
        study_dir=study_dir,
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=sampler_hash,
    )
    store_dir = point_store_dir(
        study_dir=study_dir,
        split="train",
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )
    manifest_path = store_dir / "manifest.json"
    shared_path = store_dir / "shared.pt"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    if not shared_path.exists():
        raise FileNotFoundError(f"shared.pt not found: {shared_path}")
    manifest = read_manifest(manifest_path)
    stats_path = point_store_train_stats_path(
        study_dir=study_dir,
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )

    existing: dict[str, Any] = {}
    band_stats: dict[str, dict[str, Any]] = {}
    if stats_path.exists():
        with open(stats_path, "r") as f:
            existing = json.load(f)
        band_stats = {
            str(k): dict(v) for k, v in existing.get("band_stats", {}).items()
        }

    normalize_band_ids = [ch.band_id for ch in plan.channels if ch.normalize]
    wanted = [band_id for band_id in normalize_band_ids if band_id in manifest.bands]
    if skip_existing:
        wanted = [band_id for band_id in wanted if band_id not in band_stats]

    for band_id in wanted:
        band_path = store_dir / str(manifest.bands[band_id]["path"])
        d = torch.load(band_path, map_location="cpu")
        env = d["env"]
        valid = d["valid"]
        count = int(valid.sum().item())
        if count == 0:
            mean = 0.0
            std = 1.0
        else:
            vals = env[valid]
            mean = float(vals.to(dtype=torch.float64).mean().item())
            std = float(vals.to(dtype=torch.float64).std(unbiased=False).item())
            if not math.isfinite(std) or std < 1e-8:
                std = 1.0
        band_stats[band_id] = {"mean": mean, "std": std, "count": count}

    stats = {
        "schema_version": "2",
        "created_at": utc_now_iso(),
        "source": "points",
        "source_split": "train",
        "scope": "all_candidates",
        "ddof": 0,
        "rasters_hash": str(plan.rasters_hash),
        "sampler_hash": str(sampler_hash),
        "band_stats": band_stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    return {
        "status": "existing" if existing and not wanted else "written",
        "stats_path": str(stats_path),
        "rasters_hash": str(plan.rasters_hash),
        "sampler_hash": str(sampler_hash),
        "n_bands_normalized": int(len(band_stats)),
    }
