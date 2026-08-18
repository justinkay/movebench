from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from movebench.features.config import SamplerConfig
from movebench.features.manifest import read_manifest, utc_now_iso
from movebench.features.points import (
    PointStoreReader,
    point_store_dir,
    point_store_train_stats_path,
    resolve_sampler_hash,
)
from movebench.geometry import LOCAL_GEOMETRY
from movebench.layers.plan import LayerPlan


def load_points_normalization_mean_std(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_config: SamplerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    study_dir = Path(study_dir)
    sampler_hash = resolve_sampler_hash(sampler_config)
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

    with open(stats_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if str(payload.get("schema_version")) != "4":
        raise ValueError(f"Unsupported normalization schema in {stats_path}")
    if payload.get("geometry") != LOCAL_GEOMETRY:
        raise ValueError(f"Unsupported normalization geometry in {stats_path}")
    if payload.get("rasters_hash") != str(plan.rasters_hash):
        raise ValueError(f"Normalization rasters_hash mismatch in {stats_path}")
    if payload.get("sampler_hash") != str(sampler_hash):
        raise ValueError(f"Normalization sampler_hash mismatch in {stats_path}")
    band_stats: dict[str, Any] = dict(payload.get("band_stats", {}))

    mean = torch.zeros(int(plan.num_channels()), dtype=torch.float32)
    std = torch.ones(int(plan.num_channels()), dtype=torch.float32)
    for idx, channel in enumerate(plan.channels):
        if not channel.normalize:
            continue
        stats = band_stats.get(channel.band_id)
        if stats is None:
            raise ValueError(
                f"Missing normalization stats for band_id={channel.band_id!r} in {stats_path}"
            )
        mean[idx] = float(stats["mean"])
        value = float(stats["std"])
        if not math.isfinite(value) or value < 1e-8:
            value = 1.0
        std[idx] = value
    return mean, std


def compute_points_normalization_stats(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_config: SamplerConfig,
    split: str = "train",
    skip_existing: bool = True,
) -> dict[str, Any]:
    study_dir = Path(study_dir)
    split = str(split)
    if split != "train":
        raise ValueError("Point normalization stats are stored only for split='train'")

    sampler_hash = resolve_sampler_hash(sampler_config)
    store_dir = point_store_dir(
        study_dir=study_dir,
        split="train",
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = read_manifest(manifest_path)
    reader = PointStoreReader(store_dir=store_dir, manifest=manifest)

    stats_path = point_store_train_stats_path(
        study_dir=study_dir,
        rasters_hash=str(plan.rasters_hash),
        sampler_hash=str(sampler_hash),
    )
    existing: dict[str, Any] = {}
    band_stats: dict[str, dict[str, Any]] = {}
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        band_stats = {
            str(key): dict(value) for key, value in existing.get("band_stats", {}).items()
        }

    wanted = [
        channel.band_id
        for channel in plan.channels
        if channel.normalize and channel.band_id in manifest.bands
    ]
    if skip_existing:
        wanted = [band_id for band_id in wanted if band_id not in band_stats]

    if wanted:
        sums = torch.zeros(len(wanted), dtype=torch.float64)
        sumsq = torch.zeros(len(wanted), dtype=torch.float64)
        counts = torch.zeros(len(wanted), dtype=torch.float64)
        for block in reader.iter_row_blocks(
            band_ids=wanted,
            shared_keys=[],
            block_size=2048,
        ):
            env = block["env"].to(dtype=torch.float64)
            valid = block["valid"].to(dtype=torch.float64)
            sums += (env * valid).sum(dim=(0, 2))
            sumsq += (env * env * valid).sum(dim=(0, 2))
            counts += valid.sum(dim=(0, 2))

        for idx, band_id in enumerate(wanted):
            count = int(counts[idx].item())
            if count == 0:
                mean = 0.0
                std = 1.0
            else:
                mean = float((sums[idx] / counts[idx]).item())
                var = float((sumsq[idx] / counts[idx]).item()) - (mean * mean)
                std = math.sqrt(max(var, 0.0))
                if not math.isfinite(std) or std < 1e-8:
                    std = 1.0
            band_stats[band_id] = {"mean": mean, "std": std, "count": count}

    stats = {
        "schema_version": "4",
        "created_at": utc_now_iso(),
        "source": "points",
        "geometry": LOCAL_GEOMETRY,
        "source_split": "train",
        "scope": "all_candidates",
        "ddof": 0,
        "rasters_hash": str(plan.rasters_hash),
        "sampler_hash": str(sampler_hash),
        "band_stats": band_stats,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
    return {
        "status": "existing" if existing and not wanted else "written",
        "stats_path": str(stats_path),
        "rasters_hash": str(plan.rasters_hash),
        "sampler_hash": str(sampler_hash),
        "n_bands_normalized": int(len(band_stats)),
    }
