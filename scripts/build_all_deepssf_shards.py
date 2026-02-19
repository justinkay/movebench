#!/usr/bin/env python
"""Build deepSSF shard caches for many studies.

This mirrors the benchmark setup logic:
- ensure rasters exist (download if missing),
- ensure parametric point stores exist (extract if missing),
- ensure normalization stats exist (computed from point stores),
- then build tar-sharded deepSSF caches for the requested splits.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from movebench import Study
from movebench.datasets.deepssf_shards import build_deepssf_shards, deepssf_cache_hash


def _parse_hw(s: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("output_hw must be like '101,101'")
    return int(parts[0]), int(parts[1])


def _parse_splits(s: str) -> list[str]:
    out = [p.strip() for p in str(s).split(",") if p.strip()]
    if not out:
        raise ValueError("splits must be a comma-separated list like 'train,test'")
    return out


def _parse_studies(val: str) -> list[str]:
    v = str(val).strip()
    if not v or v.lower() == "all":
        return Study.list_available()
    return [p.strip() for p in v.split(",") if p.strip()]


def _ensure_ready(
    *,
    study: Study,
    config: str,
    setup_workers: int,
    num_candidates: int,
    random_seed: int,
    step_multiplier: float,
    step_q: float,
    gap_q: float,
    bounds_buffer_deg: float,
) -> str:
    """Return sampler_hash for the default parametric point store; create prerequisites if missing."""
    from movebench.features.points import select_point_store_dir

    plan = study.layer_plan(config)
    try:
        _train_dir, train_m = select_point_store_dir(
            study_dir=study.study_dir,
            split="train",
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=int(num_candidates),
            random_seed=int(random_seed),
            step_multiplier=float(step_multiplier),
            step_q=float(step_q),
            gap_q=float(gap_q),
            bounds_buffer_deg=float(bounds_buffer_deg),
        )
        sampler_hash = str(train_m.sampler_hash)
    except FileNotFoundError:
        try:
            rep = study.extract_point_features(
                config=config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(num_candidates),
                random_seed=int(random_seed),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(step_multiplier),
                step_q=float(step_q),
                gap_q=float(gap_q),
                bounds_buffer_deg=float(bounds_buffer_deg),
            )
        except FileNotFoundError as e:
            if "rasters" not in str(e):
                raise
            study.download_rasters(
                config=config,
                splits=["train", "test"],
                n_workers=int(setup_workers),
                skip_existing=True,
            )
            rep = study.extract_point_features(
                config=config,
                splits=["train", "test"],
                parametric=True,
                num_candidates=int(num_candidates),
                random_seed=int(random_seed),
                n_workers=int(setup_workers),
                skip_existing=True,
                step_multiplier=float(step_multiplier),
                step_q=float(step_q),
                gap_q=float(gap_q),
                bounds_buffer_deg=float(bounds_buffer_deg),
            )
        sampler_hash = str(rep["sampler_hash"])

    study.compute_normalization_vals(
        from_="points",
        config=config,
        sampler_hash=sampler_hash,
        split="train",
        skip_existing=True,
    )
    return sampler_hash


def _default_cache_dir(*, study: Study, split: str, config_hash: str, cache_hash: str) -> Path:
    return study.study_dir / "cache" / "deepssf_shards" / str(split) / str(config_hash) / str(cache_hash)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deepSSF shard caches for many studies.")
    p.add_argument("--studies", type=str, default="all", help="Comma-separated study ids, or 'all' (default).")
    p.add_argument("--config", type=str, default="configs/layers_all.yaml", help="Layer config YAML.")
    p.add_argument("--splits", type=_parse_splits, default=["train", "test"], help="Comma list, e.g. train,test")
    p.add_argument("--output-hw", type=_parse_hw, default=(101, 101), help="Chip (H,W), e.g. 101,101.")
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", action="store_false", dest="normalize")

    # Match default benchmark config sampler settings.
    p.add_argument("--num-candidates", type=int, default=100)
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--step-multiplier", type=float, default=2.0)
    p.add_argument("--step-q", type=float, default=0.95)
    p.add_argument("--gap-q", type=float, default=0.999)
    p.add_argument("--bounds-buffer-deg", type=float, default=0.01)

    p.add_argument("--setup-workers", type=int, default=16, help="Workers for raster download / point extraction.")

    p.add_argument("--samples-per-shard", type=int, default=256)
    p.add_argument("--spatial-dtype", type=str, default="float16", choices=("float16", "float32"))
    p.add_argument("--target-dtype", type=str, default="uint8", choices=("uint8", "float16", "float32"))
    p.add_argument("--compression", type=str, default="none", choices=("none", "gz"))
    p.add_argument("--build-workers", type=int, default=0, help="DataLoader workers for shard building.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    p.add_argument("--fail-fast", action="store_true", help="Stop on first failure (default: continue).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    studies = _parse_studies(args.studies)
    if not studies:
        print("No studies found.")
        return 1

    ok = 0
    fail = 0

    for study_id in studies:
        try:
            study = Study(str(study_id))
            plan = study.layer_plan(str(args.config))

            sampler_hash = _ensure_ready(
                study=study,
                config=str(args.config),
                setup_workers=int(args.setup_workers),
                num_candidates=int(args.num_candidates),
                random_seed=int(args.random_seed),
                step_multiplier=float(args.step_multiplier),
                step_q=float(args.step_q),
                gap_q=float(args.gap_q),
                bounds_buffer_deg=float(args.bounds_buffer_deg),
            )

            for split in list(args.splits):
                cache_hash = deepssf_cache_hash(
                    config_hash=str(plan.config_hash),
                    rasters_hash=str(plan.rasters_hash),
                    split=str(split),
                    output_hw=tuple(args.output_hw),
                    normalize=bool(args.normalize),
                    normalization_sampler_hash=str(sampler_hash) if bool(args.normalize) else None,
                    step_multiplier=float(args.step_multiplier),
                    step_q=float(args.step_q),
                    gap_q=float(args.gap_q),
                    samples_per_shard=int(args.samples_per_shard),
                    spatial_dtype=str(args.spatial_dtype),
                    target_dtype=str(args.target_dtype),
                    compression=str(args.compression),
                )
                out_dir = _default_cache_dir(
                    study=study, split=str(split), config_hash=str(plan.config_hash), cache_hash=str(cache_hash)
                )

                if (out_dir / "manifest.json").exists() and not bool(args.overwrite):
                    print(f"[{study_id}::{split}] exists; skipping: {out_dir}")
                    continue

                ds = study.deepssf_dataset(
                    split=str(split),
                    config=str(args.config),
                    output_hw=tuple(args.output_hw),
                    normalize=bool(args.normalize),
                    normalization_sampler_hash=str(sampler_hash) if bool(args.normalize) else None,
                    step_multiplier=float(args.step_multiplier),
                    step_q=float(args.step_q),
                    gap_q=float(args.gap_q),
                )

                build_deepssf_shards(
                    dataset=ds,
                    out_dir=str(out_dir),
                    config_path=str(args.config),
                    normalization_sampler_hash=str(sampler_hash) if bool(args.normalize) else None,
                    samples_per_shard=int(args.samples_per_shard),
                    spatial_dtype=str(args.spatial_dtype),
                    target_dtype=str(args.target_dtype),
                    compression=str(args.compression),
                    num_workers=int(args.build_workers),
                    overwrite=bool(args.overwrite),
                    progress=True,
                )
                print(f"[{study_id}::{split}] wrote: {out_dir}")

            ok += 1
        except Exception as e:
            fail += 1
            print(f"[{study_id}] FAILED: {type(e).__name__}: {e}")
            if bool(args.fail_fast):
                raise

    print(f"Done. ok={ok} fail={fail} total={len(studies)}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

