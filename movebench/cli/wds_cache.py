from __future__ import annotations

import argparse


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "wds-cache",
        help="Build deepSSF WebDataset caches for one study.",
        description="Build deepSSF WebDataset caches for one study.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/benchmarks/all.yaml",
        help="Benchmark YAML config path (must include datasets.deepssf).",
    )
    parser.add_argument("--study", type=str, required=True, help="Study id.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "test"),
        default=("train", "test"),
        help="Splits to cache (default: train test).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caches.")
    parser.add_argument("--resume", action="store_true", help="Resume partially-built caches.")
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from movebench import Study
    from movebench.benchmark import ensure_study_ready, load_config
    from movebench.datasets.deepssf_wds_cache import ensure_deepssf_wds_cache_split

    cfg = load_config(str(args.config))
    dcfg = cfg.get("datasets", {}).get("deepssf", {})
    if not isinstance(dcfg, dict):
        raise ValueError("Benchmark config missing required key: datasets.deepssf")
    cache_cfg = dcfg.get("cache", {}) or {}
    if not isinstance(cache_cfg, dict):
        raise ValueError("datasets.deepssf.cache must be a mapping/dict when provided")

    study = Study(str(args.study))
    sampler_hash = ensure_study_ready(study=study, cfg=cfg, allow_setup=False)

    def ds_kwargs_for(split: str) -> dict:
        out: dict = {
            "split": str(split),
            "config": str(cfg["layers_config"]),
            "normalization_sampler_hash": str(sampler_hash),
        }
        for k in ("output_hw", "normalize", "step_multiplier", "step_q", "gap_q"):
            if k not in dcfg:
                continue
            v = dcfg[k]
            if k == "output_hw":
                if not (isinstance(v, (list, tuple)) and len(v) == 2):
                    raise ValueError("datasets.deepssf.output_hw must be a 2-item sequence like [101,101].")
                v = (int(v[0]), int(v[1]))
            out[k] = v
        return out

    samples_per_shard = int(cache_cfg.get("samples_per_shard", 256))
    spatial_dtype = str(cache_cfg.get("spatial_dtype", "float16"))
    target_dtype = str(cache_cfg.get("target_dtype", "uint8"))
    compression = str(cache_cfg.get("compression", "none"))
    seed = int(cache_cfg.get("seed", int(cfg.get("runtime", {}).get("seed", 0))))
    build_workers = int(cache_cfg.get("build_workers", 0))
    overwrite = bool(args.overwrite) or bool(cache_cfg.get("overwrite", False))
    resume = bool(args.resume) or bool(cache_cfg.get("resume", False))

    for split in args.splits:
        cache_dir, manifest = ensure_deepssf_wds_cache_split(
            study=study,
            split=str(split),
            layers_config=str(cfg["layers_config"]),
            ds_kwargs=ds_kwargs_for(str(split)),
            samples_per_shard=int(samples_per_shard),
            spatial_dtype=spatial_dtype,  # type: ignore[arg-type]
            target_dtype=target_dtype,  # type: ignore[arg-type]
            compression=compression,  # type: ignore[arg-type]
            seed=int(seed),
            build_workers=int(build_workers),
            build_if_missing=True,
            overwrite=bool(overwrite),
            resume=bool(resume),
        )
        print(f"[wds-cache] split={split} cache_dir={cache_dir} samples={manifest.get('num_samples')}")
    return 0
