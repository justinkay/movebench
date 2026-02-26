from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


def _stable_hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _legacy_rasters_hash_from_yaml_dict(data: dict[str, Any]) -> str:
    chip_groups_raw = data.get("chip_groups", {})
    if chip_groups_raw is None:
        chip_groups_raw = {}
    if not isinstance(chip_groups_raw, dict):
        raise ValueError("'chip_groups' must be a mapping for legacy rasters_hash computation")

    canonical: dict[str, Any] = {}
    for name, g in sorted(chip_groups_raw.items(), key=lambda kv: str(kv[0])):
        if not isinstance(name, str):
            raise ValueError("chip_groups keys must be strings")
        if not isinstance(g, dict):
            raise ValueError(f"chip_groups.{name} must be a mapping")
        if "layers" not in g or "target_scale_m" not in g:
            raise ValueError(f"chip_groups.{name} must include keys 'layers' and 'target_scale_m'")
        layers = g["layers"]
        if not isinstance(layers, list) or not all(isinstance(x, str) for x in layers):
            raise ValueError(f"chip_groups.{name}.layers must be a list of strings")
        canonical[name] = {"layers": list(layers), "target_scale_m": float(g["target_scale_m"])}

    payload = json.dumps({"chip_groups": canonical}, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        d = yaml.safe_load(f.read())
    if not isinstance(d, dict):
        raise ValueError(f"Expected YAML mapping at top-level: {path}")
    return dict(d)


def _remove_alphaearth_layer(data: dict[str, Any]) -> dict[str, Any]:
    out = dict(data)
    layers = out.get("layers")
    if isinstance(layers, list):
        out["layers"] = [x for x in layers if str(x) != "alphaearth_embeddings"]
    return out


def _compute_chips_meta(plan, *, registry) -> dict[str, Any]:
    chip_groups_out: dict[str, Any] = {}
    for group_name in sorted(plan.chip_groups.keys()):
        g = plan.chip_groups[group_name]
        layers = list(g["layers"])
        band_ids_in_order: list[str] = []
        for layer in layers:
            spec = registry.get(layer)
            band_ids_in_order.extend([f"{layer}.{b}" for b in spec.bands])
        payload = {
            "target_scale_m": float(g["target_scale_m"]),
            "layers": layers,
            "band_ids_in_order": band_ids_in_order,
        }
        chip_groups_out[group_name] = {
            "signature": _stable_hash_dict(payload),
            **payload,
        }
    return {"schema_version": "1", "chip_groups": chip_groups_out}

def _load_benchmark_like_deepssf_cfg(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (dcfg, cache_cfg) similar to benchmark.py parsing.

    If the file is missing or invalid, callers may fall back to these defaults.
    """
    # Defaults (match movebench/benchmark.py + movebench/datasets/deepssf_wds_cache.py)
    dcfg: dict[str, Any] = {
        "output_hw": (101, 101),
        "normalize": True,
        "step_multiplier": 2.0,
        "step_q": 0.95,
        "gap_q": 0.999,
    }
    cache_cfg: dict[str, Any] = {
        "samples_per_shard": 256,
        "spatial_dtype": "float16",
        "target_dtype": "uint8",
        "compression": "none",
    }
    raw = _load_yaml_mapping(path)
    datasets = raw.get("datasets", {})
    if not isinstance(datasets, dict):
        raise ValueError(f"Expected mapping at datasets in {path}")
    deepssf = datasets.get("deepssf", {})
    if not isinstance(deepssf, dict):
        raise ValueError(f"Expected mapping at datasets.deepssf in {path}")

    # Dataset kwargs
    if "output_hw" in deepssf:
        v = deepssf["output_hw"]
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError(f"datasets.deepssf.output_hw must be a 2-item sequence like [101,101] in {path}")
        dcfg["output_hw"] = (int(v[0]), int(v[1]))
    for k in ("normalize", "step_multiplier", "step_q", "gap_q"):
        if k in deepssf:
            dcfg[k] = deepssf[k]

    # Cache kwargs (subset used in make_deepssf_wds_train_test_loaders)
    raw_cache = deepssf.get("cache", {})
    if raw_cache is None:
        raw_cache = {}
    if not isinstance(raw_cache, dict):
        raise ValueError(f"datasets.deepssf.cache must be a mapping/dict in {path}")
    for k, default in (
        ("samples_per_shard", 256),
        ("spatial_dtype", "float16"),
        ("target_dtype", "uint8"),
        ("compression", "none"),
    ):
        if k in raw_cache:
            cache_cfg[k] = raw_cache[k]
        else:
            cache_cfg[k] = default

    return dict(dcfg), dict(cache_cfg)


def _maybe_update_wds_manifest_cache_hash(cache_dir: Path, *, cache_hash: str) -> None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        d = json.loads(manifest_path.read_text())
    except Exception:
        return
    if not isinstance(d, dict):
        return
    d["cache_hash"] = str(cache_hash)
    manifest_path.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def _detect_sampler_hashes_for_study(*, study_dir: Path, config_hash: str) -> list[str]:
    """Best-effort sampler_hash candidates for deepSSF normalization.

    This uses the presence of computed normalization stats as a proxy for
    "sampler_hash values that are likely used by downstream training".
    """
    base = Path(study_dir) / "normalization" / "points" / str(config_hash)
    if not base.exists():
        return []
    out: list[str] = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if (p / "stats.json").exists():
            out.append(p.name)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-time migration: move rasters/<old_hash> -> rasters/<new_hash> "
            "(or --revert to move rasters/<new_hash> -> rasters/<old_hash>). "
            "Optionally migrate deepSSF WebDataset caches whose cache_hash includes rasters_hash."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/layers_all.yaml",
        help="Layers YAML config path (default: configs/layers_all.yaml).",
    )
    parser.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Optional list of study ids to migrate. Default: all studies under the studies root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without renaming anything.")
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert migration: move rasters/<new_hash> -> rasters/<old_hash> and delete chips_meta.json.",
    )
    parser.add_argument(
        "--skip-deepssf-wds-cache",
        action="store_true",
        help=(
            "Skip migrating deepSSF WebDataset caches under study_dir/cache/deepssf_wds. "
            "By default, this script migrates caches using configs/benchmarks/deepssf.yaml."
        ),
    )
    parser.add_argument(
        "--deepssf-splits",
        nargs="*",
        default=None,
        help="Optional list of splits to migrate deepSSF WebDataset caches for (default: train test).",
    )
    parser.add_argument(
        "--deepssf-sampler-hashes",
        nargs="*",
        default=None,
        help=(
            "Optional list of normalization sampler_hash values to consider for deepSSF cache migration. "
            "If omitted, the script auto-detects sampler_hash candidates from normalization stats on disk."
        ),
    )
    parser.add_argument("--data-root", default=None, help="Override MOVEBENCH_DATA_ROOT for this run.")
    parser.add_argument("--paths-config", default=None, help="Override MOVEBENCH_PATHS_CONFIG for this run.")

    args = parser.parse_args(argv[1:])

    if args.data_root:
        os.environ["MOVEBENCH_DATA_ROOT"] = str(args.data_root)
    if args.paths_config:
        os.environ["MOVEBENCH_PATHS_CONFIG"] = str(args.paths_config)

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    config_path = Path(args.config)
    raw = _load_yaml_mapping(config_path)
    raw_legacy = _remove_alphaearth_layer(raw)
    old_hash = _legacy_rasters_hash_from_yaml_dict(raw_legacy)

    from movebench.layers.config import load_layers_config
    from movebench.layers.config import config_hash as _config_hash
    from movebench.layers.config import parse_layers_config
    from movebench.layers.registry import REGISTRY
    from movebench import paths

    cfg = load_layers_config(config_path)
    plan = REGISTRY.resolve(cfg)
    new_hash = str(plan.rasters_hash)
    config_hash = str(plan.config_hash)
    legacy_cfg = parse_layers_config(raw_legacy)
    legacy_config_hash = str(_config_hash(legacy_cfg))

    mode = "REVERT" if args.revert else "MIGRATE"

    chips_meta: dict[str, Any] | None = None
    if not args.revert:
        chips_meta = _compute_chips_meta(plan, registry=REGISTRY)

    migrate_wds = not bool(args.skip_deepssf_wds_cache)
    deepssf_bench_cfg_path = repo_root / "configs" / "benchmarks" / "deepssf.yaml"
    dcfg: dict[str, Any]
    cache_cfg: dict[str, Any]
    if migrate_wds:
        if deepssf_bench_cfg_path.exists():
            try:
                dcfg, cache_cfg = _load_benchmark_like_deepssf_cfg(deepssf_bench_cfg_path)
            except Exception as e:
                print(f"WARNING: Failed to parse deepSSF benchmark config; using defaults ({deepssf_bench_cfg_path}): {e}")
                dcfg = {
                    "output_hw": (101, 101),
                    "normalize": True,
                    "step_multiplier": 2.0,
                    "step_q": 0.95,
                    "gap_q": 0.999,
                }
                cache_cfg = {
                    "samples_per_shard": 256,
                    "spatial_dtype": "float16",
                    "target_dtype": "uint8",
                    "compression": "none",
                }
        else:
            print(f"WARNING: deepSSF benchmark config not found; skipping deepSSF WDS cache migration: {deepssf_bench_cfg_path}")
            migrate_wds = False
            dcfg, cache_cfg = {}, {}
    else:
        dcfg, cache_cfg = {}, {}
    deepssf_splits = [str(s) for s in (args.deepssf_splits or ["train", "test"])]
    deepssf_sampler_hashes_cli = [str(s) for s in (args.deepssf_sampler_hashes or []) if str(s).strip()]

    studies_root = paths.studies_root()
    if args.studies:
        studies = [str(s) for s in args.studies]
    else:
        if not studies_root.exists():
            print(f"Studies root not found: {studies_root}")
            return 1
        studies = sorted([p.name for p in studies_root.iterdir() if p.is_dir()])

    print(f"Config: {config_path}")
    print(f"Mode: {mode}")
    print(f"Legacy rasters_hash (chip_groups-only; layers_all without alphaearth): {old_hash}")
    print(f"New rasters_hash (raster_params-only): {new_hash}")
    if migrate_wds:
        print(f"Also migrating deepSSF WebDataset cache hashes (config: {deepssf_bench_cfg_path}).")
        print(f"deepSSF WDS splits: {deepssf_splits}")
        print(f"deepSSF WDS config_hash: {legacy_config_hash} (legacy; without alphaearth_embeddings)")
        print(f"deepSSF WDS current config_hash: {config_hash}")
        print(
            "deepSSF WDS hash params: "
            f"output_hw={dcfg.get('output_hw')}, normalize={dcfg.get('normalize')}, "
            f"step_multiplier={dcfg.get('step_multiplier')}, step_q={dcfg.get('step_q')}, gap_q={dcfg.get('gap_q')}, "
            f"samples_per_shard={cache_cfg.get('samples_per_shard')}, spatial_dtype={cache_cfg.get('spatial_dtype')}, "
            f"target_dtype={cache_cfg.get('target_dtype')}, compression={cache_cfg.get('compression')}"
        )
    print(f"Studies: {len(studies)}")
    print()

    for i, study_id in enumerate(studies, 1):
        study_dir = paths.study_dir(study_id)
        old_dir = study_dir / "rasters" / str(old_hash)
        new_dir = study_dir / "rasters" / str(new_hash)
        src_dir = new_dir if args.revert else old_dir
        dst_dir = old_dir if args.revert else new_dir
        src_hash = new_hash if args.revert else old_hash
        dst_hash = old_hash if args.revert else new_hash
        meta_path = dst_dir / "chips_meta.json"

        prefix = f"[{i}/{len(studies)}] {study_id}"

        if src_dir.exists() and dst_dir.exists():
            print(f"{prefix}: SKIP (both src and dst rasters dirs exist: {src_dir.name}, {dst_dir.name})")
            if args.revert:
                continue
        elif dst_dir.exists() and not src_dir.exists():
            print(f"{prefix}: OK (already has destination rasters dir)")
        elif not dst_dir.exists() and not src_dir.exists():
            print(f"{prefix}: SKIP (no rasters dir found for src={src_hash} or dst={dst_hash})")
            continue
        else:
            # src exists, dst does not
            if args.dry_run:
                print(f"{prefix}: RENAME {src_dir} -> {dst_dir}")
            else:
                dst_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    src_dir.rename(dst_dir)
                except OSError:
                    shutil.move(str(src_dir), str(dst_dir))
                print(f"{prefix}: RENAMED")

        if args.revert:
            if args.dry_run:
                print(f"{prefix}: DELETE {meta_path}")
                continue
            if meta_path.exists():
                meta_path.unlink()
                print(f"{prefix}: DELETED chips_meta.json")
            else:
                print(f"{prefix}: OK (no chips_meta.json to delete)")
        else:
            # Ensure chips_meta.json exists under the new dir (even if there are no chips yet).
            if args.dry_run:
                print(f"{prefix}: WRITE {meta_path}")
            else:
                assert chips_meta is not None
                dst_dir.mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w") as f:
                    json.dump(chips_meta, f, indent=2, sort_keys=True)
                print(f"{prefix}: WROTE chips_meta.json")

        if migrate_wds:
            from movebench.datasets.deepssf_wds_cache import deepssf_wds_cache_hash

            normalize = bool(dcfg.get("normalize", True))
            sampler_hashes = list(deepssf_sampler_hashes_cli)
            if not sampler_hashes and normalize:
                sampler_hashes = _detect_sampler_hashes_for_study(study_dir=study_dir, config_hash=legacy_config_hash)

            if normalize and not sampler_hashes:
                print(f"{prefix}: deepSSF WDS SKIP (no sampler_hash values detected; pass --deepssf-sampler-hashes)")
            else:
                if not sampler_hashes:
                    sampler_hashes = [""]

                for split in deepssf_splits:
                    for sampler_hash in sampler_hashes:
                        norm_hash = str(sampler_hash) if (normalize and sampler_hash) else None

                        old_cache_hash = deepssf_wds_cache_hash(
                            config_hash=str(legacy_config_hash),
                            rasters_hash=str(old_hash),
                            split=str(split),
                            output_hw=tuple(dcfg["output_hw"]),
                            normalize=normalize,
                            normalization_sampler_hash=norm_hash,
                            step_multiplier=float(dcfg.get("step_multiplier", 2.0)),
                            step_q=float(dcfg.get("step_q", 0.95)),
                            gap_q=float(dcfg.get("gap_q", 0.999)),
                            samples_per_shard=int(cache_cfg.get("samples_per_shard", 256)),
                            spatial_dtype=str(cache_cfg.get("spatial_dtype", "float16")),  # type: ignore[arg-type]
                            target_dtype=str(cache_cfg.get("target_dtype", "uint8")),  # type: ignore[arg-type]
                            compression=str(cache_cfg.get("compression", "none")),  # type: ignore[arg-type]
                        )
                        new_cache_hash = deepssf_wds_cache_hash(
                            config_hash=str(legacy_config_hash),
                            rasters_hash=str(new_hash),
                            split=str(split),
                            output_hw=tuple(dcfg["output_hw"]),
                            normalize=normalize,
                            normalization_sampler_hash=norm_hash,
                            step_multiplier=float(dcfg.get("step_multiplier", 2.0)),
                            step_q=float(dcfg.get("step_q", 0.95)),
                            gap_q=float(dcfg.get("gap_q", 0.999)),
                            samples_per_shard=int(cache_cfg.get("samples_per_shard", 256)),
                            spatial_dtype=str(cache_cfg.get("spatial_dtype", "float16")),  # type: ignore[arg-type]
                            target_dtype=str(cache_cfg.get("target_dtype", "uint8")),  # type: ignore[arg-type]
                            compression=str(cache_cfg.get("compression", "none")),  # type: ignore[arg-type]
                        )

                        src_cache_hash = new_cache_hash if args.revert else old_cache_hash
                        dst_cache_hash = old_cache_hash if args.revert else new_cache_hash
                        src_cache_dir = (
                            study_dir / "cache" / "deepssf_wds" / str(split) / str(legacy_config_hash) / str(src_cache_hash)
                        )
                        dst_cache_dir = (
                            study_dir / "cache" / "deepssf_wds" / str(split) / str(legacy_config_hash) / str(dst_cache_hash)
                        )

                        tag = f"{prefix}: deepSSF WDS split={split}"
                        if norm_hash is not None:
                            tag += f" sampler_hash={norm_hash}"

                        if src_cache_dir.exists() and dst_cache_dir.exists():
                            print(f"{tag}: SKIP (both src and dst cache dirs exist)")
                            continue
                        if dst_cache_dir.exists() and not src_cache_dir.exists():
                            print(f"{tag}: OK (already has destination cache dir)")
                            continue
                        if not src_cache_dir.exists() and not dst_cache_dir.exists():
                            continue

                        if args.dry_run:
                            print(f"{tag}: RENAME {src_cache_dir} -> {dst_cache_dir}")
                            continue

                        dst_cache_dir.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            src_cache_dir.rename(dst_cache_dir)
                        except OSError:
                            shutil.move(str(src_cache_dir), str(dst_cache_dir))
                        _maybe_update_wds_manifest_cache_hash(dst_cache_dir, cache_hash=str(dst_cache_hash))
                        print(f"{tag}: RENAMED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
