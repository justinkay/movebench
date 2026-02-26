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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="One-time migration: move rasters/<old_hash> -> rasters/<new_hash>.")
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
    from movebench.layers.registry import REGISTRY
    from movebench import paths

    cfg = load_layers_config(config_path)
    plan = REGISTRY.resolve(cfg)
    new_hash = str(plan.rasters_hash)

    chips_meta = _compute_chips_meta(plan, registry=REGISTRY)

    studies_root = paths.studies_root()
    if args.studies:
        studies = [str(s) for s in args.studies]
    else:
        if not studies_root.exists():
            print(f"Studies root not found: {studies_root}")
            return 1
        studies = sorted([p.name for p in studies_root.iterdir() if p.is_dir()])

    print(f"Config: {config_path}")
    print(f"Legacy rasters_hash (chip_groups-only; layers_all without alphaearth): {old_hash}")
    print(f"New rasters_hash (raster_params-only): {new_hash}")
    print(f"Studies: {len(studies)}")
    print()

    for i, study_id in enumerate(studies, 1):
        study_dir = paths.study_dir(study_id)
        old_dir = study_dir / "rasters" / str(old_hash)
        new_dir = study_dir / "rasters" / str(new_hash)
        meta_path = new_dir / "chips_meta.json"

        prefix = f"[{i}/{len(studies)}] {study_id}"

        if new_dir.exists() and old_dir.exists():
            print(f"{prefix}: SKIP rename (both old and new rasters dirs exist)")
        elif new_dir.exists() and not old_dir.exists():
            print(f"{prefix}: OK (already has new rasters dir)")
        elif not new_dir.exists() and not old_dir.exists():
            print(f"{prefix}: SKIP (no rasters dir found for old={old_hash} or new={new_hash})")
            continue
        else:
            # old exists, new does not
            if args.dry_run:
                print(f"{prefix}: RENAME {old_dir} -> {new_dir}")
            else:
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                try:
                    old_dir.rename(new_dir)
                except OSError:
                    shutil.move(str(old_dir), str(new_dir))
                print(f"{prefix}: RENAMED")

        # Ensure chips_meta.json exists under the new dir (even if there are no chips yet).
        if args.dry_run:
            print(f"{prefix}: WRITE {meta_path}")
            continue

        new_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(chips_meta, f, indent=2, sort_keys=True)
        print(f"{prefix}: WROTE chips_meta.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

