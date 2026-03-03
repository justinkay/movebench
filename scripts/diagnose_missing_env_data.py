#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _insert_repo_root_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Diagnose missing environmental data in downloaded studies.")
    p.add_argument("--config", required=True, help="Path to a layers YAML config.")
    p.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Optional list of study ids to scan. Default: all available studies.",
    )
    p.add_argument(
        "--from-point-stores",
        action="store_true",
        help="Diagnose missingness from pre-extracted point feature stores instead of ChipDataset (much faster).",
    )
    args = p.parse_args(argv[1:])

    _insert_repo_root_on_path()

    try:
        import torch
        from tqdm.auto import tqdm

        from movebench.layers.config import load_layers_config
        from movebench.layers.registry import REGISTRY
        from movebench.study import Study
    except ModuleNotFoundError as e:
        print(f"Import failed: {e}")
        print("Make sure movebench dependencies are installed (pandas, torch, rasterio, pyproj, tqdm).")
        return 1

    cfg = load_layers_config(args.config)

    kept_layers: list[str] = []
    dropped: list[str] = []
    for layer in cfg.layers:
        if REGISTRY.get(layer).source == "derived":
            dropped.append(layer)
            continue
        kept_layers.append(layer)

    if dropped:
        print("Skipping derived layers:", ", ".join(dropped))

    if not kept_layers:
        raise ValueError("No non-derived layers remain after filtering config.layers; nothing to diagnose.")

    if args.from_point_stores:
        # Keep the config as-is: feature store directories are keyed by config_hash.
        plan = REGISTRY.resolve(cfg)
    else:
        # ChipDataset should not compute derived layers. Different branches may add extra required
        # fields (e.g. raster_params), so keep everything except derived layers via dataclass replacement.
        filtered_cfg = replace(cfg, layers=kept_layers)
        plan = REGISTRY.resolve(filtered_cfg)

    # Variables to report: continuous channels per band_id, categorical layers collapsed per layer.
    var_names: list[str] = []
    var_channel_idxs: list[int] = []
    for layer in plan.layers_in_order:
        spec = REGISTRY.get(layer)
        if spec.source == "derived":
            continue
        start, end = plan.layer_slice(layer)
        if spec.categorical:
            var_names.append(layer)
            var_channel_idxs.append(int(start))
            continue
        for i in range(int(start), int(end)):
            var_names.append(plan.channels[i].band_id)
            var_channel_idxs.append(i)

    if not var_names:
        raise ValueError("No variables found to diagnose (after filtering derived layers).")

    var_idx_t = torch.tensor(var_channel_idxs, dtype=torch.long)

    study_ids = args.studies if args.studies else Study.list_available()
    if not study_ids:
        print("No studies found under the configured studies root.")
        return 0

    global_max_pct = [-1.0 for _ in var_names]
    global_worst_study: list[str | None] = [None for _ in var_names]

    mode = "point_stores" if args.from_point_stores else "chips"
    print(f"mode: {mode}")

    for study_id in study_ids:
        try:
            study = Study(str(study_id))
        except Exception as e:
            print(f"\n=== {study_id} ===")
            print(f"SKIP: {e}")
            continue

        missing = torch.zeros((len(var_names),), dtype=torch.int64)
        total = 0
        errors = 0
        used_splits: list[str] = []

        print(f"\n=== {study.study_id} ===")
        if args.from_point_stores:
            print(f"config_hash: {plan.config_hash}")
            for split in ("train", "test"):
                base_dir = study.study_dir / "features" / "points" / split / str(plan.config_hash)
                if not base_dir.exists():
                    continue
                candidates = [
                    d
                    for d in base_dir.iterdir()
                    if d.is_dir() and (d / "features.pt").exists() and (d / "manifest.json").exists()
                ]
                if not candidates:
                    continue
                store_dir = max(candidates, key=lambda d: (d / "manifest.json").stat().st_mtime)
                used_splits.append(split)
                try:
                    d = torch.load(store_dir / "features.pt", map_location="cpu")
                    valid = d.get("valid")
                    if valid is None:
                        raise ValueError("features.pt missing key: 'valid'")
                    if valid.ndim != 3:
                        raise ValueError(f"Expected valid tensor rank-3 [N,C,P], got shape={tuple(valid.shape)}")
                    # Strict: a variable is "present" for a step only if all candidates are valid.
                    channel_ok = valid.all(dim=2)  # (N,C)
                    selected_ok = channel_ok.index_select(1, var_idx_t)  # (N,V)
                    missing += (~selected_ok).sum(dim=0).to(torch.int64)
                    total += int(selected_ok.shape[0])
                except Exception as e:
                    errors += 1
                    print(f"{study.study_id}:{split} store error ({e})")
                finally:
                    try:
                        del d
                    except Exception:
                        pass
        else:
            print(f"rasters_hash: {plan.rasters_hash}")
            try:
                from movebench.datasets.chips import ChipDataset
            except ModuleNotFoundError as e:
                print(f"Import failed: {e}")
                print("ChipDataset mode requires rasterio and pyproj installed.")
                return 1

            for split in ("train", "test"):
                try:
                    ds = ChipDataset(
                        study_dir=study.study_dir,
                        trajectories_path=study.trajectories_path,
                        split=split,
                        plan=plan,
                        normalize=False,
                    )
                except Exception as e:
                    # Most common: split missing/empty. Keep output minimal.
                    if "No rows found for split=" not in str(e):
                        print(f"{study.study_id}:{split} skip ({e})")
                    continue

                used_splits.append(split)
                for i in tqdm(range(len(ds)), desc=f"{study.study_id}:{split}", unit="pt"):
                    try:
                        _env, valid, _transform, _crs = ds[i]
                        channel_ok = valid.view(int(valid.shape[0]), -1).all(dim=1)
                        selected_ok = channel_ok.index_select(0, var_idx_t)
                        missing += (~selected_ok).to(torch.int64)
                    except Exception:
                        errors += 1
                        missing += 1
                    finally:
                        total += 1
                        # Prevent unbounded growth: LayerIO caches chips by (group, point_id).
                        try:
                            ds.io._chip_cache.clear()
                        except Exception:
                            pass

        if total == 0:
            if args.from_point_stores:
                print("No point feature stores found for this config_hash under features/points/{train,test}/.")
            else:
                print("No points found (train/test missing or empty).")
            continue

        pct = (missing.to(torch.float64) / float(total)) * 100.0
        order = torch.argsort(pct, descending=True).tolist()

        print(f"splits: {','.join(used_splits) if used_splits else '(none)'}")
        unit = "steps" if args.from_point_stores else "points"
        print(f"{unit}: {total}  errors: {errors}")

        printed = False
        for j in order:
            miss = int(missing[j].item())
            if miss == 0:
                continue
            printed = True
            print(f"{pct[j].item():6.2f}%\t{miss}/{total}\t{var_names[j]}")
        if not printed:
            print("No missingness detected.")

        pct_list = pct.tolist()
        for i, pval in enumerate(pct_list):
            if pval > global_max_pct[i]:
                global_max_pct[i] = float(pval)
                global_worst_study[i] = study.study_id

    print("\n=== Worst missingness across studies ===")
    rows = [
        (global_max_pct[i], global_worst_study[i], var_names[i])
        for i in range(len(var_names))
        if global_max_pct[i] > 0.0 and global_worst_study[i] is not None
    ]
    rows.sort(key=lambda x: x[0], reverse=True)
    if not rows:
        print("No missingness detected in any study.")
        return 0

    for pval, worst, name in rows:
        print(f"{pval:6.2f}%\t{worst}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
