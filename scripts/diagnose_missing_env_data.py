#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import torch
from tqdm.auto import tqdm

from movebench.datasets.chips import ChipDataset
from movebench.layers.config import load_layers_config
from movebench.layers.registry import REGISTRY
from movebench.study import Study, resolve_study_selection
from one_time_scripts.triage_missing_layers._common import (
    compute_candidate_missingness_counts,
)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Diagnose missing environmental data in downloaded studies."
    )
    p.add_argument("--config", required=True, help="Path to a layers YAML config.")
    p.add_argument(
        "--studies",
        nargs="*",
        default=None,
        help="Optional list of study ids or named sets like 'mini'. Default: all available studies.",
    )
    p.add_argument(
        "--from-point-stores",
        action="store_true",
        help="Diagnose missingness from pre-extracted point feature stores instead of ChipDataset (much faster).",
    )
    args = p.parse_args(argv[1:])

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
        raise ValueError(
            "No non-derived layers remain after filtering config.layers; nothing to diagnose."
        )

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
        raise ValueError(
            "No variables found to diagnose (after filtering derived layers)."
        )

    var_idx_t = torch.tensor(var_channel_idxs, dtype=torch.long)

    # Minimal "not applicable" policy surface: annotate known land-only layers so ocean-heavy studies
    # are not mistaken for pipeline failures.
    NA_HINT_PREFIXES: dict[str, str] = {
        "alphaearth_embeddings": "land_only (expected invalid over ocean)",
    }

    def _na_hint(var_name: str) -> str | None:
        for prefix, hint in NA_HINT_PREFIXES.items():
            if var_name == prefix or var_name.startswith(prefix + "."):
                return hint
        return None

    requested = args.studies if args.studies is not None else Study.list_available()
    study_ids = resolve_study_selection(requested) if requested else []
    if not study_ids:
        print("No studies found under the configured studies root.")
        return 0

    global_max_strict_pct = [-1.0 for _ in var_names]
    global_worst_strict: list[str | None] = [None for _ in var_names]
    global_max_true_pct = [-1.0 for _ in var_names]
    global_worst_true: list[str | None] = [None for _ in var_names]
    global_max_per_cand_pct = [-1.0 for _ in var_names]
    global_worst_per_cand: list[str | None] = [None for _ in var_names]

    mode = "point_stores" if args.from_point_stores else "chips"
    print(f"mode: {mode}")

    for study_id in study_ids:
        try:
            study = Study(str(study_id))
        except Exception as e:
            print(f"\n=== {study_id} ===")
            print(f"SKIP: {e}")
            continue

        strict_missing = torch.zeros((len(var_names),), dtype=torch.int64)
        true_missing = torch.zeros((len(var_names),), dtype=torch.int64)
        per_cand_invalid = torch.zeros((len(var_names),), dtype=torch.int64)
        total = 0
        total_candidates = 0
        errors = 0
        used_splits: list[str] = []

        print(f"\n=== {study.study_id} ===")
        if args.from_point_stores:
            print(f"config_hash: {plan.config_hash}")
            for split in ("train", "test"):
                base_dir = (
                    study.study_dir
                    / "features"
                    / "points"
                    / split
                    / str(plan.config_hash)
                )
                if not base_dir.exists():
                    continue
                candidates = [
                    d
                    for d in base_dir.iterdir()
                    if d.is_dir()
                    and (d / "features.pt").exists()
                    and (d / "manifest.json").exists()
                ]
                if not candidates:
                    continue
                store_dir = max(
                    candidates, key=lambda d: (d / "manifest.json").stat().st_mtime
                )
                used_splits.append(split)
                try:
                    d = torch.load(store_dir / "features.pt", map_location="cpu")
                    valid = d.get("valid")
                    if valid is None:
                        raise ValueError("features.pt missing key: 'valid'")
                    if valid.ndim != 3:
                        raise ValueError(
                            f"Expected valid tensor rank-3 [N,C,P], got shape={tuple(valid.shape)}"
                        )
                    out = compute_candidate_missingness_counts(
                        valid, var_channel_idxs=var_idx_t
                    )
                    strict_missing += out["strict_missing_steps"].to(dtype=torch.int64)
                    true_missing += out["true_missing_steps"].to(dtype=torch.int64)
                    per_cand_invalid += out["per_candidate_invalid"].to(
                        dtype=torch.int64
                    )
                    total += int(out["n_steps"])
                    total_candidates += int(out["n_total_candidates"])
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
                for i in tqdm(
                    range(len(ds)), desc=f"{study.study_id}:{split}", unit="pt"
                ):
                    try:
                        _env, valid, _transform, _crs = ds[i]
                        channel_ok = valid.view(int(valid.shape[0]), -1).all(dim=1)
                        selected_ok = channel_ok.index_select(0, var_idx_t)
                        strict_missing += (~selected_ok).to(torch.int64)
                    except Exception:
                        errors += 1
                        strict_missing += 1
                    finally:
                        total += 1

        if total == 0:
            if args.from_point_stores:
                print(
                    "No point feature stores found for this config_hash under features/points/{train,test}/."
                )
            else:
                print("No points found (train/test missing or empty).")
            continue

        print(f"splits: {','.join(used_splits) if used_splits else '(none)'}")
        unit = "steps" if args.from_point_stores else "points"
        if args.from_point_stores:
            if total_candidates <= 0:
                raise ValueError(
                    "Internal error: total_candidates must be > 0 in point-store mode"
                )
            strict_pct = (strict_missing.to(torch.float64) / float(total)) * 100.0
            true_pct = (true_missing.to(torch.float64) / float(total)) * 100.0
            per_cand_pct = (
                per_cand_invalid.to(torch.float64) / float(total_candidates)
            ) * 100.0
            order = torch.argsort(strict_pct, descending=True).tolist()

            print(f"{unit}: {total}  candidates: {total_candidates}  errors: {errors}")
            print(
                "strict%\ttrue%\tper_cand%\tstrict/steps\ttrue/steps\tinvalid/(steps*P)\tvar\thint"
            )

            printed = False
            for j in order:
                s = int(strict_missing[j].item())
                t = int(true_missing[j].item())
                pc = int(per_cand_invalid[j].item())
                if s == 0 and t == 0 and pc == 0:
                    continue
                hint = _na_hint(var_names[j]) or ""
                printed = True
                print(
                    f"{strict_pct[j].item():6.2f}%\t{true_pct[j].item():6.2f}%\t{per_cand_pct[j].item():8.4f}%\t"
                    f"{s}/{total}\t{t}/{total}\t{pc}/{total_candidates}\t{var_names[j]}\t{hint}"
                )
            if not printed:
                print("No missingness detected.")

            for i, pval in enumerate(strict_pct.tolist()):
                if float(pval) > global_max_strict_pct[i]:
                    global_max_strict_pct[i] = float(pval)
                    global_worst_strict[i] = study.study_id
            for i, pval in enumerate(true_pct.tolist()):
                if float(pval) > global_max_true_pct[i]:
                    global_max_true_pct[i] = float(pval)
                    global_worst_true[i] = study.study_id
            for i, pval in enumerate(per_cand_pct.tolist()):
                if float(pval) > global_max_per_cand_pct[i]:
                    global_max_per_cand_pct[i] = float(pval)
                    global_worst_per_cand[i] = study.study_id
        else:
            pct = (strict_missing.to(torch.float64) / float(total)) * 100.0
            order = torch.argsort(pct, descending=True).tolist()

            print(f"{unit}: {total}  errors: {errors}")
            printed = False
            for j in order:
                miss = int(strict_missing[j].item())
                if miss == 0:
                    continue
                printed = True
                print(f"{pct[j].item():6.2f}%\t{miss}/{total}\t{var_names[j]}")
            if not printed:
                print("No missingness detected.")

            for i, pval in enumerate(pct.tolist()):
                if float(pval) > global_max_strict_pct[i]:
                    global_max_strict_pct[i] = float(pval)
                    global_worst_strict[i] = study.study_id

    print("\n=== Worst missingness across studies (strict) ===")
    rows_strict = [
        (global_max_strict_pct[i], global_worst_strict[i], var_names[i])
        for i in range(len(var_names))
        if global_max_strict_pct[i] > 0.0 and global_worst_strict[i] is not None
    ]
    rows_strict.sort(key=lambda x: x[0], reverse=True)
    if not rows_strict:
        print("No missingness detected in any study.")
        return 0

    for pval, worst, name in rows_strict:
        print(f"{pval:6.2f}%\t{worst}\t{name}")

    if args.from_point_stores:
        print("\n=== Worst missingness across studies (true endpoint) ===")
        rows_true = [
            (global_max_true_pct[i], global_worst_true[i], var_names[i])
            for i in range(len(var_names))
            if global_max_true_pct[i] > 0.0 and global_worst_true[i] is not None
        ]
        rows_true.sort(key=lambda x: x[0], reverse=True)
        if not rows_true:
            print("No true-endpoint missingness detected in any study.")
        else:
            for pval, worst, name in rows_true:
                print(f"{pval:6.2f}%\t{worst}\t{name}")

        print("\n=== Worst missingness across studies (per-candidate invalid) ===")
        rows_pc = [
            (global_max_per_cand_pct[i], global_worst_per_cand[i], var_names[i])
            for i in range(len(var_names))
            if global_max_per_cand_pct[i] > 0.0 and global_worst_per_cand[i] is not None
        ]
        rows_pc.sort(key=lambda x: x[0], reverse=True)
        if not rows_pc:
            print("No per-candidate invalidity detected in any study.")
        else:
            for pval, worst, name in rows_pc:
                print(f"{pval:8.4f}%\t{worst}\t{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
