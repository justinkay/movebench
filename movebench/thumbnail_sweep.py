from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movebench.datasets.thumbnails import detect_outlier_cols, load_thumbnail_source_df
from movebench.thumbnail_atlas import (
    DEFAULT_EMBEDDING_CONFIG,
    DEFAULT_PROJECTION_CONFIG,
    build_atlas_dataframe,
    build_thumbnail_metadata,
    save_thumbnail_pngs as save_thumbnail_png_files,
    serialize_row_index_neighbors,
    serialize_neighbors,
    thumbnail_png_bytes_list,
)
from movebench.thumbnail_atlas_plots import save_projection_diagnostics, system_speed_category
from movebench.thumbnail_config import (
    dumps_thumbnail_generation_config,
    load_thumbnail_generation_config,
    normalize_embedding_config,
    normalize_projection_config,
    thumbnail_build_kwargs,
)
from movebench.thumbnail_embedding import (
    compute_neighbors,
    compute_thumbnail_embeddings,
    preprocessed_batch_preview_png_bytes,
    project_embeddings,
)
from movebench.thumbnail_pdf import build_thumbnail_pdf_bytes, safe_filename_part
from movebench.trajectories import INDIVIDUAL_COL


DEFAULT_CHUNK_SIZES = (10, 25, 50)
DEFAULT_COLOR_MODES = ("bw", "iforest_anomaly_score", "height-above-msl", "gps:hdop", "speed_m_s", "step_length_m")
DEFAULT_SCALE_MODES = ("individual_q0.25", "individual_q0.50", "individual_q0.75", "chunk")


@dataclass(frozen=True)
class SweepRunSpec:
    chunk_size: int
    color_slug: str
    color_by: str | None
    scale_slug: str
    scale_mode: str
    scale_quantile: float
    run_slug: str
    thumbnail_config: dict[str, Any]


def _slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(text)).strip("_").lower()


def _coerce_outlier_mask(series: pd.Series) -> np.ndarray:
    values = series.to_numpy(dtype=object, copy=False)
    out = np.zeros(len(values), dtype=bool)
    truthy = {"true", "1", "1.0", "yes", "y", "t"}
    for idx, value in enumerate(values):
        if pd.isna(value):
            continue
        if isinstance(value, str):
            out[idx] = value.strip().lower() in truthy
        else:
            out[idx] = bool(value)
    return out


def _color_mode_value(mode: str) -> tuple[str, str | None]:
    if mode == "bw":
        return "bw", None
    return _slugify(mode), mode


def _scale_mode_value(mode: str) -> tuple[str, str, float]:
    if mode == "chunk":
        return "chunk", "chunk", 0.5
    if not mode.startswith("individual_q"):
        raise ValueError(f"Unsupported scale mode: {mode!r}")
    quantile_text = mode[len("individual_q") :]
    quantile = float(quantile_text)
    slug = f"individual_q{quantile_text.replace('.', '_')}"
    return slug, "individual", quantile


def build_sweep_run_specs(
    base_config_or_path: dict[str, Any] | str | Path,
    *,
    chunk_sizes: tuple[int, ...] | list[int] = DEFAULT_CHUNK_SIZES,
    color_modes: tuple[str, ...] | list[str] = DEFAULT_COLOR_MODES,
    scale_modes: tuple[str, ...] | list[str] = DEFAULT_SCALE_MODES,
) -> list[SweepRunSpec]:
    base_config = load_thumbnail_generation_config(base_config_or_path)
    specs: list[SweepRunSpec] = []
    for chunk_size in chunk_sizes:
        for color_mode in color_modes:
            color_slug, color_by = _color_mode_value(str(color_mode))
            for scale_mode in scale_modes:
                scale_slug, scale_name, scale_quantile = _scale_mode_value(str(scale_mode))
                config = json.loads(json.dumps(base_config))
                config["thumbnail"]["chunk_size"] = int(chunk_size)
                config["thumbnail"]["color_by"] = color_by
                config["thumbnail"]["scale_mode"] = scale_name
                if scale_name == "individual":
                    config["thumbnail"]["scale_quantile"] = float(scale_quantile)
                run_slug = f"chunk{int(chunk_size)}__{color_slug}__{scale_slug}"
                specs.append(
                    SweepRunSpec(
                        chunk_size=int(chunk_size),
                        color_slug=color_slug,
                        color_by=color_by,
                        scale_slug=scale_slug,
                        scale_mode=scale_name,
                        scale_quantile=float(config["thumbnail"]["scale_quantile"]),
                        run_slug=run_slug,
                        thumbnail_config=config,
                    )
                )
    return specs


def select_top_outlier_individuals(
    csv_path: str | Path,
    *,
    split: str = "all",
    top_n: int = 3,
) -> dict[str, Any]:
    split_arg = None if str(split) == "all" else str(split)
    df = load_thumbnail_source_df(csv_path, split=split_arg, include_movement_features=False)
    outlier_cols = detect_outlier_cols(df)
    rows: list[dict[str, Any]] = []
    for individual_id, df_group in df.groupby(INDIVIDUAL_COL, sort=False):
        any_flag = np.zeros(len(df_group), dtype=bool)
        for col in outlier_cols:
            any_flag |= _coerce_outlier_mask(df_group[col])
        total_fix_count = int(len(df_group))
        outlier_fix_count = int(np.count_nonzero(any_flag))
        outlier_frequency = (
            float(outlier_fix_count) / float(total_fix_count)
            if total_fix_count > 0
            else 0.0
        )
        rows.append(
            {
                "individual_id": str(individual_id),
                "total_fix_count": total_fix_count,
                "outlier_fix_count": outlier_fix_count,
                "outlier_frequency": outlier_frequency,
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -float(row["outlier_frequency"]),
            -int(row["outlier_fix_count"]),
            str(row["individual_id"]),
        ),
    )[: max(0, int(top_n))]
    return {
        "csv_path": str(Path(csv_path).resolve()),
        "split": str(split),
        "outlier_columns": outlier_cols,
        "selected_individuals": selected,
    }


def report_individuals_in_display_order(selected_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        list(selected_payload.get("selected_individuals", [])),
        key=lambda row: (
            float(row["outlier_frequency"]),
            int(row["outlier_fix_count"]),
            str(row["individual_id"]),
        ),
    )


def report_individuals_in_run_display_order(
    selected_payload: dict[str, Any],
    meta: pd.DataFrame,
) -> list[dict[str, Any]]:
    selected_rows = list(selected_payload.get("selected_individuals", []))
    if not selected_rows:
        return []

    selected_ids = [str(row["individual_id"]) for row in selected_rows]
    totals_by_individual: dict[str, dict[str, int]] = {
        individual_id: {"total_outlier_fixes": 0, "total_fixes": 0}
        for individual_id in selected_ids
    }
    if len(meta) > 0 and "individual_id" in meta.columns:
        subset = meta.loc[meta["individual_id"].astype(str).isin(selected_ids)].copy()
        if len(subset) > 0:
            grouped = (
                subset.assign(individual_id=subset["individual_id"].astype(str))
                .groupby("individual_id", sort=False)[["n_outlier_fixes", "n_fixes"]]
                .sum()
            )
            for individual_id, row in grouped.iterrows():
                totals_by_individual[str(individual_id)] = {
                    "total_outlier_fixes": int(row["n_outlier_fixes"]),
                    "total_fixes": int(row["n_fixes"]),
                }

    rows_with_run_stats: list[dict[str, Any]] = []
    for row in selected_rows:
        individual_id = str(row["individual_id"])
        totals = totals_by_individual[individual_id]
        total_fixes = int(totals["total_fixes"])
        total_outlier_fixes = int(totals["total_outlier_fixes"])
        run_outlier_frac = (
            float(total_outlier_fixes) / float(total_fixes)
            if total_fixes > 0
            else 0.0
        )
        rows_with_run_stats.append(
            {
                **row,
                "run_outlier_frac": run_outlier_frac,
                "total_outlier_fixes": total_outlier_fixes,
                "total_fixes": total_fixes,
            }
        )

    return sorted(
        rows_with_run_stats,
        key=lambda row: (
            float(row["run_outlier_frac"]),
            str(row["individual_id"]),
        ),
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _load_json_file(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_run_embedding_config(
    base_embedding_config: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_EMBEDDING_CONFIG)
    if base_embedding_config:
        cfg.update(base_embedding_config)
    if overrides:
        cfg.update(overrides)
    return normalize_embedding_config(cfg)


def _build_run_projection_config(
    base_projection_config: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_PROJECTION_CONFIG)
    if base_projection_config:
        cfg.update(base_projection_config)
    if overrides:
        cfg.update(overrides)
    return normalize_projection_config(cfg)


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _emit_progress(message: str, *, show_progress: bool) -> None:
    if show_progress:
        print(message, flush=True)


def _run_progress_prefix(*, index: int, total: int, run_slug: str) -> str:
    return f"[{index}/{total}] {run_slug}"


def _save_individual_reports(
    *,
    run_dir: Path,
    source_name: str,
    split: str,
    images: np.ndarray,
    meta: pd.DataFrame,
    params: dict[str, Any],
    display_rows: list[dict[str, Any]],
) -> dict[str, str]:
    reports_dir = run_dir / "individual_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for idx, row in enumerate(display_rows, start=1):
        individual_id = str(row["individual_id"])
        indices = meta.index[meta["individual_id"].astype(str) == individual_id].tolist()
        pdf_bytes, _pdf_name = build_thumbnail_pdf_bytes(
            source_name=source_name,
            split=split,
            export_label=f"individual_{individual_id}",
            images=images,
            meta=meta,
            indices=indices,
            params=params,
            order_mode="outlier_frac",
        )
        file_name = f"{idx:02d}_{safe_filename_part(individual_id)}.pdf"
        path = reports_dir / file_name
        path.write_bytes(pdf_bytes)
        out[individual_id] = str(path)
    return out


def _summary_label_counts(atlas_df: pd.DataFrame) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for col in ("has_any_outlier", "has_system_outlier", "has_speed_outlier"):
        if col in atlas_df.columns:
            counts[col] = int(atlas_df[col].fillna(False).astype(bool).sum())
    if {"has_system_outlier", "has_speed_outlier"}.issubset(atlas_df.columns):
        categories = system_speed_category(atlas_df)
        counts["system_speed_category"] = {
            str(label): int((categories == label).sum())
            for label in categories.categories
        }
    return counts


def run_thumbnail_sweep(
    base_config_path: str | Path,
    out_root: str | Path,
    *,
    chunk_sizes: tuple[int, ...] | list[int] = DEFAULT_CHUNK_SIZES,
    color_modes: tuple[str, ...] | list[str] = DEFAULT_COLOR_MODES,
    scale_modes: tuple[str, ...] | list[str] = DEFAULT_SCALE_MODES,
    embedding_overrides: dict[str, Any] | None = None,
    projection_overrides: dict[str, Any] | None = None,
    save_thumbnail_pngs: bool = False,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_thumbnail_generation_config(base_config_path)
    selected_payload = select_top_outlier_individuals(
        base_config["data"]["csv_path"],
        split=base_config["data"]["split"],
        top_n=3,
    )
    selected_path = out_dir / "selected_individuals.json"
    _write_json(selected_path, selected_payload)

    embedding_config = _build_run_embedding_config(
        base_config.get("embedding"),
        embedding_overrides,
    )
    projection_config = _build_run_projection_config(
        base_config.get("projection"),
        projection_overrides,
    )
    specs = build_sweep_run_specs(
        base_config,
        chunk_sizes=chunk_sizes,
        color_modes=color_modes,
        scale_modes=scale_modes,
    )
    _emit_progress(
        (
            f"Starting thumbnail sweep with {len(specs)} runs. "
            f"Output root: {out_dir.resolve()}"
        ),
        show_progress=show_progress,
    )
    _emit_progress(
        (
            "Selected report individuals: "
            + ", ".join(
                row["individual_id"]
                for row in report_individuals_in_display_order(selected_payload)
            )
        ),
        show_progress=show_progress,
    )

    manifest_runs: list[dict[str, Any]] = []
    completed_runs = 0
    skipped_runs = 0
    for run_index, spec in enumerate(specs, start=1):
        run_prefix = _run_progress_prefix(
            index=run_index,
            total=len(specs),
            run_slug=spec.run_slug,
        )
        run_dir = out_dir / spec.run_slug
        summary_path = run_dir / "experiment_summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if bool(summary.get("save_thumbnail_pngs", False)) == bool(save_thumbnail_pngs):
                skipped_runs += 1
                _emit_progress(
                    f"{run_prefix}: skipping existing outputs",
                    show_progress=show_progress,
                )
                manifest_runs.append(
                    {
                        "run_slug": spec.run_slug,
                        "chunk_size": spec.chunk_size,
                        "color_mode": spec.color_by,
                        "scale_mode": spec.scale_slug,
                        "status": "skipped_existing",
                        "summary_path": _relative(summary_path, out_dir),
                        "n_chunks": summary.get("n_chunks"),
                    }
                )
                continue

        run_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(
            f"{run_prefix}: writing thumbnail config",
            show_progress=show_progress,
        )
        (run_dir / "thumbnail_config.json").write_text(
            dumps_thumbnail_generation_config(spec.thumbnail_config),
            encoding="utf-8",
        )

        _emit_progress(
            f"{run_prefix}: building thumbnails and metadata",
            show_progress=show_progress,
        )
        config, build, meta = build_thumbnail_metadata(
            spec.thumbnail_config,
            build=None,
        )

        preview_path = run_dir / "pre_dino_batch_preview.png"
        _emit_progress(
            f"{run_prefix}: saving pre-DINO preview",
            show_progress=show_progress,
        )
        preview_max_items = 24
        preview_path.write_bytes(
            preprocessed_batch_preview_png_bytes(
                build.images,
                embedding_config=embedding_config,
                overlay_texts=(
                    [
                        f"frac={float(value):.2f}"
                        for value in meta["outlier_frac"].fillna(0.0).tolist()[:preview_max_items]
                    ]
                    if "outlier_frac" in meta.columns
                    else None
                ),
                max_items=preview_max_items,
                cols=4,
            )
        )

        atlas_dir = run_dir / "atlas"
        atlas_dir.mkdir(parents=True, exist_ok=True)
        if save_thumbnail_pngs:
            _emit_progress(
                f"{run_prefix}: writing thumbnail PNGs",
                show_progress=show_progress,
            )
            thumbnail_paths = save_thumbnail_png_files(
                build.images,
                meta,
                out_dir=atlas_dir / "thumbnails",
            )
        else:
            thumbnail_paths = [None] * len(meta)
        _emit_progress(
            f"{run_prefix}: computing DINO embeddings",
            show_progress=show_progress,
        )
        embeddings, embedding_info = compute_thumbnail_embeddings(
            build.images,
            embedding_config=embedding_config,
        )
        _emit_progress(
            f"{run_prefix}: projecting embeddings and computing neighbors",
            show_progress=show_progress,
        )
        projection, projection_info, warnings = project_embeddings(
            embeddings,
            projection_config=projection_config,
            cache_root=run_dir / ".cache",
        )
        neighbor_idx, neighbor_dist = compute_neighbors(embeddings)
        neighbors = serialize_neighbors(
            neighbor_idx,
            meta["thumbnail_id"].astype(str).tolist(),
            neighbor_distances=neighbor_dist,
        )
        atlas_neighbors = serialize_row_index_neighbors(
            neighbor_idx,
            neighbor_distances=neighbor_dist,
        )
        atlas_df = build_atlas_dataframe(
            meta,
            projection=projection,
            neighbors=neighbors,
            thumbnail_paths=thumbnail_paths,
            atlas_neighbors=atlas_neighbors,
            thumbnail_images=thumbnail_png_bytes_list(build.images) if save_thumbnail_pngs else None,
        )
        _emit_progress(
            f"{run_prefix}: writing atlas data",
            show_progress=show_progress,
        )
        atlas_df.to_parquet(atlas_dir / "thumbnail_metadata.parquet", index=False)
        np.save(atlas_dir / "dino_embeddings.npy", embeddings.astype(np.float32, copy=False))

        _emit_progress(
            f"{run_prefix}: saving projection figures",
            show_progress=show_progress,
        )
        figure_paths = save_projection_diagnostics(atlas_df, out_dir=run_dir / "figures")
        display_rows = report_individuals_in_run_display_order(selected_payload, meta)
        _emit_progress(
            f"{run_prefix}: exporting individual reports",
            show_progress=show_progress,
        )
        report_paths = _save_individual_reports(
            run_dir=run_dir,
            source_name=Path(config["data"]["csv_path"]).stem,
            split=str(config["data"]["split"]),
            images=build.images,
            meta=meta,
            params=config["thumbnail"],
            display_rows=display_rows,
        )

        summary = {
            "run_slug": spec.run_slug,
            "chunk_size": spec.chunk_size,
            "color_mode": "bw" if spec.color_by is None else spec.color_by,
            "scale_mode": spec.scale_slug,
            "save_thumbnail_pngs": bool(save_thumbnail_pngs),
            "thumbnail_storage_mode": "file" if save_thumbnail_pngs else "none",
            "n_chunks": int(len(atlas_df)),
            "n_individuals": int(atlas_df["individual_id"].astype(str).nunique()) if "individual_id" in atlas_df.columns else 0,
            "label_counts": _summary_label_counts(atlas_df),
            "warnings": warnings,
            "embedding": embedding_info,
            "projection": projection_info,
            "selected_individuals_display_order": display_rows,
            "artifact_paths": {
                "thumbnail_config": str(run_dir / "thumbnail_config.json"),
                "selected_individuals": str(selected_path),
                "pre_dino_batch_preview": str(preview_path),
                "atlas_metadata": str(atlas_dir / "thumbnail_metadata.parquet"),
                "atlas_embeddings": str(atlas_dir / "dino_embeddings.npy"),
                "thumbnail_dir": str(atlas_dir / "thumbnails") if save_thumbnail_pngs else None,
                "figures": figure_paths,
                "individual_reports": report_paths,
            },
        }
        _write_json(summary_path, summary)
        completed_runs += 1
        _emit_progress(
            f"{run_prefix}: done ({summary['n_chunks']} chunks)",
            show_progress=show_progress,
        )
        manifest_runs.append(
            {
                "run_slug": spec.run_slug,
                "chunk_size": spec.chunk_size,
                "color_mode": spec.color_by,
                "scale_mode": spec.scale_slug,
                "status": "completed",
                "summary_path": _relative(summary_path, out_dir),
                "n_chunks": summary["n_chunks"],
            }
        )

    manifest = {
        "base_config_path": str(Path(base_config_path).resolve()),
        "out_root": str(out_dir.resolve()),
        "selected_individuals_path": _relative(selected_path, out_dir),
        "embedding_config": embedding_config,
        "projection_config": projection_config,
        "save_thumbnail_pngs": bool(save_thumbnail_pngs),
        "thumbnail_storage_mode": "file" if save_thumbnail_pngs else "none",
        "n_runs": len(specs),
        "runs": manifest_runs,
    }
    _write_json(out_dir / "sweep_manifest.json", manifest)
    _emit_progress(
        (
            f"Finished thumbnail sweep. Completed: {completed_runs}, "
            f"skipped: {skipped_runs}, total: {len(specs)}"
        ),
        show_progress=show_progress,
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a thumbnail chunk/color/scale sweep.")
    parser.add_argument("base_config", help="Path to a thumbnail generation config JSON file.")
    parser.add_argument("out_root", help="Directory where run outputs should be written.")
    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CHUNK_SIZES),
        help="Chunk sizes to sweep.",
    )
    parser.add_argument(
        "--color-modes",
        nargs="+",
        default=list(DEFAULT_COLOR_MODES),
        help="Color modes to sweep. Use 'bw' for grayscale or a color_by column name.",
    )
    parser.add_argument(
        "--scale-modes",
        nargs="+",
        default=list(DEFAULT_SCALE_MODES),
        help="Scale modes to sweep, for example individual_q0.25 or chunk.",
    )
    parser.add_argument(
        "--embedding-config",
        default=None,
        help="Optional JSON file with DINO embedding config overrides.",
    )
    parser.add_argument(
        "--projection-config",
        default=None,
        help="Optional JSON file with projection config overrides.",
    )
    parser.add_argument(
        "--save-thumbnail-pngs",
        action="store_true",
        help="Write per-thumbnail PNGs into atlas/thumbnails for each run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild run outputs even if experiment_summary.json already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run progress logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_thumbnail_sweep(
        args.base_config,
        args.out_root,
        chunk_sizes=args.chunk_sizes,
        color_modes=args.color_modes,
        scale_modes=args.scale_modes,
        embedding_overrides=_load_json_file(args.embedding_config),
        projection_overrides=_load_json_file(args.projection_config),
        save_thumbnail_pngs=bool(args.save_thumbnail_pngs),
        force=bool(args.force),
        show_progress=not bool(args.quiet),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
