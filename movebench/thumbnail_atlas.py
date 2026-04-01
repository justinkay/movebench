from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from movebench.datasets.thumbnails import build_track_thumbnails
from movebench.thumbnail_config import (
    dumps_thumbnail_generation_config,
    load_thumbnail_generation_config,
    thumbnail_build_kwargs,
    thumbnail_config_run_slug,
)
from movebench.thumbnail_embedding import (
    compute_neighbors,
    compute_thumbnail_embeddings,
    project_embeddings,
)

DEFAULT_EMBEDDING_CONFIG: dict[str, Any] = {
    "model_name": "dinov2_vits14",
    "device_preference": "auto",
    "batch_size": 32,
    "image_size": 224,
    "feature_source": "cls_token",
    "intermediate_layer_offset": 2,
}

DEFAULT_PROJECTION_CONFIG: dict[str, Any] = {
    "method": "umap",
    "random_state": 17,
    "n_neighbors": 20,
    "min_dist": 0.10,
    "perplexity": 30.0,
}


def overlap_state_from_row(row: pd.Series) -> str:
    alg = bool(row.get("has_algorithm_marked_outlier", False))
    manual = bool(row.get("has_manually_marked_outlier", False))
    if alg and manual:
        return "algorithm + manual"
    if manual:
        return "manual only"
    if alg:
        return "algorithm only"
    if bool(row.get("has_any_outlier", False)):
        return "other outlier"
    return "clean"


def thumbnail_id_for_row(row: pd.Series, *, source_name: str) -> str:
    return "|".join(
        [
            str(source_name),
            str(row.get("split", "")),
            str(row.get("individual_id", "")),
            str(int(row.get("chunk_idx", 0))),
            pd.Timestamp(row["start_time"]).isoformat(),
            pd.Timestamp(row["end_time"]).isoformat(),
            str(int(row.get("n_fixes", 0))),
        ]
    )


def description_for_row(row: pd.Series) -> str:
    active_labels = sorted(
        label
        for label in row.index
        if label.startswith("has_")
        and label != "has_any_outlier"
        and bool(row.get(label, False))
    )
    label_text = "clean" if not active_labels else ", ".join(label[4:].replace("_", " ") for label in active_labels)
    return (
        f"source={row.get('source_name', '')} split={row.get('split', '')} "
        f"individual={row.get('individual_id', '')} chunk_idx={int(row.get('chunk_idx', 0))} "
        f"time={pd.Timestamp(row['start_time']).strftime('%Y-%m-%d %H:%M')} to "
        f"{pd.Timestamp(row['end_time']).strftime('%Y-%m-%d %H:%M')} "
        f"n_fixes={int(row.get('n_fixes', 0))} "
        f"overlap_state={row.get('overlap_state', '')} "
        f"outlier_frac={float(row.get('outlier_frac', 0.0)):.3f} "
        f"labels={label_text}"
    )


def build_thumbnail_metadata(
    config_or_path: dict[str, Any] | str | Path,
    *,
    build=None,
) -> tuple[dict[str, Any], Any, pd.DataFrame]:
    config = load_thumbnail_generation_config(config_or_path)
    if build is None:
        build = build_track_thumbnails(**thumbnail_build_kwargs(config))

    meta = build.metadata.copy()
    meta["start_time"] = pd.to_datetime(meta["start_time"])
    meta["end_time"] = pd.to_datetime(meta["end_time"])
    source_name = Path(config["data"]["csv_path"]).stem
    meta["source_name"] = source_name
    meta["csv_path"] = config["data"]["csv_path"]
    meta["selected_split"] = config["data"]["split"]
    meta["overlap_state"] = meta.apply(overlap_state_from_row, axis=1)
    meta["thumbnail_id"] = meta.apply(thumbnail_id_for_row, axis=1, source_name=source_name)
    meta["description"] = meta.apply(description_for_row, axis=1)
    return config, build, meta


def save_thumbnail_pngs(
    images: np.ndarray,
    metadata: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> list[str]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx, image in enumerate(images):
        thumb_id = str(metadata.iloc[idx]["thumbnail_id"])
        file_name = f"{thumb_id.replace('|', '__')}.png"
        path = target / file_name
        mode = "L" if image.ndim == 2 else "RGB"
        Image.fromarray(image, mode=mode).save(path, format="PNG", optimize=True)
        paths.append(str(path.resolve()))
    return paths


def serialize_neighbors(
    neighbor_indices: np.ndarray,
    thumbnail_ids: list[str],
    *,
    neighbor_distances: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, nbrs in enumerate(np.asarray(neighbor_indices)):
        ids = [str(thumbnail_ids[int(n)]) for n in np.asarray(nbrs).tolist() if 0 <= int(n) < len(thumbnail_ids)]
        payload: dict[str, Any] = {"ids": ids}
        if neighbor_distances is not None and len(neighbor_distances) > idx:
            payload["distances"] = [float(v) for v in np.asarray(neighbor_distances[idx]).tolist()[: len(ids)]]
        rows.append(payload)
    return rows


def serialize_row_index_neighbors(
    neighbor_indices: np.ndarray,
    *,
    neighbor_distances: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, nbrs in enumerate(np.asarray(neighbor_indices)):
        ids = [int(n) for n in np.asarray(nbrs).tolist() if int(n) >= 0]
        payload: dict[str, Any] = {"ids": ids}
        if neighbor_distances is not None and len(neighbor_distances) > idx:
            payload["distances"] = [
                float(v) for v in np.asarray(neighbor_distances[idx]).tolist()[: len(ids)]
            ]
        rows.append(payload)
    return rows


def thumbnail_png_bytes_list(images: np.ndarray) -> list[bytes]:
    out: list[bytes] = []
    for image in images:
        mode = "L" if image.ndim == 2 else "RGB"
        buf = io.BytesIO()
        Image.fromarray(image, mode=mode).save(buf, format="PNG", optimize=True)
        out.append(buf.getvalue())
    return out


def build_atlas_dataframe(
    metadata: pd.DataFrame,
    *,
    projection: np.ndarray,
    neighbors: list[dict[str, Any]],
    thumbnail_paths: list[str | None],
    atlas_neighbors: list[dict[str, Any]] | None = None,
    thumbnail_images: list[bytes] | None = None,
) -> pd.DataFrame:
    df = metadata.copy()
    df["atlas_row_id"] = np.arange(len(df), dtype=np.int64)
    df["projection_x"] = projection[:, 0] if len(projection) else np.array([], dtype=np.float32)
    df["projection_y"] = projection[:, 1] if len(projection) else np.array([], dtype=np.float32)
    df["neighbors"] = neighbors
    df["atlas_neighbors"] = atlas_neighbors if atlas_neighbors is not None else [{} for _ in range(len(df))]
    df["thumbnail_path"] = thumbnail_paths
    if thumbnail_images is not None:
        df["thumbnail_image"] = thumbnail_images
    return df


def save_atlas_artifacts(
    *,
    config: dict[str, Any],
    atlas_df: pd.DataFrame,
    embeddings: np.ndarray,
    out_root: str | Path,
) -> Path:
    run_dir = Path(out_root) / thumbnail_config_run_slug(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir = run_dir / "thumbnails"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "thumbnail_config.json").write_text(
        dumps_thumbnail_generation_config(config),
        encoding="utf-8",
    )
    atlas_df.to_parquet(run_dir / "thumbnail_metadata.parquet", index=False)
    np.save(run_dir / "dino_embeddings.npy", embeddings.astype(np.float32, copy=False))
    summary = {
        "n_thumbnails": int(len(atlas_df)),
        "n_individuals": int(atlas_df["individual_id"].astype(str).nunique()) if "individual_id" in atlas_df.columns else 0,
        "projection_columns": ["projection_x", "projection_y"],
        "thumbnail_dir": str((run_dir / "thumbnails").resolve()),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return run_dir


def build_atlas_artifacts_from_config(
    config_or_path: dict[str, Any] | str | Path,
    *,
    out_root: str | Path,
    embedding_config: dict[str, Any] | None = None,
    projection_config: dict[str, Any] | None = None,
) -> tuple[Path, pd.DataFrame, np.ndarray]:
    config, build, meta = build_thumbnail_metadata(config_or_path)
    run_dir = Path(out_root) / thumbnail_config_run_slug(config)
    thumb_paths = save_thumbnail_pngs(build.images, meta, out_dir=run_dir / "thumbnails")
    embeddings, _ = compute_thumbnail_embeddings(
        build.images,
        embedding_config=dict(DEFAULT_EMBEDDING_CONFIG if embedding_config is None else embedding_config),
    )
    projection, _projection_info, _warnings = project_embeddings(
        embeddings,
        projection_config=dict(DEFAULT_PROJECTION_CONFIG if projection_config is None else projection_config),
        cache_root=run_dir / ".cache",
    )
    neighbor_idx, neighbor_dist = compute_neighbors(embeddings)
    neighbors = serialize_neighbors(neighbor_idx, meta["thumbnail_id"].astype(str).tolist(), neighbor_distances=neighbor_dist)
    atlas_neighbors = serialize_row_index_neighbors(
        neighbor_idx,
        neighbor_distances=neighbor_dist,
    )
    atlas_df = build_atlas_dataframe(
        meta,
        projection=projection,
        neighbors=neighbors,
        thumbnail_paths=thumb_paths,
        atlas_neighbors=atlas_neighbors,
        thumbnail_images=thumbnail_png_bytes_list(build.images),
    )
    save_atlas_artifacts(
        config=config,
        atlas_df=atlas_df,
        embeddings=embeddings,
        out_root=out_root,
    )
    return run_dir, atlas_df, embeddings
