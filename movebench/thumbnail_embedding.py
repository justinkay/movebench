from __future__ import annotations

import base64
import io
import json
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn.functional as F

from movebench.datasets.thumbnails import (
    build_track_thumbnails,
    detect_color_columns,
    load_thumbnail_source_df,
)
from movebench.paths import repo_root

SOURCE_CSV_ROOT = Path(
    "/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example/cleaned_hightemporalres"
).resolve()
CACHE_ROOT = (repo_root() / ".cache" / "thumbnail_embedding").resolve()
DEFAULT_REPORT_ROOT = (CACHE_ROOT / "reports").resolve()
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SUMMARY_LABELS = (
    "clean",
    "has_any_outlier",
    "has_algorithm_marked_outlier",
    "has_manually_marked_outlier",
    "overlap_state",
)


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    manifest_path: Path
    config_path: Path
    metadata_path: Path
    embeddings_path: Path
    projections_path: Path
    neighbors_idx_path: Path
    neighbors_dist_path: Path
    thumbnails_path: Path


def _ensure_cache_dirs(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    numba_cache = cache_root / "numba_cache"
    mpl_cache = cache_root / "mplconfig"
    numba_cache.mkdir(parents=True, exist_ok=True)
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))


@lru_cache(maxsize=4)
def list_embedding_sources(source_root: Path | str = SOURCE_CSV_ROOT) -> list[dict[str, Any]]:
    root = Path(source_root).resolve()
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.csv")):
        if not path.is_file() or path.name.startswith("_"):
            continue
        header = pd.read_csv(path, nrows=0)
        if "set" in header.columns:
            split_df = pd.read_csv(path, usecols=["set"], low_memory=False)
            splits = sorted({str(v) for v in split_df["set"].dropna().tolist()})
        else:
            splits = []
        inspect_df = load_thumbnail_source_df(path, split=None, include_movement_features=True)
        rows.append(
            {
                "label": path.name,
                "path": str(path),
                "splits": ["all"] + splits,
                "default_split": "train" if "train" in splits else "all",
                "n_rows": int(len(inspect_df)),
                "color_columns": detect_color_columns(inspect_df),
            }
        )
    return rows


def default_run_config(csv_path: str | None = None) -> dict[str, Any]:
    sources = list_embedding_sources()
    first_source = csv_path or (sources[0]["path"] if sources else "")
    default_split = "train"
    for row in sources:
        if row["path"] == first_source:
            default_split = row["default_split"]
            break
    return {
        "data": {
            "csv_path": first_source,
            "split": default_split,
        },
        "thumbnail": {
            "img_size": 64,
            "chunk_size": 50,
            "stride": None,
            "min_chunk_size": 10,
            "line_width": 1,
            "marker_radius": 1,
            "padding": 2,
            "scale_mode": "individual",
            "scale_quantile": 0.5,
            "color_by": None,
        },
        "embedding": {
            "model_name": "dinov2_vits14",
            "device_preference": "auto",
            "batch_size": 32,
            "image_size": 224,
        },
        "projection": {
            "method": "umap",
            "random_state": 17,
            "n_neighbors": 20,
            "min_dist": 0.10,
            "perplexity": 30.0,
        },
    }


def normalize_run_config(config: dict[str, Any]) -> dict[str, Any]:
    base = default_run_config(config.get("data", {}).get("csv_path"))
    data = dict(base["data"])
    data.update(config.get("data", {}))
    thumb = dict(base["thumbnail"])
    thumb.update(config.get("thumbnail", {}))
    emb = dict(base["embedding"])
    emb.update(config.get("embedding", {}))
    proj = dict(base["projection"])
    proj.update(config.get("projection", {}))

    csv_path = str(Path(data["csv_path"]).resolve())
    stride = thumb.get("stride")
    split = str(data.get("split") or "all")
    color_by = thumb.get("color_by")
    if color_by in ("", "none", "None"):
        color_by = None

    return {
        "data": {
            "csv_path": csv_path,
            "split": split,
        },
        "thumbnail": {
            "img_size": int(thumb["img_size"]),
            "chunk_size": int(thumb["chunk_size"]),
            "stride": None if stride in (None, "", 0, "0") else int(stride),
            "min_chunk_size": int(thumb["min_chunk_size"]),
            "line_width": int(thumb["line_width"]),
            "marker_radius": int(thumb["marker_radius"]),
            "padding": int(thumb["padding"]),
            "scale_mode": str(thumb["scale_mode"]),
            "scale_quantile": float(thumb["scale_quantile"]),
            "color_by": color_by,
        },
        "embedding": {
            "model_name": str(emb["model_name"]),
            "device_preference": str(emb["device_preference"]).lower(),
            "batch_size": max(1, int(emb["batch_size"])),
            "image_size": max(32, int(emb["image_size"])),
        },
        "projection": {
            "method": str(proj["method"]).lower(),
            "random_state": int(proj["random_state"]),
            "n_neighbors": max(2, int(proj["n_neighbors"])),
            "min_dist": float(proj["min_dist"]),
            "perplexity": float(proj["perplexity"]),
        },
    }


def compute_run_id(config: dict[str, Any]) -> str:
    normalized = normalize_run_config(config)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()[:16]


def run_paths(run_id: str, cache_root: Path | str = CACHE_ROOT) -> RunPaths:
    root = Path(cache_root).resolve() / run_id
    return RunPaths(
        run_dir=root,
        manifest_path=root / "manifest.json",
        config_path=root / "config.json",
        metadata_path=root / "metadata.csv",
        embeddings_path=root / "embeddings.npy",
        projections_path=root / "projection.npy",
        neighbors_idx_path=root / "neighbors_idx.npy",
        neighbors_dist_path=root / "neighbors_dist.npy",
        thumbnails_path=root / "thumbnails.npz",
    )


def pick_device(device_preference: str = "auto") -> str:
    pref = str(device_preference).lower()
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "cpu":
        return "cpu"
    if pref == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def preprocess_thumbnail_image(
    image: np.ndarray,
    *,
    image_size: int = 224,
) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    pil = Image.fromarray(arr, mode="RGB")
    pil = pil.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    resized = np.asarray(pil, dtype=np.float32) / 255.0
    resized = (resized - _RGB_MEAN) / _RGB_STD
    chw = np.transpose(resized, (2, 0, 1))
    return torch.from_numpy(chw.copy())


def preprocess_thumbnail_batch(
    images: np.ndarray,
    *,
    image_size: int = 224,
) -> torch.Tensor:
    tensors = [preprocess_thumbnail_image(img, image_size=image_size) for img in images]
    if not tensors:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    return torch.stack(tensors, dim=0)


@lru_cache(maxsize=4)
def _load_dinov2_model(model_name: str, device: str):
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=True,
            trust_repo=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to load the requested DINOv2 model via torch.hub. "
            "Make sure the machine can populate the torch hub cache for "
            "'facebookresearch/dinov2', or provide a warmed cache before running the explorer."
        ) from exc
    model = model.eval().to(device)
    return model


def _extract_model_features(model, batch: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        out = model(batch)
    if isinstance(out, torch.Tensor):
        feats = out
    elif isinstance(out, dict):
        for key in ("x_norm_clstoken", "cls_token", "embedding", "last_hidden_state"):
            value = out.get(key)
            if isinstance(value, torch.Tensor):
                feats = value
                break
        else:
            raise TypeError(f"Unsupported DINO output keys: {sorted(out.keys())}")
    elif isinstance(out, (tuple, list)) and out:
        first = out[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError("The DINO model returned a non-tensor tuple/list output.")
        feats = first
    else:
        raise TypeError(f"Unsupported DINO output type: {type(out)!r}")

    if feats.ndim == 3:
        feats = feats[:, 0, :]
    if feats.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape={tuple(feats.shape)}")
    return F.normalize(feats, dim=1)


def compute_thumbnail_embeddings(
    images: np.ndarray,
    *,
    embedding_config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(images) == 0:
        return np.zeros((0, 384), dtype=np.float32), {
            "device": pick_device(embedding_config.get("device_preference", "auto")),
            "model_name": str(embedding_config.get("model_name", "dinov2_vits14")),
            "image_size": int(embedding_config.get("image_size", 224)),
            "batch_size": int(embedding_config.get("batch_size", 32)),
        }
    device = pick_device(embedding_config.get("device_preference", "auto"))
    model_name = str(embedding_config.get("model_name", "dinov2_vits14"))
    image_size = int(embedding_config.get("image_size", 224))
    batch_size = int(embedding_config.get("batch_size", 32))
    model = _load_dinov2_model(model_name, device)

    chunks: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        batch_arr = images[start : start + batch_size]
        batch = preprocess_thumbnail_batch(batch_arr, image_size=image_size).to(device)
        feats = _extract_model_features(model, batch).detach().cpu().numpy().astype(np.float32, copy=False)
        chunks.append(feats)

    embeddings = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 384), dtype=np.float32)
    return embeddings, {
        "device": device,
        "model_name": model_name,
        "image_size": image_size,
        "batch_size": batch_size,
    }


def project_embeddings(
    embeddings: np.ndarray,
    *,
    projection_config: dict[str, Any],
    cache_root: Path | str = CACHE_ROOT,
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    _ensure_cache_dirs(Path(cache_root))
    warnings: list[str] = []
    method = str(projection_config.get("method", "umap")).lower()
    random_state = int(projection_config.get("random_state", 17))
    if len(embeddings) == 0:
        return np.zeros((0, 2), dtype=np.float32), {"method": method}, warnings
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype=np.float32), {"method": method}, warnings

    arr = np.asarray(embeddings, dtype=np.float32)
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(max(2, int(projection_config.get("n_neighbors", 20))), len(arr) - 1),
                min_dist=float(projection_config.get("min_dist", 0.10)),
                random_state=random_state,
            )
            coords = reducer.fit_transform(arr).astype(np.float32, copy=False)
            return coords, {"method": "umap"}, warnings
        except Exception as exc:
            warnings.append(f"UMAP unavailable; fell back to PCA. Details: {exc}")
            method = "pca"

    if method == "tsne":
        if len(arr) < 3:
            warnings.append("t-SNE requires at least 3 samples; fell back to PCA.")
            method = "pca"
        else:
            perplexity = float(projection_config.get("perplexity", 30.0))
            perplexity = min(perplexity, max(1.0, float(len(arr) - 1) / 3.0))
            reducer = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=random_state,
            )
            coords = reducer.fit_transform(arr).astype(np.float32, copy=False)
            return coords, {"method": "tsne", "perplexity": perplexity}, warnings

    reducer = PCA(n_components=2, random_state=random_state)
    coords = reducer.fit_transform(arr).astype(np.float32, copy=False)
    return coords, {"method": "pca"}, warnings


def compute_neighbors(embeddings: np.ndarray, *, n_neighbors: int = 10) -> tuple[np.ndarray, np.ndarray]:
    if len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.int32), np.zeros((0, 0), dtype=np.float32)
    if len(embeddings) == 1:
        return np.zeros((1, 0), dtype=np.int32), np.zeros((1, 0), dtype=np.float32)

    k = min(max(2, int(n_neighbors) + 1), len(embeddings))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    return (
        indices[:, 1:].astype(np.int32, copy=False),
        distances[:, 1:].astype(np.float32, copy=False),
    )


def _overlap_state(row: pd.Series) -> str:
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


def build_chunk_ids(meta: pd.DataFrame) -> list[str]:
    rows: list[str] = []
    for _, row in meta.iterrows():
        rows.append(
            "|".join(
                [
                    str(row.get("individual_id", "")),
                    str(row.get("split", "")),
                    pd.Timestamp(row["start_time"]).isoformat(),
                    pd.Timestamp(row["end_time"]).isoformat(),
                    str(int(row.get("n_fixes", 0))),
                ]
            )
        )
    return rows


def _build_summary(meta: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_chunks": int(len(meta)),
        "n_individuals": int(meta["individual_id"].astype(str).nunique()) if "individual_id" in meta.columns else 0,
        "split_counts": {},
        "label_counts": {},
    }
    if "split" in meta.columns:
        summary["split_counts"] = {
            str(k): int(v) for k, v in meta["split"].astype(str).value_counts().sort_index().items()
        }
    for label in _SUMMARY_LABELS[:-1]:
        if label in meta.columns:
            summary["label_counts"][label] = int(meta[label].fillna(False).astype(bool).sum())
    if "overlap_state" in meta.columns:
        summary["label_counts"]["overlap_state"] = {
            str(k): int(v) for k, v in meta["overlap_state"].astype(str).value_counts().sort_index().items()
        }
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(run_id: str, *, cache_root: Path | str = CACHE_ROOT) -> dict[str, Any]:
    paths = run_paths(run_id, cache_root)
    if not paths.manifest_path.exists():
        raise FileNotFoundError(f"No manifest found for run_id={run_id!r}")
    return json.loads(paths.manifest_path.read_text(encoding="utf-8"))


def load_run_data(run_id: str, *, cache_root: Path | str = CACHE_ROOT) -> dict[str, Any]:
    paths = run_paths(run_id, cache_root)
    manifest = load_manifest(run_id, cache_root=cache_root)
    metadata = pd.read_csv(paths.metadata_path)
    projection = np.load(paths.projections_path)
    neighbors_idx = np.load(paths.neighbors_idx_path)
    neighbors_dist = np.load(paths.neighbors_dist_path)
    thumbs = np.load(paths.thumbnails_path)["images"]
    embeddings = np.load(paths.embeddings_path)
    return {
        "manifest": manifest,
        "metadata": metadata,
        "projection": projection,
        "neighbors_idx": neighbors_idx,
        "neighbors_dist": neighbors_dist,
        "thumbnails": thumbs,
        "embeddings": embeddings,
    }


def _thumbnail_params(config: dict[str, Any]) -> dict[str, Any]:
    thumb = dict(config["thumbnail"])
    return {
        "img_size": thumb["img_size"],
        "chunk_size": thumb["chunk_size"],
        "stride": thumb["stride"],
        "min_chunk_size": thumb["min_chunk_size"],
        "line_width": thumb["line_width"],
        "marker_radius": thumb["marker_radius"],
        "padding": thumb["padding"],
        "scale_mode": thumb["scale_mode"],
        "scale_quantile": thumb["scale_quantile"],
        "color_by": thumb["color_by"],
    }


def execute_run(
    config: dict[str, Any],
    *,
    cache_root: Path | str = CACHE_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    normalized = normalize_run_config(config)
    run_id = compute_run_id(normalized)
    paths = run_paths(run_id, cache_root)
    _ensure_cache_dirs(Path(cache_root))
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    if paths.manifest_path.exists() and not force:
        manifest = load_manifest(run_id, cache_root=cache_root)
        if manifest.get("status") == "completed":
            return manifest

    split_arg = None if normalized["data"]["split"] == "all" else normalized["data"]["split"]
    running_manifest = {
        "run_id": run_id,
        "status": "running",
        "config": normalized,
        "warnings": [],
        "summary": {},
    }
    _write_json(paths.manifest_path, running_manifest)
    _write_json(paths.config_path, normalized)

    try:
        build = build_track_thumbnails(
            normalized["data"]["csv_path"],
            split=split_arg,
            img_size=int(normalized["thumbnail"]["img_size"]),
            chunk_size=int(normalized["thumbnail"]["chunk_size"]),
            stride=normalized["thumbnail"]["stride"],
            min_chunk_size=int(normalized["thumbnail"]["min_chunk_size"]),
            line_width=int(normalized["thumbnail"]["line_width"]),
            marker_radius=int(normalized["thumbnail"]["marker_radius"]),
            padding=int(normalized["thumbnail"]["padding"]),
            scale_mode=str(normalized["thumbnail"]["scale_mode"]),
            scale_quantile=float(normalized["thumbnail"]["scale_quantile"]),
            color_by=normalized["thumbnail"]["color_by"],
        )
        meta = build.metadata.copy()
        meta["start_time"] = pd.to_datetime(meta["start_time"])
        meta["end_time"] = pd.to_datetime(meta["end_time"])
        meta["chunk_id"] = build_chunk_ids(meta)
        meta["overlap_state"] = meta.apply(_overlap_state, axis=1)

        embeddings, embedding_info = compute_thumbnail_embeddings(
            build.images,
            embedding_config=normalized["embedding"],
        )
        projection, projection_info, warnings = project_embeddings(
            embeddings,
            projection_config=normalized["projection"],
            cache_root=cache_root,
        )
        neighbors_idx, neighbors_dist = compute_neighbors(embeddings)

        meta_to_save = meta.copy()
        meta_to_save["start_time"] = meta_to_save["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        meta_to_save["end_time"] = meta_to_save["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
        meta_to_save.to_csv(paths.metadata_path, index=False)
        np.save(paths.embeddings_path, embeddings.astype(np.float32, copy=False))
        np.save(paths.projections_path, projection.astype(np.float32, copy=False))
        np.save(paths.neighbors_idx_path, neighbors_idx)
        np.save(paths.neighbors_dist_path, neighbors_dist)
        np.savez_compressed(paths.thumbnails_path, images=build.images)

        summary = _build_summary(meta)
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "config": normalized,
            "warnings": warnings,
            "summary": summary,
            "embedding": embedding_info,
            "projection": projection_info,
            "artifact_paths": {
                "run_dir": str(paths.run_dir),
                "metadata": str(paths.metadata_path),
                "projection": str(paths.projections_path),
                "thumbnails": str(paths.thumbnails_path),
            },
        }
        _write_json(paths.manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest = {
            "run_id": run_id,
            "status": "failed",
            "config": normalized,
            "warnings": [],
            "error": str(exc),
            "summary": {},
        }
        _write_json(paths.manifest_path, manifest)
        raise


def run_points_payload(run_id: str, *, cache_root: Path | str = CACHE_ROOT) -> dict[str, Any]:
    data = load_run_data(run_id, cache_root=cache_root)
    manifest = data["manifest"]
    meta = data["metadata"]
    coords = data["projection"]
    points: list[dict[str, Any]] = []
    for idx, row in meta.iterrows():
        points.append(
            {
                "idx": int(idx),
                "chunk_id": str(row["chunk_id"]),
                "x": float(coords[idx, 0]) if len(coords) else 0.0,
                "y": float(coords[idx, 1]) if len(coords) else 0.0,
                "individual_id": str(row.get("individual_id", "")),
                "split": str(row.get("split", "")),
                "start_time": str(row.get("start_time", "")),
                "end_time": str(row.get("end_time", "")),
                "n_fixes": int(row.get("n_fixes", 0)),
                "outlier_frac": float(row.get("outlier_frac", 0.0)),
                "clean": bool(row.get("clean", False)),
                "has_any_outlier": bool(row.get("has_any_outlier", False)),
                "has_algorithm_marked_outlier": bool(row.get("has_algorithm_marked_outlier", False)),
                "has_manually_marked_outlier": bool(row.get("has_manually_marked_outlier", False)),
                "overlap_state": str(row.get("overlap_state", "clean")),
                "thumbnail_url": f"/api/runs/{run_id}/thumbnail/{idx}",
            }
        )
    return {
        "run_id": run_id,
        "status": manifest["status"],
        "summary": manifest.get("summary", {}),
        "warnings": manifest.get("warnings", []),
        "projection": manifest.get("projection", {}),
        "color_modes": list(_SUMMARY_LABELS),
        "points": points,
    }


def chunk_detail_payload(
    run_id: str,
    *,
    idx: int,
    cache_root: Path | str = CACHE_ROOT,
) -> dict[str, Any]:
    data = load_run_data(run_id, cache_root=cache_root)
    meta = data["metadata"]
    if idx < 0 or idx >= len(meta):
        raise IndexError(f"Chunk index {idx} is out of bounds for run {run_id}.")
    row = meta.iloc[idx]
    neighbors_idx = data["neighbors_idx"]
    neighbors_dist = data["neighbors_dist"]
    neighbor_rows: list[dict[str, Any]] = []
    if len(neighbors_idx) > idx:
        for n_idx, dist in zip(neighbors_idx[idx].tolist(), neighbors_dist[idx].tolist()):
            if n_idx < 0 or n_idx >= len(meta):
                continue
            n_row = meta.iloc[int(n_idx)]
            neighbor_rows.append(
                {
                    "idx": int(n_idx),
                    "chunk_id": str(n_row["chunk_id"]),
                    "distance": float(dist),
                    "individual_id": str(n_row.get("individual_id", "")),
                    "start_time": str(n_row.get("start_time", "")),
                    "overlap_state": str(n_row.get("overlap_state", "clean")),
                    "thumbnail_url": f"/api/runs/{run_id}/thumbnail/{int(n_idx)}",
                }
            )
    return {
        "idx": int(idx),
        "chunk_id": str(row["chunk_id"]),
        "individual_id": str(row.get("individual_id", "")),
        "split": str(row.get("split", "")),
        "start_time": str(row.get("start_time", "")),
        "end_time": str(row.get("end_time", "")),
        "n_fixes": int(row.get("n_fixes", 0)),
        "outlier_frac": float(row.get("outlier_frac", 0.0)),
        "overlap_state": str(row.get("overlap_state", "clean")),
        "thumbnail_url": f"/api/runs/{run_id}/thumbnail/{idx}",
        "neighbors": neighbor_rows,
    }


def thumbnail_png_bytes(run_id: str, *, idx: int, cache_root: Path | str = CACHE_ROOT) -> bytes:
    data = load_run_data(run_id, cache_root=cache_root)
    images = data["thumbnails"]
    if idx < 0 or idx >= len(images):
        raise IndexError(f"Thumbnail index {idx} is out of bounds for run {run_id}.")
    arr = images[idx]
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _sample_report_indices(meta: pd.DataFrame, *, max_per_state: int = 6) -> list[int]:
    if len(meta) == 0:
        return []
    idxs: list[int] = []
    for state, group in meta.groupby("overlap_state", sort=True):
        g = group.sort_values(["outlier_frac", "start_time"], ascending=[False, True], kind="mergesort")
        idxs.extend(g.index[:max_per_state].tolist())
    return sorted(set(int(i) for i in idxs))


def _thumb_data_url(arr: np.ndarray) -> str:
    mode = "L" if arr.ndim == 2 else "RGB"
    pil = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def export_embedding_report(
    run_id: str,
    *,
    baseline_run_id: str | None = None,
    out_dir: Path | str | None = None,
    cache_root: Path | str = CACHE_ROOT,
) -> Path:
    current = load_run_data(run_id, cache_root=cache_root)
    baseline = load_run_data(baseline_run_id, cache_root=cache_root) if baseline_run_id else None

    target_dir = Path(out_dir).resolve() if out_dir is not None else (DEFAULT_REPORT_ROOT / run_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    def _scatter_points(data: dict[str, Any]) -> list[dict[str, Any]]:
        points = run_points_payload(data["manifest"]["run_id"], cache_root=cache_root)["points"]
        return [
            {
                "x": p["x"],
                "y": p["y"],
                "label": p["overlap_state"],
                "idx": p["idx"],
            }
            for p in points
        ]

    cur_meta = current["metadata"]
    cur_images = current["thumbnails"]
    sample_rows = []
    for idx in _sample_report_indices(cur_meta):
        row = cur_meta.iloc[idx]
        sample_rows.append(
            {
                "idx": int(idx),
                "title": f"{row['individual_id']} | {row['start_time']}",
                "subtitle": f"{row['overlap_state']} | outlier_frac={float(row.get('outlier_frac', 0.0)):.2f}",
                "thumb": _thumb_data_url(cur_images[idx]),
            }
        )

    compare_summary: dict[str, Any] | None = None
    if baseline is not None:
        cur_ids = set(cur_meta["chunk_id"].astype(str).tolist())
        base_ids = set(baseline["metadata"]["chunk_id"].astype(str).tolist())
        compare_summary = {
            "shared_chunk_ids": int(len(cur_ids & base_ids)),
            "current_only_chunk_ids": int(len(cur_ids - base_ids)),
            "baseline_only_chunk_ids": int(len(base_ids - cur_ids)),
        }

    payload = {
        "current": {
            "run_id": run_id,
            "config": current["manifest"]["config"],
            "summary": current["manifest"].get("summary", {}),
            "warnings": current["manifest"].get("warnings", []),
            "points": _scatter_points(current),
        },
        "baseline": None
        if baseline is None
        else {
            "run_id": baseline_run_id,
            "config": baseline["manifest"]["config"],
            "summary": baseline["manifest"].get("summary", {}),
            "warnings": baseline["manifest"].get("warnings", []),
            "points": _scatter_points(baseline),
        },
        "compare_summary": compare_summary,
        "samples": sample_rows,
    }
    html = _embedding_report_html(payload)
    index_path = target_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _embedding_report_html(payload: dict[str, Any]) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Thumbnail Embedding Report</title>
  <style>
    :root {{
      --bg: #f5f0e6;
      --panel: rgba(255,255,255,0.92);
      --ink: #17211f;
      --muted: #5a625e;
      --line: rgba(23,33,31,0.14);
      --shadow: 0 16px 34px rgba(17, 26, 24, 0.12);
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(76, 123, 111, 0.14), transparent 28%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }}
    .wrap {{ padding: 22px; }}
    .title {{
      margin: 0 0 14px 0;
      font-size: 30px;
      letter-spacing: -0.03em;
    }}
    .subtitle {{
      margin: 0 0 22px 0;
      color: var(--muted);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(320px, 1.3fr) minmax(280px, 0.9fr);
      gap: 18px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
    }}
    .two-up {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(90px, 1fr));
      gap: 10px;
      margin-top: 10px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: rgba(255,255,255,0.72);
    }}
    .stat .k {{ color: var(--muted); font-size: 12px; }}
    .stat .v {{ font-size: 20px; font-weight: 600; }}
    canvas {{
      width: 100%;
      height: 340px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: white;
    }}
    .samples {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .sample {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: white;
    }}
    .sample img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 10px;
      background: #f5f6f7;
      display: block;
    }}
    .sample .t {{
      margin-top: 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .sample .s {{
      margin-top: 4px;
      font-size: 11px;
      color: var(--muted);
    }}
    pre {{
      white-space: pre-wrap;
      font-size: 12px;
      margin: 0;
    }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .two-up {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Thumbnail Embedding Report</h1>
    <p class="subtitle">Run {payload["current"]["run_id"]}</p>
    <div class="layout">
      <section class="panel">
        <div class="two-up">
          <div>
            <h2>Current Run</h2>
            <canvas id="current-canvas" width="560" height="340"></canvas>
          </div>
          <div>
            <h2>{'Baseline Run' if payload.get("baseline") else 'Config'}</h2>
            {"<canvas id='baseline-canvas' width='560' height='340'></canvas>" if payload.get("baseline") else "<pre id='config-panel'></pre>"}
          </div>
        </div>
        <div class="stats" id="stats"></div>
      </section>
      <section class="panel">
        <h2>Representative Thumbnails</h2>
        <div class="samples" id="samples"></div>
      </section>
    </div>
  </div>
  <script>
    const DATA = {payload_json};
    const palette = {{
      "clean": "#2f7d4b",
      "algorithm only": "#da6e2f",
      "manual only": "#147a8a",
      "algorithm + manual": "#9e3fcb",
      "other outlier": "#c23c3c",
    }};
    function drawScatter(canvasId, points) {{
      const canvas = document.getElementById(canvasId);
      if (!canvas || !points || !points.length) return;
      const ctx = canvas.getContext("2d");
      const xs = points.map(p => p.x);
      const ys = points.map(p => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 18;
      const sx = (v) => pad + ((v - minX) / Math.max(1e-6, maxX - minX)) * (canvas.width - pad * 2);
      const sy = (v) => canvas.height - pad - ((v - minY) / Math.max(1e-6, maxY - minY)) * (canvas.height - pad * 2);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const p of points) {{
        ctx.fillStyle = palette[p.label] || "#6c7672";
        ctx.beginPath();
        ctx.arc(sx(p.x), sy(p.y), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }}
    }}
    drawScatter("current-canvas", DATA.current.points);
    if (DATA.baseline) drawScatter("baseline-canvas", DATA.baseline.points);
    const stats = document.getElementById("stats");
    const summary = DATA.current.summary || {{}};
    const statRows = [
      ["Chunks", summary.n_chunks || 0],
      ["Individuals", summary.n_individuals || 0],
      ["Warnings", (DATA.current.warnings || []).length],
    ];
    if (DATA.compare_summary) statRows.push(["Shared Chunks", DATA.compare_summary.shared_chunk_ids || 0]);
    stats.innerHTML = statRows.map(([k, v]) => `<div class="stat"><div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join("");
    const samples = document.getElementById("samples");
    samples.innerHTML = (DATA.samples || []).map((s) => `
      <div class="sample">
        <img src="${{s.thumb}}" alt="${{s.title}}" />
        <div class="t">${{s.title}}</div>
        <div class="s">${{s.subtitle}}</div>
      </div>
    `).join("");
    const cfg = document.getElementById("config-panel");
    if (cfg) cfg.textContent = JSON.stringify(DATA.current.config, null, 2);
  </script>
</body>
</html>
"""
