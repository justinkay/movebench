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
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from movebench.datasets.thumbnails import (
    build_track_thumbnails,
    detect_color_columns,
    load_thumbnail_source_df_with_anomaly,
)
from movebench.paths import repo_root
from movebench.thumbnail_config import normalize_anomaly_config, normalize_embedding_config

SOURCE_CSV_ROOT = Path(
    "/data/vision/beery/scratch/shir/codes/vibecleaning/data/movement_example/cleaned_hightemporalres"
).resolve()
CACHE_ROOT = (repo_root() / ".cache" / "thumbnail_embedding").resolve()
DEFAULT_REPORT_ROOT = (CACHE_ROOT / "reports").resolve()
_RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_EMBEDDING_FEATURE_SOURCES = (
    "cls_token",
    "patch_mean",
    "intermediate_patch_mean",
)
_EXTRA_EMBEDDING_FEATURES = (
    "extent_x_m",
    "extent_y_m",
)
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
        anomaly = normalize_anomaly_config(None)
        inspect_df = load_thumbnail_source_df_with_anomaly(
            path,
            split=None,
            include_movement_features=True,
            anomaly_config=anomaly,
        )
        rows.append(
            {
                "label": path.name,
                "path": str(path),
                "splits": ["all"] + splits,
                "default_split": "train" if "train" in splits else "all",
                "n_rows": int(len(inspect_df)),
                "color_columns": detect_color_columns(
                    inspect_df,
                    extra_columns=[anomaly["score_column"]],
                ),
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
            "individual_id": None,
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
        "anomaly": normalize_anomaly_config(None),
        "embedding": {
            "model_name": "dinov2_vits14",
            "device_preference": "auto",
            "batch_size": 32,
            "image_size": 224,
            "feature_source": "cls_token",
            "intermediate_layer_offset": 2,
            "extra_features": [],
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
    anomaly = normalize_anomaly_config(config.get("anomaly"))

    csv_path = str(Path(data["csv_path"]).resolve())
    stride = thumb.get("stride")
    split = str(data.get("split") or "all")
    individual_id = data.get("individual_id")
    if individual_id in (None, "", "all", "All individuals"):
        individual_id = None
    else:
        individual_id = str(individual_id)
    color_by = thumb.get("color_by")
    if color_by in ("", "none", "None"):
        color_by = None
    embedding = normalize_embedding_config(emb)

    return {
        "data": {
            "csv_path": csv_path,
            "split": split,
            "individual_id": individual_id,
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
        "anomaly": anomaly,
        "embedding": embedding,
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
    try:
        import torch
    except Exception:
        return "cpu"
    if pref == "cuda" and torch.cuda.is_available():
        return "cuda"
    if pref == "cpu":
        return "cpu"
    if pref == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def preprocess_thumbnail_image_array(
    image: np.ndarray,
    *,
    image_size: int = 224,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    pil = Image.fromarray(arr, mode="RGB")
    pil = pil.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)
    resized = np.asarray(pil, dtype=np.float32) / 255.0
    resized = (resized - _RGB_MEAN) / _RGB_STD
    return np.transpose(resized, (2, 0, 1)).astype(np.float32, copy=False)


def preprocess_thumbnail_image(
    image: np.ndarray,
    *,
    image_size: int = 224,
):
    import torch

    return torch.from_numpy(preprocess_thumbnail_image_array(image, image_size=image_size).copy())


def preprocess_thumbnail_batch_array(
    images: np.ndarray,
    *,
    image_size: int = 224,
) -> np.ndarray:
    arrays = [preprocess_thumbnail_image_array(img, image_size=image_size) for img in images]
    if not arrays:
        return np.zeros((0, 3, image_size, image_size), dtype=np.float32)
    return np.stack(arrays, axis=0).astype(np.float32, copy=False)


def preprocess_thumbnail_batch(
    images: np.ndarray,
    *,
    image_size: int = 224,
):
    import torch

    batch = preprocess_thumbnail_batch_array(images, image_size=image_size)
    return torch.from_numpy(batch.copy())


def denormalize_preprocessed_batch(
    batch: Any,
) -> np.ndarray:
    if hasattr(batch, "detach") and hasattr(batch, "cpu") and hasattr(batch, "numpy"):
        arr = batch.detach().cpu().numpy()
    else:
        arr = np.asarray(batch)
    if arr.ndim != 4 or arr.shape[1] != 3:
        raise ValueError(f"Expected batch with shape (N, 3, H, W), got {tuple(arr.shape)}")
    nhwc = np.transpose(arr, (0, 2, 3, 1)).astype(np.float32, copy=False)
    nhwc = nhwc * _RGB_STD[None, None, None, :] + _RGB_MEAN[None, None, None, :]
    nhwc = np.clip(nhwc, 0.0, 1.0)
    return np.rint(nhwc * 255.0).astype(np.uint8)


def preprocessed_batch_preview_png_bytes(
    images: np.ndarray,
    *,
    embedding_config: dict[str, Any],
    overlay_texts: list[str] | None = None,
    max_items: int = 16,
    cols: int = 4,
    pad: int = 4,
) -> bytes:
    image_size = int(embedding_config.get("image_size", 224))
    sample = np.asarray(images[: max(0, int(max_items))])
    if len(sample) == 0:
        preview = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(preview, mode="RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    batch = preprocess_thumbnail_batch_array(sample, image_size=image_size)
    preview_items = denormalize_preprocessed_batch(batch)
    n = len(preview_items)
    n_cols = max(1, int(cols))
    n_rows = int(math.ceil(float(n) / float(n_cols)))
    tile_h, tile_w = int(preview_items.shape[1]), int(preview_items.shape[2])
    grid = np.full(
        (
            n_rows * tile_h + pad * (n_rows + 1),
            n_cols * tile_w + pad * (n_cols + 1),
            3,
        ),
        255,
        dtype=np.uint8,
    )
    for idx, item in enumerate(preview_items):
        row = idx // n_cols
        col = idx % n_cols
        top = pad + row * (tile_h + pad)
        left = pad + col * (tile_w + pad)
        grid[top : top + tile_h, left : left + tile_w] = item

    preview_image = Image.fromarray(grid, mode="RGB")
    if overlay_texts:
        draw = ImageDraw.Draw(preview_image, mode="RGBA")
        font = ImageFont.load_default()
        overlay_items = list(overlay_texts[:n])
        for idx, text in enumerate(overlay_items):
            if not text:
                continue
            row = idx // n_cols
            col = idx % n_cols
            top = pad + row * (tile_h + pad)
            left = pad + col * (tile_w + pad)
            text_box = draw.textbbox((0, 0), str(text), font=font)
            text_w = int(text_box[2] - text_box[0])
            text_h = int(text_box[3] - text_box[1])
            rect_left = left + 4
            rect_top = top + 4
            rect_right = min(left + tile_w - 4, rect_left + text_w + 8)
            rect_bottom = min(top + tile_h - 4, rect_top + text_h + 8)
            draw.rounded_rectangle(
                (rect_left, rect_top, rect_right, rect_bottom),
                radius=4,
                fill=(0, 0, 0, 170),
            )
            draw.text(
                (rect_left + 4, rect_top + 4),
                str(text),
                font=font,
                fill=(255, 255, 255, 255),
            )

    buf = io.BytesIO()
    preview_image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@lru_cache(maxsize=4)
def _load_dinov2_model(model_name: str, device: str):
    import torch

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


def _embedding_error_info(
    *,
    device: str,
    model_name: str,
    image_size: int,
    batch_size: int,
    **extra: Any,
) -> dict[str, Any]:
    info = {
        "device": device,
        "model_name": model_name,
        "image_size": image_size,
        "batch_size": batch_size,
    }
    info.update(extra)
    return info


def _empty_extra_feature_stats(extra_features: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": str(name),
            "mean": 0.0,
            "std": 0.0,
            "n_finite": 0,
            "status": "empty",
        }
        for name in extra_features
    ]


def _append_extra_embedding_features(
    embeddings: np.ndarray,
    *,
    metadata: pd.DataFrame | None,
    extra_features: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not extra_features:
        return embeddings.astype(np.float32, copy=False), []

    if metadata is None:
        raise ValueError(
            "embedding.extra_features requires chunk metadata; pass metadata=... to "
            "compute_thumbnail_embeddings()."
        )
    if len(metadata) != len(embeddings):
        raise ValueError(
            "Embedding metadata length must match the number of thumbnails, got "
            f"{len(metadata)} rows for {len(embeddings)} embeddings."
        )

    extra_columns: list[np.ndarray] = []
    stats: list[dict[str, Any]] = []
    for feature_name in extra_features:
        if feature_name not in _EXTRA_EMBEDDING_FEATURES:
            raise ValueError(
                f"Unsupported embedding extra feature {feature_name!r}; expected one of {_EXTRA_EMBEDDING_FEATURES}."
            )
        if feature_name not in metadata.columns:
            raise KeyError(f"Requested embedding extra feature {feature_name!r} is missing from metadata.")

        values = pd.to_numeric(metadata[feature_name], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(values)
        mean = float(values[finite].mean()) if np.any(finite) else 0.0
        std = float(values[finite].std(ddof=0)) if np.any(finite) else 0.0
        normalized = np.zeros(len(values), dtype=np.float32)
        if np.any(finite) and std > 0.0:
            normalized[finite] = ((values[finite] - mean) / std).astype(np.float32, copy=False)
            status = "zscore"
        elif np.any(finite):
            status = "zero_variance"
        else:
            status = "all_missing"
        extra_columns.append(normalized[:, None])
        stats.append(
            {
                "name": str(feature_name),
                "mean": mean,
                "std": std,
                "n_finite": int(np.count_nonzero(finite)),
                "status": status,
            }
        )

    extra_arr = np.concatenate(extra_columns, axis=1).astype(np.float32, copy=False)
    augmented = np.concatenate([embeddings.astype(np.float32, copy=False), extra_arr], axis=1)
    return augmented.astype(np.float32, copy=False), stats


def _coerce_feature_tensor(raw: Any) -> Any:
    import torch

    if isinstance(raw, torch.Tensor):
        return raw
    if isinstance(raw, dict):
        for key in ("x_norm_clstoken", "cls_token", "embedding", "last_hidden_state"):
            value = raw.get(key)
            if isinstance(value, torch.Tensor):
                return value
        raise TypeError(f"Unsupported DINO output keys: {sorted(raw.keys())}")
    if isinstance(raw, (tuple, list)) and raw:
        first = raw[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError("The DINO model returned a non-tensor tuple/list output.")
        return first
    raise TypeError(f"Unsupported DINO output type: {type(raw)!r}")


def _extract_from_forward_features(
    model: Any,
    batch: Any,
    *,
    feature_source: str,
):
    import torch

    if not hasattr(model, "forward_features"):
        return None
    with torch.inference_mode():
        features = model.forward_features(batch)
    if not isinstance(features, dict):
        return None
    if feature_source == "cls_token":
        tensor = features.get("x_norm_clstoken")
        if isinstance(tensor, torch.Tensor):
            return tensor
        return None
    if feature_source == "patch_mean":
        tensor = features.get("x_norm_patchtokens")
        if isinstance(tensor, torch.Tensor):
            return tensor.mean(dim=1)
        return None
    return None


def _extract_intermediate_patch_features(
    model: Any,
    batch: Any,
    *,
    intermediate_layer_offset: int,
):
    import torch

    if hasattr(model, "get_intermediate_layers"):
        with torch.inference_mode():
            outputs = model.get_intermediate_layers(
                batch,
                n=max(1, int(intermediate_layer_offset)),
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        if not outputs:
            raise ValueError("DINO model returned no intermediate layers.")
        tensor = outputs[0]
        if isinstance(tensor, torch.Tensor):
            return tensor.mean(dim=1)
        raise TypeError(f"Unexpected intermediate-layer output type: {type(tensor)!r}")
    fallback = _extract_from_forward_features(
        model,
        batch,
        feature_source="patch_mean",
    )
    if fallback is not None:
        return fallback
    raise RuntimeError(
        "Requested feature_source='intermediate_patch_mean', but the loaded model does not expose "
        "get_intermediate_layers() or forward_features()."
    )


def _looks_like_cudnn_init_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "cudnn_status_not_initialized",
            "cudnn error",
            "cudnn_status_alloc_failed",
            "cuda error: cudnn",
        )
    )


def _clear_cuda_runtime_state() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _format_cuda_embedding_failure_message(
    *,
    initial_error: Exception,
    retry_error: Exception,
    model_name: str,
    batch_size: int,
    image_size: int,
) -> str:
    return (
        "DINO embedding on CUDA failed after retrying once with cuDNN disabled. "
        f"model={model_name!r}, batch_size={batch_size}, image_size={image_size}. "
        f"Initial CUDA error: {initial_error}. "
        f"Retry error: {retry_error}. "
        "If you want CPU embeddings, request them explicitly with "
        "device_preference='cpu'."
    )


def _compute_thumbnail_embeddings_once(
    images: np.ndarray,
    *,
    model_name: str,
    image_size: int,
    batch_size: int,
    device: str,
    feature_source: str,
    intermediate_layer_offset: int,
    use_cudnn: bool = True,
) -> np.ndarray:
    import torch

    previous_cudnn_enabled = None
    if device == "cuda":
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")
        previous_cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = bool(use_cudnn)

    try:
        model = _load_dinov2_model(model_name, device)
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch_arr = images[start : start + batch_size]
            batch = preprocess_thumbnail_batch(batch_arr, image_size=image_size).to(device)
            feats = _extract_model_features(
                model,
                batch,
                feature_source=feature_source,
                intermediate_layer_offset=intermediate_layer_offset,
            ).detach().cpu().numpy().astype(
                np.float32,
                copy=False,
            )
            chunks.append(feats)
            del batch
        return (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, 384), dtype=np.float32)
        )
    finally:
        if previous_cudnn_enabled is not None:
            torch.backends.cudnn.enabled = previous_cudnn_enabled


def _extract_model_features(
    model,
    batch: Any,
    *,
    feature_source: str,
    intermediate_layer_offset: int,
):
    import torch
    import torch.nn.functional as F

    if feature_source == "intermediate_patch_mean":
        feats = _extract_intermediate_patch_features(
            model,
            batch,
            intermediate_layer_offset=intermediate_layer_offset,
        )
    else:
        feats = _extract_from_forward_features(
            model,
            batch,
            feature_source=feature_source,
        )
        if feats is None:
            with torch.inference_mode():
                out = model(batch)
            feats = _coerce_feature_tensor(out)
            if feature_source == "patch_mean" and feats.ndim == 3:
                feats = feats.mean(dim=1)

    if feats.ndim == 3:
        feats = feats[:, 0, :]
    if feats.ndim != 2:
        raise ValueError(f"Expected a 2D feature tensor, got shape={tuple(feats.shape)}")
    return F.normalize(feats, dim=1)


def compute_thumbnail_embeddings(
    images: np.ndarray,
    *,
    embedding_config: dict[str, Any],
    metadata: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    normalized_embedding = normalize_embedding_config(embedding_config)
    model_name = str(normalized_embedding.get("model_name", "dinov2_vits14"))
    image_size = int(normalized_embedding.get("image_size", 224))
    batch_size = int(normalized_embedding.get("batch_size", 32))
    feature_source = str(normalized_embedding.get("feature_source", "cls_token")).lower()
    intermediate_layer_offset = int(normalized_embedding.get("intermediate_layer_offset", 2))
    extra_features = [str(name) for name in normalized_embedding.get("extra_features", [])]
    requested_device = pick_device(normalized_embedding.get("device_preference", "auto"))
    if len(images) == 0:
        output_dim = 384 + len(extra_features)
        return np.zeros((0, output_dim), dtype=np.float32), _embedding_error_info(
            device=requested_device,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            feature_source=feature_source,
            intermediate_layer_offset=intermediate_layer_offset,
            extra_features=extra_features,
            extra_feature_stats=_empty_extra_feature_stats(extra_features),
            base_embedding_dim=384,
            output_dim=output_dim,
        )

    try:
        base_embeddings = _compute_thumbnail_embeddings_once(
            images,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            device=requested_device,
            feature_source=feature_source,
            intermediate_layer_offset=intermediate_layer_offset,
            use_cudnn=True,
        )
        embeddings, extra_feature_stats = _append_extra_embedding_features(
            base_embeddings,
            metadata=metadata,
            extra_features=extra_features,
        )
        return embeddings, _embedding_error_info(
            device=requested_device,
            model_name=model_name,
            image_size=image_size,
            batch_size=batch_size,
            feature_source=feature_source,
            intermediate_layer_offset=intermediate_layer_offset,
            cudnn_enabled=bool(requested_device == "cuda"),
            extra_features=extra_features,
            extra_feature_stats=extra_feature_stats,
            base_embedding_dim=int(base_embeddings.shape[1]) if base_embeddings.ndim == 2 else 0,
            output_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        )
    except RuntimeError as exc:
        if requested_device != "cuda" or not _looks_like_cudnn_init_error(exc):
            raise

        _clear_cuda_runtime_state()
        try:
            base_embeddings = _compute_thumbnail_embeddings_once(
                images,
                model_name=model_name,
                image_size=image_size,
                batch_size=batch_size,
                device="cuda",
                feature_source=feature_source,
                intermediate_layer_offset=intermediate_layer_offset,
                use_cudnn=False,
            )
            embeddings, extra_feature_stats = _append_extra_embedding_features(
                base_embeddings,
                metadata=metadata,
                extra_features=extra_features,
            )
            return embeddings, _embedding_error_info(
                device="cuda",
                model_name=model_name,
                image_size=image_size,
                batch_size=batch_size,
                feature_source=feature_source,
                intermediate_layer_offset=intermediate_layer_offset,
                cudnn_enabled=False,
                cudnn_retry_reason=str(exc),
                extra_features=extra_features,
                extra_feature_stats=extra_feature_stats,
                base_embedding_dim=int(base_embeddings.shape[1]) if base_embeddings.ndim == 2 else 0,
                output_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            )
        except RuntimeError as retry_exc:
            _clear_cuda_runtime_state()
            raise RuntimeError(
                _format_cuda_embedding_failure_message(
                    initial_error=exc,
                    retry_error=retry_exc,
                    model_name=model_name,
                    batch_size=batch_size,
                    image_size=image_size,
                )
            ) from retry_exc


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
        "anomaly_config": dict(config["anomaly"]),
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
            individual_id=normalized["data"].get("individual_id"),
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
            anomaly_config=normalized["anomaly"],
        )
        meta = build.metadata.copy()
        meta["start_time"] = pd.to_datetime(meta["start_time"])
        meta["end_time"] = pd.to_datetime(meta["end_time"])
        meta["chunk_id"] = build_chunk_ids(meta)
        meta["overlap_state"] = meta.apply(_overlap_state, axis=1)

        embeddings, embedding_info = compute_thumbnail_embeddings(
            build.images,
            embedding_config=normalized["embedding"],
            metadata=meta,
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
                "extent_x_m": float(row.get("extent_x_m", 0.0)),
                "extent_y_m": float(row.get("extent_y_m", 0.0)),
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
                    "extent_x_m": float(n_row.get("extent_x_m", 0.0)),
                    "extent_y_m": float(n_row.get("extent_y_m", 0.0)),
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
        "extent_x_m": float(row.get("extent_x_m", 0.0)),
        "extent_y_m": float(row.get("extent_y_m", 0.0)),
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
