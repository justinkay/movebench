from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

CONFIG_VERSION = "thumbnail_generation.v1"
DEFAULT_ANOMALY_SCORE_COLUMN = "iforest_anomaly_score"
DEFAULT_ANOMALY_CONFIG = {
    "score_column": DEFAULT_ANOMALY_SCORE_COLUMN,
    "feature_columns": None,
    "n_estimators": 1000,
    "contamination": "auto",
    "random_state": 0,
}
DEFAULT_EMBEDDING_CONFIG = {
    "model_name": "dinov2_vits14",
    "device_preference": "auto",
    "batch_size": 32,
    "image_size": 224,
    "feature_source": "cls_token",
    "intermediate_layer_offset": 2,
    "extra_features": [],
}
DEFAULT_PROJECTION_CONFIG = {
    "method": "umap",
    "random_state": 17,
    "n_neighbors": 20,
    "min_dist": 0.10,
    "perplexity": 30.0,
}
_EMBEDDING_FEATURE_SOURCES = {
    "cls_token",
    "patch_mean",
    "intermediate_patch_mean",
}
_EXTRA_EMBEDDING_FEATURES = {
    "extent_x_m",
    "extent_y_m",
}


def normalize_anomaly_config(anomaly: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(DEFAULT_ANOMALY_CONFIG)
    if anomaly is not None:
        raw.update(dict(anomaly))
    feature_columns = raw.get("feature_columns")
    if feature_columns in (None, "", []):
        normalized_feature_columns = None
    else:
        if isinstance(feature_columns, (str, bytes)):
            feature_columns = [feature_columns]
        normalized_feature_columns = [
            str(col)
            for col in feature_columns
            if str(col).strip()
        ]
    contamination = raw.get("contamination", "auto")
    if contamination != "auto":
        contamination = float(contamination)
    return {
        "score_column": str(raw.get("score_column") or DEFAULT_ANOMALY_SCORE_COLUMN),
        "feature_columns": normalized_feature_columns,
        "n_estimators": max(1, int(raw.get("n_estimators", 1000))),
        "contamination": contamination,
        "random_state": int(raw.get("random_state", 0)),
    }


def normalize_embedding_config(embedding: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(DEFAULT_EMBEDDING_CONFIG)
    if embedding is not None:
        raw.update(dict(embedding))
    feature_source = str(raw.get("feature_source", "cls_token")).lower()
    if feature_source not in _EMBEDDING_FEATURE_SOURCES:
        raise ValueError(
            f"embedding.feature_source must be one of {sorted(_EMBEDDING_FEATURE_SOURCES)}, got {feature_source!r}"
        )
    extra_features = raw.get("extra_features", [])
    if extra_features in (None, "", []):
        normalized_extra_features: list[str] = []
    else:
        if isinstance(extra_features, (str, bytes)):
            extra_features = [extra_features]
        normalized_extra_features = []
        for feature in extra_features:
            name = str(feature).strip()
            if not name:
                continue
            if name not in _EXTRA_EMBEDDING_FEATURES:
                raise ValueError(
                    "embedding.extra_features must contain only "
                    f"{sorted(_EXTRA_EMBEDDING_FEATURES)}, got {name!r}"
                )
            if name not in normalized_extra_features:
                normalized_extra_features.append(name)
    return {
        "model_name": str(raw.get("model_name", "dinov2_vits14")),
        "device_preference": str(raw.get("device_preference", "auto")).lower(),
        "batch_size": max(1, int(raw.get("batch_size", 32))),
        "image_size": max(32, int(raw.get("image_size", 224))),
        "feature_source": feature_source,
        "intermediate_layer_offset": max(1, int(raw.get("intermediate_layer_offset", 2))),
        "extra_features": normalized_extra_features,
    }


def normalize_projection_config(projection: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(DEFAULT_PROJECTION_CONFIG)
    if projection is not None:
        raw.update(dict(projection))
    return {
        "method": str(raw.get("method", "umap")).lower(),
        "random_state": int(raw.get("random_state", 17)),
        "n_neighbors": max(2, int(raw.get("n_neighbors", 20))),
        "min_dist": float(raw.get("min_dist", 0.10)),
        "perplexity": float(raw.get("perplexity", 30.0)),
    }


def build_thumbnail_generation_config(
    *,
    csv_path: str | Path,
    split: str,
    params: dict[str, Any],
    color_by: str | None,
    individual_id: str | None = None,
    anomaly: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stride = params.get("stride")
    return normalize_thumbnail_generation_config(
        {
            "version": CONFIG_VERSION,
            "data": {
                "csv_path": str(Path(csv_path).resolve()),
                "split": str(split),
                "individual_id": individual_id,
            },
            "thumbnail": {
                "img_size": params.get("img_size"),
                "chunk_size": params.get("chunk_size"),
                "stride": stride,
                "min_chunk_size": params.get("min_chunk_size"),
                "line_width": params.get("line_width"),
                "marker_radius": params.get("marker_radius"),
                "padding": params.get("padding"),
                "scale_mode": params.get("scale_mode"),
                "scale_quantile": params.get("scale_quantile"),
                "color_by": color_by,
            },
            "anomaly": normalize_anomaly_config(anomaly),
        }
    )


def load_thumbnail_generation_config(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, (str, Path)):
        config = json.loads(Path(config_or_path).read_text(encoding="utf-8"))
    else:
        config = dict(config_or_path)
    return normalize_thumbnail_generation_config(config)


def normalize_thumbnail_generation_config(config: dict[str, Any]) -> dict[str, Any]:
    data = dict(config.get("data", {}))
    thumb = dict(config.get("thumbnail", {}))
    anomaly = normalize_anomaly_config(config.get("anomaly"))
    embedding = normalize_embedding_config(config.get("embedding")) if config.get("embedding") is not None else None
    projection = normalize_projection_config(config.get("projection")) if config.get("projection") is not None else None
    stride = thumb.get("stride")
    color_by = thumb.get("color_by")
    if color_by in ("", "none", "None"):
        color_by = None
    normalized = {
        "version": str(config.get("version", CONFIG_VERSION)),
        "data": {
            "csv_path": str(Path(data["csv_path"]).resolve()),
            "split": str(data.get("split") or "all"),
            "individual_id": None if data.get("individual_id") in (None, "", "all", "All individuals") else str(data.get("individual_id")),
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
    }
    if embedding is not None:
        normalized["embedding"] = embedding
    if projection is not None:
        normalized["projection"] = projection
    return normalized


def thumbnail_build_kwargs(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    config = load_thumbnail_generation_config(config_or_path)
    thumb = config["thumbnail"]
    split = config["data"]["split"]
    return {
        "trajectories_path": config["data"]["csv_path"],
        "split": None if split == "all" else split,
        "individual_id": config["data"].get("individual_id"),
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
        "anomaly_config": config["anomaly"],
    }


def dumps_thumbnail_generation_config(config_or_path: dict[str, Any] | str | Path) -> str:
    config = load_thumbnail_generation_config(config_or_path)
    return json.dumps(config, indent=2, sort_keys=True)


def thumbnail_config_filename(config_or_path: dict[str, Any] | str | Path) -> str:
    config = load_thumbnail_generation_config(config_or_path)
    csv_stem = Path(config["data"]["csv_path"]).stem
    split = config["data"]["split"]
    return f"{csv_stem}__{split}__thumbnail_config.json"


def thumbnail_config_run_slug(config_or_path: dict[str, Any] | str | Path) -> str:
    config = load_thumbnail_generation_config(config_or_path)
    digest = sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    csv_stem = Path(config["data"]["csv_path"]).stem
    split = config["data"]["split"]
    return f"{csv_stem}__{split}__{digest}"
