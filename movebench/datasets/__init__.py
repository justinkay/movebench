"""PyTorch dataset entrypoints built on top of movebench feature stores and rasters."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ChipDataset",
    "DeepSSFDataset",
    "SSFDataset",
    "iSSFDataset",
    "TrackThumbnailBuild",
    "TrackThumbnailDataset",
    "add_iforest_anomaly_score",
    "anomaly_feature_columns",
    "build_track_thumbnails",
    "detect_color_columns",
    "load_thumbnail_source_df",
    "load_thumbnail_source_df_with_anomaly",
]

_ATTR_TO_MODULE = {
    "ChipDataset": ("movebench.datasets.chips", "ChipDataset"),
    "DeepSSFDataset": ("movebench.datasets.deepssf", "DeepSSFDataset"),
    "SSFDataset": ("movebench.datasets.ssf", "SSFDataset"),
    "iSSFDataset": ("movebench.datasets.ssf", "iSSFDataset"),
    "TrackThumbnailBuild": ("movebench.datasets.thumbnails", "TrackThumbnailBuild"),
    "TrackThumbnailDataset": ("movebench.datasets.thumbnail_dataset", "TrackThumbnailDataset"),
    "add_iforest_anomaly_score": ("movebench.datasets.thumbnails", "add_iforest_anomaly_score"),
    "anomaly_feature_columns": ("movebench.datasets.thumbnails", "anomaly_feature_columns"),
    "build_track_thumbnails": ("movebench.datasets.thumbnails", "build_track_thumbnails"),
    "detect_color_columns": ("movebench.datasets.thumbnails", "detect_color_columns"),
    "load_thumbnail_source_df": ("movebench.datasets.thumbnails", "load_thumbnail_source_df"),
    "load_thumbnail_source_df_with_anomaly": ("movebench.datasets.thumbnails", "load_thumbnail_source_df_with_anomaly"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _ATTR_TO_MODULE[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
