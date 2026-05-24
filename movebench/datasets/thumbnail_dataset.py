"""PyTorch dataset wrapper for pre-rendered track thumbnails.

Kept separate from ``movebench.datasets.thumbnails`` so pure thumbnail
generation/rendering workflows can run without importing torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from movebench.datasets.thumbnails import build_track_thumbnails


class TrackThumbnailDataset(Dataset):
    """GPS movement tracks as MNIST-style thumbnail images."""

    def __init__(
        self,
        trajectories_path: Path | str,
        split: str | None = "train",
        img_size: int = 64,
        chunk_mode: str = "fixed",
        chunk_size: int = 50,
        stride: int | None = None,
        min_chunk_size: int = 10,
        burst_gap_seconds: float = 3600.0,
        line_width: int = 1,
        marker_radius: int = 1,
        padding: int = 2,
        scale_mode: str = "individual",
        scale_quantile: float = 0.5,
        basemap: str | None = None,
        exclude_outlier_cols: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
        transform=None,
    ) -> None:
        self._img_size = img_size
        self._transform = transform
        build = build_track_thumbnails(
            trajectories_path,
            split=split,
            img_size=img_size,
            chunk_mode=chunk_mode,
            chunk_size=chunk_size,
            stride=stride,
            min_chunk_size=min_chunk_size,
            burst_gap_seconds=burst_gap_seconds,
            line_width=line_width,
            marker_radius=marker_radius,
            padding=padding,
            scale_mode=scale_mode,
            scale_quantile=scale_quantile,
            basemap=basemap,
            exclude_outlier_cols=exclude_outlier_cols,
        )
        self._images = build.images
        self._meta = build.metadata
        self._chunk_paths = build.chunk_paths
        self._outlier_cols = build.outlier_cols

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        import torch

        img = self._images[idx]
        tensor = torch.from_numpy(img)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        else:
            raise ValueError(f"Unexpected thumbnail image shape: {tuple(tensor.shape)!r}")
        tensor = tensor.float() / 255.0
        if self._transform is not None:
            tensor = self._transform(tensor)

        row = self._meta.iloc[idx]
        label_cols = self.binary_label_columns() + self.label_columns()
        label_dict = {c: float(row[c]) for c in label_cols}
        return tensor, label_dict

    def numpy(self) -> np.ndarray:
        return self._images

    def labels(self, col: str = "outlier_frac") -> np.ndarray:
        if col not in self._meta.columns:
            available = self.label_columns()
            raise KeyError(f"{col!r} not found. Available: {available}")
        return self._meta[col].to_numpy(dtype=np.float32)

    def metadata(self) -> pd.DataFrame:
        return self._meta.copy()

    def chunk_paths(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        return self._chunk_paths

    def binary_label_columns(self) -> list[str]:
        cols = [c for c in ("clean", "has_any_outlier") if c in self._meta.columns]
        cols += sorted(
            c for c in self._meta.columns if c.startswith("has_") and c != "has_any_outlier"
        )
        return cols

    def label_columns(self) -> list[str]:
        cols = ["outlier_frac"] if "outlier_frac" in self._meta.columns else []
        cols += [c for c in self._meta.columns if c.startswith("frac_")]
        return cols
