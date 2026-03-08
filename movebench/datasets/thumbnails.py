"""MNIST-style thumbnail dataset built from GPS movement tracks.

Each thumbnail is a small B&W image showing a temporally-contiguous chunk of
one individual's trajectory rendered as a connected polyline.

Design decisions
----------------
- **Coordinate normalization**: per-chunk bounding box with equal aspect ratio
  (north-up).  Each image captures movement *shape*, not absolute location.
- **Chunking**: fixed fix-count N (default 50), non-overlapping by default.
  Chunks intentionally cross burst/time-gap boundaries — a gap appears as a
  visual jump in the polyline, which can be diagnostic.
- **Labels**: multi-label, one boolean column per anomaly-type column found in
  trajectories.csv (e.g. ``has_manually_marked_outlier``, ``has_speed_outlier``)
  plus a convenience ``any_outlier`` column.  Per-type fix counts and fractions
  are also stored in metadata for downstream re-thresholding.

Alternative chunking modes (not implemented here, but straightforward to add)
-----------------------------------------------------------------------------
- Fixed time window: group fixes whose timestamp falls within [t, t+window).
- Per-burst: use existing ``burst_id`` column; each burst becomes one image.
  High burst-length variance makes images inconsistently dense.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    SET_COL,
    TIME_COL,
    load_trajectories_csv,
)

_EPS = 1e-9  # guard against zero-range bounding boxes


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def render_track(
    lons: np.ndarray,
    lats: np.ndarray,
    img_size: int = 64,
    padding: int = 2,
    line_width: int = 1,
) -> np.ndarray:
    """Render a GPS track as a B&W polyline image.

    Parameters
    ----------
    lons, lats:
        Parallel float arrays of WGS-84 longitude / latitude values.
    img_size:
        Output image will be ``(img_size, img_size)`` pixels.
    padding:
        Blank border in pixels on each side.
    line_width:
        Polyline stroke width in pixels.

    Returns
    -------
    np.ndarray of shape ``(img_size, img_size)``, dtype ``uint8``.
    White (255) pixels on a black (0) background.
    """
    img = Image.new("L", (img_size, img_size), color=0)

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    # Edge case: fewer than 2 points — just draw a dot at the centre.
    if len(lons) < 2:
        cx = cy = img_size // 2
        draw = ImageDraw.Draw(img)
        draw.point((cx, cy), fill=255)
        return np.array(img, dtype=np.uint8)

    lon_min, lon_max = lons.min(), lons.max()
    lat_min, lat_max = lats.min(), lats.max()
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    drawable = img_size - 2 * padding

    # Edge case: stationary individual — draw a centred dot.
    if lon_range < _EPS and lat_range < _EPS:
        draw = ImageDraw.Draw(img)
        cx = cy = padding + drawable // 2
        draw.point((cx, cy), fill=255)
        return np.array(img, dtype=np.uint8)

    # Equal-aspect scale: fit the longer axis; centre the shorter one.
    scale = drawable / max(lon_range, lat_range)
    x_offset = padding + (drawable - lon_range * scale) / 2.0
    y_offset = padding + (drawable - lat_range * scale) / 2.0

    # Map to pixel coordinates; invert latitude so north is up.
    px = x_offset + (lons - lon_min) * scale
    py = y_offset + (lat_max - lats) * scale  # north-up inversion

    points = list(zip(px.tolist(), py.tolist()))
    draw = ImageDraw.Draw(img)
    draw.line(points, fill=255, width=line_width)

    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_individual(
    df_ind: pd.DataFrame,
    chunk_size: int,
    stride: int | None = None,
) -> list[pd.DataFrame]:
    """Split a single individual's track into fixed-size chunks.

    Parameters
    ----------
    df_ind:
        DataFrame for one individual, already sorted by ``TIME_COL``.
    chunk_size:
        Number of fixes per chunk.
    stride:
        Step size between chunk starts.  ``None`` means non-overlapping
        (``stride == chunk_size``).

    Returns
    -------
    List of DataFrames, one per chunk.  Trailing chunks shorter than
    ``chunk_size`` are included; the caller decides whether to drop them
    via ``min_chunk_size``.
    """
    s = stride if stride is not None else chunk_size
    n = len(df_ind)
    return [df_ind.iloc[start : start + chunk_size] for start in range(0, n, s)]


# ---------------------------------------------------------------------------
# Outlier column detection
# ---------------------------------------------------------------------------

def _detect_outlier_cols(df: pd.DataFrame) -> list[str]:
    """Return all columns whose name contains 'outlier' (case-insensitive)."""
    return [c for c in df.columns if "outlier" in c.lower()]


def _sanitize_col(col: str) -> str:
    """Convert a Movebank column name to a valid Python identifier fragment."""
    return col.replace("-", "_").replace(" ", "_").lower()


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------

def aggregate_labels(
    chunk_df: pd.DataFrame,
    outlier_cols: list[str],
) -> dict:
    """Aggregate fix-level outlier flags to chunk-level multi-label dict.

    For each outlier column ``col`` the following keys are added:
    - ``frac_<col>``: proportion of fixes in the chunk flagged by this type
    - ``n_<col>``: raw count of flagged fixes (for reference)

    Plus summary keys:
    - ``n_outlier_fixes``: fixes flagged by *any* column (OR logic)
    - ``outlier_frac``: ``n_outlier_fixes / len(chunk_df)``

    Parameters
    ----------
    chunk_df:
        DataFrame for one chunk.
    outlier_cols:
        Outlier column names present in ``chunk_df``.

    Returns
    -------
    Flat dict suitable as one row of a metadata DataFrame.
    """
    n = len(chunk_df)
    result: dict = {}
    any_flag = pd.Series([False] * n, index=chunk_df.index)

    for col in outlier_cols:
        is_out = chunk_df[col].fillna(False).astype(bool)
        any_flag = any_flag | is_out
        safe = _sanitize_col(col)
        result[f"n_{safe}"] = int(is_out.sum())
        result[f"frac_{safe}"] = float(is_out.sum()) / n

    result["n_outlier_fixes"] = int(any_flag.sum())
    result["outlier_frac"] = float(any_flag.sum()) / n

    return result


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TrackThumbnailDataset(Dataset):
    """GPS movement tracks as MNIST-style B&W thumbnail images.

    Requires only ``trajectories.csv`` — no raster downloads needed.

    Parameters
    ----------
    trajectories_path:
        Path to a Movebank-format ``trajectories.csv``.
    split:
        Which split to load (``"train"``, ``"test"``, ``None`` for all).
    img_size:
        Square image side length in pixels.
    chunk_size:
        Number of consecutive fixes per thumbnail.
    stride:
        Step between chunk start indices.  ``None`` = non-overlapping.
    min_chunk_size:
        Chunks shorter than this (e.g. trailing fragments) are dropped.
    line_width:
        Polyline stroke width.  Increase for larger ``img_size`` values.
    padding:
        Blank border in pixels on each side.
    transform:
        Optional callable applied to the ``(1, H, W)`` float32 tensor
        in ``__getitem__``.  Accepts torchvision transforms.
    """

    def __init__(
        self,
        trajectories_path: Path | str,
        split: str | None = "train",
        img_size: int = 64,
        chunk_size: int = 50,
        stride: int | None = None,
        min_chunk_size: int = 10,
        line_width: int = 1,
        padding: int = 2,
        transform=None,
    ) -> None:
        self._img_size = img_size
        self._transform = transform

        df = load_trajectories_csv(trajectories_path)
        if split is not None:
            df = df[df[SET_COL] == split].copy()

        outlier_cols = _detect_outlier_cols(df)

        images: list[np.ndarray] = []
        records: list[dict] = []

        for individual_id, df_ind in df.groupby(INDIVIDUAL_COL):
            df_ind = df_ind.sort_values(TIME_COL)
            chunks = chunk_individual(df_ind, chunk_size, stride)
            chunk_idx = 0
            for chunk in chunks:
                if len(chunk) < min_chunk_size:
                    continue
                img = render_track(
                    chunk[LON_COL].to_numpy(),
                    chunk[LAT_COL].to_numpy(),
                    img_size=img_size,
                    padding=padding,
                    line_width=line_width,
                )
                images.append(img)
                label_dict = aggregate_labels(chunk, outlier_cols)
                record = {
                    "individual_id": individual_id,
                    "chunk_idx": chunk_idx,
                    "split": split,
                    "start_time": chunk[TIME_COL].iloc[0],
                    "end_time": chunk[TIME_COL].iloc[-1],
                    "n_fixes": len(chunk),
                    "center_lat": float(chunk[LAT_COL].mean()),
                    "center_lon": float(chunk[LON_COL].mean()),
                    "lat_min": float(chunk[LAT_COL].min()),
                    "lat_max": float(chunk[LAT_COL].max()),
                    "lon_min": float(chunk[LON_COL].min()),
                    "lon_max": float(chunk[LON_COL].max()),
                }
                record.update(label_dict)
                records.append(record)
                chunk_idx += 1

        if images:
            self._images = np.stack(images, axis=0).astype(np.uint8)
        else:
            self._images = np.zeros((0, img_size, img_size), dtype=np.uint8)

        self._meta = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        """Return ``(tensor, label_dict)`` for index ``idx``.

        The tensor has shape ``(1, H, W)``, dtype ``float32``, values in
        ``[0, 1]``.  ``label_dict`` contains ``any_outlier`` plus all
        per-type ``has_*`` keys.
        """
        import torch

        img = self._images[idx]  # (H, W) uint8
        tensor = torch.from_numpy(img).unsqueeze(0).float() / 255.0
        if self._transform is not None:
            tensor = self._transform(tensor)

        row = self._meta.iloc[idx]
        label_cols = self.label_columns()
        label_dict = {c: float(row[c]) for c in label_cols}

        return tensor, label_dict

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Return all images as ``(N, H, W)`` uint8 array."""
        return self._images

    def labels(self, col: str = "outlier_frac") -> np.ndarray:
        """Return a single label column as ``(N,)`` float32 array.

        Parameters
        ----------
        col:
            One of the label columns in ``metadata()`` — e.g.
            ``"outlier_frac"``, ``"frac_speed_outlier"``, etc.
        """
        if col not in self._meta.columns:
            available = self.label_columns()
            raise KeyError(f"{col!r} not found. Available: {available}")
        return self._meta[col].to_numpy(dtype=np.float32)

    def metadata(self) -> pd.DataFrame:
        """Return per-chunk metadata DataFrame."""
        return self._meta.copy()

    def label_columns(self) -> list[str]:
        """Return all label column names (proportion-valued).

        Includes ``outlier_frac`` (any-outlier aggregate) and one
        ``frac_<type>`` column per outlier type found in trajectories.csv.
        """
        cols = ["outlier_frac"] if "outlier_frac" in self._meta.columns else []
        cols += [c for c in self._meta.columns if c.startswith("frac_")]
        return cols
