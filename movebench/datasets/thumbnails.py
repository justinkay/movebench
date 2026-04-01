"""MNIST-style thumbnail dataset built from GPS movement tracks.

Each thumbnail is a small B&W image showing a temporally-contiguous chunk of
one individual's trajectory rendered as a connected polyline with fix markers.

Design decisions
----------------
- **Coordinate normalization**: by default a per-individual fixed scale derived
  from the median chunk bounding-box extent is used (``scale_mode="individual"``).
  This means a chunk whose bounding box is much larger than the individual's
  typical range will visibly overflow the image, making spatial anomalies
  (e.g. a single GPS fix 500 km away) immediately apparent.
  Pass ``scale_mode="chunk"`` to revert to the original per-chunk bounding-box
  behaviour.
- **Fix markers**: each GPS fix is drawn as a small filled circle on top of the
  polyline so isolated outlier points are visible even at low resolution.
- **Chunking**: fixed fix-count N (default 50), non-overlapping by default.
  Chunks intentionally cross burst/time-gap boundaries — a gap appears as a
  visual jump in the polyline, which can be diagnostic.
- **Labels**: multi-label, one boolean column per anomaly-type column found in
  trajectories.csv (e.g. ``has_manually_marked_outlier``, ``has_speed_outlier``)
  plus summary ``has_any_outlier`` / ``clean`` columns.  Per-type fix counts
  and fractions are also stored in metadata for downstream re-thresholding.

Alternative chunking modes (not implemented here, but straightforward to add)
-----------------------------------------------------------------------------
- Fixed time window: group fixes whose timestamp falls within [t, t+window).
- Per-burst: use existing ``burst_id`` column; each burst becomes one image.
  High burst-length variance makes images inconsistently dense.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.ensemble import IsolationForest

from movebench.movement_math import safe_speed_m_s_numpy
from movebench.thumbnail_config import normalize_anomaly_config
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    SET_COL,
    TIME_COL,
    compute_movement_features,
    load_trajectories_csv,
)

_EPS = 1e-9  # guard against zero-range bounding boxes
_GENERATED_OUTLIER_PREFIXES = ("frac_", "n_", "has_")
_GENERATED_OUTLIER_NAMES = {"outlier_frac", "n_outlier_fixes", "has_any_outlier", "clean"}
_COLOR_MOVEMENT_COLS = ("step_length_m", "step_dt_s", "speed_m_s")
_COLOR_PREFIXES = ("gps:","height-above")
_ANOMALY_MOVEMENT_COLS = ("step_length_m", "step_dt_s", "speed_m_s", "turn_angle_rad")
_ANOMALY_RAW_MOVEMENT_COLS = ("ground-speed", "heading")
_ANOMALY_PREFIXES = ("gps:", "height-above")
_MIN_FINITE_COLOR_VALUES = 2
_COLOR_QUANTILE_RANGE = (0.005, 0.995)
_COLOR_BACKGROUND_RGB = np.array([244, 245, 247], dtype=np.uint8)
_NEUTRAL_RGB = np.array([112, 122, 132], dtype=np.uint8)
_VIRIDIS = colormaps["viridis"]
_OUTLIER_COLUMN_ALIASES: tuple[tuple[str, ...], ...] = (
    ("manually-marked-outliers", "manually-marked-outlier"),
    ("algorithm-marked-outliers", "algorithm-marked-outlier"),
    ("import-marked-outlier",),
    ("speed-outlier",),
    ("system-outlier",),
)


@dataclass(frozen=True)
class TrackThumbnailBuild:
    """Reusable thumbnail build output for apps and datasets."""

    images: np.ndarray
    metadata: pd.DataFrame
    chunk_paths: tuple[tuple[tuple[float, float], ...], ...]
    outlier_cols: tuple[str, ...]

    def proportion_label_columns(self) -> list[str]:
        cols = ["outlier_frac"] if "outlier_frac" in self.metadata.columns else []
        cols += [c for c in self.metadata.columns if c.startswith("frac_")]
        return cols

    def binary_label_columns(self) -> list[str]:
        cols = [c for c in ("clean", "has_any_outlier") if c in self.metadata.columns]
        cols += sorted(
            c for c in self.metadata.columns if c.startswith("has_") and c != "has_any_outlier"
        )
        return cols


# ---------------------------------------------------------------------------
# Shared raw CSV loading
# ---------------------------------------------------------------------------

def load_thumbnail_source_df(
    trajectories_path: Path | str,
    *,
    split: str | None = "train",
    include_movement_features: bool = True,
) -> pd.DataFrame:
    """Load and normalize a trajectories-like CSV for thumbnail workflows."""
    df = load_trajectories_csv(str(trajectories_path))
    df = df.dropna(subset=[LON_COL, LAT_COL]).copy()
    if split is not None and SET_COL in df.columns:
        df = df[df[SET_COL] == split].copy()

    df = df.sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort")
    if not include_movement_features or len(df) == 0:
        return df

    mov = compute_movement_features(df)
    step_dt_s = mov["time_since_last"].dt.total_seconds().to_numpy(dtype=np.float64)
    step_length_m = mov["step_length_m"].to_numpy(dtype=np.float64, copy=False)
    mov["step_dt_s"] = step_dt_s
    mov["speed_m_s"] = safe_speed_m_s_numpy(step_length_m, step_dt_s)
    return mov


def _coerce_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def _has_enough_finite_numeric(series: pd.Series) -> bool:
    values = _coerce_numeric(series)
    return int(np.isfinite(values).sum()) >= _MIN_FINITE_COLOR_VALUES


def _candidate_anomaly_feature_columns(
    df: pd.DataFrame,
    *,
    anomaly_config: dict[str, object] | None = None,
) -> list[str]:
    config = normalize_anomaly_config(anomaly_config)
    requested = config.get("feature_columns")
    if requested is None:
        candidates: list[str] = []
        for col in _ANOMALY_MOVEMENT_COLS + _ANOMALY_RAW_MOVEMENT_COLS:
            if col in df.columns and col not in candidates:
                candidates.append(col)
        for col in df.columns:
            if col in candidates:
                continue
            if any(col.startswith(prefix) for prefix in _ANOMALY_PREFIXES):
                candidates.append(col)
        return candidates
    return [str(col) for col in requested if str(col) in df.columns]


def anomaly_feature_columns(
    df: pd.DataFrame,
    *,
    anomaly_config: dict[str, object] | None = None,
) -> list[str]:
    cols: list[str] = []
    for col in _candidate_anomaly_feature_columns(df, anomaly_config=anomaly_config):
        values = _coerce_numeric(df[col])
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            continue
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std <= 0.0:
            continue
        cols.append(col)
    return cols


def add_iforest_anomaly_score(
    df: pd.DataFrame,
    *,
    anomaly_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    config = normalize_anomaly_config(anomaly_config)
    score_column = str(config["score_column"])
    feature_columns = anomaly_feature_columns(df, anomaly_config=config)
    if len(df) == 0 or not feature_columns:
        return df.copy()

    blocks: list[np.ndarray] = []
    for col in feature_columns:
        values = _coerce_numeric(df[col])
        finite = np.isfinite(values)
        if int(finite.sum()) < 2:
            continue
        std = float(np.nanstd(values))
        if not np.isfinite(std) or std <= 0.0:
            continue
        median = float(np.nanmedian(values))
        if not np.isfinite(median):
            continue
        mean = float(np.nanmean(values))
        if not np.isfinite(mean):
            continue
        filled = values.copy()
        filled[~finite] = median
        scaled = (filled - mean) / std
        if not np.isfinite(scaled).all():
            continue
        blocks.append(scaled.astype(np.float64, copy=False))

    if not blocks:
        return df.copy()

    feature_matrix = np.stack(blocks, axis=1)
    model = IsolationForest(
        n_estimators=int(config["n_estimators"]),
        contamination=config["contamination"],
        random_state=int(config["random_state"]),
    )
    scores = -model.fit(feature_matrix).score_samples(feature_matrix)
    out = df.copy()
    out[score_column] = np.asarray(scores, dtype=np.float64)
    return out


def load_thumbnail_source_df_with_anomaly(
    trajectories_path: Path | str,
    *,
    split: str | None = "train",
    include_movement_features: bool = True,
    anomaly_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    config = normalize_anomaly_config(anomaly_config)
    full_df = load_thumbnail_source_df(
        trajectories_path,
        split=None,
        include_movement_features=include_movement_features,
    )
    if include_movement_features:
        full_df = add_iforest_anomaly_score(full_df, anomaly_config=config)
    if split is not None and SET_COL in full_df.columns:
        return full_df[full_df[SET_COL] == split].copy()
    return full_df


def detect_color_columns(
    df: pd.DataFrame,
    *,
    extra_columns: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return numeric movement/GPS columns suitable for per-fix coloring."""
    cols: list[str] = []
    for col in _COLOR_MOVEMENT_COLS:
        if col in df.columns and _has_enough_finite_numeric(df[col]):
            cols.append(col)
    for col in extra_columns or ():
        if col in cols:
            continue
        if col in df.columns and _has_enough_finite_numeric(df[col]):
            cols.append(str(col))
    for col in df.columns:
        if col in cols:
            continue
        if not any(col.startswith(prefix) for prefix in _COLOR_PREFIXES):
            continue
        if _has_enough_finite_numeric(df[col]):
            cols.append(col)
    return cols


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _project_track_pixels(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    img_size: int,
    padding: int,
    fixed_range: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    drawable = img_size - 2 * padding

    if fixed_range is not None:
        cx = float(np.median(lons))
        cy = float(np.median(lats))
        lon_range = lat_range = fixed_range
        lon_min = cx - fixed_range / 2.0
        lat_max = cy + fixed_range / 2.0
    else:
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

    scale = drawable / max(lon_range, lat_range)
    x_offset = padding + (drawable - lon_range * scale) / 2.0
    y_offset = padding + (drawable - lat_range * scale) / 2.0

    px = x_offset + (lons - lon_min) * scale
    py = y_offset + (lat_max - lats) * scale
    return px, py


def _fit_color_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 0.0, 1.0

    lo = float(np.quantile(finite, _COLOR_QUANTILE_RANGE[0]))
    hi = float(np.quantile(finite, _COLOR_QUANTILE_RANGE[1]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if hi <= lo:
        center = float(finite[0])
        return center - 0.5, center + 0.5
    return lo, hi


def _values_to_rgb(values: np.ndarray, *, color_range: tuple[float, float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    rgb = np.repeat(_NEUTRAL_RGB[None, :], len(arr), axis=0)
    finite = np.isfinite(arr)
    if not finite.any():
        return rgb

    lo, hi = color_range
    scale = max(hi - lo, _EPS)
    normalized = np.clip((arr[finite] - lo) / scale, 0.0, 1.0)
    rgb[finite] = _VIRIDIS(normalized, bytes=True)[:, :3]
    return rgb.astype(np.uint8, copy=False)

def render_track(
    lons: np.ndarray,
    lats: np.ndarray,
    img_size: int = 64,
    padding: int = 2,
    line_width: int = 1,
    marker_radius: int = 1,
    fixed_range: float | None = None,
) -> np.ndarray:
    """Render a GPS track as a B&W polyline image with fix markers.

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
    marker_radius:
        Radius in pixels of the filled circle drawn at each fix.
        Set to 0 to draw only the polyline.
    fixed_range:
        If given, use this value (in degrees) as both the lon and lat
        extent of the rendered area, centred on the chunk median.  Points
        that fall outside are still drawn but will be clipped to the image
        boundary, making large spatial excursions visually obvious.
        If ``None``, the extent is computed from the chunk bounding box.

    Returns
    -------
    np.ndarray of shape ``(img_size, img_size)``, dtype ``uint8``.
    White (255) pixels on a black (0) background.
    """
    img = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(img)

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    drawable = img_size - 2 * padding

    # Edge case: fewer than 2 points — just draw a dot at the centre.
    if len(lons) < 2:
        cx = cy = img_size // 2
        draw.point((cx, cy), fill=255)
        return np.array(img, dtype=np.uint8)

    if fixed_range is not None:
        # Centre on median; use fixed_range as the spatial extent.
        cx = float(np.median(lons))
        cy = float(np.median(lats))
        lon_range = lat_range = fixed_range
        lon_min = cx - fixed_range / 2.0
        lat_max = cy + fixed_range / 2.0
    else:
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        # Edge case: stationary individual — draw a centred dot.
        if lon_range < _EPS and lat_range < _EPS:
            cp = padding + drawable // 2
            draw.point((cp, cp), fill=255)
            return np.array(img, dtype=np.uint8)

    px, py = _project_track_pixels(
        lons,
        lats,
        img_size=img_size,
        padding=padding,
        fixed_range=fixed_range,
    )

    # Polyline
    draw.line(list(zip(px.tolist(), py.tolist())), fill=255, width=line_width)

    # Fix markers
    if marker_radius > 0:
        r = marker_radius
        for x, y in zip(px.tolist(), py.tolist()):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    return np.array(img, dtype=np.uint8)


def render_track_color(
    lons: np.ndarray,
    lats: np.ndarray,
    values: np.ndarray,
    *,
    color_range: tuple[float, float],
    img_size: int = 64,
    padding: int = 2,
    line_width: int = 1,
    marker_radius: int = 1,
    fixed_range: float | None = None,
) -> np.ndarray:
    """Render a GPS track with per-fix/per-segment colors on a light neutral background."""
    img = Image.new("RGB", (img_size, img_size), color=tuple(_COLOR_BACKGROUND_RGB.tolist()))
    draw = ImageDraw.Draw(img)

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    values = np.asarray(values, dtype=np.float64)

    if len(values) != len(lons):
        raise ValueError("values must align with lons/lats")

    point_rgb = _values_to_rgb(values, color_range=color_range)

    if len(lons) < 2:
        cx = cy = img_size // 2
        fill = tuple(point_rgb[0].tolist()) if len(point_rgb) else tuple(_NEUTRAL_RGB.tolist())
        if marker_radius > 0:
            r = marker_radius
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        else:
            draw.point((cx, cy), fill=fill)
        return np.array(img, dtype=np.uint8)

    if fixed_range is None:
        lon_range = float(lons.max() - lons.min())
        lat_range = float(lats.max() - lats.min())
        if lon_range < _EPS and lat_range < _EPS:
            cp = padding + (img_size - 2 * padding) // 2
            finite_idx = np.flatnonzero(np.isfinite(values))
            fill = (
                tuple(point_rgb[int(finite_idx[0])].tolist())
                if len(finite_idx)
                else tuple(_NEUTRAL_RGB.tolist())
            )
            if marker_radius > 0:
                r = marker_radius
                draw.ellipse([cp - r, cp - r, cp + r, cp + r], fill=fill)
            else:
                draw.point((cp, cp), fill=fill)
            return np.array(img, dtype=np.uint8)

    px, py = _project_track_pixels(
        lons,
        lats,
        img_size=img_size,
        padding=padding,
        fixed_range=fixed_range,
    )

    if len(values) > 1:
        seg_values = np.full(len(values) - 1, np.nan, dtype=np.float64)
        left = values[:-1]
        right = values[1:]
        both = np.isfinite(left) & np.isfinite(right)
        seg_values[both] = 0.5 * (left[both] + right[both])
        only_left = np.isfinite(left) & ~np.isfinite(right)
        seg_values[only_left] = left[only_left]
        only_right = ~np.isfinite(left) & np.isfinite(right)
        seg_values[only_right] = right[only_right]
        seg_rgb = _values_to_rgb(seg_values, color_range=color_range)
        for idx in range(len(seg_values)):
            draw.line(
                [(float(px[idx]), float(py[idx])), (float(px[idx + 1]), float(py[idx + 1]))],
                fill=tuple(seg_rgb[idx].tolist()),
                width=line_width,
            )

    if marker_radius > 0:
        r = marker_radius
        for idx, (x, y) in enumerate(zip(px.tolist(), py.tolist())):
            draw.ellipse(
                [x - r, y - r, x + r, y + r],
                fill=tuple(point_rgb[idx].tolist()),
            )

    return np.array(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Per-individual scale computation
# ---------------------------------------------------------------------------

def compute_individual_scales(
    df: pd.DataFrame,
    chunk_size: int,
    stride: int | None = None,
    min_chunk_size: int = 10,
    quantile: float = 0.5,
) -> dict[str, float]:
    """Return a per-individual fixed rendering range (degrees).

    The range is the ``quantile``-th percentile of the per-chunk bounding-box
    extents across all valid chunks for that individual.  Using the median
    (default) makes the scale robust to the very outlier chunks we want to
    render as visually anomalous.

    Parameters
    ----------
    df:
        Full trajectories DataFrame (all individuals).
    chunk_size, stride, min_chunk_size:
        Same chunking parameters used when building the dataset.
    quantile:
        Which quantile of chunk extents to use as the fixed scale (0–1).
        0.5 = median.

    Returns
    -------
    Dict mapping individual ID → scale in degrees.
    """
    scales: dict[str, float] = {}
    for ind_id, df_ind in df.groupby(INDIVIDUAL_COL):
        df_ind = df_ind.sort_values(TIME_COL)
        chunks = chunk_individual(df_ind, chunk_size, stride)
        extents = []
        for ch in chunks:
            if len(ch) < min_chunk_size:
                continue
            lon_ext = float(ch[LON_COL].max() - ch[LON_COL].min())
            lat_ext = float(ch[LAT_COL].max() - ch[LAT_COL].min())
            extents.append(max(lon_ext, lat_ext))
        if extents:
            scales[str(ind_id)] = float(np.quantile(extents, quantile))
    return scales


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
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size!r}")
    s = int(stride) if stride is not None else int(chunk_size)
    if s <= 0:
        s = int(chunk_size)
    n = len(df_ind)
    return [df_ind.iloc[start : start + chunk_size] for start in range(0, n, s)]


# ---------------------------------------------------------------------------
# Outlier column detection
# ---------------------------------------------------------------------------

def detect_outlier_cols(
    df: pd.DataFrame,
    *,
    exclude: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Return whitelisted raw outlier columns present in a trajectories-like DataFrame."""
    excluded = frozenset(() if exclude is None else exclude)
    cols: list[str] = []
    present = set(df.columns)
    for alias_group in _OUTLIER_COLUMN_ALIASES:
        for col in alias_group:
            if col in excluded or col not in present:
                continue
            if col in _GENERATED_OUTLIER_NAMES:
                continue
            if any(col.startswith(prefix) for prefix in _GENERATED_OUTLIER_PREFIXES):
                continue
            cols.append(col)
            break
    return cols


def _sanitize_col(col: str) -> str:
    """Convert a Movebank column name to a valid Python identifier fragment."""
    return col.replace("-", "_").replace(" ", "_").lower()


def _coerce_outlier_mask(series: pd.Series) -> np.ndarray:
    """Coerce mixed-type outlier columns to a dense boolean mask."""
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
    if n <= 0:
        raise ValueError("chunk_df must contain at least one row")
    result: dict = {}
    any_flag = np.zeros(n, dtype=bool)

    for col in outlier_cols:
        is_out = _coerce_outlier_mask(chunk_df[col])
        any_flag |= is_out
        count = int(np.count_nonzero(is_out))
        safe = _sanitize_col(col)
        result[f"has_{safe}"] = bool(count > 0)
        result[f"n_{safe}"] = count
        result[f"frac_{safe}"] = float(count) / float(n)

    any_count = int(np.count_nonzero(any_flag))
    result["n_outlier_fixes"] = any_count
    result["has_any_outlier"] = bool(any_count > 0)
    result["outlier_frac"] = float(any_count) / float(n)
    result["clean"] = not result["has_any_outlier"]

    return result


def _validate_label_metadata(metadata: pd.DataFrame) -> None:
    if len(metadata) == 0 or "outlier_frac" not in metadata.columns:
        return

    outlier_frac = metadata["outlier_frac"].to_numpy(dtype=np.float64, copy=False)
    n_outlier_fixes = metadata["n_outlier_fixes"].to_numpy(dtype=np.float64, copy=False)
    n_fixes = metadata["n_fixes"].to_numpy(dtype=np.float64, copy=False)
    valid = np.isfinite(outlier_frac) & np.isfinite(n_outlier_fixes) & np.isfinite(n_fixes) & (n_fixes > 0)
    invalid = ~valid | (outlier_frac < 0.0) | (outlier_frac > 1.0) | (n_outlier_fixes < 0.0) | (n_outlier_fixes > n_fixes)
    if not invalid.any():
        return

    sample = metadata.loc[invalid, ["individual_id", "chunk_idx", "outlier_frac", "n_outlier_fixes", "n_fixes"]].head(5)
    raise ValueError(
        "Invalid thumbnail label statistics detected; expected 0 <= outlier_frac <= 1 and "
        f"0 <= n_outlier_fixes <= n_fixes. Sample rows:\n{sample.to_string(index=False)}"
    )


def _iter_thumbnail_groups(
    df: pd.DataFrame,
    *,
    split: str | None,
):
    if split is None and SET_COL in df.columns:
        grouped = df.groupby([INDIVIDUAL_COL, SET_COL], sort=False)
        for (individual_id, split_value), df_group in grouped:
            yield str(individual_id), str(split_value), df_group.sort_values(TIME_COL)
        return

    grouped = df.groupby(INDIVIDUAL_COL, sort=False)
    for individual_id, df_group in grouped:
        if split is not None:
            split_value = split
        elif SET_COL in df_group.columns and not df_group[SET_COL].empty:
            split_value = str(df_group[SET_COL].iloc[0])
        else:
            split_value = None
        yield str(individual_id), split_value, df_group.sort_values(TIME_COL)


def build_track_thumbnails(
    trajectories_path: Path | str,
    *,
    split: str | None = "train",
    img_size: int = 64,
    chunk_size: int = 50,
    stride: int | None = None,
    min_chunk_size: int = 10,
    line_width: int = 1,
    marker_radius: int = 1,
    padding: int = 2,
    scale_mode: str = "individual",
    scale_quantile: float = 0.5,
    color_by: str | None = None,
    anomaly_config: dict[str, object] | None = None,
    exclude_outlier_cols: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> TrackThumbnailBuild:
    """Build thumbnail images, metadata, and chunk paths for one trajectories CSV/split."""
    if scale_mode not in ("individual", "chunk"):
        raise ValueError(f"scale_mode must be 'individual' or 'chunk', got {scale_mode!r}")

    normalized_anomaly = normalize_anomaly_config(anomaly_config)
    anomaly_score_column = str(normalized_anomaly["score_column"])
    if color_by == anomaly_score_column:
        df = load_thumbnail_source_df_with_anomaly(
            trajectories_path,
            split=split,
            include_movement_features=True,
            anomaly_config=normalized_anomaly,
        )
    else:
        df = load_thumbnail_source_df(trajectories_path, split=split, include_movement_features=True)
    outlier_cols = detect_outlier_cols(df, exclude=exclude_outlier_cols)
    color_range: tuple[float, float] | None = None
    if color_by is not None:
        color_cols = detect_color_columns(df, extra_columns=[anomaly_score_column])
        if color_by not in color_cols:
            raise KeyError(f"{color_by!r} is not an available color column. Available: {color_cols}")
        color_range = _fit_color_range(_coerce_numeric(df[color_by]))

    if scale_mode == "individual":
        ind_scales = compute_individual_scales(
            df, chunk_size, stride, min_chunk_size, scale_quantile
        )
    else:
        ind_scales = {}

    images: list[np.ndarray] = []
    chunk_paths: list[tuple[tuple[float, float], ...]] = []
    records: list[dict] = []

    for individual_id, chunk_split, df_group in _iter_thumbnail_groups(df, split=split):
        chunks = chunk_individual(df_group, chunk_size, stride)
        fixed_range = ind_scales.get(individual_id)
        chunk_idx = 0
        for chunk in chunks:
            if len(chunk) < min_chunk_size:
                continue

            lons = chunk[LON_COL].to_numpy(dtype=float, copy=False)
            lats = chunk[LAT_COL].to_numpy(dtype=float, copy=False)
            if color_by is None:
                img = render_track(
                    lons,
                    lats,
                    img_size=img_size,
                    padding=padding,
                    line_width=line_width,
                    marker_radius=marker_radius,
                    fixed_range=fixed_range,
                )
            else:
                img = render_track_color(
                    lons,
                    lats,
                    _coerce_numeric(chunk[color_by]),
                    color_range=color_range if color_range is not None else (0.0, 1.0),
                    img_size=img_size,
                    padding=padding,
                    line_width=line_width,
                    marker_radius=marker_radius,
                    fixed_range=fixed_range,
                )
            images.append(img)
            chunk_paths.append(tuple((float(lon), float(lat)) for lon, lat in zip(lons.tolist(), lats.tolist())))

            label_dict = aggregate_labels(chunk, outlier_cols)
            record = {
                "global_chunk_idx": len(records),
                "individual_id": individual_id,
                "chunk_idx": chunk_idx,
                "split": chunk_split,
                "start_time": chunk[TIME_COL].iloc[0],
                "end_time": chunk[TIME_COL].iloc[-1],
                "n_fixes": len(chunk),
                "start_lon": float(chunk[LON_COL].iloc[0]),
                "start_lat": float(chunk[LAT_COL].iloc[0]),
                "end_lon": float(chunk[LON_COL].iloc[-1]),
                "end_lat": float(chunk[LAT_COL].iloc[-1]),
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
        image_arr = np.stack(images, axis=0).astype(np.uint8)
    else:
        if color_by is None:
            image_arr = np.zeros((0, img_size, img_size), dtype=np.uint8)
        else:
            image_arr = np.zeros((0, img_size, img_size, 3), dtype=np.uint8)

    metadata = pd.DataFrame(records)
    _validate_label_metadata(metadata)
    return TrackThumbnailBuild(
        images=image_arr,
        metadata=metadata,
        chunk_paths=tuple(chunk_paths),
        outlier_cols=tuple(outlier_cols),
    )


def __getattr__(name: str):
    if name != "TrackThumbnailDataset":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("movebench.datasets.thumbnail_dataset")
    value = getattr(module, name)
    globals()[name] = value
    return value
