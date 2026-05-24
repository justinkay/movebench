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
  Pass ``chunk_mode="burst"`` to split each individual/split track whenever
  the time gap between consecutive fixes is greater than ``burst_gap_seconds``.
- **Labels**: multi-label, one boolean column per anomaly-type column found in
  trajectories.csv (e.g. ``has_manually_marked_outlier``, ``has_speed_outlier``)
  plus summary ``has_any_outlier`` / ``clean`` columns.  Per-type fix counts
  and fractions are also stored in metadata for downstream re-thresholding.

Alternative chunking modes (not implemented here, but straightforward to add)
-----------------------------------------------------------------------------
- Fixed time window: group fixes whose timestamp falls within [t, t+window).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path

from matplotlib import colormaps
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import requests
from sklearn.ensemble import IsolationForest

from movebench.movement_math import safe_speed_m_s_numpy
from movebench.thumbnail_basemaps import normalize_thumbnail_basemap, thumbnail_basemap_spec
from movebench.thumbnail_config import (
    DEFAULT_BURST_GAP_MODE,
    DEFAULT_BURST_GAP_QUANTILE,
    normalize_anomaly_config,
    normalize_burst_gap_mode,
    normalize_burst_gap_quantile,
)
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
_COLOR_OUTLIER_ALIASES: tuple[tuple[str, ...], ...] = (
    ("algorithm-marked-outliers", "algorithm-marked-outlier"),
    ("manually-marked-outliers", "manually-marked-outlier"),
)
_COLOR_PREFIXES = ("gps:", "height-above")
_ANOMALY_MOVEMENT_COLS = ("step_length_m", "step_dt_s", "speed_m_s", "turn_angle_rad")
_ANOMALY_RAW_MOVEMENT_COLS = ("ground-speed", "heading")
_ANOMALY_PREFIXES = ("gps:", "height-above")
_MIN_FINITE_COLOR_VALUES = 2
_COLOR_QUANTILE_RANGE = (0.005, 0.995)
_COLOR_BACKGROUND_RGB = np.array([244, 245, 247], dtype=np.uint8)
_NEUTRAL_RGB = np.array([112, 122, 132], dtype=np.uint8)
_TRACK_RGB = np.array([27, 36, 48], dtype=np.uint8)
_TRACK_HALO_RGB = np.array([255, 255, 255], dtype=np.uint8)
_BINARY_FALSE_RGB = np.array([112, 122, 132], dtype=np.uint8)
_BINARY_TRUE_RGB = np.array([231, 76, 60], dtype=np.uint8)
_VIRIDIS = colormaps["viridis"]
_MAX_MERCATOR_LAT = 85.05112878
_TILE_FETCH_TIMEOUT_S = 20
_THUMBNAIL_BASEMAP_MAX_ZOOM = 12
DEFAULT_CHUNK_MODE = "fixed"
DEFAULT_BURST_GAP_SECONDS = 3600.0
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
    burst_gap: dict[str, object]

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
    individual_id: str | None = None,
    include_movement_features: bool = True,
) -> pd.DataFrame:
    """Load and normalize a trajectories-like CSV for thumbnail workflows."""
    df = load_trajectories_csv(str(trajectories_path))
    df = df.dropna(subset=[LON_COL, LAT_COL]).copy()
    if split is not None and SET_COL in df.columns:
        df = df[df[SET_COL] == split].copy()
    if individual_id not in (None, ""):
        df = df[df[INDIVIDUAL_COL].astype(str) == str(individual_id)].copy()

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
    individual_id: str | None = None,
    include_movement_features: bool = True,
    anomaly_config: dict[str, object] | None = None,
) -> pd.DataFrame:
    config = normalize_anomaly_config(anomaly_config)
    full_df = load_thumbnail_source_df(
        trajectories_path,
        split=None,
        individual_id=individual_id,
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
    for alias_group in _COLOR_OUTLIER_ALIASES:
        for col in alias_group:
            if col in cols:
                continue
            if col in df.columns and _has_enough_finite_numeric(df[col]):
                cols.append(col)
                break
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


def _is_binary_color_column_name(col: str | None) -> bool:
    if col in (None, ""):
        return False
    name = str(col)
    return any(name in alias_group for alias_group in _COLOR_OUTLIER_ALIASES)


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def _render_frame(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    img_size: int,
    padding: int,
    fixed_range: float | None,
) -> dict[str, float] | None:
    drawable = img_size - 2 * padding
    if drawable <= 0:
        raise ValueError(f"padding={padding!r} leaves no drawable pixels for img_size={img_size!r}")

    if fixed_range is not None:
        frame_range = float(fixed_range)
        if not np.isfinite(frame_range) or frame_range <= 0.0:
            raise ValueError(f"fixed_range must be positive and finite, got {fixed_range!r}")
        center_lon = float(np.median(lons))
        center_lat = float(np.median(lats))
    else:
        lon_min = float(lons.min())
        lon_max = float(lons.max())
        lat_min = float(lats.min())
        lat_max = float(lats.max())
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        if lon_range < _EPS and lat_range < _EPS:
            return None
        center_lon = 0.5 * (lon_min + lon_max)
        center_lat = 0.5 * (lat_min + lat_max)
        frame_range = max(lon_range, lat_range)

    return {
        "drawable": float(drawable),
        "scale": float(drawable) / frame_range,
        "full_lon_min": center_lon - frame_range / 2.0,
        "full_lon_max": center_lon + frame_range / 2.0,
        "full_lat_min": center_lat - frame_range / 2.0,
        "full_lat_max": center_lat + frame_range / 2.0,
    }


def _project_track_pixels(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    img_size: int,
    padding: int,
    fixed_range: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    frame = _render_frame(
        lons,
        lats,
        img_size=img_size,
        padding=padding,
        fixed_range=fixed_range,
    )
    if frame is None:
        raise ValueError("Cannot project a stationary chunk without a rendering frame")

    scale = float(frame["scale"])
    px = padding + (lons - float(frame["full_lon_min"])) * scale
    py = padding + (float(frame["full_lat_max"]) - lats) * scale
    return px, py


def _mercator_x_from_lon(lon: np.ndarray) -> np.ndarray:
    return (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0


def _mercator_y_from_lat(lat: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(lat, dtype=np.float64), -_MAX_MERCATOR_LAT, _MAX_MERCATOR_LAT)
    radians = np.deg2rad(clipped)
    return (1.0 - np.arcsinh(np.tan(radians)) / np.pi) / 2.0


def _choose_basemap_zoom(
    *,
    frame: dict[str, float],
    target_pixels: int,
    tile_size: int,
) -> int:
    x0 = float(_mercator_x_from_lon(np.array([frame["full_lon_min"]], dtype=np.float64))[0])
    x1 = float(_mercator_x_from_lon(np.array([frame["full_lon_max"]], dtype=np.float64))[0])
    y0 = float(_mercator_y_from_lat(np.array([frame["full_lat_max"]], dtype=np.float64))[0])
    y1 = float(_mercator_y_from_lat(np.array([frame["full_lat_min"]], dtype=np.float64))[0])
    span = max(abs(x1 - x0), abs(y1 - y0), 1e-9)
    desired_world_pixels = max(int(target_pixels), 1) * 2.0
    zoom = math.ceil(math.log2(desired_world_pixels / (float(tile_size) * span)))
    return int(np.clip(zoom, 0, _THUMBNAIL_BASEMAP_MAX_ZOOM))


@lru_cache(maxsize=1)
def _tile_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "movebench-thumbnail-renderer/1.0"})
    return session


@lru_cache(maxsize=4096)
def _fetch_basemap_tile(basemap: str, z: int, x: int, y: int) -> np.ndarray:
    spec = thumbnail_basemap_spec(basemap)
    if spec.tile_url is None:
        raise ValueError(f"Basemap {basemap!r} does not define raster tiles")
    url = spec.tile_url.format(z=int(z), x=int(x), y=int(y))
    try:
        response = _tile_http_session().get(
            url,
            timeout=_TILE_FETCH_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to fetch basemap tile for {basemap!r} at z={z}, x={x}, y={y}: {exc}"
        ) from exc
    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to decode basemap tile for {basemap!r} at z={z}, x={x}, y={y}"
        ) from exc
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(
            f"Unexpected basemap tile shape for {basemap!r} at z={z}, x={x}, y={y}: {arr.shape!r}"
        )
    if arr.shape[0] != spec.tile_size or arr.shape[1] != spec.tile_size:
        image = image.resize((spec.tile_size, spec.tile_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.uint8)
    return arr


def _render_basemap_background(
    basemap: str,
    *,
    frame: dict[str, float] | None,
    img_size: int,
    padding: int,
) -> np.ndarray:
    out = np.broadcast_to(_COLOR_BACKGROUND_RGB, (img_size, img_size, 3)).copy()
    if frame is None:
        return out

    spec = thumbnail_basemap_spec(basemap)
    if spec.tile_url is None:
        return out

    drawable = int(frame["drawable"])
    zoom = _choose_basemap_zoom(frame=frame, target_pixels=drawable, tile_size=spec.tile_size)

    sample_lon = (
        float(frame["full_lon_min"])
        + ((np.arange(drawable, dtype=np.float64) + 0.5) / float(drawable))
        * (float(frame["full_lon_max"]) - float(frame["full_lon_min"]))
    )
    sample_lat = (
        float(frame["full_lat_max"])
        - ((np.arange(drawable, dtype=np.float64) + 0.5) / float(drawable))
        * (float(frame["full_lat_max"]) - float(frame["full_lat_min"]))
    )
    sample_x = _mercator_x_from_lon(np.clip(sample_lon, -179.999999, 179.999999))
    sample_y = _mercator_y_from_lat(sample_lat)

    world_pixels = spec.tile_size * (2**zoom)
    world_x = sample_x * float(world_pixels)
    world_y = sample_y * float(world_pixels)

    min_tile_x = int(np.floor(world_x.min() / float(spec.tile_size)))
    max_tile_x = int(np.floor(world_x.max() / float(spec.tile_size)))
    min_tile_y = int(np.floor(world_y.min() / float(spec.tile_size)))
    max_tile_y = int(np.floor(world_y.max() / float(spec.tile_size)))
    max_tile_index = 2**zoom - 1

    n_tiles_x = max_tile_x - min_tile_x + 1
    n_tiles_y = max_tile_y - min_tile_y + 1
    mosaic = np.empty((n_tiles_y * spec.tile_size, n_tiles_x * spec.tile_size, 3), dtype=np.uint8)
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile = _fetch_basemap_tile(
                basemap,
                zoom,
                int(np.clip(tile_x, 0, max_tile_index)),
                int(np.clip(tile_y, 0, max_tile_index)),
            )
            y0 = (tile_y - min_tile_y) * spec.tile_size
            x0 = (tile_x - min_tile_x) * spec.tile_size
            mosaic[y0 : y0 + spec.tile_size, x0 : x0 + spec.tile_size] = tile

    sample_ix = np.clip(
        np.rint(world_x - float(min_tile_x * spec.tile_size)).astype(np.int64),
        0,
        mosaic.shape[1] - 1,
    )
    sample_iy = np.clip(
        np.rint(world_y - float(min_tile_y * spec.tile_size)).astype(np.int64),
        0,
        mosaic.shape[0] - 1,
    )
    sampled = mosaic[sample_iy[:, None], sample_ix[None, :]]
    out[padding : padding + drawable, padding : padding + drawable] = sampled
    return out


def _frame_cache_key(
    frame: dict[str, float] | None,
    *,
    img_size: int,
    padding: int,
) -> tuple[float, ...] | None:
    if frame is None:
        return None
    return (
        float(img_size),
        float(padding),
        round(float(frame["drawable"]), 6),
        round(float(frame["full_lon_min"]), 6),
        round(float(frame["full_lon_max"]), 6),
        round(float(frame["full_lat_min"]), 6),
        round(float(frame["full_lat_max"]), 6),
    )


@lru_cache(maxsize=2048)
def _render_basemap_background_cached(
    basemap: str,
    frame_key: tuple[float, ...] | None,
) -> np.ndarray:
    if frame_key is None:
        raise ValueError("frame_key must not be None")
    img_size = int(frame_key[0])
    padding = int(frame_key[1])
    frame = {
        "drawable": float(frame_key[2]),
        "full_lon_min": float(frame_key[3]),
        "full_lon_max": float(frame_key[4]),
        "full_lat_min": float(frame_key[5]),
        "full_lat_max": float(frame_key[6]),
    }
    return _render_basemap_background(
        basemap,
        frame=frame,
        img_size=img_size,
        padding=padding,
    )


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


def _binary_values_to_rgb(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    truthy = np.isfinite(arr) & (arr > 0.0)
    rgb = np.repeat(_BINARY_FALSE_RGB[None, :], len(arr), axis=0)
    rgb[truthy] = _BINARY_TRUE_RGB
    return rgb.astype(np.uint8, copy=False)


def _segment_value_colors(values: np.ndarray, *, color_range: tuple[float, float]) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros((0, 3), dtype=np.uint8)
    seg_values = np.full(len(values) - 1, np.nan, dtype=np.float64)
    left = values[:-1]
    right = values[1:]
    both = np.isfinite(left) & np.isfinite(right)
    seg_values[both] = 0.5 * (left[both] + right[both])
    only_left = np.isfinite(left) & ~np.isfinite(right)
    seg_values[only_left] = left[only_left]
    only_right = ~np.isfinite(left) & np.isfinite(right)
    seg_values[only_right] = right[only_right]
    return _values_to_rgb(seg_values, color_range=color_range)


def _binary_segment_colors(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) <= 1:
        return np.zeros((0, 3), dtype=np.uint8)
    left = np.isfinite(arr[:-1]) & (arr[:-1] > 0.0)
    right = np.isfinite(arr[1:]) & (arr[1:] > 0.0)
    truthy = left | right
    rgb = np.repeat(_BINARY_FALSE_RGB[None, :], len(arr) - 1, axis=0)
    rgb[truthy] = _BINARY_TRUE_RGB
    return rgb.astype(np.uint8, copy=False)


def _draw_track_on_rgb_background(
    background: np.ndarray,
    *,
    lons: np.ndarray,
    lats: np.ndarray,
    point_rgb: np.ndarray,
    segment_rgb: np.ndarray,
    img_size: int,
    padding: int,
    line_width: int,
    marker_radius: int,
    fixed_range: float | None,
) -> np.ndarray:
    img = Image.fromarray(np.asarray(background, dtype=np.uint8), mode="RGB")
    draw = ImageDraw.Draw(img)
    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)

    if len(lons) < 2:
        cx = cy = img_size // 2
        fill = tuple(point_rgb[0].tolist()) if len(point_rgb) else tuple(_TRACK_RGB.tolist())
        halo = tuple(_TRACK_HALO_RGB.tolist())
        if marker_radius > 0:
            r_halo = marker_radius + 1
            draw.ellipse([cx - r_halo, cy - r_halo, cx + r_halo, cy + r_halo], fill=halo)
            r = marker_radius
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
        else:
            draw.point((cx, cy), fill=fill)
        return np.asarray(img, dtype=np.uint8)

    frame = _render_frame(
        lons,
        lats,
        img_size=img_size,
        padding=padding,
        fixed_range=fixed_range,
    )
    if frame is None:
        cp = padding + (img_size - 2 * padding) // 2
        fill = tuple(point_rgb[0].tolist()) if len(point_rgb) else tuple(_TRACK_RGB.tolist())
        halo = tuple(_TRACK_HALO_RGB.tolist())
        if marker_radius > 0:
            r_halo = marker_radius + 1
            draw.ellipse([cp - r_halo, cp - r_halo, cp + r_halo, cp + r_halo], fill=halo)
            r = marker_radius
            draw.ellipse([cp - r, cp - r, cp + r, cp + r], fill=fill)
        else:
            draw.point((cp, cp), fill=fill)
        return np.asarray(img, dtype=np.uint8)

    px, py = _project_track_pixels(
        lons,
        lats,
        img_size=img_size,
        padding=padding,
        fixed_range=fixed_range,
    )
    halo_width = max(line_width + 2, 2)
    halo_fill = tuple(_TRACK_HALO_RGB.tolist())
    if len(segment_rgb) == len(lons) - 1 and len(segment_rgb) > 0:
        for idx in range(len(segment_rgb)):
            start = (float(px[idx]), float(py[idx]))
            end = (float(px[idx + 1]), float(py[idx + 1]))
            draw.line([start, end], fill=halo_fill, width=halo_width)
            draw.line([start, end], fill=tuple(segment_rgb[idx].tolist()), width=line_width)
    else:
        draw.line(list(zip(px.tolist(), py.tolist())), fill=halo_fill, width=halo_width)
        draw.line(list(zip(px.tolist(), py.tolist())), fill=tuple(_TRACK_RGB.tolist()), width=line_width)

    if marker_radius > 0:
        for idx, (x, y) in enumerate(zip(px.tolist(), py.tolist())):
            r_halo = marker_radius + 1
            draw.ellipse([x - r_halo, y - r_halo, x + r_halo, y + r_halo], fill=halo_fill)
            r = marker_radius
            fill = tuple(point_rgb[idx].tolist()) if idx < len(point_rgb) else tuple(_TRACK_RGB.tolist())
            draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    return np.asarray(img, dtype=np.uint8)


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
        seg_rgb = _segment_value_colors(values, color_range=color_range)
        for idx in range(len(seg_rgb)):
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
    *,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
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
    normalized_chunk_mode = normalize_chunk_mode(chunk_mode)
    scales: dict[str, float] = {}
    extents_by_individual: dict[str, list[float]] = {}
    for ind_id, _split_value, df_group in _iter_thumbnail_groups(df, split=None):
        chunks = thumbnail_chunks_for_group(
            df_group,
            chunk_mode=normalized_chunk_mode,
            chunk_size=chunk_size,
            stride=stride,
            burst_gap_seconds=burst_gap_seconds,
        )
        extents = []
        for ch, _info in chunks:
            if len(ch) < min_chunk_size:
                continue
            lon_ext = float(ch[LON_COL].max() - ch[LON_COL].min())
            lat_ext = float(ch[LAT_COL].max() - ch[LAT_COL].min())
            extents.append(max(lon_ext, lat_ext))
        if extents:
            extents_by_individual.setdefault(str(ind_id), []).extend(extents)
    for ind_id, extents in extents_by_individual.items():
        scales[ind_id] = float(np.quantile(extents, quantile))
    return scales


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def normalize_chunk_mode(chunk_mode: str | None) -> str:
    """Normalize thumbnail chunking mode."""
    value = DEFAULT_CHUNK_MODE if chunk_mode in (None, "") else str(chunk_mode).strip().lower()
    if value not in {"fixed", "burst"}:
        raise ValueError(f"chunk_mode must be 'fixed' or 'burst', got {chunk_mode!r}")
    return value


def _normalize_burst_gap_seconds(burst_gap_seconds: float | int | str | None) -> float:
    value = DEFAULT_BURST_GAP_SECONDS if burst_gap_seconds in (None, "") else float(burst_gap_seconds)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"burst_gap_seconds must be positive and finite, got {burst_gap_seconds!r}")
    return float(value)


def _source_gap_seconds(df: pd.DataFrame) -> list[float]:
    gaps: list[float] = []
    for _individual_id, _split_value, df_group in _iter_thumbnail_groups(df, split=None):
        times = pd.to_datetime(df_group[TIME_COL], errors="coerce")
        deltas = times.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        gaps.extend(float(value) for value in deltas if np.isfinite(value) and value >= 0.0)
    return gaps


def resolve_burst_gap_strategy(
    df: pd.DataFrame,
    *,
    burst_gap_mode: str | None = DEFAULT_BURST_GAP_MODE,
    burst_gap_seconds: float | int | str | None = DEFAULT_BURST_GAP_SECONDS,
    burst_gap_quantile: float | int | str | None = DEFAULT_BURST_GAP_QUANTILE,
) -> dict[str, object]:
    mode = normalize_burst_gap_mode(burst_gap_mode)
    fallback_seconds = _normalize_burst_gap_seconds(burst_gap_seconds)
    quantile = normalize_burst_gap_quantile(burst_gap_quantile)
    gaps = _source_gap_seconds(df)

    effective_seconds = fallback_seconds
    used_fallback = False
    if mode == "quantile":
        quantile_seconds = float(np.quantile(np.asarray(gaps, dtype=np.float64), quantile)) if gaps else math.nan
        if not np.isfinite(quantile_seconds) or quantile_seconds <= 0.0:
            used_fallback = True
        else:
            effective_seconds = quantile_seconds

    return {
        "mode": mode,
        "quantile": float(quantile),
        "fallback_seconds": float(fallback_seconds),
        "effective_seconds": float(effective_seconds),
        "gap_count": int(len(gaps)),
        "used_fallback": bool(used_fallback),
    }


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


def assign_time_gap_bursts(
    df: pd.DataFrame,
    *,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    individual_col: str = INDIVIDUAL_COL,
    time_col: str = TIME_COL,
    split_col: str | None = SET_COL,
    burst_idx_col: str = "burst_idx",
    time_delta_col: str = "burst_time_delta_s",
) -> pd.DataFrame:
    """Return a copy with per-track burst IDs based on absolute time gaps.

    A new burst starts when the time delta from the previous fix in the same
    individual/split track is strictly greater than ``burst_gap_seconds``.
    """
    gap_seconds = _normalize_burst_gap_seconds(burst_gap_seconds)
    if individual_col not in df.columns:
        raise KeyError(f"Missing individual column {individual_col!r}")
    if time_col not in df.columns:
        raise KeyError(f"Missing time column {time_col!r}")

    group_cols = [individual_col]
    if split_col and split_col in df.columns:
        group_cols.append(split_col)
    sort_cols = [*group_cols, time_col]

    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.sort_values(sort_cols, kind="mergesort")
    deltas = out.groupby(group_cols, sort=False)[time_col].diff().dt.total_seconds()
    starts = deltas.isna() | (deltas > gap_seconds)
    out[time_delta_col] = deltas
    out[burst_idx_col] = starts.groupby([out[col] for col in group_cols], sort=False).cumsum().astype(int) - 1
    return out


def burst_individual(
    df_group: pd.DataFrame,
    *,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    """Split one already-scoped individual/split track into time-gap bursts."""
    if len(df_group) == 0:
        return []
    burst_gap = _normalize_burst_gap_seconds(burst_gap_seconds)
    burst_df = assign_time_gap_bursts(
        df_group,
        burst_gap_seconds=burst_gap,
        split_col=None,
    )
    chunks: list[tuple[pd.DataFrame, dict[str, object]]] = []
    for burst_idx, chunk in burst_df.groupby("burst_idx", sort=True):
        idx = int(burst_idx)
        chunks.append(
            (
                chunk.sort_values(TIME_COL),
                {
                    "chunk_mode": "burst",
                    "burst_idx": idx,
                    "burst_id": f"burst_{idx:06d}",
                    "burst_gap_seconds": burst_gap,
                },
            )
        )
    return chunks


def thumbnail_chunks_for_group(
    df_group: pd.DataFrame,
    *,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    chunk_size: int = 50,
    stride: int | None = None,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
) -> list[tuple[pd.DataFrame, dict[str, object]]]:
    """Return thumbnail chunks plus chunking metadata for one individual/split group."""
    mode = normalize_chunk_mode(chunk_mode)
    if mode == "burst":
        return burst_individual(df_group, burst_gap_seconds=burst_gap_seconds)
    return [
        (
            chunk,
            {
                "chunk_mode": "fixed",
                "burst_idx": None,
                "burst_id": "",
                "burst_gap_seconds": _normalize_burst_gap_seconds(burst_gap_seconds),
            },
        )
        for chunk in chunk_individual(df_group, chunk_size, stride)
    ]


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
            yield str(individual_id), str(split_value), df_group.sort_values(TIME_COL, kind="mergesort")
        return

    grouped = df.groupby(INDIVIDUAL_COL, sort=False)
    for individual_id, df_group in grouped:
        if split is not None:
            split_value = split
        elif SET_COL in df_group.columns and not df_group[SET_COL].empty:
            split_value = str(df_group[SET_COL].iloc[0])
        else:
            split_value = None
        yield str(individual_id), split_value, df_group.sort_values(TIME_COL, kind="mergesort")


def _chunk_extent_metrics(
    *,
    center_lat: float,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(center_lat))
    extent_x_m = max(0.0, float(lon_max - lon_min)) * max(0.0, lon_scale)
    extent_y_m = max(0.0, float(lat_max - lat_min)) * lat_scale
    return extent_x_m, extent_y_m


def build_track_thumbnails(
    trajectories_path: Path | str,
    *,
    split: str | None = "train",
    individual_id: str | None = None,
    img_size: int = 64,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    chunk_size: int = 50,
    stride: int | None = None,
    min_chunk_size: int = 10,
    burst_gap_mode: str = DEFAULT_BURST_GAP_MODE,
    burst_gap_seconds: float = DEFAULT_BURST_GAP_SECONDS,
    burst_gap_quantile: float = DEFAULT_BURST_GAP_QUANTILE,
    line_width: int = 1,
    marker_radius: int = 1,
    padding: int = 2,
    scale_mode: str = "individual",
    scale_quantile: float = 0.5,
    color_by: str | None = None,
    basemap: str | None = None,
    anomaly_config: dict[str, object] | None = None,
    exclude_outlier_cols: set[str] | frozenset[str] | list[str] | tuple[str, ...] | None = None,
) -> TrackThumbnailBuild:
    """Build thumbnail images, metadata, and chunk paths for one trajectories CSV/split."""
    normalized_chunk_mode = normalize_chunk_mode(chunk_mode)
    normalized_burst_gap_mode = normalize_burst_gap_mode(burst_gap_mode)
    fallback_burst_gap_seconds = _normalize_burst_gap_seconds(burst_gap_seconds)
    normalized_burst_gap_quantile = normalize_burst_gap_quantile(burst_gap_quantile)
    if scale_mode not in ("individual", "chunk"):
        raise ValueError(f"scale_mode must be 'individual' or 'chunk', got {scale_mode!r}")

    normalized_basemap = normalize_thumbnail_basemap(basemap)
    if normalized_basemap == "osm":
        raise ValueError(
            "The 'osm' basemap is not supported for generated thumbnails. "
            "This batch thumbnail workflow can issue many tile requests, and direct use of "
            "tile.openstreetmap.org is intended for normal interactive viewing rather than "
            "automated thumbnail generation. Use 'positron', 'satellite', or 'none' instead."
        )
    normalized_anomaly = normalize_anomaly_config(anomaly_config)
    anomaly_score_column = str(normalized_anomaly["score_column"])
    if color_by == anomaly_score_column:
        df = load_thumbnail_source_df_with_anomaly(
            trajectories_path,
            split=split,
            individual_id=individual_id,
            include_movement_features=True,
            anomaly_config=normalized_anomaly,
        )
    else:
        df = load_thumbnail_source_df(
            trajectories_path,
            split=split,
            individual_id=individual_id,
            include_movement_features=True,
        )
    source_gap_df = load_thumbnail_source_df(
        trajectories_path,
        split=None,
        individual_id=None,
        include_movement_features=False,
    )
    burst_gap = resolve_burst_gap_strategy(
        source_gap_df,
        burst_gap_mode=normalized_burst_gap_mode,
        burst_gap_seconds=fallback_burst_gap_seconds,
        burst_gap_quantile=normalized_burst_gap_quantile,
    )
    normalized_burst_gap_seconds = float(burst_gap["effective_seconds"])
    outlier_cols = detect_outlier_cols(df, exclude=exclude_outlier_cols)
    color_range: tuple[float, float] | None = None
    color_by_is_binary = _is_binary_color_column_name(color_by)
    if color_by is not None:
        color_cols = detect_color_columns(df, extra_columns=[anomaly_score_column])
        if color_by not in color_cols:
            raise KeyError(f"{color_by!r} is not an available color column. Available: {color_cols}")
        if not color_by_is_binary:
            color_range = _fit_color_range(_coerce_numeric(df[color_by]))

    if scale_mode == "individual":
        ind_scales = compute_individual_scales(
            df,
            chunk_size,
            stride,
            min_chunk_size,
            scale_quantile,
            chunk_mode=normalized_chunk_mode,
            burst_gap_seconds=normalized_burst_gap_seconds,
        )
    else:
        ind_scales = {}

    images: list[np.ndarray] = []
    chunk_paths: list[tuple[tuple[float, float], ...]] = []
    records: list[dict] = []

    for individual_id, chunk_split, df_group in _iter_thumbnail_groups(df, split=split):
        chunks = thumbnail_chunks_for_group(
            df_group,
            chunk_mode=normalized_chunk_mode,
            chunk_size=chunk_size,
            stride=stride,
            burst_gap_seconds=normalized_burst_gap_seconds,
        )
        fixed_range = ind_scales.get(individual_id)
        chunk_idx = 0
        for chunk, chunk_info in chunks:
            if len(chunk) < min_chunk_size:
                continue

            lons = chunk[LON_COL].to_numpy(dtype=float, copy=False)
            lats = chunk[LAT_COL].to_numpy(dtype=float, copy=False)
            render_frame = _render_frame(
                lons,
                lats,
                img_size=img_size,
                padding=padding,
                fixed_range=fixed_range,
            )
            if color_by is None:
                if normalized_basemap == "none":
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
                    frame_key = _frame_cache_key(render_frame, img_size=img_size, padding=padding)
                    background = (
                        _render_basemap_background_cached(normalized_basemap, frame_key)
                        if frame_key is not None
                        else _render_basemap_background(
                            normalized_basemap,
                            frame=render_frame,
                            img_size=img_size,
                            padding=padding,
                        )
                    )
                    mono_rgb = np.repeat(_TRACK_RGB[None, :], len(lons), axis=0)
                    segment_rgb = np.repeat(_TRACK_RGB[None, :], max(len(lons) - 1, 0), axis=0)
                    img = _draw_track_on_rgb_background(
                        background,
                        lons=lons,
                        lats=lats,
                        point_rgb=mono_rgb,
                        segment_rgb=segment_rgb,
                        img_size=img_size,
                        padding=padding,
                        line_width=line_width,
                        marker_radius=marker_radius,
                        fixed_range=fixed_range,
                    )
            else:
                values = (
                    _coerce_outlier_mask(chunk[color_by]).astype(np.float64)
                    if color_by_is_binary
                    else _coerce_numeric(chunk[color_by])
                )
                if normalized_basemap == "none":
                    if color_by_is_binary:
                        background = np.broadcast_to(_COLOR_BACKGROUND_RGB, (img_size, img_size, 3)).copy()
                        point_rgb = _binary_values_to_rgb(values)
                        segment_rgb = _binary_segment_colors(values)
                        img = _draw_track_on_rgb_background(
                            background,
                            lons=lons,
                            lats=lats,
                            point_rgb=point_rgb,
                            segment_rgb=segment_rgb,
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
                            values,
                            color_range=color_range if color_range is not None else (0.0, 1.0),
                            img_size=img_size,
                            padding=padding,
                            line_width=line_width,
                            marker_radius=marker_radius,
                            fixed_range=fixed_range,
                        )
                else:
                    frame_key = _frame_cache_key(render_frame, img_size=img_size, padding=padding)
                    background = (
                        _render_basemap_background_cached(normalized_basemap, frame_key)
                        if frame_key is not None
                        else _render_basemap_background(
                            normalized_basemap,
                            frame=render_frame,
                            img_size=img_size,
                            padding=padding,
                        )
                    )
                    if color_by_is_binary:
                        point_rgb = _binary_values_to_rgb(values)
                        segment_rgb = _binary_segment_colors(values)
                    else:
                        point_rgb = _values_to_rgb(
                            values,
                            color_range=color_range if color_range is not None else (0.0, 1.0),
                        )
                        segment_rgb = _segment_value_colors(
                            values,
                            color_range=color_range if color_range is not None else (0.0, 1.0),
                        )
                    img = _draw_track_on_rgb_background(
                        background,
                        lons=lons,
                        lats=lats,
                        point_rgb=point_rgb,
                        segment_rgb=segment_rgb,
                        img_size=img_size,
                        padding=padding,
                        line_width=line_width,
                        marker_radius=marker_radius,
                        fixed_range=fixed_range,
                    )
            images.append(img)
            chunk_paths.append(tuple((float(lon), float(lat)) for lon, lat in zip(lons.tolist(), lats.tolist())))

            label_dict = aggregate_labels(chunk, outlier_cols)
            center_lat = float(chunk[LAT_COL].mean())
            center_lon = float(chunk[LON_COL].mean())
            lat_min = float(chunk[LAT_COL].min())
            lat_max = float(chunk[LAT_COL].max())
            lon_min = float(chunk[LON_COL].min())
            lon_max = float(chunk[LON_COL].max())
            extent_x_m, extent_y_m = _chunk_extent_metrics(
                center_lat=center_lat,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            )
            record = {
                "global_chunk_idx": len(records),
                "individual_id": individual_id,
                "chunk_idx": chunk_idx,
                "chunk_mode": normalized_chunk_mode,
                "burst_idx": chunk_info["burst_idx"],
                "burst_id": (
                    f"{individual_id}:{chunk_split}:{chunk_info['burst_id']}"
                    if normalized_chunk_mode == "burst"
                    else ""
                ),
                "burst_gap_seconds": normalized_burst_gap_seconds,
                "burst_gap_mode": str(burst_gap["mode"]),
                "burst_gap_quantile": float(burst_gap["quantile"]),
                "burst_gap_fallback_seconds": float(burst_gap["fallback_seconds"]),
                "burst_gap_gap_count": int(burst_gap["gap_count"]),
                "burst_gap_used_fallback": bool(burst_gap["used_fallback"]),
                "split": chunk_split,
                "start_time": chunk[TIME_COL].iloc[0],
                "end_time": chunk[TIME_COL].iloc[-1],
                "n_fixes": len(chunk),
                "start_lon": float(chunk[LON_COL].iloc[0]),
                "start_lat": float(chunk[LAT_COL].iloc[0]),
                "end_lon": float(chunk[LON_COL].iloc[-1]),
                "end_lat": float(chunk[LAT_COL].iloc[-1]),
                "center_lat": center_lat,
                "center_lon": center_lon,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "extent_x_m": extent_x_m,
                "extent_y_m": extent_y_m,
            }
            record.update(label_dict)
            records.append(record)
            chunk_idx += 1

    if images:
        image_arr = np.stack(images, axis=0).astype(np.uint8)
    else:
        if color_by is None and normalized_basemap == "none":
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
        burst_gap=dict(burst_gap),
    )


def __getattr__(name: str):
    if name != "TrackThumbnailDataset":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("movebench.datasets.thumbnail_dataset")
    value = getattr(module, name)
    globals()[name] = value
    return value
