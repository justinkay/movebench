"""MoveBench trajectory viewer (Streamlit).

Run:
  pip install -e .[apps]
  streamlit run trajectory_viz_app.py

Optional:
  MOVEBENCH_DATA_ROOT=/path/to/data/movebench streamlit run trajectory_viz_app.py
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path
import json


# If the user hasn't configured MOVEBENCH_DATA_ROOT, default to the repo-local
# data path so Study.list_available() works out-of-the-box for this checkout.
_default_data_root = (Path(__file__).resolve().parent / "data" / "movebench").resolve()
if "MOVEBENCH_DATA_ROOT" not in os.environ and _default_data_root.exists():
    os.environ["MOVEBENCH_DATA_ROOT"] = str(_default_data_root)

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from movebench import Study, paths
from movebench.trajectories import INDIVIDUAL_COL, LAT_COL, LON_COL, SET_COL, TIME_COL


BASEMAP_STYLES: dict[str, str] = {
    "Positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Dark Matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

MAX_POINTS_PER_PATH = 2_000

PATH_ALPHA = 180

BASE_SATURATION = 0.9
TRAIN_LIGHTNESS = 0.65  # lighter shade
TEST_LIGHTNESS = 0.35  # darker shade

RASTER_MAX_PX = 256
RASTER_SAMPLE_STRIDE = 10

DOT_RADIUS_MIN_PX = 4
DOT_RADIUS_MAX_PX = 14

TIMELINE_BAR_ALPHA = 200
TIMELINE_UNSELECTED_ALPHA_SCALE = 0.25

def _uniform_downsample(path: list[list[float]], max_points: int) -> list[list[float]]:
    n = len(path)
    if n <= max_points:
        return path
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return [path[int(i)] for i in idx]


def _alternating_hue_order(n: int) -> list[int]:
    """Return indices 0..n-1 in an order that maximizes adjacent hue separation."""
    half = (n + 1) // 2
    out: list[int] = []
    for i in range(half):
        out.append(i)
        j = i + half
        if j < n:
            out.append(j)
    return out[:n]


def _hls_to_rgba(h: float, l: float, s: float, alpha: int) -> list[int]:
    r, g, b = colorsys.hls_to_rgb(float(h), float(l), float(s))
    return [
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
        int(alpha),
    ]


def _make_split_palettes(individuals: list[str], alpha: int) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    n = len(individuals)
    if n == 0:
        return {}, {}

    order = _alternating_hue_order(n)
    train: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}
    for idx, indiv in enumerate(individuals):
        h = float(order[idx]) / float(n)
        train[indiv] = _hls_to_rgba(h, TRAIN_LIGHTNESS, BASE_SATURATION, alpha)
        test[indiv] = _hls_to_rgba(h, TEST_LIGHTNESS, BASE_SATURATION, alpha)
    return train, test


def _color_dot_html(rgba: list[int], *, enabled: bool) -> str:
    r, g, b, a = rgba
    alpha = float(a) / 255.0
    if not enabled:
        alpha *= 0.25
    border_alpha = 0.35 if enabled else 0.15
    return (
        "<span "
        "style="
        "\"display:inline-block;"
        "width:12px;"
        "height:12px;"
        "border-radius:999px;"
        f"background:rgba({int(r)},{int(g)},{int(b)},{alpha:.3f});"
        f"border:1px solid rgba(0,0,0,{border_alpha:.3f});"
        "\""
        "></span>"
    )


@st.cache_data(show_spinner=False)
def _load_tile_index(index_path: str) -> dict:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _compute_global_vmin_vmax(tile_paths: tuple[str, ...]) -> tuple[float, float]:
    # Approximate robust range using a stride sample to keep it lightweight.
    import rasterio  # heavy import; keep local

    samples: list[np.ndarray] = []
    for p in tile_paths:
        with rasterio.open(p) as ds:
            a = ds.read(1, masked=True).filled(np.nan).astype(np.float32, copy=False)
        a = a[::RASTER_SAMPLE_STRIDE, ::RASTER_SAMPLE_STRIDE]
        v = a[np.isfinite(a)]
        if v.size:
            samples.append(v)

    if not samples:
        return (0.0, 1.0)

    vals = np.concatenate(samples)
    vmin, vmax = np.percentile(vals, [2.0, 98.0]).tolist()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax) <= float(vmin):
        return (0.0, 1.0)
    return (float(vmin), float(vmax))


def _normalize(a: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    out = (a - float(vmin)) / float(vmax - vmin)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _hillshade(z: np.ndarray, *, azimuth_deg: float = 315.0, altitude_deg: float = 45.0) -> np.ndarray:
    # Lightweight hillshade in pixel space (good enough for visualization).
    az = np.deg2rad(float(azimuth_deg))
    alt = np.deg2rad(float(altitude_deg))

    # Avoid NaNs blowing up gradients.
    zf = z.astype(np.float32, copy=True)
    finite = np.isfinite(zf)
    fill = float(np.nanmedian(zf[finite])) if finite.any() else 0.0
    zf[~finite] = fill

    dy, dx = np.gradient(zf)
    slope = (np.pi / 2.0) - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)

    shaded = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0.0, 1.0).astype(np.float32, copy=False)


@st.cache_data(show_spinner=False)
def _tile_png_data_url(
    tile_path: str,
    *,
    style: str,
    vmin: float | None,
    vmax: float | None,
    max_px: int,
) -> str:
    import base64
    import io

    import rasterio  # heavy import; keep local
    from PIL import Image

    with rasterio.open(tile_path) as ds:
        a = ds.read(1, masked=True).filled(np.nan).astype(np.float32, copy=False)

    h, w = a.shape
    step = int(np.ceil(max(h, w) / float(max_px))) if max_px > 0 else 1
    step = max(1, step)
    a = a[::step, ::step]
    mask = np.isfinite(a)

    style_norm = style.strip().lower()
    if style_norm == "hillshade":
        shade = _hillshade(a)
        rgb = (shade * 255.0).round().astype(np.uint8)
        rgba = np.stack([rgb, rgb, rgb, (mask * PATH_ALPHA).astype(np.uint8)], axis=-1)
    else:
        if vmin is None or vmax is None:
            v = a[mask]
            if v.size:
                vmin_, vmax_ = np.percentile(v, [2.0, 98.0]).tolist()
                vmin = float(vmin_)
                vmax = float(vmax_)
            else:
                vmin, vmax = 0.0, 1.0

        norm = _normalize(a, float(vmin), float(vmax))

        if style_norm == "grayscale":
            rgb = (norm * 255.0).round().astype(np.uint8)
            rgba = np.stack([rgb, rgb, rgb, (mask * PATH_ALPHA).astype(np.uint8)], axis=-1)
        elif style_norm == "terrain colormap":
            import matplotlib.cm as cm

            cmap = cm.get_cmap("terrain")
            rgba_f = cmap(norm)  # float RGBA in [0, 1]
            rgba = (rgba_f * 255.0).round().astype(np.uint8)
            rgba[..., 3] = (mask * PATH_ALPHA).astype(np.uint8)
        else:
            raise ValueError(f"Unknown raster style: {style!r}")

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _span_to_zoom(span_deg: float) -> float:
    # Deterministic, dependency-free heuristic.
    if not np.isfinite(span_deg) or span_deg <= 0:
        return 2.0
    if span_deg > 120:
        return 1.0
    if span_deg > 60:
        return 2.0
    if span_deg > 30:
        return 3.0
    if span_deg > 15:
        return 4.0
    if span_deg > 8:
        return 5.0
    if span_deg > 4:
        return 6.0
    if span_deg > 2:
        return 7.0
    if span_deg > 1:
        return 8.0
    if span_deg > 0.5:
        return 9.0
    if span_deg > 0.25:
        return 10.0
    return 11.0


@st.cache_data(show_spinner=False)
def list_study_ids() -> list[str]:
    return Study.list_available()


@st.cache_data(show_spinner=False)
def load_study_df(study_id: str) -> pd.DataFrame:
    s = Study(study_id)
    df = pd.read_csv(s.trajectories_path)

    # Parse + clean.
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, LON_COL, LAT_COL])

    # Keep only what we need.
    keep = [INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL, SET_COL]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort")
    return df


@st.cache_data(show_spinner=False)
def build_path_rows(df: pd.DataFrame) -> list[dict]:
    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    train_palette, test_palette = _make_split_palettes(individuals, PATH_ALPHA)

    rows: list[dict] = []
    for split, palette in (("train", train_palette), ("test", test_palette)):
        sdf = df[df[SET_COL] == split]
        if len(sdf) == 0:
            continue
        for indiv, g in sdf.groupby(INDIVIDUAL_COL, sort=False):
            indiv_str = str(indiv)
            coords = g[[LON_COL, LAT_COL]].to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(coords).all(axis=1)
            coords = coords[finite]
            if len(coords) < 2:
                continue
            path = coords.tolist()
            path = _uniform_downsample(path, MAX_POINTS_PER_PATH)
            color = palette.get(indiv_str)
            if color is None:
                # Shouldn't happen, but keep the app robust to unexpected ids.
                fallback_l = TRAIN_LIGHTNESS if split == "train" else TEST_LIGHTNESS
                color = _hls_to_rgba(0.0, fallback_l, BASE_SATURATION, PATH_ALPHA)
            rows.append({"individual": indiv_str, "set": split, "path": path, "color": color})
    return rows


def compute_initial_view(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {"latitude": 0.0, "longitude": 0.0, "zoom": 1.0}

    lons = df[LON_COL].to_numpy(dtype=np.float64, copy=False)
    lats = df[LAT_COL].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(lons) & np.isfinite(lats)
    if not finite.any():
        return {"latitude": 0.0, "longitude": 0.0, "zoom": 1.0}

    lons = lons[finite]
    lats = lats[finite]

    min_lon = float(np.min(lons))
    max_lon = float(np.max(lons))
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))

    center_lon = 0.5 * (min_lon + max_lon)
    center_lat = 0.5 * (min_lat + max_lat)

    span = max(max_lon - min_lon, max_lat - min_lat)
    zoom = _span_to_zoom(float(span))

    return {"latitude": center_lat, "longitude": center_lon, "zoom": zoom}


def _build_timeline_times_and_points(
    df: pd.DataFrame,
    *,
    selected: set[str],
    enabled_splits: list[str],
) -> tuple[list[int], list[dict]]:
    if not selected or not enabled_splits:
        return [], []

    sdf = df
    sdf = sdf[sdf[INDIVIDUAL_COL].astype(str).isin(selected)]
    sdf = sdf[sdf[SET_COL].isin(enabled_splits)]
    if len(sdf) == 0:
        return [], []

    sdf = sdf.dropna(subset=[TIME_COL, LON_COL, LAT_COL]).copy()
    sdf = sdf.drop_duplicates(subset=[INDIVIDUAL_COL, SET_COL, TIME_COL], keep="last")
    if len(sdf) == 0:
        return [], []

    times = sdf[TIME_COL].sort_values(kind="mergesort").drop_duplicates()
    times_list = [pd.Timestamp(t) for t in times.tolist()]
    times_ms = [int(t.value // 1_000_000) for t in times_list]
    time_ms_to_idx = {ms: i for i, ms in enumerate(times_ms)}

    points: list[dict] = []
    for indiv, split, t, lon, lat in sdf[[INDIVIDUAL_COL, SET_COL, TIME_COL, LON_COL, LAT_COL]].itertuples(
        index=False, name=None
    ):
        ts = pd.Timestamp(t)
        ms = int(ts.value // 1_000_000)
        idx = time_ms_to_idx.get(ms)
        if idx is None:
            continue
        lon_f = float(lon)
        lat_f = float(lat)
        if not np.isfinite(lon_f) or not np.isfinite(lat_f):
            continue
        points.append(
            {
                "time_idx": int(idx),
                "position": [lon_f, lat_f],
                "individual": str(indiv),
                "set": str(split),
            }
        )

    return times_ms, points


def _build_individual_split_coverage_ms(df: pd.DataFrame) -> list[dict]:
    plot_df = df
    ranges = plot_df.groupby([INDIVIDUAL_COL, SET_COL], sort=False)[TIME_COL].agg(start="min", end="max")
    out: list[dict] = []
    for (indiv, split), r in ranges.iterrows():
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"])
        if pd.isna(start) or pd.isna(end):
            continue
        out.append(
            {
                "individual": str(indiv),
                "set": str(split),
                "start_ms": int(start.value // 1_000_000),
                "end_ms": int(end.value // 1_000_000),
            }
        )
    return out


@st.cache_data(show_spinner=False)
def _build_individual_movement_stats(df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Compute per-individual movement summary stats.

    - median fix time: median seconds between consecutive fixes
    - median step length: median meters between consecutive fixes
    - 95th percentile step length: p95 meters between consecutive fixes

    Computed within each split (train/test) and pooled, so we don't create
    artificial cross-split steps.
    """

    def _haversine_m(
        lon1_deg: np.ndarray, lat1_deg: np.ndarray, lon2_deg: np.ndarray, lat2_deg: np.ndarray
    ) -> np.ndarray:
        r = 6_371_000.0
        lon1 = np.deg2rad(lon1_deg)
        lat1 = np.deg2rad(lat1_deg)
        lon2 = np.deg2rad(lon2_deg)
        lat2 = np.deg2rad(lat2_deg)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return r * c

    out: dict[str, dict[str, float | None]] = {}
    for indiv, g_indiv in df.groupby(INDIVIDUAL_COL, sort=False):
        dt_s_parts: list[np.ndarray] = []
        step_m_parts: list[np.ndarray] = []

        for _, g in g_indiv.groupby(SET_COL, sort=False):
            g2 = g.dropna(subset=[TIME_COL, LON_COL, LAT_COL]).drop_duplicates(subset=[TIME_COL], keep="last")
            if len(g2) < 2:
                continue
            g2 = g2.sort_values(TIME_COL, kind="mergesort")

            coords = g2[[LON_COL, LAT_COL]].to_numpy(dtype=np.float64, copy=False)
            finite = np.isfinite(coords).all(axis=1)
            if not finite.all():
                g2 = g2.iloc[np.flatnonzero(finite)].copy()
                coords = coords[finite]
            if len(coords) < 2:
                continue

            times_ns = (
                g2[TIME_COL]
                .to_numpy(dtype="datetime64[ns]", copy=False)
                .astype(np.int64, copy=False)
            )
            dt_s = np.diff(times_ns).astype(np.float64) / 1_000_000_000.0
            dt_s = dt_s[np.isfinite(dt_s) & (dt_s > 0)]
            if dt_s.size:
                dt_s_parts.append(dt_s)

            lon = coords[:, 0]
            lat = coords[:, 1]
            step_m = _haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
            step_m = step_m[np.isfinite(step_m) & (step_m > 0)]
            if step_m.size:
                step_m_parts.append(step_m)

        dt_s_all = np.concatenate(dt_s_parts) if dt_s_parts else np.array([], dtype=np.float64)
        step_m_all = np.concatenate(step_m_parts) if step_m_parts else np.array([], dtype=np.float64)

        out[str(indiv)] = {
            "median_fix_s": float(np.median(dt_s_all)) if dt_s_all.size else None,
            "median_step_m": float(np.median(step_m_all)) if step_m_all.size else None,
            "p95_step_m": float(np.quantile(step_m_all, 0.95)) if step_m_all.size else None,
        }

    return out


def _render_deckgl_viewer(
    *,
    study_id: str,
    map_style: str,
    view_state: dict[str, float],
    path_rows: list[dict],
    raster_tiles: list[dict],
    individuals: list[str],
    times_ms: list[int],
    points: list[dict],
    coverage: list[dict],
    split_colors: dict[str, dict[str, str]],
    indiv_stats: dict[str, dict[str, float | None]],
) -> None:
    map_h = 600
    # Rows are taller now (checkbox + 2 lines of stats), so allocate more height
    # to avoid an internal scrollbar for typical studies (e.g., 10 individuals).
    bars_h = min(900, max(200, 60 * max(1, len(individuals)) + 24))
    total_h = int(map_h + bars_h + 220)

    payload = {
        "studyId": study_id,
        "mapStyle": map_style,
        "viewState": view_state,
        "paths": path_rows,
        "rasterTiles": raster_tiles,
        "individuals": individuals,
        "timesMs": times_ms,
        "points": points,
        "coverage": coverage,
        "splitColors": split_colors,
        "indivStats": indiv_stats,
    }
    payload_json = json.dumps(payload).replace("</", "<\\/")

    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link
      rel="stylesheet"
      href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css"
    />
    <style>
      body {{
        margin: 0;
        padding: 0;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      }}
      .mb-wrap {{
        width: 100%;
      }}
      .mb-indiv-name {{
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      #mb-map {{
        width: 100%;
        height: {map_h}px;
        border-radius: 10px;
        overflow: hidden;
      }}
      .mb-timeline {{
        padding: 12px 2px 0 2px;
      }}
      .mb-title-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin: 2px 0 8px 0;
      }}
      .mb-title {{
        font-weight: 600;
        font-size: 14px;
        margin: 0;
      }}
      .mb-title-controls {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 12px;
        font-size: 12px;
      }}
      .mb-title-controls label {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
      }}
      .mb-title-controls button {{
        appearance: none;
        border: 1px solid rgba(49, 51, 63, 0.25);
        background: rgba(49, 51, 63, 0.03);
        color: inherit;
        padding: 3px 8px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 12px;
      }}
      .mb-title-controls button:hover {{
        background: rgba(49, 51, 63, 0.06);
      }}
      #mb-bars {{
        max-height: {bars_h}px;
        overflow-y: auto;
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 10px;
        padding: 8px;
      }}
      .mb-row {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        margin: 6px 0;
      }}
      .mb-label {{
        width: 170px;
        flex: 0 0 170px;
        display: flex;
        flex-direction: column;
        align-items: stretch;
        font-size: 12px;
        gap: 2px;
        cursor: pointer;
      }}
      .mb-label-top {{
        display: flex;
        align-items: center;
        gap: 8px;
        min-height: 18px;
      }}
      .mb-label input[type="checkbox"] {{
        margin: 0;
        flex: 0 0 auto;
      }}
      .mb-indiv-stats {{
        display: flex;
        flex-wrap: wrap;
        gap: 2px 10px;
        font-size: 11px;
        color: rgba(49, 51, 63, 0.65);
      }}
      .mb-indiv-stats span {{
        white-space: nowrap;
      }}
      .mb-track {{
        flex: 1;
        height: 10px;
        margin-top: 4px;
        background: rgba(49, 51, 63, 0.08);
        border-radius: 999px;
        position: relative;
      }}
      .mb-bar {{
        position: absolute;
        height: 100%;
        border-radius: 999px;
      }}
      .mb-cursor {{
        position: absolute;
        top: -2px;
        bottom: -2px;
        width: 2px;
        background: rgba(0, 0, 0, 0.35);
        transform: translateX(-1px);
        pointer-events: none;
        border-radius: 2px;
      }}
      .mb-slider-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 8px 0 8px;
      }}
      .mb-label-spacer {{
        width: 170px;
        flex: 0 0 170px;
      }}
      .mb-slider-wrap {{
        flex: 1;
        position: relative;
      }}
      #mb-slider {{
        width: 100%;
      }}
      #mb-time {{
        position: absolute;
        right: 0;
        top: -18px;
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        white-space: nowrap;
        pointer-events: none;
        background: rgba(255, 255, 255, 0.65);
        padding: 0 4px;
        border-radius: 6px;
      }}
    </style>
  </head>
  <body>
    <div class="mb-wrap">
      <div id="mb-map"></div>
      <div class="mb-timeline">
        <div class="mb-title-row">
          <div class="mb-title">Data availability</div>
          <div class="mb-title-controls">
            <label><input type="checkbox" id="mb-show-train" /> Train</label>
            <label><input type="checkbox" id="mb-show-test" /> Test</label>
            <button type="button" id="mb-select-all">All</button>
            <button type="button" id="mb-select-none">None</button>
          </div>
        </div>
        <div id="mb-bars"></div>
        <div class="mb-slider-row">
          <div class="mb-label-spacer"></div>
          <div class="mb-slider-wrap">
            <input id="mb-slider" type="range" step="1" />
            <div id="mb-time"></div>
          </div>
        </div>
      </div>
    </div>

    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <script src="https://unpkg.com/@deck.gl/mapbox@8.9.33/dist.min.js"></script>
    <script>
      const DATA = {payload_json};
      const STUDY_ID = String(DATA.studyId || "");
      const STATE_KEY = `movebench::${{STUDY_ID}}::viewerState`;
      const VIEW_KEY = `movebench::${{STUDY_ID}}::mapView`;

      function safeNumber(v, fallback) {{
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
      }}

      function loadJSON(key) {{
        try {{
          const raw = window.localStorage.getItem(key);
          if (!raw) return null;
          return JSON.parse(raw);
        }} catch (e) {{
          return null;
        }}
      }}

      function saveJSON(key, val) {{
        try {{
          window.localStorage.setItem(key, JSON.stringify(val));
        }} catch (e) {{
          // ignore
        }}
      }}

      function fmtTime(ms) {{
        const d = new Date(ms);
        if (!Number.isFinite(d.getTime())) return "";
        return d.toISOString().replace("T", " ").replace(".000Z", "Z");
      }}

      function fmtDurationS(sec) {{
        if (!Number.isFinite(sec) || sec < 0) return "—";
        const s = sec;
        if (s < 60) return Math.round(s) + "s";
        const m = s / 60.0;
        if (m < 60) return (m < 10 ? m.toFixed(1) : m.toFixed(0)) + "m";
        const h = m / 60.0;
        if (h < 48) return (h < 10 ? h.toFixed(1) : h.toFixed(0)) + "h";
        const d = h / 24.0;
        return (d < 10 ? d.toFixed(1) : d.toFixed(0)) + "d";
      }}

      function fmtDistanceM(meters) {{
        if (!Number.isFinite(meters) || meters < 0) return "—";
        const m = meters;
        if (m < 1000) return Math.round(m) + "m";
        const km = m / 1000.0;
        return (km < 10 ? km.toFixed(1) : km.toFixed(0)) + "km";
      }}

      function upperBound(sortedArr, x) {{
        let lo = 0;
        let hi = sortedArr.length;
        while (lo < hi) {{
          const mid = (lo + hi) >> 1;
          if (sortedArr[mid] <= x) lo = mid + 1;
          else hi = mid;
        }}
        return lo;
      }}

      const indivs = Array.isArray(DATA.individuals) ? DATA.individuals.map(String) : [];
      const splitColors = (DATA.splitColors && typeof DATA.splitColors === "object") ? DATA.splitColors : {{}};
      const indivStats = (DATA.indivStats && typeof DATA.indivStats === "object") ? DATA.indivStats : {{}};

      const storedState = loadJSON(STATE_KEY) || {{}};
      let showTrain = storedState.showTrain;
      if (typeof showTrain !== "boolean") showTrain = true;
      let showTest = storedState.showTest;
      if (typeof showTest !== "boolean") showTest = true;

      const indivSet = new Set(indivs);
      let selectedIndividuals = new Set();
      if (Array.isArray(storedState.selectedIndividuals)) {{
        for (const v of storedState.selectedIndividuals.map(String)) {{
          if (indivSet.has(v)) selectedIndividuals.add(v);
        }}
      }} else {{
        selectedIndividuals = new Set(indivs);
      }}

      const timesMs = Array.isArray(DATA.timesMs) ? DATA.timesMs.map(Number).filter(Number.isFinite) : [];
      timesMs.sort((a, b) => a - b);
      let globalMin = timesMs.length ? timesMs[0] : 0;
      let globalMax = timesMs.length ? timesMs[timesMs.length - 1] : 1;
      if (!(globalMax > globalMin)) {{
        globalMin = 0;
        globalMax = 1;
      }}

      let currentTimeMs = Number(storedState.timeMs);
      if (!Number.isFinite(currentTimeMs)) {{
        currentTimeMs = timesMs.length ? timesMs[0] : globalMin;
      }}
      currentTimeMs = Math.max(globalMin, Math.min(globalMax, currentTimeMs));

      function saveStateSoon() {{
        if (saveStateSoon._t) window.clearTimeout(saveStateSoon._t);
        saveStateSoon._t = window.setTimeout(() => {{
          saveJSON(STATE_KEY, {{
            showTrain,
            showTest,
            selectedIndividuals: Array.from(selectedIndividuals),
            timeMs: currentTimeMs,
          }});
        }}, 50);
      }}

      // ---- map init (restore view state if present) ----
      const storedView = loadJSON(VIEW_KEY) || {{}};
      const initCenter = Array.isArray(storedView.center) && storedView.center.length === 2
        ? [safeNumber(storedView.center[0], 0), safeNumber(storedView.center[1], 0)]
        : [
            safeNumber(DATA.viewState && DATA.viewState.longitude, 0),
            safeNumber(DATA.viewState && DATA.viewState.latitude, 0),
          ];
      const initZoom = safeNumber(storedView.zoom, safeNumber(DATA.viewState && DATA.viewState.zoom, 1));
      const initBearing = safeNumber(storedView.bearing, 0);
      const initPitch = safeNumber(storedView.pitch, 0);

      const map = new maplibregl.Map({{
        container: "mb-map",
        style: DATA.mapStyle,
        center: initCenter,
        zoom: initZoom,
        bearing: initBearing,
        pitch: initPitch,
        attributionControl: true,
      }});
      map.addControl(new maplibregl.NavigationControl({{ showCompass: true }}), "top-right");
      map.on("moveend", () => {{
        const c = map.getCenter();
        saveJSON(VIEW_KEY, {{
          center: [c.lng, c.lat],
          zoom: map.getZoom(),
          bearing: map.getBearing(),
          pitch: map.getPitch(),
        }});
      }});

      const MapboxOverlayCtor = deck && (deck.MapboxOverlay || (deck.mapbox && deck.mapbox.MapboxOverlay));
      const overlay = MapboxOverlayCtor ? new MapboxOverlayCtor({{
        interleaved: true,
        layers: [],
        getTooltip: ({{ object, layer }}) => {{
          if (!object || !layer) return null;
          if (layer.id === "paths") return {{ text: `${{object.set}} | ${{object.individual}}` }};
          if (layer.id === "dots") return {{ text: `${{object.set}} | ${{object.individual}}` }};
          return null;
        }},
      }}) : null;
      if (overlay) {{
        map.addControl(overlay);
      }} else {{
        console.error("deck.gl MapboxOverlay not found; dots will be disabled.");
      }}

      const rasterTiles = Array.isArray(DATA.rasterTiles) ? DATA.rasterTiles : [];
      const bitmapLayers = rasterTiles.map((t) => new deck.BitmapLayer({{
        id: String(t.id || `raster_${{Math.random()}}`),
        bounds: t.bounds,
        image: t.image,
        opacity: safeNumber(t.opacity, 1.0),
        pickable: false,
      }}));

      const paths = Array.isArray(DATA.paths) ? DATA.paths : [];
      const points = Array.isArray(DATA.points) ? DATA.points : [];
      const samplesByIndiv = new Map();
      for (const p of points) {{
        const indiv = String(p.individual);
        const split = String(p.set);
        const timeIdx = Number(p.time_idx);
        if (!Number.isInteger(timeIdx) || timeIdx < 0 || timeIdx >= timesMs.length) continue;
        const ms = timesMs[timeIdx];
        const pos = (Array.isArray(p.position) && p.position.length === 2) ? p.position : null;
        if (!pos) continue;

        let splits = samplesByIndiv.get(indiv);
        if (!splits) {{
          splits = {{}};
          samplesByIndiv.set(indiv, splits);
        }}
        if (!splits[split]) splits[split] = [];
        splits[split].push([ms, pos]);
      }}

      const seriesByIndiv = new Map();
      for (const [indiv, splits] of samplesByIndiv.entries()) {{
        const outSplits = {{}};
        for (const split of ["train", "test"]) {{
          const samples = splits[split];
          if (!Array.isArray(samples) || samples.length === 0) continue;
          samples.sort((a, b) => a[0] - b[0]);
          const times = [];
          const positions = [];
          for (const s of samples) {{
            const t = s[0];
            const pos = s[1];
            if (times.length && t === times[times.length - 1]) {{
              positions[positions.length - 1] = pos;
            }} else {{
              times.push(t);
              positions.push(pos);
            }}
          }}
          outSplits[split] = {{ times, positions }};
        }}
        seriesByIndiv.set(indiv, outSplits);
      }}

      function dotsAtTime(tMs) {{
        const out = [];
        for (const indiv of selectedIndividuals) {{
          const splits = seriesByIndiv.get(String(indiv));
          if (!splits) continue;
          for (const split of ["train", "test"]) {{
            if (!splitEnabled(split)) continue;
            const s = splits[split];
            if (!s) continue;
            const times = s.times;
            const positions = s.positions;
            const n = times.length;
            if (n === 0) continue;
            if (tMs < times[0] || tMs > times[n - 1]) continue;

            const j = upperBound(times, tMs) - 1;
            if (j < 0) continue;

            const individual = String(indiv);
            if (j >= n - 1 || times[j] === tMs) {{
              out.push({{ position: positions[j], individual, set: split }});
              continue;
            }}
            const t0 = times[j];
            const t1 = times[j + 1];
            const denom = (t1 - t0);
            if (!(denom > 0)) {{
              out.push({{ position: positions[j], individual, set: split }});
              continue;
            }}
            const a = (tMs - t0) / denom;
            const p0 = positions[j];
            const p1 = positions[j + 1];
            const lon = p0[0] + a * (p1[0] - p0[0]);
            const lat = p0[1] + a * (p1[1] - p0[1]);
            out.push({{ position: [lon, lat], individual, set: split }});
          }}
        }}
        return out;
      }}

      function splitEnabled(split) {{
        return (split === "train" && showTrain) || (split === "test" && showTest);
      }}

      function makePathLayer(data) {{
        return new deck.PathLayer({{
          id: "paths",
          data: data,
          getPath: (d) => d.path,
          getColor: (d) => d.color,
          pickable: true,
          widthMinPixels: 2,
        }});
      }}

      function makeDotLayer(dots) {{
        return new deck.ScatterplotLayer({{
          id: "dots",
          data: dots,
          getPosition: (d) => d.position,
          getFillColor: [0, 0, 0, 255],
          getLineColor: [255, 255, 255, 230],
          lineWidthMinPixels: 1,
          getRadius: 1,
          radiusMinPixels: {DOT_RADIUS_MIN_PX},
          radiusMaxPixels: {DOT_RADIUS_MAX_PX},
          pickable: true,
        }});
      }}

      function renderLayers() {{
        if (!overlay) return;
        const filteredPaths = paths.filter((p) => selectedIndividuals.has(String(p.individual)) && splitEnabled(String(p.set)));
        const dots = dotsAtTime(currentTimeMs);
        overlay.setProps({{
          layers: [...bitmapLayers, makePathLayer(filteredPaths), makeDotLayer(dots)],
        }});
      }}

      // ---- data availability controls ----
      const showTrainEl = document.getElementById("mb-show-train");
      const showTestEl = document.getElementById("mb-show-test");
      const selectAllEl = document.getElementById("mb-select-all");
      const selectNoneEl = document.getElementById("mb-select-none");
      showTrainEl.checked = showTrain;
      showTestEl.checked = showTest;

      showTrainEl.addEventListener("change", () => {{
        showTrain = !!showTrainEl.checked;
        updateBarsUI();
        renderLayers();
        saveStateSoon();
      }});
      showTestEl.addEventListener("change", () => {{
        showTest = !!showTestEl.checked;
        updateBarsUI();
        renderLayers();
        saveStateSoon();
      }});

      // ---- timeline bars (split-colored) ----
      const coverage = Array.isArray(DATA.coverage) ? DATA.coverage : [];
      const coverageByIndiv = new Map();
      for (const c of coverage) {{
        const indiv = String(c.individual);
        const split = String(c.set);
        const start = Number(c.start_ms);
        const end = Number(c.end_ms);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
        if (!coverageByIndiv.has(indiv)) coverageByIndiv.set(indiv, {{}});
        coverageByIndiv.get(indiv)[split] = {{ start_ms: start, end_ms: end }};
      }}

      const barsEl = document.getElementById("mb-bars");
      const barEls = [];
      const labelEls = new Map();
      const indivCheckboxes = new Map();
      const cursorEls = [];

      function leftPct(ms) {{
        return ((ms - globalMin) / (globalMax - globalMin)) * 100.0;
      }}
      function widthPct(startMs, endMs) {{
        return Math.max(0.4, ((endMs - startMs) / (globalMax - globalMin)) * 100.0);
      }}

      for (const indiv of indivs) {{
        const row = document.createElement("div");
        row.className = "mb-row";

        const label = document.createElement("label");
        label.className = "mb-label";
        const top = document.createElement("div");
        top.className = "mb-label-top";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = selectedIndividuals.has(indiv);

        const name = document.createElement("span");
        name.className = "mb-indiv-name";
        name.textContent = indiv;
        top.appendChild(cb);
        top.appendChild(name);

        const stats = document.createElement("div");
        stats.className = "mb-indiv-stats";
        const st = indivStats[indiv] || {{}};
        const stat1 = document.createElement("span");
        stat1.textContent = "fix med: " + fmtDurationS(st.median_fix_s);
        const stat2 = document.createElement("span");
        stat2.textContent = "step med: " + fmtDistanceM(st.median_step_m);
        const stat3 = document.createElement("span");
        stat3.textContent = "step p95: " + fmtDistanceM(st.p95_step_m);
        stats.appendChild(stat1);
        stats.appendChild(stat2);
        stats.appendChild(stat3);

        label.appendChild(top);
        label.appendChild(stats);

        const track = document.createElement("div");
        track.className = "mb-track";

        const cursor = document.createElement("div");
        cursor.className = "mb-cursor";
        track.appendChild(cursor);
        cursorEls.push(cursor);

        const ranges = coverageByIndiv.get(indiv) || {{}};
        for (const split of ["train", "test"]) {{
          const r = ranges[split];
          if (!r) continue;
          const bar = document.createElement("div");
          bar.className = "mb-bar";
          bar.dataset.individual = indiv;
          bar.dataset.split = split;
          bar.style.left = `${{leftPct(r.start_ms)}}%`;
          bar.style.width = `${{widthPct(r.start_ms, r.end_ms)}}%`;
          const c = (splitColors[indiv] && splitColors[indiv][split]) ? splitColors[indiv][split] : "rgba(100,100,100,0.7)";
          bar.style.background = c;
          track.appendChild(bar);
          barEls.push(bar);
        }}

        row.appendChild(label);
        row.appendChild(track);
        barsEl.appendChild(row);

        labelEls.set(indiv, label);
        indivCheckboxes.set(indiv, cb);
        cb.addEventListener("change", () => {{
          if (cb.checked) selectedIndividuals.add(indiv);
          else selectedIndividuals.delete(indiv);
          updateBarsUI();
          renderLayers();
          saveStateSoon();
        }});
      }}

      selectAllEl.addEventListener("click", () => {{
        selectedIndividuals = new Set(indivs);
        for (const [indiv, cb] of indivCheckboxes.entries()) cb.checked = true;
        updateBarsUI();
        renderLayers();
        saveStateSoon();
      }});
      selectNoneEl.addEventListener("click", () => {{
        selectedIndividuals = new Set();
        for (const [indiv, cb] of indivCheckboxes.entries()) cb.checked = false;
        updateBarsUI();
        renderLayers();
        saveStateSoon();
      }});

      function updateCursor() {{
        const pct = leftPct(currentTimeMs);
        for (const el of cursorEls) {{
          el.style.left = `${{pct}}%`;
        }}
      }}

      function updateBarsUI() {{
        for (const indiv of indivs) {{
          const label = labelEls.get(indiv);
          if (!label) continue;
          label.style.opacity = selectedIndividuals.has(indiv) ? "1" : "{TIMELINE_UNSELECTED_ALPHA_SCALE}";
        }}
        for (const bar of barEls) {{
          const indiv = String(bar.dataset.individual);
          const split = String(bar.dataset.split);
          const splitOn = splitEnabled(split);
          if (!splitOn) {{
            bar.style.display = "none";
            continue;
          }}
          bar.style.display = "block";
          bar.style.opacity = selectedIndividuals.has(indiv) ? "1" : "{TIMELINE_UNSELECTED_ALPHA_SCALE}";
        }}
      }}

      // ---- slider ----
      const slider = document.getElementById("mb-slider");
      const timeEl = document.getElementById("mb-time");

      slider.min = String(globalMin);
      slider.max = String(globalMax);
      slider.value = String(currentTimeMs);
      slider.disabled = timesMs.length <= 1;
      timeEl.textContent = timesMs.length ? fmtTime(currentTimeMs) : "No timestamps";

      slider.addEventListener("input", (e) => {{
        const desiredMs = Number(e.target.value);
        if (!Number.isFinite(desiredMs)) return;
        currentTimeMs = Math.max(globalMin, Math.min(globalMax, desiredMs));
        timeEl.textContent = fmtTime(currentTimeMs);
        updateCursor();
        renderLayers();
        saveStateSoon();
      }});

      map.on("load", () => {{
        updateBarsUI();
        updateCursor();
        renderLayers();
      }});
    </script>
  </body>
</html>
"""

    components.html(html, height=total_h, scrolling=False)


def main() -> None:
    st.set_page_config(page_title="MoveBench Trajectory Viewer", layout="wide")

    study_ids = list_study_ids()
    if not study_ids:
        st.sidebar.error(
            "No studies found. "
            f"Expected study directories under: {paths.studies_root()}. "
            "Set MOVEBENCH_DATA_ROOT to your data root (the directory containing studies_10ind/)."
        )
        st.stop()

    study_id = st.sidebar.selectbox("Study", study_ids, key="study_id")
    df = load_study_df(study_id)

    view_state_study_id = st.session_state.get("_view_state_study_id")
    prior_view_state = st.session_state.get("_initial_view_state")
    if (
        view_state_study_id != study_id
        or not isinstance(prior_view_state, dict)
        or not {"latitude", "longitude", "zoom"}.issubset(prior_view_state.keys())
    ):
        st.session_state["_view_state_study_id"] = study_id
        st.session_state["_initial_view_state"] = compute_initial_view(df)
    view_state = st.session_state["_initial_view_state"]

    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    if not individuals:
        st.sidebar.error("No individuals found in trajectories.csv.")
        st.stop()

    study = Study(study_id)
    rasters_root = study.study_dir / "rasters"
    hash_dirs = [p for p in rasters_root.iterdir() if p.is_dir()] if rasters_root.exists() else []
    hash_dirs = sorted(hash_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    raster_hashes = [p.name for p in hash_dirs]
    dem_ready_hashes = [
        p.name for p in hash_dirs if (p / "tiles" / "dem" / "index.json").exists()
    ]
    if dem_ready_hashes:
        latest_hash = dem_ready_hashes[0]
    elif raster_hashes:
        latest_hash = raster_hashes[0]
    else:
        latest_hash = None

    if st.session_state.get("_dem_hash_study_id") != study_id:
        st.session_state["_dem_hash_study_id"] = study_id
        if latest_hash is not None:
            st.session_state["dem_raster_hash"] = latest_hash
    if "dem_overlay_enabled" not in st.session_state:
        st.session_state["dem_overlay_enabled"] = True

    train_palette, test_palette = _make_split_palettes(individuals, PATH_ALPHA)
    st.sidebar.caption(
        "Use the **Data availability** panel below the map to toggle individuals/splits without reloading."
    )

    # Build all paths; selection/filtering is handled client-side.
    rows = build_path_rows(df)

    # ---- sidebar controls ----
    basemap_options = list(BASEMAP_STYLES.keys())
    basemap_name = st.sidebar.radio("Basemap", basemap_options, index=0, key="basemap_name")
    map_style = BASEMAP_STYLES[basemap_name]

    raster_tiles: list[dict] = []
    with st.sidebar.expander("Raster overlay (DEM)", expanded=False):
        if not raster_hashes:
            st.info("No rasters found for this study.")
        else:
            st.checkbox("Show DEM overlay", key="dem_overlay_enabled")
            dem_hash = st.selectbox("Rasters hash", raster_hashes, key="dem_raster_hash")
            dem_style = st.selectbox(
                "Style",
                ["Grayscale", "Terrain colormap", "Hillshade"],
                index=2,
                key="dem_style",
            )
            dem_opacity = st.slider("Opacity", 0.0, 1.0, 0.1, 0.05, key="dem_opacity")

            if bool(st.session_state.get("dem_overlay_enabled", False)):
                dem_tiles_dir = rasters_root / str(dem_hash) / "tiles" / "dem"
                index_path = dem_tiles_dir / "index.json"
                if not index_path.exists():
                    st.warning(f"DEM tiles not found at: {dem_tiles_dir}")
                else:
                    idx = _load_tile_index(str(index_path))
                    tiles = idx.get("tiles", [])
                    tile_paths = tuple(sorted(str(dem_tiles_dir / t["file"]) for t in tiles if "file" in t))

                    vmin = vmax = None
                    if dem_style.strip().lower() != "hillshade":
                        vmin, vmax = _compute_global_vmin_vmax(tile_paths)

                    for i, t in enumerate(tiles):
                        file = t.get("file")
                        bounds = t.get("bounds") or {}
                        if not file:
                            continue

                        min_lon = bounds.get("min_lon")
                        max_lon = bounds.get("max_lon")
                        min_lat = bounds.get("min_lat")
                        max_lat = bounds.get("max_lat")
                        if min_lon is None or max_lon is None or min_lat is None or max_lat is None:
                            continue

                        tile_path = str(dem_tiles_dir / str(file))
                        if not Path(tile_path).exists():
                            continue

                        img_url = _tile_png_data_url(
                            tile_path,
                            style=dem_style,
                            vmin=vmin,
                            vmax=vmax,
                            max_px=RASTER_MAX_PX,
                        )
                        safe_style = "".join(ch if ch.isalnum() else "_" for ch in str(dem_style))
                        raster_tiles.append(
                            {
                                "id": f"dem_{dem_hash}_{safe_style}_{i}",
                                "image": img_url,
                                "bounds": [float(min_lon), float(min_lat), float(max_lon), float(max_lat)],
                                "opacity": float(dem_opacity),
                            }
                        )

    times_ms, points = _build_timeline_times_and_points(
        df, selected=set(individuals), enabled_splits=["train", "test"]
    )
    coverage = _build_individual_split_coverage_ms(df)
    indiv_stats = _build_individual_movement_stats(df)

    alpha = float(TIMELINE_BAR_ALPHA) / 255.0
    split_colors: dict[str, dict[str, str]] = {}
    for indiv in individuals:
        train_rgba = train_palette.get(indiv) or [0, 0, 0, 255]
        test_rgba = test_palette.get(indiv) or [0, 0, 0, 255]
        split_colors[indiv] = {
            "train": f"rgba({int(train_rgba[0])},{int(train_rgba[1])},{int(train_rgba[2])},{alpha:.3f})",
            "test": f"rgba({int(test_rgba[0])},{int(test_rgba[1])},{int(test_rgba[2])},{alpha:.3f})",
        }

    _render_deckgl_viewer(
        study_id=study_id,
        map_style=map_style,
        view_state=view_state,
        path_rows=rows,
        raster_tiles=raster_tiles,
        individuals=individuals,
        times_ms=times_ms,
        points=points,
        coverage=coverage,
        split_colors=split_colors,
        indiv_stats=indiv_stats,
    )


if __name__ == "__main__":
    main()
