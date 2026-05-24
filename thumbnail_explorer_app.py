"""Interactive thumbnail design explorer for cleaned raw movement CSVs.

Run:
  ~/micromamba/envs/movebench/bin/python -m streamlit run thumbnail_explorer_app.py
"""

from __future__ import annotations

import base64
import colorsys
import io
import json
import math
from textwrap import fill
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from movebench.datasets.thumbnails import (
    build_track_thumbnails,
    detect_color_columns,
    load_thumbnail_source_df,
    load_thumbnail_source_df_with_anomaly,
)
from movebench.thumbnail_basemaps import (
    DEFAULT_THUMBNAIL_BASEMAP,
    thumbnail_basemap_ids,
    thumbnail_basemap_label,
)
from movebench.thumbnail_config import (
    DEFAULT_BURST_GAP_QUANTILE,
    DEFAULT_BURST_GAP_SECONDS,
    build_thumbnail_generation_config,
    dumps_thumbnail_generation_config,
    normalize_anomaly_config,
    thumbnail_config_filename,
)
from movebench.thumbnail_pdf import build_thumbnail_pdf_bytes as _build_thumbnail_pdf_bytes
from movebench.thumbnail_pdf import class_color_palette as _pdf_class_color_palette
from movebench.thumbnail_pdf import format_class_name as _format_class_name
from movebench.thumbnail_pdf import primary_class as _pdf_primary_class
from movebench.trajectories import INDIVIDUAL_COL, LAT_COL, LON_COL, SET_COL, TIME_COL

BASEMAP_STYLES: dict[str, str] = {
    "Positron": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "Voyager": "https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    "Dark Matter": "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
}

PATH_ALPHA = 88
CHUNK_ALPHA = 165
ACTIVE_CHUNK_COLOR = [244, 98, 42, 255]
START_MARKER_COLOR = [39, 174, 96, 255]
END_MARKER_COLOR = [231, 76, 60, 255]
BACKGROUND_TRACK_COLORS = {
    "train": [109, 128, 148, PATH_ALPHA],
    "test": [144, 156, 170, PATH_ALPHA],
}
MAX_POINTS_PER_PATH = 2_000
GRID_COLS = 4
DEFAULT_SOURCE_CSV_ROOT = Path(
    "/Users/shirbar/Documents/PostDoc/codes/movebench/movement_raw"
).resolve()


def _uniform_downsample(path: list[list[float]], max_points: int) -> list[list[float]]:
    n = len(path)
    if n <= max_points:
        return path
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return [path[int(i)] for i in idx]


def _alternating_hue_order(n: int) -> list[int]:
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
        train[indiv] = _hls_to_rgba(h, 0.65, 0.90, alpha)
        test[indiv] = _hls_to_rgba(h, 0.35, 0.90, alpha)
    return train, test


def _temporal_color(idx: int, total: int, *, alpha: int = CHUNK_ALPHA) -> list[int]:
    if total <= 1:
        return [43, 108, 176, alpha]
    frac = idx / float(total - 1)
    hue = 0.60 - 0.52 * frac
    light = 0.56 - 0.08 * frac
    return _hls_to_rgba(hue, light, 0.72, alpha)


def _span_to_zoom(span_deg: float) -> float:
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
    return 10.0


def _source_label(csv_path: str) -> str:
    return Path(csv_path).name


def _source_labels(source_csvs: list[str], source_root: Path) -> dict[str, str]:
    counts: dict[str, int] = {}
    for csv_path in source_csvs:
        name = Path(csv_path).name
        counts[name] = counts.get(name, 0) + 1

    labels: dict[str, str] = {}
    for csv_path in source_csvs:
        path = Path(csv_path)
        label = path.name
        if counts[label] > 1:
            try:
                parent_label = str(path.parent.relative_to(source_root))
            except ValueError:
                parent_label = str(path.parent)
            if parent_label and parent_label != ".":
                label = f"{label} [{parent_label}]"
        labels[csv_path] = label
    return labels


def _source_stem(csv_path: str) -> str:
    return Path(csv_path).stem


def _resolve_source_root(source_root: str | Path) -> Path:
    return Path(str(source_root).strip()).expanduser().resolve()


@st.cache_data(show_spinner=False)
def list_source_csvs(source_root: str | Path) -> list[str]:
    root = _resolve_source_root(source_root)
    if not root.exists():
        print(f"Warning: source CSV root {root} does not exist")
        return []
    return [str(path) for path in sorted(root.rglob("*.csv")) if path.is_file()]


@st.cache_data(show_spinner=False)
def inspect_source_splits(csv_path: str) -> dict[str, object]:
    header = pd.read_csv(csv_path, nrows=0)
    if SET_COL not in header.columns:
        return {"has_split_column": False, "options": ["all"]}
    split_df = pd.read_csv(csv_path, usecols=[SET_COL], low_memory=False)
    splits = sorted({str(value) for value in split_df[SET_COL].dropna().tolist()})
    return {"has_split_column": True, "options": ["all"] + splits}


@st.cache_data(show_spinner=False)
def inspect_source(csv_path: str, split: str) -> dict:
    split_arg = None if split == "all" else split
    anomaly = normalize_anomaly_config(None)
    df = load_thumbnail_source_df_with_anomaly(
        csv_path,
        split=split_arg,
        include_movement_features=True,
        anomaly_config=anomaly,
    )
    return {
        "color_columns": detect_color_columns(df, extra_columns=[anomaly["score_column"]]),
        "n_rows": int(len(df)),
    }


@st.cache_data(show_spinner=False)
def load_map_df(csv_path: str, split: str) -> pd.DataFrame:
    split_arg = None if split == "all" else split
    df = load_thumbnail_source_df(csv_path, split=split_arg, include_movement_features=False)
    keep = [c for c in (INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL, SET_COL) if c in df.columns]
    return df[keep].sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort")


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
    span = max(max_lon - min_lon, max_lat - min_lat)
    return {
        "latitude": 0.5 * (min_lat + max_lat),
        "longitude": 0.5 * (min_lon + max_lon),
        "zoom": _span_to_zoom(span),
    }


@st.cache_data(show_spinner=False)
def build_full_paths(
    csv_path: str,
    split: str,
    *,
    focus_individual: str | None = None,
    context_mode: str = "active_individual",
    show_tracks: bool = True,
) -> tuple[list[dict], dict[str, float]]:
    df = load_map_df(csv_path, split)
    if context_mode == "none":
        empty_df = df.iloc[0:0].copy()
        return [], compute_initial_view(empty_df)
    if not show_tracks:
        return [], compute_initial_view(df)
    rows: list[dict] = []
    group_cols = [INDIVIDUAL_COL] + ([SET_COL] if SET_COL in df.columns else [])
    for group_key, g in df.groupby(group_cols, sort=False):
        if isinstance(group_key, tuple):
            individual_id = str(group_key[0])
            split_name = str(group_key[1]) if len(group_key) > 1 else split
        else:
            individual_id = str(group_key)
            split_name = split
        path = g[[LON_COL, LAT_COL]].to_numpy(dtype=np.float64, copy=False).tolist()
        if len(path) < 2:
            continue
        path = _uniform_downsample(path, MAX_POINTS_PER_PATH)
        color = BACKGROUND_TRACK_COLORS.get(split_name, [125, 135, 145, PATH_ALPHA])
        rows.append(
            {
                "path": path,
                "color": color,
                "individualId": individual_id,
                "split": split_name,
                "tooltip": f"{individual_id} | {split_name}",
            }
        )
    return rows, compute_initial_view(df)


def _read_generation_params() -> dict:
    stride_raw = int(st.session_state["draft_stride"])
    return {
        "img_size": int(st.session_state["draft_img_size"]),
        "chunk_mode": str(st.session_state["draft_chunk_mode"]),
        "chunk_size": int(st.session_state["draft_chunk_size"]),
        "stride": None if stride_raw <= 0 else stride_raw,
        "min_chunk_size": int(st.session_state["draft_min_chunk_size"]),
        "burst_gap_mode": str(st.session_state["draft_burst_gap_mode"]),
        "burst_gap_seconds": float(st.session_state["draft_burst_gap_seconds"]),
        "burst_gap_quantile": float(st.session_state["draft_burst_gap_quantile"]),
        "line_width": int(st.session_state["draft_line_width"]),
        "marker_radius": int(st.session_state["draft_marker_radius"]),
        "padding": int(st.session_state["draft_padding"]),
        "scale_mode": str(st.session_state["draft_scale_mode"]),
        "scale_quantile": float(st.session_state["draft_scale_quantile"]),
    }


@st.cache_data(show_spinner=False)
def build_thumbnail_bundle(
    csv_path: str,
    split: str,
    params_json: str,
    color_by: str | None,
    basemap: str,
) -> dict:
    params = json.loads(params_json)
    split_arg = None if split == "all" else split
    effective_basemap = DEFAULT_THUMBNAIL_BASEMAP if basemap == "osm" else basemap
    build = build_track_thumbnails(
        csv_path,
        split=split_arg,
        img_size=int(params["img_size"]),
        chunk_mode=str(params.get("chunk_mode", "fixed")),
        chunk_size=int(params["chunk_size"]),
        stride=params["stride"],
        min_chunk_size=int(params["min_chunk_size"]),
        burst_gap_mode=str(params.get("burst_gap_mode", "manual")),
        burst_gap_seconds=float(params.get("burst_gap_seconds", DEFAULT_BURST_GAP_SECONDS)),
        burst_gap_quantile=float(params.get("burst_gap_quantile", DEFAULT_BURST_GAP_QUANTILE)),
        line_width=int(params["line_width"]),
        marker_radius=int(params["marker_radius"]),
        padding=int(params["padding"]),
        scale_mode=str(params["scale_mode"]),
        scale_quantile=float(params["scale_quantile"]),
        color_by=color_by,
        basemap=effective_basemap,
    )
    meta = build.metadata.copy()
    if len(meta) == 0:
        return {
            "images": build.images,
            "metadata": meta,
            "chunk_paths": build.chunk_paths,
            "class_columns": [],
            "burst_gap": build.burst_gap,
        }

    meta["start_time"] = pd.to_datetime(meta["start_time"])
    meta["end_time"] = pd.to_datetime(meta["end_time"])
    class_columns = [c for c in ("clean", "has_any_outlier") if c in meta.columns]
    class_columns += sorted(c for c in meta.columns if c.startswith("has_") and c != "has_any_outlier")
    return {
        "images": build.images,
        "metadata": meta,
        "chunk_paths": build.chunk_paths,
        "class_columns": class_columns,
        "burst_gap": build.burst_gap,
    }


def format_burst_gap_quantile(value: object) -> str:
    quantile = float(value)
    percentile = quantile * 100.0
    decimals = 0 if abs(percentile - round(percentile)) < 1e-9 else 3
    formatted = f"{percentile:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return f"p{formatted}"


def format_burst_gap_summary(info: dict[str, object]) -> str:
    effective = float(info.get("effective_seconds", DEFAULT_BURST_GAP_SECONDS))
    gap_count = int(info.get("gap_count", 0))
    gap_phrase = f" from {gap_count:,} source gaps" if gap_count else " from 0 source gaps"
    if str(info.get("mode", "manual")) == "quantile":
        fallback = " (fallback)" if bool(info.get("used_fallback", False)) else ""
        return f"{format_burst_gap_quantile(info.get('quantile', DEFAULT_BURST_GAP_QUANTILE))} = {effective:,.2f} s{gap_phrase}{fallback}"
    return f"Manual = {effective:,.2f} s{gap_phrase}"


def format_class_name(col: str) -> str:
    return _format_class_name(col)


def _class_color_palette(class_columns: list[str]) -> dict[str, list[int]]:
    return _pdf_class_color_palette(class_columns)


def _primary_class(row: pd.Series, class_columns: list[str]) -> str:
    return _pdf_primary_class(row, class_columns)


def _image_to_data_url(img: np.ndarray) -> str:
    mode = "L" if img.ndim == 2 else "RGB"
    pil = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sort_class_indices(meta: pd.DataFrame, indices: list[int], sort_mode: str) -> list[int]:
    subset = meta.loc[indices].copy()
    if sort_mode == "Outlier fraction":
        subset = subset.sort_values(["outlier_frac", "start_time"], ascending=[False, True], kind="mergesort")
    elif sort_mode == "Individual then time":
        subset = subset.sort_values(["individual_id", "start_time"], ascending=[True, True], kind="mergesort")
    else:
        subset = subset.sort_values(["start_time", "individual_id"], ascending=[True, True], kind="mergesort")
    return subset.index.tolist()


def describe_chunk(row: pd.Series) -> str:
    split = row.get("split", "unknown")
    start = pd.Timestamp(row["start_time"]).strftime("%Y-%m-%d %H:%M")
    return (
        f"#{int(row['global_chunk_idx'])} | {row['individual_id']} | {split} | "
        f"{start} | outlier_frac={float(row.get('outlier_frac', 0.0)):.2f}"
    )


def _chunk_box_path(row: pd.Series) -> list[list[float]]:
    lon_min = float(row["lon_min"])
    lon_max = float(row["lon_max"])
    lat_min = float(row["lat_min"])
    lat_max = float(row["lat_max"])
    return [
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]


def build_chunk_box_rows(
    meta: pd.DataFrame,
    *,
    show_boxes: bool,
    focus_individual: str | None = None,
    context_mode: str = "active_individual",
) -> tuple[list[dict], list[int]]:
    if len(meta) == 0:
        return [], []
    ordered = meta.sort_values(["start_time", "individual_id", "chunk_idx"], kind="mergesort")
    if context_mode == "active_individual" and focus_individual is not None:
        ordered = ordered[ordered["individual_id"].astype(str) == str(focus_individual)].copy()
    box_rows: list[dict] = []
    if show_boxes:
        for _order_idx, (_, row) in enumerate(ordered.iterrows()):
            global_idx = int(row["global_chunk_idx"])
            box_rows.append(
                {
                    "path": _chunk_box_path(row),
                    "color": [46, 125, 50, 190] if bool(row.get("clean", False)) else [198, 40, 40, 190],
                    "tooltip": (
                        f"chunk #{global_idx} | {row['individual_id']} | {row['split']} | "
                        f"{pd.Timestamp(row['start_time']).strftime('%Y-%m-%d %H:%M')} | "
                        f"outlier_frac={float(row.get('outlier_frac', 0.0)):.2f}"
                    ),
                    "width": 2.0,
                }
            )
    return box_rows, ordered.index.tolist()


def build_active_chunk_payload(meta: pd.DataFrame, chunk_paths, active_idx: int | None) -> tuple[list[dict], list[dict]]:
    if active_idx is None or active_idx not in meta.index:
        return [], []

    row = meta.loc[active_idx]
    global_idx = int(row["global_chunk_idx"])
    path = [list(pt) for pt in chunk_paths[global_idx]]
    active_rows = [
        {
            "path": path,
            "color": ACTIVE_CHUNK_COLOR,
            "tooltip": describe_chunk(row),
        }
    ]
    marker_rows = [
        {
            "position": [float(row["start_lon"]), float(row["start_lat"])],
            "color": START_MARKER_COLOR,
            "tooltip": f"Start | {describe_chunk(row)}",
        },
        {
            "position": [float(row["end_lon"]), float(row["end_lat"])],
            "color": END_MARKER_COLOR,
            "tooltip": f"End | {describe_chunk(row)}",
        },
    ]
    return active_rows, marker_rows


def build_active_chunk_box(meta: pd.DataFrame, active_idx: int | None) -> list[dict]:
    if active_idx is None or active_idx not in meta.index:
        return []
    row = meta.loc[active_idx]
    return [
        {
            "path": _chunk_box_path(row),
            "color": [244, 98, 42, 255],
            "tooltip": f"Active chunk box | {describe_chunk(row)}",
            "width": 4.5,
        }
    ]


def render_interactive_browser(payload: dict, *, height: int = 760) -> None:
    payload_json = json.dumps(payload)
    html = f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <link href="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css" rel="stylesheet" />
    <style>
      html, body {{ margin: 0; padding: 0; background: transparent; }}
      body {{ font-family: ui-sans-serif, system-ui, sans-serif; color: #1f2933; }}
      .wrap {{
        display: grid;
        grid-template-columns: minmax(430px, 1.12fr) minmax(360px, 0.88fr);
        gap: 14px;
      }}
      .panel {{
        background: #ffffff;
        border: 1px solid rgba(31,41,51,0.10);
        border-radius: 14px;
        overflow: hidden;
      }}
      #map {{ width: 100%; height: {height}px; }}
      .side {{
        display: flex;
        flex-direction: column;
        height: {height}px;
      }}
      .side-main {{
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
      }}
      .active {{
        padding: 14px 16px 12px 16px;
        border-top: 1px solid rgba(31,41,51,0.10);
        background: linear-gradient(180deg, rgba(248,250,252,0.96), rgba(255,255,255,0.98));
      }}
      .active h3 {{ margin: 0 0 10px 0; font-size: 15px; }}
      .active-grid {{ display: grid; grid-template-columns: 110px 1fr; gap: 12px; }}
      .active-grid img {{
        width: 110px; height: 110px; border-radius: 10px; object-fit: contain; background: #050505;
      }}
      .active-meta {{ font-size: 13px; line-height: 1.45; }}
      .muted {{ color: #667085; }}
      .badge-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
      .badge {{
        display: inline-flex; align-items: center; padding: 3px 8px; border-radius: 999px;
        font-size: 11px; font-weight: 700; color: white;
      }}
      .filmstrip-wrap {{ flex: 1; min-height: 0; padding: 12px; display: flex; flex-direction: column; gap: 10px; }}
      .filmstrip-head {{
        display: flex; justify-content: space-between; gap: 12px; font-size: 12px; color: #667085;
      }}
      .filmstrip {{
        display: flex; gap: 10px; overflow-x: auto; overflow-y: hidden; padding-bottom: 2px; scroll-behavior: smooth;
      }}
      .card {{
        min-width: 128px; max-width: 128px; border-radius: 12px; border: 1px solid rgba(31,41,51,0.12);
        padding: 8px; background: white; cursor: pointer;
        transition: transform 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
      }}
      .card:hover {{ transform: translateY(-1px); }}
      .card.active {{ border-color: #f4622a; box-shadow: 0 0 0 2px rgba(244,98,42,0.18); }}
      .card.match {{ background: rgba(248,250,252,0.92); }}
      .swatch {{ height: 6px; border-radius: 999px; margin-bottom: 8px; }}
      .card img {{
        width: 100%; aspect-ratio: 1 / 1; border-radius: 8px; background: #050505; object-fit: contain;
      }}
      .card-title {{ margin-top: 7px; font-size: 11px; font-weight: 700; }}
      .card-sub {{ margin-top: 4px; font-size: 10px; line-height: 1.35; color: #667085; }}
      .legend {{ padding: 0 12px 12px 12px; font-size: 12px; color: #667085; }}
      .legend strong {{ color: #111827; }}
      @media (max-width: 980px) {{
        .wrap {{ grid-template-columns: 1fr; }}
        #map {{ height: 520px; }}
        .side {{ height: auto; }}
      }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="panel"><div id="map"></div></div>
      <div class="panel side">
        <div class="side-main">
          <div class="filmstrip-wrap">
            <div class="filmstrip-head">
              <span>Temporal Filmstrip</span>
              <span id="filmstrip-count"></span>
            </div>
            <div class="filmstrip" id="filmstrip"></div>
          </div>
          <div class="active">
            <h3>Active Chunk</h3>
            <div class="active-grid">
              <img id="active-thumb" alt="active thumbnail" />
              <div class="active-meta">
                <div id="active-title"></div>
                <div class="muted" id="active-time"></div>
                <div class="muted" id="active-stats"></div>
                <div class="badge-row" id="active-badges"></div>
              </div>
            </div>
          </div>
        </div>
        <div class="legend" id="legend"></div>
      </div>
    </div>
    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <script src="https://unpkg.com/@deck.gl/mapbox@8.9.33/dist.min.js"></script>
    <script>
      const DATA = {payload_json};
      const VIEW_KEY = String(DATA.viewKey || "movebench-thumbnail-view");

      function loadJSON(key) {{
        try {{
          const raw = window.localStorage.getItem(key);
          return raw ? JSON.parse(raw) : null;
        }} catch (_err) {{
          return null;
        }}
      }}

      function saveJSON(key, value) {{
        try {{
          window.localStorage.setItem(key, JSON.stringify(value));
        }} catch (_err) {{
        }}
      }}

      function safeNumber(value, fallback) {{
        const n = Number(value);
        return Number.isFinite(n) ? n : fallback;
      }}

      function rgba(arr) {{
        const [r, g, b, a] = arr || [120, 120, 120, 255];
        return `rgba(${{r}},${{g}},${{b}},${{(a / 255).toFixed(3)}})`;
      }}

      function escapeHtml(text) {{
        return String(text)
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;");
      }}

      const view = DATA.viewState || {{ latitude: 0, longitude: 0, zoom: 1 }};
      const storedView = loadJSON(VIEW_KEY) || {{}};
      const initCenter = Array.isArray(storedView.center) && storedView.center.length === 2
        ? [safeNumber(storedView.center[0], Number(view.longitude || 0)), safeNumber(storedView.center[1], Number(view.latitude || 0))]
        : [Number(view.longitude || 0), Number(view.latitude || 0)];
      const initZoom = safeNumber(storedView.zoom, Number(view.zoom || 1));
      const initBearing = safeNumber(storedView.bearing, 0);
      const initPitch = safeNumber(storedView.pitch, 0);

      const map = new maplibregl.Map({{
        container: "map",
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

      const chunks = Array.isArray(DATA.chunks) ? DATA.chunks : [];
      const chunkById = new Map(chunks.map((d) => [Number(d.id), d]));
      let activeId = Number(DATA.initialActiveId);
      if (!chunkById.has(activeId) && chunks.length) activeId = Number(chunks[0].id);

      const activeThumbEl = document.getElementById("active-thumb");
      const activeTitleEl = document.getElementById("active-title");
      const activeTimeEl = document.getElementById("active-time");
      const activeStatsEl = document.getElementById("active-stats");
      const activeBadgesEl = document.getElementById("active-badges");
      const filmstripEl = document.getElementById("filmstrip");
      const filmstripCountEl = document.getElementById("filmstrip-count");
      const legendEl = document.getElementById("legend");

      const MapboxOverlayCtor = deck && (deck.MapboxOverlay || (deck.mapbox && deck.mapbox.MapboxOverlay));
      const overlay = MapboxOverlayCtor ? new MapboxOverlayCtor({{
        interleaved: true,
        layers: [],
        getTooltip: ({{ object }}) => object && object.tooltip ? {{ text: object.tooltip }} : null,
      }}) : null;

      function selectChunk(id, scrollCard) {{
        activeId = Number(id);
        renderSide();
        renderLayers();
        if (scrollCard) {{
          const el = document.getElementById(`film-${{activeId}}`);
          if (el) el.scrollIntoView({{ behavior: "smooth", inline: "center", block: "nearest" }});
        }}
      }}

      function buildLayers() {{
        if (!overlay) return [];
        const layers = [];
        const active = chunkById.get(activeId);
        const activeIndividualId = active ? active.individualId : null;
        const contextMode = String(DATA.contextMode || "all_individuals");
        const fullPaths = Array.isArray(DATA.fullPaths) ? DATA.fullPaths : [];
        const visibleFullPaths = contextMode === "active_individual" && activeIndividualId
          ? fullPaths.filter((d) => d.individualId === activeIndividualId)
          : fullPaths;
        if (fullPaths.length) {{
          layers.push(new deck.PathLayer({{
            id: "full-paths",
            data: visibleFullPaths,
            getPath: (d) => d.path,
            getColor: (d) => d.color,
            widthMinPixels: 2,
            pickable: true,
          }}));
        }}
        if (chunks.length && Boolean(DATA.showChunkBoxes)) {{
          const visibleChunks = contextMode === "active_individual" && activeIndividualId
            ? chunks.filter((d) => d.individualId === activeIndividualId)
            : chunks;
          layers.push(new deck.PathLayer({{
            id: "chunk-boxes",
            data: visibleChunks,
            getPath: (d) => d.boxPath,
            getColor: (d) => d.boxColor,
            getWidth: (d) => Number(d.boxWidth || 2),
            widthMinPixels: 1.5,
            pickable: true,
            onClick: (info) => {{
              if (info && info.object) selectChunk(info.object.id, true);
            }},
          }}));
        }}
        if (active) {{
          if (Boolean(DATA.showChunkBoxes)) {{
            layers.push(new deck.PathLayer({{
              id: "active-chunk-box",
              data: [active],
              getPath: (d) => d.boxPath,
              getColor: [244, 98, 42, 255],
              getWidth: 4.2,
              widthMinPixels: 3,
              pickable: true,
            }}));
          }}
          layers.push(new deck.PathLayer({{
            id: "active-chunk",
            data: [active],
            getPath: (d) => d.path,
            getColor: [244, 98, 42, 255],
            widthMinPixels: 5,
            pickable: true,
          }}));
          layers.push(new deck.ScatterplotLayer({{
            id: "active-markers",
            data: [
              {{ position: active.startMarker, color: [39, 174, 96, 255], tooltip: `Start | ${{active.title}}` }},
              {{ position: active.endMarker, color: [231, 76, 60, 255], tooltip: `End | ${{active.title}}` }},
            ],
            getPosition: (d) => d.position,
            getFillColor: (d) => d.color,
            radiusMinPixels: 5,
            radiusMaxPixels: 8,
            stroked: true,
            getLineColor: [255, 255, 255, 240],
            lineWidthMinPixels: 1,
            pickable: true,
          }}));
        }}
        return layers;
      }}

      function renderLegend() {{
        legendEl.innerHTML =
          `<strong>Legend</strong><br/>` +
          `Background tracks use muted blue-gray tones so chunk classes stay visually distinct.<br/>` +
          `Chunk boxes are colored by primary class. Focus class: <strong>${{escapeHtml(DATA.focusClassLabel || "Focus")}}</strong>.`;
      }}

      function renderSide() {{
        const active = chunkById.get(activeId);
        if (!active) return;
        activeThumbEl.src = active.thumb;
        activeTitleEl.innerHTML = `<strong>${{escapeHtml(active.title)}}</strong>`;
        activeTimeEl.textContent = `${{active.startTime}} to ${{active.endTime}}`;
        activeStatsEl.textContent = `Fixes: ${{active.nFixes}} | outlier_frac=${{active.outlierFrac.toFixed(2)}} | primary=${{DATA.classLabels[active.primaryClass] || active.primaryClass}}`;
        activeBadgesEl.innerHTML = "";
        for (const cls of active.classes) {{
          const badge = document.createElement("span");
          badge.className = "badge";
          badge.style.background = rgba(DATA.classColors[cls] || [120, 120, 120, 255]);
          badge.textContent = DATA.classLabels[cls] || cls;
          activeBadgesEl.appendChild(badge);
        }}
        if (!active.classes.length) {{
          const badge = document.createElement("span");
          badge.className = "badge";
          badge.style.background = rgba(DATA.classColors.clean || [46, 125, 50, 255]);
          badge.textContent = DATA.classLabels.clean || "Clean";
          activeBadgesEl.appendChild(badge);
        }}

        filmstripEl.innerHTML = "";
        for (const chunk of chunks) {{
          const card = document.createElement("article");
          card.id = `film-${{chunk.id}}`;
          card.className = "card" + (chunk.id === activeId ? " active" : "") + (chunk.matchesFocus ? " match" : "");
          card.innerHTML = `
            <div class="swatch" style="background:${{rgba(chunk.primaryColor)}};"></div>
            <img src="${{chunk.thumb}}" alt="chunk ${{chunk.id}}" />
            <div class="card-title">#${{chunk.id}} | ${{chunk.individualId}}</div>
            <div class="card-sub">
              ${{chunk.startTime}}<br/>
              outlier_frac=${{chunk.outlierFrac.toFixed(2)}}<br/>
              ${{chunk.matchesFocus ? "focus match" : "not focus"}}
            </div>
          `;
          card.addEventListener("click", () => selectChunk(chunk.id, false));
          filmstripEl.appendChild(card);
        }}
        filmstripCountEl.textContent = `${{chunks.length}} chunks`;
      }}

      function renderLayers() {{
        if (!overlay) return;
        overlay.setProps({{ layers: buildLayers() }});
      }}

      map.on("load", () => {{
        if (!overlay) return;
        map.addControl(overlay);
        renderLegend();
        renderSide();
        renderLayers();
      }});
    </script>
  </body>
</html>
"""
    components.html(html, height=height + 28, scrolling=False)


def _safe_filename_part(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_")


def _ordered_pdf_indices(meta: pd.DataFrame, indices: list[int], order_mode: str) -> list[int]:
    if not indices:
        return []
    subset = meta.loc[indices].copy()
    if order_mode == "outlier_frac":
        subset = subset.sort_values(
            ["outlier_frac", "start_time", "individual_id", "chunk_idx"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
    else:
        subset = subset.sort_values(
            ["start_time", "individual_id", "chunk_idx"],
            ascending=[True, True, True],
            kind="mergesort",
        )
    return subset.index.tolist()


def _pdf_thumbnail_labels(row: pd.Series) -> str:
    label_cols = sorted(
        col
        for col in row.index
        if col.startswith("has_") and col != "has_any_outlier" and bool(row.get(col, False))
    )
    if not label_cols:
        return "clean"
    return ", ".join(format_class_name(col) for col in label_cols)


def _pdf_class_columns(meta: pd.DataFrame) -> list[str]:
    cols = [c for c in ("clean", "has_any_outlier") if c in meta.columns]
    cols += sorted(c for c in meta.columns if c.startswith("has_") and c != "has_any_outlier")
    return cols


def build_thumbnail_pdf_bytes(
    *,
    source_name: str,
    split: str,
    export_label: str,
    images: np.ndarray,
    meta: pd.DataFrame,
    indices: list[int],
    params: dict,
    order_mode: str,
    cols: int = 5,
    rows: int = 4,
) -> tuple[bytes, str]:
    return _build_thumbnail_pdf_bytes(
        source_name=source_name,
        split=split,
        export_label=export_label,
        images=images,
        meta=meta,
        indices=indices,
        params=params,
        order_mode=order_mode,
        cols=cols,
        rows=rows,
    )


@st.cache_data(show_spinner="Building PDF...")
def build_thumbnail_pdf_cached(
    *,
    source_name: str,
    split: str,
    export_label: str,
    images: np.ndarray,
    meta: pd.DataFrame,
    indices: tuple[int, ...],
    params_json: str,
    order_mode: str,
) -> tuple[bytes, str]:
    return build_thumbnail_pdf_bytes(
        source_name=source_name,
        split=split,
        export_label=export_label,
        images=images,
        meta=meta,
        indices=list(indices),
        params=json.loads(params_json),
        order_mode=order_mode,
    )


def _sync_filmstrip_page_from_input(max_page: int) -> None:
    st.session_state["filmstrip_page_value"] = min(
        max_page,
        max(1, int(st.session_state["filmstrip_page_input"])),
    )


def _step_filmstrip_page(delta: int, max_page: int) -> None:
    current = int(st.session_state.get("filmstrip_page_value", 1))
    st.session_state["filmstrip_page_value"] = min(max_page, max(1, current + delta))
    st.session_state["filmstrip_page_input"] = st.session_state["filmstrip_page_value"]


def ensure_generation_state() -> None:
    draft_defaults = {
        "img_size": 64,
        "chunk_mode": "fixed",
        "chunk_size": 50,
        "stride": 0,
        "min_chunk_size": 10,
        "burst_gap_mode": "quantile",
        "burst_gap_seconds": DEFAULT_BURST_GAP_SECONDS,
        "burst_gap_quantile": DEFAULT_BURST_GAP_QUANTILE,
        "line_width": 1,
        "marker_radius": 1,
        "padding": 2,
        "scale_mode": "individual",
        "scale_quantile": 0.5,
    }
    if "applied_generation_params" not in st.session_state:
        st.session_state["applied_generation_params"] = {
            **draft_defaults,
            "stride": None,
        }
    applied = st.session_state["applied_generation_params"]
    for key, value in draft_defaults.items():
        if key == "stride":
            value = 0 if applied.get("stride") in (None, 0) else int(applied["stride"])
        else:
            value = applied.get(key, value)
        st.session_state.setdefault(f"draft_{key}", value)


def render_thumbnail_grid(
    images: np.ndarray,
    meta: pd.DataFrame,
    indices: list[int],
    *,
    active_idx: int | None,
    focus_class: str | None = None,
    matched_indices: set[int] | None = None,
) -> None:
    matched_indices = matched_indices or set()
    for start in range(0, len(indices), GRID_COLS):
        cols = st.columns(GRID_COLS)
        for col, idx in zip(cols, indices[start : start + GRID_COLS]):
            row = meta.loc[idx]
            with col:
                if focus_class is not None:
                    is_match = idx in matched_indices
                    tag = f"Focus match: {format_class_name(focus_class)}" if is_match else f"Not in {format_class_name(focus_class)}"
                    st.caption(tag)
                st.image(images[idx], clamp=True, width="stretch")
                st.caption(describe_chunk(row))
                if active_idx == idx:
                    st.caption("Selected chunk")
                if st.button("Select", key=f"select_chunk_{idx}", width="stretch"):
                    st.session_state["active_chunk_idx"] = int(idx)


def main() -> None:
    st.set_page_config(page_title="MoveBench Thumbnail Explorer", layout="wide")
    ensure_generation_state()

    st.title("MoveBench Raw CSV Thumbnail Explorer")
    st.caption(
        "Explore chunk thumbnails from movement CSVs, including split-aware and unsplit files, inspect chunk overlays on a world map, and export class grids to PDF."
    )

    top_controls = st.container()
    with top_controls:
        st.session_state.setdefault("source_csv_root", str(DEFAULT_SOURCE_CSV_ROOT))
        source_root_input = st.text_input(
            "CSV root folder",
            key="source_csv_root",
            help="Choose the folder whose CSV files you want to browse.",
        )
    source_root_value = str(source_root_input).strip()
    if not source_root_value:
        st.error("Enter a CSV root folder to browse source files.")
        return

    source_root = _resolve_source_root(source_root_value)
    if not source_root.exists():
        st.error(f"CSV root folder does not exist: {source_root}")
        return
    if not source_root.is_dir():
        st.error(f"CSV root folder is not a directory: {source_root}")
        return

    source_csvs = list_source_csvs(source_root)
    if not source_csvs:
        st.error(f"No CSV files found under {source_root}.")
        return
    source_labels = _source_labels(source_csvs, source_root)

    with top_controls:
        current_source = st.session_state.get("source_csv")
        if current_source not in source_csvs:
            st.session_state["source_csv"] = source_csvs[0]
            current_source = source_csvs[0]
        c1, c2, c3, c4, c5, c6 = st.columns([1.55, 0.95, 0.95, 1.15, 0.9, 0.95], gap="medium")
        with c1:
            source_csv = st.selectbox(
                "CSV file",
                source_csvs,
                index=source_csvs.index(current_source),
                format_func=lambda value: source_labels.get(value, _source_label(value)),
                key="source_csv",
            )
        split_info = inspect_source_splits(source_csv)
        split_options = list(split_info["options"])
        has_split_column = bool(split_info["has_split_column"])
        default_split = "train" if "train" in split_options else "all"
        current_split = st.session_state.get("split_choice", default_split)
        if current_split not in split_options:
            current_split = default_split
            st.session_state["split_choice"] = current_split
        with c2:
            if has_split_column:
                split = st.selectbox("Split", split_options, index=split_options.index(current_split), key="split_choice")
            else:
                split = "all"
                st.session_state["split_choice"] = split
                st.caption("No `set` column detected; using all rows.")
        with c3:
            basemap_name = st.selectbox("Basemap", list(BASEMAP_STYLES.keys()), index=0, key="basemap_name")
        with c4:
            map_context = st.selectbox(
                "Map context",
                ["active_individual", "all_individuals", "none"],
                index=0,
                format_func=lambda v: {
                    "active_individual": "Selected individual only",
                    "all_individuals": "All individuals",
                    "none": "No background tracks",
                }[v],
                key="map_context_mode",
            )
        with c5:
            show_chunk_boxes = st.checkbox("Show chunk boxes", value=True, key="show_chunk_boxes")
        with c6:
            show_full_tracks = st.checkbox("Show full tracks", value=True, key="show_full_tracks")

    source_name = _source_stem(source_csv)
    source_info = inspect_source(source_csv, split)
    color_columns = source_info["color_columns"]
    appearance_controls = st.container()
    with appearance_controls:
        a1, a2, a3, a4 = st.columns([1.0, 1.35, 1.15, 2.1], gap="medium")
        render_modes = ["Black & white", "Color by variable"]
        current_render_mode = st.session_state.get("thumbnail_render_mode", render_modes[0])
        if current_render_mode not in render_modes:
            current_render_mode = render_modes[0]
            st.session_state["thumbnail_render_mode"] = current_render_mode
        with a1:
            render_mode = st.selectbox(
                "Thumbnail style",
                render_modes,
                index=render_modes.index(current_render_mode),
                key="thumbnail_render_mode",
            )
        current_color = st.session_state.get("thumbnail_color_by")
        if color_columns and current_color not in color_columns:
            current_color = color_columns[0]
            st.session_state["thumbnail_color_by"] = current_color
        with a2:
            if color_columns:
                if render_mode == "Color by variable":
                    selected_color = st.selectbox(
                        "Color variable",
                        color_columns,
                        index=color_columns.index(current_color or color_columns[0]),
                        key="thumbnail_color_by",
                    )
                else:
                    selected_color = current_color or color_columns[0]
                    st.caption(f"{len(color_columns)} numeric movement/GPS variables are available for coloring.")
            else:
                selected_color = None
                st.selectbox("Color variable", ["No valid variables"], index=0, disabled=True)
        thumbnail_basemap_options = [value for value in thumbnail_basemap_ids(include_none=True) if value != "osm"]
        current_thumbnail_basemap = st.session_state.get("thumbnail_basemap", DEFAULT_THUMBNAIL_BASEMAP)
        if current_thumbnail_basemap not in thumbnail_basemap_options:
            current_thumbnail_basemap = DEFAULT_THUMBNAIL_BASEMAP
            st.session_state["thumbnail_basemap"] = current_thumbnail_basemap
        with a3:
            thumbnail_basemap = st.selectbox(
                "Thumbnail basemap",
                thumbnail_basemap_options,
                index=thumbnail_basemap_options.index(current_thumbnail_basemap),
                format_func=thumbnail_basemap_label,
                key="thumbnail_basemap",
            )
        with a4:
            style_label = "black-and-white" if render_mode == "Black & white" or selected_color is None else f"colored by `{selected_color}`"
            st.caption(
                f"Source: `{source_name}` | split: `{split}` | rendering: {style_label} | "
                f"basemap: `{thumbnail_basemap}` | rows after filtering: {source_info['n_rows']:,}"
            )
    color_by = selected_color if render_mode == "Color by variable" and selected_color is not None else None
    if render_mode == "Color by variable" and not color_columns:
        st.caption("No valid movement or numeric `gps:*` columns are available for coloring in the current file/split.")

    apply_clicked = False
    with st.expander("Generation settings", expanded=False):
        g1, g2, g3, g4, g5, g6 = st.columns(6, gap="medium")
        with g1:
            st.number_input("Image size", min_value=16, max_value=256, step=8, key="draft_img_size")
            st.radio("Chunk mode", ["fixed", "burst"], key="draft_chunk_mode")
        with g2:
            burst_mode = st.session_state.get("draft_chunk_mode") == "burst"
            st.number_input("Chunk size", min_value=2, max_value=500, step=1, key="draft_chunk_size", disabled=burst_mode)
            st.number_input("Stride (0 = chunk size)", min_value=0, max_value=500, step=1, key="draft_stride", disabled=burst_mode)
        with g3:
            st.number_input("Minimum chunk size", min_value=1, max_value=500, step=1, key="draft_min_chunk_size")
            st.radio(
                "Burst gap mode",
                ["quantile", "manual"],
                key="draft_burst_gap_mode",
                format_func=lambda value: "Quantile" if value == "quantile" else "Manual seconds",
            )
        with g4:
            burst_gap_mode = st.session_state.get("draft_burst_gap_mode", "quantile")
            st.number_input(
                "Fallback seconds" if burst_gap_mode == "quantile" else "Burst gap seconds",
                min_value=1.0,
                max_value=604800.0,
                step=300.0,
                key="draft_burst_gap_seconds",
            )
            st.number_input(
                "Burst gap quantile",
                min_value=0.001,
                max_value=1.0,
                step=0.001,
                format="%.3f",
                key="draft_burst_gap_quantile",
                disabled=burst_gap_mode != "quantile",
            )
        with g5:
            st.number_input("Line width", min_value=1, max_value=8, step=1, key="draft_line_width")
            st.number_input("Marker radius", min_value=0, max_value=8, step=1, key="draft_marker_radius")
        with g6:
            st.number_input("Padding", min_value=0, max_value=16, step=1, key="draft_padding")
            st.radio("Scale mode", ["individual", "chunk"], key="draft_scale_mode")
            st.slider("Scale quantile", min_value=0.05, max_value=0.95, step=0.05, key="draft_scale_quantile")
            apply_clicked = st.button("Apply", width="stretch", type="primary")

    if apply_clicked:
        st.session_state["applied_generation_params"] = _read_generation_params()

    params = st.session_state["applied_generation_params"]
    effective_params = {**params, "basemap": thumbnail_basemap}
    export_config = build_thumbnail_generation_config(
        csv_path=source_csv,
        split=split,
        params=params,
        color_by=color_by,
        basemap=thumbnail_basemap,
    )
    export_config_json = dumps_thumbnail_generation_config(export_config)
    export_config_name = thumbnail_config_filename(export_config)
    params_json = json.dumps(effective_params, sort_keys=True)
    signature = (source_csv, split, params_json, color_by, thumbnail_basemap)
    if st.session_state.get("_thumbnail_signature") != signature:
        st.session_state["_thumbnail_signature"] = signature
        st.session_state["active_chunk_idx"] = None
        st.session_state["selected_class"] = None
        st.session_state["selected_individual"] = None

    bundle = build_thumbnail_bundle(source_csv, split, params_json, color_by, thumbnail_basemap)
    images = bundle["images"]
    meta = bundle["metadata"]
    chunk_paths = bundle["chunk_paths"]
    class_columns = bundle["class_columns"]
    burst_gap = bundle["burst_gap"]
    st.caption(f"Burst threshold: {format_burst_gap_summary(burst_gap)}")

    if len(meta) == 0:
        st.warning("No thumbnail chunks were produced for the current file/split/parameter combination.")
        return

    browse_controls = st.container()
    with browse_controls:
        b1, b2, b3, b4, b5 = st.columns([1.15, 1.1, 1.1, 0.95, 0.7], gap="medium")
        with b1:
            selected_class = st.selectbox(
                "Class focus",
                class_columns,
                index=0 if st.session_state.get("selected_class") not in class_columns else class_columns.index(st.session_state["selected_class"]),
                format_func=format_class_name,
            )
            st.session_state["selected_class"] = selected_class
        with b2:
            class_display_mode = st.selectbox(
                "Class handling",
                ["Highlight in temporal grid", "Filter to class only"],
                index=0,
            )
        with b3:
            sort_mode = st.selectbox("Sort chunks", ["Time", "Outlier fraction", "Individual then time"])
        with b4:
            page_size = int(st.select_slider("Filmstrip size", options=[8, 12, 16, 24, 32], value=16))
        with b5:
            st.caption("PDF export")

    focus_match_set = set(meta.index[meta[selected_class].fillna(False).astype(bool)].tolist())
    ordered_indices = sort_class_indices(meta, meta.index.tolist(), sort_mode)
    if not focus_match_set:
        st.warning(f"No chunks available for class: {format_class_name(selected_class)}")
        return

    individual_options = ["All individuals"] + sorted(meta["individual_id"].astype(str).unique().tolist())
    current_individual = st.session_state.get("selected_individual")
    if current_individual not in individual_options:
        current_individual = "All individuals"
    c5, c6 = st.columns([1.6, 1.0], gap="medium")
    with c5:
        selected_individual = st.selectbox("Individual", individual_options, index=individual_options.index(current_individual))
        st.session_state["selected_individual"] = selected_individual
    if selected_individual == "All individuals":
        visible_indices_all = ordered_indices
    else:
        visible_indices_all = [idx for idx in ordered_indices if str(meta.loc[idx, "individual_id"]) == selected_individual]
    if not visible_indices_all:
        visible_indices_all = ordered_indices
        st.session_state["selected_individual"] = "All individuals"
        selected_individual = "All individuals"

    if class_display_mode == "Filter to class only":
        visible_indices = [idx for idx in visible_indices_all if idx in focus_match_set]
    else:
        visible_indices = visible_indices_all
    if not visible_indices:
        visible_indices = visible_indices_all
        class_display_mode = "Highlight in temporal grid"

    active_idx = st.session_state.get("active_chunk_idx")
    if active_idx not in visible_indices:
        active_idx = visible_indices[0]
        st.session_state["active_chunk_idx"] = int(active_idx)
    active_individual = str(meta.loc[active_idx, "individual_id"])

    max_page = max(1, math.ceil(len(visible_indices) / float(page_size)))
    page_signature = (
        source_csv,
        split,
        selected_class,
        class_display_mode,
        sort_mode,
        selected_individual,
        page_size,
        len(visible_indices),
    )
    if st.session_state.get("_filmstrip_page_signature") != page_signature:
        active_page = 1 + (visible_indices.index(active_idx) // page_size)
        st.session_state["_filmstrip_page_signature"] = page_signature
        st.session_state["filmstrip_page_value"] = min(max_page, max(1, active_page))
        st.session_state["filmstrip_page_input"] = st.session_state["filmstrip_page_value"]
    else:
        st.session_state["filmstrip_page_value"] = min(
            max_page,
            max(1, int(st.session_state.get("filmstrip_page_value", 1))),
        )
        st.session_state.setdefault("filmstrip_page_input", st.session_state["filmstrip_page_value"])
    with c6:
        p1, p2, p3 = st.columns([0.8, 1.1, 0.8], gap="small")
        with p1:
            st.button(
                f"Prev {page_size}",
                width="stretch",
                disabled=st.session_state["filmstrip_page_value"] <= 1,
                on_click=_step_filmstrip_page,
                args=(-1, max_page),
            )
        with p2:
            page = int(
                st.number_input(
                    "Filmstrip page",
                    min_value=1,
                    max_value=max_page,
                    step=1,
                    key="filmstrip_page_input",
                    on_change=_sync_filmstrip_page_from_input,
                    args=(max_page,),
                )
            )
        with p3:
            st.button(
                f"Next {page_size}",
                width="stretch",
                disabled=st.session_state["filmstrip_page_value"] >= max_page,
                on_click=_step_filmstrip_page,
                args=(1, max_page),
            )
    page = int(st.session_state["filmstrip_page_value"])
    start = (page - 1) * page_size
    page_indices = visible_indices[start : start + page_size]
    if active_idx not in page_indices and page_indices:
        active_idx = page_indices[0]
        st.session_state["active_chunk_idx"] = int(active_idx)
        active_individual = str(meta.loc[active_idx, "individual_id"])

    st.caption(
        f"Showing {len(page_indices)} chunks on page {page} of {max_page}; "
        f"{len([idx for idx in page_indices if idx in focus_match_set])} match {format_class_name(selected_class)}."
    )
    if selected_individual == "All individuals" and len(visible_indices) > page_size:
        st.caption("Choose an individual to get a continuous per-animal filmstrip and less visual clutter.")

    focused_individual = active_individual if selected_individual == "All individuals" else selected_individual
    map_rows, view_state = build_full_paths(
        source_csv,
        split,
        focus_individual=focused_individual,
        context_mode=map_context,
        show_tracks=show_full_tracks,
    )
    class_colors = _class_color_palette(class_columns)
    class_labels = {col: format_class_name(col) for col in class_columns}
    browser_chunks: list[dict] = []
    for idx in page_indices:
        row = meta.loc[idx]
        positive_classes = [col for col in class_columns if bool(row.get(col, False))]
        primary_class = _primary_class(row, class_columns)
        primary_color = class_colors.get(primary_class, [120, 120, 120, 210])
        box_alpha = 230 if idx in focus_match_set else 150
        browser_chunks.append(
            {
                "id": int(idx),
                "globalChunkIdx": int(row["global_chunk_idx"]),
                "title": describe_chunk(row),
                "individualId": str(row["individual_id"]),
                "startTime": pd.Timestamp(row["start_time"]).strftime("%Y-%m-%d %H:%M"),
                "endTime": pd.Timestamp(row["end_time"]).strftime("%Y-%m-%d %H:%M"),
                "nFixes": int(row.get("n_fixes", 0)),
                "outlierFrac": float(row.get("outlier_frac", 0.0)),
                "classes": positive_classes,
                "primaryClass": primary_class,
                "primaryColor": primary_color,
                "matchesFocus": bool(idx in focus_match_set),
                "path": [list(pt) for pt in chunk_paths[int(row["global_chunk_idx"])]],
                "boxPath": _chunk_box_path(row),
                "boxColor": primary_color[:3] + [box_alpha],
                "boxWidth": 3.0 if idx in focus_match_set else 2.0,
                "startMarker": [float(row["start_lon"]), float(row["start_lat"])],
                "endMarker": [float(row["end_lon"]), float(row["end_lat"])],
                "thumb": _image_to_data_url(images[idx]),
            }
        )

    st.subheader("Map And Filmstrip")
    render_interactive_browser(
        {
            "mapStyle": BASEMAP_STYLES[basemap_name],
            "viewState": view_state,
            "viewKey": f"movebench-thumbnail-map::{source_name}::{split}::{map_context}",
            "contextMode": map_context,
            "showChunkBoxes": bool(show_chunk_boxes),
            "fullPaths": map_rows,
            "chunks": browser_chunks,
            "initialActiveId": int(active_idx),
            "focusClassLabel": format_class_name(selected_class),
            "classLabels": class_labels,
            "classColors": class_colors,
        }
    )

    pdf_section = st.container()
    with pdf_section:
        st.subheader("PDF Export")
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.markdown("**Class PDF**")
            export_indices = tuple(idx for idx in visible_indices_all if idx in focus_match_set)
            class_pdf_key = (
                "class",
                source_name,
                split,
                selected_class,
                export_indices,
                params_json,
                "temporal",
                color_by,
            )
            if st.button("Prepare class PDF", key="prepare_class_pdf", width="content", disabled=len(export_indices) == 0):
                st.session_state["prepared_class_pdf_key"] = class_pdf_key
            if len(export_indices) == 0:
                st.caption("No chunks in the current temporal scope match the focused class for PDF export.")
            elif st.session_state.get("prepared_class_pdf_key") == class_pdf_key:
                pdf_bytes, pdf_name = build_thumbnail_pdf_cached(
                    source_name=source_name,
                    split=split,
                    export_label=format_class_name(selected_class),
                    images=images,
                    meta=meta,
                    indices=export_indices,
                    params_json=params_json,
                    order_mode="temporal",
                )
                st.download_button(
                    "Download class PDF",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    width="stretch",
                    key="download_class_pdf",
                )
            else:
                st.caption("Class PDF export is generated on demand so browsing stays responsive.")

        with e2:
            st.markdown("**Individual PDF**")
            individual_export_options = sorted(meta["individual_id"].astype(str).unique().tolist())
            current_export_individual = st.session_state.get("pdf_individual")
            if current_export_individual not in individual_export_options:
                current_export_individual = selected_individual if selected_individual != "All individuals" else individual_export_options[0]
            pdf_individual = st.selectbox(
                "Individual to print",
                individual_export_options,
                index=individual_export_options.index(current_export_individual),
                key="pdf_individual",
            )
            pdf_order_mode = st.selectbox(
                "Thumbnail order",
                ["temporal", "outlier_frac"],
                index=0,
                format_func=lambda value: {
                    "temporal": "Temporal order",
                    "outlier_frac": "Outlier fraction: low to high",
                }[value],
                key="pdf_order_mode",
            )
            individual_export_indices = tuple(
                idx for idx in ordered_indices if str(meta.loc[idx, "individual_id"]) == pdf_individual
            )
            individual_pdf_key = (
                "individual",
                source_name,
                split,
                pdf_individual,
                pdf_order_mode,
                individual_export_indices,
                params_json,
                color_by,
            )
            if st.button(
                "Prepare individual PDF",
                key="prepare_individual_pdf",
                width="content",
                disabled=len(individual_export_indices) == 0,
            ):
                st.session_state["prepared_individual_pdf_key"] = individual_pdf_key
            if len(individual_export_indices) == 0:
                st.caption("No chunks are available for the selected individual under the current file and split.")
            elif st.session_state.get("prepared_individual_pdf_key") == individual_pdf_key:
                pdf_bytes, pdf_name = build_thumbnail_pdf_cached(
                    source_name=source_name,
                    split=split,
                    export_label=f"individual_{pdf_individual}",
                    images=images,
                    meta=meta,
                    indices=individual_export_indices,
                    params_json=params_json,
                    order_mode=pdf_order_mode,
                )
                st.download_button(
                    "Download individual PDF",
                    data=pdf_bytes,
                    file_name=pdf_name,
                    mime="application/pdf",
                    width="stretch",
                    key="download_individual_pdf",
                )
            else:
                st.caption("Choose an individual and ordering, then prepare the PDF.")

    bottom_stats = st.container()
    with bottom_stats:
        summary_rows = [
            {"class": format_class_name(col), "count": int(meta[col].fillna(False).astype(bool).sum())}
            for col in class_columns
        ]
        summary_df = pd.DataFrame(summary_rows).sort_values(["count", "class"], ascending=[False, True], kind="mergesort")
        s1, s2 = st.columns([1.0, 1.2], gap="medium")
        with s1:
            st.subheader("Class Summary")
            st.dataframe(summary_df, hide_index=True, width="stretch")
        with s2:
            st.subheader("Current View")
            current_view_df = pd.DataFrame(
                [
                    {"field": "selected csv", "value": str(_source_label(source_csv))},
                    {"field": "selected individual", "value": str(selected_individual)},
                    {"field": "class handling", "value": str(class_display_mode)},
                    {"field": "focus class", "value": str(format_class_name(selected_class))},
                    {"field": "thumbnail style", "value": "Black & white" if color_by is None else f"Color by {color_by}"},
                    {"field": "thumbnail basemap", "value": thumbnail_basemap_label(thumbnail_basemap)},
                    {"field": "burst threshold", "value": format_burst_gap_summary(burst_gap)},
                    {"field": "split", "value": str(split)},
                    {"field": "filmstrip page", "value": f"{page} / {max_page}"},
                    {"field": "chunks on page", "value": str(len(page_indices))},
                    {"field": "focus-class matches on page", "value": str(len([idx for idx in page_indices if idx in focus_match_set]))},
                    {"field": "map context", "value": str({"active_individual": "Selected individual only", "all_individuals": "All individuals", "none": "No background tracks"}[map_context])},
                ]
            )
            st.dataframe(
                current_view_df,
                hide_index=True,
                width="stretch",
            )
            st.caption("Live active chunk details now stay inside the map-and-filmstrip panel so map clicks update instantly.")

    export_section = st.container()
    with export_section:
        st.subheader("Export Config")
        st.caption("This JSON captures the currently applied thumbnail settings so the notebook can rebuild the exact same thumbnails.")
        st.code(export_config_json, language="json")
        st.download_button(
            "Download thumbnail config",
            data=export_config_json,
            file_name=export_config_name,
            mime="application/json",
            width="content",
        )


if __name__ == "__main__":
    main()
