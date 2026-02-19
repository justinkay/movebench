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
) -> tuple[list[str], list[dict]]:
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
    time_to_idx = {t: i for i, t in enumerate(times_list)}

    points: list[dict] = []
    for indiv, split, t, lon, lat in sdf[[INDIVIDUAL_COL, SET_COL, TIME_COL, LON_COL, LAT_COL]].itertuples(
        index=False, name=None
    ):
        ts = pd.Timestamp(t)
        idx = time_to_idx.get(ts)
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

    times_iso = [t.isoformat() for t in times_list]
    return times_iso, points


def _build_individual_coverage_ms(df: pd.DataFrame, *, enabled_splits: list[str]) -> list[dict]:
    plot_df = df
    if enabled_splits:
        plot_df = plot_df[plot_df[SET_COL].isin(enabled_splits)]

    ranges = plot_df.groupby(INDIVIDUAL_COL, sort=False)[TIME_COL].agg(start="min", end="max")
    out: list[dict] = []
    for indiv, r in ranges.iterrows():
        start = pd.Timestamp(r["start"])
        end = pd.Timestamp(r["end"])
        if pd.isna(start) or pd.isna(end):
            continue
        out.append(
            {
                "individual": str(indiv),
                "start_ms": int(start.value // 1_000_000),
                "end_ms": int(end.value // 1_000_000),
            }
        )
    return out


def _render_deckgl_viewer(
    *,
    map_style: str,
    view_state: dict[str, float],
    path_rows: list[dict],
    raster_tiles: list[dict],
    individuals: list[str],
    selected: set[str],
    times: list[str],
    points: list[dict],
    coverage: list[dict],
    bar_colors: dict[str, str],
) -> None:
    map_h = 600
    bars_h = min(260, max(120, 22 * max(1, len(individuals))))
    total_h = int(map_h + bars_h + 140)

    payload = {
        "mapStyle": map_style,
        "viewState": view_state,
        "paths": path_rows,
        "rasterTiles": raster_tiles,
        "individuals": individuals,
        "selected": sorted(selected),
        "times": times,
        "points": points,
        "coverage": coverage,
        "barColors": bar_colors,
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
      #mb-map {{
        width: 100%;
        height: {map_h}px;
        border-radius: 10px;
        overflow: hidden;
      }}
      .mb-timeline {{
        padding: 10px 2px 0 2px;
      }}
      .mb-title {{
        font-weight: 600;
        font-size: 14px;
        margin: 2px 0 8px 0;
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
        align-items: center;
        gap: 10px;
        margin: 6px 0;
      }}
      .mb-label {{
        width: 170px;
        font-size: 12px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
      .mb-track {{
        flex: 1;
        height: 10px;
        background: rgba(49, 51, 63, 0.08);
        border-radius: 999px;
        position: relative;
      }}
      .mb-bar {{
        position: absolute;
        height: 100%;
        border-radius: 999px;
      }}
      .mb-slider-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px 2px 0 2px;
      }}
      #mb-slider {{
        flex: 1;
      }}
      #mb-time {{
        font-size: 12px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        white-space: nowrap;
      }}
    </style>
  </head>
  <body>
    <div class="mb-wrap">
      <div id="mb-map"></div>
      <div class="mb-timeline">
        <div class="mb-title">Data availability</div>
        <div id="mb-bars"></div>
        <div class="mb-slider-row">
          <input id="mb-slider" type="range" min="0" step="1" />
          <div id="mb-time"></div>
        </div>
      </div>
    </div>

    <script src="https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js"></script>
    <script src="https://unpkg.com/deck.gl@8.9.33/dist.min.js"></script>
    <script src="https://unpkg.com/@deck.gl/mapbox@8.9.33/dist.min.js"></script>
    <script>
      const DATA = {payload_json};

      function safeNumber(v, fallback) {{
        const n = Number(v);
        return Number.isFinite(n) ? n : fallback;
      }}

      const map = new maplibregl.Map({{
        container: "mb-map",
        style: DATA.mapStyle,
        center: [
          safeNumber(DATA.viewState && DATA.viewState.longitude, 0),
          safeNumber(DATA.viewState && DATA.viewState.latitude, 0),
        ],
        zoom: safeNumber(DATA.viewState && DATA.viewState.zoom, 1),
        attributionControl: true,
      }});
      map.addControl(new maplibregl.NavigationControl({{ showCompass: true }}), "top-right");

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
      const pathLayer = new deck.PathLayer({{
        id: "paths",
        data: paths,
        getPath: (d) => d.path,
        getColor: (d) => d.color,
        pickable: true,
        widthMinPixels: 2,
      }});

      const times = Array.isArray(DATA.times) ? DATA.times : [];
      const points = Array.isArray(DATA.points) ? DATA.points : [];
      const dotsByIdx = new Array(times.length);
      for (let i = 0; i < dotsByIdx.length; i++) dotsByIdx[i] = [];
      for (const p of points) {{
        const idx = Number(p.time_idx);
        if (!Number.isInteger(idx) || idx < 0 || idx >= dotsByIdx.length) continue;
        dotsByIdx[idx].push(p);
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

      function setDotIdx(idx) {{
        if (!overlay) return;
        const dots = (idx >= 0 && idx < dotsByIdx.length) ? dotsByIdx[idx] : [];
        overlay.setProps({{
          layers: [...bitmapLayers, pathLayer, makeDotLayer(dots)],
        }});
      }}

      // ---- timeline bars ----
      const selectedSet = new Set(Array.isArray(DATA.selected) ? DATA.selected : []);
      const barColors = DATA.barColors || {{}};
      const coverage = Array.isArray(DATA.coverage) ? DATA.coverage : [];
      const covMap = new Map();
      let globalMin = null;
      let globalMax = null;
      for (const c of coverage) {{
        const indiv = String(c.individual);
        const start = Number(c.start_ms);
        const end = Number(c.end_ms);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;
        covMap.set(indiv, {{ start_ms: start, end_ms: end }});
        if (globalMin === null || start < globalMin) globalMin = start;
        if (globalMax === null || end > globalMax) globalMax = end;
      }}
      if (globalMin === null || globalMax === null || globalMax <= globalMin) {{
        globalMin = 0;
        globalMax = 1;
      }}

      const barsEl = document.getElementById("mb-bars");
      const indivs = Array.isArray(DATA.individuals) ? DATA.individuals : [];
      for (const indiv of indivs) {{
        const row = document.createElement("div");
        row.className = "mb-row";

        const label = document.createElement("div");
        label.className = "mb-label";
        label.textContent = indiv;

        const track = document.createElement("div");
        track.className = "mb-track";

        const r = covMap.get(indiv);
        if (r) {{
          const leftPct = ((r.start_ms - globalMin) / (globalMax - globalMin)) * 100.0;
          const widthPct = Math.max(0.4, ((r.end_ms - r.start_ms) / (globalMax - globalMin)) * 100.0);
          const bar = document.createElement("div");
          bar.className = "mb-bar";
          bar.style.left = `${{leftPct}}%`;
          bar.style.width = `${{widthPct}}%`;
          bar.style.background = barColors[indiv] || "rgba(100, 100, 100, 0.7)";
          if (selectedSet.size > 0 && !selectedSet.has(indiv)) {{
            bar.style.opacity = "{TIMELINE_UNSELECTED_ALPHA_SCALE}";
            label.style.opacity = "{TIMELINE_UNSELECTED_ALPHA_SCALE}";
          }}
          track.appendChild(bar);
        }} else {{
          if (selectedSet.size > 0 && !selectedSet.has(indiv)) {{
            label.style.opacity = "{TIMELINE_UNSELECTED_ALPHA_SCALE}";
          }}
        }}

        row.appendChild(label);
        row.appendChild(track);
        barsEl.appendChild(row);
      }}

      // ---- slider ----
      const slider = document.getElementById("mb-slider");
      const timeEl = document.getElementById("mb-time");

      const maxIdx = Math.max(0, times.length - 1);
      slider.max = String(maxIdx);
      slider.value = "0";
      slider.disabled = times.length <= 1;

      if (times.length === 0) {{
        timeEl.textContent = "No timestamps";
      }} else {{
        timeEl.textContent = times[0];
      }}

      slider.addEventListener("input", (e) => {{
        const idx = Number(e.target.value);
        if (!Number.isInteger(idx)) return;
        setDotIdx(idx);
      }});

      // Timeline label only updates on release (change), per request.
      slider.addEventListener("change", (e) => {{
        const idx = Number(e.target.value);
        if (!Number.isInteger(idx)) return;
        timeEl.textContent = times[idx] || "";
      }});

      map.on("load", () => {{
        setDotIdx(0);
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

    if "show_train" not in st.session_state:
        st.session_state["show_train"] = True
    if "show_test" not in st.session_state:
        st.session_state["show_test"] = True

    train_palette, test_palette = _make_split_palettes(individuals, PATH_ALPHA)

    def _indiv_visible_key(indiv: str) -> str:
        return f"visible::{study_id}::{indiv}"

    last_study_id = st.session_state.get("_last_study_id")
    if last_study_id != study_id:
        st.session_state["_last_study_id"] = study_id
        for indiv in individuals:
            st.session_state[_indiv_visible_key(indiv)] = True

    # ---- table header ----
    h1, h2, h3 = st.sidebar.columns([3, 1, 1])
    h1.markdown("**Individual**")
    h2.checkbox("Train", key="show_train")
    h3.checkbox("Test", key="show_test")

    # ---- table rows ----
    for indiv in individuals:
        k = _indiv_visible_key(indiv)
        if k not in st.session_state:
            st.session_state[k] = True

        c1, c2, c3 = st.sidebar.columns([3, 1, 1])
        visible = c1.checkbox(indiv, key=k)

        train_color = train_palette[indiv]
        test_color = test_palette[indiv]
        c2.markdown(
            _color_dot_html(train_color, enabled=visible and bool(st.session_state.get("show_train", True))),
            unsafe_allow_html=True,
        )
        c3.markdown(
            _color_dot_html(test_color, enabled=visible and bool(st.session_state.get("show_test", True))),
            unsafe_allow_html=True,
        )

    sidebar_messages = st.sidebar.container()
    selected = {indiv for indiv in individuals if bool(st.session_state.get(_indiv_visible_key(indiv), False))}
    if not selected:
        sidebar_messages.warning("No individuals selected.")

    enabled_splits: list[str] = []
    if bool(st.session_state.get("show_train", True)):
        enabled_splits.append("train")
    if bool(st.session_state.get("show_test", True)):
        enabled_splits.append("test")

    if not enabled_splits:
        sidebar_messages.warning("No splits selected.")
    else:
        missing_splits = [s for s in enabled_splits if (df[SET_COL] == s).sum() == 0]
        if missing_splits:
            sidebar_messages.warning(f"No data for enabled split(s): {', '.join(missing_splits)} (rendering what exists).")

    all_rows = build_path_rows(df)
    if selected and enabled_splits:
        rows = [r for r in all_rows if r.get("individual") in selected and r.get("set") in enabled_splits]
    else:
        rows = []

    # ---- map-side controls ----
    basemap_options = list(BASEMAP_STYLES.keys())
    try:
        basemap_name = st.radio(
            "Basemap",
            basemap_options,
            index=0,
            horizontal=True,
            key="basemap_name",
        )
    except TypeError:
        basemap_name = st.radio("Basemap", basemap_options, index=0, key="basemap_name")
    map_style = BASEMAP_STYLES[basemap_name]

    raster_tiles: list[dict] = []
    with st.expander("Raster overlay (DEM)", expanded=False):
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
            dem_opacity = st.slider("Opacity", 0.0, 1.0, 0.6, 0.05, key="dem_opacity")

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

    times, points = _build_timeline_times_and_points(df, selected=selected, enabled_splits=enabled_splits)
    coverage = _build_individual_coverage_ms(df, enabled_splits=enabled_splits)

    bar_colors: dict[str, str] = {}
    for indiv, rgba in train_palette.items():
        r, g, b, _ = rgba
        bar_colors[str(indiv)] = f"rgba({int(r)},{int(g)},{int(b)},{float(TIMELINE_BAR_ALPHA) / 255.0:.3f})"

    _render_deckgl_viewer(
        map_style=map_style,
        view_state=view_state,
        path_rows=rows,
        raster_tiles=raster_tiles,
        individuals=individuals,
        selected=selected,
        times=times,
        points=points,
        coverage=coverage,
        bar_colors=bar_colors,
    )


if __name__ == "__main__":
    main()
