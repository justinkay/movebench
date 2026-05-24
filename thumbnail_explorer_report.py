"""Generate a static thumbnail explorer report with no live server.

Example:
  ~/micromamba/envs/movebench/bin/python thumbnail_explorer_report.py \
    --study atlas_bats \
    --split train \
    --out /tmp/atlas_bats_thumbnail_report
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import io
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from PIL import Image

from movebench import Study
from movebench.datasets.thumbnails import build_track_thumbnails
from movebench.thumbnail_basemaps import DEFAULT_THUMBNAIL_BASEMAP, thumbnail_basemap_spec
from movebench.trajectories import INDIVIDUAL_COL, LAT_COL, LON_COL, SET_COL, TIME_COL

PATH_ALPHA = 0.25


def _alternating_hue_order(n: int) -> list[int]:
    half = (n + 1) // 2
    out: list[int] = []
    for i in range(half):
        out.append(i)
        j = i + half
        if j < n:
            out.append(j)
    return out[:n]


def _hls_to_hex(h: float, l: float, s: float) -> str:
    r, g, b = colorsys.hls_to_rgb(float(h), float(l), float(s))
    return "#{:02x}{:02x}{:02x}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


def _make_palette(individuals: list[str], *, lightness: float) -> dict[str, str]:
    n = len(individuals)
    if n == 0:
        return {}
    order = _alternating_hue_order(n)
    colors: dict[str, str] = {}
    for idx, indiv in enumerate(individuals):
        h = float(order[idx]) / float(n)
        colors[indiv] = _hls_to_hex(h, lightness, 0.82)
    return colors


def _temporal_color(idx: int, total: int) -> str:
    if total <= 1:
        return "#2b6cb0"
    frac = idx / float(total - 1)
    hue = 0.60 - 0.52 * frac
    light = 0.56 - 0.08 * frac
    return _hls_to_hex(hue, light, 0.72)


def _image_to_data_url(img: np.ndarray) -> str:
    mode = "L" if img.ndim == 2 else "RGB"
    pil = Image.fromarray(img, mode=mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text)).strip("_")


def _format_class_name(col: str) -> str:
    if col == "clean":
        return "Clean"
    if col == "has_any_outlier":
        return "Any Outlier"
    if col.startswith("has_"):
        col = col[4:]
    return col.replace("_", " ").title()


def _load_full_paths(study_id: str, split: str) -> list[dict]:
    df = pd.read_csv(Study(study_id).trajectories_path, low_memory=False)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, LON_COL, LAT_COL]).copy()
    if split != "all" and SET_COL in df.columns:
        df = df[df[SET_COL] == split].copy()
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL], kind="mergesort")
    individuals = sorted(df[INDIVIDUAL_COL].dropna().astype(str).unique().tolist())
    train_palette = _make_palette(individuals, lightness=0.64)
    test_palette = _make_palette(individuals, lightness=0.36)

    rows: list[dict] = []
    for (individual_id, split_name), group in df.groupby([INDIVIDUAL_COL, SET_COL], sort=False):
        coords = group[[LAT_COL, LON_COL]].to_numpy(dtype=float, copy=False)
        if len(coords) < 2:
            continue
        color = train_palette.get(str(individual_id)) if str(split_name) == "train" else test_palette.get(str(individual_id))
        rows.append(
            {
                "individual": str(individual_id),
                "split": str(split_name),
                "coords": coords.tolist(),
                "color": color or "#777777",
            }
        )
    return rows


def _build_class_pdfs(
    out_dir: Path,
    *,
    study_id: str,
    split: str,
    images: np.ndarray,
    meta: pd.DataFrame,
    class_columns: list[str],
    params: dict,
) -> dict[str, str]:
    pdf_dir = out_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    links: dict[str, str] = {}
    for class_name in class_columns:
        idxs = meta.index[meta[class_name].fillna(False).astype(bool)].tolist()
        if not idxs:
            continue
        file_name = (
            f"{_safe_name(study_id)}__{_safe_name(split)}__{_safe_name(class_name)}__"
            f"chunk{params['chunk_size']}__img{params['img_size']}.pdf"
        )
        path = pdf_dir / file_name
        with PdfPages(path) as pdf:
            per_page = 20
            cols = 5
            rows = 4
            title = (
                f"{study_id} | {_format_class_name(class_name)} | split={split} | "
                f"mode={params['chunk_mode']} chunk={params['chunk_size']} "
                f"stride={params['stride'] or params['chunk_size']} "
                f"img={params['img_size']} scale={params['scale_mode']}"
            )
            if params.get("basemap"):
                title += f" basemap={params['basemap']}"
            for start in range(0, len(idxs), per_page):
                page = idxs[start : start + per_page]
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.6))
                axes_arr = np.atleast_1d(axes).ravel()
                for ax in axes_arr:
                    ax.axis("off")
                for ax, idx in zip(axes_arr, page):
                    row = meta.loc[idx]
                    if images[idx].ndim == 2:
                        ax.imshow(images[idx], cmap="gray", vmin=0, vmax=255)
                    else:
                        ax.imshow(images[idx])
                    ax.set_title(
                        f"#{int(row['global_chunk_idx'])} | {row['individual_id']}\n"
                        f"{pd.Timestamp(row['start_time']).strftime('%Y-%m-%d')}",
                        fontsize=8,
                    )
                    ax.axis("off")
                fig.suptitle(title, fontsize=12)
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        links[class_name] = f"pdf/{file_name}"
    return links


def _build_payload(
    *,
    study_id: str,
    split: str,
    out_dir: Path,
    params: dict,
) -> dict:
    split_arg = None if split == "all" else split
    build = build_track_thumbnails(
        Study(study_id).trajectories_path,
        split=split_arg,
        img_size=params["img_size"],
        chunk_mode=params["chunk_mode"],
        chunk_size=params["chunk_size"],
        stride=params["stride"],
        min_chunk_size=params["min_chunk_size"],
        burst_gap_seconds=params["burst_gap_seconds"],
        line_width=params["line_width"],
        marker_radius=params["marker_radius"],
        padding=params["padding"],
        scale_mode=params["scale_mode"],
        scale_quantile=params["scale_quantile"],
        basemap=params["basemap"],
    )
    meta = build.metadata.copy()
    if len(meta) == 0:
        raise RuntimeError("No thumbnail chunks produced for the requested study/split/parameters.")

    meta["_orig_idx"] = np.arange(len(meta), dtype=int)
    meta["start_time"] = pd.to_datetime(meta["start_time"])
    meta["end_time"] = pd.to_datetime(meta["end_time"])
    meta = meta.sort_values(["start_time", "individual_id", "chunk_idx"], kind="mergesort").reset_index(drop=True)
    meta["global_chunk_idx"] = np.arange(len(meta), dtype=int)

    reordered_paths = [build.chunk_paths[int(idx)] for idx in meta["_orig_idx"].tolist()]
    images = build.images[meta["_orig_idx"].to_numpy(dtype=int)]

    class_columns = [c for c in ("clean", "has_any_outlier") if c in meta.columns]
    class_columns += sorted(c for c in meta.columns if c.startswith("has_") and c != "has_any_outlier")
    pdf_links = _build_class_pdfs(
        out_dir,
        study_id=study_id,
        split=split,
        images=images,
        meta=meta,
        class_columns=class_columns,
        params=params,
    )

    chunks: list[dict] = []
    total = len(meta)
    for order_idx, row in meta.iterrows():
        classes = [c for c in class_columns if bool(row.get(c, False))]
        chunks.append(
            {
                "id": int(order_idx),
                "individual_id": str(row["individual_id"]),
                "split": row.get("split"),
                "chunk_idx": int(row["chunk_idx"]),
                "start_time": pd.Timestamp(row["start_time"]).isoformat(),
                "end_time": pd.Timestamp(row["end_time"]).isoformat(),
                "n_fixes": int(row["n_fixes"]),
                "outlier_frac": float(row.get("outlier_frac", 0.0)),
                "classes": classes,
                "path": [[float(lat), float(lon)] for lon, lat in reordered_paths[order_idx]],
                "start_marker": [float(row["start_lat"]), float(row["start_lon"])],
                "end_marker": [float(row["end_lat"]), float(row["end_lon"])],
                "thumb": _image_to_data_url(images[order_idx]),
                "temporal_color": _temporal_color(order_idx, total),
            }
        )

    meta = meta.drop(columns=["_orig_idx"])
    return {
        "study_id": study_id,
        "split": split,
        "params": params,
        "class_columns": class_columns,
        "class_labels": {c: _format_class_name(c) for c in class_columns},
        "pdf_links": pdf_links,
        "full_paths": _load_full_paths(study_id, split),
        "chunks": chunks,
    }


def _html_template(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Thumbnail Explorer Report</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: rgba(255,255,255,0.92);
      --ink: #1c221f;
      --muted: #5c655f;
      --line: rgba(28,34,31,0.15);
      --accent: #d95f18;
      --shadow: 0 18px 40px rgba(25, 36, 31, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(63, 119, 104, 0.14), transparent 28%),
        linear-gradient(180deg, #f7f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{
      min-height: 100vh;
      padding: 22px;
    }}
    .topbar {{
      margin-bottom: 18px;
      padding: 18px 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .title {{
      margin: 0 0 6px 0;
      font-size: 28px;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 0.95fr) minmax(480px, 1.05fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-head {{
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .panel-head h2 {{
      margin: 0;
      font-size: 18px;
    }}
    #map {{
      width: 100%;
      height: 720px;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      padding: 14px 18px 0 18px;
    }}
    .controls label {{
      font-size: 13px;
      color: var(--muted);
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    select, button {{
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      padding: 8px 10px;
      font: inherit;
      color: inherit;
    }}
    .summary {{
      padding: 0 18px 10px 18px;
      display: grid;
      grid-template-columns: repeat(3, minmax(120px, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.7);
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }}
    .stat .value {{
      font-size: 22px;
      font-weight: 600;
    }}
    .active {{
      margin: 0 18px 14px 18px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(250,248,243,0.95);
    }}
    .active h3 {{
      margin: 0 0 10px 0;
      font-size: 15px;
    }}
    .active dl {{
      margin: 0;
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }}
    .active dt {{
      color: var(--muted);
    }}
    .grid {{
      padding: 0 18px 18px 18px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 12px;
      max-height: 860px;
      overflow: auto;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px;
      background: white;
      cursor: pointer;
      transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
    }}
    .card:hover {{
      transform: translateY(-1px);
      box-shadow: 0 10px 22px rgba(25,36,31,0.10);
    }}
    .card.active-card {{
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(217,95,24,0.12);
    }}
    .card img {{
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #060606;
      border-radius: 10px;
      display: block;
    }}
    .card-title {{
      font-size: 12px;
      font-weight: 600;
      margin-top: 8px;
    }}
    .card-sub {{
      font-size: 11px;
      color: var(--muted);
      margin-top: 4px;
      line-height: 1.35;
    }}
    .pdf-link {{
      margin-left: auto;
      font-size: 13px;
      color: #0f5f53;
      text-decoration: none;
      font-weight: 600;
    }}
    .pdf-link.disabled {{
      color: var(--muted);
      pointer-events: none;
      text-decoration: none;
    }}
    @media (max-width: 1200px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      #map {{
        height: 520px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <h1 class="title">Thumbnail Explorer Report</h1>
      <p class="subtitle" id="subtitle"></p>
    </div>
    <div class="layout">
      <section class="panel">
        <div class="panel-head">
          <h2>Map</h2>
          <label><input type="checkbox" id="toggle-all-chunks" /> Show all chunks</label>
        </div>
        <div id="map"></div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <h2>Thumbnail Classes</h2>
          <a id="pdf-link" class="pdf-link" href="#" target="_blank" rel="noopener">Download class PDF</a>
        </div>
        <div class="controls">
          <label>Class
            <select id="class-select"></select>
          </label>
          <label>Sort
            <select id="sort-select">
              <option>Time</option>
              <option>Outlier fraction</option>
              <option>Individual then time</option>
            </select>
          </label>
        </div>
        <div class="summary" id="summary"></div>
        <div class="active">
          <h3>Active Chunk</h3>
          <dl id="active-details"></dl>
        </div>
        <div class="grid" id="grid"></div>
      </section>
    </div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const DATA = {payload_json};
    const subtitle = document.getElementById("subtitle");
    subtitle.textContent =
      `${{DATA.study_id}} | split=${{DATA.split}} | chunk=${{DATA.params.chunk_size}} | ` +
      `stride=${{DATA.params.stride ?? DATA.params.chunk_size}} | img=${{DATA.params.img_size}} | ` +
      `scale=${{DATA.params.scale_mode}} | basemap=${{DATA.params.basemap}}`;

    const map = L.map("map", {{ preferCanvas: true }});
    L.tileLayer("{thumbnail_basemap_spec(DEFAULT_THUMBNAIL_BASEMAP).tile_url}", {{
      maxZoom: 19,
      attribution: "{thumbnail_basemap_spec(DEFAULT_THUMBNAIL_BASEMAP).attribution}"
    }}).addTo(map);

    const fullTrackLayer = L.layerGroup().addTo(map);
    const chunkLayer = L.layerGroup().addTo(map);
    const activeLayer = L.layerGroup().addTo(map);
    const bounds = [];

    for (const row of DATA.full_paths) {{
      const poly = L.polyline(row.coords, {{
        color: row.color,
        weight: 2,
        opacity: {PATH_ALPHA},
      }}).bindTooltip(`${{row.individual}} | ${{row.split}}`);
      poly.addTo(fullTrackLayer);
      bounds.push(...row.coords);
    }}
    if (bounds.length) {{
      map.fitBounds(bounds, {{ padding: [20, 20] }});
    }} else {{
      map.setView([0, 0], 2);
    }}

    const classSelect = document.getElementById("class-select");
    const sortSelect = document.getElementById("sort-select");
    const toggleAllChunks = document.getElementById("toggle-all-chunks");
    const summaryEl = document.getElementById("summary");
    const activeDetailsEl = document.getElementById("active-details");
    const gridEl = document.getElementById("grid");
    const pdfLink = document.getElementById("pdf-link");

    function labelFor(cls) {{
      return DATA.class_labels[cls] || cls;
    }}

    for (const cls of DATA.class_columns) {{
      const opt = document.createElement("option");
      opt.value = cls;
      opt.textContent = labelFor(cls);
      classSelect.appendChild(opt);
    }}

    let activeClass = DATA.class_columns[0];
    let activeChunkId = null;

    function chunksForClass(cls) {{
      let rows = DATA.chunks.filter((row) => row.classes.includes(cls));
      const sortMode = sortSelect.value;
      if (sortMode === "Outlier fraction") {{
        rows = rows.sort((a, b) => (b.outlier_frac - a.outlier_frac) || a.start_time.localeCompare(b.start_time));
      }} else if (sortMode === "Individual then time") {{
        rows = rows.sort((a, b) => (a.individual_id.localeCompare(b.individual_id) || a.start_time.localeCompare(b.start_time)));
      }} else {{
        rows = rows.sort((a, b) => (a.start_time.localeCompare(b.start_time) || a.individual_id.localeCompare(b.individual_id)));
      }}
      return rows;
    }}

    function renderSummary() {{
      summaryEl.innerHTML = "";
      const rows = [
        ["Class", labelFor(activeClass)],
        ["Chunks", String(chunksForClass(activeClass).length)],
        ["PDF", DATA.pdf_links[activeClass] ? "Ready" : "Unavailable"],
      ];
      for (const [label, value] of rows) {{
        const el = document.createElement("div");
        el.className = "stat";
        el.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}</div>`;
        summaryEl.appendChild(el);
      }}
      const href = DATA.pdf_links[activeClass];
      if (href) {{
        pdfLink.href = href;
        pdfLink.classList.remove("disabled");
      }} else {{
        pdfLink.href = "#";
        pdfLink.classList.add("disabled");
      }}
    }}

    function renderChunkOverlay(rows) {{
      chunkLayer.clearLayers();
      if (!toggleAllChunks.checked) return;
      for (const row of rows) {{
        L.polyline(row.path, {{
          color: row.temporal_color,
          weight: 2,
          opacity: 0.7,
        }}).bindTooltip(`#${{row.id}} | ${{row.individual_id}} | ${{row.start_time}}`).addTo(chunkLayer);
      }}
    }}

    function renderActiveChunk(row) {{
      activeLayer.clearLayers();
      if (!row) return;
      const poly = L.polyline(row.path, {{
        color: "#f4622a",
        weight: 5,
        opacity: 0.95,
      }}).addTo(activeLayer);
      L.circleMarker(row.start_marker, {{
        radius: 6,
        color: "#ffffff",
        weight: 1,
        fillColor: "#27ae60",
        fillOpacity: 1,
      }}).bindTooltip("Start").addTo(activeLayer);
      L.circleMarker(row.end_marker, {{
        radius: 6,
        color: "#ffffff",
        weight: 1,
        fillColor: "#e74c3c",
        fillOpacity: 1,
      }}).bindTooltip("End").addTo(activeLayer);
      map.fitBounds(poly.getBounds(), {{ padding: [24, 24], maxZoom: 12 }});

      activeDetailsEl.innerHTML = "";
      const details = [
        ["Chunk", `#${{row.id}}`],
        ["Individual", row.individual_id],
        ["Split", row.split],
        ["Fixes", String(row.n_fixes)],
        ["Start", row.start_time],
        ["End", row.end_time],
        ["Outlier frac", row.outlier_frac.toFixed(3)],
      ];
      for (const [label, value] of details) {{
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        activeDetailsEl.appendChild(dt);
        activeDetailsEl.appendChild(dd);
      }}
    }}

    function renderGrid(rows) {{
      gridEl.innerHTML = "";
      for (const row of rows) {{
        const card = document.createElement("article");
        card.className = "card";
        if (row.id === activeChunkId) card.classList.add("active-card");
        card.innerHTML = `
          <img src="${{row.thumb}}" alt="thumbnail chunk ${{row.id}}" />
          <div class="card-title">#${{row.id}} | ${{row.individual_id}}</div>
          <div class="card-sub">
            ${{row.start_time.slice(0, 16).replace("T", " ")}}<br/>
            split=${{row.split}} | outlier_frac=${{row.outlier_frac.toFixed(2)}}
          </div>
        `;
        card.addEventListener("click", () => {{
          activeChunkId = row.id;
          refresh();
        }});
        gridEl.appendChild(card);
      }}
    }}

    function refresh() {{
      const rows = chunksForClass(activeClass);
      if (!rows.length) return;
      if (!rows.some((row) => row.id === activeChunkId)) {{
        activeChunkId = rows[0].id;
      }}
      const activeRow = rows.find((row) => row.id === activeChunkId) || rows[0];
      renderSummary();
      renderChunkOverlay(rows);
      renderActiveChunk(activeRow);
      renderGrid(rows);
    }}

    classSelect.addEventListener("change", () => {{
      activeClass = classSelect.value;
      activeChunkId = null;
      refresh();
    }});
    sortSelect.addEventListener("change", refresh);
    toggleAllChunks.addEventListener("change", refresh);

    activeClass = classSelect.value;
    refresh();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static thumbnail explorer report.")
    parser.add_argument("--study", required=True, help="Study id under the MoveBench studies root.")
    parser.add_argument("--split", default="train", choices=["train", "test", "all"])
    parser.add_argument("--out", required=True, help="Output directory for index.html and PDFs.")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--chunk-mode", choices=["fixed", "burst"], default="fixed")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=0, help="0 means use chunk size.")
    parser.add_argument("--min-chunk-size", type=int, default=10)
    parser.add_argument("--burst-gap-seconds", type=float, default=3600.0)
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--marker-radius", type=int, default=1)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--scale-mode", choices=["individual", "chunk"], default="individual")
    parser.add_argument("--scale-quantile", type=float, default=0.5)
    parser.add_argument(
        "--basemap",
        choices=["positron", "topographic", "osm", "satellite", "none"],
        default=DEFAULT_THUMBNAIL_BASEMAP,
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "img_size": int(args.img_size),
        "chunk_mode": str(args.chunk_mode),
        "chunk_size": int(args.chunk_size),
        "stride": None if int(args.stride) <= 0 else int(args.stride),
        "min_chunk_size": int(args.min_chunk_size),
        "burst_gap_seconds": float(args.burst_gap_seconds),
        "line_width": int(args.line_width),
        "marker_radius": int(args.marker_radius),
        "padding": int(args.padding),
        "scale_mode": str(args.scale_mode),
        "scale_quantile": float(args.scale_quantile),
        "basemap": str(args.basemap),
    }

    payload = _build_payload(
        study_id=str(args.study),
        split=str(args.split),
        out_dir=out_dir,
        params=params,
    )
    html = _html_template(payload)
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
