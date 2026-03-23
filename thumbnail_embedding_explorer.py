from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from movebench.thumbnail_embedding import (
    CACHE_ROOT,
    SOURCE_CSV_ROOT,
    compute_run_id,
    default_run_config,
    execute_run,
    export_embedding_report,
    list_embedding_sources,
    load_manifest,
    normalize_run_config,
    run_points_payload,
    thumbnail_png_bytes,
    chunk_detail_payload,
)

_JOBS: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def _build_handler(cache_root: Path, source_root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MoveBenchThumbnailEmbedding/0.1"

        def _json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text: str, *, status: int = 200) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _png(self, data: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def log_message(self, fmt: str, *args) -> None:
            print(fmt % args)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._html(_index_html())
                return
            if path == "/api/sources":
                self._json(
                    {
                        "default_config": default_run_config(),
                        "sources": list_embedding_sources(source_root),
                    }
                )
                return
            if path.startswith("/api/runs/") and path.endswith("/points"):
                run_id = path.split("/")[3]
                try:
                    self._json(run_points_payload(run_id, cache_root=cache_root))
                except FileNotFoundError as exc:
                    self._json({"error": str(exc)}, status=404)
                return
            if path.startswith("/api/runs/") and "/thumbnail/" in path:
                parts = path.split("/")
                run_id = parts[3]
                idx = int(parts[5])
                try:
                    self._png(thumbnail_png_bytes(run_id, idx=idx, cache_root=cache_root))
                except Exception as exc:
                    self._json({"error": str(exc)}, status=404)
                return
            if path.startswith("/api/runs/") and path.endswith("/chunk"):
                run_id = path.split("/")[3]
                query = parse_qs(parsed.query)
                idx = int(query.get("idx", ["-1"])[0])
                try:
                    self._json(chunk_detail_payload(run_id, idx=idx, cache_root=cache_root))
                except Exception as exc:
                    self._json({"error": str(exc)}, status=404)
                return
            if path.startswith("/api/runs/"):
                run_id = path.split("/")[3]
                try:
                    self._json(load_manifest(run_id, cache_root=cache_root))
                except FileNotFoundError as exc:
                    self._json({"error": str(exc)}, status=404)
                return
            self._json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/runs":
                payload = self._read_json()
                config = normalize_run_config(payload)
                run_id = compute_run_id(config)
                running_payload = _manifest_running_payload(run_id, config)
                with _LOCK:
                    existing = _JOBS.get(run_id)
                    if existing is None or not existing.is_alive():
                        thread = threading.Thread(
                            target=_run_job,
                            kwargs={"config": config, "cache_root": cache_root},
                            daemon=True,
                            name=f"thumbnail-run-{run_id}",
                        )
                        _JOBS[run_id] = thread
                        if not _manifest_completed(run_id, cache_root):
                            run_dir = cache_root / run_id
                            run_dir.mkdir(parents=True, exist_ok=True)
                            (run_dir / "manifest.json").write_text(
                                json.dumps(running_payload, indent=2, sort_keys=True),
                                encoding="utf-8",
                            )
                            thread.start()
                if _manifest_completed(run_id, cache_root):
                    self._json(load_manifest(run_id, cache_root=cache_root))
                else:
                    self._json(running_payload, status=HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/reports":
                payload = self._read_json()
                run_id = str(payload["run_id"])
                baseline_run_id = payload.get("baseline_run_id")
                out_dir = payload.get("out_dir")
                try:
                    report_path = export_embedding_report(
                        run_id,
                        baseline_run_id=baseline_run_id,
                        out_dir=out_dir,
                        cache_root=cache_root,
                    )
                except Exception as exc:
                    self._json({"error": str(exc)}, status=500)
                    return
                self._json({"run_id": run_id, "report_path": str(report_path)})
                return
            self._json({"error": "Not found"}, status=404)

    return Handler


def _manifest_completed(run_id: str, cache_root: Path) -> bool:
    try:
        manifest = load_manifest(run_id, cache_root=cache_root)
    except FileNotFoundError:
        return False
    return manifest.get("status") == "completed"


def _manifest_running_payload(run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "running",
        "config": config,
        "warnings": [],
        "summary": {},
    }


def _run_job(*, config: dict[str, Any], cache_root: Path) -> None:
    try:
        execute_run(config, cache_root=cache_root)
    except Exception as exc:
        print(f"Run {compute_run_id(config)} failed: {exc}")


def _index_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Thumbnail Embedding Explorer</title>
  <style>
    :root {
      --bg: #f3ede2;
      --panel: rgba(255,255,255,0.92);
      --ink: #17211f;
      --muted: #5c655f;
      --line: rgba(23,33,31,0.12);
      --accent: #d26531;
      --shadow: 0 18px 40px rgba(23,33,31,0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(76, 123, 111, 0.14), transparent 28%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
    }
    .wrap {
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(540px, 1fr) minmax(320px, 360px);
      gap: 18px;
      min-height: 100vh;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 16px;
    }
    h1, h2, h3 { margin: 0 0 10px 0; }
    h1 { font-size: 28px; letter-spacing: -0.03em; }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 12px;
    }
    label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }
    input, select, button {
      width: 100%;
      margin-top: 6px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: white;
      padding: 8px 10px;
      font: inherit;
      color: inherit;
    }
    button {
      cursor: pointer;
      background: linear-gradient(180deg, #ffffff 0%, #f7f2e7 100%);
    }
    button.primary {
      background: linear-gradient(180deg, #eea06d 0%, #d26531 100%);
      color: white;
      border-color: rgba(210,101,49,0.45);
    }
    .grid2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .scatter-wrap {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 8px;
    }
    .scatter-panel h3 { font-size: 15px; }
    canvas {
      width: 100%;
      height: 390px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: white;
      display: block;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .legend span { display: inline-flex; align-items: center; gap: 6px; }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }
    .status {
      min-height: 20px;
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
    }
    .thumb {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 12px;
      background: #f4f5f7;
      border: 1px solid var(--line);
    }
    .meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
      margin-top: 8px;
      white-space: pre-wrap;
    }
    .neighbors {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
      max-height: 340px;
      overflow: auto;
    }
    .neighbor {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 8px;
      background: white;
    }
    .neighbor img {
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      border-radius: 10px;
      background: #f4f5f7;
    }
    .neighbor .txt {
      font-size: 11px;
      color: var(--muted);
      margin-top: 6px;
    }
    @media (max-width: 1350px) {
      .wrap { grid-template-columns: 1fr; }
      .scatter-wrap { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <section class="panel">
      <h1>Thumbnail Embedding Explorer</h1>
      <div class="sub">GPU-backed when available, CPU-safe otherwise. Create runs, inspect embedding space, pin a baseline, and export a static report from cached artifacts.</div>
      <label>CSV file<select id="source-select"></select></label>
      <div class="grid2">
        <label>Split<select id="split-select"></select></label>
        <label>Projection<select id="projection-select">
          <option value="umap">UMAP</option>
          <option value="tsne">t-SNE</option>
          <option value="pca">PCA</option>
        </select></label>
      </div>
      <div class="grid2">
        <label>Chunk size<input id="chunk-size" type="number" min="2" value="50" /></label>
        <label>Stride<input id="stride" type="number" min="0" value="0" /></label>
      </div>
      <div class="grid2">
        <label>Min chunk<input id="min-chunk-size" type="number" min="2" value="10" /></label>
        <label>Image size<input id="img-size" type="number" min="16" value="64" /></label>
      </div>
      <div class="grid2">
        <label>Line width<input id="line-width" type="number" min="1" value="1" /></label>
        <label>Marker radius<input id="marker-radius" type="number" min="0" value="1" /></label>
      </div>
      <div class="grid2">
        <label>Padding<input id="padding" type="number" min="0" value="2" /></label>
        <label>Batch size<input id="batch-size" type="number" min="1" value="32" /></label>
      </div>
      <div class="grid2">
        <label>Scale mode<select id="scale-mode">
          <option value="individual">Individual</option>
          <option value="chunk">Chunk</option>
        </select></label>
        <label>Color by<select id="color-by-select"></select></label>
      </div>
      <div class="grid2">
        <label>Device<select id="device-pref">
          <option value="auto">Auto</option>
          <option value="cuda">CUDA</option>
          <option value="cpu">CPU</option>
        </select></label>
        <label>Label mode<select id="label-mode">
          <option value="overlap_state">Overlap state</option>
          <option value="clean">Clean</option>
          <option value="has_any_outlier">Any outlier</option>
          <option value="has_algorithm_marked_outlier">Algorithm</option>
          <option value="has_manually_marked_outlier">Manual</option>
        </select></label>
      </div>
      <button class="primary" id="run-btn">Create Run</button>
      <div class="grid2" style="margin-top:10px;">
        <button id="pin-baseline-btn">Pin Current As Baseline</button>
        <button id="export-report-btn">Export Report</button>
      </div>
      <div class="status" id="status"></div>
    </section>

    <section class="panel">
      <h2>Embedding Space</h2>
      <div class="scatter-wrap">
        <div class="scatter-panel">
          <h3>Current Run</h3>
          <canvas id="current-canvas" width="620" height="390"></canvas>
          <div class="status" id="current-summary"></div>
        </div>
        <div class="scatter-panel">
          <h3>Baseline Run</h3>
          <canvas id="baseline-canvas" width="620" height="390"></canvas>
          <div class="status" id="baseline-summary">No baseline pinned.</div>
        </div>
      </div>
      <div class="legend">
        <span><i class="dot" style="background:#2f7d4b"></i> clean</span>
        <span><i class="dot" style="background:#da6e2f"></i> algorithm only</span>
        <span><i class="dot" style="background:#147a8a"></i> manual only</span>
        <span><i class="dot" style="background:#9e3fcb"></i> algorithm + manual</span>
        <span><i class="dot" style="background:#c23c3c"></i> other outlier</span>
      </div>
    </section>

    <section class="panel">
      <h2>Selection</h2>
      <img id="selection-thumb" class="thumb" alt="selected thumbnail" />
      <div class="meta" id="selection-meta">No point selected.</div>
      <h3 style="margin-top:14px;">Nearest Neighbors</h3>
      <div class="neighbors" id="neighbors"></div>
    </section>
  </div>
  <script>
    const state = {
      sources: [],
      currentRunId: null,
      currentPoints: null,
      baselineRunId: null,
      baselinePoints: null,
      currentRendered: [],
      baselineRendered: [],
    };
    const palette = {
      "clean": "#2f7d4b",
      "algorithm only": "#da6e2f",
      "manual only": "#147a8a",
      "algorithm + manual": "#9e3fcb",
      "other outlier": "#c23c3c",
      "true": "#d26531",
      "false": "#7f8a84",
    };

    async function loadSources() {
      const data = await fetchJson("/api/sources");
      state.sources = data.sources || [];
      const sourceSelect = document.getElementById("source-select");
      sourceSelect.innerHTML = state.sources.map((s, idx) => `<option value="${idx}">${s.label}</option>`).join("");
      if (state.sources.length) applySource(0, data.default_config || null);
      sourceSelect.addEventListener("change", () => applySource(Number(sourceSelect.value), null));
    }

    function applySource(idx, defaultConfig) {
      const source = state.sources[idx];
      if (!source) return;
      const splitSelect = document.getElementById("split-select");
      splitSelect.innerHTML = (source.splits || []).map((s) => `<option value="${s}">${s}</option>`).join("");
      splitSelect.value = source.default_split || "all";
      const colorBy = document.getElementById("color-by-select");
      colorBy.innerHTML = [`<option value="">Black & white</option>`].concat((source.color_columns || []).map((c) => `<option value="${c}">${c}</option>`)).join("");
      if (defaultConfig) {
        document.getElementById("chunk-size").value = defaultConfig.thumbnail.chunk_size;
        document.getElementById("stride").value = defaultConfig.thumbnail.stride || 0;
      }
    }

    function currentSource() {
      return state.sources[Number(document.getElementById("source-select").value)] || null;
    }

    function buildConfig() {
      const source = currentSource();
      return {
        data: {
          csv_path: source ? source.path : "",
          split: document.getElementById("split-select").value,
        },
        thumbnail: {
          img_size: Number(document.getElementById("img-size").value),
          chunk_size: Number(document.getElementById("chunk-size").value),
          stride: Number(document.getElementById("stride").value) || null,
          min_chunk_size: Number(document.getElementById("min-chunk-size").value),
          line_width: Number(document.getElementById("line-width").value),
          marker_radius: Number(document.getElementById("marker-radius").value),
          padding: Number(document.getElementById("padding").value),
          scale_mode: document.getElementById("scale-mode").value,
          scale_quantile: 0.5,
          color_by: document.getElementById("color-by-select").value || null,
        },
        embedding: {
          model_name: "dinov2_vits14",
          device_preference: document.getElementById("device-pref").value,
          batch_size: Number(document.getElementById("batch-size").value),
          image_size: 224,
        },
        projection: {
          method: document.getElementById("projection-select").value,
          random_state: 17,
          n_neighbors: 20,
          min_dist: 0.10,
          perplexity: 30.0,
        },
      };
    }

    async function fetchJson(url, options) {
      const res = await fetch(url, options);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Request failed: ${res.status}`);
      return data;
    }

    function setStatus(text) {
      document.getElementById("status").textContent = text || "";
    }

    async function createRun() {
      try {
        setStatus("Creating run...");
        const manifest = await fetchJson("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(buildConfig()),
        });
        state.currentRunId = manifest.run_id;
        await pollRun(manifest.run_id, false);
      } catch (err) {
        setStatus(err.message);
      }
    }

    async function pollRun(runId, isBaseline) {
      while (true) {
        const manifest = await fetchJson(`/api/runs/${runId}`);
        if (manifest.status === "completed") {
          await loadPoints(runId, isBaseline);
          if (!isBaseline) setStatus(`Run ${runId} completed.`);
          return;
        }
        if (manifest.status === "failed") {
          throw new Error(manifest.error || `Run ${runId} failed.`);
        }
        setStatus(`Run ${runId} is ${manifest.status}...`);
        await new Promise((resolve) => setTimeout(resolve, 1200));
      }
    }

    async function loadPoints(runId, isBaseline) {
      const payload = await fetchJson(`/api/runs/${runId}/points`);
      if (isBaseline) {
        state.baselinePoints = payload;
        drawScatter("baseline-canvas", payload, true);
        document.getElementById("baseline-summary").textContent = summaryText(payload);
      } else {
        state.currentPoints = payload;
        drawScatter("current-canvas", payload, false);
        document.getElementById("current-summary").textContent = summaryText(payload);
      }
    }

    function summaryText(payload) {
      const summary = payload.summary || {};
      return `${summary.n_chunks || 0} chunks | ${summary.n_individuals || 0} individuals | ${(payload.warnings || []).join(" ")}`;
    }

    function pointColor(point) {
      const mode = document.getElementById("label-mode").value;
      if (mode === "overlap_state") return palette[point.overlap_state] || "#6d7671";
      return palette[String(Boolean(point[mode]))] || "#6d7671";
    }

    function drawScatter(canvasId, payload, isBaseline) {
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext("2d");
      const points = (payload && payload.points) || [];
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      if (!points.length) return;
      const xs = points.map((p) => p.x);
      const ys = points.map((p) => p.y);
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const pad = 18;
      const sx = (v) => pad + ((v - minX) / Math.max(1e-6, maxX - minX)) * (canvas.width - pad * 2);
      const sy = (v) => canvas.height - pad - ((v - minY) / Math.max(1e-6, maxY - minY)) * (canvas.height - pad * 2);
      const rendered = [];
      for (const point of points) {
        const x = sx(point.x);
        const y = sy(point.y);
        rendered.push({ ...point, screenX: x, screenY: y });
        ctx.fillStyle = pointColor(point);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      if (isBaseline) state.baselineRendered = rendered;
      else state.currentRendered = rendered;
    }

    async function selectPoint(runId, idx) {
      const detail = await fetchJson(`/api/runs/${runId}/chunk?idx=${idx}`);
      document.getElementById("selection-thumb").src = detail.thumbnail_url;
      document.getElementById("selection-meta").textContent = [
        `individual: ${detail.individual_id}`,
        `split: ${detail.split}`,
        `time: ${detail.start_time} → ${detail.end_time}`,
        `n_fixes: ${detail.n_fixes}`,
        `outlier_frac: ${detail.outlier_frac.toFixed(3)}`,
        `label: ${detail.overlap_state}`,
      ].join("\\n");
      document.getElementById("neighbors").innerHTML = (detail.neighbors || []).map((n) => `
        <div class="neighbor">
          <img src="${n.thumbnail_url}" alt="${n.chunk_id}" />
          <div class="txt">${n.individual_id}<br/>${n.start_time}<br/>dist=${n.distance.toFixed(3)}<br/>${n.overlap_state}</div>
        </div>
      `).join("");
      if (state.baselinePoints && state.currentPoints && runId === state.currentRunId) {
        const currentPoint = state.currentPoints.points.find((p) => p.idx === idx);
        if (currentPoint) {
          const match = (state.baselinePoints.points || []).find((p) => p.chunk_id === currentPoint.chunk_id);
          if (match) {
            document.getElementById("baseline-summary").textContent = `${summaryText(state.baselinePoints)} | linked baseline chunk found`;
          }
        }
      }
    }

    function attachCanvasClick(canvasId, getRunId, getPoints) {
      const canvas = document.getElementById(canvasId);
      canvas.addEventListener("click", async (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width);
        const y = (event.clientY - rect.top) * (canvas.height / rect.height);
        const points = getPoints();
        if (!points.length) return;
        let best = null;
        for (const point of points) {
          const dx = point.screenX - x;
          const dy = point.screenY - y;
          const dist = dx * dx + dy * dy;
          if (!best || dist < best.dist) best = { point, dist };
        }
        if (best && best.dist < 90) {
          await selectPoint(getRunId(), best.point.idx);
        }
      });
    }

    async function pinBaseline() {
      if (!state.currentRunId || !state.currentPoints) {
        setStatus("Create a current run first.");
        return;
      }
      state.baselineRunId = state.currentRunId;
      state.baselinePoints = state.currentPoints;
      drawScatter("baseline-canvas", state.baselinePoints, true);
      document.getElementById("baseline-summary").textContent = summaryText(state.baselinePoints);
      setStatus(`Pinned ${state.baselineRunId} as baseline.`);
    }

    async function exportReport() {
      if (!state.currentRunId) {
        setStatus("Create a run first.");
        return;
      }
      try {
        const res = await fetchJson("/api/reports", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: state.currentRunId,
            baseline_run_id: state.baselineRunId,
          }),
        });
        setStatus(`Report written to ${res.report_path}`);
      } catch (err) {
        setStatus(err.message);
      }
    }

    document.getElementById("run-btn").addEventListener("click", createRun);
    document.getElementById("pin-baseline-btn").addEventListener("click", pinBaseline);
    document.getElementById("export-report-btn").addEventListener("click", exportReport);
    document.getElementById("label-mode").addEventListener("change", () => {
      if (state.currentPoints) drawScatter("current-canvas", state.currentPoints, false);
      if (state.baselinePoints) drawScatter("baseline-canvas", state.baselinePoints, true);
    });
    attachCanvasClick("current-canvas", () => state.currentRunId, () => state.currentRendered);
    attachCanvasClick("baseline-canvas", () => state.baselineRunId, () => state.baselineRendered);
    loadSources().catch((err) => setStatus(err.message));
  </script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local thumbnail embedding explorer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--cache-root", default=str(CACHE_ROOT))
    parser.add_argument("--source-root", default=str(SOURCE_CSV_ROOT))
    args = parser.parse_args(argv)

    cache_root = Path(args.cache_root).resolve()
    source_root = Path(args.source_root).resolve()
    handler_cls = _build_handler(cache_root, source_root)
    server = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"Thumbnail embedding explorer serving on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
