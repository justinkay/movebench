const BASEMAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const PATH_ALPHA = 180;
const POINT_ALPHA = 210;
const STORAGE_VERSION = 2;

let assetPromise = null;

async function ensureAssetsLoaded() {
  if (assetPromise) {
    return assetPromise;
  }
  assetPromise = Promise.all([
    loadCss("/static/vendor/maplibre-gl/maplibre-gl.css"),
    loadScript("/static/vendor/maplibre-gl/maplibre-gl.js"),
    loadScript("/static/vendor/deckgl/deck.gl.min.js"),
    loadScript("/static/vendor/deckgl/deck.gl-mapbox.min.js"),
  ]);
  return assetPromise;
}

function loadCss(href) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`link[data-movebench-asset="${href}"]`);
    if (existing) {
      resolve();
      return;
    }
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.dataset.movebenchAsset = href;
    link.onload = () => resolve();
    link.onerror = () => reject(new Error(`Failed to load ${href}`));
    document.head.appendChild(link);
  });
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-movebench-asset="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "true") {
        resolve();
      } else {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
      }
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = false;
    script.dataset.movebenchAsset = src;
    script.addEventListener("load", () => {
      script.dataset.loaded = "true";
      resolve();
    }, { once: true });
    script.addEventListener("error", () => reject(new Error(`Failed to load ${src}`)), { once: true });
    document.head.appendChild(script);
  });
}

class MoveBenchTrajectoryApp {
  constructor({ mountEl }) {
    this.mountEl = mountEl;
    this.uiState = this.loadUiState();
    this.artifacts = [];
    this.layerConfigs = [];
    this.studiesRoot = "";
    this.currentArtifact = "";
    this.currentArtifactEntry = null;
    this.currentLayerConfig = "";
    this.covariateSteps = [];
    this.covariateStepOptionsKey = "";
    this.activeTab = this.uiState.activeTab || "trajectory";
    this.data = null;
    this.covariates = null;
    this.currentTimeMs = 0;
    this.map = null;
    this.overlay = null;
    this.mapLoaded = false;
    this.loadRequestId = 0;
  }

  async init() {
    await ensureAssetsLoaded();
    this.renderShell();
    this.bindEvents();
    await this.loadArtifacts();
    this.syncTabVisibility();
  }

  async destroy() {
    if (this.map) {
      this.map.remove();
      this.map = null;
      this.overlay = null;
      this.mapLoaded = false;
    }
  }

  loadUiState() {
    try {
      const raw = localStorage.getItem("movebench.app.trajectory");
      const parsed = JSON.parse(raw || "");
      if (!parsed || parsed.version !== STORAGE_VERSION) {
        throw new Error("stale");
      }
      return parsed;
    } catch {
      return {
        version: STORAGE_VERSION,
        artifact: "",
        activeTab: "trajectory",
        layerConfig: "",
        pointId: "",
        showTrain: true,
        showTest: true,
        showPoints: false,
        selectedIndividuals: [],
      };
    }
  }

  saveUiState() {
    const payload = {
      version: STORAGE_VERSION,
      artifact: this.currentArtifact,
      activeTab: this.activeTab,
      layerConfig: this.currentLayerConfig,
      pointId: this.refs?.covPointId?.value || "",
      showTrain: this.refs.showTrain.checked,
      showTest: this.refs.showTest.checked,
      showPoints: this.refs.showPoints.checked,
      selectedIndividuals: this.getCheckedIndividuals(),
    };
    localStorage.setItem("movebench.app.trajectory", JSON.stringify(payload));
  }

  renderShell() {
    this.mountEl.innerHTML = `
      <style>
        .traj-root {
          display: grid;
          grid-template-rows: auto auto auto auto minmax(0, 1fr);
          min-height: 100%;
          color: #e5edf7;
          font-family: "Segoe UI", sans-serif;
          background: linear-gradient(180deg, rgba(8, 17, 27, 0.96), rgba(5, 10, 18, 0.98));
        }
        .traj-root.mode-trajectory {
          height: calc(100vh - 64px);
          min-height: calc(100vh - 64px);
        }
        .traj-root.mode-covariates {
          height: auto;
          min-height: calc(100vh - 64px);
        }
        .traj-toolbar,
        .cov-toolbar {
          display: flex;
          flex-wrap: wrap;
          gap: 10px 14px;
          align-items: center;
          padding: 14px 16px 10px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
          background: rgba(8, 17, 27, 0.8);
        }
        .traj-toolbar label,
        .traj-toggle,
        .cov-toolbar label {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 12px;
          color: #92a4bd;
        }
        .traj-toolbar select,
        .traj-toolbar button,
        .cov-toolbar select,
        .cov-toolbar input,
        .cov-toolbar button {
          font: inherit;
        }
        .traj-toolbar select,
        .cov-toolbar select,
        .cov-toolbar input {
          min-width: min(420px, calc(100vw - 48px));
          padding: 8px 10px;
          border-radius: 10px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(15, 23, 42, 0.92);
          color: #e5edf7;
        }
        .cov-toolbar input {
          min-width: 120px;
          width: 120px;
        }
        .traj-toolbar button,
        .cov-toolbar button,
        .traj-tab {
          padding: 8px 12px;
          border: none;
          border-radius: 10px;
          background: rgba(255, 255, 255, 0.08);
          color: #e5edf7;
          cursor: pointer;
          font-size: 14px;
          white-space: nowrap;
        }
        .traj-toolbar button:disabled,
        .cov-toolbar button:disabled {
          cursor: not-allowed;
          opacity: 0.45;
        }
        .traj-meta,
        .traj-status {
          padding: 0 16px 10px;
          font-size: 12px;
        }
        .traj-meta {
          color: #6ea7c9;
        }
        .traj-meta code {
          color: #b9e7ff;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .traj-status {
          min-height: 24px;
          color: #92a4bd;
        }
        .traj-status.error,
        .cov-status.error {
          color: #ffb3c2;
        }
        .traj-tabs {
          display: flex;
          gap: 8px;
          padding: 0 16px 12px;
        }
        .traj-tab.active {
          background: rgba(139, 216, 255, 0.18);
          color: #dff5ff;
        }
        .traj-body,
        .cov-root {
          min-height: 0;
        }
        .traj-root.mode-trajectory .traj-body,
        .traj-root.mode-trajectory .traj-main {
          height: 100%;
        }
        .traj-body.hidden,
        .cov-root.hidden {
          display: none;
        }
        .traj-main {
          display: grid;
          grid-template-columns: minmax(0, 1.2fr) minmax(300px, 0.8fr);
          gap: 12px;
          min-height: 0;
          padding: 0 16px 14px;
        }
        .traj-map-wrap,
        .traj-side,
        .cov-section,
        .cov-card {
          min-height: 0;
          border-radius: 16px;
          overflow: hidden;
          border: 1px solid rgba(255, 255, 255, 0.07);
          background: rgba(255, 255, 255, 0.03);
        }
        .traj-map-wrap {
          position: relative;
          background: #08111b;
        }
        .traj-map {
          width: 100%;
          height: 100%;
        }
        .traj-overlay {
          position: absolute;
          inset: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
          text-align: center;
          color: #d6deea;
          background: linear-gradient(180deg, rgba(8, 17, 27, 0.18), rgba(8, 17, 27, 0.8));
        }
        .traj-overlay.hidden {
          display: none;
        }
        .traj-overlay-card {
          max-width: 560px;
          padding: 18px 20px;
          border-radius: 16px;
          background: rgba(7, 11, 22, 0.88);
          border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .traj-overlay-card h3 {
          font-size: 16px;
          margin-bottom: 8px;
        }
        .traj-overlay-card p {
          font-size: 13px;
          line-height: 1.5;
        }
        .traj-side {
          display: grid;
          grid-template-rows: auto minmax(0, 1fr) auto;
        }
        .traj-side-head,
        .traj-slider-row,
        .cov-section-head {
          padding: 12px 14px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }
        .traj-side-head,
        .cov-section-head {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          font-size: 12px;
          color: #92a4bd;
        }
        .traj-side-head strong,
        .cov-section-head strong {
          color: #dbe5f0;
        }
        .traj-individuals {
          overflow-y: auto;
          padding: 10px 12px;
        }
        .traj-individual {
          display: grid;
          gap: 6px;
          padding: 10px 12px;
          margin-bottom: 10px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.04);
          border: 1px solid rgba(255, 255, 255, 0.05);
          cursor: pointer;
        }
        .traj-name {
          display: flex;
          justify-content: space-between;
          gap: 10px;
          font-size: 12px;
          color: #eef4fb;
        }
        .traj-name-left {
          display: flex;
          gap: 8px;
          align-items: center;
          min-width: 0;
        }
        .traj-name-left input {
          margin: 0;
        }
        .traj-name-label,
        .traj-name-species {
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
        }
        .traj-name-species,
        .traj-stats,
        .cov-muted,
        .cov-status {
          color: #92a4bd;
        }
        .traj-name-species,
        .traj-stats,
        .cov-muted,
        .cov-status {
          font-size: 11px;
        }
        .traj-stats {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }
        .traj-track {
          position: relative;
          height: 10px;
          width: 100%;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.06);
          overflow: hidden;
        }
        .traj-bar {
          position: absolute;
          top: 0;
          height: 100%;
          border-radius: 999px;
        }
        .traj-slider-row {
          display: grid;
          gap: 10px;
          border-bottom: none;
        }
        .traj-slider {
          width: 100%;
        }
        .traj-time {
          font-size: 12px;
          color: #dbe5f0;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .cov-root {
          display: grid;
          grid-template-rows: auto auto minmax(0, 1fr);
          gap: 12px;
          padding: 0 16px 14px;
        }
        .cov-summary {
          padding: 0 16px;
          font-size: 12px;
          color: #92a4bd;
        }
        .cov-summary code {
          color: #dff5ff;
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        }
        .cov-content {
          display: grid;
          gap: 12px;
          min-height: auto;
        }
        .cov-family {
          display: grid;
          gap: 12px;
          padding: 12px;
        }
        .cov-family-head {
          display: flex;
          justify-content: space-between;
          gap: 12px;
          align-items: baseline;
        }
        .cov-family-head h4 {
          font-size: 16px;
          color: #eef4fb;
        }
        .cov-family-summary {
          font-size: 11px;
          color: #92a4bd;
          text-align: right;
        }
        .cov-family-grid {
          display: grid;
          gap: 12px;
        }
        .cov-family-stream {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
          gap: 14px;
        }
        .cov-source-title {
          font-size: 13px;
          color: #dff5ff;
        }
        .cov-band-source {
          font-size: 11px;
          color: #92a4bd;
          text-transform: uppercase;
          letter-spacing: 0.06em;
        }
        .cov-coverage-card {
          display: grid;
          gap: 8px;
          padding: 12px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
          align-content: start;
        }
        .cov-coverage-meta {
          font-size: 11px;
          color: #92a4bd;
        }
        .cov-section-body {
          display: grid;
          gap: 12px;
          padding: 12px;
        }
        .cov-card {
          padding: 12px;
          display: grid;
          gap: 10px;
        }
        .cov-card h4 {
          font-size: 14px;
          color: #eef4fb;
        }
        .cov-band-image {
          width: 100%;
          height: auto;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background: rgba(0, 0, 0, 0.2);
        }
        .cov-layer-grid {
          display: grid;
          gap: 12px;
          align-content: start;
        }
        .cov-band-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
          gap: 14px;
        }
        .cov-band-card {
          display: grid;
          gap: 8px;
          padding: 12px;
          border-radius: 12px;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .cov-band-title {
          font-size: 12px;
          color: #eef4fb;
          word-break: break-word;
        }
        .cov-band-meta {
          font-size: 11px;
          color: #92a4bd;
        }
        .cov-two-up {
          display: grid;
          grid-template-columns: minmax(0, 1.8fr) minmax(320px, 520px);
          gap: 12px;
          align-items: start;
        }
        .cov-coverage {
          display: block;
          width: 100%;
          max-width: 520px;
          height: auto;
          max-height: 300px;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          background:
            radial-gradient(circle at top left, rgba(139, 216, 255, 0.08), transparent 24%),
            linear-gradient(180deg, rgba(12, 18, 30, 0.96), rgba(7, 12, 22, 0.98));
        }
        .cov-empty {
          padding: 16px;
          color: #92a4bd;
          font-size: 12px;
        }
        @media (max-width: 980px) {
          .traj-main {
            grid-template-columns: 1fr;
            grid-template-rows: minmax(280px, 1fr) minmax(260px, 0.9fr);
          }
          .traj-toolbar select,
          .cov-toolbar select {
            min-width: 0;
            width: 100%;
          }
          .cov-band-grid,
          .cov-two-up {
            grid-template-columns: 1fr;
          }
          .cov-coverage {
            max-width: none;
            max-height: 240px;
          }
        }
      </style>
      <div class="traj-root">
        <div class="traj-toolbar">
          <label>Artifact <select data-role="artifact"></select></label>
          <label class="traj-toggle"><input type="checkbox" data-role="show-train"> Train</label>
          <label class="traj-toggle"><input type="checkbox" data-role="show-test"> Test</label>
          <label class="traj-toggle"><input type="checkbox" data-role="show-points"> Points</label>
          <button type="button" data-role="select-all">All</button>
          <button type="button" data-role="select-none">None</button>
          <button type="button" data-role="reset-view">Reset view</button>
        </div>
        <div class="traj-meta" data-role="meta"></div>
        <div class="traj-status" data-role="status"></div>
        <div class="traj-tabs">
          <button type="button" class="traj-tab" data-role="tab-trajectory">Trajectories</button>
          <button type="button" class="traj-tab" data-role="tab-covariates">Covariates</button>
        </div>
        <div class="traj-body" data-role="trajectory-panel">
          <div class="traj-main">
            <div class="traj-map-wrap">
              <div class="traj-map" data-role="map"></div>
              <div class="traj-overlay" data-role="overlay">
                <div class="traj-overlay-card">
                  <h3>MoveBench trajectories</h3>
                  <p>Select a study artifact from the configured studies directory to explore the trajectory CSV directly.</p>
                </div>
              </div>
            </div>
            <div class="traj-side">
              <div class="traj-side-head">
                <span>Individuals and coverage</span>
                <strong data-role="count"></strong>
              </div>
              <div class="traj-individuals" data-role="individuals"></div>
              <div class="traj-slider-row">
                <input class="traj-slider" data-role="slider" type="range" min="0" max="0" value="0" step="1">
                <div class="traj-time" data-role="time"></div>
              </div>
            </div>
          </div>
        </div>
        <div class="cov-root hidden" data-role="covariates-panel">
          <div class="cov-toolbar">
            <label>Layer config <select data-role="cov-config"></select></label>
            <label>Step <select data-role="cov-point-id"></select></label>
            <button type="button" data-role="cov-load">Load covariates</button>
            <div class="cov-status" data-role="cov-status"></div>
          </div>
          <div class="cov-summary" data-role="cov-summary"></div>
          <div class="cov-content" data-role="cov-content">
            <div class="cov-empty">Open the Covariates tab to load notebook-style raster views for the selected study.</div>
          </div>
        </div>
      </div>
    `;

    this.refs = {
      root: this.mountEl.querySelector(".traj-root"),
      artifact: this.mountEl.querySelector('[data-role="artifact"]'),
      showTrain: this.mountEl.querySelector('[data-role="show-train"]'),
      showTest: this.mountEl.querySelector('[data-role="show-test"]'),
      showPoints: this.mountEl.querySelector('[data-role="show-points"]'),
      selectAll: this.mountEl.querySelector('[data-role="select-all"]'),
      selectNone: this.mountEl.querySelector('[data-role="select-none"]'),
      resetView: this.mountEl.querySelector('[data-role="reset-view"]'),
      meta: this.mountEl.querySelector('[data-role="meta"]'),
      status: this.mountEl.querySelector('[data-role="status"]'),
      count: this.mountEl.querySelector('[data-role="count"]'),
      individuals: this.mountEl.querySelector('[data-role="individuals"]'),
      slider: this.mountEl.querySelector('[data-role="slider"]'),
      time: this.mountEl.querySelector('[data-role="time"]'),
      map: this.mountEl.querySelector('[data-role="map"]'),
      overlay: this.mountEl.querySelector('[data-role="overlay"]'),
      tabTrajectory: this.mountEl.querySelector('[data-role="tab-trajectory"]'),
      tabCovariates: this.mountEl.querySelector('[data-role="tab-covariates"]'),
      trajectoryPanel: this.mountEl.querySelector('[data-role="trajectory-panel"]'),
      covariatesPanel: this.mountEl.querySelector('[data-role="covariates-panel"]'),
      covConfig: this.mountEl.querySelector('[data-role="cov-config"]'),
      covPointId: this.mountEl.querySelector('[data-role="cov-point-id"]'),
      covLoad: this.mountEl.querySelector('[data-role="cov-load"]'),
      covStatus: this.mountEl.querySelector('[data-role="cov-status"]'),
      covSummary: this.mountEl.querySelector('[data-role="cov-summary"]'),
      covContent: this.mountEl.querySelector('[data-role="cov-content"]'),
    };

    this.refs.showTrain.checked = this.uiState.showTrain !== false;
    this.refs.showTest.checked = this.uiState.showTest !== false;
    this.refs.showPoints.checked = this.uiState.showPoints === true;
  }

  bindEvents() {
    this.refs.artifact.addEventListener("change", async () => {
      this.currentArtifact = this.refs.artifact.value;
      await this.loadArtifact();
    });
    this.refs.showTrain.addEventListener("change", () => this.handleVisibilityChange());
    this.refs.showTest.addEventListener("change", () => this.handleVisibilityChange());
    this.refs.showPoints.addEventListener("change", () => {
      this.saveUiState();
      this.renderLayers();
    });
    this.refs.selectAll.addEventListener("click", () => {
      if (!this.data) return;
      this.data.selectedIndividuals = new Set(this.data.individuals);
      this.saveUiState();
      this.renderIndividuals();
      this.renderLayers();
    });
    this.refs.selectNone.addEventListener("click", () => {
      if (!this.data) return;
      this.data.selectedIndividuals = new Set();
      this.saveUiState();
      this.renderIndividuals();
      this.renderLayers();
    });
    this.refs.resetView.addEventListener("click", () => this.resetView());
    this.refs.slider.addEventListener("input", () => {
      this.currentTimeMs = Number(this.refs.slider.value) || 0;
      this.updateTimeLabel();
      this.renderLayers();
    });
    this.refs.tabTrajectory.addEventListener("click", () => {
      void this.setActiveTab("trajectory");
    });
    this.refs.tabCovariates.addEventListener("click", () => {
      void this.setActiveTab("covariates");
    });
    this.refs.covConfig.addEventListener("change", async () => {
      this.currentLayerConfig = this.refs.covConfig.value;
      this.saveUiState();
      await this.loadCovariateSteps(true);
      if (this.activeTab === "covariates") {
        await this.loadCovariates();
      }
    });
    this.refs.covPointId.addEventListener("change", () => {
      this.saveUiState();
    });
    this.refs.covLoad.addEventListener("click", () => {
      void this.loadCovariates();
    });
  }

  async setActiveTab(tab) {
    this.activeTab = tab;
    this.saveUiState();
    this.syncTabVisibility();
    if (tab === "trajectory") {
      this.renderLayers();
      return;
    }
    await this.loadCovariates();
  }

  syncTabVisibility() {
    const showTrajectory = this.activeTab !== "covariates";
    this.refs.root.classList.toggle("mode-trajectory", showTrajectory);
    this.refs.root.classList.toggle("mode-covariates", !showTrajectory);
    this.refs.tabTrajectory.classList.toggle("active", showTrajectory);
    this.refs.tabCovariates.classList.toggle("active", !showTrajectory);
    this.refs.trajectoryPanel.classList.toggle("hidden", !showTrajectory);
    this.refs.covariatesPanel.classList.toggle("hidden", showTrajectory);
  }

  handleVisibilityChange() {
    this.saveUiState();
    this.renderLayers();
    this.renderIndividuals();
  }

  async loadArtifacts() {
    this.setStatus("Loading artifacts...");
    let payload;
    try {
      payload = await fetchJSON("/api/artifacts");
    } catch (error) {
      this.setStatus(error.message, true);
      this.showOverlay("The studies index could not be loaded.");
      return;
    }

    this.studiesRoot = String(payload.studies_root || "");
    this.artifacts = Array.isArray(payload.artifacts) ? payload.artifacts : [];
    this.layerConfigs = Array.isArray(payload.layer_configs) ? payload.layer_configs : [];
    this.refs.meta.innerHTML = this.studiesRoot
      ? `Studies root: <code>${escapeHtml(this.studiesRoot)}</code>`
      : "";

    this.populateArtifacts();
    this.populateLayerConfigs(payload.default_layer_config || "");

    if (!this.artifacts.length) {
      this.setStatus("No artifacts found.", true);
      this.showOverlay("No study artifacts were found under the configured studies directory.");
      return;
    }

    this.currentArtifact = this.artifacts.some(artifact => artifact.id === this.uiState.artifact)
      ? this.uiState.artifact
      : this.artifacts[0].id;
    this.refs.artifact.value = this.currentArtifact;
    await this.loadArtifact();
  }

  populateArtifacts() {
    this.refs.artifact.innerHTML = "";
    for (const artifact of this.artifacts) {
      const option = document.createElement("option");
      option.value = artifact.id;
      option.textContent = artifact.label || artifact.id;
      this.refs.artifact.appendChild(option);
    }
  }

  populateLayerConfigs(defaultConfig) {
    this.refs.covConfig.innerHTML = "";
    for (const config of this.layerConfigs) {
      const option = document.createElement("option");
      option.value = config.path;
      option.textContent = config.label || config.path;
      this.refs.covConfig.appendChild(option);
    }
    this.currentLayerConfig = this.layerConfigs.some(config => config.path === this.uiState.layerConfig)
      ? this.uiState.layerConfig
      : defaultConfig || this.layerConfigs[0]?.path || "";
    this.refs.covConfig.value = this.currentLayerConfig;
    this.refs.covPointId.innerHTML = "";
  }

  async loadArtifact() {
    const requestId = ++this.loadRequestId;
    this.currentArtifactEntry = this.artifacts.find(artifact => artifact.id === this.currentArtifact) || null;
    if (!this.currentArtifactEntry) {
      return;
    }

    this.setStatus(`Loading ${this.currentArtifactEntry.label || this.currentArtifact}...`);
    try {
      const summary = await fetchJSON(`/api/artifacts/${encodeURIComponent(this.currentArtifact)}/summary`);
      if (requestId !== this.loadRequestId) {
        return;
      }

      this.data = buildDatasetFromSummary(summary);
      this.currentTimeMs = this.data.minTimeMs;
      this.refs.slider.min = String(this.data.minTimeMs);
      this.refs.slider.max = String(this.data.maxTimeMs);
      this.refs.slider.value = String(this.currentTimeMs);
      this.data.selectedIndividuals = restoreSelectedIndividuals(this.data.individuals, this.uiState.selectedIndividuals);

      this.covariates = null;
      this.covariateSteps = [];
      this.covariateStepOptionsKey = "";
      this.saveUiState();
      this.renderIndividuals();
      this.updateTimeLabel();
      this.hideOverlay();
      await this.rebuildMap();
      this.resetView();
      this.setStatus(
        `Loaded ${formatCount(this.data.totalRows)} fixes across ${formatCount(this.data.individuals.length)} individuals from ${this.currentArtifactEntry.label || this.currentArtifact}.`,
      );
      if (this.activeTab === "covariates") {
        await this.loadCovariates();
      }
    } catch (error) {
      if (requestId !== this.loadRequestId) {
        return;
      }
      this.data = null;
      this.covariates = null;
      this.refs.count.textContent = "";
      this.renderIndividuals();
      this.renderLayers();
      this.renderCovariates();
      this.setStatus(error.message, true);
      this.showOverlay(`Could not render ${this.currentArtifact}.`);
    }
  }

  async loadCovariateSteps(force = false) {
    if (!this.currentArtifact || !this.currentLayerConfig) {
      this.covariateSteps = [];
      this.populateCovariateSteps([], "");
      return [];
    }

    const cacheKey = `${this.currentArtifact}|${this.currentLayerConfig}`;
    if (!force && this.covariateStepOptionsKey === cacheKey && this.covariateSteps.length) {
      return this.covariateSteps;
    }

    const query = new URLSearchParams({ config: this.currentLayerConfig });
    const preferredPointId = String(this.uiState.pointId || this.refs.covPointId.value || "").trim();
    if (preferredPointId) {
      query.set("point_id", preferredPointId);
    }

    this.setCovariateStatus(`Loading valid steps for ${this.currentArtifact}...`);
    try {
      const payload = await fetchJSON(
        `/api/artifacts/${encodeURIComponent(this.currentArtifact)}/covariates/steps?${query.toString()}`,
      );
      this.currentLayerConfig = payload.config || this.currentLayerConfig;
      this.refs.covConfig.value = this.currentLayerConfig;
      this.covariateStepOptionsKey = cacheKey;
      this.covariateSteps = Array.isArray(payload.steps) ? payload.steps : [];
      this.populateCovariateSteps(this.covariateSteps, String(payload.selected_point_id ?? ""));
      if (!this.covariateSteps.length) {
        this.setCovariateStatus("No valid steps are available for this layer config.", true);
      }
      return this.covariateSteps;
    } catch (error) {
      this.covariateSteps = [];
      this.covariateStepOptionsKey = "";
      this.populateCovariateSteps([], "");
      this.setCovariateStatus(error.message, true);
      return [];
    }
  }

  populateCovariateSteps(steps, selectedPointId) {
    this.refs.covPointId.innerHTML = "";
    if (!steps.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No valid steps";
      this.refs.covPointId.appendChild(option);
      this.refs.covPointId.disabled = true;
      this.refs.covLoad.disabled = true;
      return;
    }

    const preferred = String(selectedPointId || this.uiState.pointId || "");
    for (const step of steps) {
      const option = document.createElement("option");
      option.value = String(step.point_id ?? "");
      option.textContent = step.label || String(step.point_id ?? "");
      this.refs.covPointId.appendChild(option);
    }

    const values = new Set(steps.map(step => String(step.point_id ?? "")));
    this.refs.covPointId.value = values.has(preferred) ? preferred : String(steps[0].point_id ?? "");
    this.refs.covPointId.disabled = false;
    this.refs.covLoad.disabled = false;
    this.saveUiState();
  }

  async loadCovariates() {
    if (!this.currentArtifact) {
      return;
    }
    if (!this.currentLayerConfig) {
      this.setCovariateStatus("No layer config available.", true);
      return;
    }
    const steps = await this.loadCovariateSteps();
    if (!steps.length) {
      this.covariates = null;
      this.renderCovariates();
      return;
    }

    const pointValue = this.refs.covPointId.value.trim();
    const query = new URLSearchParams({ config: this.currentLayerConfig });
    if (pointValue) {
      query.set("point_id", pointValue);
    }

    this.setCovariateStatus(`Loading covariates for ${this.currentArtifact}...`);
    try {
      const payload = await fetchJSON(
        `/api/artifacts/${encodeURIComponent(this.currentArtifact)}/covariates?${query.toString()}`,
      );
      this.covariates = payload;
      this.currentLayerConfig = payload.config || this.currentLayerConfig;
      this.refs.covConfig.value = this.currentLayerConfig;
      this.refs.covPointId.value = String(payload.selected_point?.point_id ?? pointValue);
      this.saveUiState();
      this.renderCovariates();
      this.setCovariateStatus(`Loaded raster grids for point ${payload.selected_point?.point_id ?? "?"}.`);
    } catch (error) {
      this.covariates = null;
      this.renderCovariates();
      this.setCovariateStatus(error.message, true);
    }
  }

  renderCovariates() {
    if (!this.covariates) {
      this.refs.covSummary.innerHTML = "";
      this.refs.covContent.innerHTML = `
        <div class="cov-empty">
          Select a layer config and valid step to load the raster inspection views from
          <code>notebooks/eda_download.ipynb</code>.
        </div>
      `;
      return;
    }

    const selectedPoint = this.covariates.selected_point || {};
    this.refs.covSummary.innerHTML = `
      Study: <code>${escapeHtml(this.covariates.study_id || "")}</code>
      &nbsp;|&nbsp; Config: <code>${escapeHtml(this.covariates.config || "")}</code>
      &nbsp;|&nbsp; Step start: <code>${escapeHtml(String(selectedPoint.point_id ?? ""))}</code>
      &nbsp;|&nbsp; Start time: <code>${escapeHtml(selectedPoint.timestamp || "")}</code>
      ${selectedPoint.step_end_point_id != null ? `&nbsp;|&nbsp; Step end: <code>${escapeHtml(String(selectedPoint.step_end_point_id))}</code>` : ""}
      ${selectedPoint.step_end_timestamp ? `&nbsp;|&nbsp; End time: <code>${escapeHtml(selectedPoint.step_end_timestamp)}</code>` : ""}
      &nbsp;|&nbsp; Rasters hash: <code>${escapeHtml(this.covariates.rasters_hash || "")}</code>
    `;

    const families = buildCovariateFamilies(this.covariates);
    this.refs.covContent.innerHTML = "";
    this.refs.covContent.appendChild(renderSection(
      "Covariate Families",
      families,
      families.map(renderCovariateFamily).join(""),
    ));
  }

  setCovariateStatus(message, isError = false) {
    this.refs.covStatus.textContent = message;
    this.refs.covStatus.classList.toggle("error", isError);
  }

  renderIndividuals() {
    this.refs.individuals.innerHTML = "";
    if (!this.data) {
      return;
    }

    this.refs.count.textContent = `${formatCount(this.data.individuals.length)} shown`;
    for (const individual of this.data.individuals) {
      const stats = this.data.stats[individual];
      const coverage = this.data.coverageByIndividual[individual] || {};
      const isSelected = this.data.selectedIndividuals.has(individual);
      const card = document.createElement("div");
      card.className = "traj-individual";
      card.style.opacity = isSelected ? "1" : "0.34";

      const header = htmlElement(`
        <div class="traj-name">
          <div class="traj-name-left">
            <input type="checkbox" ${isSelected ? "checked" : ""}>
            <span class="traj-name-label">${escapeHtml(individual)}</span>
            ${this.data.speciesByIndividual[individual] ? `<span class="traj-name-species">• ${escapeHtml(this.data.speciesByIndividual[individual])}</span>` : ""}
          </div>
          <span>${escapeHtml(formatCount(stats.rowCount))}</span>
        </div>
      `);
      const checkbox = header.querySelector("input");
      const applySelection = shouldSelect => {
        if (!this.data) return;
        if (shouldSelect) {
          this.data.selectedIndividuals.add(individual);
        } else {
          this.data.selectedIndividuals.delete(individual);
        }
        this.saveUiState();
        this.renderIndividuals();
        this.renderLayers();
      };

      checkbox.addEventListener("click", event => event.stopPropagation());
      checkbox.addEventListener("change", () => applySelection(checkbox.checked));

      card.appendChild(header);
      card.appendChild(htmlElement(`
        <div class="traj-stats">
          <span>median fix ${escapeHtml(formatDuration(stats.medianFixS))}</span>
          <span>median step ${escapeHtml(formatMaybeNumber(stats.medianStepM, "m"))}</span>
          <span>q95 step ${escapeHtml(formatMaybeNumber(stats.p95StepM, "m"))}</span>
        </div>
      `));

      const track = document.createElement("div");
      track.className = "traj-track";
      for (const setName of visibleSets(this.refs.showTrain.checked, this.refs.showTest.checked)) {
        const bar = coverage[setName];
        if (!bar) {
          continue;
        }
        const bounds = rangeToPercent(this.data.minTimeMs, this.data.maxTimeMs, bar.startMs, bar.endMs);
        const el = document.createElement("div");
        el.className = "traj-bar";
        el.style.left = `${bounds.left}%`;
        el.style.width = `${bounds.width}%`;
        el.style.background = colorCss(this.data.palette[individual], setName, isSelected ? 0.88 : 0.25);
        track.appendChild(el);
      }
      card.appendChild(track);
      card.addEventListener("click", () => applySelection(!isSelected));
      this.refs.individuals.appendChild(card);
    }
  }

  async rebuildMap() {
    if (!this.data) {
      this.renderLayers();
      return;
    }

    if (!this.map) {
      this.map = new maplibregl.Map({
        container: this.refs.map,
        style: BASEMAP_STYLE,
        center: [this.data.initialView.longitude, this.data.initialView.latitude],
        zoom: this.data.initialView.zoom,
        attributionControl: false,
      });
      this.map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
      this.map.on("load", () => {
        this.mapLoaded = true;
        this.renderLayers();
      });
      this.overlay = new deck.MapboxOverlay({ interleaved: true, layers: [] });
      this.map.addControl(this.overlay);
      return;
    }

    this.renderLayers();
  }

  renderLayers() {
    if (!this.data || !this.mapLoaded || !this.overlay) {
      if (this.overlay) {
        this.overlay.setProps({ layers: [] });
      }
      return;
    }

    const pathData = [];
    const pointData = [];
    const cursorData = [];
    const visibleIndividuals = this.data.individuals.filter(individual => this.data.selectedIndividuals.has(individual));
    for (const individual of visibleIndividuals) {
      for (const setName of visibleSets(this.refs.showTrain.checked, this.refs.showTest.checked)) {
        const series = this.data.seriesByIndividual[individual]?.[setName];
        if (!series) {
          continue;
        }
        if (series.positions.length >= 2) {
          pathData.push({
            individual,
            setName,
            path: series.positions,
            color: splitColor(this.data.palette[individual], setName, PATH_ALPHA),
          });
        }
        if (this.refs.showPoints.checked) {
          pointData.push(...series.positions.map(position => ({
            individual,
            setName,
            position,
            color: splitColor(this.data.palette[individual], setName, POINT_ALPHA),
          })));
        }
        const cursorPosition = interpolateSeriesPosition(series, this.currentTimeMs);
        if (cursorPosition) {
          cursorData.push({ position: cursorPosition });
        }
      }
    }

    const layers = [
      new deck.PathLayer({
        id: "trajectory-paths",
        data: pathData,
        getPath: item => item.path,
        getColor: item => item.color,
        getWidth: 3,
        widthMinPixels: 2,
        pickable: false,
      }),
    ];

    if (this.refs.showPoints.checked) {
      layers.push(new deck.ScatterplotLayer({
        id: "trajectory-points",
        data: pointData,
        getPosition: item => item.position,
        getFillColor: item => item.color,
        getRadius: 36,
        radiusMinPixels: 2,
        radiusMaxPixels: 6,
        pickable: false,
      }));
    }

    if (cursorData.length) {
      layers.push(new deck.ScatterplotLayer({
        id: "trajectory-cursor",
        data: cursorData,
        getPosition: item => item.position,
        getFillColor: [12, 12, 12, 255],
        getLineColor: [255, 255, 255, 235],
        getRadius: 64,
        radiusMinPixels: 4,
        radiusMaxPixels: 8,
        lineWidthMinPixels: 1.5,
        stroked: true,
        filled: true,
        pickable: false,
      }));
    }

    this.overlay.setProps({ layers });
  }

  resetView() {
    if (!this.data || !this.map) {
      return;
    }
    this.map.jumpTo({
      center: [this.data.initialView.longitude, this.data.initialView.latitude],
      zoom: this.data.initialView.zoom,
      bearing: 0,
      pitch: 0,
    });
  }

  updateTimeLabel() {
    this.refs.time.textContent = this.data ? formatTimestamp(this.currentTimeMs) : "No timestamps";
  }

  getCheckedIndividuals() {
    if (!this.data || !(this.data.selectedIndividuals instanceof Set)) {
      return [];
    }
    return Array.from(this.data.selectedIndividuals).sort((left, right) => left.localeCompare(right));
  }

  setStatus(message, isError = false) {
    this.refs.status.textContent = message;
    this.refs.status.classList.toggle("error", isError);
  }

  showOverlay(message) {
    this.refs.overlay.classList.remove("hidden");
    this.refs.overlay.querySelector("p").textContent = message;
  }

  hideOverlay() {
    this.refs.overlay.classList.add("hidden");
  }
}

function renderSection(title, items, bodyHtml) {
  const body = items.length ? bodyHtml : `<div class="cov-empty">No ${escapeHtml(title.toLowerCase())} available for this layer config.</div>`;
  return htmlElement(`
    <section class="cov-section">
      <div class="cov-section-head">
        <span>${escapeHtml(title)}</span>
        <strong>${escapeHtml(formatCount(items.length))}</strong>
      </div>
      <div class="cov-section-body">${body}</div>
    </section>
  `);
}

function buildCovariateFamilies(covariates) {
  const families = new Map();
  const ensureFamily = (name, kind) => {
    const info = covariateFamilyInfo(name, kind);
    if (!families.has(info.key)) {
      families.set(info.key, {
        key: info.key,
        title: info.title,
        sortOrder: info.sortOrder,
        chipGroups: [],
        tileLayers: [],
        derivedLayers: [],
      });
    }
    return families.get(info.key);
  };

  for (const group of covariates.chip_groups || []) {
    ensureFamily(group.name, "chip_group").chipGroups.push(group);
  }
  for (const layer of covariates.gee_tile_layers || []) {
    ensureFamily(layer.name, "tile_layer").tileLayers.push(layer);
  }
  for (const layer of covariates.derived_layers || []) {
    ensureFamily(layer.name, "derived_layer").derivedLayers.push(layer);
  }

  return Array.from(families.values()).sort((left, right) => {
    const leftPriority = covariateFamilySourcePriority(left);
    const rightPriority = covariateFamilySourcePriority(right);
    if (leftPriority !== rightPriority) {
      return leftPriority - rightPriority;
    }
    if (left.sortOrder !== right.sortOrder) {
      return left.sortOrder - right.sortOrder;
    }
    return left.title.localeCompare(right.title);
  });
}

function covariateFamilySourcePriority(family) {
  if ((family.chipGroups || []).length) {
    return 0;
  }
  if ((family.tileLayers || []).length) {
    return 1;
  }
  if ((family.derivedLayers || []).length) {
    return 2;
  }
  return 3;
}

function covariateFamilyInfo(name, kind) {
  const value = String(name || "").toLowerCase();
  if (value === "landsat" || value.startsWith("landsat_")) {
    return { key: "landsat", title: "Landsat", sortOrder: 10 };
  }
  if (value === "dem" || value === "slope" || value === "aspect" || value === "rugosity") {
    return { key: "dem", title: "DEM", sortOrder: 20 };
  }
  if (value.includes("uplift")) {
    return { key: "uplift", title: "Uplift", sortOrder: 30 };
  }
  if (kind === "chip_group") {
    return { key: `chip:${value}`, title: labelizeCovariateName(value), sortOrder: 90 };
  }
  const base = value.split("_")[0] || value;
  return { key: base, title: labelizeCovariateName(base), sortOrder: 80 };
}

function labelizeCovariateName(name) {
  return String(name || "")
    .split(/[_-]+/)
    .filter(Boolean)
    .map(part => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function renderCovariateFamily(family) {
  const sourceCount = family.chipGroups.length + family.tileLayers.length + family.derivedLayers.length;
  const items = [];

  for (const group of family.chipGroups) {
    for (const band of group.bands || []) {
      items.push(renderBandCard(band, {
        sourceLabel: `Chip group · ${group.name}`,
      }));
    }
  }

  for (const layer of family.tileLayers) {
    for (const band of layer.bands || []) {
      items.push(renderBandCard(band, {
        sourceLabel: `Tile layer · ${layer.name}`,
      }));
    }
    items.push(renderCoverageCard(layer.coverage, {
      sourceLabel: `Tile coverage · ${layer.name}`,
      tileCount: layer.coverage?.tiles?.length || 0,
    }));
  }

  for (const layer of family.derivedLayers) {
    for (const band of layer.bands || []) {
      items.push(renderBandCard(band, {
        sourceLabel: `Derived layer · ${layer.name}`,
      }));
    }
  }

  return `
    <div class="cov-family">
      <div class="cov-family-head">
        <h4>${escapeHtml(family.title)}</h4>
        <div class="cov-family-summary">${escapeHtml(formatCount(sourceCount))} sources</div>
      </div>
      <div class="cov-family-grid">
        <div class="cov-family-stream">${items.join("")}</div>
      </div>
    </div>
  `;
}

function renderBandCard(band, options = {}) {
  return `
    <div class="cov-band-card">
      ${options.sourceLabel ? `<div class="cov-band-source">${escapeHtml(options.sourceLabel)}</div>` : ""}
      <div class="cov-band-title">${escapeHtml(band.band_id || "")}</div>
      <div class="cov-band-meta">
        ${escapeHtml(String(band.width || "?"))} x ${escapeHtml(String(band.height || "?"))}
        · NaN ${escapeHtml(((Number(band.nan_frac) || 0) * 100).toFixed(1))}%
      </div>
      <img class="cov-band-image" loading="lazy" src="${escapeHtml(band.image_url || "")}" alt="${escapeHtml(band.band_id || "raster band")}">
    </div>
  `;
}

function renderCoverageCard(coverage, options = {}) {
  return `
    <div class="cov-coverage-card">
      ${options.sourceLabel ? `<div class="cov-band-source">${escapeHtml(options.sourceLabel)}</div>` : ""}
      <div class="cov-band-title">Coverage</div>
      <div class="cov-coverage-meta">
        ${escapeHtml(formatCount(options.tileCount || 0))} tiles
      </div>
      ${renderCoverageSvg(coverage)}
    </div>
  `;
}

function renderCoverageSvg(coverage) {
  if (!coverage || !coverage.extent) {
    return `<div class="cov-empty">No coverage data available.</div>`;
  }
  const width = 640;
  const height = 480;
  const minLon = Number(coverage.extent.min_lon);
  const maxLon = Number(coverage.extent.max_lon);
  const minLat = Number(coverage.extent.min_lat);
  const maxLat = Number(coverage.extent.max_lat);
  const lonSpan = Math.max(1e-9, maxLon - minLon);
  const latSpan = Math.max(1e-9, maxLat - minLat);
  const projectX = lon => ((Number(lon) - minLon) / lonSpan) * width;
  const projectY = lat => height - (((Number(lat) - minLat) / latSpan) * height);

  const rects = Array.isArray(coverage.tiles)
    ? coverage.tiles.map(tile => {
      const x = projectX(tile.min_lon);
      const y = projectY(tile.max_lat);
      const rectWidth = Math.max(1, projectX(tile.max_lon) - projectX(tile.min_lon));
      const rectHeight = Math.max(1, projectY(tile.min_lat) - projectY(tile.max_lat));
      return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${rectWidth.toFixed(2)}" height="${rectHeight.toFixed(2)}" fill="none" stroke="rgba(139,216,255,0.85)" stroke-width="1.25" />`;
    }).join("")
    : "";

  const points = Array.isArray(coverage.points)
    ? coverage.points.map(point => {
      const cx = projectX(point[0]);
      const cy = projectY(point[1]);
      return `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="1.4" fill="rgba(255,255,255,0.38)" />`;
    }).join("")
    : "";

  return `
    <svg class="cov-coverage" viewBox="0 0 ${width} ${height}" role="img" aria-label="Tile coverage map">
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
      ${rects}
      ${points}
    </svg>
  `;
}

function restoreSelectedIndividuals(individuals, storedSelection) {
  const valid = new Set(individuals);
  const restored = Array.isArray(storedSelection)
    ? storedSelection.filter(individual => valid.has(individual))
    : [];
  return new Set(restored.length ? restored : individuals);
}

function buildDatasetFromSummary(summary) {
  const individuals = Array.isArray(summary.individuals)
    ? [...summary.individuals].sort((left, right) => left.localeCompare(right))
    : [];
  if (!individuals.length) {
    throw new Error("Dataset summary did not contain any individuals.");
  }

  const seriesByIndividual = {};
  const coverageByIndividual = {};
  const stats = {};
  const palette = {};

  Object.entries(summary.series_by_individual || {}).forEach(([individual, sets]) => {
    seriesByIndividual[individual] = {};
    Object.entries(sets || {}).forEach(([setName, series]) => {
      seriesByIndividual[individual][setName] = {
        times: Array.isArray(series?.times) ? series.times.map(value => Number(value) || 0) : [],
        positions: Array.isArray(series?.positions)
          ? series.positions.map(position => [Number(position?.[0]) || 0, Number(position?.[1]) || 0])
          : [],
      };
    });
  });

  Object.entries(summary.coverage_by_individual || {}).forEach(([individual, sets]) => {
    coverageByIndividual[individual] = {};
    Object.entries(sets || {}).forEach(([setName, coverage]) => {
      coverageByIndividual[individual][setName] = {
        startMs: Number(coverage?.start_ms) || 0,
        endMs: Number(coverage?.end_ms) || 0,
      };
    });
  });

  Object.entries(summary.stats || {}).forEach(([individual, item]) => {
    stats[individual] = {
      rowCount: Number(item?.row_count) || 0,
      medianFixS: finiteOrNull(item?.median_fix_s),
      medianStepM: finiteOrNull(item?.median_step_m),
      p95StepM: finiteOrNull(item?.p95_step_m),
    };
  });

  individuals.forEach((individual, index) => {
    if (!stats[individual]) {
      stats[individual] = { rowCount: 0, medianFixS: null, medianStepM: null, p95StepM: null };
    }
    palette[individual] = hslToRgb((index * 137.508) % 360, 0.76, 0.54);
  });

  return {
    totalRows: Number(summary.total_rows) || 0,
    individuals,
    speciesByIndividual: summary.species_by_individual || {},
    seriesByIndividual,
    coverageByIndividual,
    stats,
    palette,
    minTimeMs: Number(summary.min_time_ms) || 0,
    maxTimeMs: Number(summary.max_time_ms) || 0,
    initialView: {
      longitude: Number(summary.initial_view?.longitude) || 0,
      latitude: Number(summary.initial_view?.latitude) || 0,
      zoom: Number(summary.initial_view?.zoom) || 1,
    },
    selectedIndividuals: new Set(individuals),
  };
}

function interpolateSeriesPosition(series, currentTimeMs) {
  if (!series || !Array.isArray(series.times) || !Array.isArray(series.positions) || !series.times.length) {
    return null;
  }
  if (currentTimeMs <= series.times[0]) {
    return series.positions[0];
  }
  const lastIndex = series.times.length - 1;
  if (currentTimeMs >= series.times[lastIndex]) {
    return series.positions[lastIndex];
  }
  for (let index = 1; index < series.times.length; index += 1) {
    const leftTime = series.times[index - 1];
    const rightTime = series.times[index];
    if (currentTimeMs > rightTime) {
      continue;
    }
    const leftPos = series.positions[index - 1];
    const rightPos = series.positions[index];
    const span = rightTime - leftTime;
    if (span <= 0) {
      return rightPos;
    }
    const ratio = (currentTimeMs - leftTime) / span;
    return [
      leftPos[0] + (rightPos[0] - leftPos[0]) * ratio,
      leftPos[1] + (rightPos[1] - leftPos[1]) * ratio,
    ];
  }
  return series.positions[lastIndex];
}

function visibleSets(showTrain, showTest) {
  const sets = [];
  if (showTrain) sets.push("train");
  if (showTest) sets.push("test");
  return sets;
}

function splitColor(baseColor, setName, alpha) {
  const factor = setName === "test" ? 0.7 : 1.12;
  return [
    clampColor(baseColor[0] * factor),
    clampColor(baseColor[1] * factor),
    clampColor(baseColor[2] * factor),
    alpha,
  ];
}

function colorCss(baseColor, setName, alpha) {
  const color = splitColor(baseColor, setName, Math.round(alpha * 255));
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`;
}

function clampColor(value) {
  return Math.max(0, Math.min(255, Math.round(value)));
}

function hslToRgb(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs(hp % 2 - 1));
  let rgb = [0, 0, 0];
  if (hp >= 0 && hp < 1) rgb = [c, x, 0];
  else if (hp < 2) rgb = [x, c, 0];
  else if (hp < 3) rgb = [0, c, x];
  else if (hp < 4) rgb = [0, x, c];
  else if (hp < 5) rgb = [x, 0, c];
  else rgb = [c, 0, x];
  const m = l - c / 2;
  return rgb.map(value => Math.round((value + m) * 255));
}

function rangeToPercent(min, max, start, end) {
  const span = Math.max(1, max - min);
  const left = ((start - min) / span) * 100;
  const width = Math.max(0.8, ((end - start) / span) * 100);
  return { left, width };
}

function formatCount(value) {
  return new Intl.NumberFormat().format(value);
}

function formatTimestamp(value) {
  const date = new Date(value);
  return Number.isFinite(date.getTime()) ? date.toLocaleString() : "Unknown time";
}

function formatMaybeNumber(value, suffix) {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${suffix}`;
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) {
    return "n/a";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(seconds >= 10 ? 1 : 2)} s`;
  }
  if (seconds < 3600) {
    const minutes = seconds / 60;
    return `${minutes.toFixed(minutes >= 10 ? 1 : 2)} min`;
  }
  const hours = seconds / 3600;
  return `${hours.toFixed(hours >= 10 ? 1 : 2)} hr`;
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function htmlElement(html) {
  const template = document.createElement("template");
  template.innerHTML = html.trim();
  return template.content.firstElementChild;
}

async function apiFetch(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }
  return response;
}

async function fetchJSON(url) {
  const response = await apiFetch(url);
  return response.json();
}

function showBootError(message) {
  const mountEl = document.getElementById("app");
  if (!mountEl) {
    return;
  }
  mountEl.innerHTML = `
    <div style="display:grid;place-items:center;height:100%;padding:24px;color:#ffb3c2;font-family:'Segoe UI',sans-serif;">
      <div style="max-width:560px;padding:20px 24px;border-radius:16px;background:rgba(7,11,22,0.92);border:1px solid rgba(255,255,255,0.08);">
        <h2 style="margin:0 0 8px;font-size:18px;">Trajectory app failed to load</h2>
        <p style="margin:0;font-size:13px;line-height:1.5;">${escapeHtml(message)}</p>
      </div>
    </div>
  `;
}

async function main() {
  const mountEl = document.getElementById("app");
  if (!mountEl) {
    throw new Error("Missing #app mount element.");
  }

  const app = new MoveBenchTrajectoryApp({ mountEl });
  window.addEventListener("beforeunload", () => {
    void app.destroy();
  });
  await app.init();
}

main().catch(error => {
  showBootError(error instanceof Error ? error.message : String(error));
});
