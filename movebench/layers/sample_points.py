"""Point-based GEE environmental covariate sampling for movebench.

Unlike the chip-based download (which downloads spatial raster patches requiring a
buffer computed from cleaned movement data), this module samples GEE ImageCollections
at exact GPS coordinates and writes the results to a flat Parquet file:

    study_dir/point_values/<config_hash>/values.parquet
    study_dir/point_values/<config_hash>/metadata.json

Each row corresponds to one GPS point (identified by point_id) and each column
corresponds to one band using dot notation, e.g.:
    era5_land.temperature_2m
    modis_lst.LST_Day_1km

No buffer is required. All raw GPS points, including temporal or spatial outliers,
receive environmental covariates. `get_buffer_m_for_study()` is never called.

Parallelism strategy
--------------------
Points are grouped per-layer by temporal bins: ``(layer, temporal_bucket)``.
Each worker samples one layer for one temporal bucket using ``sampleRegions``
for up to ``batch_size`` points. This keeps temporal semantics layer-specific
while still batching requests to reduce RPC overhead.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from movebench.layers.download import (
    GEE_NODATA_SENTINEL,
    LON_COL,
    LAT_COL,
    TIME_COL,
    SET_COL,
    POINT_ID_COL,
    _fully_masked_image,
    _nearest_from_window,
    _parse_iso_timestamp,
    ensure_ee_initialized,
)
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import LayerRegistry, REGISTRY


# Maximum points per sampleRegions call.  GEE limits FeatureCollection size
# returned by getInfo(); keeping batches ≤500 avoids memory errors.
_DEFAULT_BATCH_SIZE = 500

# Temporal order from finest to coarsest, used for point sampling bins.
_TEMPORAL_ORDER = ["hourly", "daily", "8day", "16day", "32day"]


# ---------------------------------------------------------------------------
# Temporal bucketing helpers
# ---------------------------------------------------------------------------


def _temporal_bucket_and_anchor(ts: pd.Timestamp, temporal: str) -> tuple[str, str]:
    """Return (temporal bucket key, canonical timestamp ISO) for one timestamp.

    Points sharing a bucket key will be batched into one ``sampleRegions`` call
    because they all see the same GEE composite image.

    Args:
        ts:       A pandas Timestamp (UTC-aware or naive).
        temporal: One of "hourly", "daily", "8day", "16day", "32day".

    Returns:
        (bucket_key, timestamp_iso) where ``timestamp_iso`` is the canonical bin
        anchor used when constructing the per-layer GEE image.
    """
    ts_utc = ts if ts.tzinfo is not None else ts.tz_localize("UTC")
    ts_utc = ts_utc.tz_convert("UTC")

    if temporal == "hourly":
        anchor = ts_utc.floor("h")
        return anchor.isoformat(), anchor.isoformat()
    elif temporal == "daily":
        anchor = ts_utc.normalize()
        return anchor.isoformat(), anchor.isoformat()
    elif temporal == "8day":
        # MODIS 8-day composites: images on fixed DOY multiples of 8.
        doy = int(ts_utc.dayofyear)
        period_start_doy = ((doy - 1) // 8) * 8 + 1
        period_start = pd.Timestamp(year=int(ts_utc.year), month=1, day=1, tz="UTC") + pd.Timedelta(days=period_start_doy - 1)
        return period_start.isoformat(), period_start.isoformat()
    elif temporal == "16day":
        doy = int(ts_utc.dayofyear)
        period_start_doy = ((doy - 1) // 16) * 16 + 1
        period_start = pd.Timestamp(year=int(ts_utc.year), month=1, day=1, tz="UTC") + pd.Timedelta(days=period_start_doy - 1)
        return period_start.isoformat(), period_start.isoformat()
    elif temporal == "32day":
        epoch = pd.Timestamp("2014-01-01", tz="UTC")
        days_since = (ts_utc.normalize() - epoch).days
        period_start = epoch + pd.Timedelta(days=(days_since // 32) * 32)
        return period_start.isoformat(), period_start.isoformat()
    else:
        raise ValueError(f"Unsupported temporal={temporal!r} for GEE point sampling")


# ---------------------------------------------------------------------------
# GEE image builder (mirrors _chip_layer_image from download.py)
# ---------------------------------------------------------------------------


def _build_point_layer_image(layer: str, *, spec, timestamp_iso: str, point_geom) -> Any:
    """Build an ``ee.Image`` for one gee_chip layer, adapted for point sampling.

    Mirrors ``_chip_layer_image()`` from ``download.py`` exactly, with the
    following differences:
    * ``filterBounds`` receives ``point_geom`` (an ``ee.Geometry.MultiPoint`` or
      ``ee.Geometry.Point``) instead of a rectangular chip region.
    * ``scale_factor`` is applied inline (same as the chip path).
    * Bands are renamed ``{layer}__{band}`` to avoid collisions when multiple layers
      from different collections are stacked via ``addBands``.

    Args:
        layer:         Layer name (must be a gee_chip layer).
        spec:          LayerSpec for the layer.
        timestamp_iso: ISO 8601 timestamp string (representative for this batch).
        point_geom:    ``ee.Geometry`` covering the batch points, used for
                       ``filterBounds`` where needed.

    Returns:
        ``ee.Image`` with bands named ``{layer}__{band}``, unmasked to
        ``GEE_NODATA_SENTINEL``.
    """
    ee = ensure_ee_initialized()

    if layer == "landsat_ndwi":
        ts = ee.Date(timestamp_iso)
        col = (
            ee.ImageCollection(spec.gee_collection)
            .filterDate(ts.advance(-32, "day"), ts.advance(32, "day"))
            .filterBounds(point_geom)
            .select(spec.bands)
        )
        fallback = _fully_masked_image(spec.bands)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), fallback))
        img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
        return img

    if layer == "dynamic_world":
        DW_START = "2015-06-23"
        ts_dt = _parse_iso_timestamp(timestamp_iso)
        dw_start_dt = datetime(2015, 6, 23, tzinfo=ts_dt.tzinfo)
        was_clamped = ts_dt < dw_start_dt

        if was_clamped:
            col = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(DW_START, "2016-06-23")
                .filterBounds(point_geom)
                .select(spec.bands)
            )
        else:
            ts = ee.Date(timestamp_iso)
            col30 = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(ts.advance(-15, "day"), ts.advance(15, "day"))
                .filterBounds(point_geom)
                .select(spec.bands)
            )
            col60 = (
                ee.ImageCollection(spec.gee_collection)
                .filterDate(ts.advance(-30, "day"), ts.advance(30, "day"))
                .filterBounds(point_geom)
                .select(spec.bands)
            )
            col = ee.ImageCollection(ee.Algorithms.If(col30.size().gt(0), col30, col60))

        fallback = _fully_masked_image(spec.bands)
        img = ee.Image(ee.Algorithms.If(col.size().gt(0), col.median(), fallback))
        img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
        return img

    temporal = spec.temporal
    if temporal == "hourly":
        img = _nearest_from_window(
            spec.gee_collection, timestamp_iso, bands=spec.bands, start=-1, end=1, unit="hour"
        )
    elif temporal == "daily":
        if layer in {"modis_ndsi", "modis_bai"}:
            img = _nearest_from_window(
                spec.gee_collection, timestamp_iso, bands=spec.bands, start=-10, end=10, unit="day"
            )
        else:
            img = _nearest_from_window(
                spec.gee_collection, timestamp_iso, bands=spec.bands, start=-1, end=1, unit="day"
            )
    elif temporal == "8day":
        img = _nearest_from_window(
            spec.gee_collection, timestamp_iso, bands=spec.bands, start=-4, end=4, unit="day"
        )
    elif temporal == "16day":
        img = _nearest_from_window(
            spec.gee_collection, timestamp_iso, bands=spec.bands, start=-10, end=10, unit="day"
        )
    else:
        raise ValueError(f"Unsupported temporal={temporal!r} for gee_chip layer {layer!r}")

    sf = getattr(spec, "scale_factor", 1.0)
    if sf not in (None, 1.0):
        img = img.multiply(float(sf))

    img = img.rename([f"{layer}__{b}" for b in spec.bands]).toFloat()
    return img


# ---------------------------------------------------------------------------
# Request dataclass and worker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PointBatchRequest:
    """A batch of GPS points that share one temporal bin for one gee_chip layer.

    All points in this batch are sampled against the same per-layer GEE image.

    Attributes:
        layer:            Layer name (must be a gee_chip layer).
        temporal_bin_key: Temporal bucket string.
        timestamp_iso:    Canonical timestamp for this bucket, used for
                          ``filterDate`` when building the layer image.
        point_ids:        List of ``point_id`` integers.
        lons:             Corresponding longitudes.
        lats:             Corresponding latitudes.
    """

    layer: str
    temporal_bin_key: str
    timestamp_iso: str
    point_ids: list[int]
    lons: list[float]
    lats: list[float]


def _sample_points_worker(
    args: tuple[int, PointBatchRequest, dict[str, Any]]
) -> dict[str, Any]:
    """Multiprocessing worker: sample one temporal batch for one layer.

    Each worker issues a single ``sampleRegions`` GEE call for all points in the
    batch, then parses the result into per-point band-value records.

    Args:
        args: ``(batch_index, request, context_dict)``
              context_dict keys: "registry" (LayerRegistry)

    Returns:
        dict with keys:
            ``layer``:       str
            ``temporal_bin_key``: str
            ``records``:     list[dict] — one dict per point, keys are
                             ``"point_id"`` and one key per band_id in dot notation.
            ``status``:      ``"ok"`` or ``"failed"``
            ``error``:       str (only when status == "failed")
    """
    _idx, req, ctx = args

    registry: LayerRegistry = ctx["registry"]

    ee = ensure_ee_initialized()

    spec = registry.get(req.layer)
    if spec.source != "gee_chip":
        return {
            "layer": req.layer,
            "temporal_bin_key": req.temporal_bin_key,
            "status": "failed",
            "error": f"layer_is_not_gee_chip: {req.layer}",
            "records": [],
        }
    if spec.native_scale_m is None:
        return {
            "layer": req.layer,
            "temporal_bin_key": req.temporal_bin_key,
            "status": "failed",
            "error": f"missing_native_scale_m: {req.layer}",
            "records": [],
        }
    sample_scale = float(spec.native_scale_m)

    # Build a geometry covering all batch points (for filterBounds).
    coords = [[float(lon), float(lat)] for lon, lat in zip(req.lons, req.lats)]
    if len(coords) == 1:
        point_geom = ee.Geometry.Point(coords[0])
    else:
        point_geom = ee.Geometry.MultiPoint(coords)

    # Band name mapping: GEE-side name → dot-notation band_id in output.
    gee_band_names: list[str] = []   # in stack order
    band_id_map: dict[str, str] = {}  # gee_name → dot-notation

    layer_img = _build_point_layer_image(
        req.layer,
        spec=spec,
        timestamp_iso=req.timestamp_iso,
        point_geom=point_geom,
    ).unmask(GEE_NODATA_SENTINEL).toFloat()
    for band in spec.bands:
        gee_name = f"{req.layer}__{band}"
        gee_band_names.append(gee_name)
        band_id_map[gee_name] = f"{req.layer}.{band}"

    # Build a FeatureCollection with point_id as a property.
    features = [
        ee.Feature(
            ee.Geometry.Point([float(lon), float(lat)]),
            {"point_id": int(pid)},
        )
        for pid, lon, lat in zip(req.point_ids, req.lons, req.lats)
    ]
    fc = ee.FeatureCollection(features)

    def _do_sample():
        return layer_img.sampleRegions(
            collection=fc,
            scale=sample_scale,
            geometries=False,
        ).getInfo()

    try:
        sampled_info = _do_sample()
    except Exception as e:
        time.sleep(5.0)
        try:
            sampled_info = _do_sample()
        except Exception as e2:
            return {
                "layer": req.layer,
                "temporal_bin_key": req.temporal_bin_key,
                "status": "failed",
                "error": f"sampleRegions_failed: {e2}",
                "records": [],
            }

    # Parse sampleRegions result into per-point records.
    # GEE returns a GeoJSON FeatureCollection; each Feature's "properties" dict
    # contains the sampled band values and the "point_id" property we attached.
    features_out = (sampled_info or {}).get("features", [])

    records: list[dict[str, Any]] = []
    for feat in features_out:
        props = feat.get("properties", {})
        pid = int(props.get("point_id", -1))
        record: dict[str, Any] = {"point_id": pid}
        for gee_name in gee_band_names:
            raw = props.get(gee_name)
            val = float(raw) if raw is not None else GEE_NODATA_SENTINEL
            # Convert sentinel to NaN (consistent with LayerIO behaviour).
            if val == GEE_NODATA_SENTINEL:
                val = float("nan")
            record[band_id_map[gee_name]] = val
        records.append(record)

    # GEE silently drops fully-masked pixels; fill gaps with NaN rows.
    returned_pids = {r["point_id"] for r in records}
    for pid in req.point_ids:
        if pid not in returned_pids:
            record = {"point_id": int(pid)}
            for gee_name in gee_band_names:
                record[band_id_map[gee_name]] = float("nan")
            records.append(record)

    return {
        "layer": req.layer,
        "temporal_bin_key": req.temporal_bin_key,
        "status": "ok",
        "records": records,
    }


def _gee_chip_layers_in_plan_order(plan: LayerPlan, registry: LayerRegistry) -> list[str]:
    """Return unique gee_chip layers in plan channel order."""
    out: list[str] = []
    seen: set[str] = set()
    for ch in plan.channels:
        layer = str(ch.layer)
        if layer in seen:
            continue
        if registry.get(layer).source != "gee_chip":
            continue
        out.append(layer)
        seen.add(layer)
    return out


def _build_per_layer_batch_requests(
    valid_df: pd.DataFrame,
    *,
    gee_chip_layers: list[str],
    registry: LayerRegistry,
    batch_size: int,
) -> tuple[list[PointBatchRequest], dict[str, int], dict[str, int]]:
    """Build per-layer temporal-bin requests and per-layer batch/bin counters."""
    batch_requests: list[PointBatchRequest] = []
    per_layer_batch_counts: dict[str, int] = {}
    per_layer_temporal_bin_counts: dict[str, int] = {}

    pid_vals = valid_df[POINT_ID_COL].astype(int).to_list()
    lon_vals = valid_df[LON_COL].astype(float).to_list()
    lat_vals = valid_df[LAT_COL].astype(float).to_list()
    ts_vals = valid_df[TIME_COL].tolist()

    for layer_name in gee_chip_layers:
        spec = registry.get(layer_name)
        temporal = str(spec.temporal)
        if temporal not in _TEMPORAL_ORDER:
            raise ValueError(
                f"Unsupported temporal={temporal!r} for gee_chip layer {layer_name!r}"
            )

        buckets: dict[str, dict[str, Any]] = {}
        for pid, lon, lat, ts in zip(pid_vals, lon_vals, lat_vals, ts_vals):
            bucket_key, timestamp_iso = _temporal_bucket_and_anchor(pd.Timestamp(ts), temporal)
            b = buckets.get(bucket_key)
            if b is None:
                b = {"timestamp_iso": timestamp_iso, "entries": []}
                buckets[bucket_key] = b
            b["entries"].append((int(pid), float(lon), float(lat)))

        per_layer_temporal_bin_counts[layer_name] = len(buckets)
        layer_batch_count = 0

        for bucket_key in sorted(buckets.keys()):
            bucket = buckets[bucket_key]
            entries = list(bucket["entries"])
            entries.sort(key=lambda e: e[0])  # deterministic order by point_id
            timestamp_iso = str(bucket["timestamp_iso"])
            for start in range(0, len(entries), batch_size):
                sub = entries[start : start + batch_size]
                batch_requests.append(
                    PointBatchRequest(
                        layer=layer_name,
                        temporal_bin_key=bucket_key,
                        timestamp_iso=timestamp_iso,
                        point_ids=[int(e[0]) for e in sub],
                        lons=[float(e[1]) for e in sub],
                        lats=[float(e[2]) for e in sub],
                    )
                )
                layer_batch_count += 1

        per_layer_batch_counts[layer_name] = layer_batch_count

    return batch_requests, per_layer_batch_counts, per_layer_temporal_bin_counts


# ---------------------------------------------------------------------------
# Output path helpers
# ---------------------------------------------------------------------------


def _point_values_dir(study_dir: Path, config_hash: str) -> Path:
    """Return the directory for point values for a given config hash.

    Layout: ``study_dir/point_values/<config_hash>/``
    """
    return Path(study_dir) / "point_values" / str(config_hash)


def _parquet_path(study_dir: Path, config_hash: str) -> Path:
    return _point_values_dir(study_dir, config_hash) / "values.parquet"


def _metadata_path(study_dir: Path, config_hash: str) -> Path:
    return _point_values_dir(study_dir, config_hash) / "metadata.json"


# ---------------------------------------------------------------------------
# Skip-existing helpers
# ---------------------------------------------------------------------------


def _load_existing_point_ids(parquet_path: Path) -> set[int] | None:
    """Return the set of point_ids already in the parquet file, or None.

    Returns None if the file is missing or cannot be read.
    """
    if not parquet_path.exists():
        return None
    try:
        existing = pd.read_parquet(parquet_path, columns=["point_id"])
        return set(int(p) for p in existing["point_id"].tolist())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sample_point_values(
    df: pd.DataFrame,
    *,
    study_dir: Path,
    plan: LayerPlan,
    registry: LayerRegistry = REGISTRY,
    splits: list[str] | None = None,
    n_workers: int = 16,
    skip_existing: bool = True,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Sample GEE gee_chip layers at exact GPS coordinates for all points in ``df``.

    Unlike ``download_chips()``, this function:
    * Does **not** require a ``buffer_km`` — no outlier pre-cleaning needed.
    * No step-length or time-gap filtering is applied to any GPS point.
    * Returns scalar covariate values, not raster patches.
    * Writes results to ``study_dir/point_values/<config_hash>/values.parquet``.

    The output Parquet schema::

        point_id              int64       GPS point identifier
        <layer>.<band>        float32     One column per gee_chip band in plan order,
                                          e.g. era5_land.temperature_2m.
                                          NaN for masked / missing pixels.

    A companion ``metadata.json`` is written alongside ``values.parquet``.

    Args:
        df:            DataFrame with at least ``point_id``, ``location-long``,
                       ``location-lat``, ``timestamp`` columns.
        study_dir:     Root directory of the study.
        plan:          Resolved LayerPlan (from ``LayerRegistry.resolve()``).
        registry:      LayerRegistry (default: global REGISTRY).
        splits:        Optional list of split values to process (e.g.
                       ``["train", "test"]``).  If None, all rows are processed.
                       Requires a ``set`` column in ``df`` when specified.
        n_workers:     Number of multiprocessing workers.
        skip_existing: If True and the parquet file already covers all
                       ``point_id`` values in ``df``, return immediately.
                       If the file is partial, only missing points are
                       downloaded and merged.
        batch_size:    Maximum points per ``sampleRegions`` call (default 500).
                       Reduce if GEE returns memory errors.

    Returns:
        dict with keys ``config_hash``, ``parquet_path``, ``n_points``,
        ``n_downloaded``, ``n_existing``, ``n_failed``, ``failures``.
    """
    study_dir = Path(study_dir)
    batch_size = max(1, int(batch_size))

    # ------------------------------------------------------------------
    # 1. Validate inputs and apply split filter
    # ------------------------------------------------------------------
    required_cols = [POINT_ID_COL, LON_COL, LAT_COL, TIME_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"sample_point_values: df is missing required columns: {missing_cols}"
        )

    if splits is not None:
        if SET_COL not in df.columns:
            raise ValueError(
                f"sample_point_values: splits={splits!r} requested but df has no "
                f"{SET_COL!r} column"
            )
        df = df[df[SET_COL].isin(splits)].copy()
        if len(df) == 0:
            raise ValueError(f"No rows matched splits={splits!r}")

    # Identify gee_chip layers present in the plan (stable channel order).
    gee_chip_layers = _gee_chip_layers_in_plan_order(plan, registry)

    if not gee_chip_layers:
        raise ValueError(
            "No gee_chip layers found in plan.channels. "
            "sample_point_values only handles gee_chip layers."
        )

    # All band_ids for gee_chip layers, in plan order.
    gee_chip_band_ids = [
        ch.band_id
        for ch in plan.channels
        if registry.get(ch.layer).source == "gee_chip"
    ]

    # ------------------------------------------------------------------
    # 2. Determine which point_ids need downloading
    # ------------------------------------------------------------------
    parquet_path = _parquet_path(study_dir, plan.config_hash)
    all_point_ids = set(int(p) for p in df[POINT_ID_COL].tolist())

    if skip_existing:
        existing_pids = _load_existing_point_ids(parquet_path)
        if existing_pids is not None and all_point_ids.issubset(existing_pids):
            return {
                "config_hash": plan.config_hash,
                "parquet_path": str(parquet_path),
                "n_points": len(all_point_ids),
                "n_downloaded": 0,
                "n_existing": len(all_point_ids),
                "n_failed": 0,
                "failures": [],
            }
        missing_pids = all_point_ids if existing_pids is None else (all_point_ids - existing_pids)
        n_existing = len(all_point_ids) - len(missing_pids)
    else:
        existing_pids = None
        missing_pids = all_point_ids
        n_existing = 0

    work_df = df[df[POINT_ID_COL].astype(int).isin(missing_pids)].copy()

    # ------------------------------------------------------------------
    # 3. Parse timestamps; separate NaT points (will get all-NaN output)
    # ------------------------------------------------------------------
    work_df[TIME_COL] = pd.to_datetime(work_df[TIME_COL], errors="coerce", utc=True)
    nat_mask = work_df[TIME_COL].isna()
    nat_pids = set(int(p) for p in work_df.loc[nat_mask, POINT_ID_COL].tolist())
    valid_df = work_df[~nat_mask].copy()

    # ------------------------------------------------------------------
    # 4. Group valid points by (layer, temporal_bucket) and build
    #    PointBatchRequest list
    # ------------------------------------------------------------------
    batch_requests, per_layer_batch_counts, per_layer_temporal_bin_counts = _build_per_layer_batch_requests(
        valid_df,
        gee_chip_layers=gee_chip_layers,
        registry=registry,
        batch_size=batch_size,
    )

    # ------------------------------------------------------------------
    # 5. Run workers
    # ------------------------------------------------------------------
    out_dir = _point_values_dir(study_dir, plan.config_hash)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialise EE in the main process before spawning the pool.
    # On Linux, mp.Pool uses fork() so workers inherit the already-initialised
    # EE client and skip re-authentication.  This also surfaces credential
    # errors clearly in the main process rather than as per-batch failures.
    if batch_requests:
        ensure_ee_initialized()

    ctx: dict[str, Any] = {
        "registry": registry,
    }

    from tqdm import tqdm

    worker_args = [(i, req, ctx) for i, req in enumerate(batch_requests)]

    if n_workers > 1 and worker_args:
        with mp.Pool(int(n_workers)) as pool:
            worker_results = list(
                tqdm(
                    pool.imap_unordered(_sample_points_worker, worker_args),
                    total=len(worker_args),
                    desc="Sampling point values",
                    unit="batch",
                )
            )
    else:
        worker_results = [
            _sample_points_worker(a)
            for a in tqdm(worker_args, desc="Sampling point values", unit="batch")
        ]

    # ------------------------------------------------------------------
    # 6. Collect records keyed by (layer, point_id)
    # ------------------------------------------------------------------
    # We collect records per layer separately, then join horizontally.
    records_by_layer: dict[str, dict[int, dict[str, Any]]] = {
        layer: {} for layer in gee_chip_layers
    }

    failures: list[dict[str, Any]] = []
    n_failed_batches = 0

    for result in worker_results:
        layer = result["layer"]
        if result["status"] == "failed":
            n_failed_batches += 1
            failures.append(
                {
                    "layer": layer,
                    "temporal_bin_key": result["temporal_bin_key"],
                    "error": result.get("error", "unknown"),
                }
            )
            continue
        for rec in result["records"]:
            pid = int(rec["point_id"])
            dest = records_by_layer[layer]
            if pid not in dest:
                dest[pid] = {"point_id": pid}
            dest[pid].update(rec)

    # ------------------------------------------------------------------
    # 7. Build output DataFrame
    # ------------------------------------------------------------------
    # All-NaN rows for NaT-timestamp points.
    nat_rows: list[dict[str, Any]] = []
    for pid in nat_pids:
        row: dict[str, Any] = {"point_id": int(pid)}
        for band_id in gee_chip_band_ids:
            row[band_id] = float("nan")
        nat_rows.append(row)

    # Merge band values from all layers into per-point dicts.
    merged: dict[int, dict[str, Any]] = {}
    for pid in missing_pids - nat_pids:
        merged[pid] = {"point_id": int(pid)}
        for layer_name in gee_chip_layers:
            layer_rec = records_by_layer[layer_name].get(pid, {})
            merged[pid].update(layer_rec)

    # Build rows in point_id order.
    rows: list[dict[str, Any]] = []
    for pid in sorted(missing_pids - nat_pids):
        base = merged.get(pid, {"point_id": pid})
        row = {"point_id": int(pid)}
        for band_id in gee_chip_band_ids:
            row[band_id] = float(base.get(band_id, float("nan")))
        rows.append(row)
    rows.extend(nat_rows)

    new_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 8. Merge with existing parquet (partial resume)
    # ------------------------------------------------------------------
    if skip_existing and existing_pids is not None:
        try:
            existing_df = pd.read_parquet(parquet_path)
        except Exception:
            existing_df = pd.DataFrame(columns=["point_id"] + gee_chip_band_ids)

        keep_mask = ~existing_df["point_id"].astype(int).isin(missing_pids)
        keep_existing = existing_df[keep_mask].copy()
        # Ensure column alignment.
        for col in gee_chip_band_ids:
            if col not in keep_existing.columns:
                keep_existing[col] = float("nan")
        combined_df = pd.concat(
            [keep_existing[["point_id"] + gee_chip_band_ids], new_df],
            ignore_index=True,
        )
    else:
        combined_df = new_df

    combined_df = combined_df.sort_values("point_id").reset_index(drop=True)
    combined_df["point_id"] = combined_df["point_id"].astype("int64")
    for col in gee_chip_band_ids:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].astype("float32")
        else:
            combined_df[col] = float("nan")
            combined_df[col] = combined_df[col].astype("float32")

    # Reorder columns: point_id first, then bands in plan order.
    combined_df = combined_df[["point_id"] + gee_chip_band_ids]

    # ------------------------------------------------------------------
    # 9. Atomic write
    # ------------------------------------------------------------------
    tmp_parquet = parquet_path.with_suffix(".parquet.tmp")
    combined_df.to_parquet(tmp_parquet, index=False)
    tmp_parquet.replace(parquet_path)

    # ------------------------------------------------------------------
    # 10. Write metadata.json
    # ------------------------------------------------------------------
    n_downloaded = len(missing_pids - nat_pids)
    metadata: dict[str, Any] = {
        "schema_version": "1",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "config_hash": plan.config_hash,
        "band_ids": gee_chip_band_ids,
        "batching_mode": "per_layer_temporal_bins",
        "n_batches": len(batch_requests),
        "n_unique_temporal_bins": int(sum(per_layer_temporal_bin_counts.values())),
        "per_layer_batch_counts": {k: int(v) for k, v in per_layer_batch_counts.items()},
        "per_layer_temporal_bin_counts": {k: int(v) for k, v in per_layer_temporal_bin_counts.items()},
        "n_points": len(combined_df),
        "n_downloaded": n_downloaded,
        "n_existing": n_existing,
        "n_nat_timestamp": len(nat_pids),
        "n_failed_batches": n_failed_batches,
        "failures": failures[:50],
    }
    meta_path = _metadata_path(study_dir, plan.config_hash)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return {
        "config_hash": plan.config_hash,
        "parquet_path": str(parquet_path),
        "n_points": len(all_point_ids),
        "n_downloaded": n_downloaded,
        "n_existing": n_existing,
        "n_failed": n_failed_batches,
        "failures": failures,
    }
