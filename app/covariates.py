from __future__ import annotations

import io
from typing import Any
from urllib.parse import urlencode

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import colormaps
from movebench.layers import REGISTRY
from movebench.study import Study
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    fit_interval_gap_cutoffs,
    fit_interval_step_length_cutoff_m,
    select_step_starts,
)
from PIL import Image

from app.configs import REPO_ROOT, resolve_layer_config


matplotlib.use("Agg")

MAX_BANDS_DISPLAY = 16
MAX_COVERAGE_POINTS = 2500


def _study_frame(study_id: str) -> tuple[Study, pd.DataFrame]:
    study = Study(str(study_id))
    df = pd.read_csv(study.trajectories_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "point_id" not in df.columns:
        raise ValueError("Study trajectories.csv is missing point_id")
    return study, df


def _step_time_seconds(plan) -> int | None:
    step = getattr(plan, "step", None)
    if step is None:
        return None
    return int(step.step_time_s)


def _selectable_rows(df: pd.DataFrame, plan) -> pd.DataFrame:
    step_time_s = _step_time_seconds(plan)
    if step_time_s is None:
        return df.copy()

    step = plan.step_spec
    train_df = (
        df[df[SET_COL] == "train"].copy()
        if SET_COL in df.columns
        else df.iloc[0:0]
    )
    fit_df = train_df if len(train_df) else df
    if len(fit_df) == 0:
        raise ValueError("No trajectory rows available to prepare interval steps")

    gap_cutoffs = fit_interval_gap_cutoffs(fit_df, gap_q=float(step.gap_q))
    max_step_length_m = fit_interval_step_length_cutoff_m(
        fit_df,
        step_time_s=int(step.step_time_s),
        gap_q=float(step.gap_q),
        gap_cutoffs=gap_cutoffs,
        step_q=float(step.step_q),
        max_step_q=float(step.max_step_q),
        step_multiplier=float(step.step_multiplier),
    )
    starts = select_step_starts(
        df,
        step_time_s=int(step.step_time_s),
        gap_q=float(step.gap_q),
        gap_cutoffs=gap_cutoffs,
        max_step_length_m=float(max_step_length_m),
    )
    if len(starts) == 0:
        raise ValueError(
            "No valid interval steps available for the selected layer config"
        )

    out = pd.DataFrame(
        {
            POINT_ID_COL: starts["point_id_start"].astype(int),
            "step_end_point_id": starts["point_id_end"].astype(int),
            "timestamp": starts["timestamp"],
            "step_end_timestamp": starts["timestamp_end"],
            LON_COL: starts["start_lon"].astype(float),
            LAT_COL: starts["start_lat"].astype(float),
            SET_COL: starts[SET_COL].astype(object),
            INDIVIDUAL_COL: starts[INDIVIDUAL_COL].astype(object),
        }
    )
    return out


def _default_point_row(df: pd.DataFrame) -> pd.Series:
    train_df = df[df["set"] == "train"] if "set" in df.columns else df.iloc[0:0]
    source = train_df if len(train_df) else df
    if len(source) == 0:
        raise ValueError("Study did not contain any usable trajectory rows")
    index = 100 if len(source) > 100 else len(source) // 2
    return source.iloc[int(index)]


def _selected_point_row(df: pd.DataFrame, point_id: int | None) -> pd.Series:
    if point_id is None:
        return _default_point_row(df)
    matches = df[df["point_id"] == int(point_id)]
    if len(matches) == 0:
        raise ValueError(f"Unknown point_id: {point_id}")
    return matches.iloc[0]


def _point_payload(row: pd.Series) -> dict[str, Any]:
    timestamp = row["timestamp"]
    end_point_id = row.get("step_end_point_id")
    end_timestamp = row.get("step_end_timestamp")
    return {
        "point_id": int(row["point_id"]),
        "timestamp": str(timestamp.isoformat()),
        "lon": float(row["location-long"]),
        "lat": float(row["location-lat"]),
        "set": str(row.get("set", "")),
        "individual": str(
            row.get("individual-local-identifier")
            or row.get("individual")
            or row.get("animal-id")
            or ""
        ),
        "step_end_point_id": int(end_point_id) if pd.notna(end_point_id) else None,
        "step_end_timestamp": (
            str(pd.Timestamp(end_timestamp).isoformat()) if pd.notna(end_timestamp) else None
        ),
    }


def _step_options_payload(df: pd.DataFrame) -> list[dict[str, Any]]:
    options = []
    ordered = df.sort_values("timestamp").reset_index(drop=True)
    for _, row in ordered.iterrows():
        point_id = int(row["point_id"])
        timestamp = pd.Timestamp(row["timestamp"])
        individual = str(
            row.get("individual-local-identifier", "")
            or row.get("individual", "")
            or row.get("animal-id", "")
            or ""
        )
        split = str(row.get("set", "") or "")
        end_point_id = row.get("step_end_point_id")
        end_timestamp = row.get("step_end_timestamp")
        label_parts = [str(point_id)]
        if individual:
            label_parts.append(individual)
        if split:
            label_parts.append(split)
        label_parts.append(str(timestamp.isoformat()))
        if pd.notna(end_timestamp):
            label_parts.append(f"-> {pd.Timestamp(end_timestamp).isoformat()}")
        options.append({
            "point_id": point_id,
            "timestamp": str(timestamp.isoformat()),
            "individual": individual,
            "set": split,
            "step_end_point_id": int(end_point_id) if pd.notna(end_point_id) else None,
            "step_end_timestamp": (
                str(pd.Timestamp(end_timestamp).isoformat()) if pd.notna(end_timestamp) else None
            ),
            "label": " | ".join(part for part in label_parts if part),
        })
    return options


def _robust_vmin_vmax(x: np.ndarray, p_lo: int = 2, p_hi: int = 98) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(x, [p_lo, p_hi]).tolist()
    if float(lo) == float(hi):
        hi = float(lo) + 1.0
    return float(lo), float(hi)


def _band_to_png_bytes(band: np.ndarray, *, cmap_name: str = "viridis") -> bytes:
    band = np.asarray(band, dtype=np.float32)
    mask = np.isfinite(band)
    vmin, vmax = _robust_vmin_vmax(band)
    if vmax <= vmin:
        vmax = vmin + 1.0
    normalized = np.clip((np.nan_to_num(band, nan=vmin) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = (colormaps[cmap_name](normalized) * 255).astype(np.uint8)
    rgba[~mask, 3] = 0
    image = Image.fromarray(rgba, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sample_points(df: pd.DataFrame, *, limit: int = MAX_COVERAGE_POINTS) -> list[list[float]]:
    coords = df[["location-long", "location-lat"]].dropna().to_numpy(dtype=float)
    if len(coords) <= limit:
        return [[float(lon), float(lat)] for lon, lat in coords]
    step = max(1, len(coords) // limit)
    sampled = coords[::step][:limit]
    return [[float(lon), float(lat)] for lon, lat in sampled]


def _coverage_payload(df: pd.DataFrame, tiles: list[dict[str, Any]]) -> dict[str, Any]:
    points = _sample_points(df)
    lon_values = [point[0] for point in points]
    lat_values = [point[1] for point in points]
    for tile in tiles:
        bounds = tile["bounds"]
        lon_values.extend([float(bounds["min_lon"]), float(bounds["max_lon"])])
        lat_values.extend([float(bounds["min_lat"]), float(bounds["max_lat"])])

    if lon_values and lat_values:
        extent = {
            "min_lon": min(lon_values),
            "max_lon": max(lon_values),
            "min_lat": min(lat_values),
            "max_lat": max(lat_values),
        }
    else:
        extent = {"min_lon": -180.0, "max_lon": 180.0, "min_lat": -90.0, "max_lat": 90.0}

    return {
        "points": points,
        "tiles": [
            {
                "min_lon": float(tile["bounds"]["min_lon"]),
                "max_lon": float(tile["bounds"]["max_lon"]),
                "min_lat": float(tile["bounds"]["min_lat"]),
                "max_lat": float(tile["bounds"]["max_lat"]),
            }
            for tile in tiles
        ],
        "extent": extent,
    }


def _query(config: str, point_id: int) -> str:
    return urlencode({"config": config, "point_id": point_id})


def _band_payloads(*, base_url: str, config: str, point_id: int, raster) -> list[dict[str, Any]]:
    array = np.asarray(raster.array)
    show = min(int(array.shape[0]), MAX_BANDS_DISPLAY)
    query = _query(config, point_id)
    payloads = []
    for index in range(show):
        band = array[index]
        payloads.append({
            "index": index,
            "band_id": str(raster.band_ids[index]),
            "height": int(band.shape[0]),
            "width": int(band.shape[1]),
            "nan_frac": float(np.isnan(band).mean()) if np.issubdtype(band.dtype, np.floating) else 0.0,
            "image_url": f"{base_url}/bands/{index}.png?{query}",
        })
    return payloads


def _covariate_context(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
) -> dict[str, Any]:
    config_value, config_path = resolve_layer_config(config, repo_root=REPO_ROOT)
    study, df = _study_frame(study_id)
    plan = study.layer_plan(config_path)
    selectable_df = _selectable_rows(df, plan)
    row = _selected_point_row(selectable_df, point_id)
    point = _point_payload(row)
    io_obj = study.layer_io(config_path)
    gee_tile_layers = [layer for layer in plan.layers_in_order if REGISTRY.get(layer).source == "gee_tile"]
    derived_layers = [layer for layer in plan.layers_in_order if REGISTRY.get(layer).source == "derived"]

    return {
        "study": study,
        "df": df,
        "selectable_df": selectable_df,
        "row": row,
        "point": point,
        "plan": plan,
        "io": io_obj,
        "config": config_value,
        "gee_tile_layers": gee_tile_layers,
        "derived_layers": derived_layers,
    }


def _bounds_latlon_for_point(io_obj, plan, *, point_id: int) -> tuple[float, float, float, float]:
    chip_group_name = next(iter(plan.chip_groups.keys()), None)
    if chip_group_name is None:
        raise ValueError("Selected layer config did not define any chip groups")
    chip_group = io_obj.load_chip_group(chip_group_name, point_id=point_id)
    return tuple(map(float, chip_group.bounds))


def build_covariates_payload(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
) -> dict[str, Any]:
    ctx = _covariate_context(study_id, config=config, point_id=point_id)
    point = ctx["point"]
    config_value = ctx["config"]
    study = ctx["study"]
    plan = ctx["plan"]
    io_obj = ctx["io"]
    row = ctx["row"]
    bounds_latlon = _bounds_latlon_for_point(io_obj, plan, point_id=point["point_id"])

    base = f"/api/artifacts/{study_id}/covariates"

    chip_groups = []
    for group_name in plan.chip_groups.keys():
        raster = io_obj.load_chip_group(group_name, point_id=point["point_id"])
        array = np.asarray(raster.array)
        chip_groups.append({
            "name": group_name,
            "total_bands": int(array.shape[0]),
            "shown_bands": min(int(array.shape[0]), MAX_BANDS_DISPLAY),
            "shape": list(array.shape),
            "bands": _band_payloads(
                base_url=f"{base}/chip-groups/{group_name}",
                config=config_value,
                point_id=point["point_id"],
                raster=raster,
            ),
        })

    gee_tile_layers = []
    for layer in ctx["gee_tile_layers"]:
        spec = REGISTRY.get(layer)
        raster = io_obj.load_layer(
            layer,
            bounds_latlon=bounds_latlon,
            timestamp=row["timestamp"],
        )
        idx = io_obj._tile_index(layer, timestamp=row["timestamp"])
        gee_tile_layers.append({
            "name": layer,
            "total_bands": len(spec.bands),
            "shown_bands": min(len(spec.bands), MAX_BANDS_DISPLAY),
            "bands": _band_payloads(
                base_url=f"{base}/tile-layers/{layer}",
                config=config_value,
                point_id=point["point_id"],
                raster=raster,
            ),
            "coverage": _coverage_payload(ctx["df"], idx.get("tiles", [])),
        })

    derived_layers = []
    for layer in ctx["derived_layers"]:
        raster = io_obj.load_layer(
            layer,
            point_id=point["point_id"],
            bounds_latlon=bounds_latlon,
            timestamp=row["timestamp"],
        )
        derived_layers.append({
            "name": layer,
            "total_bands": int(np.asarray(raster.array).shape[0]),
            "shown_bands": min(int(np.asarray(raster.array).shape[0]), MAX_BANDS_DISPLAY),
            "bands": _band_payloads(
                base_url=f"{base}/derived-layers/{layer}",
                config=config_value,
                point_id=point["point_id"],
                raster=raster,
            ),
        })

    selectable_point_ids = sorted(
        int(value) for value in pd.Series(ctx["selectable_df"]["point_id"]).dropna().unique().tolist()
    )
    return {
        "study_id": study.study_id,
        "study_dir": str(study.study_dir),
        "config": config_value,
        "rasters_hash": plan.rasters_hash,
        "step_time_s": _step_time_seconds(plan),
        "selected_point": {
            **point,
            "bounds_latlon": list(bounds_latlon),
        },
        "point_id_min": selectable_point_ids[0] if selectable_point_ids else None,
        "point_id_max": selectable_point_ids[-1] if selectable_point_ids else None,
        "available_steps_count": len(ctx["selectable_df"]),
        "chip_groups": chip_groups,
        "gee_tile_layers": gee_tile_layers,
        "derived_layers": derived_layers,
    }


def build_covariate_steps_payload(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
) -> dict[str, Any]:
    ctx = _covariate_context(study_id, config=config, point_id=point_id)
    steps = _step_options_payload(ctx["selectable_df"])
    return {
        "study_id": ctx["study"].study_id,
        "config": ctx["config"],
        "rasters_hash": ctx["plan"].rasters_hash,
        "step_time_s": _step_time_seconds(ctx["plan"]),
        "selected_point_id": int(ctx["point"]["point_id"]),
        "available_steps_count": len(steps),
        "steps": steps,
    }


def chip_group_band_png(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
    group_name: str,
    band_index: int,
) -> bytes:
    ctx = _covariate_context(study_id, config=config, point_id=point_id)
    raster = ctx["io"].load_chip_group(group_name, point_id=ctx["point"]["point_id"])
    array = np.asarray(raster.array)
    index = int(band_index)
    if index < 0 or index >= array.shape[0]:
        raise ValueError(f"Unknown band index: {band_index}")
    return _band_to_png_bytes(array[index])


def tile_layer_band_png(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
    layer: str,
    band_index: int,
) -> bytes:
    ctx = _covariate_context(study_id, config=config, point_id=point_id)
    bounds_latlon = _bounds_latlon_for_point(
        ctx["io"],
        ctx["plan"],
        point_id=ctx["point"]["point_id"],
    )
    raster = ctx["io"].load_layer(
        layer,
        bounds_latlon=bounds_latlon,
        timestamp=ctx["row"]["timestamp"],
    )
    array = np.asarray(raster.array)
    index = int(band_index)
    if index < 0 or index >= array.shape[0]:
        raise ValueError(f"Unknown band index: {band_index}")
    return _band_to_png_bytes(array[index])


def derived_layer_band_png(
    study_id: str,
    *,
    config: str | None,
    point_id: int | None,
    layer: str,
    band_index: int,
) -> bytes:
    ctx = _covariate_context(study_id, config=config, point_id=point_id)
    bounds_latlon = _bounds_latlon_for_point(
        ctx["io"],
        ctx["plan"],
        point_id=ctx["point"]["point_id"],
    )
    raster = ctx["io"].load_layer(
        layer,
        point_id=ctx["point"]["point_id"],
        bounds_latlon=bounds_latlon,
        timestamp=ctx["row"]["timestamp"],
    )
    array = np.asarray(raster.array)
    index = int(band_index)
    if index < 0 or index >= array.shape[0]:
        raise ValueError(f"Unknown band index: {band_index}")
    return _band_to_png_bytes(array[index])
