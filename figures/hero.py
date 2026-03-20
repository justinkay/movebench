# %% [markdown]
# # Hero Map
#
# Run each `# %%` block with `Shift+Enter` in VS Code.

# %%
from __future__ import annotations

import csv
import hashlib
import math
import os
import pickle
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import shapereader
from matplotlib.collections import LineCollection
from shapely.geometry import Point
from shapely.prepared import prep


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "pngs" / "hero_atlas_blue_log.png"
OUTPUT_PATH.parent.mkdir(exist_ok=True)
CACHE_DIR = REPO_ROOT / ".cache" / "hero"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
COUNTRY_POINTS_CACHE_VERSION = "v1"
TRAJECTORY_OVERLAY_CACHE_VERSION = "v8"

MARINE_SPECIES = {
    "Chelonia mydas",
    "Eretmochelys imbricata",
    "Eumetopias jubatus",
    "Halichoerus grypus",
    "Mirounga angustirostris",
    "Mobula alfredi",
    "Physeter macrocephalus",
}

_COUNTRY_RECORDS_CACHE: list[dict[str, object]] | None = None
_GPS_POINT_COUNTS_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
_TRAJECTORY_OVERLAY_CACHE: dict[tuple[str, float], dict[str, object]] = {}
EPOCH = datetime(1970, 1, 1)


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value.strip()
    return value.split("#", 1)[0].strip()


def _cache_digest(namespace: str, version: str, *parts: object) -> str:
    raw = "|".join([namespace, version, *(str(part) for part in parts)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _country_points_cache_path(studies_root: Path, point_round_decimals: int) -> Path:
    return CACHE_DIR / (
        "country_gps_points_"
        f"{_cache_digest('country_points', COUNTRY_POINTS_CACHE_VERSION, studies_root.resolve(), point_round_decimals)}.pkl"
    )


def _trajectory_overlay_cache_path(
    studies_root: Path,
    outlier_q: float,
) -> Path:
    return CACHE_DIR / (
        "trajectory_overlay_"
        f"{_cache_digest('trajectory_overlay', TRAJECTORY_OVERLAY_CACHE_VERSION, studies_root.resolve(), outlier_q)}.pkl"
    )


def _load_default_studies_root() -> Path:
    data_root = os.environ.get("MOVEBENCH_DATA_ROOT")
    if data_root:
        return Path(data_root) / "studies"

    config_path = REPO_ROOT / "configs" / "paths.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find default paths config: {config_path}")

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() != "data_root":
            continue
        cleaned = _strip_inline_comment(value).strip().strip("'\"")
        if cleaned:
            return Path(cleaned) / "studies"
        break

    raise ValueError(f"Could not resolve data_root from {config_path}")


def _study_dirs(studies_root: Path) -> list[Path]:
    if not studies_root.exists():
        raise FileNotFoundError(f"Studies root not found: {studies_root}")

    return sorted(
        [
            path
            for path in studies_root.iterdir()
            if path.is_dir() and (path / "trajectories.csv").exists()
        ],
        key=lambda path: path.name,
    )


def _build_light_blue_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "movebench_light_blue",
        ["#EAF4FB", "#D5EAF7", "#B8D9EE", "#8EC0DF", "#5F9FC8"],
    )


def _hide_geo_outline(ax: plt.Axes) -> None:
    geo_spine = ax.spines.get("geo")
    if geo_spine is not None:
        geo_spine.set_visible(False)


def _in_ipython_session() -> bool:
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return get_ipython() is not None


def _parse_timestamp(raw: str) -> datetime | None:
    raw = raw.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _seconds_since_epoch(dt: datetime) -> float:
    return (dt - EPOCH).total_seconds()


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _haversine_distance_m(
    start_lat_deg: float,
    start_lon_deg: float,
    end_lat_deg: float,
    end_lon_deg: float,
) -> float:
    start_lat = math.radians(start_lat_deg)
    start_lon = math.radians(start_lon_deg)
    end_lat = math.radians(end_lat_deg)
    end_lon = math.radians(end_lon_deg)

    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(start_lat) * math.cos(end_lat) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return 6_371_000.0 * c


def _clamp_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12 or norm <= max_norm:
        return vector
    return vector * (max_norm / norm)


def _build_smooth_gap_bridge(
    points: list[tuple[float, float, float]],
    idx: int,
    *,
    gap_cutoff_s: float,
    step_cutoff_m: float,
    speed_cutoff_mps: float,
) -> np.ndarray | None:
    if idx < 1 or idx >= len(points):
        return None

    prev_time, prev_lon, prev_lat = points[idx - 1]
    cur_time, cur_lon, cur_lat = points[idx]
    prev_prev_point = points[idx - 2] if idx >= 2 else None
    next_point = points[idx + 1] if idx + 1 < len(points) else None

    if abs(cur_lon - prev_lon) > 180.0:
        return None
    if prev_prev_point is not None and abs(prev_lon - prev_prev_point[1]) > 180.0:
        return None
    if next_point is not None and abs(next_point[1] - cur_lon) > 180.0:
        return None

    gap_s = cur_time - prev_time
    if gap_s <= gap_cutoff_s or gap_s > gap_cutoff_s * 20.0:
        return None

    bridge_step_m = _haversine_distance_m(prev_lat, prev_lon, cur_lat, cur_lon)
    if not math.isfinite(bridge_step_m) or bridge_step_m <= 0.0:
        return None

    bridge_speed_mps = bridge_step_m / gap_s
    typical_speed_mps = max(speed_cutoff_mps, step_cutoff_m / max(gap_cutoff_s, 1.0))
    if bridge_speed_mps > typical_speed_mps * 2.6:
        return None
    if bridge_step_m > max(step_cutoff_m * 12.0, typical_speed_mps * gap_s * 2.75):
        return None

    start = np.asarray([prev_lon, prev_lat], dtype=float)
    end = np.asarray([cur_lon, cur_lat], dtype=float)

    segment_length = float(np.linalg.norm(end - start))
    if segment_length <= 1e-9:
        return None

    chord = end - start
    tangent_start = 0.45 * chord
    tangent_end = 0.45 * chord

    if prev_prev_point is not None:
        before = np.asarray([prev_prev_point[1], prev_prev_point[2]], dtype=float)
        tangent_start = 0.5 * (end - before)
    elif next_point is not None:
        after = np.asarray([next_point[1], next_point[2]], dtype=float)
        tangent_start = 0.35 * (after - start)

    if next_point is not None:
        after = np.asarray([next_point[1], next_point[2]], dtype=float)
        tangent_end = 0.5 * (after - start)
    elif prev_prev_point is not None:
        before = np.asarray([prev_prev_point[1], prev_prev_point[2]], dtype=float)
        tangent_end = 0.35 * (end - before)

    max_tangent_norm = segment_length * 1.35
    tangent_start = _clamp_vector_norm(tangent_start, max_tangent_norm)
    tangent_end = _clamp_vector_norm(tangent_end, max_tangent_norm)

    interior_count = max(4, min(18, int(math.ceil(gap_s / gap_cutoff_s)) + 3))
    t_values = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
    t = t_values[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    bridge_points = h00 * start + h10 * tangent_start + h01 * end + h11 * tangent_end
    return bridge_points.astype(float)


def _read_study_species_name(study_dir: Path) -> str | None:
    with (study_dir / "trajectories.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            species_name = (
                row.get("individual-taxon-canonical-name")
                or row.get("individual_taxon_canonical_name")
                or ""
            ).strip()
            if species_name:
                return species_name
    return None


def _load_country_records() -> list[dict[str, object]]:
    global _COUNTRY_RECORDS_CACHE

    if _COUNTRY_RECORDS_CACHE is not None:
        return _COUNTRY_RECORDS_CACHE

    shapefile_path = shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )

    country_records: list[dict[str, object]] = []
    for record in shapereader.Reader(shapefile_path).records():
        geometry = record.geometry
        country_name = (
            record.attributes.get("NAME_LONG")
            or record.attributes.get("NAME_EN")
            or record.attributes.get("ADMIN")
            or "Unknown"
        )
        country_records.append(
            {
                "country_name": country_name,
                "geometry": geometry,
                "prepared_geometry": prep(geometry),
                "bounds": geometry.bounds,
            }
        )

    _COUNTRY_RECORDS_CACHE = country_records
    return country_records


def _assign_country_to_point(
    lon: float,
    lat: float,
    country_records: list[dict[str, object]],
    *,
    marine_credit_nearest: bool,
) -> tuple[str | None, bool]:
    point = Point(lon, lat)

    for country in country_records:
        min_lon, min_lat, max_lon, max_lat = country["bounds"]
        if lon < min_lon or lon > max_lon or lat < min_lat or lat > max_lat:
            continue
        if country["prepared_geometry"].intersects(point):
            return str(country["country_name"]), False

    if not marine_credit_nearest:
        return None, False

    nearest_country_name: str | None = None
    nearest_distance = np.inf
    for country in country_records:
        distance = point.distance(country["geometry"])
        if distance < nearest_distance:
            nearest_distance = float(distance)
            nearest_country_name = str(country["country_name"])

    return nearest_country_name, nearest_country_name is not None


def load_country_gps_point_counts(
    studies_root: Path | None = None,
    *,
    point_round_decimals: int = 4,
) -> pd.DataFrame:
    if studies_root is None:
        studies_root = _load_default_studies_root()

    cache_key = (str(studies_root.resolve()), point_round_decimals)
    cached = _GPS_POINT_COUNTS_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    cache_path = _country_points_cache_path(studies_root, point_round_decimals)
    if cache_path.exists():
        summary = pd.read_pickle(cache_path)
        _GPS_POINT_COUNTS_CACHE[cache_key] = summary.copy()
        return summary.copy()

    country_records = _load_country_records()
    gps_point_counts: dict[str, int] = defaultdict(int)
    point_assignment_cache: dict[tuple[float, float, bool], str | None] = {}

    for study_dir in _study_dirs(studies_root):
        study_species_name = _read_study_species_name(study_dir) or ""
        is_marine_study = study_species_name in MARINE_SPECIES

        with (study_dir / "trajectories.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    lat = float(row.get("location-lat", ""))
                    lon = float(row.get("location-long", ""))
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue

                cache_point = (
                    round(lon, point_round_decimals),
                    round(lat, point_round_decimals),
                    is_marine_study,
                )
                if cache_point not in point_assignment_cache:
                    country_name, _ = _assign_country_to_point(
                        float(cache_point[0]),
                        float(cache_point[1]),
                        country_records,
                        marine_credit_nearest=is_marine_study,
                    )
                    point_assignment_cache[cache_point] = country_name

                country_name = point_assignment_cache[cache_point]
                if country_name is None:
                    continue
                gps_point_counts[country_name] += 1

    rows = [
        {"country_name": country["country_name"], "gps_point_count": gps_point_counts[str(country["country_name"])]}
        for country in country_records
        if gps_point_counts[str(country["country_name"])] > 0
    ]
    if rows:
        summary = pd.DataFrame(rows).sort_values(
            ["gps_point_count", "country_name"], ascending=[False, True]
        ).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=["country_name", "gps_point_count"])
    _GPS_POINT_COUNTS_CACHE[cache_key] = summary.copy()
    summary.to_pickle(cache_path)
    return summary


def plot_hero_map(
    country_summary: pd.DataFrame | None = None,
    *,
    output_path: Path = OUTPUT_PATH,
) -> tuple[plt.Figure, plt.Axes]:
    if country_summary is None:
        country_summary = load_country_gps_point_counts()

    country_records = _load_country_records()
    country_df = pd.DataFrame(country_records)[["country_name", "geometry"]]
    plot_df = country_df.merge(country_summary, on="country_name", how="left")
    plot_df["gps_point_count"] = plot_df["gps_point_count"].fillna(0)

    positive_values = plot_df.loc[plot_df["gps_point_count"] > 0, "gps_point_count"]
    max_value = float(positive_values.max()) if not positive_values.empty else 1.0

    cmap = _build_light_blue_cmap()
    norm = (
        mcolors.LogNorm(vmin=1, vmax=max_value)
        if max_value > 1
        else mcolors.Normalize(vmin=0, vmax=max_value)
    )

    fig = plt.figure(figsize=(18, 8.5), facecolor="#FFFFFF")
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_facecolor("#FFFFFF")
    ax.set_global()
    _hide_geo_outline(ax)

    for row in plot_df.itertuples(index=False):
        value = float(row.gps_point_count)
        facecolor = "#FFFFFF"
        if value > 0:
            facecolor = cmap(norm(value))
        ax.add_geometries(
            [row.geometry],
            crs=ccrs.PlateCarree(),
            facecolor=facecolor,
            edgecolor="#D3E1EC",
            linewidth=0.35,
            zorder=2,
        )

    if output_path is not None:
        fig.savefig(
            output_path,
            dpi=220,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )

    return fig, ax


def load_trajectory_overlay_data(
    studies_root: Path | None = None,
    *,
    outlier_q: float = 0.95,
) -> dict[str, object]:
    if studies_root is None:
        studies_root = _load_default_studies_root()

    cache_key = (str(studies_root.resolve()), outlier_q)
    cached = _TRAJECTORY_OVERLAY_CACHE.get(cache_key)
    if cached is not None:
        return cached
    cache_path = _trajectory_overlay_cache_path(
        studies_root, outlier_q
    )
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            overlay_data = pickle.load(handle)
        _TRAJECTORY_OVERLAY_CACHE[cache_key] = overlay_data
        return overlay_data

    line_segments: list[np.ndarray] = []
    kept_segment_count = 0

    for study_dir in _study_dirs(studies_root):
        tracks_by_individual: dict[str, list[tuple[float, float, float]]] = defaultdict(
            list
        )
        with (study_dir / "trajectories.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    lat = float(row.get("location-lat", ""))
                    lon = float(row.get("location-long", ""))
                except (TypeError, ValueError):
                    continue

                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue

                timestamp = _parse_timestamp(row.get("timestamp") or "")
                if timestamp is None:
                    continue

                individual_id = (
                    row.get("individual-local-identifier")
                    or row.get("individual_local_identifier")
                    or f"{study_dir.name}__unknown"
                )
                tracks_by_individual[str(individual_id)].append(
                    (_seconds_since_epoch(timestamp), lon, lat)
                )

        for points in tracks_by_individual.values():
            if len(points) < 2:
                continue

            points.sort(key=lambda item: item[0])
            point_array = np.asarray([(point[1], point[2]) for point in points], dtype=float)
            lon_q1, lon_q3 = np.quantile(point_array[:, 0], [0.25, 0.75])
            lat_q1, lat_q3 = np.quantile(point_array[:, 1], [0.25, 0.75])
            lon_iqr = max(float(lon_q3 - lon_q1), 1e-6)
            lat_iqr = max(float(lat_q3 - lat_q1), 1e-6)
            lon_lo = float(lon_q1 - 8.0 * lon_iqr)
            lon_hi = float(lon_q3 + 8.0 * lon_iqr)
            lat_lo = float(lat_q1 - 8.0 * lat_iqr)
            lat_hi = float(lat_q3 + 8.0 * lat_iqr)
            points = [
                point
                for point in points
                if lon_lo <= point[1] <= lon_hi and lat_lo <= point[2] <= lat_hi
            ]
            if len(points) < 2:
                continue

            gaps_s: list[float] = []
            step_lengths_m: list[float] = []
            step_speeds_mps: list[float] = []

            for idx in range(1, len(points)):
                prev_time, prev_lon, prev_lat = points[idx - 1]
                cur_time, cur_lon, cur_lat = points[idx]
                gap_s = cur_time - prev_time
                step_length_m = _haversine_distance_m(
                    prev_lat, prev_lon, cur_lat, cur_lon
                )
                if gap_s <= 0 or not math.isfinite(step_length_m) or step_length_m <= 0:
                    continue
                gaps_s.append(gap_s)
                step_lengths_m.append(step_length_m)
                step_speeds_mps.append(step_length_m / gap_s)

            gap_cutoff_s = _quantile(gaps_s, outlier_q)
            step_cutoff_m = _quantile(step_lengths_m, outlier_q)
            speed_cutoff_mps = _quantile(step_speeds_mps, outlier_q)
            if gap_cutoff_s is None or step_cutoff_m is None or speed_cutoff_mps is None:
                continue

            valid_point_mask = [True] * len(points)
            for idx in range(1, len(points) - 1):
                prev_time, prev_lon, prev_lat = points[idx - 1]
                cur_time, cur_lon, cur_lat = points[idx]
                next_time, next_lon, next_lat = points[idx + 1]

                prev_gap_s = cur_time - prev_time
                next_gap_s = next_time - cur_time
                prev_step_m = _haversine_distance_m(prev_lat, prev_lon, cur_lat, cur_lon)
                next_step_m = _haversine_distance_m(cur_lat, cur_lon, next_lat, next_lon)
                bridge_gap_s = next_time - prev_time
                bridge_step_m = _haversine_distance_m(
                    prev_lat, prev_lon, next_lat, next_lon
                )

                is_isolated_spatial_outlier = (
                    prev_gap_s > 0
                    and next_gap_s > 0
                    and prev_step_m >= step_cutoff_m
                    and next_step_m >= step_cutoff_m
                    and bridge_gap_s > 0
                    and bridge_gap_s <= gap_cutoff_s
                    and bridge_step_m <= step_cutoff_m
                )
                if is_isolated_spatial_outlier:
                    valid_point_mask[idx] = False

            filtered_points = [
                point for point, is_valid in zip(points, valid_point_mask, strict=True) if is_valid
            ]
            if len(filtered_points) < 2:
                continue

            current_segment: list[tuple[float, float]] = [
                (filtered_points[0][1], filtered_points[0][2])
            ]
            for idx in range(1, len(filtered_points)):
                prev_time, prev_lon, prev_lat = filtered_points[idx - 1]
                cur_time, cur_lon, cur_lat = filtered_points[idx]
                gap_s = cur_time - prev_time
                step_length_m = _haversine_distance_m(
                    prev_lat, prev_lon, cur_lat, cur_lon
                )
                keep_edge = (
                    gap_s > 0
                    and math.isfinite(step_length_m)
                    and 0 < step_length_m < step_cutoff_m
                    and gap_s < gap_cutoff_s
                )
                if not keep_edge:
                    bridge_points = _build_smooth_gap_bridge(
                        filtered_points,
                        idx,
                        gap_cutoff_s=gap_cutoff_s,
                        step_cutoff_m=step_cutoff_m,
                        speed_cutoff_mps=speed_cutoff_mps,
                    )
                    if bridge_points is not None:
                        current_segment.extend(
                            [tuple(point) for point in bridge_points]
                        )
                        current_segment.append((cur_lon, cur_lat))
                        continue

                    if len(current_segment) >= 2:
                        line_segments.append(np.asarray(current_segment, dtype=float))
                        kept_segment_count += 1
                    current_segment = [(cur_lon, cur_lat)]
                    continue

                # Avoid artifacts across the antimeridian on a rectangular Plate Carree map.
                if abs(cur_lon - prev_lon) > 180:
                    if len(current_segment) >= 2:
                        line_segments.append(np.asarray(current_segment, dtype=float))
                        kept_segment_count += 1
                    current_segment = [(cur_lon, cur_lat)]
                    continue

                current_segment.append((cur_lon, cur_lat))

            if len(current_segment) >= 2:
                line_segments.append(np.asarray(current_segment, dtype=float))
                kept_segment_count += 1

    overlay_data = {
        "line_segments": line_segments,
        "kept_segment_count": kept_segment_count,
    }
    _TRAJECTORY_OVERLAY_CACHE[cache_key] = overlay_data
    with cache_path.open("wb") as handle:
        pickle.dump(overlay_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return overlay_data


def plot_hero_with_spaghetti(
    country_summary: pd.DataFrame | None = None,
    *,
    overlay_data: dict[str, object] | None = None,
    line_color: str = "#8F2D1F",
    halo_color: str = "#F6D7CC",
    line_alpha: float = 0.62,
    halo_alpha: float = 0.26,
    line_width: float = 1.6,
    halo_width: float = 3.2,
    output_path: Path | None = REPO_ROOT / "pngs" / "hero_atlas_blue_log_spaghetti.png",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plot_hero_map(country_summary=country_summary, output_path=None)
    if overlay_data is None:
        overlay_data = load_trajectory_overlay_data()

    segments = overlay_data["line_segments"]
    halo_collection = LineCollection(
        segments,
        colors=halo_color,
        linewidths=halo_width,
        alpha=halo_alpha,
        zorder=3,
        capstyle="round",
        joinstyle="round",
    )
    halo_collection.set_transform(ccrs.PlateCarree())
    ax.add_collection(halo_collection)

    collection = LineCollection(
        segments,
        colors=line_color,
        linewidths=line_width,
        alpha=line_alpha,
        zorder=4,
        capstyle="round",
        joinstyle="round",
    )
    collection.set_transform(ccrs.PlateCarree())
    ax.add_collection(collection)

    if output_path is not None:
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, ax


# %% [markdown]
# ## Load Cached Country GPS Point Counts

# %%
if _in_ipython_session():
    country_gps_point_counts = load_country_gps_point_counts()
    country_gps_point_counts.head(12)


# %% [markdown]
# ## Atlas Blue Log Hero Map

# %%
if _in_ipython_session():
    fig, ax = plot_hero_map(country_gps_point_counts)
    plt.show()


# %% [markdown]
# ## Load Cached Trajectory Overlays

# %%
if _in_ipython_session():
    trajectory_overlay_data = load_trajectory_overlay_data()
    trajectory_overlay_data["kept_segment_count"]


# %% [markdown]
# ## Cleaned Spaghetti Lines

# %%
if _in_ipython_session():
    fig, ax = plot_hero_with_spaghetti(
        country_summary=country_gps_point_counts,
        overlay_data=trajectory_overlay_data,
    )
    plt.show()


if __name__ == "__main__":
    plot_hero_map()
    plt.show()
