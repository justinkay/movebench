# %% [markdown]
# # Hero Map
#
# Run each `# %%` block with `Shift+Enter` in VS Code.

# %%
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cartopy.io import shapereader
from shapely.geometry import Point
from shapely.prepared import prep


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "pngs" / "hero_atlas_blue_log.png"
OUTPUT_PATH.parent.mkdir(exist_ok=True)

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


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value.strip()
    return value.split("#", 1)[0].strip()


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


# %% [markdown]
# ## Load Cached Country GPS Point Counts

# %%
country_gps_point_counts = load_country_gps_point_counts()
country_gps_point_counts.head(12)


# %% [markdown]
# ## Atlas Blue Log Hero Map

# %%
fig, ax = plot_hero_map(country_gps_point_counts)
plt.show()


if __name__ == "__main__":
    plot_hero_map()
    plt.show()
