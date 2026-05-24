from __future__ import annotations

from dataclasses import dataclass

DEFAULT_THUMBNAIL_BASEMAP = "positron"
THUMBNAIL_BASEMAP_NONE = "none"


@dataclass(frozen=True)
class ThumbnailBasemapSpec:
    key: str
    label: str
    tile_url: str | None
    attribution: str | None
    tile_size: int = 256


_THUMBNAIL_BASEMAP_SPECS: dict[str, ThumbnailBasemapSpec] = {
    "positron": ThumbnailBasemapSpec(
        key="positron",
        label="Positron",
        tile_url="https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        attribution="&copy; OpenStreetMap contributors &copy; CARTO",
    ),
    "topographic": ThumbnailBasemapSpec(
        key="topographic",
        label="Topographic",
        tile_url="https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}.png",
        attribution=(
            '&copy; <a href="https://stadiamaps.com/" target="_blank">Stadia Maps</a> '
            '&copy; <a href="https://stamen.com/" target="_blank">Stamen Design</a> '
            '&copy; <a href="https://openmaptiles.org/" target="_blank">OpenMapTiles</a> '
            '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
        ),
    ),
    "osm": ThumbnailBasemapSpec(
        key="osm",
        label="OpenStreetMap",
        tile_url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        attribution="&copy; OpenStreetMap contributors",
    ),
    "satellite": ThumbnailBasemapSpec(
        key="satellite",
        label="Satellite",
        tile_url=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attribution="Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
    ),
    "none": ThumbnailBasemapSpec(
        key="none",
        label="None",
        tile_url=None,
        attribution=None,
    ),
}


def normalize_thumbnail_basemap(basemap: str | None) -> str:
    value = DEFAULT_THUMBNAIL_BASEMAP if basemap in (None, "") else str(basemap).strip().lower()
    if value not in _THUMBNAIL_BASEMAP_SPECS:
        raise ValueError(
            "thumbnail basemap must be one of "
            f"{sorted(_THUMBNAIL_BASEMAP_SPECS)}, got {basemap!r}"
        )
    return value


def thumbnail_basemap_spec(basemap: str | None) -> ThumbnailBasemapSpec:
    return _THUMBNAIL_BASEMAP_SPECS[normalize_thumbnail_basemap(basemap)]


def thumbnail_basemap_ids(*, include_none: bool = True) -> list[str]:
    ids = ["positron", "topographic", "osm", "satellite"]
    if include_none:
        ids.append("none")
    return ids


def thumbnail_basemap_label(basemap: str | None) -> str:
    return thumbnail_basemap_spec(basemap).label
