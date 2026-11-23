from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.transform import from_bounds

from movebench import paths
import movebench.layers.derived as derived
from movebench.layers.constants import GEE_NODATA_SENTINEL, WGS84_CRS
from movebench.layers.plan import LayerPlan
from movebench.layers.registry import REGISTRY, LayerRegistry
from movebench.rasters import (
    Raster,
    read_geotiff,
    read_geotiff_window,
)


def _copy_first_valid_gee_tile(
    merged_data, new_data, merged_mask, new_mask, **_kwargs
):
    data = np.asarray(new_data)
    np.copyto(
        merged_data,
        data,
        where=merged_mask
        & ~new_mask
        & (data != GEE_NODATA_SENTINEL),
    )


def _year_from_timestamp(ts: Any) -> int:
    if ts is None:
        raise ValueError("timestamp is required to resolve a year-based layer")
    if hasattr(ts, "year"):
        return int(ts.year)
    raise TypeError(f"timestamp must have a .year attribute, got {type(ts)!r}")


def _clamp_year(year: int, year_bounds: tuple[int, int] | None) -> int:
    if year_bounds is None:
        return year
    lo, hi = year_bounds
    return max(min(year, hi), lo)


class LayerIO:
    """Loads rasters for a study + LayerPlan.

    This is a thin IO façade used by feature extraction and derived computation.
    It does not download anything; it only reads from disk and local datasets.
    """

    def __init__(
        self, study_dir: Path, plan: LayerPlan, registry: LayerRegistry = REGISTRY
    ):
        self.study_dir = Path(study_dir)
        self.plan = plan
        self.registry = registry

        self._tile_index_cache: dict[tuple[str, int | None], dict[str, Any]] = {}
        self._tile_index_years_cache: dict[str, list[int]] = {}

        self._chip_group_layer_slices = self._build_chip_group_layer_slices()

    def _rasters_root(self) -> Path:
        return self.study_dir / "rasters" / self.plan.rasters_hash

    def _chip_path(self, chip_group: str, point_id: int) -> Path:
        return self._rasters_root() / "chips" / chip_group / f"{int(point_id):05d}.tif"

    def _tile_dir(self, layer: str) -> Path:
        return self._rasters_root() / "tiles" / layer

    def _build_chip_group_layer_slices(self) -> dict[str, dict[str, tuple[int, int]]]:
        out: dict[str, dict[str, tuple[int, int]]] = {}
        for group_name, g in self.plan.chip_groups.items():
            offset = 0
            layer_slices: dict[str, tuple[int, int]] = {}
            for layer in g["layers"]:
                spec = self.registry.get(layer)
                n = len(spec.bands)
                layer_slices[layer] = (offset, offset + n)
                offset += n
            out[group_name] = layer_slices
        return out

    def load_layer(
        self,
        layer: str,
        *,
        point_id: int | None = None,
        bounds_latlon: tuple[float, float, float, float] | None = None,
        timestamp: Any | None = None,
    ) -> Raster:
        spec = self.registry.get(layer)
        if spec.metric_grid_derived:
            raise ValueError(
                f"{layer} must be derived after reprojection to a local metric grid"
            )

        if spec.source == "gee_chip":
            if point_id is None:
                raise ValueError(
                    f"point_id is required to load gee_chip layer {layer!r}"
                )
            return self._load_gee_chip_layer(layer, point_id)

        if spec.source == "gee_tile":
            if bounds_latlon is None:
                raise ValueError(
                    f"bounds_latlon is required to load gee_tile layer {layer!r}"
                )
            return self._load_gee_tile_layer(layer, bounds_latlon, timestamp=timestamp)

        if spec.source == "local_tiff":
            if bounds_latlon is None:
                raise ValueError(
                    f"bounds_latlon is required to load local_tiff layer {layer!r}"
                )
            return self._load_local_tiff_layer(layer, bounds_latlon, timestamp)

        if spec.source == "derived":
            return self._load_derived_layer(
                layer,
                point_id=point_id,
                bounds_latlon=bounds_latlon,
                timestamp=timestamp,
            )

        raise ValueError(f"Unknown LayerSource for {layer!r}: {spec.source!r}")

    def iter_layers(
        self,
        *,
        point_id: int,
        timestamp: Any,
        bounds_parts: tuple[tuple[float, float, float, float], ...],
        dst_crs=None,
        support_bounds_m: tuple[float, float, float, float] | None = None,
        source_cache: dict[str, list[Raster]] | None = None,
        chip_groups: dict[str, Raster] | None = None,
    ):
        """Load the plan in order, sharing source reads and terrain work per request."""
        source_cache = {} if source_cache is None else source_cache
        chip_groups = {} if chip_groups is None else chip_groups
        metric_cache: dict[str, Raster] = {}

        def load(layer: str) -> list[Raster]:
            if layer in source_cache:
                return source_cache[layer]
            spec = self.registry.get(layer)
            if spec.metric_grid_derived:
                if layer not in metric_cache:
                    requested = {
                        name
                        for name in self.plan.layers_in_order
                        if self.registry.get(name).metric_grid_derived
                    }
                    wind = (
                        load("era5_land_hourly")
                        if "orographic_uplift_hourly" in requested
                        else None
                    )
                    metric_cache.update(
                        derived.derive_metric_terrain_layers(
                            dem_parts=load("dem"),
                            dst_crs=dst_crs,
                            support_bounds_m=support_bounds_m,
                            requested_layers=requested,
                            era5_land_hourly=wind[0] if wind else None,
                        )
                    )
                return [metric_cache[layer]]

            if spec.source == "gee_chip":
                group = self._chip_group_for_layer(layer)
                if group not in chip_groups:
                    chip_groups[group] = self.load_chip_group(group, point_id=point_id)
                raster = chip_groups[group]
                start, end = self._chip_group_layer_slices[group][layer]
                parts = [
                    Raster(
                        array=raster.array[start:end],
                        transform=raster.transform,
                        crs=raster.crs,
                        nodata=raster.nodata,
                        bounds=raster.bounds,
                        band_ids=[f"{layer}.{band}" for band in spec.bands],
                    )
                ]
            elif spec.source == "derived":
                if not spec.derived_from:
                    raise ValueError(f"Derived layer {layer!r} must define derived_from")
                parent_parts = {parent: load(parent) for parent in spec.derived_from}
                compute = getattr(derived, f"compute_{layer}", None)
                if compute is None:
                    raise ValueError(f"Missing derived compute function compute_{layer}")
                parts = []
                for index in range(max(len(rasters) for rasters in parent_parts.values())):
                    parents = {
                        parent: rasters[index] if len(rasters) > 1 else rasters[0]
                        for parent, rasters in parent_parts.items()
                    }
                    base = parents[spec.derived_from[0]]
                    parts.append(
                        Raster(
                            array=compute(**parents),
                            transform=base.transform,
                            crs=base.crs,
                            nodata=np.nan,
                            bounds=base.bounds,
                            band_ids=[f"{layer}.{band}" for band in spec.bands],
                        )
                    )
            else:
                parts = [
                    self.load_layer(layer, bounds_latlon=bounds, timestamp=timestamp)
                    for bounds in bounds_parts
                ]
            source_cache[layer] = parts
            return parts

        for layer in self.plan.layers_in_order:
            yield load(layer)

    def load_chip_group(self, chip_group: str, *, point_id: int) -> Raster:
        """Load the full stacked GeoTIFF for a chip group (all layers/bands in that group)."""
        return self._load_gee_chip_group(chip_group, int(point_id))

    def _chip_group_for_layer(self, layer: str) -> str:
        group = self.plan.chip_group_for_layer(layer)
        if group is None:
            raise ValueError(
                f"gee_chip layer {layer!r} is not listed in any chip group"
            )
        return group

    def _chip_group_band_ids(self, chip_group: str) -> list[str]:
        band_ids: list[str] = []
        group = self.plan.chip_groups.get(chip_group)
        if group is None:
            raise KeyError(f"Unknown chip group: {chip_group!r}")
        for layer in group["layers"]:
            spec = self.registry.get(layer)
            band_ids.extend([f"{layer}.{b}" for b in spec.bands])
        return band_ids

    def _load_gee_chip_group(self, chip_group: str, point_id: int) -> Raster:
        chip_path = self._chip_path(chip_group, point_id)
        r = read_geotiff(chip_path)

        # GEE downloads use a numeric sentinel for masked pixels. Convert it to NaN
        # immediately so downstream valid masking and derived math are correct.
        arr = r.array.astype(np.float32, copy=False)
        arr[arr == GEE_NODATA_SENTINEL] = np.nan
        r = Raster(
            array=arr,
            transform=r.transform,
            crs=r.crs,
            nodata=float("nan"),
            bounds=r.bounds,
            band_ids=self._chip_group_band_ids(chip_group),
        )
        return r

    def _load_gee_chip_layer(self, layer: str, point_id: int) -> Raster:
        chip_group = self._chip_group_for_layer(layer)
        group_r = self._load_gee_chip_group(chip_group, point_id)
        start, end = self._chip_group_layer_slices[chip_group][layer]
        return Raster(
            array=group_r.array[start:end],
            transform=group_r.transform,
            crs=group_r.crs,
            nodata=group_r.nodata,
            bounds=group_r.bounds,
            band_ids=group_r.band_ids[start:end],
        )

    def _tile_index_years_on_disk(self, layer: str) -> list[int]:
        cached = self._tile_index_years_cache.get(layer)
        if cached is not None:
            return cached

        tile_dir = self._tile_dir(layer)
        years: list[int] = []
        for p in tile_dir.glob("index_*.json"):
            stem = p.stem  # index_<year>
            parts = stem.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                years.append(int(parts[1]))
            except ValueError:
                continue
        years = sorted(set(years))
        # Avoid caching an empty result so newly-downloaded years can be seen
        # without rebuilding the LayerIO instance.
        if years:
            self._tile_index_years_cache[layer] = years
        return years

    def _tile_index(
        self, layer: str, *, timestamp: Any | None = None
    ) -> dict[str, Any]:
        spec = self.registry.get(layer)
        resolved_year: int | None = None
        index_path = self._tile_dir(layer) / "index.json"

        if spec.temporal == "annual":
            requested_year = _year_from_timestamp(timestamp)
            available_years = self._tile_index_years_on_disk(layer)
            if available_years:
                resolved_year = min(
                    available_years,
                    key=lambda a: (abs(int(a) - int(requested_year)), int(a)),
                )
                index_path = self._tile_dir(layer) / f"index_{int(resolved_year)}.json"

        cache_key = (layer, resolved_year)
        cached = self._tile_index_cache.get(cache_key)
        if cached is not None:
            return cached

        if not index_path.exists():
            tile_dir = self._tile_dir(layer)
            if spec.temporal == "annual":
                raise FileNotFoundError(
                    f"Tile index not found for annual gee_tile layer={layer!r}: {index_path}. "
                    f"Expected {tile_dir}/index_<year>.json; run downloads for this study."
                )
            raise FileNotFoundError(
                f"Tile index.json not found for layer={layer!r}: {index_path}"
            )

        with open(index_path, "r") as f:
            d = json.load(f)
        if not isinstance(d, dict) or "tiles" not in d:
            raise ValueError(
                f"Invalid tile index.json for layer {layer!r}: {index_path}"
            )
        n_tiles_reported = d.get("n_tiles_reported")
        n_tiles_effective = d.get("n_tiles_effective", len(d.get("tiles", [])))
        if n_tiles_reported is not None and int(n_tiles_effective) != int(
            n_tiles_reported
        ):
            warnings.warn(
                (
                    f"Incomplete GEE tile index for layer={layer!r}: "
                    f"reported={int(n_tiles_reported)} effective={int(n_tiles_effective)} "
                    f"at {index_path}. Rerun downloads for this study."
                ),
                RuntimeWarning,
            )
        self._tile_index_cache[cache_key] = d
        return d

    def _load_gee_tile_layer(
        self,
        layer: str,
        bounds_latlon: tuple[float, float, float, float],
        *,
        timestamp: Any | None,
    ) -> Raster:
        idx = self._tile_index(layer, timestamp=timestamp)
        tile_dir = self._tile_dir(layer)
        tiles = idx.get("tiles", [])

        spec = self.registry.get(layer)
        band_ids = [f"{layer}.{b}" for b in spec.bands]

        def _warn_missing(reason: str) -> None:
            warnings.warn(
                f"Missing GEE tile data for layer={layer!r} ({reason}); returning all-NaN raster (samples become invalid).",
                RuntimeWarning,
            )

        def _missing_raster() -> Raster:
            min_lon, min_lat, max_lon, max_lat = bounds_latlon
            # Some derived layers (e.g. slope/aspect) require >=2 pixels to run
            # numerical gradients without crashing.
            transform = from_bounds(min_lon, min_lat, max_lon, max_lat, 2, 2)
            arr = np.full((len(band_ids), 2, 2), np.nan, dtype=np.float32)
            return Raster(
                array=arr,
                transform=transform,
                crs=WGS84_CRS,
                nodata=float("nan"),
                bounds=bounds_latlon,
                band_ids=band_ids,
            )

        min_lon, min_lat, max_lon, max_lat = bounds_latlon
        matches: list[Path] = []
        for t in tiles:
            b = t.get("bounds", {})
            if (
                float(b["min_lon"]) <= max_lon
                and float(b["max_lon"]) >= min_lon
                and float(b["min_lat"]) <= max_lat
                and float(b["max_lat"]) >= min_lat
            ):
                matches.append(tile_dir / t["file"])

        if not matches:
            _warn_missing("no_coverage")
            return _missing_raster()

        try:
            if len(matches) == 1:
                base = read_geotiff_window(matches[0], bounds_latlon)
            else:
                datasets = [rasterio.open(p) for p in matches]
                try:
                    mosaic, out_transform = merge(
                        datasets,
                        bounds=bounds_latlon,
                        nodata=GEE_NODATA_SENTINEL,
                        method=_copy_first_valid_gee_tile,
                    )
                    mosaic_bounds = bounds_latlon
                    if 0 in mosaic.shape[-2:]:
                        min_lon, min_lat, max_lon, max_lat = bounds_latlon
                        center_lon = (min_lon + max_lon) / 2.0
                        center_lat = (min_lat + max_lat) / 2.0
                        width = max(
                            max_lon - min_lon, 2.0 * abs(datasets[0].res[0])
                        )
                        height = max(
                            max_lat - min_lat, 2.0 * abs(datasets[0].res[1])
                        )
                        mosaic_bounds = (
                            center_lon - width / 2.0,
                            center_lat - height / 2.0,
                            center_lon + width / 2.0,
                            center_lat + height / 2.0,
                        )
                        mosaic, out_transform = merge(
                            datasets,
                            bounds=mosaic_bounds,
                            nodata=GEE_NODATA_SENTINEL,
                            method=_copy_first_valid_gee_tile,
                        )
                    base = Raster(
                        array=mosaic,
                        transform=out_transform,
                        crs=datasets[0].crs,
                        nodata=GEE_NODATA_SENTINEL,
                        bounds=mosaic_bounds,
                        band_ids=[f"band_{i + 1}" for i in range(mosaic.shape[0])],
                    )
                finally:
                    for ds in datasets:
                        ds.close()
        except Exception as exc:
            if isinstance(exc, FileNotFoundError) or (isinstance(exc, RasterioIOError)):
                _warn_missing("io_error")
                return _missing_raster()
            raise

        arr = base.array.astype(np.float32, copy=False)
        arr[arr == GEE_NODATA_SENTINEL] = np.nan

        if spec.clip_max is not None:
            arr = np.clip(arr, 0.0, float(spec.clip_max))
        return Raster(
            array=arr,
            transform=base.transform,
            crs=base.crs,
            nodata=float("nan"),
            bounds=base.bounds,
            band_ids=band_ids,
        )

    def _load_local_tiff_layer(
        self,
        layer: str,
        bounds_latlon: tuple[float, float, float, float],
        timestamp: Any | None,
    ) -> Raster:
        spec = self.registry.get(layer)
        if not spec.path_pattern:
            raise ValueError(f"local_tiff layer {layer!r} missing spec.path_pattern")

        cfg = paths.paths_config()
        max_window_hw = (250, 250) # save us from OOM for very large study extents

        if "{band_num}" in spec.path_pattern:
            rasters: list[Raster] = []
            for i in range(1, len(spec.bands) + 1):
                p = Path(spec.path_pattern.format(**cfg, band_num=i))
                rasters.append(
                    read_geotiff_window(
                        p,
                        bounds_latlon,
                        max_window_hw=max_window_hw,
                        resampling=Resampling.bilinear,
                    )
                )

            base = rasters[0]
            for r in rasters[1:]:
                if (
                    r.array.shape[1:] != base.array.shape[1:]
                    or r.transform != base.transform
                    or r.crs != base.crs
                ):
                    raise ValueError(f"Bioclim bands did not align for layer {layer!r}")

            arr = np.concatenate([r.array for r in rasters], axis=0)
            return Raster(
                array=arr,
                transform=base.transform,
                crs=base.crs,
                nodata=base.nodata,
                bounds=base.bounds,
                band_ids=[f"{layer}.{b}" for b in spec.bands],
            )

        year = _clamp_year(
            _year_from_timestamp(timestamp) if "{year}" in spec.path_pattern else 0,
            spec.year_bounds,
        )
        p = Path(spec.path_pattern.format(**cfg, year=year))
        r = read_geotiff_window(
            p,
            bounds_latlon,
            max_window_hw=max_window_hw,
            resampling=Resampling.bilinear,
        )
        return Raster(
            array=r.array,
            transform=r.transform,
            crs=r.crs,
            nodata=r.nodata,
            bounds=r.bounds,
            band_ids=[f"{layer}.{b}" for b in spec.bands],
        )

    def _load_derived_layer(
        self,
        layer: str,
        *,
        point_id: int | None,
        bounds_latlon: tuple[float, float, float, float] | None,
        timestamp: Any | None,
    ) -> Raster:
        spec = self.registry.get(layer)
        if not spec.derived_from:
            raise ValueError(f"Derived layer {layer!r} must define derived_from")

        parents: dict[str, Raster] = {}
        for parent in spec.derived_from:
            parents[parent] = self.load_layer(
                parent,
                point_id=point_id,
                bounds_latlon=bounds_latlon,
                timestamp=timestamp,
            )

        fn_name = f"compute_{layer}"
        fn = getattr(derived, fn_name, None)
        if fn is None:
            raise ValueError(
                f"Derived layer {layer!r} missing compute function {fn_name} in movebench.layers.derived"
            )

        arr = fn(**parents)
        if arr.ndim != 3:
            raise ValueError(
                f"{fn_name} must return an array shaped (C,H,W); got shape={getattr(arr, 'shape', None)}"
            )

        base = parents[spec.derived_from[0]]
        return Raster(
            array=arr,
            transform=base.transform,
            crs=base.crs,
            nodata=float("nan"),
            bounds=base.bounds,
            band_ids=[f"{layer}.{b}" for b in spec.bands],
        )
