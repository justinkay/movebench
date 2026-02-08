from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from movebench2 import paths
from movebench2.layers.plan import LayerPlan
from movebench2.layers.registry import LayerRegistry, REGISTRY
from movebench2.rasters import Raster, merge_hdf_window, read_geotiff, read_geotiff_window


def _year_from_timestamp(ts: Any) -> int:
    if ts is None:
        raise ValueError("timestamp is required to resolve a year-based local layer")
    if hasattr(ts, "year"):
        return int(ts.year)
    raise TypeError(f"timestamp must have a .year attribute, got {type(ts)!r}")


def _clamp_year(year: int, year_bounds: tuple[int, int] | None) -> int:
    if year_bounds is None:
        return year
    lo, hi = year_bounds
    return max(min(year, hi), lo)


@lru_cache(maxsize=1)
def _modis_grid_rows() -> list[tuple[int, int, float, float, float, float]]:
    """Rows from sn_bound_10deg.txt as (h, v, lon_min, lon_max, lat_min, lat_max)."""

    grid_path = paths.get_path("modis_grid")
    rows: list[tuple[int, int, float, float, float, float]] = []
    with open(grid_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or not s[0].isdigit():
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            v = int(parts[0])
            h = int(parts[1])
            lon_min = float(parts[2])
            lon_max = float(parts[3])
            lat_min = float(parts[4])
            lat_max = float(parts[5])
            rows.append((h, v, lon_min, lon_max, lat_min, lat_max))
    return rows


def _modis_tiles_for_bounds(bounds_latlon: tuple[float, float, float, float]) -> list[tuple[int, int]]:
    lon_min, lat_min, lon_max, lat_max = bounds_latlon
    out: list[tuple[int, int]] = []
    for h, v, lo0, lo1, la0, la1 in _modis_grid_rows():
        if lo0 <= lon_max and lo1 >= lon_min and la0 <= lat_max and la1 >= lat_min:
            out.append((h, v))
    out.sort(key=lambda hv: (hv[1], hv[0]))
    return out


@dataclass
class _ModisFileIndex:
    base_dir: Path
    product_prefix: str

    def __post_init__(self) -> None:
        self._index: dict[tuple[int, int, int], str] = {}
        pattern = str(self.base_dir / f"{self.product_prefix}.A*.h*v*.hdf")
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            try:
                year = int(fname.split(".A")[1][:4])
                h = int(fname.split(".h")[1][:2])
                v = int(fname.split("v")[1][:2])
            except (IndexError, ValueError):
                continue
            self._index[(year, h, v)] = path

    def get_path(self, year: int, h: int, v: int) -> str | None:
        return self._index.get((year, h, v))


class LayerIO:
    """Loads rasters for a study + LayerPlan.

    This is a thin IO façade used by feature extraction and derived computation.
    It does not download anything; it only reads from disk and local datasets.
    """

    def __init__(self, study_dir: Path, plan: LayerPlan, registry: LayerRegistry = REGISTRY):
        self.study_dir = Path(study_dir)
        self.plan = plan
        self.registry = registry

        self._chip_cache: dict[tuple[str, int], Raster] = {}
        self._tile_index_cache: dict[str, dict[str, Any]] = {}
        self._hdf_index_cache: dict[str, _ModisFileIndex] = {}

        self._chip_group_layer_slices = self._build_chip_group_layer_slices()

    def _rasters_root(self) -> Path:
        return self.study_dir / "rasters" / self.plan.config_hash

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

        if spec.source == "gee_chip":
            if point_id is None:
                raise ValueError(f"point_id is required to load gee_chip layer {layer!r}")
            return self._load_gee_chip_layer(layer, point_id)

        if spec.source == "gee_tile":
            if bounds_latlon is None:
                raise ValueError(f"bounds_latlon is required to load gee_tile layer {layer!r}")
            return self._load_gee_tile_layer(layer, bounds_latlon)

        if spec.source == "local_tiff":
            if bounds_latlon is None:
                raise ValueError(f"bounds_latlon is required to load local_tiff layer {layer!r}")
            return self._load_local_tiff_layer(layer, bounds_latlon, timestamp)

        if spec.source == "local_hdf":
            if bounds_latlon is None:
                raise ValueError(f"bounds_latlon is required to load local_hdf layer {layer!r}")
            return self._load_local_hdf_layer(layer, bounds_latlon, timestamp)

        if spec.source == "derived":
            return self._load_derived_layer(layer, point_id=point_id, bounds_latlon=bounds_latlon, timestamp=timestamp)

        raise ValueError(f"Unknown LayerSource for {layer!r}: {spec.source!r}")

    def _chip_group_for_layer(self, layer: str) -> str:
        group = self.plan.chip_group_for_layer(layer)
        if group is None:
            raise ValueError(f"gee_chip layer {layer!r} is not listed in any chip group")
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
        key = (chip_group, int(point_id))
        cached = self._chip_cache.get(key)
        if cached is not None:
            return cached

        chip_path = self._chip_path(chip_group, point_id)
        r = read_geotiff(chip_path)
        r = Raster(
            array=r.array,
            transform=r.transform,
            crs=r.crs,
            nodata=r.nodata,
            bounds=r.bounds,
            band_ids=self._chip_group_band_ids(chip_group),
        )
        self._chip_cache[key] = r
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

    def _tile_index(self, layer: str) -> dict[str, Any]:
        cached = self._tile_index_cache.get(layer)
        if cached is not None:
            return cached
        index_path = self._tile_dir(layer) / "index.json"
        with open(index_path, "r") as f:
            d = json.load(f)
        if not isinstance(d, dict) or "tiles" not in d:
            raise ValueError(f"Invalid tile index.json for layer {layer!r}: {index_path}")
        self._tile_index_cache[layer] = d
        return d

    def _load_gee_tile_layer(self, layer: str, bounds_latlon: tuple[float, float, float, float]) -> Raster:
        idx = self._tile_index(layer)
        tile_dir = self._tile_dir(layer)
        tiles = idx.get("tiles", [])

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
            raise FileNotFoundError(f"No tiles found for {layer!r} covering bounds={bounds_latlon}")

        if len(matches) == 1:
            base = read_geotiff_window(matches[0], bounds_latlon)
        else:
            import rasterio
            from rasterio.merge import merge

            datasets = [rasterio.open(p) for p in matches]
            try:
                mosaic, out_transform = merge(datasets, bounds=bounds_latlon, nodata=datasets[0].nodata)
                base = Raster(
                    array=mosaic,
                    transform=out_transform,
                    crs=datasets[0].crs,
                    nodata=datasets[0].nodata,
                    bounds=bounds_latlon,
                    band_ids=[f"band_{i+1}" for i in range(mosaic.shape[0])],
                )
            finally:
                for ds in datasets:
                    ds.close()

        spec = self.registry.get(layer)
        return Raster(
            array=base.array,
            transform=base.transform,
            crs=base.crs,
            nodata=base.nodata,
            bounds=base.bounds,
            band_ids=[f"{layer}.{b}" for b in spec.bands],
        )

    def _load_local_tiff_layer(self, layer: str, bounds_latlon: tuple[float, float, float, float], timestamp: Any | None) -> Raster:
        spec = self.registry.get(layer)
        if not spec.path_pattern:
            raise ValueError(f"local_tiff layer {layer!r} missing spec.path_pattern")

        cfg = paths.paths_config()

        if "{band_num}" in spec.path_pattern:
            rasters: list[Raster] = []
            for i in range(1, len(spec.bands) + 1):
                p = Path(spec.path_pattern.format(**cfg, band_num=i))
                rasters.append(read_geotiff_window(p, bounds_latlon))

            base = rasters[0]
            for r in rasters[1:]:
                if r.array.shape[1:] != base.array.shape[1:] or r.transform != base.transform or r.crs != base.crs:
                    raise ValueError(f"Bioclim bands did not align for layer {layer!r}")

            import numpy as np

            arr = np.concatenate([r.array for r in rasters], axis=0)
            return Raster(
                array=arr,
                transform=base.transform,
                crs=base.crs,
                nodata=base.nodata,
                bounds=base.bounds,
                band_ids=[f"{layer}.{b}" for b in spec.bands],
            )

        year = _clamp_year(_year_from_timestamp(timestamp) if "{year}" in spec.path_pattern else 0, spec.year_bounds)
        p = Path(spec.path_pattern.format(**cfg, year=year))
        r = read_geotiff_window(p, bounds_latlon)
        return Raster(
            array=r.array,
            transform=r.transform,
            crs=r.crs,
            nodata=r.nodata,
            bounds=r.bounds,
            band_ids=[f"{layer}.{b}" for b in spec.bands],
        )

    def _hdf_index_for_layer(self, layer: str) -> _ModisFileIndex:
        cached = self._hdf_index_cache.get(layer)
        if cached is not None:
            return cached

        spec = self.registry.get(layer)
        if not spec.path_pattern:
            raise ValueError(f"local_hdf layer {layer!r} missing spec.path_pattern")

        # Built-in patterns begin with "{paths_key}/...".
        if spec.path_pattern.startswith("{"):
            end = spec.path_pattern.find("}")
            root_key = spec.path_pattern[1:end]
            base_dir = paths.get_path(root_key)
        else:
            base_dir = Path(os.path.dirname(spec.path_pattern))

        filename_pattern = spec.path_pattern.split("/")[-1]
        product_prefix = filename_pattern.split(".")[0]

        idx = _ModisFileIndex(base_dir=Path(base_dir), product_prefix=product_prefix)
        self._hdf_index_cache[layer] = idx
        return idx

    def _load_local_hdf_layer(self, layer: str, bounds_latlon: tuple[float, float, float, float], timestamp: Any | None) -> Raster:
        spec = self.registry.get(layer)
        if not spec.hdf_path_suffix:
            raise ValueError(f"local_hdf layer {layer!r} missing spec.hdf_path_suffix")

        year = _clamp_year(_year_from_timestamp(timestamp), spec.year_bounds)
        hvs = _modis_tiles_for_bounds(bounds_latlon)

        idx = self._hdf_index_for_layer(layer)

        subdataset_paths: list[str] = []
        for h, v in hvs:
            hdf_path = idx.get_path(year, int(h), int(v))
            if hdf_path is None:
                raise FileNotFoundError(f"No MODIS file for layer={layer!r} year={year} h={h} v={v}")
            subdataset_paths.append(f"HDF4_EOS:EOS_GRID:{hdf_path}{spec.hdf_path_suffix}")

        r = merge_hdf_window(tuple(subdataset_paths), bounds_latlon)
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

        import movebench2.layers.derived as derived

        fn_name = f"compute_{layer}"
        fn = getattr(derived, fn_name, None)
        if fn is None:
            raise ValueError(f"Derived layer {layer!r} missing compute function {fn_name} in movebench2.layers.derived")

        arr = fn(**parents)
        if arr.ndim != 3:
            raise ValueError(f"{fn_name} must return an array shaped (C,H,W); got shape={getattr(arr, 'shape', None)}")

        base = parents[spec.derived_from[0]]
        return Raster(
            array=arr,
            transform=base.transform,
            crs=base.crs,
            nodata=float("nan"),
            bounds=base.bounds,
            band_ids=[f"{layer}.{b}" for b in spec.bands],
        )

