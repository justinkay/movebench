from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from movebench.datasets.deepssf import (
    _bearing_movebench_to_deepssf,
    _time_features_notebook_order,
)
from movebench.geometry import local_xy_m_from_lonlat, lonlat_from_local_xy_m
from movebench.layers.constants import GEE_NODATA_SENTINEL
from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    SET_COL,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)

ALPHAEARTH_LAYER = "alphaearth_embeddings"
NUM_BANDS = 64


def alphaearth_query_geometry(
    centers_deg: np.ndarray,
    pixel_size_m: float,
    output_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(centers_deg, dtype=np.float64)
    height, width = (int(value) for value in output_hw)
    rows = (height - 1) / 2.0 - np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    edge = np.linspace(-1.0, 1.0, 257)
    longitude = np.empty((len(centers), height, width), dtype=np.float64)
    latitude = np.empty_like(longitude)
    bounds = np.empty((len(centers), 4), dtype=np.float64)

    for start in range(0, len(centers), 32):
        stop = start + 32
        chunk_centers = centers[start:stop]
        chunk_size = len(chunk_centers)
        chunk_longitude, chunk_latitude = lonlat_from_local_xy_m(
            chunk_centers[:, 0, None, None],
            chunk_centers[:, 1, None, None],
            float(pixel_size_m) * cols[None, None, :],
            float(pixel_size_m) * rows[None, :, None],
        )

        edge_x = width * float(pixel_size_m) / 2.0
        edge_y = height * float(pixel_size_m) / 2.0
        ones = np.ones((chunk_size, len(edge)))
        boundary_x = np.concatenate(
            (
                edge_x * np.broadcast_to(edge, (chunk_size, len(edge))),
                edge_x * np.broadcast_to(edge, (chunk_size, len(edge))),
                -edge_x * ones,
                edge_x * ones,
            ),
            axis=1,
        )
        boundary_y = np.concatenate(
            (
                edge_y * ones,
                -edge_y * ones,
                edge_y * np.broadcast_to(edge, (chunk_size, len(edge))),
                edge_y * np.broadcast_to(edge, (chunk_size, len(edge))),
            ),
            axis=1,
        )
        bound_longitude, bound_latitude = lonlat_from_local_xy_m(
            chunk_centers[:, 0, None],
            chunk_centers[:, 1, None],
            boundary_x,
            boundary_y,
        )
        center_lon = chunk_centers[:, 0]
        chunk_longitude = center_lon[:, None, None] + (
            chunk_longitude - center_lon[:, None, None] + 180.0
        ) % 360.0 - 180.0
        bound_longitude = center_lon[:, None] + (
            bound_longitude - center_lon[:, None] + 180.0
        ) % 360.0 - 180.0
        longitude[start:stop] = chunk_longitude
        latitude[start:stop] = chunk_latitude
        epsilon = 1e-10
        bounds[start:stop] = np.column_stack(
            (
                bound_longitude.min(axis=1) - epsilon,
                np.maximum(-90.0, bound_latitude.min(axis=1) - epsilon),
                bound_longitude.max(axis=1) + epsilon,
                np.minimum(90.0, bound_latitude.max(axis=1) + epsilon),
            )
        )
    return longitude, latitude, bounds


@dataclass(frozen=True)
class AlphaEarthTile:
    path: str
    bounds: tuple[float, float, float, float]
    order: int


@dataclass(frozen=True)
class AlphaEarthGroup:
    group_id: int
    study_id: str
    year: int | None
    tiles: tuple[AlphaEarthTile, ...]


@dataclass(frozen=True)
class AlphaEarthCatalog:
    groups: tuple[AlphaEarthGroup, ...]

    @classmethod
    def from_tile_directories(
        cls,
        tile_directories: Iterable[tuple[str, str | Path]],
    ) -> AlphaEarthCatalog:
        groups: list[AlphaEarthGroup] = []
        for study_id, raw_tile_dir in tile_directories:
            tile_dir = Path(raw_tile_dir)
            annual: list[tuple[int, Path]] = []
            for path in tile_dir.glob("index_*.json"):
                match = re.fullmatch(r"index_(\d+)\.json", path.name)
                if match:
                    annual.append((int(match.group(1)), path))
            indexes: list[tuple[int | None, Path]] = sorted(annual)
            if not indexes and (tile_dir / "index.json").exists():
                indexes = [(None, tile_dir / "index.json")]
            if not indexes:
                raise FileNotFoundError(
                    f"No AlphaEarth tile indexes found under {tile_dir}"
                )

            for year, index_path in indexes:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                tiles: list[AlphaEarthTile] = []
                for order, tile in enumerate(index["tiles"]):
                    path = (tile_dir / str(tile["file"])).resolve()
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"AlphaEarth tile listed by {index_path} is missing: {path}"
                        )
                    raw_bounds = tile["bounds"]
                    tiles.append(
                        AlphaEarthTile(
                            path=str(path),
                            bounds=(
                                float(raw_bounds["min_lon"]),
                                float(raw_bounds["min_lat"]),
                                float(raw_bounds["max_lon"]),
                                float(raw_bounds["max_lat"]),
                            ),
                            order=order,
                        )
                    )
                if not tiles:
                    raise ValueError(f"AlphaEarth tile index is empty: {index_path}")
                groups.append(
                    AlphaEarthGroup(
                        group_id=len(groups),
                        study_id=str(study_id),
                        year=year,
                        tiles=tuple(tiles),
                    )
                )
        return cls(tuple(groups))

    def resolve_group_id(self, study_id: str, year: int) -> int:
        groups = [group for group in self.groups if group.study_id == str(study_id)]
        if not groups:
            raise KeyError(f"No AlphaEarth tiles found for study {study_id!r}")
        if groups[0].year is None:
            return groups[0].group_id
        return min(
            groups,
            key=lambda group: (abs(int(group.year) - int(year)), int(group.year)),
        ).group_id

    @property
    def tile_count(self) -> int:
        return sum(len(group.tiles) for group in self.groups)


@dataclass(frozen=True)
class _TileRuntime:
    group_id: int
    tile: AlphaEarthTile
    value_offset: int
    mask_offset: int
    height: int
    width: int
    left: float
    bottom: float
    right: float
    top: float
    xres: float
    yres: float
    nodata: float | None

    @property
    def pixels(self) -> int:
        return self.height * self.width


def _read_header(entry: tuple[int, AlphaEarthTile]):
    group_id, tile = entry
    with rasterio.open(tile.path) as source:
        if source.crs is None or source.crs.to_epsg() != 4326:
            raise ValueError(f"AlphaEarth tile must use EPSG:4326: {tile.path}")
        if (
            float(source.transform.a) <= 0.0
            or float(source.transform.e) >= 0.0
            or float(source.transform.b) != 0.0
            or float(source.transform.d) != 0.0
        ):
            raise ValueError(f"AlphaEarth tile must be north-up: {tile.path}")
        return (
            group_id,
            tile,
            source.count,
            source.height,
            source.width,
            float(source.bounds.left),
            float(source.bounds.bottom),
            float(source.bounds.right),
            float(source.bounds.top),
            float(source.transform.a),
            float(-source.transform.e),
            source.nodata,
        )


def _decode_tile(runtime: _TileRuntime) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(runtime.tile.path) as source:
        values = source.read()
    valid = np.isfinite(values) & (values != GEE_NODATA_SENTINEL)
    if runtime.nodata is not None and math.isfinite(runtime.nodata):
        valid &= values != runtime.nodata
    if not np.array_equal(valid, np.broadcast_to(valid[:1], valid.shape)):
        raise ValueError(
            f"AlphaEarth bands have different validity: {runtime.tile.path}"
        )
    validity = valid[0]
    values[:, ~validity] = 0.0
    return values.astype(np.float16), validity


@torch.compile(dynamic=True)
def _route_tile(
    source_validity,
    source_x,
    source_y,
    xscale,
    yscale,
    merged,
    crop,
    mosaic_shape,
    source_window,
    region,
    coverage,
    nearest_coverage,
    row_offsets,
    col_offsets,
):
    base_x = torch.floor(source_x - 0.5).long()
    base_y = torch.floor(source_y - 0.5).long()
    delta_x = source_x - 0.5 - base_x
    delta_y = source_y - 0.5 - base_y

    def map_indices(col, row):
        merged_col = torch.floor(
            source_window[None, :, 0, None, None]
            + (col - region[None, :, 0, None, None] + 0.5)
            * source_window[None, :, 2, None, None]
        ).long()
        merged_row = torch.floor(
            source_window[None, :, 1, None, None]
            + (row - region[None, :, 1, None, None] + 0.5)
            * source_window[None, :, 3, None, None]
        ).long()
        tile_col = torch.where(merged[None, :, None, None], merged_col, col)
        tile_row = torch.where(merged[None, :, None, None], merged_row, row)
        direct_inside = (
            (tile_col >= crop[None, :, 0, None, None])
            & (tile_col < crop[None, :, 2, None, None])
            & (tile_row >= crop[None, :, 1, None, None])
            & (tile_row < crop[None, :, 3, None, None])
        )
        merged_inside = (
            (col >= region[None, :, 0, None, None])
            & (
                col
                < region[None, :, 0, None, None]
                + region[None, :, 2, None, None]
            )
            & (row >= region[None, :, 1, None, None])
            & (
                row
                < region[None, :, 1, None, None]
                + region[None, :, 3, None, None]
            )
            & (col >= 0)
            & (col < mosaic_shape[None, :, 0, None, None])
            & (row >= 0)
            & (row < mosaic_shape[None, :, 1, None, None])
        )
        inside = torch.where(
            merged[None, :, None, None], merged_inside, direct_inside
        )
        inside &= (
            (tile_col >= 0)
            & (tile_col < source_validity.shape[1])
            & (tile_row >= 0)
            & (tile_row < source_validity.shape[0])
        )
        return tile_col, tile_row, inside

    source_col = base_x[None] + col_offsets[:, None, None, None]
    source_row = base_y[None] + row_offsets[:, None, None, None]
    col, row, inside = map_indices(source_col, source_row)
    col.clamp_(0, source_validity.shape[1] - 1)
    row.clamp_(0, source_validity.shape[0] - 1)
    owned = inside & source_validity[row, col]
    available = owned & ~coverage
    weight = (
        (
            1.0
            - (
                (col_offsets[:, None, None, None] - delta_x[None])
                * xscale[None, :, None, None]
            ).abs()
        ).clamp_min(0.0)
        * (
            1.0
            - (
                (row_offsets[:, None, None, None] - delta_y[None])
                * yscale[None, :, None, None]
            ).abs()
        ).clamp_min(0.0)
        * available
    ).float()
    local_pixel = row * source_validity.shape[1] + col

    nearest_col, nearest_row, nearest_inside = map_indices(
        torch.floor(source_x).long()[None],
        torch.floor(source_y).long()[None],
    )
    nearest_col = nearest_col.squeeze(0).clamp_(0, source_validity.shape[1] - 1)
    nearest_row = nearest_row.squeeze(0).clamp_(0, source_validity.shape[0] - 1)
    nearest_inside = nearest_inside.squeeze(0)
    nearest = nearest_inside & source_validity[nearest_row, nearest_col]
    return local_pixel, weight, coverage | owned, nearest_coverage | nearest


@torch.compile(dynamic=True)
def _gather_samples(values, pixels, weights):
    numerator = (
        values.view(-1, NUM_BANDS)[pixels].float() * weights[..., None]
    ).sum(dim=0).permute(0, 3, 1, 2)
    return numerator, weights.sum(dim=0)[:, None]


class AlphaEarthGPUCache:
    def __init__(
        self,
        catalog: AlphaEarthCatalog,
        device: torch.device | str,
        load_workers: int = 16,
    ) -> None:
        self.catalog = catalog
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("AlphaEarthGPUCache requires CUDA")

        entries = [
            (group.group_id, tile)
            for group in catalog.groups
            for tile in group.tiles
        ]
        with ThreadPoolExecutor(max_workers=int(load_workers)) as executor:
            raw_headers = list(executor.map(_read_header, entries))
            runtimes: list[_TileRuntime] = []
            value_offset = 0
            mask_offset = 0
            for (
                group_id,
                tile,
                count,
                height,
                width,
                left,
                bottom,
                right,
                top,
                xres,
                yres,
                nodata,
            ) in raw_headers:
                if count != NUM_BANDS:
                    raise ValueError(
                        f"Expected {NUM_BANDS} AlphaEarth bands: {tile.path}"
                    )
                runtime = _TileRuntime(
                    group_id=group_id,
                    tile=tile,
                    value_offset=value_offset,
                    mask_offset=mask_offset,
                    height=height,
                    width=width,
                    left=left,
                    bottom=bottom,
                    right=right,
                    top=top,
                    xres=xres,
                    yres=yres,
                    nodata=nodata,
                )
                runtimes.append(runtime)
                value_offset += NUM_BANDS * runtime.pixels
                mask_offset += runtime.pixels

            self.values = torch.empty(
                value_offset, dtype=torch.float16, device=self.device
            )
            self.validity = torch.empty(
                mask_offset, dtype=torch.bool, device=self.device
            )
            loaded = 0
            try:
                for start in range(0, len(runtimes), 32):
                    chunk = runtimes[start : start + 32]
                    for runtime, (values, validity) in zip(
                        chunk, executor.map(_decode_tile, chunk)
                    ):
                        self._value_view(runtime).copy_(
                            torch.from_numpy(values).permute(1, 2, 0).to(self.device)
                        )
                        self._mask_view(runtime).copy_(
                            torch.from_numpy(validity).to(self.device)
                        )
                        loaded += 1
            except Exception as exc:
                raise RuntimeError(
                    f"AlphaEarth cache load failed after {loaded}/{len(runtimes)} tiles"
                ) from exc

        self.runtimes = tuple(runtimes)
        self.group_tiles = tuple(
            tuple(
                index
                for index, runtime in enumerate(self.runtimes)
                if runtime.group_id == group.group_id
            )
            for group in catalog.groups
        )

    def _value_view(self, runtime: _TileRuntime) -> torch.Tensor:
        return self.values[
            runtime.value_offset : runtime.value_offset
            + NUM_BANDS * runtime.pixels
        ].view(runtime.height, runtime.width, NUM_BANDS)

    def _mask_view(self, runtime: _TileRuntime) -> torch.Tensor:
        return self.validity[
            runtime.mask_offset : runtime.mask_offset + runtime.pixels
        ].view(runtime.height, runtime.width)

    @staticmethod
    def _approximate_scanlines(
        longitude: torch.Tensor,
        latitude: torch.Tensor,
        xres: torch.Tensor,
        yres: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        exact = torch.stack(
            (
                longitude / xres[:, None, None],
                -latitude / yres[:, None, None],
            ),
            dim=0,
        )
        output = exact.clone()
        active = torch.ones_like(longitude[:, :, 0], dtype=torch.bool)

        def interpolate(start: int, count: int, rows: torch.Tensor) -> None:
            if count <= 5:
                return
            middle = start + (count - 1) // 2
            end = start + count - 1
            fraction = (middle - start) / (end - start)
            estimate = exact[..., start] + fraction * (
                exact[..., end] - exact[..., start]
            )
            accepted = rows & (
                (estimate - exact[..., middle]).abs().sum(dim=0) <= 0.125
            )
            offsets = torch.linspace(
                0.0,
                1.0,
                count,
                dtype=torch.float64,
                device=exact.device,
            )
            line = exact[..., start, None] + offsets * (
                exact[..., end] - exact[..., start]
            )[..., None]
            output[..., start : end + 1] = torch.where(
                accepted[None, :, :, None],
                line,
                output[..., start : end + 1],
            )
            rejected = rows & ~accepted
            first_count = middle - start
            interpolate(start, first_count, rejected)
            interpolate(middle, count - first_count, rejected)

        interpolate(0, longitude.shape[-1], active)
        return (
            output[0] * xres[:, None, None],
            -output[1] * yres[:, None, None],
        )

    def materialize(
        self,
        group_ids: torch.Tensor,
        longitude: torch.Tensor,
        latitude: torch.Tensor,
        bounds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        record_groups = group_ids.to(self.device, non_blocking=True)
        longitude = longitude.to(self.device, non_blocking=True)
        latitude = latitude.to(self.device, non_blocking=True)
        bounds_gpu = bounds.to(self.device, non_blocking=True)
        resolutions = torch.tensor(
            [
                (
                    self.runtimes[self.group_tiles[group_id][0]].xres,
                    self.runtimes[self.group_tiles[group_id][0]].yres,
                )
                for group_id in record_groups.tolist()
            ],
            dtype=torch.float64,
            device=self.device,
        )
        longitude, latitude = self._approximate_scanlines(
            longitude,
            latitude,
            resolutions[:, 0],
            resolutions[:, 1],
        )

        candidates: dict[int, list[tuple[object, ...]]] = {}
        for record_index, (group_id, query_bounds) in enumerate(
            zip(record_groups.tolist(), bounds.tolist())
        ):
            matches: list[tuple[int, float]] = []
            for tile_index in self.group_tiles[group_id]:
                runtime = self.runtimes[tile_index]
                qleft, qbottom, qright, qtop = query_bounds
                left, bottom, right, top = runtime.tile.bounds
                for shift in (-360.0, 0.0, 360.0):
                    if (
                        left + shift <= qright
                        and right + shift >= qleft
                        and bottom <= qtop
                        and top >= qbottom
                    ):
                        matches.append((tile_index, shift))
                        break
            if not matches:
                raise FileNotFoundError(
                    "No AlphaEarth tile intersects query bounds "
                    f"{query_bounds} for group_id={group_id}"
                )
            for tile_index, shift in matches:
                candidates.setdefault(tile_index, []).append(
                    (
                        record_index,
                        shift,
                        qleft,
                        qbottom,
                        qright,
                        qtop,
                        len(matches) > 1
                        and qleft >= -180.0
                        and qright <= 180.0,
                    )
                )

        count = len(record_groups)
        height, width = longitude.shape[-2:]
        max_xradius = max(
            1,
            math.ceil(
                (
                    (bounds_gpu[:, 2] - bounds_gpu[:, 0])
                    / resolutions[:, 0]
                    / width
                ).max().item()
            ),
        )
        max_yradius = max(
            1,
            math.ceil(
                (
                    (bounds_gpu[:, 3] - bounds_gpu[:, 1])
                    / resolutions[:, 1]
                    / height
                ).max().item()
            ),
        )
        offsets = [
            (row_offset, col_offset)
            for row_offset in range(-max_yradius, max_yradius + 1)
            for col_offset in range(-max_xradius, max_xradius + 1)
        ]
        offset_index = {offset: index for index, offset in enumerate(offsets)}
        shape = (len(offsets), count, height, width)
        kernel_pixels = torch.zeros(shape, dtype=torch.int32, device=self.device)
        kernel_weights = torch.zeros(shape, dtype=torch.float32, device=self.device)
        kernel_coverage = torch.zeros(shape, dtype=torch.bool, device=self.device)
        nearest_coverage = torch.zeros(
            count, height, width, dtype=torch.bool, device=self.device
        )

        for tile_index in sorted(candidates):
            records = candidates[tile_index]
            runtime = self.runtimes[tile_index]
            record_indices = torch.tensor(
                [record[0] for record in records],
                dtype=torch.long,
                device=self.device,
            )
            metadata = torch.tensor(
                [record[1:] for record in records],
                dtype=torch.float64,
                device=self.device,
            )
            shifts = metadata[:, 0, None, None]
            query_bounds = metadata[:, 1:5]
            merged = metadata[:, 5].bool()
            direct_x = (
                longitude[record_indices] - shifts - runtime.left
            ) / runtime.xres
            direct_y = (
                runtime.top - latitude[record_indices]
            ) / runtime.yres
            source_x = torch.where(
                merged[:, None, None],
                (longitude[record_indices] - query_bounds[:, 0, None, None])
                / runtime.xres,
                direct_x,
            )
            source_y = torch.where(
                merged[:, None, None],
                (query_bounds[:, 3, None, None] - latitude[record_indices])
                / runtime.yres,
                direct_y,
            )
            source_width = (
                query_bounds[:, 2] - query_bounds[:, 0]
            ) / runtime.xres
            source_height = (
                query_bounds[:, 3] - query_bounds[:, 1]
            ) / runtime.yres
            mosaic_width = torch.round(source_width).long()
            mosaic_height = torch.round(source_height).long()
            xscale = width / torch.where(
                merged,
                torch.minimum(source_width, mosaic_width.double()),
                source_width,
            )
            yscale = height / torch.where(
                merged,
                torch.minimum(source_height, mosaic_height.double()),
                source_height,
            )
            for scales in (xscale, yscale):
                reciprocal = 1.0 / scales
                rounded = torch.floor(reciprocal + 0.5)
                scales.copy_(
                    torch.where(
                        (scales < 1.0)
                        & ((reciprocal - rounded).abs() < 0.05),
                        1.0 / rounded,
                        scales,
                    ).clamp_max(1.0)
                )
            four_samples = (xscale >= 0.95) & (yscale >= 0.95)
            xscale = torch.where(four_samples, 1.0, xscale)
            yscale = torch.where(four_samples, 1.0, yscale)
            xradius = math.ceil(1.0 / xscale.min().item())
            yradius = math.ceil(1.0 / yscale.min().item())
            crop = torch.stack(
                (
                    torch.floor(
                        (query_bounds[:, 0] - metadata[:, 0] - runtime.left)
                        / runtime.xres
                    ),
                    torch.floor(
                        (runtime.top - query_bounds[:, 3]) / runtime.yres
                    ),
                    torch.ceil(
                        (query_bounds[:, 2] - metadata[:, 0] - runtime.left)
                        / runtime.xres
                    ),
                    torch.ceil(
                        (runtime.top - query_bounds[:, 1]) / runtime.yres
                    ),
                ),
                dim=1,
            ).long()
            crop[:, 2] = torch.maximum(crop[:, 2], crop[:, 0] + 2)
            crop[:, 3] = torch.maximum(crop[:, 3], crop[:, 1] + 2)
            tile_left = runtime.left + metadata[:, 0]
            intersection_left = torch.maximum(tile_left, query_bounds[:, 0])
            intersection_right = torch.minimum(
                runtime.right + metadata[:, 0],
                query_bounds[:, 0] + mosaic_width * runtime.xres,
            )
            intersection_top = torch.minimum(
                torch.full_like(tile_left, runtime.top), query_bounds[:, 3]
            )
            intersection_bottom = torch.maximum(
                torch.full_like(tile_left, runtime.bottom),
                query_bounds[:, 3] - mosaic_height * runtime.yres,
            )
            source_window_left = (intersection_left - tile_left) / runtime.xres
            source_window_top = (
                runtime.top - intersection_top
            ) / runtime.yres
            region = torch.stack(
                (
                    torch.floor(
                        (intersection_left - query_bounds[:, 0])
                        / runtime.xres
                        + 0.1
                    ),
                    torch.floor(
                        (query_bounds[:, 3] - intersection_top)
                        / runtime.yres
                        + 0.1
                    ),
                    torch.floor(
                        (intersection_right - intersection_left)
                        / runtime.xres
                        + 0.5
                    ),
                    torch.floor(
                        (intersection_top - intersection_bottom)
                        / runtime.yres
                        + 0.5
                    ),
                ),
                dim=1,
            ).long()
            source_window = torch.stack(
                (
                    source_window_left,
                    source_window_top,
                    (intersection_right - intersection_left)
                    / runtime.xres
                    / region[:, 2].clamp_min(1),
                    (intersection_top - intersection_bottom)
                    / runtime.yres
                    / region[:, 3].clamp_min(1),
                ),
                dim=1,
            )
            mosaic_shape = torch.stack((mosaic_width, mosaic_height), dim=1)

            tile_offsets = [
                (row_offset, col_offset)
                for row_offset in range(-yradius, yradius + 1)
                for col_offset in range(-xradius, xradius + 1)
            ]
            offset_indices = torch.tensor(
                [offset_index[offset] for offset in tile_offsets],
                dtype=torch.long,
                device=self.device,
            )
            offset_tensor = torch.tensor(
                tile_offsets, dtype=torch.long, device=self.device
            )
            route_index = (offset_indices[:, None], record_indices[None, :])
            (
                local_pixels,
                tile_weights,
                tile_coverage,
                tile_nearest_coverage,
            ) = _route_tile(
                self._mask_view(runtime),
                source_x,
                source_y,
                xscale,
                yscale,
                merged,
                crop,
                mosaic_shape,
                source_window,
                region,
                kernel_coverage[route_index],
                nearest_coverage[record_indices],
                offset_tensor[:, 0],
                offset_tensor[:, 1],
            )
            selected = tile_weights > 0
            kernel_pixels[route_index] = torch.where(
                selected,
                (local_pixels + runtime.mask_offset).int(),
                kernel_pixels[route_index],
            )
            kernel_weights[route_index] = torch.where(
                selected, tile_weights, kernel_weights[route_index]
            )
            kernel_coverage[route_index] = tile_coverage
            nearest_coverage.index_copy_(
                0, record_indices, tile_nearest_coverage
            )

        numerator, denominator = _gather_samples(
            self.values, kernel_pixels, kernel_weights
        )
        valid = (denominator > 0) & nearest_coverage[:, None]
        values = torch.where(
            valid,
            numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny),
            0.0,
        ).half()
        return values.contiguous(), valid.contiguous()


class DeepSSFGPUMetadataDataset(Dataset[int]):
    def __init__(
        self,
        *,
        study_id: str,
        steps: pd.DataFrame,
        catalog: AlphaEarthCatalog,
        output_hw: tuple[int, int],
        step_buffer_m: float,
    ) -> None:
        required = {
            "point_id_start",
            "timestamp",
            "start_lon",
            "start_lat",
            "end_lon",
            "end_lat",
            "start_bearing",
        }
        missing = sorted(required.difference(steps.columns))
        if missing:
            raise ValueError(f"Step metadata is missing columns: {missing}")
        self.study_id = str(study_id)
        self.output_hw = tuple(int(value) for value in output_hw)
        self.step_buffer_m = float(step_buffer_m)
        self.pixel_size_m = (2.0 * self.step_buffer_m) / (
            float(self.output_hw[1]) - 1.0
        )
        self.centers = steps[["start_lon", "start_lat"]].to_numpy(
            dtype=np.float64, copy=True
        )
        self.endpoints = steps[["end_lon", "end_lat"]].to_numpy(
            dtype=np.float64, copy=True
        )
        self.scalars = _time_features_notebook_order(steps["timestamp"])
        bearing = _bearing_movebench_to_deepssf(
            steps["start_bearing"].to_numpy(dtype=np.float64, copy=False)
        ).astype(np.float32, copy=False)
        self.bearing = torch.from_numpy(bearing).unsqueeze(1)
        self.point_ids = torch.from_numpy(
            steps["point_id_start"].to_numpy(dtype=np.int64, copy=True)
        )
        years = pd.to_datetime(steps["timestamp"], utc=True).dt.year.to_numpy()
        self.group_ids = torch.tensor(
            [catalog.resolve_group_id(self.study_id, int(year)) for year in years],
            dtype=torch.int64,
        )

    def __len__(self) -> int:
        return len(self.point_ids)

    def __getitem__(self, index: int) -> int:
        return int(index)


def build_deepssf_gpu_datasets(
    *,
    study_id: str,
    trajectories_path: str | Path,
    plan: LayerPlan,
    splits: list[str],
    catalog: AlphaEarthCatalog,
    output_hw: tuple[int, int],
) -> dict[str, DeepSSFGPUMetadataDataset]:
    height, width = (int(value) for value in output_hw)
    if height != width or height < 2:
        raise ValueError(
            f"DeepSSF GPU datasets require square output_hw >= 2; got {output_hw}"
        )
    df = load_trajectories_csv(trajectories_path)
    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("DeepSSF GPU datasets require a non-empty train split")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        step_q=float(plan.step_spec.step_q),
        max_step_q=float(plan.step_spec.max_step_q),
        step_multiplier=float(plan.step_spec.step_multiplier),
    )
    selected = df[df[SET_COL].isin(splits)].copy()
    steps = select_step_starts_with_support(
        selected,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        support=support,
    )
    datasets: dict[str, DeepSSFGPUMetadataDataset] = {}
    for split in splits:
        split_steps = steps[steps[SET_COL] == str(split)].reset_index(drop=True)
        if len(split_steps) == 0:
            raise ValueError(
                f"No valid step starts found for split={split!r} in {trajectories_path}"
            )
        datasets[str(split)] = DeepSSFGPUMetadataDataset(
            study_id=study_id,
            steps=split_steps,
            catalog=catalog,
            output_hw=(height, width),
            step_buffer_m=float(support.max_step_length_m),
        )
    return datasets


class _DeepSSFGPUCollator:
    def __init__(self, dataset: DeepSSFGPUMetadataDataset) -> None:
        self.dataset = dataset

    def __call__(self, indices: list[int]) -> dict[str, torch.Tensor]:
        positions = np.asarray(indices, dtype=np.int64)
        centers = self.dataset.centers[positions]
        longitude, latitude, bounds = alphaearth_query_geometry(
            centers,
            self.dataset.pixel_size_m,
            self.dataset.output_hw,
        )
        x_m, y_m = local_xy_m_from_lonlat(
            centers[:, 0],
            centers[:, 1],
            self.dataset.endpoints[positions, 0],
            self.dataset.endpoints[positions, 1],
        )
        pixel_size_m = self.dataset.pixel_size_m
        edge = self.dataset.step_buffer_m + 0.5 * pixel_size_m
        target_row = np.floor((edge - y_m) / pixel_size_m).astype(np.int64)
        target_col = np.floor((x_m + edge) / pixel_size_m).astype(np.int64)
        height, width = self.dataset.output_hw
        target = torch.zeros((len(indices), height, width), dtype=torch.float32)
        valid = (
            (target_row >= 0)
            & (target_row < height)
            & (target_col >= 0)
            & (target_col < width)
        )
        batch_index = torch.arange(len(indices), dtype=torch.long)[
            torch.from_numpy(valid)
        ]
        target[
            batch_index,
            torch.from_numpy(target_row[valid]),
            torch.from_numpy(target_col[valid]),
        ] = 1.0
        tensor_positions = torch.from_numpy(positions)
        return {
            "group_ids": self.dataset.group_ids[tensor_positions],
            "longitude": torch.from_numpy(longitude),
            "latitude": torch.from_numpy(latitude),
            "bounds": torch.from_numpy(bounds),
            "scalars": self.dataset.scalars[tensor_positions],
            "bearing": self.dataset.bearing[tensor_positions],
            "target": target,
            "point_id_start": self.dataset.point_ids[tensor_positions],
        }


class _DeepSSFGPULoader:
    def __init__(self, loader: DataLoader, cache: AlphaEarthGPUCache) -> None:
        self.loader = loader
        self.cache = cache

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            spatial, _valid = self.cache.materialize(
                batch["group_ids"],
                batch["longitude"],
                batch["latitude"],
                batch["bounds"],
            )
            yield (
                spatial,
                batch["scalars"].to(self.cache.device, non_blocking=True),
                batch["bearing"].to(self.cache.device, non_blocking=True),
                batch["target"].to(self.cache.device, non_blocking=True),
                batch["point_id_start"].to(self.cache.device, non_blocking=True),
            )


def make_deepssf_gpu_loader(
    dataset: DeepSSFGPUMetadataDataset,
    *,
    cache: AlphaEarthGPUCache,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> _DeepSSFGPULoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=_DeepSSFGPUCollator(dataset),
        pin_memory=True,
        persistent_workers=False,
        generator=generator,
    )
    return _DeepSSFGPULoader(loader, cache)
