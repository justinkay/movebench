from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from movebench.rasters import Raster


@torch.no_grad()
def sample_from_raster(
    raster: Raster,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    bilinear: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample raster at point locations (EPSG:4326) in raster's native CRS."""
    from pyproj import Transformer  # imported lazily to avoid requiring pyproj for point-store readers

    arr = raster.array
    transform = raster.transform
    src_crs = raster.crs
    nodata = raster.nodata

    C, H, W = arr.shape
    N = int(len(lons))

    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    xs, ys = transformer.transform(lons, lats)

    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    cols = np.asarray(cols)
    rows = np.asarray(rows)

    if nodata is not None:
        if isinstance(nodata, float) and np.isnan(nodata):
            nodata_mask = ~np.isnan(arr)
        elif np.isinf(nodata):
            nodata_mask = ~np.isinf(arr)
        else:
            nodata_mask = arr != nodata
    else:
        nodata_mask = np.isfinite(arr)

    if bilinear:
        arr_clean = np.where(nodata_mask, arr, 0.0).astype(np.float32, copy=False)
        arr_t = torch.from_numpy(arr_clean)

        in_bounds = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_clamped = np.clip(rows, 0, H - 1)
        cols_clamped = np.clip(cols, 0, W - 1)

        x_norm = 2.0 * (cols_clamped / max(W - 1, 1)) - 1.0
        y_norm = 2.0 * (rows_clamped / max(H - 1, 1)) - 1.0

        grid = torch.stack(
            [
                torch.tensor(x_norm, dtype=torch.float32),
                torch.tensor(y_norm, dtype=torch.float32),
            ],
            dim=-1,
        ).view(1, N, 1, 2)

        samples = F.grid_sample(
            arr_t.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        ).squeeze(0).squeeze(-1)

        mask_t = torch.from_numpy(nodata_mask.astype(np.float32, copy=False))
        mask_interp = F.grid_sample(
            mask_t.unsqueeze(0),
            grid,
            mode="bilinear",
            align_corners=True,
            padding_mode="zeros",
        ).squeeze(0).squeeze(-1)

        valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
        valid_t = valid_t & (mask_interp > 0.99)
        return samples, valid_t

    rows_int = np.clip(np.round(rows).astype(int), 0, H - 1)
    cols_int = np.clip(np.round(cols).astype(int), 0, W - 1)
    in_bounds = (rows >= -0.5) & (rows < H - 0.5) & (cols >= -0.5) & (cols < W - 0.5)

    samples = torch.from_numpy(arr[:, rows_int, cols_int]).float()
    valid_t = torch.tensor(in_bounds, dtype=torch.bool).unsqueeze(0).expand(C, -1)
    valid_t = valid_t & torch.from_numpy(nodata_mask[:, rows_int, cols_int])
    return samples, valid_t


def apply_valid_range(
    values: torch.Tensor,
    valid: torch.Tensor,
    valid_range: tuple[float, float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if valid_range is None:
        return values, valid
    lo, hi = float(valid_range[0]), float(valid_range[1])
    in_range = (values >= lo) & (values <= hi)
    valid2 = valid & in_range
    values2 = torch.where(valid2, values, torch.zeros_like(values))
    return values2, valid2

