"""deepSSF dataset wrapper on top of ChipDataset(mode='steps').

deepSSF is trained/evaluated on a per-chip (H,W) grid with a one-hot target at
the observed endpoint pixel.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.datasets.chips import ChipDataset


def _bearing_movebench_to_deepssf(bearing_north0: np.ndarray) -> np.ndarray:
    """Convert MoveBench bearing convention to deepSSF convention.

    MoveBench: 0=north, +pi/2=east
    deepSSF:   0=east,  +pi/2=north
    """
    bearing_east0 = (math.pi / 2.0) - bearing_north0
    return (bearing_east0 + math.pi) % (2.0 * math.pi) - math.pi


def _time_features_notebook_order(timestamps: pd.Series) -> torch.Tensor:
    """Return (N,4) time features matching notebooks/deepssf.ipynb ordering.

    Ordering: [sin_tod, cos_tod, sin_doy, cos_doy]
    """
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if ts.isna().any():
        bad = int(ts.isna().sum())
        raise ValueError(f"Found {bad} NaT timestamps in step starts; cannot build deepSSF time features")

    doy = ts.dt.dayofyear.to_numpy(dtype=np.float64)
    hour = ts.dt.hour.to_numpy(dtype=np.float64)
    minute = ts.dt.minute.to_numpy(dtype=np.float64)
    second = ts.dt.second.to_numpy(dtype=np.float64)

    hour_frac = hour + minute / 60.0 + second / 3600.0

    sin_doy = np.sin(2.0 * math.pi * doy / 365.25)
    cos_doy = np.cos(2.0 * math.pi * doy / 365.25)
    sin_tod = np.sin(2.0 * math.pi * hour_frac / 24.0)
    cos_tod = np.cos(2.0 * math.pi * hour_frac / 24.0)

    feats = np.stack([sin_tod, cos_tod, sin_doy, cos_doy], axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(feats)


class DeepSSFDataset(Dataset):
    """Wrap a ChipDataset(mode='steps') to provide deepSSF model inputs."""

    def __init__(self, chips: ChipDataset) -> None:
        if getattr(chips, "mode", None) != "steps":
            raise ValueError("DeepSSFDataset requires ChipDataset(mode='steps')")
        self.chips = chips

        steps = chips.step_starts_df
        for col in ("timestamp", "start_bearing"):
            if col not in steps.columns:
                raise ValueError(f"ChipDataset.step_starts_df missing required column: {col!r}")

        bearing_north0 = steps["start_bearing"].to_numpy(dtype=np.float64, copy=False)
        bearing_east0 = _bearing_movebench_to_deepssf(bearing_north0).astype(np.float32, copy=False)
        self._bearing = torch.from_numpy(bearing_east0)  # (N,)

        self._time = _time_features_notebook_order(steps["timestamp"])  # (N,4)

        if int(len(self._bearing)) != int(len(self.chips)) or int(self._time.shape[0]) != int(len(self.chips)):
            raise RuntimeError(
                "Internal error: ChipDataset length does not match step_starts_df length "
                f"(chips={len(self.chips)} steps={len(steps)})"
            )

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial, _valid, target, _xfm, _crs = self.chips[int(idx)]
        spatial = spatial.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)

        scalars_to_grid = self._time[int(idx)]  # (4,)
        bearing = self._bearing[int(idx)].unsqueeze(0)  # (1,)

        return spatial, scalars_to_grid, bearing, target
