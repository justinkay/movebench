"""Shared low-level movement math utilities."""

from __future__ import annotations

import numpy as np
import torch

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1_deg: np.ndarray | float,
    lon1_deg: np.ndarray | float,
    lat2_deg: np.ndarray | float,
    lon2_deg: np.ndarray | float,
) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    lat1 = np.deg2rad(lat1_deg)
    lat2 = np.deg2rad(lat2_deg)
    dlat = lat2 - lat1
    dlon = np.deg2rad(np.asarray(lon2_deg) - np.asarray(lon1_deg))
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))
    return EARTH_RADIUS_M * c


def safe_speed_m_s_numpy(
    step_length_m: np.ndarray | float,
    step_dt_s: np.ndarray | float,
) -> np.ndarray:
    """Compute speed in m/s with nonpositive dt mapped to NaN."""
    sl = np.asarray(step_length_m, dtype=np.float64)
    dt = np.asarray(step_dt_s, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dt > 0, sl / dt, np.nan)


def safe_speed_m_s_torch(step_length_m: torch.Tensor, step_dt_s: torch.Tensor) -> torch.Tensor:
    """Compute speed in m/s with nonpositive dt mapped to NaN.

    ``step_dt_s`` may have lower rank than ``step_length_m`` and will be unsqueezed
    on trailing axes for broadcasting.
    """
    dt = step_dt_s
    while dt.ndim < step_length_m.ndim:
        dt = dt.unsqueeze(-1)
    return torch.where(dt > 0, step_length_m / dt, torch.full_like(step_length_m, float("nan")))
