from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from movebench.features.points.sampler import sample_pseudo_steps


TARGET_CRS = "EPSG:3857"  # meters


@lru_cache(maxsize=1)
def _transformers():
    """Construct cached CRS transformers lazily (pyproj is an optional dependency for readers)."""
    from pyproj import Transformer  # imported lazily to avoid requiring pyproj for point-store readers

    to_target = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    to_4326 = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)
    return to_target, to_4326


@dataclass(frozen=True)
class StepRequest:
    point_id_start: int
    point_id_end: int
    timestamp_start: Any
    timestamp_end: Any | None
    start_lon: float
    start_lat: float
    start_bearing: float
    end_lon: float
    end_lat: float


@dataclass(frozen=True)
class CandidateSet:
    lons: np.ndarray
    lats: np.ndarray
    bounds_4326: tuple[float, float, float, float]
    step_lengths_m: np.ndarray
    turn_angles_rad: np.ndarray


def compute_candidates(
    req: StepRequest,
    *,
    sampler_state: dict[str, Any],
    bounds_buffer_deg: float,
) -> CandidateSet:
    pseudo_lens, pseudo_turns = sample_pseudo_steps(sampler_state, req.point_id_start)

    to_target, to_4326 = _transformers()
    x0, y0 = to_target.transform(float(req.start_lon), float(req.start_lat))
    x1, y1 = to_target.transform(float(req.end_lon), float(req.end_lat))

    headings = float(req.start_bearing) + pseudo_turns
    headings = (headings + np.pi) % (2.0 * np.pi) - np.pi

    dx = pseudo_lens * np.sin(headings)
    dy = pseudo_lens * np.cos(headings)
    xs = x0 + dx
    ys = y0 + dy

    # True endpoint appended last.
    xs = np.append(xs, x1)
    ys = np.append(ys, y1)

    # Step lengths / turn angles for all candidates.
    true_sl = float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))
    true_heading = float(np.arctan2(x1 - x0, y1 - y0))
    true_ta = true_heading - float(req.start_bearing)
    true_ta = float((true_ta + np.pi) % (2.0 * np.pi) - np.pi)

    step_lengths = np.append(pseudo_lens, true_sl)
    turn_angles = np.append(pseudo_turns, true_ta)

    lons, lats = to_4326.transform(xs, ys)
    lons = np.asarray(lons, dtype=np.float64)
    lats = np.asarray(lats, dtype=np.float64)

    b = float(bounds_buffer_deg)
    bounds_4326 = (
        float(np.min(lons) - b),
        float(np.min(lats) - b),
        float(np.max(lons) + b),
        float(np.max(lats) + b),
    )

    return CandidateSet(
        lons=lons,
        lats=lats,
        bounds_4326=bounds_4326,
        step_lengths_m=step_lengths.astype(np.float32, copy=False),
        turn_angles_rad=turn_angles.astype(np.float32, copy=False),
    )

