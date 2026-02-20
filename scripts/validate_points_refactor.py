#!/usr/bin/env python
"""Lightweight, data-free validation for the point-store refactor.

This script intentionally avoids any raster IO. It checks:
- step selection returns (point_id_start, point_id_end) correctly,
- PointStoreConfig.store_id is stable/deterministic,
- bounds_buffer_deg expands candidate bounds as expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_step_selection() -> None:
    from movebench.trajectories import (
        INDIVIDUAL_COL,
        LAT_COL,
        LON_COL,
        POINT_ID_COL,
        SET_COL,
        TIME_COL,
        select_step_starts,
    )

    df = pd.DataFrame(
        {
            INDIVIDUAL_COL: ["a", "a", "a", "b", "b", "b"],
            TIME_COL: pd.to_datetime(
                [
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                    "2020-01-01T02:00:00Z",
                    "2020-01-01T00:00:00Z",
                    "2020-01-01T01:00:00Z",
                    "2020-01-01T02:00:00Z",
                ]
            ),
            LON_COL: [0.0, 0.001, 0.002, 10.0, 10.001, 10.002],
            LAT_COL: [0.0, 0.0, 0.0, 45.0, 45.0, 45.0],
            POINT_ID_COL: [100, 101, 102, 200, 201, 202],
            SET_COL: ["train"] * 6,
        }
    )

    steps = select_step_starts(df, step_multiplier=100.0, step_q=1.0, gap_q=1.0)
    got = list(zip(steps["point_id_start"].tolist(), steps["point_id_end"].tolist()))
    want = [(101, 102), (201, 202)]
    if got != want:
        raise AssertionError(f"select_step_starts returned {got}, expected {want}")

    for col in ("timestamp", "end_timestamp", "start_lon", "start_lat", "start_bearing", "end_lon", "end_lat"):
        if col not in steps.columns:
            raise AssertionError(f"select_step_starts missing expected column: {col!r}")


def _check_store_id_determinism() -> None:
    from movebench.features.points import PointStoreConfig

    cfg1 = PointStoreConfig(
        sampler_type="parametric",
        num_candidates=10,
        random_seed=0,
        step_multiplier=2.0,
        step_q=0.95,
        gap_q=0.999,
        bounds_buffer_deg=0.01,
    )
    cfg2 = PointStoreConfig(
        sampler_type="parametric",
        num_candidates=10,
        random_seed=0,
        step_multiplier=2.0,
        step_q=0.95,
        gap_q=0.999,
        bounds_buffer_deg=0.01,
    )
    if cfg1.store_id() != cfg2.store_id():
        raise AssertionError("PointStoreConfig.store_id is not deterministic for identical configs")

    cfg3 = PointStoreConfig(
        sampler_type="parametric",
        num_candidates=10,
        random_seed=0,
        step_multiplier=2.0,
        step_q=0.95,
        gap_q=0.999,
        bounds_buffer_deg=0.02,
    )
    if cfg1.store_id() == cfg3.store_id():
        raise AssertionError("PointStoreConfig.store_id did not change when bounds_buffer_deg changed")


def _check_candidate_bounds_buffer() -> None:
    from movebench.features.points import StepRequest, compute_candidates

    sampler_state = {
        "valid_lengths": np.array([100.0, 200.0], dtype=np.float64),
        "valid_turns": np.array([0.0, 0.1], dtype=np.float64),
        "num_candidates": 5,
        "random_seed": 123,
        "gamma_used": False,
        "gamma_params": None,
        "vonmises_used": False,
        "vonmises_params": None,
    }
    req = StepRequest(
        point_id_start=42,
        point_id_end=43,
        timestamp_start=pd.Timestamp("2020-01-01T01:00:00Z"),
        timestamp_end=pd.Timestamp("2020-01-01T02:00:00Z"),
        start_lon=0.0,
        start_lat=0.0,
        start_bearing=0.0,
        end_lon=0.01,
        end_lat=0.0,
    )

    cand0 = compute_candidates(req, sampler_state=sampler_state, bounds_buffer_deg=0.0)
    cand1 = compute_candidates(req, sampler_state=sampler_state, bounds_buffer_deg=0.5)

    if not np.allclose(cand0.lons, cand1.lons) or not np.allclose(cand0.lats, cand1.lats):
        raise AssertionError("bounds_buffer_deg should not affect candidate locations")

    b0 = cand0.bounds_4326
    b1 = cand1.bounds_4326
    if not np.isclose(b1[0], b0[0] - 0.5) or not np.isclose(b1[1], b0[1] - 0.5):
        raise AssertionError("bounds_buffer_deg did not expand min bounds as expected")
    if not np.isclose(b1[2], b0[2] + 0.5) or not np.isclose(b1[3], b0[3] + 0.5):
        raise AssertionError("bounds_buffer_deg did not expand max bounds as expected")


def main() -> int:
    _check_step_selection()
    _check_store_id_determinism()
    _check_candidate_bounds_buffer()
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

