from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch

from movebench.datasets import DataSplit
from movebench.features.points import PointStoreReader
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    TIME_COL,
    IntervalStepSupport,
    bearing_rad,
    fit_train_step_support,
    geodesic_distance_m,
    load_trajectories_csv,
)


class PointContext(TypedDict):
    """CPU [N] tensors in point-store order: int64 IDs and float32 movement context."""

    point_id_start: torch.Tensor
    start_bearing_rad: torch.Tensor
    prev_step_length_m: torch.Tensor


def _step_context(shared: dict[str, torch.Tensor], trajectories_path: Path) -> PointContext:
    point_ids = shared.get("point_id_start")
    if point_ids is None:
        raise ValueError("point store shared rows missing point_id_start required by baselines")

    rows: list[dict[str, object]] = []
    with open(trajectories_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = str(row.get(TIME_COL, "")).strip()
            if not timestamp:
                continue
            rows.append(
                {
                    "point_id": int(row[POINT_ID_COL]),
                    "timestamp": timestamp,
                    "lon": float(row[LON_COL]),
                    "lat": float(row[LAT_COL]),
                    "individual": str(row[INDIVIDUAL_COL]),
                }
            )

    rows.sort(key=lambda row: (str(row["individual"]), str(row["timestamp"])))
    prev_by_individual: dict[str, dict[str, object]] = {}
    by_point: dict[int, dict[str, object]] = {}
    for row in rows:
        prev = prev_by_individual.get(str(row["individual"]))
        if prev is None:
            prev_step_length_m = float("nan")
        else:
            prev_step_length_m = float(
                geodesic_distance_m(
                    [float(prev["lat"])],
                    [float(prev["lon"])],
                    [float(row["lat"])],
                    [float(row["lon"])],
                )[0]
            )
        row["prev_step_length_m"] = prev_step_length_m
        by_point[int(row["point_id"])] = row
        prev_by_individual[str(row["individual"])] = row

    wanted = [int(point_id) for point_id in point_ids.tolist()]
    selected_rows = [by_point[point_id] for point_id in wanted]
    start_lon = [float(row["lon"]) for row in selected_rows]
    start_lat = [float(row["lat"]) for row in selected_rows]

    true_lon = shared["candidate_lon"][:, -1].to(dtype=torch.float64).numpy()
    true_lat = shared["candidate_lat"][:, -1].to(dtype=torch.float64).numpy()
    true_heading = bearing_rad(start_lat, start_lon, true_lat, true_lon)
    true_turn = shared["turn_angle_rad"][:, -1].to(dtype=torch.float64).numpy()
    start_bearing = (true_heading - true_turn + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "point_id_start": point_ids.to(dtype=torch.int64).cpu(),
        "start_bearing_rad": torch.tensor(start_bearing, dtype=torch.float32),
        "prev_step_length_m": torch.tensor(
            [float(row["prev_step_length_m"]) for row in selected_rows],
            dtype=torch.float32,
        ),
    }


def _load_shared(
    *,
    store_dir: Path,
    manifest,
) -> dict[str, torch.Tensor]:
    reader = PointStoreReader(store_dir=store_dir, manifest=manifest)
    return reader.load_shared_rows(
        keys=["candidate_lon", "candidate_lat", "turn_angle_rad", "point_id_start"],
        row_idxs=np.arange(manifest.n_rows, dtype=np.int64),
    )


def point_context(data: DataSplit) -> PointContext:
    store = data.inputs["point_stores"][data.split_key]
    shared = _load_shared(store_dir=store["store_dir"], manifest=store["manifest"])
    return _step_context(shared, data.inputs["trajectories_path"])


def training_steps(train: DataSplit) -> tuple[IntervalStepSupport, pd.DataFrame]:
    return fit_train_step_support(
        load_trajectories_csv(train.inputs["trajectories_path"]),
        step_spec=train.inputs["plan"].step_spec,
    )
