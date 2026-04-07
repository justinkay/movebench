from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

import movebench.bench.config as bench_config
import movebench.bench.executors.ssf_grid as ssf_grid
from pyproj import Transformer

from movebench.features.points import PointStoreReader
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    TIME_COL,
    bearing_rad,
    haversine_distance_m,
    load_trajectories_csv,
)

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _step_context(shared: dict[str, torch.Tensor], trajectories_path: Path) -> dict[str, torch.Tensor]:
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
                haversine_distance_m(
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
    start_x, start_y = _TO_3857.transform(start_lon, start_lat)

    true_lon = shared["candidate_lon"][:, -1].to(dtype=torch.float64).numpy()
    true_lat = shared["candidate_lat"][:, -1].to(dtype=torch.float64).numpy()
    true_heading = bearing_rad(start_lat, start_lon, true_lat, true_lon)
    true_turn = shared["turn_angle_rad"][:, -1].to(dtype=torch.float64).numpy()
    start_bearing = (true_heading - true_turn + math.pi) % (2.0 * math.pi) - math.pi

    return {
        "point_id_start": point_ids.to(dtype=torch.int64).cpu(),
        "start_x_m": torch.tensor(start_x, dtype=torch.float32),
        "start_y_m": torch.tensor(start_y, dtype=torch.float32),
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


def _write_predictions(
    run_dir: Path,
    *,
    split_key: str,
    payload: dict[str, torch.Tensor | str],
) -> None:
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, pred_dir / f"{split_key}.pt")


def run_nomove(*, run_payload, resolved_artifacts, run_dir, tracker):
    trajectories_path = Path(resolved_artifacts["trajectories_path"])
    for split_key in run_payload["evaluation_split_keys"]:
        shared = _load_shared(
            store_dir=Path(resolved_artifacts["point_stores"][split_key]["store_dir"]),
            manifest=resolved_artifacts["point_stores"][split_key]["manifest"],
        )
        ctx = _step_context(shared, trajectories_path)
        zeros = torch.zeros_like(ctx["start_x_m"])
        _write_predictions(
            Path(run_dir),
            split_key=str(split_key),
            payload={
                "prediction_kind": "deterministic",
                "point_id_start": ctx["point_id_start"],
                "dx_m": zeros,
                "dy_m": zeros,
            },
        )
    return {
        "status": "completed",
        "train_steps": 0,
        "early_stop_split_key": run_payload["early_stop_split_key"],
    }


def run_constvel(*, run_payload, resolved_artifacts, run_dir, tracker):
    trajectories_path = Path(resolved_artifacts["trajectories_path"])
    for split_key in run_payload["evaluation_split_keys"]:
        shared = _load_shared(
            store_dir=Path(resolved_artifacts["point_stores"][split_key]["store_dir"]),
            manifest=resolved_artifacts["point_stores"][split_key]["manifest"],
        )
        ctx = _step_context(shared, trajectories_path)
        dx_m = ctx["prev_step_length_m"] * torch.sin(ctx["start_bearing_rad"])
        dy_m = ctx["prev_step_length_m"] * torch.cos(ctx["start_bearing_rad"])
        _write_predictions(
            Path(run_dir),
            split_key=str(split_key),
            payload={
                "prediction_kind": "deterministic",
                "point_id_start": ctx["point_id_start"],
                "dx_m": dx_m,
                "dy_m": dy_m,
            },
        )
    return {
        "status": "completed",
        "train_steps": 0,
        "early_stop_split_key": run_payload["early_stop_split_key"],
    }


def run_randomwalk(*, run_payload, resolved_artifacts, run_dir, tracker):
    trajectories_path = Path(resolved_artifacts["trajectories_path"])
    trajectories_df = load_trajectories_csv(trajectories_path)
    _support, train_steps = ssf_grid.fit_train_step_support(
        trajectories_df,
        step_spec=resolved_artifacts["plan"].step_spec,
    )
    train_manifest = resolved_artifacts["point_stores"]["train"]["manifest"]
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams.get("batch_size", 256))
    num_workers = int(method_hparams.get("num_workers", 0) or 0)
    output_hw = tuple(
        int(v)
        for v in bench_config.resolve_timescale(str(run_payload["timescale"]))[
            "output_hw"
        ]
    )
    for split_key in run_payload["evaluation_split_keys"]:
        split_manifest = resolved_artifacts["point_stores"][split_key]["manifest"]
        grid_dataset = ssf_grid.make_grid_step_dataset(
            study_dir=resolved_artifacts["study_dir"],
            trajectories_path=trajectories_path,
            split_key=str(split_key),
            plan=resolved_artifacts["plan"],
            sampler_config=resolved_artifacts["sampler"],
            output_hw=output_hw,
            selected_splits=ssf_grid.selected_splits_from_manifest(split_manifest),
            include_env=False,
        )
        _write_predictions(
            Path(run_dir),
            split_key=str(split_key),
            payload=ssf_grid.build_ssf_grid_artifact(
                grid_dataset=grid_dataset,
                train_manifest=train_manifest,
                train_steps=train_steps,
                batch_size=batch_size,
                num_workers=num_workers,
                device=torch.device("cpu"),
                model=None,
                feature_adapter=None,
            ),
        )
    return {
        "status": "completed",
        "train_steps": 0,
        "early_stop_split_key": run_payload["early_stop_split_key"],
    }
