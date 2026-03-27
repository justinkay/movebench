from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
from typing import Any

import torch
from pyproj import Transformer
from rasterio.transform import from_bounds, rowcol

import movebench.paths as paths

import movebench.bench_v2.config as bench_config
import movebench.bench_v2.mlflow as bench_mlflow
from movebench.bench_v2.evaluation import energy
from movebench.trajectories import (
    SET_COL,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def _load_json(path: Path) -> dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _metric_split_key(split_key: str) -> str:
    return str(split_key).replace("-", "_")


def _prediction_path(run_dir: Path, split_key: str) -> Path:
    return run_dir / "predictions" / f"{split_key}.pt"


def _truth_xy_m_for_step(
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
) -> torch.Tensor:
    start_x_m, start_y_m = _TO_3857.transform(float(start_lon), float(start_lat))
    end_x_m, end_y_m = _TO_3857.transform(float(end_lon), float(end_lat))
    return torch.tensor(
        [float(end_x_m) - float(start_x_m), float(end_y_m) - float(start_y_m)],
        dtype=torch.float64,
    )


def _support_transform(*, output_hw: tuple[int, int], buffer_m: float):
    height, width = (int(output_hw[0]), int(output_hw[1]))
    if height != width:
        raise ValueError(f"expected square output_hw, got {list(output_hw)}")
    if width < 2:
        raise ValueError(f"expected output_hw width >= 2, got {list(output_hw)}")
    pixel_size_m = (2.0 * float(buffer_m)) / float(width - 1)
    return (
        from_bounds(
            -float(buffer_m) - 0.5 * pixel_size_m,
            -float(buffer_m) - 0.5 * pixel_size_m,
            float(buffer_m) + 0.5 * pixel_size_m,
            float(buffer_m) + 0.5 * pixel_size_m,
            width,
            height,
        ),
        pixel_size_m,
    )


def _is_truth_in_support(
    truth_xy_m: torch.Tensor,
    *,
    output_hw: tuple[int, int],
    buffer_m: float,
) -> bool:
    transform, _pixel_size_m = _support_transform(
        output_hw=output_hw,
        buffer_m=buffer_m,
    )
    rr, cc = rowcol(
        transform,
        float(truth_xy_m[0]),
        float(truth_xy_m[1]),
    )
    return 0 <= int(rr) < int(output_hw[0]) and 0 <= int(cc) < int(output_hw[1])


def _support_rows_for_split(
    *,
    study_id: str,
    split_key: str,
    timescale_key: str,
) -> tuple[list[dict[str, object]], float]:
    study_dir = paths.study_dir(str(study_id))
    trajectories_path = study_dir / "trajectories.csv"
    df = load_trajectories_csv(trajectories_path)
    timescale = bench_config.resolve_timescale(str(timescale_key))
    plan = timescale["plan"]

    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError(f"Study {study_id!r} has no train split for support reconstruction")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(timescale["step_time_s"]),
        gap_q=float(plan.step_spec.gap_q),
        step_q=float(plan.step_spec.step_q),
        max_step_q=float(plan.step_spec.max_step_q),
        step_multiplier=float(plan.step_spec.step_multiplier),
    )
    steps = select_step_starts_with_support(
        df[df[SET_COL] == str(split_key)].copy(),
        step_time_s=int(timescale["step_time_s"]),
        gap_q=float(plan.step_spec.gap_q),
        support=support,
    )

    rows: list[dict[str, object]] = []
    output_hw = tuple(int(v) for v in timescale["output_hw"])
    for step_index, row in enumerate(steps.itertuples(index=False)):
        truth_xy_m = _truth_xy_m_for_step(
            row.start_lon,
            row.start_lat,
            row.end_lon,
            row.end_lat,
        )
        rows.append(
            {
                "step_index": int(step_index),
                "point_id_start": int(row.point_id_start),
                "point_id_end": int(row.point_id_end),
                "truth_xy_m": truth_xy_m,
                "in_support": _is_truth_in_support(
                    truth_xy_m,
                    output_hw=output_hw,
                    buffer_m=float(support.max_step_length_m),
                ),
            }
        )
    return rows, float(support.max_step_length_m)


def _point_rows_by_id(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for row in rows:
        point_id = int(row["point_id_start"])
        if point_id in out:
            raise ValueError(f"duplicate point_id_start {point_id} in support reconstruction")
        out[point_id] = row
    return out


def _tensor_1d(value: Any, *, field: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1:
        raise ValueError(f"{field} must be rank-1, got {tuple(tensor.shape)}")
    return tensor


def _deterministic_contracts(pred_artifact: dict[str, object]) -> tuple[torch.Tensor, list[dict[str, object]]]:
    point_ids = _tensor_1d(pred_artifact["point_id_start"], field="point_id_start").to(dtype=torch.int64)
    dx_m = _tensor_1d(pred_artifact["dx_m"], field="dx_m").to(dtype=torch.float64)
    dy_m = _tensor_1d(pred_artifact["dy_m"], field="dy_m").to(dtype=torch.float64)
    if int(point_ids.shape[0]) != int(dx_m.shape[0]) or int(point_ids.shape[0]) != int(dy_m.shape[0]):
        raise ValueError("deterministic artifact fields must have the same number of steps")
    rows = [
        {
            "prediction_kind": "deterministic",
            "dx_m": float(dx_m[idx].item()),
            "dy_m": float(dy_m[idx].item()),
        }
        for idx in range(int(point_ids.shape[0]))
    ]
    return point_ids, rows


def _discrete_contracts(pred_artifact: dict[str, object]) -> tuple[torch.Tensor, list[dict[str, object]]]:
    point_ids = _tensor_1d(pred_artifact["point_id_start"], field="point_id_start").to(dtype=torch.int64)
    coords_m = torch.as_tensor(pred_artifact["coords_m"], dtype=torch.float64)
    probs = torch.as_tensor(pred_artifact["probs"], dtype=torch.float64)
    if coords_m.ndim != 3 or int(coords_m.shape[2]) != 2:
        raise ValueError(f"coords_m must have shape (N,P,2), got {tuple(coords_m.shape)}")
    if probs.ndim != 2:
        raise ValueError(f"probs must have shape (N,P), got {tuple(probs.shape)}")
    if int(point_ids.shape[0]) != int(coords_m.shape[0]) or int(point_ids.shape[0]) != int(probs.shape[0]):
        raise ValueError("discrete artifact fields must have the same number of steps")
    if int(coords_m.shape[1]) != int(probs.shape[1]):
        raise ValueError("discrete artifact coords_m and probs must share the same support size")
    rows = [
        {
            "prediction_kind": "discrete",
            "coords_m": coords_m[idx],
            "probs": probs[idx],
        }
        for idx in range(int(point_ids.shape[0]))
    ]
    return point_ids, rows


def _grid_coordinates(*, width: int, height: int, pixel_size_m: float) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [(col - (width // 2)) * float(pixel_size_m) for col in range(width)],
            dtype=torch.float64,
        ),
        torch.tensor(
            [((height // 2) - row) * float(pixel_size_m) for row in range(height)],
            dtype=torch.float64,
        ),
    )


def _grid_contracts(
    pred_artifact: dict[str, object],
    *,
    expected_pixel_size_m: float,
    expected_output_hw: tuple[int, int],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    if "point_id_start" not in pred_artifact:
        raise ValueError("grid prediction artifact must include point_id_start")
    point_ids = _tensor_1d(
        pred_artifact["point_id_start"],
        field="point_id_start",
    ).to(dtype=torch.int64)
    if "scores" in pred_artifact:
        scores = torch.as_tensor(pred_artifact["scores"], dtype=torch.float64)
        if scores.ndim != 3:
            raise ValueError(f"grid scores must have shape (N,H,W), got {tuple(scores.shape)}")
        pixel_size_m = float(pred_artifact["pixel_size_m"])
        height, width = int(scores.shape[1]), int(scores.shape[2])
        x_coords_m, y_coords_m = _grid_coordinates(
            width=width,
            height=height,
            pixel_size_m=pixel_size_m,
        )
        contracts = [
            {
                "prediction_kind": "grid",
                "probs_hw": torch.softmax(scores[idx].reshape(-1), dim=0).reshape(height, width),
                "x_coords_m": x_coords_m,
                "y_coords_m": y_coords_m,
            }
            for idx in range(int(scores.shape[0]))
        ]
    else:
        probs_hw = torch.as_tensor(pred_artifact["probs_hw"], dtype=torch.float64)
        if probs_hw.ndim == 2:
            probs_hw = probs_hw.unsqueeze(0)
        if probs_hw.ndim != 3:
            raise ValueError(f"grid probs_hw must have shape (N,H,W), got {tuple(probs_hw.shape)}")
        height, width = int(probs_hw.shape[1]), int(probs_hw.shape[2])
        x_coords_m = torch.as_tensor(pred_artifact["x_coords_m"], dtype=torch.float64)
        y_coords_m = torch.as_tensor(pred_artifact["y_coords_m"], dtype=torch.float64)
        contracts = [
            {
                "prediction_kind": "grid",
                "probs_hw": probs_hw[idx],
                "x_coords_m": x_coords_m,
                "y_coords_m": y_coords_m,
            }
            for idx in range(int(probs_hw.shape[0]))
        ]
        if int(x_coords_m.numel()) != width or int(y_coords_m.numel()) != height:
            raise ValueError("grid coordinate arrays must match probs_hw shape")
        if width > 1:
            pixel_size_m = float(torch.diff(x_coords_m)[0].abs().item())
        elif height > 1:
            pixel_size_m = float(torch.diff(y_coords_m)[0].abs().item())
        else:
            pixel_size_m = 0.0

    if (height, width) != tuple(int(v) for v in expected_output_hw):
        raise ValueError(
            f"grid prediction shape {height}x{width} does not match expected output_hw {list(expected_output_hw)}"
        )
    if not math.isclose(float(pixel_size_m), float(expected_pixel_size_m), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            f"grid prediction pixel_size_m={pixel_size_m} does not match expected {expected_pixel_size_m}"
        )

    if int(point_ids.shape[0]) != len(contracts):
        raise ValueError(
            f"grid artifact point_id_start has {int(point_ids.shape[0])} rows but predictions contain {len(contracts)} rows"
        )
    return point_ids, contracts


def _score_split(
    *,
    split_key: str,
    pred_artifact: dict[str, object],
    run_payload: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    ordered_rows, buffer_m = _support_rows_for_split(
        study_id=str(run_payload["study_id"]),
        split_key=str(split_key),
        timescale_key=str(run_payload["timescale"]),
    )
    truth_by_point = _point_rows_by_id(ordered_rows)
    expected_point_ids = [int(row["point_id_start"]) for row in ordered_rows]
    pred_kind = str(pred_artifact.get("prediction_kind", "")).strip()
    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    output_hw = tuple(int(v) for v in timescale["output_hw"])
    _transform, expected_pixel_size_m = _support_transform(
        output_hw=output_hw,
        buffer_m=float(buffer_m),
    )

    if pred_kind == "deterministic":
        point_ids, contracts = _deterministic_contracts(pred_artifact)
    elif pred_kind == "discrete":
        point_ids, contracts = _discrete_contracts(pred_artifact)
    elif pred_kind == "grid":
        point_ids, contracts = _grid_contracts(
            pred_artifact,
            expected_pixel_size_m=float(expected_pixel_size_m),
            expected_output_hw=output_hw,
        )
    else:
        raise ValueError(f"Unsupported prediction_kind {pred_kind!r}")

    if len(contracts) != len(ordered_rows):
        raise ValueError(
            f"prediction artifact for split {split_key!r} contains {len(contracts)} rows but {len(ordered_rows)} steps were reconstructed"
        )
    actual_point_ids = [int(point_id) for point_id in point_ids.tolist()]
    if len(set(actual_point_ids)) != len(actual_point_ids):
        raise ValueError(f"prediction artifact for split {split_key!r} contains duplicate point_id_start values")
    if set(actual_point_ids) != set(expected_point_ids):
        raise ValueError(
            f"prediction artifact for split {split_key!r} point_id_start values do not match reconstructed steps"
        )
    aligned_rows = [truth_by_point[point_id] for point_id in actual_point_ids]

    by_step_rows: list[dict[str, object]] = []
    excluded = 0
    for point_id, contract, meta in zip(actual_point_ids, contracts, aligned_rows):
        if int(meta["point_id_start"]) != int(point_id):
            raise ValueError(
                f"prediction row point_id_start={int(point_id)} does not align with reconstructed step point_id_start={int(meta['point_id_start'])}"
            )
        if not bool(meta["in_support"]):
            excluded += 1
            continue

        truth_xy_m = meta["truth_xy_m"]
        if pred_kind == "deterministic":
            score, accuracy, calibration = energy._energy_terms_deterministic(contract, truth_xy_m)
        elif pred_kind == "discrete":
            score, accuracy, calibration = energy._energy_terms_discrete(contract, truth_xy_m)
        else:
            score, accuracy, calibration = energy._energy_terms_grid(contract, truth_xy_m)
        by_step_rows.append(
            {
                "step_index": int(meta["step_index"]),
                "point_id_start": int(point_id),
                "energy_score": float(score),
                "accuracy": float(accuracy),
                "calibration": float(calibration),
            }
        )

    summary: dict[str, object] = {
        "split_key": str(split_key),
        "prediction_kind": pred_kind,
        "n_steps_total": int(len(ordered_rows)),
        "n_steps_scored": int(len(by_step_rows)),
        "n_steps_excluded_out_of_support": int(excluded),
        "support_buffer_m": float(buffer_m),
        "support_pixel_size_m": float(expected_pixel_size_m),
    }
    if by_step_rows:
        summary.update(
            {
                "energy_score": float(sum(float(row["energy_score"]) for row in by_step_rows) / float(len(by_step_rows))),
                "accuracy": float(sum(float(row["accuracy"]) for row in by_step_rows) / float(len(by_step_rows))),
                "calibration": float(sum(float(row["calibration"]) for row in by_step_rows) / float(len(by_step_rows))),
            }
        )
    else:
        summary.update(
            {
                "energy_score": None,
                "accuracy": None,
                "calibration": None,
            }
        )
    return by_step_rows, summary


def _write_energy_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step_index", "point_id_start", "energy_score", "accuracy", "calibration"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _log_metrics_to_mlflow(run_key: str, metrics: dict[str, float]) -> None:
    latest_runs = bench_mlflow.search_latest_runs_by_key([str(run_key)])
    latest_run = latest_runs.get(str(run_key))
    if latest_run is None:
        raise ValueError(f"No MLflow run found for run_key {run_key!r}")
    client = bench_mlflow._client(create=True)
    if client is None:
        raise ValueError("MLflow tracking client is unavailable")
    run_id = str(getattr(getattr(latest_run, "info", None), "run_id", "")).strip()
    if not run_id:
        raise ValueError(f"Latest MLflow run for {run_key!r} is missing run_id")
    for key, value in metrics.items():
        client.log_metric(run_id, str(key), float(value))


def evaluate_run_dir(run_dir: str | Path) -> dict[str, object]:
    """
    Prediction scorer contract at the evaluation boundary:

    `prediction_kind == "deterministic"`:
        {"prediction_kind": "deterministic", "point_id_start": ..., "dx_m": ..., "dy_m": ...}

    `prediction_kind == "discrete"`:
        {"prediction_kind": "discrete", "point_id_start": ..., "coords_m": ..., "probs": ...}

    `prediction_kind == "grid"`:
        {
            "prediction_kind": "grid",
            "point_id_start": ...,
            "scores": ...,
            "pixel_size_m": ...,
        }

    The runner converts each split artifact into per-step scorer inputs before
    computing exact energy terms.
    """

    run_dir = Path(run_dir)
    run_payload = _load_json(run_dir / "resolved_run.json")
    training_summary = _load_json(run_dir / "training_summary.json")
    if str(training_summary.get("status", "")) != "completed":
        raise ValueError(
            f"Only completed runs can be evaluated, got status={training_summary.get('status')!r}"
        )

    split_summaries: dict[str, dict[str, object]] = {}
    mlflow_metrics: dict[str, float] = {}
    for split_key in list(run_payload.get("evaluation_split_keys", [])):
        pred_path = _prediction_path(run_dir, str(split_key))
        if not pred_path.exists():
            continue
        pred_artifact = torch.load(pred_path, map_location="cpu")
        rows, summary = _score_split(
            split_key=str(split_key),
            pred_artifact=dict(pred_artifact),
            run_payload=run_payload,
        )
        out_dir = run_dir / "final_metrics" / str(split_key)
        _write_energy_rows(out_dir / "energy_by_step.csv.gz", rows)
        _write_json(out_dir / "summary.json", summary)
        split_summaries[str(split_key)] = summary

        metric_split = _metric_split_key(str(split_key))
        if summary["energy_score"] is not None:
            mlflow_metrics[f"energy_score.{metric_split}"] = float(summary["energy_score"])
            mlflow_metrics[f"energy_score.accuracy.{metric_split}"] = float(summary["accuracy"])
            mlflow_metrics[f"energy_score.calibration.{metric_split}"] = float(summary["calibration"])
        mlflow_metrics[f"n_steps_scored.{metric_split}"] = float(summary["n_steps_scored"])
        mlflow_metrics[f"n_steps_excluded_out_of_support.{metric_split}"] = float(
            summary["n_steps_excluded_out_of_support"]
        )

    if not split_summaries:
        raise ValueError(f"No held-out prediction artifacts found under {run_dir / 'predictions'}")

    _log_metrics_to_mlflow(str(run_payload["run_key"]), mlflow_metrics)
    return {
        "run_key": str(run_payload["run_key"]),
        "run_dir": str(run_dir),
        "splits": split_summaries,
    }


def evaluate_run(run_key: str) -> dict[str, object]:
    return evaluate_run_dir(
        paths.repo_root() / "results" / "bench_v2" / "runs" / str(run_key)
    )
