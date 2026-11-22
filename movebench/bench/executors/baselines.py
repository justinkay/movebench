from __future__ import annotations

from pathlib import Path

import torch

import movebench.ssf_grid as ssf_grid
from movebench.datasets.chips import grid_dataset
from movebench.datasets.points import point_context, training_steps
from movebench.geometry import LOCAL_GEOMETRY


def _write_predictions(
    run_dir: Path,
    *,
    split_key: str,
    payload: dict[str, torch.Tensor | str],
) -> None:
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**payload, "geometry": LOCAL_GEOMETRY},
        pred_dir / f"{split_key}.pt",
    )


def run_nomove(*, run_payload, resolved_artifacts, run_dir, tracker):
    for split_key in run_payload["evaluation_split_keys"]:
        ctx = point_context(resolved_artifacts["splits"][split_key])
        zeros = torch.zeros_like(ctx["start_bearing_rad"])
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
    for split_key in run_payload["evaluation_split_keys"]:
        ctx = point_context(resolved_artifacts["splits"][split_key])
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
    splits = resolved_artifacts["splits"]
    train = splits["train"]
    _support, train_steps = training_steps(train)
    train_manifest = train.inputs["point_stores"]["train"]["manifest"]
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams.get("batch_size", 256))
    num_workers = int(method_hparams.get("num_workers", 0) or 0)
    for split_key in run_payload["evaluation_split_keys"]:
        grid = grid_dataset(splits[split_key], plan=train.inputs["plan"], include_env=False)
        _write_predictions(
            Path(run_dir),
            split_key=str(split_key),
            payload=ssf_grid.build_ssf_grid_artifact(
                grid_dataset=grid,
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
