from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow

import movebench.paths as paths

MLFLOW_EXPERIMENT_NAME = "movebench_bench_v2"
MLFLOW_TRACKING_URI_ENV = "MOVEBENCH_BENCH_V2_MLFLOW_TRACKING_URI"


def _tracking_uri() -> str:
    env_value = os.environ.get(MLFLOW_TRACKING_URI_ENV)
    if env_value:
        return env_value
    db_path = paths.repo_root() / "results" / "bench_v2" / "mlflow.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.as_posix()}"


def _artifacts_root() -> Path:
    root = paths.repo_root() / "results" / "bench_v2" / "mlartifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _client():
    tracking_uri = _tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def get_or_create_experiment() -> str:
    client = _client()
    existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if existing is not None:
        return str(existing.experiment_id)
    return str(
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=_artifacts_root().as_uri(),
        )
    )


def start_benchmark_run(run_payload: dict[str, object]) -> object:
    experiment_id = get_or_create_experiment()
    tags = {
        "run_key": str(run_payload["run_key"]),
        "study_id": str(run_payload["study_id"]),
        "timescale": str(run_payload["timescale"]),
        "covariate_set": str(run_payload["covariate_set"]),
        "method": str(run_payload["method"]),
        "early_stop_split": str(run_payload["early_stop_split_key"]),
        "status": str(run_payload.get("status", "running")),
    }
    params: dict[str, object] = {
        "step_time_s": int(run_payload["step_time_s"]),
        "seed": int(run_payload["seed"]),
    }
    params.update(dict(run_payload.get("method_hparams", {})))
    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=str(run_payload["run_key"]),
        tags=tags,
    )
    mlflow.log_params(params)
    return run


def search_all_benchmark_runs(filter_string: str | None = None) -> list[object]:
    client = _client()
    experiment_id = get_or_create_experiment()
    runs = client.search_runs(
        [experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=50000,
    )
    return list(runs)


def search_latest_runs_by_key(run_keys: list[str]) -> dict[str, object]:
    wanted = {str(run_key) for run_key in run_keys}
    if not wanted:
        return {}
    runs = sorted(
        search_all_benchmark_runs(),
        key=lambda run: int(getattr(run.info, "start_time", 0) or 0),
        reverse=True,
    )
    latest: dict[str, object] = {}
    for run in runs:
        run_key = str(getattr(run.data, "tags", {}).get("run_key", "")).strip()
        if not run_key or run_key not in wanted or run_key in latest:
            continue
        latest[run_key] = run
    return latest
