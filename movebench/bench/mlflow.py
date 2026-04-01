from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json

import mlflow

import movebench.paths as paths

MLFLOW_EXPERIMENT_NAME = "movebench_bench"
MLFLOW_TRACKING_URI_ENV = "MOVEBENCH_BENCH_MLFLOW_TRACKING_URI"


def _default_db_path() -> Path:
    return paths.repo_root() / "results" / "bench" / "mlflow.db"


def _tracking_uri(*, create: bool) -> str | None:
    env_value = os.environ.get(MLFLOW_TRACKING_URI_ENV)
    if env_value:
        return env_value
    db_path = _default_db_path()
    if create:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    elif not db_path.exists():
        return None
    return f"sqlite:///{db_path.as_posix()}"


def _artifacts_root() -> Path:
    root = paths.repo_root() / "results" / "bench" / "mlartifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _client(*, create: bool):
    tracking_uri = _tracking_uri(create=create)
    if tracking_uri is None:
        return None
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)


def get_or_create_experiment() -> str:
    client = _client(create=True)
    assert client is not None
    existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if existing is not None:
        return str(existing.experiment_id)
    return str(
        client.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=_artifacts_root().as_uri(),
        )
    )


def get_existing_experiment_id() -> str | None:
    client = _client(create=False)
    if client is None:
        return None
    existing = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if existing is None:
        return None
    return str(existing.experiment_id)


def _flatten_model_params(method_model: dict[str, object]) -> dict[str, object]:
    params: dict[str, object] = {}
    for key, value in dict(method_model).items():
        param_key = f"model.{key}"
        if isinstance(value, (str, int, float, bool)):
            params[param_key] = value
            continue
        params[param_key] = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    return params


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
    params.update(_flatten_model_params(dict(run_payload.get("method_model", {}))))
    run = mlflow.start_run(
        experiment_id=experiment_id,
        run_name=str(run_payload["run_key"]),
        tags=tags,
    )
    mlflow.log_params(params)
    return run


def search_all_benchmark_runs(filter_string: str | None = None) -> list[object]:
    experiment_id = get_existing_experiment_id()
    if experiment_id is None:
        return []
    client = _client(create=False)
    assert client is not None
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


def _run_record(run: object) -> dict[str, object]:
    info = getattr(run, "info", None)
    data = getattr(run, "data", None)
    tags = dict(getattr(data, "tags", {}) or {})
    params = dict(getattr(data, "params", {}) or {})
    metrics = {
        str(key): float(value)
        for key, value in dict(getattr(data, "metrics", {}) or {}).items()
    }
    return {
        "run_id": str(getattr(info, "run_id", "") or ""),
        "start_time": int(getattr(info, "start_time", 0) or 0),
        "end_time": int(getattr(info, "end_time", 0) or 0),
        "status": str(tags.get("status", "")),
        "tags": {str(key): str(value) for key, value in tags.items()},
        "params": {str(key): str(value) for key, value in params.items()},
        "metrics": metrics,
    }


def search_latest_run_records_by_key(
    run_keys: list[str],
) -> dict[str, dict[str, object]]:
    return {
        str(run_key): _run_record(run)
        for run_key, run in search_latest_runs_by_key(run_keys).items()
    }
