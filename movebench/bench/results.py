from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import movebench.bench.config as bench_config
import movebench.bench.run as bench_run
import movebench.bench.run_state as bench_run_state

TIMESCALE_ORDER = list(bench_config.TIMESCALES.keys())
RESULT_COLUMNS = [
    "experiment_name",
    "experiment_names",
    "run_key",
    "run_id",
    "run_status",
    "run_dir",
    "study_id",
    "timescale",
    "method",
    "method_variant",
    "covariate_set",
    "seed",
    "geometry",
    "split_key",
    "energy_score",
    "accuracy",
    "calibration",
    "n_steps_scored",
    "n_steps_excluded_out_of_support",
]


def empty_results_df() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)


def _metric_split_key(split_key: str) -> str:
    return str(split_key).replace("-", "_")


def expanded_checked_in_experiments() -> list[dict[str, object]]:
    return bench_run.expanded_checked_in_experiments()


def planned_run_index(
    expanded_experiments: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    expanded_experiments = (
        expanded_checked_in_experiments()
        if expanded_experiments is None
        else expanded_experiments
    )
    planned: dict[str, dict[str, object]] = {}
    for entry in expanded_experiments:
        experiment_name = str(dict(entry.get("experiment", {})).get("name", ""))
        for run in entry.get("runs", []):
            run_payload = dict(run)
            run_key = str(run_payload["run_key"])
            if run_key not in planned:
                planned[run_key] = {
                    **run_payload,
                    "experiment_names": [experiment_name],
                }
            elif experiment_name not in planned[run_key]["experiment_names"]:
                planned[run_key]["experiment_names"].append(experiment_name)
    return planned


def _sort_results_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    order = {value: index for index, value in enumerate(TIMESCALE_ORDER)}
    df = df.assign(
        _timescale_order=df["timescale"].map(
            lambda value: order.get(str(value), len(order))
        )
    )
    return (
        df.sort_values(
            [
                "study_id",
                "split_key",
                "_timescale_order",
                "method",
                "method_variant",
                "covariate_set",
                "run_key",
            ]
        )
        .drop(columns="_timescale_order")
        .reset_index(drop=True)
    )


def load_results(
    experiment_names: str | list[str] | tuple[str, ...],
    *,
    experiment_module_path: str | Path | None = None,
    latest_run_records: dict[str, dict[str, object]] | None = None,
    expanded_experiments: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    """Load one row per completed planned run and evaluation split."""
    names = (
        [str(experiment_names)]
        if isinstance(experiment_names, str)
        else [str(name) for name in experiment_names]
    )
    wanted = set(names)
    if expanded_experiments is None and experiment_module_path is not None:
        module_path = str(Path(experiment_module_path).resolve())
        experiment = bench_run.load_experiment(module_path)
        expanded_experiments = [
            {
                "module_path": module_path,
                "experiment": experiment,
                "runs": bench_run.expand_experiment(experiment),
            }
        ]
    elif expanded_experiments is not None and experiment_module_path is not None:
        module_path = Path(experiment_module_path).resolve()
        expanded_experiments = [
            entry
            for entry in expanded_experiments
            if Path(str(entry["module_path"])).resolve() == module_path
        ]
    planned_runs = {
        run_key: run
        for run_key, run in planned_run_index(expanded_experiments).items()
        if wanted.intersection(run["experiment_names"])
    }
    if not planned_runs:
        return empty_results_df()
    if latest_run_records is None:
        latest_run_records = bench_run_state.search_latest_run_records_by_key(
            list(planned_runs)
        )

    rows: list[dict[str, object]] = []
    for run_key, planned_run in planned_runs.items():
        latest = dict(latest_run_records.get(run_key, {}))
        if str(latest.get("status", "")) != "completed":
            continue
        metrics = dict(latest.get("metrics", {}))
        experiment_names = tuple(
            str(name) for name in planned_run["experiment_names"]
        )
        matched_experiments = tuple(
            name for name in names if name in planned_run["experiment_names"]
        )
        for split_key in planned_run.get("evaluation_split_keys", []):
            split_key = str(split_key)
            metric_split = _metric_split_key(split_key)
            energy_score = metrics.get(f"energy_score.{metric_split}")
            if energy_score is None:
                continue
            accuracy = metrics.get(f"energy_score.accuracy.{metric_split}")
            calibration = metrics.get(f"energy_score.calibration.{metric_split}")
            n_steps_scored = metrics.get(f"n_steps_scored.{metric_split}")
            n_steps_excluded = metrics.get(
                f"n_steps_excluded_out_of_support.{metric_split}"
            )
            rows.append(
                {
                    "experiment_name": matched_experiments[0],
                    "experiment_names": experiment_names,
                    "run_key": run_key,
                    "run_id": str(latest.get("run_id", "")),
                    "run_status": "completed",
                    "run_dir": str(bench_run_state.run_dir(run_key)),
                    "study_id": str(planned_run["study_id"]),
                    "timescale": str(planned_run["timescale"]),
                    "method": str(planned_run["method"]),
                    "method_variant": str(planned_run.get("method_variant", "")),
                    "covariate_set": str(planned_run["covariate_set"]),
                    "seed": int(planned_run.get("seed", 0)),
                    "geometry": str(planned_run.get("geometry", "")),
                    "split_key": split_key,
                    "energy_score": float(energy_score),
                    "accuracy": np.nan if accuracy is None else float(accuracy),
                    "calibration": (
                        np.nan if calibration is None else float(calibration)
                    ),
                    "n_steps_scored": (
                        np.nan if n_steps_scored is None else float(n_steps_scored)
                    ),
                    "n_steps_excluded_out_of_support": (
                        np.nan if n_steps_excluded is None else float(n_steps_excluded)
                    ),
                }
            )
    if not rows:
        return empty_results_df()
    return _sort_results_df(pd.DataFrame.from_records(rows, columns=RESULT_COLUMNS))


def normalize_results(
    results: pd.DataFrame,
    *,
    baseline_method: str,
    by: list[str],
    value: str = "energy_score",
) -> pd.DataFrame:
    """Normalize each non-baseline row by its matching baseline value."""
    baseline_value = f"baseline_{value}"
    normalized_value = f"normalized_{value}"
    baseline = results.loc[
        results["method"] == baseline_method,
        [*by, value],
    ].rename(columns={value: baseline_value})
    normalized = results[results["method"] != baseline_method].merge(
        baseline,
        on=by,
        how="inner",
        validate="many_to_one",
    )
    normalized = normalized[normalized[baseline_value] > 0].copy()
    normalized[normalized_value] = normalized[value] / normalized[baseline_value]
    return normalized


def aggregate_results(
    results: pd.DataFrame,
    *,
    by: list[str],
    value: str = "normalized_energy_score",
) -> pd.DataFrame:
    """Compute equal-study summary statistics for the requested groups."""
    study_by = list(dict.fromkeys([*by, "study_id"]))
    if results.duplicated(study_by).any():
        raise ValueError(f"Multiple result rows for {study_by}")
    mean_col = f"{value}_mean"
    std_col = f"{value}_std"
    sem_col = f"{value}_sem"
    aggregate = results.groupby(by, as_index=False).agg(
        **{
            mean_col: (value, "mean"),
            std_col: (value, "std"),
            "n_studies": ("study_id", "nunique"),
        }
    )
    aggregate[sem_col] = aggregate[std_col].fillna(0.0) / np.sqrt(
        aggregate["n_studies"]
    )
    return aggregate
