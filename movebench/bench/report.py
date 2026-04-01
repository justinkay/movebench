from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from math import ceil
from pathlib import Path
import shutil
from typing import Any
import pandas as pd

import movebench.paths as paths

import movebench.bench.config as bench_config
import movebench.bench.mlflow as bench_mlflow
import movebench.bench.run as bench_run

EXPECTED_HELD_OUT_SPLITS = ["test_temporal", "test_individual"]
TIMESCALE_ORDER = ["1h", "12h", "1d", "7d"]
METHOD_ORDER = list(bench_config.METHODS.keys())
COVARIATE_ORDER = list(bench_config.COVARIATE_SETS.keys())
METHOD_COLORS = {
    "baseline_nomove": "#6a4c93",
    "baseline_constvel": "#1982c4",
    "baseline_randomwalk": "#4f772d",
    "issf_linear": "#c1121f",
    "issf_mlp": "#f77f00",
    "deepssf": "#003049",
    "moveformer": "#2a9d8f",
}
COVARIATE_COLORS = {
    "all": "#003049",
    "all_environmental": "#c1121f",
    "alphaearth_only": "#2a9d8f",
    "lite": "#f77f00",
}


def _report_root() -> Path:
    return paths.repo_root() / "results" / "bench" / "report"


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


def _empty_results_df() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [],
        columns=[
            "experiment_name",
            "run_key",
            "run_id",
            "run_status",
            "study_id",
            "timescale",
            "method",
            "covariate_set",
            "seed",
            "split_key",
            "energy_score",
        ],
    )


def _ordered(values: list[str], *, preferred: list[str]) -> list[str]:
    order = {value: idx for idx, value in enumerate(preferred)}
    return sorted(
        values,
        key=lambda value: (order.get(value, len(order)), str(value)),
    )


def _method_label(method_key: str) -> str:
    return str(method_key).replace("_", " ")


def _covariate_label(covariate_key: str) -> str:
    return str(covariate_key).replace("_", " ")


def _split_label(split_key: str) -> str:
    return str(split_key).replace("-", " ").replace("_", " ")


def _expanded_checked_in_experiments() -> list[dict[str, object]]:
    return bench_run.expanded_checked_in_experiments()


def _planned_run_index(
    expanded_experiments: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    expanded_experiments = (
        _expanded_checked_in_experiments()
        if expanded_experiments is None
        else expanded_experiments
    )
    planned: dict[str, dict[str, object]] = {}
    for entry in expanded_experiments:
        experiment = dict(entry.get("experiment", {}))
        experiment_name = str(experiment.get("name", ""))
        for run in entry.get("runs", []):
            run_payload = dict(run)
            run_key = str(run_payload["run_key"])
            if run_key not in planned:
                planned[run_key] = {
                    **run_payload,
                    "experiment_names": [experiment_name],
                }
                continue
            if experiment_name not in planned[run_key]["experiment_names"]:
                planned[run_key]["experiment_names"].append(experiment_name)
    return planned


def _planned_runs_for_experiment(
    experiment_name: str,
    *,
    expanded_experiments: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    planned = _planned_run_index(expanded_experiments)
    return {
        run_key: run
        for run_key, run in planned.items()
        if str(experiment_name) in list(run["experiment_names"])
    }


def _sort_results_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    order = {value: idx for idx, value in enumerate(TIMESCALE_ORDER)}
    df = df.copy()
    df["timescale_order"] = df["timescale"].map(
        lambda value: order.get(str(value), len(order))
    )
    df = df.sort_values(
        ["study_id", "split_key", "timescale_order", "method", "covariate_set", "run_key"]
    ).drop(columns=["timescale_order"])
    return df.reset_index(drop=True)


def _result_rows_for_experiment(
    experiment_name: str,
    *,
    latest_run_records: dict[str, dict[str, object]] | None = None,
    expanded_experiments: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    planned_runs = _planned_runs_for_experiment(
        experiment_name,
        expanded_experiments=expanded_experiments,
    )
    if not planned_runs:
        return _empty_results_df()
    if latest_run_records is None:
        latest_run_records = bench_mlflow.search_latest_run_records_by_key(
            list(planned_runs.keys())
        )

    rows: list[dict[str, object]] = []
    for run_key, planned_run in planned_runs.items():
        latest = dict(latest_run_records.get(run_key, {}))
        if str(latest.get("status", "")) != "completed":
            continue
        metrics = dict(latest.get("metrics", {}))
        for split_key in list(planned_run.get("evaluation_split_keys", [])):
            metric_key = f"energy_score.{_metric_split_key(str(split_key))}"
            if metric_key not in metrics:
                continue
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "run_key": run_key,
                    "run_id": str(latest.get("run_id", "")),
                    "run_status": "completed",
                    "study_id": str(planned_run["study_id"]),
                    "timescale": str(planned_run["timescale"]),
                    "method": str(planned_run["method"]),
                    "covariate_set": str(planned_run["covariate_set"]),
                    "seed": int(planned_run.get("seed", 0)),
                    "split_key": str(split_key),
                    "energy_score": float(metrics[metric_key]),
                }
            )
    if not rows:
        return _empty_results_df()
    return _sort_results_df(pd.DataFrame.from_records(rows))


def load_main_results(
    *,
    latest_run_records: dict[str, dict[str, object]] | None = None,
    expanded_experiments: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    return _result_rows_for_experiment(
        "main_results",
        latest_run_records=latest_run_records,
        expanded_experiments=expanded_experiments,
    )


def load_covariate_results(
    *,
    latest_run_records: dict[str, dict[str, object]] | None = None,
    expanded_experiments: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    return _result_rows_for_experiment(
        "covariate_comparison",
        latest_run_records=latest_run_records,
        expanded_experiments=expanded_experiments,
    )


def _study_split_map(status_snapshot: dict[str, object]) -> dict[str, list[str]]:
    study_splits: dict[str, list[str]] = {}
    for study in status_snapshot.get("studies", []):
        seen: list[str] = []
        for readiness in study.get("timescales", []):
            for split_key in readiness.get("held_out_splits", []):
                split_text = str(split_key)
                if split_text in seen:
                    continue
                seen.append(split_text)
        study_splits[str(study["study_id"])] = _ordered(
            seen,
            preferred=EXPECTED_HELD_OUT_SPLITS,
        )
    return study_splits


def _plot_asset_entry(
    *,
    section: str,
    study_id: str,
    split_key: str,
    status: str,
    relative_path: str | None = None,
) -> dict[str, object]:
    return {
        "section": section,
        "study_id": study_id,
        "split_key": split_key,
        "status": status,
        "relative_path": relative_path,
    }


def write_main_results_plots(
    main_results: pd.DataFrame,
    *,
    status_snapshot: dict[str, object],
    report_root: str | Path,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if main_results.empty and "study_id" not in main_results.columns:
        main_results = _empty_results_df()
    report_root = Path(report_root)
    asset_dir = report_root / "assets" / "main_results"
    asset_dir.mkdir(parents=True, exist_ok=True)
    study_splits = _study_split_map(status_snapshot)
    plot_entries: list[dict[str, object]] = []
    x_index = {value: idx for idx, value in enumerate(TIMESCALE_ORDER)}

    for study_id in _ordered(list(study_splits.keys()), preferred=[]):
        held_out_splits = set(study_splits[study_id])
        for split_key in EXPECTED_HELD_OUT_SPLITS:
            if split_key not in held_out_splits:
                plot_entries.append(
                    _plot_asset_entry(
                        section="main_results",
                        study_id=study_id,
                        split_key=split_key,
                        status="not_applicable",
                    )
                )
                continue

            subset = main_results[
                (main_results["study_id"] == study_id)
                & (main_results["split_key"] == split_key)
            ]
            if subset.empty:
                plot_entries.append(
                    _plot_asset_entry(
                        section="main_results",
                        study_id=study_id,
                        split_key=split_key,
                        status="no_data",
                    )
                )
                continue

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            methods = _ordered(
                [str(value) for value in subset["method"].drop_duplicates().tolist()],
                preferred=METHOD_ORDER,
            )
            for method_key in methods:
                method_rows = subset[subset["method"] == method_key].copy()
                method_rows["x"] = method_rows["timescale"].map(
                    lambda value: x_index.get(str(value), len(x_index))
                )
                method_rows = method_rows.sort_values("x")
                ax.plot(
                    method_rows["x"].tolist(),
                    method_rows["energy_score"].astype(float).tolist(),
                    marker="o",
                    linewidth=2,
                    color=METHOD_COLORS.get(method_key),
                    label=_method_label(method_key),
                )
            ax.set_title(f"{study_id} | {_split_label(split_key)}")
            ax.set_xlabel("Timescale")
            ax.set_ylabel("Mean energy score")
            ax.set_xticks(list(range(len(TIMESCALE_ORDER))), TIMESCALE_ORDER)
            ax.grid(True, linewidth=0.4, alpha=0.4)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()

            relative_path = f"assets/main_results/{study_id}__{split_key}.png"
            fig.savefig(report_root / relative_path, dpi=160)
            plt.close(fig)
            plot_entries.append(
                _plot_asset_entry(
                    section="main_results",
                    study_id=study_id,
                    split_key=split_key,
                    status="plotted",
                    relative_path=relative_path,
                )
            )

    return plot_entries


def write_covariate_comparison_plots(
    covariate_results: pd.DataFrame,
    *,
    status_snapshot: dict[str, object],
    report_root: str | Path,
) -> list[dict[str, object]]:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if covariate_results.empty and "study_id" not in covariate_results.columns:
        covariate_results = _empty_results_df()
    report_root = Path(report_root)
    asset_dir = report_root / "assets" / "covariate_comparison"
    asset_dir.mkdir(parents=True, exist_ok=True)
    study_splits = _study_split_map(status_snapshot)
    plot_entries: list[dict[str, object]] = []
    x_index = {value: idx for idx, value in enumerate(TIMESCALE_ORDER)}

    for study_id in _ordered(list(study_splits.keys()), preferred=[]):
        held_out_splits = set(study_splits[study_id])
        for split_key in EXPECTED_HELD_OUT_SPLITS:
            if split_key not in held_out_splits:
                plot_entries.append(
                    _plot_asset_entry(
                        section="covariate_comparison",
                        study_id=study_id,
                        split_key=split_key,
                        status="not_applicable",
                    )
                )
                continue

            subset = covariate_results[
                (covariate_results["study_id"] == study_id)
                & (covariate_results["split_key"] == split_key)
            ]
            if subset.empty:
                plot_entries.append(
                    _plot_asset_entry(
                        section="covariate_comparison",
                        study_id=study_id,
                        split_key=split_key,
                        status="no_data",
                    )
                )
                continue

            methods = _ordered(
                [str(value) for value in subset["method"].drop_duplicates().tolist()],
                preferred=METHOD_ORDER,
            )
            ncols = 2 if len(methods) > 1 else 1
            nrows = int(ceil(len(methods) / float(ncols)))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(8.5, max(4.8, 3.6 * nrows)),
                squeeze=False,
            )
            flat_axes = list(axes.reshape(-1))
            for ax, method_key in zip(flat_axes, methods):
                method_rows = subset[subset["method"] == method_key]
                covariates = _ordered(
                    [
                        str(value)
                        for value in method_rows["covariate_set"].drop_duplicates().tolist()
                    ],
                    preferred=COVARIATE_ORDER,
                )
                for covariate_key in covariates:
                    covariate_rows = method_rows[
                        method_rows["covariate_set"] == covariate_key
                    ].copy()
                    covariate_rows["x"] = covariate_rows["timescale"].map(
                        lambda value: x_index.get(str(value), len(x_index))
                    )
                    covariate_rows = covariate_rows.sort_values("x")
                    ax.plot(
                        covariate_rows["x"].tolist(),
                        covariate_rows["energy_score"].astype(float).tolist(),
                        marker="o",
                        linewidth=2,
                        color=COVARIATE_COLORS.get(covariate_key),
                        label=_covariate_label(covariate_key),
                    )
                ax.set_title(_method_label(method_key))
                ax.set_xticks(list(range(len(TIMESCALE_ORDER))), TIMESCALE_ORDER)
                ax.set_ylabel("Mean energy score")
                ax.grid(True, linewidth=0.4, alpha=0.4)
                ax.legend(frameon=False, fontsize=8)
            for ax in flat_axes[len(methods) :]:
                ax.axis("off")
            fig.suptitle(f"{study_id} | {_split_label(split_key)}", y=1.02)
            fig.tight_layout()

            relative_path = f"assets/covariate_comparison/{study_id}__{split_key}.png"
            fig.savefig(report_root / relative_path, dpi=160)
            plt.close(fig)
            plot_entries.append(
                _plot_asset_entry(
                    section="covariate_comparison",
                    study_id=study_id,
                    split_key=split_key,
                    status="plotted",
                    relative_path=relative_path,
                )
            )

    return plot_entries


def _final_summary_path(run_key: str, split_key: str) -> Path:
    return (
        paths.repo_root()
        / "results"
        / "bench"
        / "runs"
        / str(run_key)
        / "final_metrics"
        / str(split_key)
        / "summary.json"
    )


def _final_summaries(
    run_key: str,
    split_keys: list[str],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    summaries: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for split_key in split_keys:
        path = _final_summary_path(run_key, split_key)
        if not path.exists():
            missing.append(str(split_key))
            continue
        summaries[str(split_key)] = _load_json(path)
    return summaries, missing


def _missing_mlflow_metric_splits(
    latest_run_record: dict[str, object],
    split_keys: list[str],
) -> list[str]:
    metrics = dict(latest_run_record.get("metrics", {}))
    missing: list[str] = []
    for split_key in split_keys:
        metric_key = f"energy_score.{_metric_split_key(str(split_key))}"
        if metric_key not in metrics:
            missing.append(str(split_key))
    return missing


def _planner_blocked_runs(
    expanded_experiments: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    expanded_experiments = (
        _expanded_checked_in_experiments()
        if expanded_experiments is None
        else expanded_experiments
    )
    blocked: list[dict[str, object]] = []
    seen: set[str] = set()
    for expanded in expanded_experiments:
        experiment = dict(expanded.get("experiment", {}))
        for entry in experiment.get("blocked", []):
            payload = {str(key): value for key, value in dict(entry).items()}
            dedupe_key = json.dumps(payload, sort_keys=True)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            blocked.append(payload)
    blocked.sort(
        key=lambda entry: (
            str(entry.get("study_id", "")),
            str(entry.get("timescale", "")),
            str(entry.get("covariate_set", "")),
            str(entry.get("method", "")),
            str(entry.get("reason", "")),
        )
    )
    return blocked


def _refresh_final_metrics(
    planned_runs: dict[str, dict[str, object]],
    *,
    latest_run_records: dict[str, dict[str, object]],
    refresh_metrics: bool,
    allow_metric_repair: bool,
) -> tuple[dict[str, dict[str, object]], list[str], list[dict[str, object]]]:
    if not bool(allow_metric_repair):
        return latest_run_records, [], []
    refreshed: list[str] = []
    errors: list[dict[str, object]] = []
    evaluate_run_fn = None
    for run_key, planned_run in planned_runs.items():
        latest = dict(latest_run_records.get(run_key, {}))
        if str(latest.get("status", "")) != "completed":
            continue
        split_keys = [
            str(split_key) for split_key in planned_run.get("evaluation_split_keys", [])
        ]
        _summaries, missing_splits = _final_summaries(
            run_key,
            split_keys,
        )
        missing_mlflow_splits = _missing_mlflow_metric_splits(latest, split_keys)
        if (not refresh_metrics) and not missing_splits and not missing_mlflow_splits:
            continue
        try:
            if evaluate_run_fn is None:
                from movebench.bench.evaluation.runner import (
                    evaluate_run as evaluate_run_fn,
                )
            evaluate_run_fn(run_key)
            refreshed.append(run_key)
        except Exception as exc:
            errors.append(
                {
                    "kind": "final_metrics",
                    "run_key": run_key,
                    "reason": str(exc),
                }
            )
    if refreshed:
        latest_run_records = bench_mlflow.search_latest_run_records_by_key(
            list(planned_runs.keys())
        )
    return latest_run_records, refreshed, errors


def _planned_run_entries(
    planned_runs: dict[str, dict[str, object]],
    *,
    latest_run_records: dict[str, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    entries: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    missing_artifacts: list[dict[str, object]] = []

    for run_key in _ordered(list(planned_runs.keys()), preferred=[]):
        planned_run = planned_runs[run_key]
        latest = dict(latest_run_records.get(run_key, {}))
        status = str(latest.get("status", "") or "pending")
        split_keys = [str(split_key) for split_key in planned_run.get("evaluation_split_keys", [])]
        final_metrics, missing_summary_splits = _final_summaries(run_key, split_keys)
        missing_metric_splits = _missing_mlflow_metric_splits(latest, split_keys)
        if status != "completed":
            missing_summary_splits = []
            missing_metric_splits = []
        missing_splits = _ordered(
            list(dict.fromkeys([*missing_summary_splits, *missing_metric_splits])),
            preferred=EXPECTED_HELD_OUT_SPLITS,
        )
        entry = {
            "run_key": run_key,
            "study_id": str(planned_run["study_id"]),
            "timescale": str(planned_run["timescale"]),
            "method": str(planned_run["method"]),
            "covariate_set": str(planned_run["covariate_set"]),
            "seed": int(planned_run.get("seed", 0)),
            "latest_status": status,
            "run_id": str(latest.get("run_id", "")),
            "experiment_names": list(planned_run.get("experiment_names", [])),
            "final_metrics": final_metrics,
            "missing_final_metrics_splits": missing_splits,
        }
        entries.append(entry)
        if status == "completed":
            completed.append(entry)
            for split_key in missing_summary_splits:
                missing_artifacts.append(
                    {
                        "kind": "final_metrics",
                        "run_key": run_key,
                        "split_key": split_key,
                        "reason": "missing final metric summary",
                    }
                )
            for split_key in missing_metric_splits:
                missing_artifacts.append(
                    {
                        "kind": "mlflow_metric",
                        "run_key": run_key,
                        "split_key": split_key,
                        "reason": "missing MLflow final metric",
                    }
                )

    return entries, completed, missing_artifacts


def _blocked_studies(
    status_snapshot: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blocked_studies: list[dict[str, object]] = []
    missing_artifacts: list[dict[str, object]] = []
    for study in status_snapshot.get("studies", []):
        blocked_timescales: list[dict[str, object]] = []
        reasons: list[str] = []
        for readiness in study.get("timescales", []):
            blocked_reasons = [str(reason) for reason in readiness.get("blocked_reasons", [])]
            if not blocked_reasons:
                continue
            blocked_timescales.append(
                {
                    "timescale": str(readiness["timescale"]),
                    "reasons": blocked_reasons,
                }
            )
            for reason in blocked_reasons:
                if reason not in reasons:
                    reasons.append(reason)
                missing_artifacts.append(
                    {
                        "kind": "study_prerequisite",
                        "study_id": str(study["study_id"]),
                        "timescale": str(readiness["timescale"]),
                        "reason": reason,
                    }
                )
        if blocked_timescales:
            blocked_studies.append(
                {
                    "study_id": str(study["study_id"]),
                    "timescales": blocked_timescales,
                    "reasons": reasons,
                }
            )
    return blocked_studies, missing_artifacts


def _render_summary_cards(counts: dict[str, int]) -> str:
    labels = [
        ("Studies", counts["studies"]),
        ("Planned runs", counts["planned_runs"]),
        ("Completed runs", counts["completed_runs"]),
        ("Pending runs", counts["pending_runs"]),
        ("Running runs", counts["running_runs"]),
        ("Failed runs", counts["failed_runs"]),
        ("Blocked runs", counts["blocked_runs"]),
        ("Not applicable cases", counts["not_applicable_cases"]),
    ]
    return "\n".join(
        [
            '<div class="summary-grid">',
            *[
                (
                    '<div class="summary-card">'
                    f"<dt>{html.escape(label)}</dt>"
                    f"<dd>{value}</dd>"
                    "</div>"
                )
                for label, value in labels
            ],
            "</div>",
        ]
    )


def _render_study_readiness(status_snapshot: dict[str, object]) -> str:
    rows = [
        "<table>",
        "<thead><tr><th>Study</th><th>Timescale</th><th>Status</th><th>Details</th></tr></thead>",
        "<tbody>",
    ]
    for study in status_snapshot.get("studies", []):
        for readiness in study.get("timescales", []):
            parts: list[str] = []
            if readiness.get("point_ready"):
                parts.append("SSF-ready")
            if readiness.get("raster_ready"):
                parts.append("deepSSF-cache-ready")
            if readiness.get("fully_runnable"):
                parts.append("fully-runnable")
            else:
                parts.append("blocked")
            detail = "; ".join([str(reason) for reason in readiness.get("blocked_reasons", [])])
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(study['study_id']))}</td>"
                f"<td>{html.escape(str(readiness['timescale']))}</td>"
                f"<td>{html.escape(', '.join(parts))}</td>"
                f"<td>{html.escape(detail)}</td>"
                "</tr>"
            )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _render_completed_runs(completed_runs: list[dict[str, object]]) -> str:
    if not completed_runs:
        return '<p class="muted">No completed runs yet.</p>'
    rows = [
        "<table>",
        "<thead><tr><th>Study</th><th>Timescale</th><th>Method</th><th>Covariates</th><th>Final metrics</th></tr></thead>",
        "<tbody>",
    ]
    for run in completed_runs:
        parts: list[str] = []
        for split_key in _ordered(
            list(run["final_metrics"].keys()),
            preferred=EXPECTED_HELD_OUT_SPLITS,
        ):
            summary = dict(run["final_metrics"][split_key])
            energy_score = summary.get("energy_score")
            if energy_score is None:
                parts.append(f"{split_key}: no scored steps")
            else:
                parts.append(f"{split_key}: {float(energy_score):.3f}")
        if run["missing_final_metrics_splits"]:
            parts.append(
                "missing: " + ", ".join(run["missing_final_metrics_splits"])
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(run['study_id'])}</td>"
            f"<td>{html.escape(run['timescale'])}</td>"
            f"<td>{html.escape(_method_label(run['method']))}</td>"
            f"<td>{html.escape(_covariate_label(run['covariate_set']))}</td>"
            f"<td>{html.escape('; '.join(parts))}</td>"
            "</tr>"
        )
    rows.extend(["</tbody>", "</table>"])
    return "\n".join(rows)


def _render_plot_section(
    title: str,
    entries: list[dict[str, object]],
) -> str:
    sections = [f"<section><h2>{html.escape(title)}</h2>"]
    by_study: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        by_study.setdefault(str(entry["study_id"]), []).append(entry)

    for study_id in _ordered(list(by_study.keys()), preferred=[]):
        sections.append(f"<h3>{html.escape(study_id)}</h3>")
        for entry in _ordered(
            [str(item["split_key"]) for item in by_study[study_id]],
            preferred=EXPECTED_HELD_OUT_SPLITS,
        ):
            item = next(
                row
                for row in by_study[study_id]
                if str(row["split_key"]) == entry
            )
            sections.append(
                f"<article><h4>{html.escape(_split_label(str(item['split_key'])))}</h4>"
            )
            if item["status"] == "plotted":
                sections.append(
                    f'<img src="{html.escape(str(item["relative_path"]))}" alt="{html.escape(study_id)} {html.escape(_split_label(str(item["split_key"])))}">'
                )
            elif item["status"] == "not_applicable":
                sections.append('<p class="muted">Not applicable for this study.</p>')
            else:
                sections.append('<p class="muted">No completed runs with final metrics yet.</p>')
            sections.append("</article>")
    sections.append("</section>")
    return "\n".join(sections)


def _render_html(
    *,
    generated_at: str,
    counts: dict[str, int],
    status_snapshot: dict[str, object],
    completed_runs: list[dict[str, object]],
    main_plots: list[dict[str, object]],
    covariate_plots: list[dict[str, object]],
) -> str:
    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"en\">",
            "<head>",
            "<meta charset=\"utf-8\">",
            "<title>MoveBench Bench Report</title>",
            "<style>",
            "body { font-family: Georgia, 'Times New Roman', serif; margin: 24px auto; max-width: 1200px; line-height: 1.45; color: #222; background: linear-gradient(180deg, #fbf7ef 0%, #ffffff 280px); }",
            "h1, h2, h3, h4 { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; }",
            "section { background: rgba(255, 255, 255, 0.92); border: 1px solid #d7d0c5; padding: 18px 20px; margin: 18px 0; box-shadow: 0 10px 24px rgba(52, 49, 43, 0.06); }",
            ".summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin: 0; }",
            ".summary-card { background: #f4efe5; border: 1px solid #d7d0c5; padding: 12px; }",
            ".summary-card dt { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; color: #6a5f52; }",
            ".summary-card dd { margin: 8px 0 0; font-size: 1.6rem; font-weight: 700; color: #0f3b57; }",
            "table { width: 100%; border-collapse: collapse; }",
            "th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #e2ddd3; vertical-align: top; }",
            "th { font-size: 0.9rem; color: #6a5f52; }",
            "img { width: 100%; border: 1px solid #d7d0c5; background: #fff; }",
            ".muted { color: #6a5f52; }",
            "</style>",
            "</head>",
            "<body>",
            "<header>",
            "<h1>MoveBench Bench Report</h1>",
            f"<p class=\"muted\">Generated {html.escape(generated_at)}</p>",
            "</header>",
            "<section>",
            "<h2>Global Summary</h2>",
            _render_summary_cards(counts),
            "</section>",
            "<section>",
            "<h2>Study Readiness</h2>",
            _render_study_readiness(status_snapshot),
            "</section>",
            "<section>",
            "<h2>Completed Runs</h2>",
            _render_completed_runs(completed_runs),
            "</section>",
            _render_plot_section("Main Results", main_plots),
            _render_plot_section("Covariate Comparison", covariate_plots),
            "</body>",
            "</html>",
        ]
    )


def write_report(
    *,
    status_snapshot: dict[str, object],
    refresh_metrics: bool = False,
    allow_metric_repair: bool = True,
) -> dict[str, object]:
    expanded_experiments = _expanded_checked_in_experiments()
    planned_runs = _planned_run_index(expanded_experiments)
    planner_blocked_runs = _planner_blocked_runs(expanded_experiments)
    latest_run_records = bench_mlflow.search_latest_run_records_by_key(
        list(planned_runs.keys())
    )
    latest_run_records, refreshed, refresh_errors = _refresh_final_metrics(
        planned_runs,
        latest_run_records=latest_run_records,
        refresh_metrics=bool(refresh_metrics),
        allow_metric_repair=bool(allow_metric_repair),
    )

    main_results = load_main_results(
        latest_run_records=latest_run_records,
        expanded_experiments=expanded_experiments,
    )
    covariate_results = load_covariate_results(
        latest_run_records=latest_run_records,
        expanded_experiments=expanded_experiments,
    )

    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    assets_root = report_root / "assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)

    main_plots = write_main_results_plots(
        main_results,
        status_snapshot=status_snapshot,
        report_root=report_root,
    )
    covariate_plots = write_covariate_comparison_plots(
        covariate_results,
        status_snapshot=status_snapshot,
        report_root=report_root,
    )

    blocked_studies, study_missing_artifacts = _blocked_studies(status_snapshot)
    planned_run_entries, completed_runs, run_missing_artifacts = _planned_run_entries(
        planned_runs,
        latest_run_records=latest_run_records,
    )
    counts = {
        "studies": int(status_snapshot.get("counts", {}).get("studies", 0)),
        "planned_runs": int(len(planned_runs) + len(planner_blocked_runs)),
        "completed_runs": int(
            sum(1 for entry in planned_run_entries if entry["latest_status"] == "completed")
        ),
        "pending_runs": int(
            sum(1 for entry in planned_run_entries if entry["latest_status"] == "pending")
        ),
        "running_runs": int(
            sum(1 for entry in planned_run_entries if entry["latest_status"] == "running")
        ),
        "failed_runs": int(
            sum(1 for entry in planned_run_entries if entry["latest_status"] == "failed")
        ),
        "blocked_runs": int(
            len(planner_blocked_runs)
            + sum(1 for entry in planned_run_entries if entry["latest_status"] == "blocked")
        ),
        "not_applicable_cases": int(
            sum(
                1
                for entry in [*main_plots, *covariate_plots]
                if entry["status"] == "not_applicable"
            )
        ),
        "main_results_plots": int(
            sum(1 for entry in main_plots if entry["status"] == "plotted")
        ),
        "covariate_comparison_plots": int(
            sum(1 for entry in covariate_plots if entry["status"] == "plotted")
        ),
        "blocked_studies": int(len(blocked_studies)),
    }
    generated_at = datetime.now(timezone.utc).isoformat()
    status_report = {
        "generated_at": generated_at,
        "counts": counts,
        "studies": list(status_snapshot.get("studies", [])),
        "blocked_studies": blocked_studies,
        "blocked_runs": planner_blocked_runs,
        "planned_runs": planned_run_entries,
        "completed_runs": completed_runs,
        "missing_artifacts": [
            *study_missing_artifacts,
            *run_missing_artifacts,
            *refresh_errors,
        ],
        "refreshed_run_keys": refreshed,
        "plots": {
            "main_results": main_plots,
            "covariate_comparison": covariate_plots,
        },
    }
    status_report_path = report_root / "status_report.json"
    _write_json(status_report_path, status_report)

    html_text = _render_html(
        generated_at=generated_at,
        counts=counts,
        status_snapshot=status_snapshot,
        completed_runs=completed_runs,
        main_plots=main_plots,
        covariate_plots=covariate_plots,
    )
    index_path = report_root / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    return {
        "index_path": str(index_path),
        "status_report_path": str(status_report_path),
    }
