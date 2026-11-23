from __future__ import annotations

import hashlib
import importlib
import json
import shutil
from pathlib import Path

import torch

import movebench.bench.artifacts as bench_artifacts
import movebench.bench.config as bench_config
import movebench.bench.evaluation.runner as evaluation_runner
import movebench.bench.planning as bench_planning
import movebench.bench.run_state as bench_run_state
import movebench.paths as paths
from movebench.training import RestorationReference, seed_everything


def _log_progress(message: str) -> None:
    print(str(message), flush=True)


def _load_implementation(reference: str):
    module_name, name = reference.split(":", 1)
    return getattr(importlib.import_module(module_name), name)


def _predict_and_score(method, data, *, run_payload, run_dir):
    split_key = data.split_key
    pred_path = run_dir / "predictions" / f"{split_key}.pt"
    payload = None
    scored = None
    if pred_path.exists():
        try:
            payload = dict(torch.load(pred_path, map_location="cpu"))
            scored = evaluation_runner._score_split(
                split_key=split_key,
                pred_artifact=payload,
                run_payload=run_payload,
            )
        except Exception:
            payload = None
    if scored is None:
        payload = method.predict(data)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, pred_path)
        scored = evaluation_runner._score_split(
            split_key=split_key,
            pred_artifact=payload,
            run_payload=run_payload,
        )
    evaluation_runner._write_split_metrics(run_dir, split_key, *scored)


def _write_json(path: Path, payload: dict[str, object] | list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _history_steps_completed(run_dir: Path) -> int:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return 0
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    steps = [
        int(float(entry["step"]))
        for entry in history_payload
        if isinstance(entry, dict) and "step" in entry
    ]
    return max(steps) if steps else 0


def _interrupted_summary(
    run_payload: dict[str, object],
    *,
    run_dir: Path,
    reason: str,
) -> dict[str, object]:
    return {
        "status": "interrupted",
        "reason": str(reason),
        "train_steps": _history_steps_completed(run_dir),
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
    }


def _manifest_slug(name: str) -> str:
    slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
        for ch in str(name).strip()
    ).strip("_")
    return slug or "experiment"


def _resolved_manifest_payload(
    *,
    experiment: dict[str, object],
    planned_runs: list[dict[str, object]],
    pending_runs: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "experiment": dict(experiment),
        "planned_runs": [dict(run_payload) for run_payload in planned_runs],
        "planner_blocked_runs": [dict(run) for run in experiment.get("blocked", [])],
        "pending_run_keys": [str(run["run_key"]) for run in pending_runs],
    }


def _resolved_manifest_path(
    *,
    experiment: dict[str, object],
    manifest_payload: dict[str, object],
) -> Path:
    payload_hash = hashlib.sha256(
        bench_config.stable_payload(manifest_payload).encode("utf-8")
    ).hexdigest()[:16]
    return (
        paths.repo_root()
        / "results"
        / "bench"
        / "manifests"
        / f"{_manifest_slug(str(experiment['name']))}__{payload_hash}.json"
    )


def _clear_prediction_outputs(run_dir: Path) -> None:
    summary_path = run_dir / "training_summary.json"
    if summary_path.exists():
        summary_path.unlink()
    predictions_dir = run_dir / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
    final_metrics_dir = run_dir / "final_metrics"
    if final_metrics_dir.exists():
        shutil.rmtree(final_metrics_dir)


def _clear_fitted_state(run_dir: Path) -> None:
    bench_run_state.remove_active_run(run_dir)
    for path in (run_dir / "history.json", run_dir / "checkpoint_best.pt"):
        if path.exists():
            path.unlink()


def _clear_shared_dependents(training_run_key: str) -> None:
    runs_root = bench_run_state.run_dir(training_run_key).parent
    dependents = [
        path.parent
        for path in sorted(runs_root.glob("*/resolved_run.json"))
        if _load_json(path).get("shared_checkpoint_run_key") == training_run_key
    ]
    active = [path.name for path in dependents if bench_run_state.active_run_is_live(path)]
    if active:
        raise RuntimeError(
            f"Cannot replace shared trainer {training_run_key}: active scorers {active}. "
            "Retry training after scoring finishes."
        )
    for path in dependents:
        _clear_prediction_outputs(path)


def _write_blocked_run(
    run_payload: dict[str, object],
    *,
    run_dir: Path,
    reason: str,
) -> dict[str, object]:
    _log_progress(f"{run_payload['run_key']}: blocked: {reason}")
    summary = {
        "status": "blocked",
        "reason": reason,
        "train_steps": 0,
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
    }
    _write_json(run_dir / "training_summary.json", summary)
    return summary


def _execute_run(
    run_payload: dict[str, object],
    *,
    resume: bool = True,
) -> dict[str, object]:
    """
    Note this deletes existing artifacts! It should only be called when you *really* want to execute the run;
    that's why it is a private method.
    """
    run_dir = bench_run_state.run_dir(str(run_payload["run_key"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    canonical_payload = bench_planning._canonical_run_payload(run_payload)
    resume_run = False
    resolved_run_path = run_dir / "resolved_run.json"
    if bool(resume) and resolved_run_path.exists():
        try:
            resume_run = (
                bench_config.stable_payload(bench_planning._canonical_run_payload(_load_json(resolved_run_path)))
                == bench_config.stable_payload(canonical_payload)
            )
        except Exception:
            resume_run = False

    checkpoint_path = run_dir / "checkpoint_best.pt"
    replacing_shared = bool(run_payload.get("shared_training_run")) and (
        not resume_run or not checkpoint_path.exists()
    )
    if replacing_shared:
        _clear_shared_dependents(str(run_payload["run_key"]))

    if resume_run:
        training_summary_path = run_dir / "training_summary.json"
        if training_summary_path.exists():
            training_summary_path.unlink()
        final_metrics_dir = run_dir / "final_metrics"
        if final_metrics_dir.exists():
            shutil.rmtree(final_metrics_dir)
    else:
        _clear_prediction_outputs(run_dir)
        _clear_fitted_state(run_dir)
    _write_json(resolved_run_path, canonical_payload)

    method_config = bench_config.METHODS.get(str(run_payload["method"]), {})
    if not method_config.get("implementation"):
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=f"implementation not available for method {run_payload['method']}",
        )

    resolved = bench_artifacts._resolve_artifacts_for_run(run_payload)
    if resolved["status"] != "resolved":
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=str(resolved["reason"]),
        )

    restore_from: RestorationReference | None = None
    if "shared_checkpoint_path" in resolved:
        restore_from = {
            "checkpoint_path": resolved["shared_checkpoint_path"],
            "training_run_key": str(run_payload["shared_checkpoint_run_key"]),
            "training_signature": str(run_payload["shared_training_signature"]),
        }
    elif method_config.get("fit_reuse") == "checkpoint":
        if resume_run and checkpoint_path.exists():
            restore_from = {
                "checkpoint_path": checkpoint_path,
                "training_run_key": (
                    str(run_payload["run_key"])
                    if run_payload.get("shared_training_run")
                    else None
                ),
                "training_signature": run_payload.get("shared_training_signature"),
            }
        else:
            if run_payload.get("shared_training_run") and not replacing_shared:
                _clear_shared_dependents(str(run_payload["run_key"]))
            _clear_prediction_outputs(run_dir)
            _clear_fitted_state(run_dir)

    elif method_config.get("fit_reuse") != "rebuild":
        _clear_prediction_outputs(run_dir)

    _log_progress(f"{run_payload['run_key']}: starting method {run_payload['method']}")
    summary = {
        "train_steps": 0,
        "early_stop_split_key": run_payload["early_stop_split_key"],
    }
    with bench_run_state.active_run(run_dir):
        try:
            seed_everything(int(run_payload["seed"]))
            implementation = _load_implementation(str(run_payload["implementation"]))
            method = implementation(
                model_config=run_payload.get("method_model", {}),
                hparams=run_payload.get("method_hparams", {}),
                seed=run_payload["seed"],
            )
            if run_payload.get("shared_training_run"):
                train, validation = resolved["train"], None
            else:
                splits = resolved["splits"]
                train = splits["train"]
                validation = splits.get(run_payload["early_stop_split_key"])
            method.fit(train, validation, work_dir=run_dir, restore_from=restore_from)
            summary.update(dict(getattr(method, "training_summary_", {})))
            summary["fit_completed"] = True
            summary = bench_run_state.write_training_summary(
                run_dir, summary, status="running"
            )
            if run_payload.get("evaluates", True):
                for split_key in run_payload["evaluation_split_keys"]:
                    _predict_and_score(
                        method,
                        splits[split_key],
                        run_payload=run_payload,
                        run_dir=run_dir,
                    )
            error = bench_run_state.completion_error(run_dir, run_payload, summary)
            if error:
                raise ValueError(error)
            return bench_run_state.write_training_summary(
                run_dir, summary, status="completed"
            )
        except (KeyboardInterrupt, bench_run_state.RunInterrupted) as exc:
            reason = bench_run_state.interrupt_reason(exc)
            _log_progress(f"{run_payload['run_key']}: interrupted: {reason}")
            if not summary.get("fit_completed"):
                summary = _interrupted_summary(
                    run_payload, run_dir=run_dir, reason=reason
                )
            bench_run_state.write_training_summary(
                run_dir, summary, status="interrupted", reason=reason
            )
            raise
        except Exception as exc:
            _log_progress(f"{run_payload['run_key']}: failed: {exc}")
            return bench_run_state.write_training_summary(
                run_dir, summary, status="failed", reason=str(exc)
            )


def execute_run_payload(
    run_payload: dict[str, object],
    *,
    resume: bool = True,
) -> dict[str, object]:
    return _execute_run(dict(run_payload), resume=resume)


def resolve_pending_runs(
    experiment_module: str,
    *,
    studies_override: list[str] | None = None,
    methods: list[str] | None = None,
    max_runs: int | None = None,
    rerun: bool = False,
) -> dict[str, object]:
    print("Loading experiment")
    experiment = bench_planning.load_experiment(experiment_module)
    if methods is not None:
        experiment["methods"] = [str(method) for method in methods]

    print("Expanding experiment")
    planned_runs = bench_planning.expand_experiment(experiment, studies_override=studies_override)
    planner_blocked_runs = [dict(run) for run in experiment.get("blocked", [])]

    print("Loading latest runs")
    latest_runs = bench_run_state.search_latest_run_records_by_key(
        [str(run["run_key"]) for run in planned_runs]
    )

    print("Prepping pending runs")
    pending_runs: list[dict[str, object]] = []
    skipped_runs = 0
    for run_payload in planned_runs:
        run_key = str(run_payload["run_key"])
        latest_run = dict(latest_runs.get(run_key, {}))
        latest_status = str(latest_run.get("status", ""))
        if not bool(rerun):
            if latest_status == "completed":
                skipped_runs += 1
                continue
            if latest_status == "running":
                skipped_runs += 1
                continue
            failure_reason = str(dict(latest_run.get("tags", {})).get("failure_reason", ""))
            if failure_reason == "stale interrupted run":
                bench_run_state.remove_active_run(bench_run_state.run_dir(run_key))
        pending_runs.append(dict(run_payload))

    if max_runs is not None:
        pending_runs = pending_runs[: int(max_runs)]

    manifest_payload = _resolved_manifest_payload(
        experiment=experiment,
        planned_runs=planned_runs,
        pending_runs=pending_runs,
    )
    manifest_path = _resolved_manifest_path(
        experiment=experiment,
        manifest_payload=manifest_payload,
    )

    return {
        "experiment": experiment,
        "planned_runs": [dict(run_payload) for run_payload in planned_runs],
        "pending_runs": pending_runs,
        "planner_blocked_runs": planner_blocked_runs,
        "latest_runs": latest_runs,
        "manifest_payload": manifest_payload,
        "manifest_path": manifest_path,
        "counts": {
            "planned_runs": len(planned_runs) + len(planner_blocked_runs),
            "pending_runs": len(pending_runs),
            "skipped_runs": skipped_runs,
            "blocked_runs": len(planner_blocked_runs),
        },
    }


def load_resolved_run(manifest_path: str | Path, run_key: str) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    for run_payload in manifest.get("planned_runs", []):
        if str(run_payload.get("run_key")) == str(run_key):
            return dict(run_payload)
    raise KeyError(f"Run key {run_key!r} not found in manifest {manifest_path}")


def run_experiment(
    experiment_module: str,
    *,
    studies_override: list[str] | None = None,
    methods: list[str] | None = None,
    max_runs: int | None = None,
    rerun: bool = False,
) -> dict[str, object]:
    resolved = resolve_pending_runs(
        experiment_module,
        studies_override=studies_override,
        methods=methods,
        max_runs=max_runs,
        rerun=rerun,
    )

    executed_runs = 0
    method_blocked_runs = 0
    failed_runs = 0
    pending_runs = list(resolved["pending_runs"])
    total_pending_runs = len(pending_runs)
    skipped_runs = int(resolved["counts"]["skipped_runs"])
    planner_blocked_runs = int(resolved["counts"]["blocked_runs"])
    if total_pending_runs:
        _log_progress(
            "Executing "
            f"{total_pending_runs} run(s); skipped={skipped_runs}, planner_blocked={planner_blocked_runs}"
        )
    for run_index, run_payload in enumerate(pending_runs, start=1):
        _log_progress(
            f"[{run_index}/{total_pending_runs}] starting {run_payload['run_key']}"
        )
        summary = execute_run_payload(run_payload, resume=not bool(rerun))
        executed_runs += 1
        status = str(summary.get("status", ""))
        if status == "blocked":
            method_blocked_runs += 1
        elif status == "failed":
            failed_runs += 1
        _log_progress(
            f"[{run_index}/{total_pending_runs}] finished {run_payload['run_key']} status={status or 'completed'}"
        )

    return {
        "planned_runs": int(resolved["counts"]["planned_runs"]),
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "blocked_runs": planner_blocked_runs + method_blocked_runs,
        "failed_runs": failed_runs,
    }
