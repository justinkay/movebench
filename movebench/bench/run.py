from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import signal
from pathlib import Path

import movebench.bench.artifacts as bench_artifacts
import movebench.bench.config as bench_config
import movebench.bench.evaluation.runner as evaluation_runner
import movebench.bench.planning as bench_planning
import movebench.bench.report as bench_report
import movebench.bench.run_state as bench_run_state
import movebench.bench.summary as bench_summary
import movebench.paths as paths
from movebench.training import RunTracker

_SUPPORTED_METHODS = {
    str(method_key)
    for method_key, method_cfg in bench_config.METHODS.items()
    if bool(method_cfg.get("executor"))
}


class _RunInterrupted(BaseException):
    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def _log_progress(message: str) -> None:
    print(str(message), flush=True)


def _load_executor(executor_ref: str):
    module_name, func_name = str(executor_ref).split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


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


def _interrupt_reason(exc: BaseException) -> str:
    if isinstance(exc, _RunInterrupted):
        return str(exc.reason)
    return exc.__class__.__name__


def _install_interrupt_handlers() -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handle_signal(signum, frame) -> None:
        del frame
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{int(signum)}"
        raise _RunInterrupted(name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError, RuntimeError):
            continue
    return previous


def _restore_interrupt_handlers(previous: dict[int, object]) -> None:
    for signum, handler in previous.items():
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError, RuntimeError):
            continue


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


def _clear_runtime_outputs(run_dir: Path) -> None:
    bench_run_state.remove_active_run(run_dir)
    for path in (
        run_dir / "training_summary.json",
        run_dir / "history.json",
        run_dir / "checkpoint_best.pt",
    ):
        if path.exists():
            path.unlink()
    predictions_dir = run_dir / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)
    final_metrics_dir = run_dir / "final_metrics"
    if final_metrics_dir.exists():
        shutil.rmtree(final_metrics_dir)


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

    if resume_run:
        training_summary_path = run_dir / "training_summary.json"
        if training_summary_path.exists():
            training_summary_path.unlink()
        final_metrics_dir = run_dir / "final_metrics"
        if final_metrics_dir.exists():
            shutil.rmtree(final_metrics_dir)
    else:
        _clear_runtime_outputs(run_dir)
    _write_json(resolved_run_path, canonical_payload)

    if str(run_payload["method"]) not in _SUPPORTED_METHODS:
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=f"executor not implemented for method {run_payload['method']}",
        )

    resolved = bench_artifacts._resolve_artifacts_for_run(run_payload)
    if resolved["status"] != "resolved":
        return _write_blocked_run(
            run_payload,
            run_dir=run_dir,
            reason=str(resolved["reason"]),
        )

    _log_progress(f"{run_payload['run_key']}: starting executor {run_payload['method']}")
    tracker = RunTracker(run_dir)
    bench_run_state.write_active_run(run_dir)
    previous_handlers = _install_interrupt_handlers()
    try:
        try:
            executor = _load_executor(str(run_payload["executor"]))
            executor_kwargs = {}
            if "shared_checkpoint_path" in resolved:
                executor_kwargs["restore_from"] = {
                    "checkpoint_path": resolved["shared_checkpoint_path"],
                    "training_run_key": str(run_payload["shared_checkpoint_run_key"]),
                    "training_signature": str(run_payload["shared_training_signature"]),
                }
            summary = dict(
                executor(
                    run_payload=run_payload,
                    resolved_artifacts=resolved,
                    run_dir=run_dir,
                    tracker=tracker,
                    **executor_kwargs,
                )
            )
            summary.setdefault("status", "completed")
            if str(summary["status"]) == "completed":
                _write_json(run_dir / "training_summary.json", summary)
                if bool(run_payload.get("evaluates", True)):
                    _log_progress(f"{run_payload['run_key']}: starting evaluation")
                    evaluation_runner.evaluate_run_dir(run_dir)
                    _log_progress(f"{run_payload['run_key']}: finished evaluation")
            _write_json(run_dir / "training_summary.json", summary)
            return summary
        except (KeyboardInterrupt, _RunInterrupted) as exc:
            reason = _interrupt_reason(exc)
            _log_progress(f"{run_payload['run_key']}: interrupted: {reason}")
            summary = _interrupted_summary(run_payload, run_dir=run_dir, reason=reason)
            _write_json(run_dir / "training_summary.json", summary)
            raise
        except Exception as exc:
            _log_progress(f"{run_payload['run_key']}: failed: {exc}")
            summary = {
                "status": "failed",
                "reason": str(exc),
                "train_steps": 0,
                "early_stop_split_key": str(run_payload["early_stop_split_key"]),
            }
            _write_json(run_dir / "training_summary.json", summary)
            return summary
    finally:
        _restore_interrupt_handlers(previous_handlers)
        bench_run_state.remove_active_run(run_dir)


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
    executor_blocked_runs = 0
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
            executor_blocked_runs += 1
        elif status == "failed":
            failed_runs += 1
        _log_progress(
            f"[{run_index}/{total_pending_runs}] finished {run_payload['run_key']} status={status or 'completed'}"
        )

    return {
        "planned_runs": int(resolved["counts"]["planned_runs"]),
        "executed_runs": executed_runs,
        "skipped_runs": skipped_runs,
        "blocked_runs": planner_blocked_runs + executor_blocked_runs,
        "failed_runs": failed_runs,
    }


def _render_status_text(snapshot: dict[str, object]) -> str:
    counts = snapshot["counts"]
    lines = [
        f"Studies: {counts['studies']}",
        f"Blocked studies: {counts['blocked_studies']}",
        f"SSF-ready pairs: {counts['ssf_ready_pairs']}",
        f"deepSSF-cache-ready pairs: {counts['raster_ready_pairs']}",
        f"Fully runnable pairs: {counts['fully_runnable_pairs']}",
        f"Blocked pairs: {counts['blocked_pairs']}",
        f"Planned runs: {counts['planned_runs']}",
        f"Completed runs: {counts['completed_runs']}",
    ]
    for study in snapshot["studies"]:
        for readiness in study["timescales"]:
            parts: list[str] = []
            if readiness["point_ready"]:
                parts.append("ssf-ready")
            if readiness["raster_ready"]:
                parts.append("deepSSF-cache-ready")
            if readiness["fully_runnable"]:
                parts.append("fully-runnable")
            else:
                parts.append("blocked")
            if readiness["blocked_reasons"]:
                parts.append(f"reasons={'; '.join(readiness['blocked_reasons'])}")
            lines.append(
                f"{study['study_id']} / {readiness['timescale']}: {', '.join(parts)}"
            )
    return "\n".join(lines)


def _status_selection_label(studies_override: list[str] | None) -> str:
    if not studies_override:
        return "all"
    return " ".join(str(value) for value in studies_override)


def _command_plan(args: argparse.Namespace) -> int:
    experiment = bench_planning.load_experiment(args.experiment_module)
    runs = bench_planning.expand_experiment(experiment, studies_override=args.studies)
    planned_runs = int(len(runs) + len(experiment["blocked"]))
    existing_runs = len(
        bench_run_state.search_latest_run_records_by_key(
            [str(run["run_key"]) for run in runs]
        )
    )
    print(f"Planned runs: {planned_runs}")
    print(f"Existing runs: {existing_runs}")
    print(f"Blocked runs: {len(experiment['blocked'])}")
    print(f"Runnable runs: {len(runs)}")
    return 0


def _command_status(args: argparse.Namespace) -> int:
    if args.format == "text":
        selections = [list(args.studies)] if args.studies else [["all"], ["mini"]]
        sections = []
        for studies_override in selections:
            snapshot = bench_planning.status_snapshot(studies_override=studies_override)
            sections.append(
                "\n".join(
                    [
                        f"Study selection: {_status_selection_label(studies_override)}",
                        _render_status_text(snapshot),
                    ]
                )
            )
        print("\n\n".join(sections))
        return 0
    if args.format == "html":
        result = bench_report.write_report(
            status_snapshot=bench_planning.status_snapshot(studies_override=args.studies),
            refresh_metrics=False,
            allow_metric_repair=False,
        )
        print(result["index_path"])
        return 0
    raise ValueError(f"Unsupported status format: {args.format}")


def _command_report(args: argparse.Namespace) -> int:
    result = bench_report.write_report(
        status_snapshot=bench_planning.status_snapshot(),
        refresh_metrics=bool(args.refresh_metrics),
        allow_metric_repair=True,
    )
    print(result["index_path"])
    return 0


def _command_summary(args: argparse.Namespace) -> int:
    experiments_override = bench_summary.normalize_experiment_selection(args.experiments)
    selections = [list(args.studies)] if args.studies else [["all"], ["mini"]]
    sections = []
    for studies_override in selections:
        lines = [f"Study selection: {_status_selection_label(studies_override)}"]
        if experiments_override:
            lines.append(
                "Experiment selection: " + ", ".join(experiments_override)
            )
        lines.append(
            bench_summary.render_current_summary(
                studies_override=studies_override,
                experiments_override=experiments_override,
            )
        )
        sections.append("\n".join(lines))
    print("\n\n".join(sections))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    summary = run_experiment(
        args.experiment_module,
        studies_override=args.studies,
        methods=args.methods,
        max_runs=args.max_runs,
        rerun=bool(args.rerun),
    )
    print(f"Planned runs: {summary['planned_runs']}")
    print(f"Executed runs: {summary['executed_runs']}")
    print(f"Skipped runs: {summary['skipped_runs']}")
    print(f"Blocked runs: {summary['blocked_runs']}")
    print(f"Failed runs: {summary['failed_runs']}")
    return 0


def register(subparsers) -> None:
    parser = subparsers.add_parser("bench", help="Plan and inspect the benchmark runtime")
    bench_subparsers = parser.add_subparsers(dest="bench_command", required=True)

    plan_parser = bench_subparsers.add_parser("plan", help="Plan a benchmark experiment")
    plan_parser.add_argument("experiment_module")
    plan_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    plan_parser.set_defaults(func=_command_plan)

    status_parser = bench_subparsers.add_parser("status", help="Inspect artifact and run readiness")
    status_parser.add_argument("--format", choices=["text", "html"], default="text")
    status_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default text output prints both 'all' and 'mini'.",
    )
    status_parser.set_defaults(func=_command_status)

    report_parser = bench_subparsers.add_parser("report", help="Generate the benchmark report")
    report_parser.add_argument("--refresh-metrics", action="store_true")
    report_parser.set_defaults(func=_command_report)

    summary_parser = bench_subparsers.add_parser(
        "summary",
        help="Print high-level benchmark progress and blocker summaries",
    )
    summary_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default text output prints both 'all' and 'mini'.",
    )
    summary_parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional experiment names like 'main_results', 'covariate_comparison', "
            "'baselines', or 'smoke_one_study'. Aliases: 'main', 'covariates', 'smoke'."
        ),
    )
    summary_parser.set_defaults(func=_command_summary)

    run_parser = bench_subparsers.add_parser("run", help="Execute runnable benchmark runs")
    run_parser.add_argument("experiment_module")
    run_parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    run_parser.add_argument("--methods", nargs="+", default=None)
    run_parser.add_argument("--max-runs", type=int, default=None)
    run_parser.add_argument("--rerun", action="store_true")
    run_parser.set_defaults(func=_command_run)
