from __future__ import annotations

import argparse

import movebench.bench.planning as bench_planning
import movebench.bench.report as bench_report
import movebench.bench.run as bench_run
import movebench.bench.run_state as bench_run_state
import movebench.bench.summary as bench_summary


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
    summary = bench_run.run_experiment(
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
    return 2 if summary["failed_runs"] else 0


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
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=_command_run)


def add_run_arguments(parser) -> None:
    parser.add_argument("experiment_module")
    parser.add_argument(
        "--studies",
        nargs="+",
        default=None,
        help="Optional study ids or named sets like 'mini'. Default: use the experiment's studies value.",
    )
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--rerun", action="store_true")
