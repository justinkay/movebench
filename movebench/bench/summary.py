from __future__ import annotations

from collections import Counter, defaultdict
import json
import re

import movebench.bench.run_state as bench_run_state

STATUS_KEYS = (
    "completed",
    "not_started",
    "running",
    "failed",
    "blocked",
    "status_unknown",
)
KNOWN_RUN_STATUSES = {"completed", "running", "failed", "blocked"}
EXPERIMENT_SELECTION_ALIASES = {
    "main": "main_results",
    "main_results": "main_results",
    "main-results": "main_results",
    "covariates": "covariate_comparison",
    "covariate": "covariate_comparison",
    "covariate_comparison": "covariate_comparison",
    "covariate-comparison": "covariate_comparison",
    "baselines": "baselines",
    "baseline": "baselines",
    "smoke": "smoke_one_study",
    "smoke_one_study": "smoke_one_study",
    "smoke-one-study": "smoke_one_study",
}
EXPERIMENT_ORDER = [
    "main_results",
    "covariate_comparison",
    "baselines",
    "smoke_one_study",
]
TIMESCALE_ORDER = ["1h", "12h", "1d", "7d"]
METHOD_ORDER = [
    "baseline_nomove",
    "baseline_constvel",
    "baseline_randomwalk",
    "issf_linear",
    "issf_mlp",
    "deepssf",
    "moveformer",
]
TOP_BLOCKER_LIMIT = 5
SMALL_FAILED_DOWNLOAD_THRESHOLD = 25


def _ordered(values: list[str], *, preferred: list[str]) -> list[str]:
    order = {value: idx for idx, value in enumerate(preferred)}
    return sorted(values, key=lambda value: (order.get(value, len(order)), str(value)))


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _format_pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * float(numerator) / float(denominator)):.1f}%"


def normalize_experiment_selection(
    experiments_override: list[str] | None,
) -> list[str] | None:
    if experiments_override is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    unknown: list[str] = []
    for value in experiments_override:
        key = str(value).strip().lower().replace(" ", "_")
        canonical = EXPERIMENT_SELECTION_ALIASES.get(key)
        if canonical is None:
            unknown.append(str(value))
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    if unknown:
        raise ValueError(
            "Unknown experiment selection(s): "
            f"{unknown}. Allowed: {EXPERIMENT_ORDER}"
        )
    return normalized


def _classify_latest_status(
    latest_record: dict[str, object] | None,
) -> tuple[str, str | None]:
    if latest_record is None:
        return "not_started", "no latest run record"
    status = str(dict(latest_record).get("status", "")).strip().lower()
    if status in KNOWN_RUN_STATUSES:
        return status, None
    if not status:
        return "status_unknown", "missing status tag"
    return "status_unknown", f"unrecognized status tag: {status}"


def _failed_reason(latest_record: dict[str, object] | None) -> str:
    tags = dict(dict(latest_record or {}).get("tags", {}) or {})
    reason = str(tags.get("failure_reason", "")).strip()
    return reason or "missing failure_reason tag"


def _summarize_failure_reason(reason: str) -> str:
    reason = str(reason).strip()
    if reason.startswith("Too many open files"):
        return "Too many open files"
    if reason.startswith("CUDA unknown error"):
        return "CUDA environment/setup error"
    if "database is locked" in reason:
        return "MLflow database locked"
    match = re.match(r"No segments found for split='([^']+)'", reason)
    if match:
        return f"No segments found (split={match.group(1)})"
    match = re.match(r"No rows found for split='([^']+)'", reason)
    if match:
        return f"No rows found in trajectories.csv (split={match.group(1)})"
    match = re.match(r"No valid step starts found for split='([^']+)'", reason)
    if match:
        return f"No valid step starts found (split={match.group(1)})"
    return reason


def _summary_entry(name: str, counts: Counter[str]) -> dict[str, object]:
    planned_runs = sum(int(counts.get(status, 0)) for status in STATUS_KEYS)
    completed_runs = int(counts.get("completed", 0))
    not_started_runs = int(counts.get("not_started", 0))
    running_runs = int(counts.get("running", 0))
    failed_runs = int(counts.get("failed", 0))
    blocked_runs = int(counts.get("blocked", 0))
    status_unknown_runs = int(counts.get("status_unknown", 0))
    remaining_runs = planned_runs - completed_runs
    return {
        "name": str(name),
        "planned_runs": int(planned_runs),
        "completed_runs": completed_runs,
        "not_started_runs": not_started_runs,
        "running_runs": running_runs,
        "failed_runs": failed_runs,
        "blocked_runs": blocked_runs,
        "status_unknown_runs": status_unknown_runs,
        "remaining_runs": int(remaining_runs),
        "completion_pct": _format_pct(completed_runs, planned_runs),
    }


def _ranked_counts(counter: Counter[str]) -> list[dict[str, object]]:
    return [
        {"reason": str(reason), "count": int(count)}
        for reason, count in sorted(
            counter.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]


def _download_status_summary(
    entries: list[dict[str, object]] | None,
) -> dict[str, object]:
    entries = [] if entries is None else [dict(entry) for entry in entries]
    counts = Counter(str(entry.get("classification", "unknown")) for entry in entries)
    failed_examples = [
        entry
        for entry in sorted(
            entries,
            key=lambda entry: (
                -int(entry.get("failed_files", 0)),
                -int(entry.get("failed_points", 0)),
                str(entry.get("study_id", "")),
                str(entry.get("timescale", "")),
            ),
        )
        if str(entry.get("classification", "")) == "failed"
    ]
    return {
        "pairs_considered": int(len(entries)),
        "failed_pairs": int(counts.get("failed", 0)),
        "small_failed_pairs": int(
            sum(
                1
                for entry in failed_examples
                if int(entry.get("failed_files", 0)) <= SMALL_FAILED_DOWNLOAD_THRESHOLD
            )
        ),
        "missing_rasters_root_pairs": int(counts.get("missing_rasters_root", 0)),
        "no_request_status_pairs": int(counts.get("no_request_status", 0)),
        "downloads_ok_pairs": int(counts.get("ok", 0)),
        "status_error_pairs": int(counts.get("status_error", 0)),
        "not_applicable_pairs": int(counts.get("not_applicable", 0)),
        "unknown_pairs": int(counts.get("unknown", 0)),
        "failed_examples": failed_examples[:TOP_BLOCKER_LIMIT],
        "not_started_examples": [
            {
                "study_id": str(entry.get("study_id", "")),
                "timescale": str(entry.get("timescale", "")),
                "classification": str(entry.get("classification", "")),
                "reasons": list(entry.get("reasons", [])),
            }
            for entry in entries
            if str(entry.get("classification", "")) in {
                "missing_rasters_root",
                "no_request_status",
            }
        ][:TOP_BLOCKER_LIMIT],
    }


def _planned_run_index(
    expanded_experiments: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    planned: dict[str, dict[str, object]] = {}
    for expanded in expanded_experiments:
        experiment = dict(expanded.get("experiment", {}))
        experiment_name = str(experiment.get("name", ""))
        for run in expanded.get("runs", []):
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


def _blocked_run_entries(
    expanded_experiments: list[dict[str, object]],
) -> list[dict[str, object]]:
    blocked: dict[str, dict[str, object]] = {}
    for expanded in expanded_experiments:
        experiment = dict(expanded.get("experiment", {}))
        experiment_name = str(experiment.get("name", ""))
        for entry in experiment.get("blocked", []):
            payload = {str(key): value for key, value in dict(entry).items()}
            dedupe_key = json.dumps(payload, sort_keys=True)
            if dedupe_key not in blocked:
                blocked[dedupe_key] = {
                    **payload,
                    "experiment_names": [experiment_name],
                }
                continue
            if experiment_name not in blocked[dedupe_key]["experiment_names"]:
                blocked[dedupe_key]["experiment_names"].append(experiment_name)
    return [
        blocked[key]
        for key in sorted(
            blocked.keys(),
            key=lambda value: (
                str(blocked[value].get("study_id", "")),
                str(blocked[value].get("timescale", "")),
                str(blocked[value].get("covariate_set", "")),
                str(blocked[value].get("method", "")),
                str(blocked[value].get("reason", "")),
            ),
        )
    ]


def _filter_status_snapshot_by_studies(
    status_snapshot: dict[str, object],
    study_ids: set[str],
) -> dict[str, object]:
    filtered_studies = [
        dict(study)
        for study in status_snapshot.get("studies", [])
        if str(study.get("study_id", "")) in study_ids
    ]
    blocked_studies = 0
    ssf_ready_pairs = 0
    raster_ready_pairs = 0
    fully_runnable_pairs = 0
    blocked_pairs = 0
    study_timescale_pairs = 0
    for study in filtered_studies:
        study_has_blocked_pair = False
        timescales = [dict(readiness) for readiness in study.get("timescales", [])]
        study["timescales"] = timescales
        study_timescale_pairs += len(timescales)
        for readiness in timescales:
            if bool(readiness.get("point_ready")):
                ssf_ready_pairs += 1
            if bool(readiness.get("raster_ready")):
                raster_ready_pairs += 1
            if bool(readiness.get("fully_runnable")):
                fully_runnable_pairs += 1
            else:
                blocked_pairs += 1
                study_has_blocked_pair = True
        if study_has_blocked_pair:
            blocked_studies += 1
    counts = {
        **dict(status_snapshot.get("counts", {})),
        "studies": int(len(filtered_studies)),
        "study_timescale_pairs": int(study_timescale_pairs),
        "blocked_studies": int(blocked_studies),
        "ssf_ready_pairs": int(ssf_ready_pairs),
        "raster_ready_pairs": int(raster_ready_pairs),
        "fully_runnable_pairs": int(fully_runnable_pairs),
        "blocked_pairs": int(blocked_pairs),
    }
    return {
        **dict(status_snapshot),
        "counts": counts,
        "studies": filtered_studies,
    }


def build_summary_payload(
    *,
    status_snapshot: dict[str, object],
    planned_runs: dict[str, dict[str, object]],
    blocked_runs: list[dict[str, object]],
    latest_run_records: dict[str, dict[str, object]],
    download_entries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    overall_counts: Counter[str] = Counter()
    experiment_counts: dict[str, Counter[str]] = defaultdict(Counter)
    method_counts: dict[str, Counter[str]] = defaultdict(Counter)
    timescale_counts: dict[str, Counter[str]] = defaultdict(Counter)
    blocked_run_reasons: Counter[str] = Counter()
    blocked_pair_reasons: Counter[str] = Counter()
    status_unknown_reasons: Counter[str] = Counter()
    failed_run_reasons: Counter[str] = Counter()
    status_unknown_examples: list[dict[str, object]] = []
    failed_run_examples: list[dict[str, object]] = []

    for run_key, run_payload in planned_runs.items():
        latest_record = latest_run_records.get(run_key)
        status, status_reason = _classify_latest_status(latest_record)
        overall_counts[status] += 1
        method = str(run_payload["method"])
        timescale = str(run_payload["timescale"])
        method_counts[method][status] += 1
        timescale_counts[timescale][status] += 1
        for experiment_name in run_payload.get("experiment_names", []):
            experiment_counts[str(experiment_name)][status] += 1
        if status == "status_unknown":
            reason = str(status_reason or "unknown status")
            status_unknown_reasons[reason] += 1
            status_unknown_examples.append(
                {
                    "study_id": str(run_payload["study_id"]),
                    "timescale": timescale,
                    "method": method,
                    "covariate_set": str(run_payload["covariate_set"]),
                    "run_key": str(run_key),
                    "reason": reason,
                }
            )
        elif status == "failed":
            exact_reason = _failed_reason(latest_record)
            reason = _summarize_failure_reason(exact_reason)
            failed_run_reasons[reason] += 1
            failed_run_examples.append(
                {
                    "study_id": str(run_payload["study_id"]),
                    "timescale": timescale,
                    "method": method,
                    "covariate_set": str(run_payload["covariate_set"]),
                    "run_key": str(run_key),
                    "reason_group": reason,
                    "reason": exact_reason,
                }
            )

    for blocked_run in blocked_runs:
        overall_counts["blocked"] += 1
        method = str(blocked_run["method"])
        timescale = str(blocked_run["timescale"])
        reason = str(blocked_run.get("reason", "missing prerequisites"))
        method_counts[method]["blocked"] += 1
        timescale_counts[timescale]["blocked"] += 1
        blocked_run_reasons[reason] += 1
        for experiment_name in blocked_run.get("experiment_names", []):
            experiment_counts[str(experiment_name)]["blocked"] += 1

    for study in status_snapshot.get("studies", []):
        for readiness in study.get("timescales", []):
            for reason in readiness.get("blocked_reasons", []):
                blocked_pair_reasons[str(reason)] += 1

    experiment_names = _ordered(
        list(experiment_counts.keys()),
        preferred=EXPERIMENT_ORDER,
    )
    method_names = _ordered(list(method_counts.keys()), preferred=METHOD_ORDER)
    timescale_names = _ordered(list(timescale_counts.keys()), preferred=TIMESCALE_ORDER)

    overall_planned = sum(int(overall_counts.get(status, 0)) for status in STATUS_KEYS)
    overall_completed = int(overall_counts.get("completed", 0))
    counts = dict(status_snapshot.get("counts", {}))
    summary = {
        "overall": {
            "planned_runs": int(overall_planned),
            "completed_runs": overall_completed,
            "not_started_runs": int(overall_counts.get("not_started", 0)),
            "running_runs": int(overall_counts.get("running", 0)),
            "failed_runs": int(overall_counts.get("failed", 0)),
            "blocked_runs": int(overall_counts.get("blocked", 0)),
            "status_unknown_runs": int(overall_counts.get("status_unknown", 0)),
            "remaining_runs": int(overall_planned - overall_completed),
            "completion_pct": _format_pct(overall_completed, overall_planned),
            "studies": int(counts.get("studies", 0)),
            "blocked_studies": int(counts.get("blocked_studies", 0)),
            "study_timescale_pairs": int(counts.get("study_timescale_pairs", 0)),
            "fully_runnable_pairs": int(counts.get("fully_runnable_pairs", 0)),
            "blocked_pairs": int(counts.get("blocked_pairs", 0)),
        },
        "experiments": [
            {
                "experiment_name": entry["name"],
                **{
                    key: value
                    for key, value in entry.items()
                    if key != "name"
                },
            }
            for entry in (
                _summary_entry(name, experiment_counts[name])
                for name in experiment_names
            )
        ],
        "methods": [
            {
                "method": entry["name"],
                **{
                    key: value
                    for key, value in entry.items()
                    if key != "name"
                },
            }
            for entry in (
                _summary_entry(name, method_counts[name])
                for name in method_names
            )
        ],
        "timescales": [
            {
                "timescale": entry["name"],
                **{
                    key: value
                    for key, value in entry.items()
                    if key != "name"
                },
            }
            for entry in (
                _summary_entry(name, timescale_counts[name])
                for name in timescale_names
            )
        ],
        "blockers": {
            "blocked_runs_by_reason": _ranked_counts(blocked_run_reasons),
            "blocked_pairs_by_reason": _ranked_counts(blocked_pair_reasons),
            "failed_runs_by_method": _ranked_counts(
                Counter(
                    {
                        str(name): int(method_counts[name].get("failed", 0))
                        for name in method_names
                        if int(method_counts[name].get("failed", 0)) > 0
                    }
                )
            ),
            "running_runs_by_method": _ranked_counts(
                Counter(
                    {
                        str(name): int(method_counts[name].get("running", 0))
                        for name in method_names
                        if int(method_counts[name].get("running", 0)) > 0
                    }
                )
            ),
            "status_unknown_runs_by_reason": _ranked_counts(status_unknown_reasons),
            "status_unknown_examples": sorted(
                status_unknown_examples,
                key=lambda entry: (
                    str(entry["reason"]),
                    str(entry["study_id"]),
                    str(entry["timescale"]),
                    str(entry["method"]),
                ),
            )[:TOP_BLOCKER_LIMIT],
            "failed_runs_by_reason": _ranked_counts(failed_run_reasons),
            "failed_run_examples": sorted(
                failed_run_examples,
                key=lambda entry: (
                    -int(failed_run_reasons[str(entry["reason_group"])]),
                    str(entry["reason_group"]),
                    str(entry["study_id"]),
                    str(entry["timescale"]),
                    str(entry["method"]),
                ),
            )[:TOP_BLOCKER_LIMIT],
        },
        "download_status": _download_status_summary(download_entries),
    }
    return summary


def _render_rollup(items: list[dict[str, object]]) -> str:
    if not items:
        return "none"
    return ", ".join(
        f"{item['reason']} {_format_count(int(item['count']))}" for item in items
    )


def _render_method_rollup(items: list[dict[str, object]]) -> str:
    if not items:
        return "none"
    return ", ".join(
        f"{item['reason']} {_format_count(int(item['count']))}" for item in items
    )


def _render_reasons(reasons: list[object]) -> str:
    if not reasons:
        return "unknown"
    return "; ".join(str(reason) for reason in reasons)


def _render_download_examples(entries: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        lines.append(
            (
                f"  {entry['study_id']} / {entry['timescale']}: "
                f"failed_files={_format_count(int(entry.get('failed_files', 0)))}"
                f", failed_points={_format_count(int(entry.get('failed_points', 0)))}"
                f", requested_points={_format_count(int(entry.get('requested_points', 0)))}"
                f", downloaded_files={_format_count(int(entry.get('downloaded_files', 0)))}"
                f", existing_files={_format_count(int(entry.get('existing_files', 0)))}"
            )
        )
    return lines


def _render_not_started_examples(entries: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        label = (
            "no raster cache root"
            if str(entry.get("classification", "")) == "missing_rasters_root"
            else "no matching request report"
        )
        lines.append(
            f"  {entry['study_id']} / {entry['timescale']}: {label}; {_render_reasons(list(entry.get('reasons', [])))}"
        )
    return lines


def _render_run_reason_examples(entries: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for entry in entries:
        reason = str(entry["reason"])
        reason_group = str(entry.get("reason_group", reason))
        detail = f"{reason_group}: {reason}" if reason_group != reason else reason
        lines.append(
            f"  {entry['study_id']} / {entry['timescale']} / {entry['method']}: {detail}"
        )
    return lines


def render_summary_text(summary: dict[str, object]) -> str:
    overall = dict(summary["overall"])
    download_status = dict(summary.get("download_status", {}))
    blockers = dict(summary["blockers"])
    lines = [
        "Overall",
        (
            f"  completed {_format_count(overall['completed_runs'])} / "
            f"{_format_count(overall['planned_runs'])} runs ({overall['completion_pct']})"
        ),
        (
            f"  remaining {_format_count(overall['remaining_runs'])} = "
            f"{_format_count(overall['not_started_runs'])} not started + "
            f"{_format_count(overall['running_runs'])} running + "
            f"{_format_count(overall['failed_runs'])} failed + "
            f"{_format_count(overall['blocked_runs'])} blocked + "
            f"{_format_count(overall['status_unknown_runs'])} status unknown"
        ),
        (
            f"  readiness {_format_count(overall['fully_runnable_pairs'])} / "
            f"{_format_count(overall['study_timescale_pairs'])} study-timescale pairs fully runnable; "
            f"{_format_count(overall['blocked_studies'])} / "
            f"{_format_count(overall['studies'])} studies blocked"
        ),
        "",
        "Experiments",
    ]
    for entry in summary["experiments"]:
        lines.append(
            (
                f"  {entry['experiment_name']}: "
                f"{_format_count(entry['completed_runs'])} / "
                f"{_format_count(entry['planned_runs'])} completed "
                f"({entry['completion_pct']}); remaining "
                f"{_format_count(entry['remaining_runs'])} = "
                f"{_format_count(entry['not_started_runs'])} not started + "
                f"{_format_count(entry['running_runs'])} running + "
                f"{_format_count(entry['failed_runs'])} failed + "
                f"{_format_count(entry['blocked_runs'])} blocked + "
                f"{_format_count(entry['status_unknown_runs'])} status unknown"
            )
        )

    lines.append("")
    lines.append("Remaining by method")
    for entry in summary["methods"]:
        if int(entry["remaining_runs"]) <= 0:
            continue
        lines.append(
            (
                f"  {entry['method']}: remaining {_format_count(entry['remaining_runs'])} = "
                f"{_format_count(entry['not_started_runs'])} not started + "
                f"{_format_count(entry['running_runs'])} running + "
                f"{_format_count(entry['failed_runs'])} failed + "
                f"{_format_count(entry['blocked_runs'])} blocked + "
                f"{_format_count(entry['status_unknown_runs'])} status unknown"
            )
        )

    timescale_entries = [entry for entry in summary["timescales"] if int(entry["remaining_runs"]) > 0]
    if timescale_entries:
        lines.append("")
        lines.append("Remaining by timescale")
        for entry in timescale_entries:
            lines.append(
                (
                    f"  {entry['timescale']}: remaining {_format_count(entry['remaining_runs'])} = "
                    f"{_format_count(entry['not_started_runs'])} not started + "
                    f"{_format_count(entry['running_runs'])} running + "
                    f"{_format_count(entry['failed_runs'])} failed + "
                    f"{_format_count(entry['blocked_runs'])} blocked + "
                    f"{_format_count(entry['status_unknown_runs'])} status unknown"
                )
            )

    lines.extend(
        [
            "",
            "Top blockers",
            (
                "  blocked logical runs: "
                + _render_rollup(
                    list(blockers["blocked_runs_by_reason"])[:TOP_BLOCKER_LIMIT]
                )
            ),
            (
                "  blocked study-timescale pairs: "
                + _render_rollup(
                    list(blockers["blocked_pairs_by_reason"])[:TOP_BLOCKER_LIMIT]
                )
            ),
            "",
            "Active problems",
            (
                f"  failed runs: {_format_count(overall['failed_runs'])} = "
                + _render_method_rollup(
                    list(blockers["failed_runs_by_method"])[:TOP_BLOCKER_LIMIT]
                )
            ),
            (
                f"  running runs: {_format_count(overall['running_runs'])} = "
                + _render_method_rollup(
                    list(blockers["running_runs_by_method"])[:TOP_BLOCKER_LIMIT]
                )
            ),
        ]
    )
    if int(overall.get("status_unknown_runs", 0)) > 0:
        lines.extend(
            [
                "",
                "Why status unknown",
                (
                    f"  status unknown runs: {_format_count(overall['status_unknown_runs'])} = "
                    + _render_rollup(
                        list(blockers["status_unknown_runs_by_reason"])[:TOP_BLOCKER_LIMIT]
                    )
                ),
            ]
        )
        status_unknown_examples = list(blockers.get("status_unknown_examples", []))
        if status_unknown_examples:
            lines.append("  examples:")
            lines.extend(_render_run_reason_examples(status_unknown_examples))
    if int(overall.get("failed_runs", 0)) > 0:
        lines.extend(
            [
                "",
                "Failed run reasons",
                (
                    f"  failed runs by reason: "
                    + _render_rollup(
                        list(blockers["failed_runs_by_reason"])[:TOP_BLOCKER_LIMIT]
                    )
                ),
            ]
        )
        failed_run_examples = list(blockers.get("failed_run_examples", []))
        if failed_run_examples:
            lines.append("  examples:")
            lines.extend(_render_run_reason_examples(failed_run_examples))
    if int(download_status.get("pairs_considered", 0)) > 0:
        incomplete_pairs = (
            int(download_status.get("failed_pairs", 0))
            + int(download_status.get("missing_rasters_root_pairs", 0))
            + int(download_status.get("no_request_status_pairs", 0))
            + int(download_status.get("status_error_pairs", 0))
            + int(download_status.get("unknown_pairs", 0))
        )
        lines.extend(
            [
                "",
                "Download status behind blocked pairs",
                (
                    "  incomplete downloads likely block "
                    f"{_format_count(incomplete_pairs)} / "
                    f"{_format_count(int(download_status['pairs_considered']))} blocked study-timescale pairs"
                ),
                (
                    "  known failed download requests: "
                    f"{_format_count(int(download_status['failed_pairs']))}"
                    f" (small residual <= {SMALL_FAILED_DOWNLOAD_THRESHOLD} failed files: "
                    f"{_format_count(int(download_status['small_failed_pairs']))})"
                ),
                (
                    "  likely not started (missing raster cache root): "
                    f"{_format_count(int(download_status['missing_rasters_root_pairs']))}"
                ),
                (
                    "  raster cache exists but no matching request report: "
                    f"{_format_count(int(download_status['no_request_status_pairs']))}"
                ),
                (
                    "  downloads already marked ok; blocker is downstream extraction/cache work: "
                    f"{_format_count(int(download_status['downloads_ok_pairs']))}"
                ),
            ]
        )
        if int(download_status.get("status_error_pairs", 0)) > 0:
            lines.append(
                "  download status inspection errors: "
                f"{_format_count(int(download_status['status_error_pairs']))}"
            )
        failed_examples = list(download_status.get("failed_examples", []))
        if failed_examples:
            lines.append("  largest failed requests:")
            lines.extend(_render_download_examples(failed_examples))
        not_started_examples = list(download_status.get("not_started_examples", []))
        if not_started_examples:
            lines.append("  not started / no report examples:")
            lines.extend(_render_not_started_examples(not_started_examples))
    return "\n".join(lines)


def build_current_summary_payload(
    studies_override: list[str] | None = None,
    experiments_override: list[str] | None = None,
) -> dict[str, object]:
    import movebench.bench.config as bench_config
    import movebench.bench.run as bench_run
    from movebench.download_reports import load_chip_request_status_for_extract
    from movebench.study import Study

    status_snapshot, expanded_experiments = bench_run._status_context(
        studies_override=studies_override
    )
    selected_experiments = normalize_experiment_selection(experiments_override)
    if selected_experiments is not None:
        selected_names = set(selected_experiments)
        expanded_experiments = [
            expanded
            for expanded in expanded_experiments
            if str(dict(expanded.get("experiment", {})).get("name", "")) in selected_names
        ]
        selected_studies = {
            str(study_id)
            for expanded in expanded_experiments
            for study_id in dict(expanded.get("experiment", {})).get(
                "resolved_studies", []
            )
        }
        status_snapshot = _filter_status_snapshot_by_studies(
            status_snapshot,
            selected_studies,
        )
    planned_runs = _planned_run_index(expanded_experiments)
    blocked_runs = _blocked_run_entries(expanded_experiments)
    latest_run_records = bench_run_state.search_latest_run_records_by_key(
        list(planned_runs.keys())
    )
    download_entries: list[dict[str, object]] = []
    for study_entry in status_snapshot.get("studies", []):
        blocked_timescales = [
            dict(readiness)
            for readiness in study_entry.get("timescales", [])
            if list(readiness.get("blocked_reasons", []))
        ]
        if not blocked_timescales:
            continue
        study = Study(str(study_entry["study_id"]))
        for readiness in blocked_timescales:
            reasons = [str(reason) for reason in readiness.get("blocked_reasons", [])]
            entry = {
                "study_id": str(study_entry["study_id"]),
                "timescale": str(readiness["timescale"]),
                "reasons": reasons,
                "classification": "unknown",
            }
            if any(reason == "study has no train split" for reason in reasons):
                entry["classification"] = "not_applicable"
                download_entries.append(entry)
                continue
            timescale = bench_config.resolve_timescale(
                str(readiness["timescale"]),
                study_id=str(study_entry["study_id"]),
            )
            rasters_root = study.study_dir / "rasters" / str(timescale["rasters_hash"])
            if not rasters_root.exists():
                entry["classification"] = "missing_rasters_root"
                download_entries.append(entry)
                continue
            splits = ["train", *[str(split) for split in readiness.get("held_out_splits", [])]]
            try:
                request_status = load_chip_request_status_for_extract(
                    study=study,
                    plan=timescale["plan"],
                    splits=splits,
                )
            except Exception as exc:
                entry["classification"] = "status_error"
                entry["error"] = str(exc)
                download_entries.append(entry)
                continue
            if not isinstance(request_status, dict):
                entry["classification"] = "no_request_status"
                download_entries.append(entry)
                continue
            entry["request_status"] = str(request_status.get("status", ""))
            entry["requested_points"] = int(request_status.get("requested_point_count", 0))
            report = request_status.get("report")
            chip_groups = report.get("chip_groups", {}) if isinstance(report, dict) else {}
            failed_points: set[int] = set()
            failed_files = 0
            downloaded_files = 0
            existing_files = 0
            for group_stats in chip_groups.values():
                if not isinstance(group_stats, dict):
                    continue
                failed_files += int(group_stats.get("failed", 0))
                downloaded_files += int(group_stats.get("downloaded", 0))
                existing_files += int(group_stats.get("existing", 0))
                failures = group_stats.get("failures")
                if not isinstance(failures, list):
                    continue
                for item in failures:
                    if not isinstance(item, dict):
                        continue
                    point_id = item.get("point_id")
                    if point_id is None:
                        continue
                    try:
                        failed_points.add(int(point_id))
                    except (TypeError, ValueError):
                        continue
            entry["failed_files"] = int(failed_files)
            entry["failed_points"] = int(len(failed_points))
            entry["downloaded_files"] = int(downloaded_files)
            entry["existing_files"] = int(existing_files)
            status = str(request_status.get("status", "")).strip().lower()
            if status == "failed":
                entry["classification"] = "failed"
            elif status == "ok":
                entry["classification"] = "ok"
            else:
                entry["classification"] = "unknown"
            download_entries.append(entry)
    return build_summary_payload(
        status_snapshot=status_snapshot,
        planned_runs=planned_runs,
        blocked_runs=blocked_runs,
        latest_run_records=latest_run_records,
        download_entries=download_entries,
    )


def render_current_summary(
    studies_override: list[str] | None = None,
    experiments_override: list[str] | None = None,
) -> str:
    return render_summary_text(
        build_current_summary_payload(
            studies_override=studies_override,
            experiments_override=experiments_override,
        )
    )
