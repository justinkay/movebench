from __future__ import annotations

import argparse
import contextlib
import io
import sys
import types
import unittest
from unittest import mock


def _fake_mlflow_module() -> types.ModuleType:
    module = types.ModuleType("mlflow")
    module.set_tracking_uri = lambda uri: None
    module.start_run = lambda **kwargs: None
    module.log_params = lambda params: None
    module.tracking = types.SimpleNamespace(MlflowClient=lambda **kwargs: None)
    return module


_RUN_MODULE_CACHE: types.ModuleType | None = None


def _load_run_module():
    global _RUN_MODULE_CACHE
    if _RUN_MODULE_CACHE is not None:
        return _RUN_MODULE_CACHE
    with mock.patch.dict(sys.modules, {"mlflow": _fake_mlflow_module()}):
        sys.modules.pop("movebench.bench.mlflow", None)
        sys.modules.pop("movebench.bench.run", None)
        import movebench.bench.run as run_module

    _RUN_MODULE_CACHE = run_module
    return run_module


class BenchSummaryTest(unittest.TestCase):
    def test_normalize_experiment_selection_accepts_aliases(self) -> None:
        from movebench.bench import summary as summary_module

        self.assertEqual(
            summary_module.normalize_experiment_selection(
                ["covariates", "main-results", "smoke"]
            ),
            ["covariate_comparison", "main_results", "smoke_one_study"],
        )

    def test_blocked_run_entries_dedupe_and_collect_experiment_names(self) -> None:
        from movebench.bench import summary as summary_module

        blocked_runs = summary_module._blocked_run_entries(
            [
                {
                    "experiment": {
                        "name": "main_results",
                        "blocked": [
                            {
                                "study_id": "study-a",
                                "timescale": "1d",
                                "covariate_set": "all",
                                "method": "deepssf",
                                "reason": "missing deepSSF cache manifest for train",
                            }
                        ],
                    }
                },
                {
                    "experiment": {
                        "name": "covariate_comparison",
                        "blocked": [
                            {
                                "study_id": "study-a",
                                "timescale": "1d",
                                "covariate_set": "all",
                                "method": "deepssf",
                                "reason": "missing deepSSF cache manifest for train",
                            }
                        ],
                    }
                },
            ]
        )

        self.assertEqual(len(blocked_runs), 1)
        self.assertEqual(
            blocked_runs[0]["experiment_names"],
            ["main_results", "covariate_comparison"],
        )

    def test_build_summary_payload_rolls_up_experiments_and_blockers(self) -> None:
        from movebench.bench import summary as summary_module

        payload = summary_module.build_summary_payload(
            status_snapshot={
                "counts": {
                    "studies": 2,
                    "blocked_studies": 1,
                    "study_timescale_pairs": 4,
                    "fully_runnable_pairs": 3,
                    "blocked_pairs": 1,
                },
                "studies": [
                    {
                        "study_id": "study-a",
                        "timescales": [
                            {"timescale": "1h", "blocked_reasons": []},
                            {
                                "timescale": "1d",
                                "blocked_reasons": [
                                    "missing deepSSF cache manifest for train"
                                ],
                            },
                        ],
                    },
                    {
                        "study_id": "study-b",
                        "timescales": [
                            {"timescale": "1h", "blocked_reasons": []},
                            {"timescale": "1d", "blocked_reasons": []},
                        ],
                    },
                ],
            },
            planned_runs={
                "run-completed": {
                    "study_id": "study-a",
                    "timescale": "1h",
                    "method": "issf_linear",
                    "covariate_set": "all",
                    "experiment_names": ["main_results", "smoke_one_study"],
                },
                "run-pending": {
                    "study_id": "study-a",
                    "timescale": "1d",
                    "method": "deepssf",
                    "covariate_set": "all",
                    "experiment_names": ["main_results"],
                },
                "run-running": {
                    "study_id": "study-b",
                    "timescale": "1h",
                    "method": "moveformer",
                    "covariate_set": "all",
                    "experiment_names": ["covariate_comparison"],
                },
                "run-failed": {
                    "study_id": "study-b",
                    "timescale": "1d",
                    "method": "issf_mlp",
                    "covariate_set": "all",
                    "experiment_names": ["covariate_comparison"],
                },
                "run-unknown": {
                    "study_id": "study-c",
                    "timescale": "12h",
                    "method": "moveformer",
                    "covariate_set": "all",
                    "experiment_names": ["covariate_comparison"],
                },
            },
            blocked_runs=[
                {
                    "study_id": "study-a",
                    "timescale": "1d",
                    "covariate_set": "all",
                    "method": "deepssf",
                    "reason": "missing deepSSF cache manifest for train",
                    "experiment_names": ["covariate_comparison"],
                },
                {
                    "study_id": "study-b",
                    "timescale": "1d",
                    "covariate_set": "all",
                    "method": "issf_linear",
                    "reason": "missing point store manifest for train",
                    "experiment_names": ["main_results"],
                },
            ],
            latest_run_records={
                "run-completed": {"status": "completed"},
                "run-running": {"status": "running"},
                "run-failed": {
                    "status": "failed",
                    "tags": {"failure_reason": "CUDA out of memory"},
                },
                "run-unknown": {
                    "status": "",
                    "run_id": "unknown-run-id",
                    "tags": {},
                },
            },
            download_entries=[
                {
                    "study_id": "study-a",
                    "timescale": "1d",
                    "reasons": ["missing point store manifest for train"],
                    "classification": "failed",
                    "requested_points": 100,
                    "failed_files": 2,
                    "failed_points": 2,
                    "downloaded_files": 8,
                    "existing_files": 190,
                },
                {
                    "study_id": "study-b",
                    "timescale": "7d",
                    "reasons": ["missing point store manifest for train"],
                    "classification": "missing_rasters_root",
                },
                {
                    "study_id": "study-c",
                    "timescale": "12h",
                    "reasons": ["missing deepSSF cache manifest for train"],
                    "classification": "no_request_status",
                },
                {
                    "study_id": "study-d",
                    "timescale": "1h",
                    "reasons": ["missing deepSSF cache manifest for train"],
                    "classification": "ok",
                },
            ],
        )

        self.assertEqual(payload["overall"]["planned_runs"], 7)
        self.assertEqual(payload["overall"]["completed_runs"], 1)
        self.assertEqual(payload["overall"]["not_started_runs"], 1)
        self.assertEqual(payload["overall"]["running_runs"], 1)
        self.assertEqual(payload["overall"]["failed_runs"], 1)
        self.assertEqual(payload["overall"]["blocked_runs"], 2)
        self.assertEqual(payload["overall"]["status_unknown_runs"], 1)
        self.assertEqual(payload["overall"]["remaining_runs"], 6)

        by_experiment = {
            entry["experiment_name"]: entry for entry in payload["experiments"]
        }
        self.assertEqual(by_experiment["main_results"]["planned_runs"], 3)
        self.assertEqual(by_experiment["main_results"]["completed_runs"], 1)
        self.assertEqual(by_experiment["main_results"]["not_started_runs"], 1)
        self.assertEqual(by_experiment["main_results"]["blocked_runs"], 1)
        self.assertEqual(by_experiment["covariate_comparison"]["planned_runs"], 4)
        self.assertEqual(by_experiment["covariate_comparison"]["running_runs"], 1)
        self.assertEqual(by_experiment["covariate_comparison"]["failed_runs"], 1)
        self.assertEqual(by_experiment["covariate_comparison"]["blocked_runs"], 1)
        self.assertEqual(by_experiment["covariate_comparison"]["status_unknown_runs"], 1)
        self.assertEqual(by_experiment["smoke_one_study"]["planned_runs"], 1)
        self.assertEqual(by_experiment["smoke_one_study"]["completed_runs"], 1)

        self.assertEqual(
            payload["blockers"]["blocked_runs_by_reason"][0],
            {
                "count": 1,
                "reason": "missing deepSSF cache manifest for train",
            },
        )
        self.assertEqual(
            payload["blockers"]["blocked_pairs_by_reason"][0],
            {
                "count": 1,
                "reason": "missing deepSSF cache manifest for train",
            },
        )
        self.assertEqual(payload["download_status"]["pairs_considered"], 4)
        self.assertEqual(payload["download_status"]["failed_pairs"], 1)
        self.assertEqual(payload["download_status"]["small_failed_pairs"], 1)
        self.assertEqual(payload["download_status"]["missing_rasters_root_pairs"], 1)
        self.assertEqual(payload["download_status"]["no_request_status_pairs"], 1)
        self.assertEqual(payload["download_status"]["downloads_ok_pairs"], 1)
        self.assertEqual(
            payload["blockers"]["status_unknown_runs_by_reason"],
            [{"count": 1, "reason": "missing status tag"}],
        )
        self.assertEqual(
            payload["blockers"]["failed_runs_by_reason"],
            [{"count": 1, "reason": "CUDA out of memory"}],
        )

        rendered = summary_module.render_summary_text(payload)
        self.assertIn("Overall", rendered)
        self.assertIn("Experiments", rendered)
        self.assertIn("main_results: 1 / 3 completed (33.3%)", rendered)
        self.assertIn("covariate_comparison: 0 / 4 completed (0.0%)", rendered)
        self.assertIn(
            "remaining 6 = 1 not started + 1 running + 1 failed + 2 blocked + 1 status unknown",
            rendered,
        )
        self.assertIn("Top blockers", rendered)
        self.assertIn("failed runs: 1 = issf_mlp 1", rendered)
        self.assertIn("running runs: 1 = moveformer 1", rendered)
        self.assertIn("Why status unknown", rendered)
        self.assertIn("status unknown runs: 1 = missing status tag 1", rendered)
        self.assertIn("study-c / 12h / moveformer: missing status tag", rendered)
        self.assertIn("Failed run reasons", rendered)
        self.assertIn("failed runs by reason: CUDA out of memory 1", rendered)
        self.assertIn("study-b / 1d / issf_mlp: CUDA out of memory", rendered)
        self.assertIn("Download status behind blocked pairs", rendered)
        self.assertIn(
            "incomplete downloads likely block 3 / 4 blocked study-timescale pairs",
            rendered,
        )
        self.assertIn("known failed download requests: 1", rendered)
        self.assertIn("largest failed requests:", rendered)
        self.assertIn(
            "study-a / 1d: failed_files=2, failed_points=2, requested_points=100",
            rendered,
        )
        self.assertIn(
            "study-b / 7d: no raster cache root; missing point store manifest for train",
            rendered,
        )

    def test_build_summary_payload_groups_failed_reason_categories(self) -> None:
        from movebench.bench import summary as summary_module

        payload = summary_module.build_summary_payload(
            status_snapshot={"counts": {}, "studies": []},
            planned_runs={
                "run-a": {
                    "study_id": "study-a",
                    "timescale": "1d",
                    "method": "issf_linear",
                    "covariate_set": "all",
                    "experiment_names": ["main_results"],
                },
                "run-b": {
                    "study_id": "study-b",
                    "timescale": "1d",
                    "method": "issf_linear",
                    "covariate_set": "all",
                    "experiment_names": ["main_results"],
                },
                "run-c": {
                    "study_id": "study-c",
                    "timescale": "7d",
                    "method": "moveformer",
                    "covariate_set": "all",
                    "experiment_names": ["main_results"],
                },
            },
            blocked_runs=[],
            latest_run_records={
                "run-a": {
                    "status": "failed",
                    "tags": {
                        "failure_reason": (
                            "No rows found for split='test_individual' in "
                            "/tmp/study-a/trajectories.csv"
                        )
                    },
                },
                "run-b": {
                    "status": "failed",
                    "tags": {
                        "failure_reason": (
                            "No rows found for split='test_individual' in "
                            "/tmp/study-b/trajectories.csv"
                        )
                    },
                },
                "run-c": {
                    "status": "failed",
                    "tags": {
                        "failure_reason": (
                            "CUDA unknown error - this may be due to an incorrectly "
                            "set up environment"
                        )
                    },
                },
            },
        )

        self.assertEqual(
            payload["blockers"]["failed_runs_by_reason"],
            [
                {
                    "count": 2,
                    "reason": "No rows found in trajectories.csv (split=test_individual)",
                },
                {"count": 1, "reason": "CUDA environment/setup error"},
            ],
        )

        rendered = summary_module.render_summary_text(payload)
        self.assertIn(
            (
                "failed runs by reason: No rows found in trajectories.csv "
                "(split=test_individual) 2, CUDA environment/setup error 1"
            ),
            rendered,
        )
        self.assertIn(
            (
                "study-a / 1d / issf_linear: No rows found in trajectories.csv "
                "(split=test_individual): No rows found for split='test_individual'"
            ),
            rendered,
        )

    def test_build_current_summary_payload_uses_filesystem_run_records(self) -> None:
        from movebench.bench import summary as summary_module

        run_module = _load_run_module()
        status_snapshot = {
            "counts": {
                "studies": 1,
                "blocked_studies": 0,
                "study_timescale_pairs": 1,
                "fully_runnable_pairs": 1,
                "blocked_pairs": 0,
            },
            "studies": [
                {
                    "study_id": "study-a",
                    "timescales": [
                        {
                            "timescale": "1h",
                            "held_out_splits": ["test_temporal"],
                            "point_ready": True,
                            "raster_ready": True,
                            "fully_runnable": True,
                            "blocked_reasons": [],
                        }
                    ],
                }
            ],
        }
        expanded_experiments = [
            {
                "experiment": {
                    "name": "main_results",
                    "resolved_studies": ["study-a"],
                    "blocked": [],
                },
                "runs": [
                    {
                        "run_key": "run-a",
                        "study_id": "study-a",
                        "timescale": "1h",
                        "method": "issf_linear",
                        "covariate_set": "all",
                    }
                ],
            }
        ]
        failing_mlflow = types.ModuleType("movebench.bench.mlflow")
        failing_mlflow.search_latest_run_records_by_key = lambda run_keys: (_ for _ in ()).throw(
            AssertionError(f"unexpected MLflow lookup for {run_keys}")
        )

        with (
            mock.patch.object(
                run_module,
                "_status_context",
                return_value=(status_snapshot, expanded_experiments),
            ),
            mock.patch.object(
                summary_module,
                "bench_run_state",
                types.SimpleNamespace(
                    search_latest_run_records_by_key=mock.Mock(
                        return_value={"run-a": {"status": "completed", "tags": {}}}
                    )
                ),
                create=True,
            ),
            mock.patch.dict(
                sys.modules,
                {"movebench.bench.mlflow": failing_mlflow},
            ),
        ):
            payload = summary_module.build_current_summary_payload(
                studies_override=["study-a"]
            )

        self.assertEqual(payload["overall"]["planned_runs"], 1)
        self.assertEqual(payload["overall"]["completed_runs"], 1)
        self.assertEqual(payload["overall"]["remaining_runs"], 0)

    def test_bench_summary_command_prints_text(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(["bench", "summary", "--studies", "all"])
        stdout = io.StringIO()

        with (
            mock.patch.object(
                run_module,
                "_bench_summary_module",
                return_value=types.SimpleNamespace(
                    normalize_experiment_selection=lambda values: values,
                    render_current_summary=lambda studies_override=None, experiments_override=None: (
                        f"summary output for {' '.join(studies_override or [])}"
                    )
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        self.assertEqual(
            stdout.getvalue().strip(),
            "Study selection: all\nsummary output for all",
        )

    def test_bench_summary_prints_all_and_mini_by_default(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(["bench", "summary"])
        stdout = io.StringIO()
        observed: list[list[str]] = []

        def _render_current_summary(
            studies_override: list[str] | None = None,
            experiments_override: list[str] | None = None,
        ) -> str:
            self.assertIsNone(experiments_override)
            observed.append(list(studies_override or []))
            return f"summary for {' '.join(studies_override or [])}"

        with (
            mock.patch.object(
                run_module,
                "_bench_summary_module",
                return_value=types.SimpleNamespace(
                    normalize_experiment_selection=lambda values: values,
                    render_current_summary=_render_current_summary
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        self.assertEqual(observed, [["all"], ["mini"]])
        self.assertEqual(
            stdout.getvalue().strip(),
            (
                "Study selection: all\nsummary for all\n\n"
                "Study selection: mini\nsummary for mini"
            ),
        )

    def test_bench_summary_studies_override_prints_only_requested_selection(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(["bench", "summary", "--studies", "mini"])
        stdout = io.StringIO()

        render_current_summary = mock.Mock(return_value="summary for mini")
        with (
            mock.patch.object(
                run_module,
                "_bench_summary_module",
                return_value=types.SimpleNamespace(
                    normalize_experiment_selection=lambda values: values,
                    render_current_summary=render_current_summary
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        render_current_summary.assert_called_once_with(
            studies_override=["mini"],
            experiments_override=None,
        )
        self.assertEqual(
            stdout.getvalue().strip(),
            "Study selection: mini\nsummary for mini",
        )

    def test_bench_summary_experiments_override_prints_only_requested_selection(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(
            ["bench", "summary", "--studies", "mini", "--experiments", "covariates"]
        )
        stdout = io.StringIO()

        render_current_summary = mock.Mock(return_value="summary for mini covariates")
        with (
            mock.patch.object(
                run_module,
                "_bench_summary_module",
                return_value=types.SimpleNamespace(
                    render_current_summary=render_current_summary,
                    normalize_experiment_selection=lambda values: [
                        "covariate_comparison"
                    ],
                ),
            ),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        render_current_summary.assert_called_once_with(
            studies_override=["mini"],
            experiments_override=["covariate_comparison"],
        )
        self.assertEqual(
            stdout.getvalue().strip(),
            (
                "Study selection: mini\n"
                "Experiment selection: covariate_comparison\n"
                "summary for mini covariates"
            ),
        )
