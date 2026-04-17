from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _resolved_payload(
    *,
    run_key: str,
    method_hparams: dict[str, object] | None = None,
    method_model: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "run_key": run_key,
        "study_id": "study-a",
        "timescale": "1d",
        "covariate_set": "all",
        "method": "issf_linear",
        "seed": 0,
        "step_time_s": 86400,
        "early_stop_split_key": "test_temporal",
        "evaluation_split_keys": ["test_temporal"],
        "method_hparams": {} if method_hparams is None else dict(method_hparams),
        "method_model": {} if method_model is None else dict(method_model),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


class RunStateTest(unittest.TestCase):
    def _load_run_state_module(self):
        import movebench.bench.run_state as run_state_module

        return run_state_module

    def test_search_latest_run_records_reads_completed_run_metrics_without_mlflow_db(self):
        run_state_module = self._load_run_state_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            run_dir = repo_root / "results" / "bench" / "runs" / "run-a"
            _write_json(run_dir / "resolved_run.json", _resolved_payload(run_key="run-a"))
            _write_json(
                run_dir / "training_summary.json",
                {
                    "status": "completed",
                    "train_steps": 17,
                    "early_stop_split_key": "test_temporal",
                },
            )
            _write_json(
                run_dir / "final_metrics" / "test_temporal" / "summary.json",
                {
                    "energy_score": 1.25,
                    "accuracy": 0.75,
                    "calibration": 0.5,
                    "n_steps_scored": 12,
                    "n_steps_excluded_out_of_support": 3,
                },
            )

            with mock.patch.object(run_state_module.paths, "repo_root", return_value=repo_root):
                records = run_state_module.search_latest_run_records_by_key(["run-a"])

        self.assertEqual(records["run-a"]["status"], "completed")
        self.assertEqual(records["run-a"]["tags"]["run_key"], "run-a")
        self.assertEqual(records["run-a"]["tags"]["study_id"], "study-a")
        self.assertEqual(records["run-a"]["params"]["step_time_s"], "86400")
        self.assertEqual(records["run-a"]["metrics"]["energy_score.test_temporal"], 1.25)
        self.assertEqual(
            records["run-a"]["metrics"]["energy_score.accuracy.test_temporal"],
            0.75,
        )
        self.assertEqual(
            records["run-a"]["metrics"]["energy_score.calibration.test_temporal"],
            0.5,
        )
        self.assertEqual(records["run-a"]["metrics"]["n_steps_scored.test_temporal"], 12.0)
        self.assertEqual(
            records["run-a"]["metrics"]["n_steps_excluded_out_of_support.test_temporal"],
            3.0,
        )

    def test_search_latest_run_records_classifies_blocked_failed_and_interrupted_runs(self):
        run_state_module = self._load_run_state_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            blocked_dir = repo_root / "results" / "bench" / "runs" / "blocked-run"
            failed_dir = repo_root / "results" / "bench" / "runs" / "failed-run"
            interrupted_dir = repo_root / "results" / "bench" / "runs" / "interrupted-run"
            _write_json(
                blocked_dir / "resolved_run.json",
                _resolved_payload(run_key="blocked-run"),
            )
            _write_json(
                blocked_dir / "training_summary.json",
                {
                    "status": "blocked",
                    "reason": "missing point store manifest for train",
                    "early_stop_split_key": "test_temporal",
                },
            )
            _write_json(
                failed_dir / "resolved_run.json",
                _resolved_payload(run_key="failed-run"),
            )
            _write_json(
                failed_dir / "training_summary.json",
                {
                    "status": "failed",
                    "reason": "CUDA out of memory",
                    "early_stop_split_key": "test_temporal",
                },
            )
            _write_json(
                interrupted_dir / "resolved_run.json",
                _resolved_payload(run_key="interrupted-run"),
            )
            _write_json(
                interrupted_dir / "training_summary.json",
                {
                    "status": "interrupted",
                    "reason": "SIGTERM",
                    "early_stop_split_key": "test_temporal",
                },
            )

            with mock.patch.object(run_state_module.paths, "repo_root", return_value=repo_root):
                records = run_state_module.search_latest_run_records_by_key(
                    ["blocked-run", "failed-run", "interrupted-run"]
                )

        self.assertEqual(records["blocked-run"]["status"], "blocked")
        self.assertEqual(
            records["blocked-run"]["tags"]["blocked_reason"],
            "missing point store manifest for train",
        )
        self.assertEqual(records["failed-run"]["status"], "failed")
        self.assertEqual(
            records["failed-run"]["tags"]["failure_reason"],
            "CUDA out of memory",
        )
        self.assertEqual(records["interrupted-run"]["status"], "failed")
        self.assertEqual(records["interrupted-run"]["tags"]["failure_reason"], "SIGTERM")
        self.assertEqual(records["interrupted-run"]["tags"]["interrupted"], "true")

    def test_search_latest_run_records_distinguishes_running_stale_incomplete_and_missing_runs(self):
        run_state_module = self._load_run_state_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            running_dir = repo_root / "results" / "bench" / "runs" / "running-run"
            stale_dir = repo_root / "results" / "bench" / "runs" / "stale-run"
            incomplete_dir = repo_root / "results" / "bench" / "runs" / "incomplete-run"
            _write_json(
                running_dir / "resolved_run.json",
                _resolved_payload(run_key="running-run"),
            )
            _write_json(
                running_dir / "active_run.json",
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "slurm_job_id": "",
                },
            )
            _write_json(
                stale_dir / "resolved_run.json",
                _resolved_payload(run_key="stale-run"),
            )
            _write_json(
                stale_dir / "active_run.json",
                {
                    "pid": os.getpid(),
                    "hostname": "different-host",
                    "slurm_job_id": "",
                },
            )
            _write_json(
                incomplete_dir / "resolved_run.json",
                _resolved_payload(run_key="incomplete-run"),
            )
            pred_path = incomplete_dir / "predictions" / "test_temporal.pt"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_path.write_text("prediction", encoding="utf-8")

            with mock.patch.object(run_state_module.paths, "repo_root", return_value=repo_root):
                records = run_state_module.search_latest_run_records_by_key(
                    ["running-run", "stale-run", "incomplete-run", "missing-run"]
                )

        self.assertEqual(records["running-run"]["status"], "running")
        self.assertEqual(records["running-run"]["tags"]["status"], "running")
        self.assertEqual(records["stale-run"]["status"], "failed")
        self.assertEqual(
            records["stale-run"]["tags"]["failure_reason"],
            "stale interrupted run",
        )
        self.assertEqual(records["incomplete-run"]["status"], "failed")
        self.assertEqual(
            records["incomplete-run"]["tags"]["failure_reason"],
            "incomplete run missing training_summary",
        )
        self.assertNotIn("missing-run", records)

    def test_search_latest_run_records_preserves_unknown_status_strings(self):
        run_state_module = self._load_run_state_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            run_dir = repo_root / "results" / "bench" / "runs" / "unknown-run"
            _write_json(
                run_dir / "resolved_run.json",
                _resolved_payload(run_key="unknown-run"),
            )
            _write_json(
                run_dir / "training_summary.json",
                {
                    "status": "mystery_state",
                    "reason": "unexpected state",
                    "early_stop_split_key": "test_temporal",
                },
            )

            with mock.patch.object(run_state_module.paths, "repo_root", return_value=repo_root):
                records = run_state_module.search_latest_run_records_by_key(["unknown-run"])

        self.assertEqual(records["unknown-run"]["status"], "mystery_state")
        self.assertEqual(records["unknown-run"]["tags"]["status"], "mystery_state")

