from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class BenchReportTest(unittest.TestCase):
    def test_write_report_reads_completed_runs_from_local_files(self) -> None:
        from movebench.bench import report as report_module

        status_snapshot = {
            "counts": {"studies": 1},
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
                "experiment": {"name": "main_results", "blocked": []},
                "runs": [
                    {
                        "run_key": "run-a",
                        "study_id": "study-a",
                        "timescale": "1h",
                        "method": "issf_linear",
                        "covariate_set": "all",
                        "seed": 0,
                        "evaluation_split_keys": ["test_temporal"],
                    }
                ],
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            run_dir = repo_root / "results" / "bench" / "runs" / "run-a"
            (run_dir / "final_metrics" / "test_temporal").mkdir(parents=True)
            (run_dir / "resolved_run.json").write_text(
                json.dumps(
                    {
                        "run_key": "run-a",
                        "study_id": "study-a",
                        "timescale": "1h",
                        "method": "issf_linear",
                        "covariate_set": "all",
                        "seed": 0,
                        "step_time_s": 3600,
                        "evaluation_split_keys": ["test_temporal"],
                        "early_stop_split_key": "test_temporal",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "train_steps": 4,
                        "early_stop_split_key": "test_temporal",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "final_metrics" / "test_temporal" / "summary.json").write_text(
                json.dumps(
                    {
                        "split_key": "test_temporal",
                        "prediction_kind": "grid",
                        "n_steps_total": 3,
                        "n_steps_scored": 2,
                        "n_steps_excluded_out_of_support": 1,
                        "support_buffer_m": 100.0,
                        "support_pixel_size_m": 25.0,
                        "energy_score": 0.75,
                        "accuracy": 0.5,
                        "calibration": 0.25,
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(report_module.paths, "repo_root", return_value=repo_root),
                mock.patch.object(
                    report_module,
                    "_expanded_checked_in_experiments",
                    return_value=expanded_experiments,
                ),
            ):
                result = report_module.write_report(
                    status_snapshot=status_snapshot,
                    refresh_metrics=False,
                    allow_metric_repair=False,
                )
                index_exists = Path(result["index_path"]).exists()

            status_report = json.loads(
                (repo_root / "results" / "bench" / "report" / "status_report.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(index_exists)
        self.assertEqual(status_report["counts"]["completed_runs"], 1)
        self.assertEqual(status_report["counts"]["pending_runs"], 0)
        self.assertEqual(status_report["counts"]["main_results_plots"], 1)
        self.assertEqual(status_report["planned_runs"][0]["latest_status"], "completed")
        self.assertEqual(
            status_report["planned_runs"][0]["missing_final_metrics_splits"],
            [],
        )
        self.assertEqual(status_report["missing_artifacts"], [])


if __name__ == "__main__":
    unittest.main()
