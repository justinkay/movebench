from __future__ import annotations

import argparse
import contextlib
import gzip
import io
import json
import os
import socket
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from movebench.features.points import resolve_sampler_hash
from tests.support import (
    make_plan,
    make_sampler_config,
    write_schema4_point_store,
    write_train_stats,
    write_trajectories_csv,
)


def _fake_mlflow_module() -> types.ModuleType:
    module = types.ModuleType("mlflow")
    module.set_tracking_uri = lambda uri: None
    module.start_run = lambda **kwargs: None
    module.log_params = lambda params: None
    module.tracking = types.SimpleNamespace(MlflowClient=lambda **kwargs: None)
    return module


def _load_run_module():
    with mock.patch.dict(sys.modules, {"mlflow": _fake_mlflow_module()}):
        sys.modules.pop("movebench.bench.mlflow", None)
        sys.modules.pop("movebench.bench.run", None)
        import movebench.bench.run as run_module

    return run_module


def _load_evaluation_runner_module():
    with mock.patch.dict(sys.modules, {"mlflow": _fake_mlflow_module()}):
        sys.modules.pop("movebench.bench.mlflow", None)
        sys.modules.pop("movebench.bench.evaluation.runner", None)
        import movebench.bench.evaluation.runner as runner_module

    return runner_module


def _latest_record(
    status: str,
    *,
    tags: dict[str, str] | None = None,
) -> dict[str, object]:
    return {"status": status, "tags": dict(tags or {})}


class RunnerPointArtifactsTest(unittest.TestCase):
    def test_supported_methods_tracks_method_registry(self):
        run_module = _load_run_module()

        self.assertEqual(
            run_module._SUPPORTED_METHODS,
            {
                str(method_key)
                for method_key, method_cfg in run_module.bench_config.METHODS.items()
                if bool(method_cfg.get("executor"))
            },
        )
        self.assertIn("issf_linear_time", run_module._SUPPORTED_METHODS)
        self.assertIn("issf_mlp_time", run_module._SUPPORTED_METHODS)

    def test_resolve_point_artifacts_accepts_schema_4_store(self):
        run_module = _load_run_module()
        plan = make_plan()
        sampler_config = make_sampler_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            study_dir = data_root / "studies" / "demo"
            study_dir.mkdir(parents=True)
            row_specs = [
                {
                    "point_id_start": 100,
                    "point_id_end": 101,
                    "start_lon": 0.01,
                    "start_lat": 0.02,
                    "individual_id": "alpha",
                    "burst_id": 1,
                    "timestamp_ns": 1_704_067_200_000_000_000,
                },
                {
                    "point_id_start": 102,
                    "point_id_end": 103,
                    "start_lon": 0.02,
                    "start_lat": 0.03,
                    "individual_id": "alpha",
                    "burst_id": 1,
                    "timestamp_ns": 1_704_070_800_000_000_000,
                },
            ]
            write_trajectories_csv(study_dir, row_specs=row_specs, split="train")
            write_schema4_point_store(
                study_dir=study_dir,
                split="train",
                plan=plan,
                sampler_config=sampler_config,
                row_specs=row_specs,
            )
            write_schema4_point_store(
                study_dir=study_dir,
                split="test_temporal",
                plan=plan,
                sampler_config=sampler_config,
                row_specs=row_specs,
            )
            write_train_stats(
                study_dir=study_dir,
                plan=plan,
                sampler_config=sampler_config,
                stats={
                    band_id: {"mean": 0.0, "std": 1.0, "count": 1}
                    for band_id in plan.band_ids_in_order()
                },
            )

            run_payload = {
                "study_id": "demo",
                "method": "issf_linear",
                "rasters_hash": str(plan.rasters_hash),
                "sampler_hash": str(resolve_sampler_hash(sampler_config)),
                "evaluation_split_keys": ["test_temporal"],
                "band_ids": plan.band_ids_in_order(),
            }
            method = {"requires_normalization_stats": True}
            timescale = {
                "plan": plan,
                "sampler": sampler_config,
                "all_band_ids": plan.band_ids_in_order(),
            }

            with mock.patch.dict(os.environ, {"MOVEBENCH_DATA_ROOT": str(data_root)}):
                resolved = run_module._resolve_point_artifacts(
                    run_payload,
                    method=method,
                    timescale=timescale,
                )

        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(
            resolved["point_stores"]["train"]["manifest"].schema_version,
            "4",
        )

    def test_resolve_point_artifacts_rejects_missing_schema_4_payload(self):
        run_module = _load_run_module()
        plan = make_plan()
        sampler_config = make_sampler_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            study_dir = data_root / "studies" / "demo"
            study_dir.mkdir(parents=True)
            write_trajectories_csv(
                study_dir,
                row_specs=[
                    {
                        "point_id_start": 100,
                        "point_id_end": 101,
                        "start_lon": 0.01,
                        "start_lat": 0.02,
                        "individual_id": "alpha",
                        "burst_id": 1,
                        "timestamp_ns": 1_704_067_200_000_000_000,
                    }
                ],
                split="train",
            )
            fixture = write_schema4_point_store(
                study_dir=study_dir,
                split="train",
                plan=plan,
                sampler_config=sampler_config,
            )
            write_schema4_point_store(
                study_dir=study_dir,
                split="test_temporal",
                plan=plan,
                sampler_config=sampler_config,
            )
            write_train_stats(
                study_dir=study_dir,
                plan=plan,
                sampler_config=sampler_config,
                stats={
                    band_id: {"mean": 0.0, "std": 1.0, "count": 1}
                    for band_id in plan.band_ids_in_order()
                },
            )
            (Path(fixture["store_dir"]) / "shared" / "shared.npy").unlink()

            run_payload = {
                "study_id": "demo",
                "method": "issf_linear",
                "rasters_hash": str(plan.rasters_hash),
                "sampler_hash": str(resolve_sampler_hash(sampler_config)),
                "evaluation_split_keys": ["test_temporal"],
                "band_ids": plan.band_ids_in_order(),
            }
            method = {"requires_normalization_stats": True}
            timescale = {
                "plan": plan,
                "sampler": sampler_config,
                "all_band_ids": plan.band_ids_in_order(),
            }

            with mock.patch.dict(os.environ, {"MOVEBENCH_DATA_ROOT": str(data_root)}):
                resolved = run_module._resolve_point_artifacts(
                    run_payload,
                    method=method,
                    timescale=timescale,
                )

        self.assertEqual(resolved["status"], "blocked")
        self.assertIn("invalid point store manifest", resolved["reason"])


class EvaluationRunnerGridArtifactsTest(unittest.TestCase):
    def test_evaluate_run_dir_scores_deepssf_style_grid_artifact(self):
        runner_module = _load_evaluation_runner_module()
        scores = torch.tensor([[[1.0, 0.0], [2.0, -1.0]]], dtype=torch.float32)
        truth_xy_m = torch.tensor([0.5, -0.25], dtype=torch.float64)
        x_coords_m, y_coords_m = runner_module._grid_coordinates(
            width=2,
            height=2,
            pixel_size_m=2.0,
        )
        expected_probs_hw = torch.softmax(
            scores[0].reshape(-1).to(dtype=torch.float64),
            dim=0,
        ).reshape(2, 2)
        expected_score, expected_accuracy, expected_calibration = (
            runner_module.energy._energy_terms_grid(
                {
                    "probs_hw": expected_probs_hw,
                    "x_coords_m": x_coords_m,
                    "y_coords_m": y_coords_m,
                },
                truth_xy_m,
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
            (run_dir / "resolved_run.json").write_text(
                json.dumps(
                    {
                        "run_key": "grid-demo",
                        "study_id": "demo",
                        "timescale": "tiny",
                        "rasters_hash": "rasters",
                        "sampler_hash": "sampler",
                        "evaluation_split_keys": ["test_temporal"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "training_summary.json").write_text(
                json.dumps({"status": "completed"}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "prediction_kind": "grid",
                    "point_id_start": torch.tensor([100], dtype=torch.int64),
                    "scores": scores,
                    "pixel_size_m": 2.0,
                },
                run_dir / "predictions" / "test_temporal.pt",
            )

            with (
                mock.patch.object(
                    runner_module,
                    "_support_rows_for_split",
                    return_value=(
                        [
                            {
                                "step_index": 0,
                                "point_id_start": 100,
                                "point_id_end": 101,
                                "truth_xy_m": truth_xy_m,
                                "in_support": True,
                            }
                        ],
                        1.0,
                    ),
                ),
                mock.patch.object(
                    runner_module,
                    "_selected_splits_for_split",
                    return_value=None,
                ),
                mock.patch.object(
                    runner_module.bench_config,
                    "resolve_timescale",
                    return_value={"output_hw": (2, 2)},
                ),
            ):
                result = runner_module.evaluate_run_dir(run_dir)

            summary_path = run_dir / "final_metrics" / "test_temporal" / "summary.json"
            rows_path = (
                run_dir / "final_metrics" / "test_temporal" / "energy_by_step.csv.gz"
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with gzip.open(rows_path, "rt", encoding="utf-8") as handle:
                rows = [line.strip() for line in handle.readlines()]

        self.assertEqual(result["splits"]["test_temporal"]["prediction_kind"], "grid")
        self.assertEqual(summary["prediction_kind"], "grid")
        self.assertEqual(summary["n_steps_scored"], 1)
        self.assertAlmostEqual(summary["energy_score"], expected_score)
        self.assertAlmostEqual(summary["accuracy"], expected_accuracy)
        self.assertAlmostEqual(summary["calibration"], expected_calibration)
        self.assertEqual(len(rows), 2)


class RunnerSharedHelpersTest(unittest.TestCase):
    def test_resolve_studies_accepts_named_study_set_for_experiment_and_override(self):
        run_module = _load_run_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            studies_root = Path(tmpdir) / "studies"
            for study_id in (
                "forrest2024_buffalo",
                "Accio_967b7813_raw_merged",
                "study-z",
            ):
                (studies_root / study_id).mkdir(parents=True, exist_ok=True)

            with mock.patch.dict(os.environ, {"MOVEBENCH_DATA_ROOT": str(Path(tmpdir))}):
                self.assertEqual(
                    run_module._resolve_studies({"studies": "mini"}),
                    [
                        "forrest2024_buffalo",
                        "Accio_967b7813_raw_merged",
                    ],
                )
                self.assertEqual(
                    run_module._resolve_studies(
                        {"studies": "all"},
                        studies_override=["mini"],
                    ),
                    [
                        "forrest2024_buffalo",
                        "Accio_967b7813_raw_merged",
                    ],
                )

    def test_expand_experiment_builds_moveformer_run_payload(self):
        run_module = _load_run_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir)
            (data_root / "studies" / "demo").mkdir(parents=True, exist_ok=True)
            experiment = {
                "name": "demo",
                "studies": ["demo"],
                "timescales": ["1h"],
                "covariate_sets": ["all"],
                "methods": ["moveformer"],
                "seed": 0,
            }
            readiness = {
                "available_point_splits": ["test_temporal"],
                "early_stop_point_split": "test_temporal",
                "point_ready": True,
            }
            fake_timescale = {
                "step_time_s": 3600,
                "layers_config": "configs/layers/all_ae_1h_1h.yaml",
                "rasters_hash": "rh-demo",
                "sampler_config": "configs/samplers/issf_defaults.yaml",
                "sampler_hash": "sh-demo",
            }

            with (
                mock.patch.dict(os.environ, {"MOVEBENCH_DATA_ROOT": str(data_root)}),
                mock.patch.object(
                    run_module,
                    "_study_timescale_readiness",
                    return_value=readiness,
                ),
                mock.patch.object(
                    run_module,
                    "_resolve_artifacts_for_run",
                    return_value={"status": "resolved"},
                ),
                mock.patch.object(
                    run_module.bench_config,
                    "resolve_covariate_band_ids",
                    return_value=["layer_a.band_0"],
                ),
                mock.patch.object(
                    run_module.bench_config,
                    "resolve_timescale",
                    return_value=fake_timescale,
                ),
            ):
                runs = run_module.expand_experiment(experiment)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["method"], "moveformer")
        self.assertNotIn("moveformer_context_version", runs[0])

    def test_expand_experiment_resolves_layers_per_study(self):
        run_module = _load_run_module()
        experiment = {
            "name": "demo",
            "studies": ["study-a", "costa2024_elephant_seals"],
            "timescales": ["1d"],
            "covariate_sets": ["all"],
            "methods": ["issf_linear"],
            "seed": 0,
        }
        readiness = {
            "available_point_splits": ["test_temporal"],
            "early_stop_point_split": "test_temporal",
            "point_ready": True,
        }

        def fake_resolve_covariate_band_ids(timescale_key, covariate_set_key, study_id):
            self.assertEqual(timescale_key, "1d")
            self.assertEqual(covariate_set_key, "all")
            return [f"{study_id}.band"]

        def fake_resolve_timescale(timescale_key, study_id):
            self.assertEqual(timescale_key, "1d")
            return {
                "step_time_s": 86400,
                "layers_config": f"configs/layers/{study_id}.yaml",
                "rasters_hash": f"{study_id}-rasters",
                "sampler_config": "configs/samplers/issf_defaults.yaml",
                "sampler_hash": f"{study_id}-sampler",
            }

        with (
            mock.patch.object(
                run_module,
                "_study_timescale_readiness",
                return_value=readiness,
            ),
            mock.patch.object(
                run_module,
                "_resolve_artifacts_for_run",
                return_value={"status": "resolved"},
            ),
            mock.patch.object(
                run_module.bench_config,
                "resolve_covariate_band_ids",
                side_effect=fake_resolve_covariate_band_ids,
            ),
            mock.patch.object(
                run_module.bench_config,
                "resolve_timescale",
                side_effect=fake_resolve_timescale,
            ),
        ):
            runs = run_module.expand_experiment(experiment)

        self.assertEqual(
            {run["study_id"]: run["layers_config"] for run in runs},
            {
                "study-a": "configs/layers/study-a.yaml",
                "costa2024_elephant_seals": "configs/layers/costa2024_elephant_seals.yaml",
            },
        )
        self.assertEqual(
            {run["study_id"]: run["band_ids"] for run in runs},
            {
                "study-a": ["study-a.band"],
                "costa2024_elephant_seals": ["costa2024_elephant_seals.band"],
            },
        )

    def test_resolve_pending_runs_filters_latest_runs_and_builds_manifest_payload(self):
        run_module = _load_run_module()
        experiment = {"name": "demo", "methods": ["issf_linear"], "blocked": []}
        planned_runs = [
            {"run_key": "run-a", "method": "issf_linear"},
            {"run_key": "run-b", "method": "deepssf"},
            {"run_key": "run-c", "method": "moveformer"},
        ]

        def _expand(exp: dict[str, object], studies_override=None):
            exp["blocked"] = [{"reason": "missing prerequisites"}]
            return [dict(run) for run in planned_runs]

        with (
            mock.patch.object(run_module, "load_experiment", return_value=experiment),
            mock.patch.object(run_module, "expand_experiment", side_effect=_expand),
            mock.patch.object(
                run_module.bench_run_state,
                "search_latest_run_records_by_key",
                return_value={
                    "run-a": _latest_record("completed"),
                    "run-b": _latest_record("blocked"),
                },
                create=True,
            ) as search_latest_runs,
        ):
            resolved = run_module.resolve_pending_runs(
                "experiments/demo.py",
                studies_override=["study-a"],
                methods=["issf_linear"],
                max_runs=1,
                rerun=False,
            )

        search_latest_runs.assert_called_once_with(["run-a", "run-b", "run-c"])
        self.assertEqual([run["run_key"] for run in resolved["planned_runs"]], ["run-a", "run-b", "run-c"])
        self.assertEqual([run["run_key"] for run in resolved["pending_runs"]], ["run-b"])
        self.assertEqual(resolved["counts"]["planned_runs"], 4)
        self.assertEqual(resolved["counts"]["pending_runs"], 1)
        self.assertEqual(resolved["counts"]["skipped_runs"], 1)
        self.assertEqual(resolved["counts"]["blocked_runs"], 1)
        self.assertIn("results/bench/manifests", str(resolved["manifest_path"]))
        self.assertEqual(resolved["manifest_payload"]["pending_run_keys"], ["run-b"])

    def test_resolve_pending_runs_keeps_live_running_run_skipped(self):
        run_module = _load_run_module()
        experiment = {"name": "demo", "methods": ["deepssf"], "blocked": []}
        planned_runs = [{"run_key": "run-a", "method": "deepssf"}]

        def _expand(exp: dict[str, object], studies_override=None):
            del studies_override
            exp["blocked"] = []
            return [dict(run) for run in planned_runs]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "active_run.json").write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "hostname": socket.gethostname(),
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(run_module, "load_experiment", return_value=experiment),
                mock.patch.object(run_module, "expand_experiment", side_effect=_expand),
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module.bench_run_state,
                    "search_latest_run_records_by_key",
                    return_value={"run-a": _latest_record("running")},
                    create=True,
                ),
            ):
                resolved = run_module.resolve_pending_runs(
                    "experiments/demo.py",
                    rerun=False,
                )

        self.assertEqual([run["run_key"] for run in resolved["pending_runs"]], [])
        self.assertEqual(resolved["counts"]["skipped_runs"], 1)

    def test_resolve_pending_runs_requeues_stale_running_run(self):
        run_module = _load_run_module()
        experiment = {"name": "demo", "methods": ["deepssf"], "blocked": []}
        planned_runs = [{"run_key": "run-a", "method": "deepssf"}]

        def _expand(exp: dict[str, object], studies_override=None):
            del studies_override
            exp["blocked"] = []
            return [dict(run) for run in planned_runs]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "active_run.json").write_text(
                json.dumps(
                    {
                        "pid": 999999,
                        "hostname": socket.gethostname(),
                    }
                ),
                encoding="utf-8",
            )

            with (
                mock.patch.object(run_module, "load_experiment", return_value=experiment),
                mock.patch.object(run_module, "expand_experiment", side_effect=_expand),
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module.bench_run_state,
                    "search_latest_run_records_by_key",
                    return_value={
                        "run-a": _latest_record(
                            "failed",
                            tags={
                                "failure_reason": "stale interrupted run",
                                "interrupted": "true",
                            },
                        )
                    },
                    create=True,
                ),
            ):
                resolved = run_module.resolve_pending_runs(
                    "experiments/demo.py",
                    rerun=False,
                )

        self.assertEqual([run["run_key"] for run in resolved["pending_runs"]], ["run-a"])
        self.assertEqual(resolved["counts"]["skipped_runs"], 0)
        self.assertFalse((run_dir / "active_run.json").exists())

    def test_load_resolved_run_reads_manifest_by_run_key(self):
        run_module = _load_run_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "planned_runs": [
                            {"run_key": "run-a", "method": "issf_linear"},
                            {"run_key": "run-b", "method": "deepssf"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            resolved = run_module.load_resolved_run(manifest_path, "run-b")

        self.assertEqual(resolved["run_key"], "run-b")
        self.assertEqual(resolved["method"], "deepssf")

    def test_execute_run_payload_resumes_matching_issf_run_dir(self):
        run_module = _load_run_module()
        run_payload = {
            "run_key": "run-a",
            "method": "issf_linear",
            "executor": "movebench.bench.executors.issf:run_linear",
            "early_stop_split_key": "test_temporal",
            "covariate_set": "all",
        }
        observed: dict[str, bool] = {}

        def _fake_executor(**kwargs):
            run_dir = Path(kwargs["run_dir"])
            observed["checkpoint_exists"] = (run_dir / "checkpoint_best.pt").exists()
            observed["history_exists"] = (run_dir / "history.json").exists()
            observed["predictions_exist"] = (
                run_dir / "predictions" / "test_temporal.pt"
            ).exists()
            observed["final_metrics_removed"] = not (run_dir / "final_metrics").exists()
            return {
                "status": "completed",
                "train_steps": 1,
                "best_step": 1,
                "early_stop_split_key": "test_temporal",
                "train_metrics": {},
                "eval_metrics": {},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "resolved_run.json").write_text(
                json.dumps(run_module._canonical_run_payload(run_payload)),
                encoding="utf-8",
            )
            (run_dir / "checkpoint_best.pt").write_text("checkpoint", encoding="utf-8")
            (run_dir / "history.json").write_text("[]", encoding="utf-8")
            (run_dir / "predictions").mkdir()
            (run_dir / "predictions" / "test_temporal.pt").write_text(
                "prediction",
                encoding="utf-8",
            )
            (run_dir / "final_metrics" / "test_temporal").mkdir(parents=True)
            fake_eval = types.SimpleNamespace(evaluate_run_dir=mock.Mock(return_value={}))

            with (
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module,
                    "_resolve_artifacts_for_run",
                    return_value={"status": "resolved"},
                ),
                mock.patch.object(
                    run_module,
                    "_load_executor",
                    return_value=_fake_executor,
                ),
                mock.patch.object(
                    run_module,
                    "_evaluation_runner_module",
                    return_value=fake_eval,
                ),
            ):
                summary = run_module.execute_run_payload(run_payload)

            self.assertEqual(summary["status"], "completed")
            self.assertTrue(observed["checkpoint_exists"])
            self.assertTrue(observed["history_exists"])
            self.assertTrue(observed["predictions_exist"])
            self.assertTrue(observed["final_metrics_removed"])
            self.assertTrue((run_dir / "checkpoint_best.pt").exists())
            self.assertTrue((run_dir / "history.json").exists())
            self.assertTrue((run_dir / "predictions" / "test_temporal.pt").exists())
            fake_eval.evaluate_run_dir.assert_called_once()

    def test_execute_run_payload_clears_runtime_outputs_when_resume_disabled(self):
        run_module = _load_run_module()
        run_payload = {
            "run_key": "run-a",
            "method": "issf_linear",
            "executor": "movebench.bench.executors.issf:run_linear",
            "early_stop_split_key": "test_temporal",
            "covariate_set": "all",
        }
        observed: dict[str, bool] = {}

        def _fake_executor(**kwargs):
            run_dir = Path(kwargs["run_dir"])
            observed["checkpoint_removed"] = not (run_dir / "checkpoint_best.pt").exists()
            observed["history_removed"] = not (run_dir / "history.json").exists()
            observed["predictions_removed"] = not (run_dir / "predictions").exists()
            return {
                "status": "completed",
                "train_steps": 1,
                "best_step": 1,
                "early_stop_split_key": "test_temporal",
                "train_metrics": {},
                "eval_metrics": {},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "resolved_run.json").write_text(
                json.dumps(run_module._canonical_run_payload(run_payload)),
                encoding="utf-8",
            )
            (run_dir / "checkpoint_best.pt").write_text("checkpoint", encoding="utf-8")
            (run_dir / "history.json").write_text("[]", encoding="utf-8")
            (run_dir / "predictions").mkdir()
            (run_dir / "predictions" / "test_temporal.pt").write_text(
                "prediction",
                encoding="utf-8",
            )
            fake_eval = types.SimpleNamespace(evaluate_run_dir=mock.Mock(return_value={}))

            with (
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module,
                    "_resolve_artifacts_for_run",
                    return_value={"status": "resolved"},
                ),
                mock.patch.object(
                    run_module,
                    "_load_executor",
                    return_value=_fake_executor,
                ),
                mock.patch.object(
                    run_module,
                    "_evaluation_runner_module",
                    return_value=fake_eval,
                ),
            ):
                summary = run_module.execute_run_payload(run_payload, resume=False)

            self.assertEqual(summary["status"], "completed")
            self.assertTrue(observed["checkpoint_removed"])
            self.assertTrue(observed["history_removed"])
            self.assertTrue(observed["predictions_removed"])

    def test_execute_run_payload_clears_runtime_outputs_on_payload_mismatch(self):
        run_module = _load_run_module()
        run_payload = {
            "run_key": "run-a",
            "method": "issf_linear",
            "executor": "movebench.bench.executors.issf:run_linear",
            "early_stop_split_key": "test_temporal",
            "covariate_set": "all",
        }
        observed: dict[str, bool] = {}

        def _fake_executor(**kwargs):
            run_dir = Path(kwargs["run_dir"])
            observed["checkpoint_removed"] = not (run_dir / "checkpoint_best.pt").exists()
            observed["history_removed"] = not (run_dir / "history.json").exists()
            observed["predictions_removed"] = not (run_dir / "predictions").exists()
            return {
                "status": "completed",
                "train_steps": 1,
                "best_step": 1,
                "early_stop_split_key": "test_temporal",
                "train_metrics": {},
                "eval_metrics": {},
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "resolved_run.json").write_text(
                json.dumps(
                    run_module._canonical_run_payload(
                        {**run_payload, "covariate_set": "lite"}
                    )
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoint_best.pt").write_text("checkpoint", encoding="utf-8")
            (run_dir / "history.json").write_text("[]", encoding="utf-8")
            (run_dir / "predictions").mkdir()
            (run_dir / "predictions" / "test_temporal.pt").write_text(
                "prediction",
                encoding="utf-8",
            )
            fake_eval = types.SimpleNamespace(evaluate_run_dir=mock.Mock(return_value={}))

            with (
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module,
                    "_resolve_artifacts_for_run",
                    return_value={"status": "resolved"},
                ),
                mock.patch.object(
                    run_module,
                    "_load_executor",
                    return_value=_fake_executor,
                ),
                mock.patch.object(
                    run_module,
                    "_evaluation_runner_module",
                    return_value=fake_eval,
                ),
            ):
                summary = run_module.execute_run_payload(run_payload)

            self.assertEqual(summary["status"], "completed")
            self.assertTrue(observed["checkpoint_removed"])
            self.assertTrue(observed["history_removed"])
            self.assertTrue(observed["predictions_removed"])

    def test_execute_run_payload_marks_keyboard_interrupt_as_interrupted(self):
        run_module = _load_run_module()
        run_payload = {
            "run_key": "run-a",
            "method": "deepssf",
            "executor": "movebench.bench.executors.deepssf:run",
            "early_stop_split_key": "test_temporal",
            "covariate_set": "all",
        }
        observed: dict[str, bool] = {}

        def _fake_executor(**kwargs):
            run_dir = Path(kwargs["run_dir"])
            observed["active_exists"] = (run_dir / "active_run.json").exists()
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-a"

            with (
                mock.patch.object(run_module, "_run_dir", return_value=run_dir),
                mock.patch.object(
                    run_module,
                    "_resolve_artifacts_for_run",
                    return_value={"status": "resolved"},
                ),
                mock.patch.object(
                    run_module,
                    "_load_executor",
                    return_value=_fake_executor,
                ),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_module.execute_run_payload(run_payload)

            summary = json.loads(
                (run_dir / "training_summary.json").read_text(encoding="utf-8")
            )

        self.assertTrue(observed["active_exists"])
        self.assertEqual(summary["status"], "interrupted")
        self.assertFalse((run_dir / "active_run.json").exists())

    def test_run_experiment_uses_shared_pending_run_helpers(self):
        run_module = _load_run_module()
        resolved = {
            "planned_runs": [{"run_key": "run-a"}, {"run_key": "run-b"}],
            "pending_runs": [{"run_key": "run-a"}, {"run_key": "run-b"}],
            "counts": {
                "planned_runs": 3,
                "pending_runs": 2,
                "skipped_runs": 1,
                "blocked_runs": 1,
            },
        }

        with (
            mock.patch.object(
                run_module,
                "resolve_pending_runs",
                return_value=resolved,
                create=True,
            ) as resolve_pending_runs,
            mock.patch.object(
                run_module,
                "execute_run_payload",
                side_effect=[
                    {"status": "completed"},
                    {"status": "failed"},
                ],
                create=True,
            ) as execute_run_payload,
        ):
            summary = run_module.run_experiment(
                "experiments/demo.py",
                studies_override=["study-a"],
                methods=["issf_linear"],
                max_runs=2,
                rerun=True,
            )

        resolve_pending_runs.assert_called_once_with(
            "experiments/demo.py",
            studies_override=["study-a"],
            methods=["issf_linear"],
            max_runs=2,
            rerun=True,
        )
        self.assertEqual(
            [call.args[0]["run_key"] for call in execute_run_payload.call_args_list],
            ["run-a", "run-b"],
        )
        self.assertEqual(
            [call.kwargs for call in execute_run_payload.call_args_list],
            [{"resume": False}, {"resume": False}],
        )
        self.assertEqual(
            summary,
            {
                "planned_runs": 3,
                "executed_runs": 2,
                "skipped_runs": 1,
                "blocked_runs": 1,
                "failed_runs": 1,
            },
        )


class RunnerCommandTest(unittest.TestCase):
    def test_bench_status_prints_all_and_mini_by_default(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(["bench", "status"])
        stdout = io.StringIO()
        observed: list[list[str] | None] = []

        def _fake_status_snapshot(
            studies_override: list[str] | None = None,
        ) -> dict[str, object]:
            observed.append(None if studies_override is None else list(studies_override))
            counts = {"studies": 3 if studies_override == ["all"] else 2}
            return {"counts": counts, "studies": []}

        def _fake_render(snapshot: dict[str, object]) -> str:
            return f"Studies: {snapshot['counts']['studies']}"

        with (
            mock.patch.object(run_module, "status_snapshot", side_effect=_fake_status_snapshot),
            mock.patch.object(run_module, "_render_status_text", side_effect=_fake_render),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        self.assertEqual(observed, [["all"], ["mini"]])
        self.assertEqual(
            stdout.getvalue().strip(),
            "Study selection: all\nStudies: 3\n\nStudy selection: mini\nStudies: 2",
        )

    def test_bench_status_studies_override_prints_only_requested_selection(self) -> None:
        run_module = _load_run_module()
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        run_module.register(subparsers)
        args = parser.parse_args(["bench", "status", "--studies", "mini"])
        stdout = io.StringIO()

        with (
            mock.patch.object(
                run_module,
                "status_snapshot",
                return_value={"counts": {"studies": 2}, "studies": []},
            ) as status_snapshot,
            mock.patch.object(run_module, "_render_status_text", return_value="Studies: 2"),
            contextlib.redirect_stdout(stdout),
        ):
            rc = args.func(args)

        self.assertEqual(rc, 0)
        status_snapshot.assert_called_once_with(studies_override=["mini"])
        self.assertEqual(stdout.getvalue().strip(), "Study selection: mini\nStudies: 2")


if __name__ == "__main__":
    unittest.main()
