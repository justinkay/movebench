import copy
import gzip
import json
import signal
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from movebench.bench import artifacts, config, planning, report, results, run, run_state, summary
from movebench.bench.evaluation import runner as evaluation
from movebench.datasets import DataSplit
from movebench.geometry import LOCAL_GEOMETRY
from movebench.slurm import bench as slurm_bench


class OriginMethod:
    def __init__(self, *, model_config, hparams, seed):
        pass

    def fit(self, train, validation, *, work_dir, restore_from=None):
        self.training_summary_ = {
            "train_steps": 7, "best_step": 3, "train_metrics": {"loss": 0.25},
        }
        return self

    def predict(self, data):
        return {
            "prediction_kind": "deterministic", "geometry": LOCAL_GEOMETRY,
            "point_id_start": torch.tensor([20, 10]),
            "dx_m": torch.zeros(2), "dy_m": torch.zeros(2),
        }


class BenchExecutionTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch.object(run.paths, "repo_root", return_value=self.root))
        self.entry = {
            "implementation": f"{__name__}:OriginMethod", "trains": False,
            "artifact_requirement": "points", "requires_normalization_stats": False,
            "fit_reuse": "rebuild", "prediction_kind": "deterministic", "gpus": 2,
            "model": {}, "hparams": {},
        }
        self.enterContext(patch.dict(config.METHODS, origin=self.entry))
        self.splits = {key: DataSplit(key, {}, {}) for key in (
            "train", "test_temporal", "test_individual",
        )}
        self.enterContext(patch.object(artifacts, "_resolve_artifacts_for_run", return_value={
            "status": "resolved", "splits": self.splits,
        }))
        self.timescale = {
            "step_time_s": 3600, "alphaearth_layers_config": "ae.yaml",
            "layers_config": "all.yaml", "rasters_hash": "rasters",
            "sampler_config": "sampler.yaml", "sampler_hash": "sampler",
            "geometry": LOCAL_GEOMETRY, "output_hw": (3, 3),
        }
        self.enterContext(patch.object(config, "resolve_timescale", return_value=self.timescale))
        self.enterContext(patch.object(evaluation, "_selected_splits_for_split", return_value=None))
        self.truth = [
            {"point_id_start": point, "step_index": i, "in_support": True,
             "truth_xy_m": torch.tensor([3., 4.], dtype=torch.float64)}
            for i, point in enumerate((10, 20))
        ]
        self.enterContext(patch.object(evaluation, "_support_rows_for_split", return_value=(self.truth, 3.)))
        self.payload = {
            "run_key": "origin", "method": "origin", "implementation": self.entry["implementation"],
            "seed": 0, "study_id": "study", "timescale": "1h", "covariate_set": "all",
            "geometry": LOCAL_GEOMETRY, "early_stop_split_key": "test_temporal",
            "evaluation_split_keys": ["test_temporal", "test_individual"],
        }
        self.directory = run_state.run_dir("origin")

    def summary(self):
        return run_state.load_training_summary(self.directory)

    def status(self):
        return run_state._load_run_record("origin")["status"]

    def execute(self):
        return run.execute_run_payload(self.payload)

    def refresh(self, *, force=False, allow=True):
        return report._refresh_final_metrics(
            {"origin": self.payload},
            latest_run_records=run_state.search_latest_run_records_by_key(["origin"]),
            refresh_metrics=force, allow_metric_repair=allow,
        )

    def test_registration_after_runner_import_plans_executes_and_loads_results(self):
        experiment = {"name": "contribution", "studies": ["study"], "timescales": ["1h"],
                      "covariate_sets": ["all"], "methods": ["origin"]}
        with (
            patch.object(planning, "_resolve_studies", return_value=["study"]),
            patch.object(config, "resolve_covariate_band_ids", return_value=["band"]),
            patch.object(artifacts, "_study_timescale_readiness", return_value={
                "available_point_splits": ["test_temporal", "test_individual"],
                "early_stop_point_split": "test_temporal", "point_ready": True,
            }),
        ):
            planned = planning.expand_experiment(experiment)
        self.assertEqual(len(planned), 1)
        payload = planned[0]
        for field in ("executor", "fit_reuse", "prediction_kind", "gpus"):
            self.assertNotIn(field, payload)
        self.assertEqual(payload["implementation"], self.entry["implementation"])
        self.assertEqual(run.execute_run_payload(payload)["status"], "completed")
        frame = results.load_results("contribution", expanded_experiments=[{
            "experiment": experiment, "runs": planned,
        }])
        self.assertEqual(frame["energy_score"].tolist(), [5., 5.])
        self.assertEqual(slurm_bench._default_gpus_for_method("origin"), 2)
        args = SimpleNamespace(gpus_per_node=0, partition=None, exclude=None, qos=None,
                               account=None, time=None, mem=None, cpus_per_task=1)
        self.assertEqual(slurm_bench._resources_for_run(args, run_payload=payload).gpus_per_node, 0)
        self.entry.pop("gpus")
        self.assertEqual(slurm_bench._default_gpus_for_method("origin"), 0)
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"planned_runs": planned}))
        self.assertEqual(run.load_resolved_run(manifest, payload["run_key"]), payload)

    def test_saved_predictions_score_once_without_predicting(self):
        self.execute()
        before = {p.name: p.read_bytes() for p in (self.directory / "predictions").iterdir()}
        with (
            patch.object(OriginMethod, "predict", side_effect=AssertionError("predicted")),
            patch.object(evaluation, "_score_split", wraps=evaluation._score_split) as score,
            patch.object(evaluation, "evaluate_run_dir", side_effect=AssertionError("rescored")),
        ):
            self.assertEqual(self.execute()["status"], "completed")
        self.assertEqual(score.call_count, 2)
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.directory / "predictions").iterdir()})
        self.assertEqual(self.status(), "completed")
        with gzip.open(self.directory / "final_metrics/test_temporal/energy_by_step.csv.gz", "rt") as handle:
            self.assertEqual(handle.read().splitlines()[1:], ["1,20,5.0,5.0,0.0", "0,10,5.0,5.0,0.0"])

    def test_saved_prediction_recovery_and_missing_split(self):
        for damage in ("unreadable", "wrong_kind", "wrong_ids", "missing"):
            with self.subTest(damage=damage):
                self.execute()
                path = self.directory / "predictions/test_temporal.pt"
                prediction = OriginMethod(model_config={}, hparams={}, seed=0).predict(None)
                if damage == "unreadable":
                    path.write_bytes(b"damaged")
                elif damage == "missing":
                    path.unlink()
                else:
                    if damage == "wrong_kind":
                        prediction.update(prediction_kind="discrete", coords_m=torch.zeros(2, 1, 2), probs=torch.ones(2, 1))
                    else:
                        prediction["point_id_start"] = torch.tensor([10, 10])
                    torch.save(prediction, path)
                with (
                    patch.object(OriginMethod, "predict", autospec=True, side_effect=OriginMethod.predict) as predict,
                    patch.object(evaluation, "_score_split", wraps=evaluation._score_split) as score,
                ):
                    self.assertEqual(self.execute()["status"], "completed")
                self.assertEqual(predict.call_count, 1)
                self.assertEqual(score.call_count, 3 if damage == "wrong_ids" else 2)

    def test_invalid_fresh_prediction_fails_once_and_retains_fit(self):
        with patch.object(OriginMethod, "predict", return_value={}) as predict:
            summary = self.execute()
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["fit_completed"])
        self.assertEqual(summary["train_steps"], 7)
        self.assertEqual(self.status(), "failed")
        self.assertFalse((self.directory / "active_run.json").exists())

    def test_writer_failure_preserves_valid_predictions_and_report_repairs(self):
        self.execute()
        with (
            patch.object(OriginMethod, "predict", side_effect=AssertionError("predicted")),
            patch.object(evaluation, "_write_split_metrics", side_effect=OSError("disk full")),
        ):
            summary = self.execute()
        self.assertEqual(summary["status"], "failed")
        self.assertTrue(summary["fit_completed"])
        self.assertEqual(summary["train_metrics"], {"loss": .25})
        with patch.object(run, "_load_implementation", side_effect=AssertionError("constructed")):
            records, refreshed, errors = self.refresh()
        self.assertEqual((refreshed, errors), (["origin"], []))
        self.assertEqual(records["origin"]["status"], "completed")
        self.assertNotIn("reason", self.summary())

    def test_metric_refresh_marks_unfinished_before_writes_and_reloads_failed_status(self):
        self.execute()
        writer = evaluation._write_split_metrics
        calls = []

        def fail_second(*args):
            self.assertEqual(self.summary()["status"], "running")
            self.assertEqual(self.status(), "running")
            calls.append(args[1])
            if len(calls) == 2:
                raise OSError("second split")
            writer(*args)

        with patch.object(evaluation, "_write_split_metrics", side_effect=fail_second):
            records, refreshed, errors = self.refresh(force=True)
        self.assertEqual((records["origin"]["status"], refreshed, len(errors)), ("failed", [], 1))
        self.assertTrue(self.summary()["fit_completed"])
        self.assertEqual(self.summary()["train_steps"], 7)
        self.assertFalse((self.directory / "active_run.json").exists())
        self.assertEqual(self.refresh()[1], ["origin"])

    def test_report_repairs_missing_unreadable_and_unfinished_metrics(self):
        self.execute()
        for damage in ("csv", "summary", "unreadable", "unfinished"):
            with self.subTest(damage=damage):
                metric = self.directory / "final_metrics/test_individual"
                if damage == "csv":
                    (metric / "energy_by_step.csv.gz").unlink()
                elif damage == "summary":
                    (metric / "summary.json").unlink()
                elif damage == "unreadable":
                    (metric / "summary.json").write_text("bad json")
                else:
                    run_state.write_training_summary(self.directory, self.summary(), status="failed")
                before = {p: p.read_bytes() for p in self.directory.rglob("*") if p.is_file()}
                self.assertEqual(self.refresh(allow=False)[1:], ([], []))
                self.assertEqual(before, {p: p.read_bytes() for p in self.directory.rglob("*") if p.is_file()})
                self.assertEqual(self.refresh()[1:], (["origin"], []))
                self.assertEqual(self.status(), "completed")

    def test_repair_skips_unfitted_missing_prediction_active_and_trainonly(self):
        self.execute()
        original = copy.deepcopy(self.summary())
        for case in ("unfitted", "missing_prediction", "active", "trainonly"):
            with self.subTest(case=case):
                run_state.write_training_summary(self.directory, original, status="failed")
                payload = dict(self.payload)
                if case == "unfitted":
                    run_state.write_training_summary(self.directory, {"train_steps": 0}, status="failed")
                elif case == "missing_prediction":
                    (self.directory / "predictions/test_temporal.pt").unlink()
                elif case == "active":
                    run_state.write_active_run(self.directory)
                else:
                    payload["evaluates"] = False
                with patch.object(evaluation, "evaluate_run") as evaluate:
                    report._refresh_final_metrics({"origin": payload}, latest_run_records={}, refresh_metrics=True, allow_metric_repair=True)
                    evaluate.assert_not_called()
                run_state.remove_active_run(self.directory)
                torch.save(OriginMethod(model_config={}, hparams={}, seed=0).predict(None), self.directory / "predictions/test_temporal.pt")

    def test_direct_reevaluation_requires_every_prediction_and_never_regenerates(self):
        self.execute()
        (self.directory / "predictions/test_temporal.pt").unlink()
        with self.assertRaisesRegex(ValueError, "not eligible"):
            evaluation.evaluate_run("origin")
        self.assertEqual(self.status(), "failed")
        torch.save({}, self.directory / "predictions/test_temporal.pt")
        with self.assertRaises(ValueError), patch.object(OriginMethod, "predict", side_effect=AssertionError("predicted")):
            evaluation.evaluate_run("origin")
        self.assertEqual(self.summary()["status"], "failed")
        self.assertTrue(self.summary()["fit_completed"])

    def test_completion_requires_all_outputs_and_a_final_fitted_summary(self):
        self.execute()
        files = [self.directory / "training_summary.json", *list((self.directory / "predictions").iterdir()), *list((self.directory / "final_metrics").rglob("*.json")), *list((self.directory / "final_metrics").rglob("*.gz"))]
        for path in files:
            before = path.read_bytes()
            path.unlink()
            self.assertEqual(self.status(), "failed", str(path))
            path.write_bytes(before)
        for content in ("bad json", '{"status":"completed"}', '{"status":"running","fit_completed":true}'):
            (self.directory / "training_summary.json").write_text(content)
            self.assertEqual(self.status(), "failed")
            with run_state.active_run(self.directory):
                self.assertEqual(self.status(), "running")

    def test_interruption_after_fitting_and_during_refresh_preserves_fit_and_handlers(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for operation in (self.execute, lambda: evaluation.evaluate_run("origin")):
            self.execute()
            with patch.object(evaluation, "_write_split_metrics", side_effect=lambda *args: signal.raise_signal(signal.SIGTERM)):
                with self.assertRaises(run_state.RunInterrupted):
                    operation()
            self.assertEqual(self.summary()["status"], "interrupted")
            self.assertEqual(self.summary()["train_steps"], 7)
            self.assertTrue(self.summary()["fit_completed"])
            self.assertFalse((self.directory / "active_run.json").exists())
            self.assertEqual({sig: signal.getsignal(sig) for sig in handlers}, handlers)

    def test_custom_persistence_without_reuse_clears_before_fit_even_on_interruption(self):
        self.execute()
        self.entry.pop("fit_reuse")

        def fit(method, train, validation, *, work_dir, restore_from):
            self.assertIsNone(restore_from)
            for name in ("predictions", "final_metrics", "training_summary.json"):
                self.assertFalse((work_dir / name).exists())
            (work_dir / "custom.state").write_text("new weights")
            method.training_summary_ = {"fit_completed": True}
            raise KeyboardInterrupt()

        with patch.object(OriginMethod, "fit", new=fit), self.assertRaises(KeyboardInterrupt):
            self.execute()
        self.assertNotIn("fit_completed", self.summary())
        self.assertFalse((self.directory / "predictions").exists())

    def test_trainonly_requires_checkpoint_without_predictions_or_scores(self):
        self.payload["evaluates"] = False
        fit = OriginMethod.fit

        def save(method, train, validation, *, work_dir, restore_from):
            (work_dir / "checkpoint_best.pt").write_bytes(b"completed model")
            return fit(method, train, validation, work_dir=work_dir)

        with (
            patch.object(OriginMethod, "fit", new=save),
            patch.object(OriginMethod, "predict", side_effect=AssertionError("predicted")),
            patch.object(evaluation, "_score_split", side_effect=AssertionError("scored")),
        ):
            self.assertEqual(self.execute()["status"], "completed")
        self.assertEqual(self.status(), "completed")
        self.assertFalse((self.directory / "predictions").exists())
        (self.directory / "checkpoint_best.pt").unlink()
        self.assertEqual(self.status(), "failed")

    def test_predictions_and_score_rows_are_released_between_splits_and_before_replacement(self):
        references = []
        predict = OriginMethod.predict
        score = evaluation._score_split

        class Rows(list):
            pass

        def watched_predict(method, data):
            self.assertTrue(all(ref() is None for ref in references))
            return predict(method, data)

        def watched_score(**kwargs):
            rows, summary = score(**kwargs)
            rows = Rows(rows)
            references.extend([weakref.ref(rows), weakref.ref(kwargs["pred_artifact"]["point_id_start"])])
            return rows, summary

        with patch.object(OriginMethod, "predict", new=watched_predict), patch.object(evaluation, "_score_split", new=watched_score):
            self.assertEqual(self.execute()["status"], "completed")
        self.assertTrue(all(ref() is None for ref in references))
        references.clear()

        def reject(**kwargs):
            if not references:
                references.append(weakref.ref(kwargs["pred_artifact"]["point_id_start"]))
                raise ValueError("reject cache")
            return score(**kwargs)

        with patch.object(OriginMethod, "predict", new=watched_predict), patch.object(evaluation, "_score_split", new=reject):
            self.assertEqual(self.execute()["status"], "completed")

    def test_optional_prediction_kind_accepts_supported_representations(self):
        self.entry.pop("prediction_kind")
        common = OriginMethod(model_config={}, hparams={}, seed=0).predict(None)
        variants = [
            common,
            {**common, "prediction_kind": "discrete", "coords_m": torch.zeros(2, 1, 2), "probs": torch.ones(2, 1)},
            {**common, "prediction_kind": "grid", "scores": torch.zeros(2, 3, 3), "pixel_size_m": 3.},
            {**common, "prediction_kind": "grid", "probs_hw": torch.full((2, 3, 3), 1 / 9), "x_coords_m": torch.tensor([-3., 0., 3.]), "y_coords_m": torch.tensor([-3., 0., 3.])},
        ]
        for prediction in variants:
            with patch.object(OriginMethod, "predict", return_value=prediction):
                self.assertEqual(run.execute_run_payload(self.payload, resume=False)["status"], "completed")
        for row in self.truth:
            row["in_support"] = False
        with patch.object(OriginMethod, "predict", return_value=common):
            self.assertEqual(run.execute_run_payload(self.payload, resume=False)["status"], "completed")
        metrics = json.loads((self.directory / "final_metrics/test_temporal/summary.json").read_text())
        self.assertIsNone(metrics["energy_score"])
        self.assertEqual(metrics["n_steps_excluded_out_of_support"], 2)


class BenchRegistrationTest(unittest.TestCase):
    def test_builtin_declarations_and_experiment_groups(self):
        self.assertEqual(len(config.METHODS), 11)
        self.assertNotIn("fit_reuse", config.METHODS["moveformer_full_predictonly"])
        for key, entry in config.METHODS.items():
            self.assertNotIn("executor", entry)
            self.assertTrue(callable(run._load_implementation(entry["implementation"])))
            self.assertEqual(entry["gpus"], int(key == "deepssf" or key.startswith("moveformer")))
        self.assertEqual(summary.normalize_experiment_selection(["ablation"]), ["ablation"])
        perstudy = planning.load_experiment("experiments/moveformer_perstudy.py")
        self.assertEqual(perstudy["name"], "ablation")
        self.assertEqual(perstudy["covariate_sets"], ["all", "alphaearth_only"])
        self.assertEqual(perstudy["methods"], ["moveformer_perstudy"])
        for name in ("main_results", "moveformer_full_trainonly", "moveformer_full_predictonly"):
            experiment = planning.load_experiment(f"experiments/{name}.py")
            self.assertEqual(experiment["name"], "main_results")
            self.assertNotIn("moveformer_perstudy", experiment["methods"])

    def test_main_shared_and_repair_runs_resolve_the_same_full_cohort(self):
        names = ("main_results", "moveformer_full_trainonly", "moveformer_full_predictonly",
                 "main_results_valid_1d_repairs", "main_results_valid_7d_repairs")
        experiments = {name: planning.load_experiment(f"experiments/{name}.py") for name in names}
        cohort = ["other", *experiments[names[-2]]["studies"], *experiments[names[-1]]["studies"]]
        with (
            patch.object(planning, "resolve_study_selection", side_effect=lambda value: cohort if value == "all" else list(value)),
            patch.object(artifacts, "_study_timescale_readiness", return_value={
                "available_point_splits": ["test_temporal"], "early_stop_point_split": "test_temporal",
                "point_ready": True, "train_point_ready": True,
            }),
            patch.object(artifacts, "_resolve_artifacts_for_run", return_value={"status": "resolved"}),
        ):
            expanded = {name: planning.expand_experiment(experiment) for name, experiment in experiments.items()}
        trainers = {payload["timescale"]: payload for payload in expanded["moveformer_full_trainonly"]}
        main = [payload for payload in expanded["main_results"] if payload["method"] == "moveformer_full_predictonly"]
        self.assertEqual([payload["run_key"] for payload in main], [payload["run_key"] for payload in expanded["moveformer_full_predictonly"]])
        for name in ("main_results", "moveformer_full_predictonly", *names[-2:]):
            for payload in expanded[name]:
                if payload["method"] != "moveformer_full_predictonly":
                    continue
                trainer = trainers[payload["timescale"]]
                self.assertEqual(payload["shared_checkpoint_run_key"], trainer["run_key"])
                self.assertEqual(set(payload["shared_training_study_ids"]), set(cohort))
        planned = results.planned_run_index([
            {"experiment": experiments[name], "runs": expanded[name]} for name in names[:3]
        ])
        self.assertEqual(sum(payload["method"] == "moveformer_full_predictonly" for payload in planned.values()), len(main))


if __name__ == "__main__":
    unittest.main()
