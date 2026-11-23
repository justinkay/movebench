import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from movebench.bench import artifacts, config, planning, run, run_state
from movebench.bench.evaluation import runner as evaluation
from movebench.geometry import LOCAL_GEOMETRY
from movebench.cli.slurm import bench as slurm_bench
from movebench.training import read_checkpoint, save_checkpoint

TRAINER = "moveformer_full_trainonly"
SCORER = "moveformer_full_predictonly"


class SharedWorkflowTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch.object(run.paths, "repo_root", return_value=self.root))
        self.enterContext(patch("movebench.study._list_available_studies", return_value=["a", "b"]))
        self.enterContext(patch.object(config, "resolve_covariate_band_ids", return_value=["band"]))
        self.enterContext(patch.object(config, "resolve_timescale", return_value={
            "step_time_s": 3600, "alphaearth_layers_config": "ae.yaml",
            "layers_config": "all.yaml", "rasters_hash": "rasters",
            "sampler_config": "sampler.yaml", "sampler_hash": "sampler",
            "geometry": LOCAL_GEOMETRY, "output_hw": (3, 3),
        }))
        self.enterContext(patch.object(artifacts, "_study_timescale_readiness", return_value={
            "available_point_splits": ["test_temporal"],
            "early_stop_point_split": "test_temporal", "train_point_ready": True,
            "point_ready": True, "point_blocked_reasons": [],
        }))
        self.enterContext(patch.object(
            artifacts, "_resolve_point_artifacts",
            side_effect=lambda payload, **kwargs: artifacts._resolved_splits(payload, {}, (3, 3)),
        ))
        self.enterContext(patch.object(evaluation, "_selected_splits_for_split", return_value=None))
        self.enterContext(patch.object(evaluation, "_support_rows_for_split", return_value=([
            {"point_id_start": 10, "step_index": 0, "in_support": True,
             "truth_xy_m": torch.zeros(2, dtype=torch.float64)},
        ], 3.)))
        self.events = []
        self.enterContext(patch.object(run, "_load_implementation", side_effect=self.implementation))
        self.experiments = {}
        for method in (TRAINER, SCORER):
            experiment = {
                "name": "shared-workflow", "studies": ["a", "b"],
                "timescales": ["1h"], "covariate_sets": ["alphaearth_only"],
                "methods": [method], "trained_on_studies": ["a", "b"],
            }
            path = self.root / f"{method}.py"
            path.write_text(f"EXPERIMENT = {experiment!r}\n")
            self.experiments[method] = str(path)

    def implementation(self, reference):
        is_trainer = reference == config.METHODS[TRAINER]["implementation"]
        method = SimpleNamespace()

        def fit(train, validation, *, work_dir, restore_from=None):
            if restore_from is not None:
                checkpoint = read_checkpoint(restore_from)
                self.events.append(("restore" if is_trainer else "load", work_dir.name))
            else:
                self.assertTrue(is_trainer, "a scorer must never train")
                self.events.append(("train", work_dir.name))
                checkpoint = save_checkpoint(
                    work_dir / "checkpoint_best.pt", model=torch.nn.Linear(1, 1),
                    fit={"best_step": 2, "completed_steps": 3,
                         "train_metrics": {"loss": 1.}, "eval_metrics": {"loss": 2.}},
                    schedule={}, model_metadata={
                        "shared_training_signature": train.training_signature,
                        "shared_study_ids": list(train.requested_study_ids),
                        "shared_data_study_ids": list(train.requested_study_ids),
                        "shared_zero_step_studies": [], "taxa": [],
                        "eval_split_by_study": {}, "skipped_eval_study_ids": [],
                    },
                )
            method.training_summary_ = {"train_steps": checkpoint["completed_steps"]}
            return method

        def predict(data):
            self.events.append(("predict", data.metadata["study_id"]))
            return {
                "prediction_kind": "grid", "geometry": LOCAL_GEOMETRY,
                "point_id_start": torch.tensor([10]),
                "scores": torch.zeros(1, 3, 3), "pixel_size_m": 3.,
            }

        method.fit = fit
        if not is_trainer:
            method.predict = predict
        return lambda **kwargs: method

    def resolve(self, method, studies=None, **kwargs):
        return run.resolve_pending_runs(
            self.experiments[method], studies_override=studies, **kwargs
        )

    def local(self, method, studies=None, *, failed=0, **kwargs):
        result = run.run_experiment(
            self.experiments[method], studies_override=studies, **kwargs
        )
        self.assertEqual(result["failed_runs"], failed)
        return result

    def snapshot(self, directory):
        return {
            str(path.relative_to(directory)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()
        }

    def train(self):
        payload, = self.resolve(TRAINER)["pending_runs"]
        self.assertEqual(self.local(TRAINER)["executed_runs"], 1)
        return run_state.run_dir(payload["run_key"])

    def launch(self, method, studies, *, workers=False, **overrides):
        args = SimpleNamespace(
            experiment_module=self.experiments[method], studies=studies,
            methods=None, max_runs=None, rerun=False, dry_run=False,
            gpus_per_node=None, cpus_per_task=2, partition="gpu",
            exclude=None, qos=None, account=None, time="00:17:00", mem="8G",
            max_cpu_jobs=3, max_gpu_jobs=1, polling_interval=1,
            log_dir=self.root / "logs",
        )
        vars(args).update(overrides)
        requests = []

        def queue(**kwargs):
            self.assertEqual(set(kwargs), {"label", "requests", "polling_interval", "dry_run"})
            requests.extend(kwargs["requests"])
            parser = argparse.ArgumentParser()
            slurm_bench.register(parser.add_subparsers())
            for request in requests:
                for worker_args in request.worker_args_list:
                    worker = parser.parse_args(worker_args[1:])
                    self.assertEqual(worker.rerun, args.rerun)
                    payload = run.load_resolved_run(worker.manifest_path, worker.run_key)
                    self.assertIn(payload["method"], args.methods or [method])
                    if workers:
                        self.assertEqual(slurm_bench._run_worker(worker), 0)
            return 0

        with patch.object(slurm_bench, "run_sbatch_queue", side_effect=queue):
            self.assertEqual(slurm_bench._run_launcher(args), 0)
        return requests

    def test_separate_commands_reuse_trainer_and_completed_subset(self):
        self.assertEqual(self.local(SCORER, ["a"])["blocked_runs"], 1)
        self.assertEqual(self.launch(SCORER, ["a", "b"]), [])
        self.assertEqual(self.events, [])
        trainer = self.train()
        trainer_before = self.snapshot(trainer)
        self.assertEqual(self.local(TRAINER)["skipped_runs"], 1)
        score_a, = self.resolve(SCORER, ["a"])["pending_runs"]
        self.assertEqual(score_a["shared_checkpoint_run_key"], trainer.name)
        self.assertEqual(self.local(SCORER, ["a"])["executed_runs"], 1)
        a_dir = run_state.run_dir(score_a["run_key"])
        a_before = self.snapshot(a_dir)
        pending = self.resolve(SCORER, ["a", "b"], max_runs=1)
        self.assertEqual(pending["counts"]["skipped_runs"], 1)
        self.assertEqual([p["study_id"] for p in pending["pending_runs"]], ["b"])
        self.assertEqual(pending["pending_runs"][0]["shared_checkpoint_run_key"], trainer.name)
        self.assertEqual(self.local(SCORER, ["a", "b"], max_runs=1)["executed_runs"], 1)
        self.assertEqual(self.snapshot(trainer), trainer_before)
        self.assertEqual(self.snapshot(a_dir), a_before)
        self.assertEqual([event for event in self.events if event[0] == "predict"], [("predict", "a"), ("predict", "b")])

    def test_scoring_default_cohort_and_membership(self):
        trainer = self.train()
        before = self.snapshot(trainer)
        experiment = planning.load_experiment(self.experiments[SCORER])
        experiment.pop("trained_on_studies")
        self.assertEqual(planning.expand_experiment(experiment, ["a"]), [])
        self.assertIn("not completed", experiment["blocked"][0]["reason"])
        runs = planning.expand_experiment(experiment, ["a", "b"])
        self.assertEqual({p["shared_checkpoint_run_key"] for p in runs}, {trainer.name})
        experiment["trained_on_studies"] = ["a", "b"]
        self.assertEqual(planning.expand_experiment(experiment, ["outside"]), [])
        self.assertIn("no compatible shared training run", experiment["blocked"][0]["reason"])
        self.assertEqual(self.local(TRAINER, ["a"])["executed_runs"], 1)
        subset, = self.resolve(TRAINER, ["a"])["planned_runs"]
        self.assertNotEqual(subset["run_key"], trainer.name)
        self.assertEqual(self.snapshot(trainer), before)

    def test_unavailable_trainer_blocks_planner_and_saved_worker(self):
        trainer = self.train()
        resolved = self.resolve(SCORER, ["a"])
        scorer, = resolved["pending_runs"]
        slurm_bench._write_manifest(resolved["manifest_path"], resolved["manifest_payload"])
        checkpoint = trainer / "checkpoint_best.pt"
        summary = trainer / "training_summary.json"
        original_summary, original_checkpoint = summary.read_bytes(), checkpoint.read_bytes()
        for condition in ("missing_summary", "missing_checkpoint", "failed", "running", "active", "no_fit", "signature"):
            with self.subTest(condition=condition):
                summary.write_bytes(original_summary)
                checkpoint.write_bytes(original_checkpoint)
                if condition == "missing_summary":
                    summary.unlink()
                elif condition == "missing_checkpoint":
                    checkpoint.unlink()
                elif condition == "active":
                    run_state.write_active_run(trainer)
                elif condition == "signature":
                    path = trainer / "resolved_run.json"
                    payload = json.loads(path.read_text())
                    payload["shared_training_signature"] = "wrong"
                    path.write_text(json.dumps(payload))
                else:
                    summary.write_text(json.dumps({
                        "status": "completed" if condition == "no_fit" else condition,
                        "fit_completed": condition != "no_fit",
                    }))
                before = self.snapshot(trainer)
                with patch.object(run, "_load_implementation") as construct:
                    blocked = self.resolve(SCORER, ["a"])
                    self.assertEqual(blocked["pending_runs"], [])
                    self.assertEqual(blocked["counts"]["blocked_runs"], 1)
                    self.assertEqual(slurm_bench._run_worker(SimpleNamespace(
                        manifest_path=resolved["manifest_path"], run_key=scorer["run_key"], rerun=False,
                    )), 2)
                    construct.assert_not_called()
                self.assertEqual(self.snapshot(trainer), before)
                self.assertEqual(run_state.load_training_summary(run_state.run_dir(scorer["run_key"]))["status"], "blocked")
                run_state.remove_active_run(trainer)

    def test_prediction_rerun_and_failure_leave_other_results_intact(self):
        trainer = self.train()
        self.local(SCORER)
        runs = {p["study_id"]: p for p in self.resolve(SCORER)["planned_runs"]}
        b_dir = run_state.run_dir(runs["b"]["run_key"])
        trainer_before, b_before = self.snapshot(trainer), self.snapshot(b_dir)
        self.events.clear()
        with patch.object(evaluation, "_write_split_metrics", side_effect=OSError("disk full")):
            self.local(SCORER, ["a"], rerun=True, failed=1)
        self.assertEqual(self.events, [("load", runs["a"]["run_key"]), ("predict", "a")])
        self.assertEqual(self.snapshot(trainer), trainer_before)
        self.assertEqual(self.snapshot(b_dir), b_before)
        self.events.clear()
        self.assertEqual(self.local(SCORER, ["a"])["executed_runs"], 1)
        self.assertEqual(self.events, [("load", runs["a"]["run_key"])])
        self.local(SCORER, ["a"], rerun=True)
        self.assertEqual(self.events[-1], ("predict", "a"))
        self.assertEqual(self.snapshot(trainer), trainer_before)
        self.assertEqual(self.snapshot(b_dir), b_before)

    def test_trainer_summary_repair_and_replacement_are_distinct(self):
        trainer = self.train()
        self.local(SCORER)
        runs = self.resolve(SCORER)["planned_runs"]
        dependents = [run_state.run_dir(p["run_key"]) for p in runs]
        before = [self.snapshot(path) for path in dependents]
        checkpoint = trainer / "checkpoint_best.pt"
        checkpoint_before = checkpoint.read_bytes()
        (trainer / "training_summary.json").unlink()
        self.events.clear()
        self.assertEqual(self.local(TRAINER)["executed_runs"], 1)
        self.assertEqual(self.events, [("restore", trainer.name)])
        self.assertEqual(checkpoint.read_bytes(), checkpoint_before)
        self.assertEqual([self.snapshot(path) for path in dependents], before)
        for missing in (False, True):
            with self.subTest(missing_checkpoint=missing):
                if missing:
                    checkpoint.unlink()
                    self.assertEqual(self.local(SCORER)["blocked_runs"], 2)
                self.events.clear()
                self.assertEqual(self.local(TRAINER, rerun=not missing)["executed_runs"], 1)
                self.assertEqual(self.events, [("train", trainer.name)])
                for path in dependents:
                    self.assertEqual(set(self.snapshot(path)), {"resolved_run.json"})
                self.assertEqual(self.local(SCORER, ["a"])["executed_runs"], 1)
                self.assertNotEqual(run_state._load_run_record(runs[1]["run_key"])["status"], "completed")
                self.assertEqual(self.events[-1], ("predict", "a"))
                self.local(SCORER, ["b"])

    def test_invalid_checkpoint_load_never_falls_back_to_training(self):
        trainer = self.train()
        self.local(SCORER)
        scorers = self.resolve(SCORER)["planned_runs"]
        before = {p["run_key"]: self.snapshot(run_state.run_dir(p["run_key"])) for p in scorers}
        (trainer / "checkpoint_best.pt").write_bytes(b"invalid checkpoint")
        (trainer / "training_summary.json").unlink()
        self.events.clear()
        self.local(TRAINER, failed=1)
        self.assertEqual(self.events, [])
        self.assertEqual((trainer / "checkpoint_best.pt").read_bytes(), b"invalid checkpoint")
        self.assertEqual({p["run_key"]: self.snapshot(run_state.run_dir(p["run_key"])) for p in scorers}, before)

    def test_saved_worker_arrays_keep_resources_and_command_rerun(self):
        training_requests = self.launch(TRAINER, ["a", "b"], workers=True)
        self.assertEqual(len(training_requests), 1)
        self.assertEqual(len(training_requests[0].worker_args_list), 1)
        trainer = run_state.run_dir(self.resolve(TRAINER)["planned_runs"][0]["run_key"])
        trainer_before = self.snapshot(trainer)
        requests = self.launch(SCORER, ["a", "b"], workers=True)
        request, = requests
        self.assertEqual((len(request.worker_args_list), request.array_concurrency), (2, 1))
        self.assertEqual((request.resources.gpus_per_node, request.resources.cpus_per_task), (1, 2))
        self.assertEqual((request.resources.time, request.resources.mem), ("00:17:00", "8G"))
        self.assertEqual(self.launch(SCORER, ["a", "b"]), [])
        requests = self.launch(SCORER, ["a", "b"], workers=True, rerun=True, max_runs=1, gpus_per_node=0)
        self.assertEqual(len(requests[0].worker_args_list), 1)
        self.assertEqual(requests[0].resources.gpus_per_node, 0)
        self.assertEqual(self.snapshot(trainer), trainer_before)
        requests = self.launch(
            SCORER, ["a", "b"], methods=["baseline_nomove", SCORER], rerun=True,
        )
        self.assertEqual([
            (r.resources.gpus_per_node, r.array_concurrency, len(r.worker_args_list))
            for r in requests
        ], [(0, 2, 2), (1, 1, 2)])
        self.assertTrue(all(r.resources.time == "00:17:00" for r in requests))


if __name__ == "__main__":
    unittest.main()
