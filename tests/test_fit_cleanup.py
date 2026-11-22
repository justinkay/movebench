import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from movebench.bench import artifacts, planning, run, run_state
from movebench.datasets import DataSplit
from movebench.training import RunTracker, TrainerConfig, fit_or_restore, save_checkpoint


class FitCleanupTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch.object(run.paths, "repo_root", return_value=self.root))
        self.enterContext(patch.object(artifacts, "_resolve_artifacts_for_run", return_value={"status": "resolved", "train": object(), "splits": {"train": DataSplit("train", {}, {})}}))
        self.model = torch.nn.Linear(1, 1)
        self.fit = {
            "best_step": 2, "completed_steps": 3,
            "train_metrics": {"nll": 1.0}, "eval_metrics": {"nll": 2.0},
        }

    def payload(self, key="local", *, shared=False):
        return {
            "run_key": key,
            "method": "moveformer_full_trainonly" if shared else "issf_linear",
            "implementation": "unused:method",
            "seed": 17,
            "early_stop_split_key": "test_temporal",
            "evaluates": False,
            **({"shared_training_run": True, "shared_training_signature": "signature"} if shared else {}),
        }

    def results(self, payload):
        directory = run_state.run_dir(payload["run_key"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "resolved_run.json").write_text(json.dumps(planning._canonical_run_payload(payload)))
        (directory / "training_summary.json").write_text('{"status": "completed", "fit_completed": true}')
        for name in ("predictions", "final_metrics"):
            (directory / name).mkdir(exist_ok=True)
            (directory / name / "old").write_text("old model results")
        return directory

    def checkpoint(self, directory, *, shared=False):
        metadata = {
            "shared_training_signature": "signature",
            "shared_study_ids": ["a", "outside"],
            "shared_data_study_ids": ["a", "outside"],
            "shared_zero_step_studies": [], "taxa": {},
            "eval_split_by_study": {}, "skipped_eval_study_ids": [],
        } if shared else {}
        save_checkpoint(
            directory / "checkpoint_best.pt", model=self.model,
            fit=self.fit, schedule={}, model_metadata=metadata,
        )

    def execute(self, payload, fit_callback, *, resume=True):
        def fit(train, validation, *, work_dir, restore_from=None):
            method.training_summary_ = fit_callback(
                run_dir=work_dir, restore_from=restore_from, tracker=RunTracker(work_dir),
            )
            return method
        method = SimpleNamespace(fit=fit)
        with patch.object(run, "_load_implementation", return_value=lambda **kwargs: method):
            return run.execute_run_payload(payload, resume=resume)

    def restore(self, *, restore_from, run_dir, tracker, **kwargs):
        result = fit_or_restore(
            SimpleNamespace(model=self.model), TrainerConfig(3, 1, 2), tracker,
            work_dir=run_dir, restore_from=restore_from,
        )
        return {"status": "completed", "train_steps": result["completed_steps"]}

    def assert_cleared(self, directory):
        for name in ("predictions", "final_metrics", "training_summary.json"):
            self.assertFalse((directory / name).exists(), str(directory / name))

    def test_fresh_fit_clears_results_before_interruption_and_retry(self):
        for after_save in (False, True):
            with self.subTest(after_save=after_save):
                payload = self.payload(str(after_save))
                directory = self.results(payload)

                def interrupt(**kwargs):
                    self.assert_cleared(directory)
                    self.assertIsNone(kwargs["restore_from"])
                    if after_save:
                        self.checkpoint(directory)
                    raise KeyboardInterrupt()

                with self.assertRaises(KeyboardInterrupt):
                    self.execute(payload, interrupt)
                self.assertFalse((directory / "predictions").exists())
                self.assertFalse(run_state.active_run_is_live(directory))

                def retry(**kwargs):
                    self.assertFalse((directory / "predictions").exists())
                    self.assertEqual(kwargs["restore_from"] is not None, after_save)
                    if after_save:
                        return self.restore(**kwargs)
                    self.checkpoint(directory)
                    return {"status": "completed", "train_steps": 0}

                self.assertEqual(self.execute(payload, retry)["status"], "completed")

    def test_selected_checkpoint_disappears_without_retraining(self):
        payload = self.payload()
        directory = self.results(payload)
        self.checkpoint(directory)

        def delete_then_restore(**kwargs):
            self.assertIsNotNone(kwargs["restore_from"])
            (directory / "checkpoint_best.pt").unlink()
            return self.restore(**kwargs)

        with patch("movebench.training.run_training_loop") as fit:
            result = self.execute(payload, delete_then_restore)
        self.assertEqual(result["status"], "failed")
        fit.assert_not_called()
        self.assertTrue((directory / "predictions" / "old").exists())
        self.assertFalse(run_state.active_run_is_live(directory))

    def test_prediction_failure_after_save_restores_checkpoint_on_retry(self):
        payload = self.payload()
        directory = self.results(payload)

        def fail_prediction(**kwargs):
            self.assert_cleared(directory)
            self.assertIsNone(kwargs["restore_from"])
            self.checkpoint(directory)
            raise RuntimeError("prediction failed")

        self.assertEqual(self.execute(payload, fail_prediction)["status"], "failed")
        with patch("movebench.training.run_training_loop") as fit:
            result = self.execute(payload, self.restore)
        self.assertEqual(result["train_steps"], 3)
        fit.assert_not_called()

    def test_valid_checkpoint_preserves_predictions_and_ignores_damaged_history(self):
        payload = self.payload()
        directory = self.results(payload)
        self.checkpoint(directory)
        (directory / "history.json").write_text("damaged")
        with patch("movebench.training.run_training_loop") as fit:
            result = self.execute(payload, self.restore)
        self.assertEqual(result["train_steps"], 3)
        self.assertTrue((directory / "predictions" / "old").exists())
        self.assertEqual((directory / "history.json").read_text(), "damaged")
        fit.assert_not_called()

    def shared_results(self):
        payload = self.payload("trainer", shared=True)
        trainer = self.results(payload)
        dependents = [self.results({
            **self.payload(study), "method": "moveformer_full_predictonly",
            "shared_checkpoint_run_key": "trainer",
        }) for study in ("a", "outside")]
        unrelated = self.results({**self.payload("unrelated"), "shared_checkpoint_run_key": "other"})
        return payload, trainer, dependents, unrelated

    def test_shared_replacement_clears_all_dependents_before_fit(self):
        for reason in ("missing", "rerun", "mismatch"):
            with self.subTest(reason=reason):
                payload, trainer, dependents, unrelated = self.shared_results()
                if reason != "missing":
                    self.checkpoint(trainer, shared=True)
                if reason == "mismatch":
                    payload["seed"] += 1

                def fresh(**kwargs):
                    self.assertIsNone(kwargs["restore_from"])
                    for directory in [trainer, *dependents]:
                        self.assert_cleared(directory)
                        self.assertTrue((directory / "resolved_run.json").exists())
                    self.assertFalse((trainer / "checkpoint_best.pt").exists())
                    self.assertTrue((unrelated / "predictions" / "old").exists())
                    raise KeyboardInterrupt()

                with self.assertRaises(KeyboardInterrupt):
                    self.execute(payload, fresh, resume=reason != "rerun")

    def test_active_shared_dependent_prevents_any_deletion(self):
        payload, trainer, dependents, unrelated = self.shared_results()
        self.checkpoint(trainer, shared=True)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with (
            patch.object(run_state, "active_run_is_live", side_effect=lambda directory: directory == dependents[-1]),
            patch.object(run, "_load_implementation") as executor,
        ):
            with self.assertRaisesRegex(RuntimeError, "Retry training after scoring finishes"):
                run.execute_run_payload(payload, resume=False)
        executor.assert_not_called()
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()})

    def test_shared_cleanup_failure_prevents_checkpoint_replacement(self):
        payload, trainer, dependents, unrelated = self.shared_results()
        self.checkpoint(trainer, shared=True)
        checkpoint = (trainer / "checkpoint_best.pt").read_bytes()
        clear = run._clear_prediction_outputs

        def fail_second(directory):
            if directory == dependents[-1]:
                raise OSError("cleanup failed")
            clear(directory)

        with (
            patch.object(run, "_clear_prediction_outputs", side_effect=fail_second),
            patch.object(run, "_load_implementation") as executor,
        ):
            with self.assertRaisesRegex(OSError, "cleanup failed"):
                run.execute_run_payload(payload, resume=False)
        self.assert_cleared(dependents[0])
        self.assertEqual((trainer / "checkpoint_best.pt").read_bytes(), checkpoint)
        executor.assert_not_called()

    def test_shared_summary_repair_restores_without_clearing_dependents(self):
        payload, trainer, dependents, unrelated = self.shared_results()
        self.checkpoint(trainer, shared=True)
        (trainer / "training_summary.json").unlink()
        with patch("movebench.training.run_training_loop") as fit:
            result = self.execute(payload, self.restore)
        self.assertEqual(result["train_steps"], 3)
        for directory in [*dependents, unrelated]:
            self.assertTrue((directory / "predictions" / "old").exists())
            self.assertTrue((directory / "training_summary.json").exists())
        fit.assert_not_called()

    def test_shared_checkpoint_loss_is_unfinished_but_live_status_takes_precedence(self):
        payload = self.payload("trainer", shared=True)
        directory = self.results(payload)
        record = run_state._load_run_record("trainer")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["tags"]["failure_reason"], "missing checkpoint_best.pt")
        with (
            patch.object(planning, "load_experiment", return_value={"name": "retry"}),
            patch.object(planning, "expand_experiment", return_value=[payload]),
        ):
            pending = run.resolve_pending_runs("unused")
        self.assertEqual(pending["pending_runs"], [payload])
        with patch.object(run_state, "active_run_is_live", return_value=True):
            self.assertEqual(run_state._load_run_record("trainer")["status"], "running")
        self.checkpoint(directory, shared=True)
        self.assertEqual(run_state._load_run_record("trainer")["status"], "completed")

    def test_runner_seeds_all_generators_before_method_setup(self):
        draws = []

        def executor(**kwargs):
            draws.append((random.random(), float(np.random.rand()), torch.rand(3)))
            self.checkpoint(kwargs["run_dir"])
            return {"status": "completed"}

        for initial in (3, 88):
            random.seed(initial)
            np.random.seed(initial)
            torch.manual_seed(initial)
            self.execute(self.payload(str(initial)), executor)
        self.assertEqual(draws[0][:2], draws[1][:2])
        torch.testing.assert_close(draws[0][2], draws[1][2], rtol=0, atol=0)

    def test_interruption_history_reporting_remains_separate(self):
        payload = self.payload()
        directory = self.results(payload)
        self.checkpoint(directory)
        (directory / "history.json").write_text('[{"step": 7}]')

        def interrupt(**kwargs):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.execute(payload, interrupt)
        summary = json.loads((directory / "training_summary.json").read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["train_steps"], 7)


if __name__ == "__main__":
    unittest.main()
