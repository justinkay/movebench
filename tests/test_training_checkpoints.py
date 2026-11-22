import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from movebench.training import (
    RunTracker,
    TrainerConfig,
    fit_or_restore,
    read_checkpoint,
    run_training_loop,
    save_checkpoint,
)


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.model = torch.nn.Linear(1, 1)
        self.reference = {
            "checkpoint_path": self.directory / "checkpoint_best.pt",
            "training_run_key": None,
            "training_signature": None,
        }
        self.fit = {
            "best_step": 4,
            "completed_steps": 6,
            "train_metrics": {"nll": 1.5},
            "eval_metrics": {"nll": 2.0},
        }
        self.schedule = {"steps_per_epoch": 2, "max_steps": 10}
        self.config = TrainerConfig(10, 2, 1)

    def save(self, metadata=None):
        return save_checkpoint(
            self.reference["checkpoint_path"],
            model=self.model,
            fit=self.fit,
            schedule=self.schedule,
            model_metadata=metadata or {},
        )

    def test_round_trip_uses_saved_statistics_without_history_or_training(self):
        runtime = SimpleNamespace(model=self.model)
        with patch("movebench.training.run_training_loop", return_value=self.fit) as fit:
            result = fit_or_restore(
                runtime, self.config, object(), work_dir=self.directory,
                restore_from=None, schedule=self.schedule,
            )
        fit.assert_called_once()
        checkpoint = read_checkpoint(self.reference)
        self.assertEqual(set(checkpoint), {
            "model_state", "best_step", "completed_steps", "train_metrics",
            "eval_metrics", "schedule", "model_metadata",
        })
        for tensor in checkpoint["model_state"].values():
            self.assertEqual(tensor.device.type, "cpu")
        for history in (None, "damaged history"):
            with self.subTest(history=history):
                if history is not None:
                    (self.directory / "history.json").write_text(history)
                with torch.no_grad():
                    self.model.weight.fill_(-10)
                with (
                    patch("movebench.training.run_training_loop") as fit,
                    patch.object(Path, "read_text", side_effect=AssertionError("history read")),
                ):
                    restored = fit_or_restore(
                        runtime, self.config, object(), work_dir=self.directory,
                        restore_from=self.reference, schedule={"max_steps": 999},
                        model_metadata={"changed": True},
                    )
                fit.assert_not_called()
                self.assertEqual(restored, result)
                for name, tensor in self.model.state_dict().items():
                    torch.testing.assert_close(tensor, checkpoint["model_state"][name])

    def test_requested_load_never_falls_back_to_training(self):
        valid = self.save()
        cases = {"missing": None, "corrupt": b"not a checkpoint", "old_format": {
            "state_dict": self.model.state_dict(), "meta": {"best_step": 4},
        }}
        for field in valid:
            cases[f"missing_{field}"] = {key: value for key, value in valid.items() if key != field}
        cases["invalid_steps"] = {**valid, "completed_steps": 2}
        cases["invalid_metrics"] = {**valid, "eval_metrics": None}
        cases["invalid_weights"] = {**valid, "model_state": {"weight": torch.zeros(4)}}
        for name, checkpoint in cases.items():
            with self.subTest(name=name):
                path = self.reference["checkpoint_path"]
                path.unlink(missing_ok=True)
                if isinstance(checkpoint, bytes):
                    path.write_bytes(checkpoint)
                elif checkpoint is not None:
                    torch.save(checkpoint, path)
                with patch("movebench.training.run_training_loop") as fit:
                    with self.assertRaises(Exception):
                        fit_or_restore(
                            SimpleNamespace(model=self.model), self.config, object(),
                            work_dir=self.directory, restore_from=self.reference,
                        )
                fit.assert_not_called()

    def test_shared_reader_and_trainer_require_signature_and_metadata(self):
        metadata = {
            "shared_training_signature": "expected",
            "shared_study_ids": ["a", "b", "zero"],
            "shared_data_study_ids": ["a", "b"],
            "shared_zero_step_studies": [{"study_id": "zero", "reason": "no steps"}],
            "taxa": {"a": {"species": "a"}, "b": {"species": "b"}},
            "eval_split_by_study": {"a": "test_temporal"},
            "skipped_eval_study_ids": ["b"],
        }
        self.reference.update(training_run_key="trainer", training_signature="expected")
        self.save(metadata)
        self.assertEqual(read_checkpoint(self.reference)["model_metadata"], metadata)
        invalid = [{**metadata, "shared_training_signature": "wrong"}]
        invalid += [{key: value for key, value in metadata.items() if key != missing}
                    for missing in metadata]
        for candidate in invalid:
            with self.subTest(metadata=candidate):
                self.save(candidate)
                with self.assertRaises(ValueError):
                    read_checkpoint(self.reference)
                with patch("movebench.training.run_training_loop") as fit:
                    with self.assertRaises(ValueError):
                        fit_or_restore(
                            SimpleNamespace(model=self.model), self.config, object(),
                            work_dir=self.directory, restore_from=self.reference,
                        )
                fit.assert_not_called()

    def test_controlled_fit_saves_best_weights_and_actual_completed_steps(self):
        model = self.model
        calls = []

        class Runtime:
            def __init__(self):
                self.model = model
                self.step = 0
                self.closed = False

            def train_step(self, step):
                self.step = step
                with torch.no_grad():
                    self.model.weight.fill_(step)
                return {"loss": float(step)}

            def evaluate(self, split):
                calls.append((self.step, split))
                return {"nll": 100.0 + self.step if split == "train" else {2: 3., 4: 2., 6: 2.}[self.step]}

            def state_dict(self):
                return {"step": self.step, "model": copy.deepcopy(self.model.state_dict())}

            def load_state_dict(self, state):
                self.step = state["step"]
                self.model.load_state_dict(state["model"])

            def close(self):
                self.closed = True

        runtime = Runtime()
        result = fit_or_restore(
            runtime, self.config, RunTracker(self.directory),
            work_dir=self.directory, restore_from=None,
        )
        self.assertEqual(result["best_step"], 4)
        self.assertEqual(result["completed_steps"], 6)
        self.assertEqual(result["train_metrics"], {"nll": 104.0})
        self.assertEqual(result["eval_metrics"], {"nll": 2.0})
        self.assertEqual(float(read_checkpoint(self.reference)["model_state"]["weight"].item()), 4.0)
        self.assertEqual(calls, [(step, split) for step in (2, 4, 6, 4) for split in ("train", "test_temporal")])
        history = json.loads((self.directory / "history.json").read_text())
        self.assertEqual([entry["step"] for entry in history], [2., 4., 6.])
        self.assertTrue(runtime.closed)

    def test_loop_closes_runtime_on_failure_or_interruption(self):
        for error in (RuntimeError("update failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error)):
                runtime = Mock()
                runtime.train_step.side_effect = error
                with self.assertRaises(type(error)):
                    run_training_loop(runtime, self.config, object())
                runtime.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
