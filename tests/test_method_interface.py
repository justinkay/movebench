import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import torch

from movebench.bench import run
from movebench.datasets import DataSplit
from movebench.models import baselines, deepssf, moveformer, ssf


class MethodInterfaceTest(unittest.TestCase):
    def test_constructors_only_remember_settings(self):
        constructors = (
            baselines.NoMove, baselines.ConstantVelocity, baselines.RandomWalk,
            ssf.linear_method, ssf.mlp_method, deepssf.DeepSSFMethod,
            moveformer.MoveFormerMethod, moveformer.SharedMoveFormerTrainer,
            moveformer.SharedMoveFormerScorer,
        )
        with (
            patch.object(baselines, "training_steps") as steps,
            patch.object(ssf, "candidate_dataset") as candidates,
            patch.object(deepssf, "PreparedRasters") as rasters,
            patch.object(moveformer, "sequence_dataset") as sequences,
            patch.object(torch.cuda, "is_available") as cuda,
        ):
            state = torch.random.get_rng_state()
            for constructor in constructors:
                with self.subTest(constructor=constructor):
                    config, hparams = {"setting": 3}, {"batch_size": 2}
                    method = constructor(model_config=config, hparams=hparams, seed=7)
                    self.assertEqual(method.model_config, config)
                    self.assertEqual(method.hparams, hparams)
                    self.assertIsNot(method.model_config, config)
                    self.assertIsNot(method.hparams, hparams)
                    self.assertEqual(method.seed, 7)
                    self.assertFalse(hasattr(method, "model_"))
            torch.testing.assert_close(torch.random.get_rng_state(), state)
        for operation in (steps, candidates, rasters, sequences, cuda):
            operation.assert_not_called()

    def test_random_walk_prepares_distribution_once_for_multiple_predictions(self):
        data = DataSplit("train", {}, {"plan": object(), "point_stores": {
            "train": {"manifest": object()},
        }})
        state = {"num_candidates": 10}
        with (
            patch.object(baselines, "training_steps", return_value=(None, pd.DataFrame({
                "step_length_m": [1.], "turn_angle_rad": [0.],
            }))) as steps,
            patch.object(baselines.ssf_grid, "_replay_sampler_state", return_value=state) as replay,
            patch.object(baselines, "grid_dataset") as grid,
            patch.object(baselines.ssf_grid, "build_ssf_grid_artifact", return_value={}) as predict,
        ):
            method = baselines.RandomWalk(model_config={}, hparams={}, seed=0)
            self.assertIs(method.fit(data, None, work_dir=Path("unused")), method)
            for split in ("test_temporal", "test_individual"):
                method.predict(DataSplit(split, data.metadata, data.inputs))
        steps.assert_called_once_with(data)
        replay.assert_called_once()
        self.assertEqual(grid.call_count, 2)
        for call in grid.call_args_list:
            self.assertFalse(call.kwargs["include_env"])
        for call in predict.call_args_list:
            self.assertIs(call.kwargs["sampler_state"], state)

    def test_issf_defers_prediction_support_and_reuses_split_datasets(self):
        train = DataSplit("train", {}, {"point_stores": {"train": {"manifest": object()}}})
        validation = DataSplit("test_temporal", {}, train.inputs)
        dataset = SimpleNamespace(c_env=2, include_time=False, n_individuals=0, plan=object())
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(torch.multiprocessing, "set_sharing_strategy") as sharing,
            patch.object(ssf, "candidate_dataset", return_value=dataset) as candidates,
            patch.object(ssf, "candidate_loader", return_value=[]),
            patch.object(ssf, "candidate_eval_loader", return_value=[]),
            patch.object(ssf, "fit_or_restore", return_value={
                "completed_steps": 2, "best_step": 1,
                "train_metrics": {}, "eval_metrics": {},
            }),
            patch.object(ssf, "training_steps", return_value=(None, pd.DataFrame({
                "step_length_m": [1.], "turn_angle_rad": [0.],
            }))) as steps,
            patch.object(ssf.ssf_grid, "_replay_sampler_state", return_value={}) as replay,
            patch.object(ssf, "grid_dataset"),
            patch.object(ssf.ssf_grid, "build_ssf_grid_artifact", return_value={}),
        ):
            method = ssf.linear_method(model_config={}, seed=0, hparams={
                "lr": .01, "weight_decay": 0., "max_steps": 2,
                "eval_every_steps": 1, "patience_evals": 2,
            })
            method.fit(train, validation, work_dir=Path(directory))
            sharing.assert_called_once_with("file_system")
            steps.assert_not_called()
            self.assertFalse(hasattr(method, "runtime_"))
            method.predict(validation)
            method.predict(validation)
        self.assertEqual(candidates.call_count, 2)
        steps.assert_called_once_with(train)
        replay.assert_called_once()

    def test_runner_owns_saved_predictions_and_reuses_valid_files(self):
        validation = DataSplit("test_temporal", {}, {})
        prediction = {"prediction_kind": "grid", "point_id_start": torch.tensor([8]),
                      "scores": torch.zeros(1, 2, 2)}
        method = Mock()
        method.predict.return_value = prediction
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(run_payload={}, run_dir=Path(directory))
            with patch.object(run.evaluation_runner, "_score_split", return_value=([], {})) as score:
                run._predict_and_score(method, validation, **kwargs)
                run._predict_and_score(method, validation, **kwargs)
                self.assertEqual(score.call_count, 2)
            path = Path(directory) / "predictions/test_temporal.pt"
            torch.testing.assert_close(torch.load(path)["scores"], prediction["scores"])
            method.predict.assert_called_once_with(validation)
            with patch.object(run.evaluation_runner, "_score_split", side_effect=ValueError("invalid")):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    run._predict_and_score(method, validation, **kwargs)
            self.assertEqual(method.predict.call_count, 2)
            method.fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
