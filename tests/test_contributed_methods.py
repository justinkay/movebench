import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from examples import candidate_method, origin_method
from movebench.bench import artifacts, config, planning, report, results, run, run_state
from movebench.bench.evaluation import runner as evaluation
from movebench.datasets.ssf import iSSFDataset
from movebench.geometry import LOCAL_GEOMETRY


class TinyGrid(Dataset):
    output_hw = (3, 3)
    pixel_size_m = 3.0
    collate = staticmethod(default_collate)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "point_id_start": torch.tensor(10 + 10 * index),
            "start_bearing_rad": torch.tensor(0.),
            "timestamp_ns": torch.tensor(1_700_000_000_000_000_000),
            "individual_id": "animal",
            "env": torch.ones(1, 3, 3),
            "valid": torch.ones(1, 3, 3, dtype=torch.bool),
        }


class ContributedMethodsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch.object(run.paths, "repo_root", return_value=self.root))
        self.enterContext(patch("movebench.study._list_available_studies", return_value=["study"]))
        self.enterContext(patch.dict(config.METHODS))
        # Loading this local example registers methods after the runner is imported.
        self.experiment = planning.load_experiment(str(
            Path(__file__).resolve().parents[1] / "examples/methods_experiment.py"
        ))
        self.enterContext(patch.object(config, "resolve_covariate_band_ids", return_value=["band"]))
        self.enterContext(patch.object(config, "resolve_timescale", return_value={
            "step_time_s": 43200, "layers_config": "all.yaml", "rasters_hash": "rasters",
            "sampler_config": "sampler.yaml", "sampler_hash": "sampler",
            "geometry": LOCAL_GEOMETRY, "output_hw": (3, 3),
        }))
        self.enterContext(patch.object(artifacts, "_study_timescale_readiness", return_value={
            "available_point_splits": ["test_temporal"],
            "early_stop_point_split": "test_temporal", "point_ready": True,
        }))
        self.manifest = SimpleNamespace(n_rows=2, sampler={"num_candidates": 2, "random_seed": 0})
        self.enterContext(patch.object(artifacts, "_resolve_point_artifacts", side_effect=
            lambda payload, **kwargs: artifacts._resolved_splits(payload, {
                "point_stores": {"train": {"manifest": self.manifest}},
            }, (3, 3))
        ))
        self.enterContext(patch.object(evaluation, "_selected_splits_for_split", return_value=None))
        self.enterContext(patch.object(evaluation, "_support_rows_for_split", return_value=([
            {"point_id_start": point, "step_index": index, "in_support": True,
             "truth_xy_m": torch.zeros(2, dtype=torch.float64)}
            for index, point in enumerate((10, 20))
        ], 3.)))

    def execute(self, method):
        self.experiment["methods"] = [method]
        payload, = planning.expand_experiment(self.experiment, studies_override=["study"])
        summary = run.execute_run_payload(payload)
        self.assertEqual(summary["status"], "completed", summary)
        frame = results.load_results(self.experiment["name"], expanded_experiments=[{
            "experiment": self.experiment, "runs": [payload],
        }])
        self.assertEqual(frame["method"].tolist(), [method])
        self.assertEqual(frame["n_steps_scored"].tolist(), [2])
        return payload, summary, frame

    def test_origin_example_plans_scores_loads_and_renders_without_core_edits(self):
        with patch.object(origin_method, "point_context", return_value={
            "point_id_start": torch.tensor([20, 10]), "start_bearing_rad": torch.ones(2),
        }):
            payload, summary, frame = self.execute("example_origin")
        self.assertEqual(summary["train_steps"], 0)
        self.assertEqual(frame["energy_score"].tolist(), [0.])
        entries = report.write_main_results_plots(
            frame, status_snapshot={"studies": [{
                "study_id": "study", "timescales": [{"held_out_splits": ["test_temporal"]}],
            }]}, report_root=self.root,
        )
        plotted, = [entry for entry in entries if entry["status"] == "plotted"]
        self.assertTrue((self.root / plotted["relative_path"]).is_file())
        self.assertNotIn("example_origin", report.METHOD_COLORS)

    def test_candidate_example_uses_model_batches_and_restores_without_updates(self):
        dataset = iSSFDataset.__new__(iSSFDataset)
        dataset.manifest, dataset.plan = self.manifest, object()
        dataset.c_env, dataset._n_individuals = 1, 0
        dataset.include_time, dataset.include_individual_id = True, False
        dataset.normalize, dataset._movement_stats = True, None
        dataset._band_ids = ["band"]
        dataset._mean, dataset._std = torch.tensor([10.]), torch.tensor([2.])
        raw_env = torch.arange(10., 22., 2.).view(2, 1, 3)
        shared = {
            "step_length_m": torch.tensor([[1., 2., 3.], [3., 2., 1.]]),
            "turn_angle_rad": torch.zeros(2, 3),
            "timestamp_ns": torch.full((2,), 1_700_000_000_000_000_000),
        }
        dataset.reader = SimpleNamespace(
            load_env_valid_rows=lambda **kw: (raw_env[kw["row_idxs"]], torch.ones_like(raw_env[kw["row_idxs"]], dtype=torch.bool)),
            load_shared_rows=lambda **kw: {key: shared[key][kw["row_idxs"]] for key in kw["keys"]},
        )
        batches = []
        assemble = iSSFDataset.make_model_batch

        def record(dataset, **kwargs):
            batch = assemble(dataset, **kwargs)
            batches.append(batch)
            return batch

        with (
            patch.object(candidate_method, "candidate_dataset", return_value=dataset),
            patch.object(candidate_method, "training_steps", return_value=(None, pd.DataFrame({
                "step_length_m": [1., 2.], "turn_angle_rad": [0., .5],
            }))),
            patch.object(candidate_method, "grid_dataset", return_value=TinyGrid()),
            patch.object(iSSFDataset, "make_model_batch", record),
        ):
            payload, summary, frame = self.execute("example_candidates")
            self.assertEqual(summary["train_steps"], 1)
            self.assertTrue(torch.isfinite(torch.tensor(frame["energy_score"].tolist())).all())
            self.assertEqual({batch["env"].shape[-1] for batch in batches}, {3, 9})
            for batch in batches:
                self.assertEqual(batch["time"].shape[-1], 4)
                if batch["env"].shape[-1] == 3:
                    torch.testing.assert_close(batch["env"].flatten().sort().values, torch.arange(6.))
                else:
                    torch.testing.assert_close(batch["env"], torch.ones_like(batch["env"]))
            with (
                patch.object(torch.optim, "Adam", side_effect=AssertionError("restoration optimized")),
                patch.object(candidate_method.CandidateMethod, "predict", side_effect=AssertionError("saved prediction regenerated")),
            ):
                restored = run.execute_run_payload(payload)
            self.assertEqual(restored, summary)
        directory = run_state.run_dir(payload["run_key"])
        prediction = torch.load(directory / "predictions/test_temporal.pt", weights_only=True)
        self.assertEqual(prediction["scores"].shape, (2, 3, 3))
        torch.testing.assert_close(prediction["point_id_start"], torch.tensor([10, 20]))


if __name__ == "__main__":
    unittest.main()
