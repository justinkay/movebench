import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import movebench.bench.config as bench_config
import movebench.models.deepssf as deepssf_model
import movebench.bench.artifacts as bench_artifacts
import movebench.bench.planning as bench_planning
from movebench.training import TrainerConfig, run_training_loop
from movebench.datasets.deepssf_wds_cache import steps_per_epoch_from_manifest


class DeepSSFObjectiveTest(unittest.TestCase):
    def test_evaluation_reports_combined_and_component_nll(self):
        output = torch.tensor([[[[0.0, 0.0], [1.0, 2.0]]]])

        class FixedModel(torch.nn.Module):
            def forward(self, _inputs):
                return output

        spatial = torch.zeros((1, 1, 1, 2))
        scalars = torch.zeros((1, 4))
        bearing = torch.zeros((1, 1))
        target = torch.tensor([[[0.0, 1.0]]])
        loader = [(spatial, scalars, bearing, target, torch.tensor([1]))]

        metrics = deepssf_model._evaluate_loader(
            FixedModel(),
            loader,
            device=torch.device("cpu"),
            channel_indices=None,
        )
        true_idx = torch.tensor([1])
        self.assertAlmostEqual(
            metrics["nll_habitat"],
            F.cross_entropy(torch.tensor([[0.0, 1.0]]), true_idx).item(),
        )
        self.assertAlmostEqual(
            metrics["nll_movement"],
            F.cross_entropy(torch.tensor([[0.0, 2.0]]), true_idx).item(),
        )
        self.assertAlmostEqual(
            metrics["nll"],
            F.cross_entropy(torch.tensor([[0.0, 3.0]]), true_idx).item(),
        )

    def test_component_schedulers_decay_independently(self):
        class TwoPartModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.movement = torch.nn.Parameter(torch.tensor(0.0))
                self.habitat = torch.nn.Parameter(torch.tensor(0.0))

            def parameter_groups(self):
                return {
                    "movement": [self.movement],
                    "habitat": [self.habitat],
                }

        runtime = deepssf_model._DeepSSFRuntime(
            model=TwoPartModel(),
            train_loader=[],
            eval_loaders={},
            early_stop_split_key="test_temporal",
            device=torch.device("cpu"),
            lr_movement=1e-3,
            lr_habitat=1e-2,
            weight_decay=0.0,
            steps_per_epoch=3,
            lr_scheduler_factor=0.1,
            lr_scheduler_patience_epochs=0,
            channel_indices=[],
            eval_max_items=None,
        )
        runtime.step_lr_schedulers(
            {"nll_movement": 1.0, "nll_habitat": 1.0}
        )
        rates = runtime.step_lr_schedulers(
            {"nll_movement": 1.1, "nll_habitat": 0.9}
        )
        self.assertAlmostEqual(rates["lr_movement"], 1e-4)
        self.assertAlmostEqual(rates["lr_habitat"], 1e-2)
        self.assertTrue(runtime.lr_scheduler_due(3))
        self.assertFalse(runtime.lr_scheduler_due(4))
        self.assertEqual(
            set(runtime.state_dict()),
            {"model", "optimizers", "lr_schedulers"},
        )

    def test_wds_epoch_size_uses_manifest_sample_count(self):
        self.assertEqual(
            steps_per_epoch_from_manifest({"num_samples": 65}, batch_size=32),
            3,
        )


class TrainingScheduleTest(unittest.TestCase):
    def test_scheduler_events_share_evaluation_and_do_not_advance_patience(self):
        class Runtime:
            early_stop_metric = "nll"
            early_stop_split_key = "test_temporal"

            def __init__(self):
                self.step = 0
                self.final = False
                self.evaluate_calls = []
                self.scheduler_steps = []
                self.state_dict_calls = 0

            def train_step(self, step):
                self.step = step
                return {"loss": 0.0}

            def lr_scheduler_due(self, step):
                return step % 3 == 0

            def step_lr_schedulers(self, _metrics):
                self.scheduler_steps.append(self.step)
                return {"lr_movement": 1e-5, "lr_habitat": 1e-4}

            def evaluate(self, split):
                self.evaluate_calls.append((self.step, split, self.final))
                return {
                    "nll": 1.0,
                    "nll_movement": 0.4,
                    "nll_habitat": 0.6,
                }

            def state_dict(self):
                self.state_dict_calls += 1
                return {"step": self.step}

            def load_state_dict(self, _state):
                self.final = True

        runtime = Runtime()
        result = run_training_loop(
            runtime,
            TrainerConfig(
                max_steps=12,
                eval_every_steps=4,
                patience_evals=3,
            ),
            object(),
        )

        self.assertFalse(result["stopped_early"])
        self.assertEqual(result["best_step"], 4)
        self.assertEqual(runtime.scheduler_steps, [3, 6, 9, 12])
        self.assertEqual(
            [int(entry["step"]) for entry in result["history"]],
            [3, 4, 6, 8, 9, 12],
        )
        training_calls = [
            (step, split)
            for step, split, final in runtime.evaluate_calls
            if not final
        ]
        self.assertEqual(
            training_calls,
            [
                (3, "test_temporal"),
                (4, "train"),
                (4, "test_temporal"),
                (6, "test_temporal"),
                (8, "train"),
                (8, "test_temporal"),
                (9, "test_temporal"),
                (12, "train"),
                (12, "test_temporal"),
            ],
        )
        self.assertEqual(runtime.state_dict_calls, 1)


class DeepSSFSweepTest(unittest.TestCase):
    sweep_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "sweeps"
        / "deepssf.py"
    )

    def test_deepssf_defaults_and_variant_inheritance(self):
        method = bench_config.METHODS["deepssf"]
        self.assertEqual(
            method["model"],
            {
                "image_dim": 101,
                "output_channels": 4,
                "dense_dim_hidden": 128,
                "dropout": 0.1,
            },
        )
        self.assertEqual(method["hparams"]["lr_scheduler_factor"], 0.1)
        self.assertEqual(method["hparams"]["lr_scheduler_patience_epochs"], 5)

        experiment = bench_config.normalize_experiment(
            {
                "name": "variants",
                "methods": ["deepssf"],
                "method_variants": {
                    "deepssf": [
                        {
                            "key": "small",
                            "model": {"output_channels": 2},
                            "hparams": {"lr_movement": 3e-6},
                        },
                        {
                            "key": "large",
                            "model": {"dense_dim_hidden": 256},
                        },
                    ]
                },
            }
        )
        variants = bench_planning._expand_method_variants(
            experiment,
            "deepssf",
            method,
        )
        self.assertEqual([variant["key"] for variant in variants], ["small", "large"])
        self.assertEqual(variants[0]["model"]["output_channels"], 2)
        self.assertEqual(variants[0]["model"]["dense_dim_hidden"], 128)
        self.assertEqual(variants[0]["hparams"]["batch_size"], 32)
        self.assertEqual(variants[1]["model"]["output_channels"], 4)

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bench_config.normalize_experiment(
                {
                    "name": "duplicates",
                    "methods": ["deepssf"],
                    "method_variants": {
                        "deepssf": [{"key": "same"}, {"key": "same"}]
                    },
                }
            )

    def test_sweep_has_15_variants_and_600_theoretical_runs(self):
        experiment = bench_planning.load_experiment(str(self.sweep_path))
        variants = experiment["method_variants"]["deepssf"]
        self.assertEqual(len(variants), 15)
        self.assertEqual(len({variant["key"] for variant in variants}), 15)
        self.assertNotIn(
            str(self.sweep_path),
            bench_planning.checked_in_experiment_module_paths(),
        )

        studies = [f"mini-{index}" for index in range(10)]
        timescale = {
            "step_time_s": 3600,
            "alphaearth_layers_config": "alphaearth.yaml",
            "layers_config": "all.yaml",
            "rasters_hash": "rasters",
            "sampler_config": "sampler.yaml",
            "sampler_hash": "sampler",
            "geometry": "local",
            "plan": SimpleNamespace(
                channels=[
                    SimpleNamespace(
                        band_id="alphaearth.A00",
                        layer=bench_planning.ALPHAEARTH_LAYER,
                    )
                ]
            ),
        }
        readiness = {
            "has_train": True,
            "held_out_splits": ["test_temporal"],
            "available_point_splits": ["test_temporal"],
            "point_ready": True,
        }
        with (
            patch.object(bench_planning, "_resolve_studies", return_value=studies),
            patch.object(
                bench_artifacts,
                "_study_timescale_readiness",
                return_value=readiness,
            ),
            patch.object(
                bench_planning.bench_config,
                "resolve_covariate_band_ids",
                return_value=["alphaearth.A00"],
            ),
            patch.object(
                bench_planning.bench_config,
                "resolve_timescale",
                return_value=timescale,
            ),
            patch.object(
                bench_artifacts,
                "_resolve_artifacts_for_run",
                return_value={"status": "resolved"},
            ),
        ):
            runs = bench_planning.expand_experiment(experiment)

        self.assertEqual(len(runs), 10 * 4 * 15)
        self.assertEqual(len({run["run_key"] for run in runs}), len(runs))
        self.assertEqual(len({run["method_variant"] for run in runs}), 15)
        self.assertTrue(
            all(
                "__variant=" in run["run_key"]
                and run["method_hparams"]["batch_size"] == 32
                and run["method_hparams"]["lr_scheduler_patience_epochs"] == 5
                for run in runs
            )
        )


if __name__ == "__main__":
    unittest.main()
