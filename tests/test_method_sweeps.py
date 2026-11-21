import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from x_transformers.x_transformers import Attention

import movebench.bench.config as bench_config
import movebench.bench.executors.moveformer as moveformer_executor
import movebench.bench.artifacts as bench_artifacts
import movebench.bench.planning as bench_planning
from movebench.models.moveformer import build_moveformer_selection_model


class MethodSweepTest(unittest.TestCase):
    sweep_dir = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "sweeps"
    )

    def _load(self, filename):
        return bench_planning.load_experiment(str(self.sweep_dir / filename))

    def test_sweep_definitions(self):
        expected = {
            "issf_linear.py": ("issf_linear", 6),
            "issf_mlp.py": ("issf_mlp", 12),
            "moveformer.py": ("moveformer", 8),
        }
        checked_in = set(bench_planning.checked_in_experiment_module_paths())
        for filename, (method, count) in expected.items():
            with self.subTest(filename=filename):
                path = self.sweep_dir / filename
                experiment = self._load(filename)
                variants = experiment["method_variants"][method]
                self.assertEqual(experiment["methods"], [method])
                self.assertEqual(experiment["studies"], "mini")
                self.assertEqual(
                    experiment["timescales"],
                    ["1h", "12h", "1d", "7d"],
                )
                self.assertEqual(
                    experiment["covariate_sets"],
                    ["alphaearth_only"],
                )
                self.assertEqual(len(variants), count)
                self.assertEqual(
                    len({variant["key"] for variant in variants}),
                    count,
                )
                self.assertNotIn(str(path), checked_in)

        linear = self._load("issf_linear.py")["method_variants"]["issf_linear"]
        self.assertEqual(
            {
                (
                    variant["hparams"]["batch_size"],
                    variant["hparams"]["lr"],
                )
                for variant in linear
            },
            set(product([256, 1048], [3e-3, 1e-2, 3e-2])),
        )

        mlp = self._load("issf_mlp.py")["method_variants"]["issf_mlp"]
        self.assertEqual(
            {
                (
                    tuple(variant["model"]["hidden_dims"]),
                    variant["hparams"]["batch_size"],
                    variant["hparams"]["lr"],
                )
                for variant in mlp
            },
            {
                (hidden_dims, batch_size, learning_rate)
                for hidden_dims in [(64,), (128,), (128, 128)]
                for batch_size, learning_rate in [
                    (256, 3e-4),
                    (256, 1e-3),
                    (1048, 1e-3),
                    (1048, 3e-3),
                ]
            },
        )

        moveformer = {
            variant["key"]: variant
            for variant in self._load("moveformer.py")["method_variants"][
                "moveformer"
            ]
        }
        self.assertEqual(
            set(moveformer),
            {
                "current",
                "publication_schedule",
                "lr1e-5",
                "lr2e-4",
                "d2_h4",
                "d4_h8",
                "ctx64",
                "fullctx",
            },
        )
        self.assertEqual(
            moveformer["publication_schedule"]["hparams"],
            {
                "batch_size": 24,
                "min_len": 2,
                "warmup_steps": 1000,
                "lr_gamma": 0.98,
            },
        )
        self.assertEqual(moveformer["ctx64"]["hparams"], {"max_len": 64})
        self.assertEqual(
            moveformer["fullctx"]["model"],
            {"var_len_training_v2": False},
        )

    def test_variants_inherit_method_defaults(self):
        linear = self._load("issf_linear.py")
        linear_variant = bench_planning._expand_method_variants(
            linear,
            "issf_linear",
            bench_config.METHODS["issf_linear"],
        )[0]
        self.assertEqual(linear_variant["hparams"]["weight_decay"], 0.001)

        mlp = self._load("issf_mlp.py")
        mlp_variant = bench_planning._expand_method_variants(
            mlp,
            "issf_mlp",
            bench_config.METHODS["issf_mlp"],
        )[0]
        self.assertEqual(mlp_variant["model"]["dropout"], 0.6)

        moveformer = self._load("moveformer.py")
        variants = {
            variant["key"]: variant
            for variant in bench_planning._expand_method_variants(
                moveformer,
                "moveformer",
                bench_config.METHODS["moveformer"],
            )
        }
        self.assertEqual(
            variants["current"]["model"],
            {
                "depth": 6,
                "heads": 8,
                "var_len_training_v2": True,
            },
        )
        self.assertEqual(variants["current"]["hparams"]["max_len"], 64)
        self.assertEqual(variants["ctx64"]["model"], variants["current"]["model"])
        self.assertEqual(variants["ctx64"]["hparams"]["max_len"], 64)

        with self.assertRaisesRegex(ValueError, "Unknown moveformer model"):
            bench_config.normalize_experiment(
                {
                    "name": "bad_moveformer_variant",
                    "methods": ["moveformer"],
                    "method_variants": {
                        "moveformer": [
                            {"key": "bad", "model": {"width": 128}}
                        ]
                    },
                }
            )

    def test_theoretical_expansion_counts(self):
        studies = [f"mini-{index}" for index in range(10)]
        timescale = {
            "step_time_s": 3600,
            "alphaearth_layers_config": "alphaearth.yaml",
            "layers_config": "all.yaml",
            "rasters_hash": "rasters",
            "sampler_config": "sampler.yaml",
            "sampler_hash": "sampler",
            "geometry": "local",
        }
        readiness = {
            "available_point_splits": ["test_temporal"],
            "early_stop_point_split": "test_temporal",
            "point_ready": True,
        }
        expected = {
            "issf_linear.py": 240,
            "issf_mlp.py": 480,
            "moveformer.py": 320,
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
            for filename, expected_count in expected.items():
                with self.subTest(filename=filename):
                    runs = bench_planning.expand_experiment(self._load(filename))
                    self.assertEqual(len(runs), expected_count)
                    self.assertEqual(
                        len({run["run_key"] for run in runs}),
                        expected_count,
                    )
                    self.assertTrue(
                        all(
                            "__variant=" in run["run_key"]
                            and "method_variant" in run
                            for run in runs
                        )
                    )


class MoveFormerConfigurationTest(unittest.TestCase):
    def test_builder_applies_architecture_and_context_settings(self):
        baseline = build_moveformer_selection_model(cov_dim=4)
        compact = build_moveformer_selection_model(
            cov_dim=4,
            depth=2,
            heads=4,
            var_len_training_v2=False,
            max_len=64,
        )

        self.assertEqual(baseline.encoder.depth, 6)
        self.assertEqual(compact.encoder.depth, 2)
        self.assertEqual(
            {
                module.heads
                for module in compact.encoder.modules()
                if isinstance(module, Attention)
            },
            {4},
        )
        self.assertTrue(baseline.var_len_training_v2)
        self.assertFalse(compact.var_len_training_v2)
        self.assertEqual(compact.max_len, 64)
        self.assertLess(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in baseline.parameters()),
        )

    def test_executor_forwards_resolved_model_settings(self):
        class FakeDataset:
            cov_dim = 7
            num_candidates_total = 101
            plan = object()
            _mov_vec_scale = 1.0

            def __len__(self):
                return 1

        method_hparams = dict(bench_config.METHODS["moveformer"]["hparams"])
        method_hparams["max_len"] = 64
        run_payload = {
            "method": "moveformer",
            "method_model": {
                "depth": 2,
                "heads": 4,
                "var_len_training_v2": False,
            },
            "method_hparams": method_hparams,
            "seed": 0,
            "early_stop_split_key": "test_temporal",
            "evaluation_split_keys": [],
            "band_ids": ["alphaearth.A00"],
            "timescale": "1h",
            "study_id": "study",
            "covariate_set": "alphaearth_only",
        }
        resolved_artifacts = {
            "normalization_stats_path": "stats.json",
            "point_stores": {},
            "study_dir": "/tmp/study",
            "trajectories_path": "/tmp/study/trajectories.csv",
            "plan": object(),
            "sampler": object(),
        }
        model = torch.nn.Linear(1, 1)
        runtime = SimpleNamespace(model=model)
        metrics = {"nll": 1.0, "nce": 1.0, "num_tokens": 1.0}

        with (
            tempfile.TemporaryDirectory() as run_dir,
            patch.object(
                moveformer_executor,
                "MoveFormerPointStoreDataset",
                return_value=FakeDataset(),
            ),
            patch.object(moveformer_executor, "_loader", return_value=object()),
            patch.object(
                moveformer_executor,
                "build_moveformer_selection_model",
                return_value=model,
            ) as build_model,
            patch.object(
                moveformer_executor,
                "_MoveFormerRuntime",
                return_value=runtime,
            ),
            patch.object(
                moveformer_executor,
                "run_training_loop",
                return_value={"completed_steps": 10, "best_step": 10},
            ),
            patch.object(
                moveformer_executor,
                "_eval_metrics",
                return_value=metrics,
            ),
            patch.object(
                moveformer_executor.bench_config,
                "resolve_timescale",
                return_value={"output_hw": (101, 101)},
            ),
            patch.object(
                moveformer_executor.torch.cuda,
                "is_available",
                return_value=False,
            ),
            patch.object(moveformer_executor.torch, "save"),
        ):
            result = moveformer_executor.run(
                run_payload=run_payload,
                resolved_artifacts=resolved_artifacts,
                run_dir=run_dir,
                tracker=object(),
            )

        build_model.assert_called_once_with(
            cov_dim=7,
            max_len=64,
            depth=2,
            heads=4,
            var_len_training_v2=False,
        )
        self.assertEqual(result["status"], "completed")

if __name__ == "__main__":
    unittest.main()
