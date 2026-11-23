import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import from_bounds

import movebench.bench.config as bench_config
import movebench.models.deepssf as deepssf_model
import movebench.bench.artifacts as bench_artifacts
import movebench.bench.planning as bench_planning
import movebench.datasets.rasters as raster_data
from movebench.datasets.chips import ChipDataset
from movebench.datasets.deepssf import DeepSSFDataset
from movebench.datasets.deepssf_gpu import (
    ALPHAEARTH_LAYER,
    AlphaEarthCatalog,
    AlphaEarthGPUCache,
    AlphaEarthGroup,
    DeepSSFGPUMetadataDataset,
    _DeepSSFGPUCollator,
    alphaearth_query_geometry,
    build_deepssf_gpu_datasets,
)
from movebench.datasets.types import DataSplit
from movebench.geometry import lonlat_from_local_xy_m
from movebench.layers import REGISTRY
from movebench.layers.config import FixedStepSpec, load_layers_config
from movebench.layers.plan import Channel, LayerPlan
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
)


def _plan() -> LayerPlan:
    return LayerPlan(
        rasters_hash="test",
        layers_in_order=[ALPHAEARTH_LAYER],
        channels=[
            Channel(
                band_id=f"{ALPHAEARTH_LAYER}.A00",
                layer=ALPHAEARTH_LAYER,
                categorical=False,
                normalize=False,
            )
        ],
        chip_groups={},
        derived_dependencies={},
        step=FixedStepSpec(
            step_time_s=3600,
            step_multiplier=2.0,
            step_q=0.95,
            max_step_q=0.99,
            gap_q=0.999,
        ),
    )


def _readiness() -> dict[str, object]:
    return {
        "has_train": True,
        "held_out_splits": ["test_temporal"],
        "available_point_splits": ["test_temporal"],
        "early_stop_point_split": "test_temporal",
        "point_ready": True,
        "raster_blocked_reasons": [],
        "point_blocked_reasons": [],
    }


class DeepSSFGPUConfigTest(unittest.TestCase):
    def test_config_defaults_to_wds_and_validates_backend(self):
        normalized = bench_config.normalize_experiment({"name": "test"})
        self.assertEqual(normalized["deepssf_data"], "wds")
        self.assertEqual(
            bench_config.normalize_experiment(
                {"name": "test", "deepssf_data": "GPU"}
            )["deepssf_data"],
            "gpu",
        )
        with self.assertRaisesRegex(ValueError, "deepssf_data"):
            bench_config.normalize_experiment(
                {"name": "test", "deepssf_data": "disk"}
            )

    def test_gpu_planning_probes_point_stores_but_not_wds(self):
        experiment = {
            "name": "gpu",
            "studies": ["study"],
            "timescales": ["1h"],
            "covariate_sets": ["alphaearth_only"],
            "methods": ["deepssf"],
            "deepssf_data": "gpu",
        }

        def point_state(_study_id, _timescale, split):
            if split == "test_individual":
                return {"available": False, "reason": "zero valid steps"}
            return {"available": True}

        with (
            patch.object(bench_planning, "_resolve_studies", return_value=["study"]),
            patch.object(
                bench_artifacts,
                "_study_split_keys",
                return_value={
                    "has_train": True,
                    "held_out_splits": ["test_temporal", "test_individual"],
                },
            ),
            patch.object(
                bench_artifacts,
                "_alphaearth_tile_state",
                return_value={"available": True},
            ),
            patch.object(
                bench_artifacts,
                "_deepssf_cache_state",
                side_effect=AssertionError("WDS probe"),
            ),
            patch.object(
                bench_artifacts,
                "_point_store_state",
                side_effect=point_state,
            ),
            patch.object(
                bench_artifacts,
                "_resolve_artifacts_for_run",
                return_value={"status": "resolved"},
            ),
        ):
            runs = bench_planning.expand_experiment(experiment)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["artifact_requirement"], "deepssf_gpu")
        self.assertEqual(runs[0]["deepssf_data"], "gpu")
        self.assertEqual(
            runs[0]["evaluation_split_keys"], ["test_temporal"]
        )
        self.assertEqual(
            runs[0]["selected_splits"], ["train", "test_temporal"]
        )

    def test_gpu_blocks_only_incompatible_deepssf_runs(self):
        experiment = {
            "name": "mixed",
            "studies": ["study"],
            "timescales": ["1h"],
            "covariate_sets": ["all"],
            "methods": ["deepssf", "baseline_nomove"],
            "deepssf_data": "gpu",
        }
        with (
            patch.object(bench_planning, "_resolve_studies", return_value=["study"]),
            patch.object(
                bench_artifacts, "_study_timescale_readiness", return_value=_readiness()
            ),
            patch.object(
                bench_artifacts,
                "_resolve_artifacts_for_run",
                return_value={"status": "resolved"},
            ),
        ):
            runs = bench_planning.expand_experiment(experiment)
        self.assertEqual([run["method"] for run in runs], ["baseline_nomove"])
        self.assertEqual(len(experiment["blocked"]), 1)
        self.assertIn("only supports AlphaEarth", experiment["blocked"][0]["reason"])
        self.assertNotIn("deepssf_data", runs[0])

    def test_default_wds_run_payload_is_unchanged(self):
        experiment = {
            "name": "wds",
            "studies": ["study"],
            "timescales": ["1h"],
            "covariate_sets": ["alphaearth_only"],
            "methods": ["deepssf"],
        }
        with (
            patch.object(bench_planning, "_resolve_studies", return_value=["study"]),
            patch.object(
                bench_artifacts, "_study_timescale_readiness", return_value=_readiness()
            ),
            patch.object(
                bench_artifacts,
                "_resolve_artifacts_for_run",
                return_value={"status": "resolved"},
            ),
        ):
            runs = bench_planning.expand_experiment(experiment)
        self.assertEqual(runs[0]["artifact_requirement"], "deepssf_cache")
        self.assertNotIn("deepssf_data", runs[0])
        self.assertNotIn("selected_splits", runs[0])


class DeepSSFGPUExecutorTest(unittest.TestCase):
    def test_training_evaluation_and_prediction_do_not_call_wds(self):
        class FakeDataset:
            pixel_size_m = 100.0

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.movement = torch.nn.Parameter(torch.tensor(0.0))
                self.habitat = torch.nn.Parameter(torch.tensor(0.0))

            def parameter_groups(self):
                return {
                    "movement": [self.movement],
                    "habitat": [self.habitat],
                }

            def forward(self, inputs):
                spatial, _scalars, _bearing = inputs
                batch, _channels, height, width = spatial.shape
                score = (self.movement + self.habitat).expand(
                    batch, height, width
                )
                return torch.stack((score, torch.zeros_like(score)), dim=-1)

        spatial = torch.zeros((1, 64, 5, 5), dtype=torch.float32)
        scalars = torch.zeros((1, 4), dtype=torch.float32)
        bearing = torch.zeros((1, 1), dtype=torch.float32)
        target = torch.zeros((1, 5, 5), dtype=torch.float32)
        target[0, 2, 2] = 1.0
        point_id = torch.tensor([7], dtype=torch.int64)
        loader = [(spatial, scalars, bearing, target, point_id)]
        run_payload = {
            "study_id": "study",
            "method": "deepssf",
            "covariate_set": "alphaearth_only",
            "seed": 0,
            "deepssf_data": "gpu",
            "selected_splits": ["train", "test_temporal"],
            "evaluation_split_keys": ["test_temporal"],
            "early_stop_split_key": "test_temporal",
            "method_model": {
                "image_dim": 5,
                "output_channels": 4,
                "dense_dim_hidden": 128,
                "dropout": 0.1,
            },
            "method_hparams": {
                "batch_size": 1,
                "max_eval_items": 10,
                "max_steps": 1,
                "eval_every_steps": 1,
                "patience_evals": 1,
                "lr_movement": 1e-5,
                "lr_habitat": 1e-4,
                "weight_decay": 0.0,
                "lr_scheduler_factor": 0.1,
                "lr_scheduler_patience_epochs": 5,
            },
        }
        inputs = {
            "kind": "gpu",
            "trajectories_path": Path("unused.csv"),
            "plan": object(),
            "catalog": object(),
            "channel_indices": [0],
            "selected_splits": ("train", "test_temporal"),
        }
        resolved_artifacts = {
            "splits": {
                "train": DataSplit(
                    "train", {"study_id": "study", "output_hw": (5, 5)}, inputs
                )
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    deepssf_model,
                    "_device_for_backend",
                    return_value=torch.device("cpu"),
                ),
                patch.object(
                    raster_data,
                    "build_deepssf_gpu_datasets",
                    return_value={
                        "train": FakeDataset(),
                        "test_temporal": FakeDataset(),
                    },
                ),
                patch.object(raster_data, "AlphaEarthGPUCache") as gpu_cache,
                patch.object(
                    raster_data,
                    "make_deepssf_gpu_loader",
                    return_value=loader,
                ) as make_gpu_loader,
                patch.object(
                    raster_data,
                    "make_deepssf_wds_loader",
                    side_effect=AssertionError("WDS loader"),
                ),
                patch.object(
                    deepssf_model,
                    "DeepSSF",
                    side_effect=lambda *_args, **_kwargs: FakeModel(),
                ) as deep_ssf,
            ):
                method = deepssf_model.DeepSSFMethod(
                    model_config=run_payload["method_model"],
                    hparams=run_payload["method_hparams"],
                    seed=run_payload["seed"],
                )
                train = resolved_artifacts["splits"]["train"]
                validation = DataSplit("test_temporal", train.metadata, train.inputs)
                self.assertIs(
                    method.fit(train, validation, work_dir=Path(tmpdir)), method
                )
                prediction = method.predict(validation)
                result = method.training_summary_
            self.assertFalse((Path(tmpdir) / "predictions").exists())
            gpu_cache.assert_called_once()
            self.assertEqual(make_gpu_loader.call_count, 4)
            self.assertEqual(result["schedule"]["steps_per_epoch"], 1)
            self.assertEqual(result["schedule"]["lr_scheduler_factor"], 0.1)
            self.assertEqual(
                result["schedule"]["lr_scheduler_patience_epochs"],
                5,
            )
            self.assertEqual(deep_ssf.call_args.kwargs["dense_dim_hidden"], 128)
            self.assertEqual(prediction["point_id_start"].tolist(), [7])


class AlphaEarthCatalogTest(unittest.TestCase):
    def test_static_and_annual_catalogs_and_earlier_year_tie(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            static = root / "static"
            static.mkdir()
            (static / "tile.tif").touch()
            (static / "index.json").write_text(
                json.dumps(
                    {
                        "tiles": [
                            {
                                "file": "tile.tif",
                                "bounds": {
                                    "min_lon": -1,
                                    "min_lat": -1,
                                    "max_lon": 1,
                                    "max_lat": 1,
                                },
                            }
                        ]
                    }
                )
            )
            annual = root / "annual"
            annual.mkdir()
            for year in (2019, 2021):
                (annual / f"tile_{year}.tif").touch()
                (annual / f"index_{year}.json").write_text(
                    json.dumps(
                        {
                            "tiles": [
                                {
                                    "file": f"tile_{year}.tif",
                                    "bounds": {
                                        "min_lon": -1,
                                        "min_lat": -1,
                                        "max_lon": 1,
                                        "max_lat": 1,
                                    },
                                }
                            ]
                        }
                    )
                )
            catalog = AlphaEarthCatalog.from_tile_directories(
                [("static", static), ("annual", annual)]
            )
            self.assertEqual(catalog.tile_count, 3)
            self.assertEqual(catalog.resolve_group_id("static", 2030), 0)
            tied = catalog.resolve_group_id("annual", 2020)
            self.assertEqual(catalog.groups[tied].year, 2019)

    def test_catalog_rejects_missing_indexed_tile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "tiles": [
                            {
                                "file": "missing.tif",
                                "bounds": {
                                    "min_lon": -1,
                                    "min_lat": -1,
                                    "max_lon": 1,
                                    "max_lat": 1,
                                },
                            }
                        ]
                    }
                )
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing"):
                AlphaEarthCatalog.from_tile_directories([("study", root)])


class DeepSSFGPUMetadataTest(unittest.TestCase):
    def test_query_geometry_is_centered_at_high_latitude_and_dateline(self):
        longitude, latitude, bounds = alphaearth_query_geometry(
            np.array([[179.999, 75.0]]),
            1_000.0,
            (5, 5),
        )
        self.assertAlmostEqual(longitude[0, 2, 2], 179.999, places=9)
        self.assertAlmostEqual(latitude[0, 2, 2], 75.0, places=9)
        self.assertTrue(np.isfinite(longitude).all())
        self.assertTrue(np.isfinite(latitude).all())
        self.assertLess(bounds[0, 0], 179.999)
        self.assertGreater(bounds[0, 2], 179.999)

    def test_metadata_target_time_and_bearing_contract(self):
        end_lon, end_lat = lonlat_from_local_xy_m(10.0, 75.0, 1_000.0, 1_000.0)
        steps = pd.DataFrame(
            {
                "point_id_start": [7],
                "timestamp": [pd.Timestamp("2020-06-01T06:00:00Z")],
                "start_lon": [10.0],
                "start_lat": [75.0],
                "end_lon": [float(end_lon)],
                "end_lat": [float(end_lat)],
                "start_bearing": [0.0],
            }
        )
        catalog = AlphaEarthCatalog(
            (AlphaEarthGroup(0, "study", None, ()),)
        )
        dataset = DeepSSFGPUMetadataDataset(
            study_id="study",
            steps=steps,
            catalog=catalog,
            output_hw=(5, 5),
            step_buffer_m=2_000.0,
        )
        batch = _DeepSSFGPUCollator(dataset)([0])
        self.assertEqual(batch["point_id_start"].tolist(), [7])
        self.assertEqual(torch.nonzero(batch["target"][0]).tolist(), [[1, 3]])
        torch.testing.assert_close(
            batch["bearing"], torch.tensor([[np.pi / 2]], dtype=torch.float32)
        )
        torch.testing.assert_close(
            batch["scalars"][0, :2],
            torch.tensor([1.0, 0.0]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_step_selection_matches_chip_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trajectories_path = root / "trajectories.csv"
            trajectory = pd.DataFrame(
                {
                    POINT_ID_COL: np.arange(7),
                    TIME_COL: pd.date_range("2020-01-01", periods=7, freq="h"),
                    LON_COL: np.arange(7) * 0.001,
                    LAT_COL: np.zeros(7),
                    SET_COL: ["train"] * 4 + ["test_temporal"] * 3,
                    INDIVIDUAL_COL: ["animal"] * 7,
                }
            )
            trajectory.to_csv(trajectories_path, index=False)
            catalog = AlphaEarthCatalog(
                (AlphaEarthGroup(0, "study", None, ()),)
            )
            selected_splits = ["train", "test_temporal"]
            chips = ChipDataset(
                study_dir=root,
                trajectories_path=trajectories_path,
                split="test_temporal",
                plan=_plan(),
                output_hw=(5, 5),
                normalize=False,
                mode="steps",
                selected_splits=selected_splits,
            )
            cpu_dataset = DeepSSFDataset(chips)
            gpu_dataset = build_deepssf_gpu_datasets(
                study_id="study",
                trajectories_path=trajectories_path,
                plan=_plan(),
                splits=selected_splits,
                catalog=catalog,
                output_hw=(5, 5),
            )["test_temporal"]
            self.assertEqual(
                gpu_dataset.point_ids.tolist(),
                chips.step_starts_df["point_id_start"].tolist(),
            )
            self.assertAlmostEqual(gpu_dataset.pixel_size_m, chips.pixel_size_m)
            torch.testing.assert_close(gpu_dataset.scalars, cpu_dataset._time)
            torch.testing.assert_close(gpu_dataset.bearing[:, 0], cpu_dataset._bearing)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AlphaEarthGPUParityTest(unittest.TestCase):
    def test_single_tile_matches_chip_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            plan = REGISTRY.resolve(
                load_layers_config("configs/layers/alphaearth_only_1h_1h.yaml")
            )
            tile_dir = (
                study_dir
                / "rasters"
                / plan.rasters_hash
                / "tiles"
                / ALPHAEARTH_LAYER
            )
            tile_dir.mkdir(parents=True)
            tile_path = tile_dir / "tile_2020.tif"
            height = width = 100
            rows, cols = np.meshgrid(
                np.arange(height, dtype=np.float32),
                np.arange(width, dtype=np.float32),
                indexing="ij",
            )
            values = np.stack(
                [0.01 * band + 0.001 * rows + 0.002 * cols for band in range(64)]
            ).astype(np.float32)
            values[:, 48:50, 48:50] = -9999.0
            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=64,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_bounds(-0.1, -0.1, 0.1, 0.1, width, height),
                nodata=-9999.0,
            ) as destination:
                destination.write(values)
            (tile_dir / "index_2020.json").write_text(
                json.dumps(
                    {
                        "tiles": [
                            {
                                "file": tile_path.name,
                                "bounds": {
                                    "min_lon": -0.1,
                                    "min_lat": -0.1,
                                    "max_lon": 0.1,
                                    "max_lat": 0.1,
                                },
                            }
                        ]
                    }
                )
            )
            trajectories_path = study_dir / "trajectories.csv"
            pd.DataFrame(
                {
                    POINT_ID_COL: np.arange(6),
                    TIME_COL: pd.date_range("2020-01-01", periods=6, freq="h"),
                    LON_COL: np.arange(6) * 0.001,
                    LAT_COL: np.zeros(6),
                    SET_COL: ["train"] * 4 + ["test_temporal"] * 2,
                    INDIVIDUAL_COL: ["animal"] * 6,
                }
            ).to_csv(trajectories_path, index=False)
            selected_splits = ["train", "test_temporal"]
            chips = ChipDataset(
                study_dir=study_dir,
                trajectories_path=trajectories_path,
                split="test_temporal",
                plan=plan,
                output_hw=(5, 5),
                normalize=False,
                mode="steps",
                selected_splits=selected_splits,
            )
            cpu_values, cpu_valid, *_ = chips[0]
            catalog = AlphaEarthCatalog.from_tile_directories(
                [("study", tile_dir)]
            )
            metadata = build_deepssf_gpu_datasets(
                study_id="study",
                trajectories_path=trajectories_path,
                plan=plan,
                splits=selected_splits,
                catalog=catalog,
                output_hw=(5, 5),
            )["test_temporal"]
            batch = _DeepSSFGPUCollator(metadata)([0])
            cache = AlphaEarthGPUCache(catalog, "cuda")
            gpu_values, gpu_valid = cache.materialize(
                batch["group_ids"],
                batch["longitude"],
                batch["latitude"],
                batch["bounds"],
            )
            torch.testing.assert_close(
                gpu_values[0].float().cpu(),
                cpu_values,
                atol=0.02,
                rtol=0.02,
            )
            self.assertTrue(
                torch.equal(gpu_valid[0].cpu().expand_as(cpu_valid), cpu_valid)
            )


if __name__ == "__main__":
    unittest.main()
