import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from rasterio.transform import from_bounds

import movebench.bench.artifacts as bench_artifacts
from movebench.extract import uses_chip_downloads
from movebench.layers.io import LayerIO


class ExtractionRepairTest(unittest.TestCase):
    def test_tile_only_plan_does_not_use_chip_report(self):
        self.assertFalse(uses_chip_downloads(SimpleNamespace(chip_groups={})))
        self.assertTrue(
            uses_chip_downloads(SimpleNamespace(chip_groups={"era5": {}}))
        )

    def test_zero_width_tile_mosaic_retries_with_two_pixels(self):
        loader = LayerIO.__new__(LayerIO)
        loader.registry = SimpleNamespace(
            get=lambda _layer: SimpleNamespace(bands=["A00"], clip_max=None)
        )
        tile = {
            "bounds": {
                "min_lon": 0.0,
                "min_lat": 0.0,
                "max_lon": 1.0,
                "max_lat": 1.0,
            },
            "file": "tile.tif",
        }
        datasets = [MagicMock(), MagicMock()]
        for dataset in datasets:
            dataset.res = (0.1, 0.1)
            dataset.crs = "EPSG:4326"
        transform = from_bounds(0.0, 0.0, 1.0, 1.0, 2, 2)

        with (
            patch.object(
                loader,
                "_tile_index",
                return_value={"tiles": [tile, {**tile, "file": "tile2.tif"}]},
            ),
            patch.object(loader, "_tile_dir", return_value=Path("/tiles")),
            patch(
                "movebench.layers.io.rasterio.open",
                side_effect=datasets,
            ),
            patch(
                "movebench.layers.io.merge",
                side_effect=[
                    (np.empty((1, 2, 0), dtype=np.float32), transform),
                    (np.ones((1, 2, 2), dtype=np.float32), transform),
                ],
            ) as merge_mock,
        ):
            raster = loader._load_gee_tile_layer(
                "alphaearth_embeddings",
                (0.499, 0.499, 0.501, 0.501),
                timestamp=None,
            )

        self.assertEqual(raster.array.shape, (1, 2, 2))
        self.assertAlmostEqual(raster.bounds[2] - raster.bounds[0], 0.2)
        self.assertAlmostEqual(raster.bounds[3] - raster.bounds[1], 0.2)
        self.assertEqual(merge_mock.call_count, 2)

    def test_readiness_accepts_any_available_heldout_store(self):
        def point_state(_study_id, _timescale, split):
            if split == "test_individual":
                return {"available": False, "reason": "zero valid steps"}
            return {"available": True}

        with (
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
                "_point_store_state",
                side_effect=point_state,
            ),
            patch.object(
                bench_artifacts,
                "_alphaearth_tile_state",
                return_value={"available": True},
            ),
        ):
            readiness = bench_artifacts._study_timescale_readiness(
                "study", "1d", deepssf_data="gpu"
            )

        self.assertTrue(readiness["point_ready"])
        self.assertEqual(readiness["available_point_splits"], ["test_temporal"])
        self.assertEqual(readiness["point_blocked_reasons"], [])

    def test_zero_row_manifest_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            manifest_path.touch()
            with (
                patch.object(
                    bench_artifacts.bench_config,
                    "resolve_timescale",
                    return_value={"rasters_hash": "r", "sampler_hash": "s"},
                ),
                patch.object(
                    bench_artifacts,
                    "point_store_manifest_path",
                    return_value=manifest_path,
                ),
                patch.object(
                    bench_artifacts,
                    "read_manifest",
                    return_value=SimpleNamespace(
                        study_id="study",
                        split="test_temporal",
                        n_rows=0,
                    ),
                ),
            ):
                state = bench_artifacts._point_store_state(
                    "study", "1d", "test_temporal"
                )

        self.assertFalse(state["available"])
        self.assertEqual(state["n_rows"], 0)
        self.assertIn("zero valid steps", state["reason"])


if __name__ == "__main__":
    unittest.main()
