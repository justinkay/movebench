import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from movebench.bench.evaluation.runner import _grid_coordinates, _truth_xy_m_for_step
from movebench.datasets.deepssf_wds_cache import read_deepssf_wds_manifest
from movebench.datasets.moveformer import _set_local_movement_vectors
from movebench.datasets.ssf import SSFDataset
from movebench.geometry import (
    LOCAL_GEOMETRY,
    local_xy_m_from_lonlat,
    lonlat_from_local_xy_m,
)


class PipelineGeometryTest(unittest.TestCase):
    def test_evaluation_truth_and_grid_share_local_meter_coordinates(self):
        end_lon, end_lat = lonlat_from_local_xy_m(10.0, 75.0, 2_000.0, -3_000.0)
        truth = _truth_xy_m_for_step(10.0, 75.0, float(end_lon), float(end_lat))
        torch.testing.assert_close(
            truth,
            torch.tensor([2_000.0, -3_000.0], dtype=torch.float64),
            atol=1e-6,
            rtol=0.0,
        )
        x_m, y_m = _grid_coordinates(width=5, height=5, pixel_size_m=1_000.0)
        torch.testing.assert_close(
            x_m,
            torch.tensor([-2_000.0, -1_000.0, 0.0, 1_000.0, 2_000.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            y_m,
            torch.tensor([2_000.0, 1_000.0, 0.0, -1_000.0, -2_000.0], dtype=torch.float64),
        )

    def test_moveformer_metric_vectors_use_stored_offsets(self):
        next_lon, next_lat = lonlat_from_local_xy_m(0.0, 70.0, 1_000.0, 500.0)
        start_lon = np.array([0.0, float(next_lon)])
        start_lat = np.array([70.0, float(next_lat)])
        candidate_x_m = torch.tensor([[100.0, 200.0], [300.0, 400.0]])
        candidate_y_m = torch.tensor([[500.0, 600.0], [700.0, 800.0]])
        encoded = {}
        _set_local_movement_vectors(
            encoded,
            start_lon=start_lon,
            start_lat=start_lat,
            candidate_x_m=candidate_x_m,
            candidate_y_m=candidate_y_m,
            mov_vec_scale=0.01,
        )
        prev_x_m, prev_y_m = local_xy_m_from_lonlat(
            start_lon[1],
            start_lat[1],
            start_lon[0],
            start_lat[0],
        )
        np.testing.assert_allclose(
            encoded["location_mov"][1],
            -np.array([prev_x_m, prev_y_m]) * 0.01,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            encoded["location_cand_mov"],
            np.stack([candidate_x_m.numpy(), candidate_y_m.numpy()], axis=-1) * 0.01,
        )

    def test_wds_manifest_requires_geometry(self):
        manifest = {
            "schema_version": "5",
            "payload_format": "pth",
            "geometry": LOCAL_GEOMETRY,
            "num_samples": 1,
            "pixel_size_m": 100.0,
            "env_channels": 1,
            "band_ids": ["dem.elevation"],
            "sample_identity": "point_id_start",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(read_deepssf_wds_manifest(path)["geometry"], LOCAL_GEOMETRY)
            manifest.pop("geometry")
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "geometry"):
                read_deepssf_wds_manifest(path)

    def test_ssf_metadata_does_not_construct_crs_objects(self):
        dataset = SSFDataset.__new__(SSFDataset)
        dataset.start_lon = torch.tensor([10.0])
        dataset.start_lat = torch.tensor([75.0])
        dataset._get_env_valid = lambda _idx: (
            torch.zeros((1, 1)),
            torch.ones((1, 1), dtype=torch.bool),
        )
        with patch(
            "movebench.datasets.ssf.local_aeqd_proj4",
            return_value="+proj=aeqd",
        ) as proj4:
            _env, _valid, _metadata, crs = dataset[0]
        self.assertEqual(crs, "+proj=aeqd")
        proj4.assert_called_once_with(10.0, 75.0)

if __name__ == "__main__":
    unittest.main()
