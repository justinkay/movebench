"""Minimal contribution: predict zero displacement from prepared point inputs."""

import torch

from movebench.datasets.points import point_context


class OriginMethod:
    def __init__(self, *, model_config, hparams, seed):
        pass

    def fit(self, train, validation, *, work_dir, restore_from=None):
        return self

    def predict(self, data):
        context = point_context(data)
        zeros = torch.zeros_like(context["start_bearing_rad"])
        return {
            "prediction_kind": "deterministic",
            "geometry": data.metadata["geometry"],
            "point_id_start": context["point_id_start"],
            "dx_m": zeros,
            "dy_m": zeros,
        }
