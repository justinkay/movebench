"""Deterministic and fitted movement baselines."""

from __future__ import annotations

import torch

import movebench.ssf_grid as ssf_grid
from movebench.datasets.chips import grid_dataset
from movebench.datasets.points import point_context, training_steps
from movebench.geometry import LOCAL_GEOMETRY


class NoMove:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        return self

    def predict(self, data):
        ctx = point_context(data)
        zeros = torch.zeros_like(ctx["start_bearing_rad"])
        return {
            "prediction_kind": "deterministic",
            "geometry": LOCAL_GEOMETRY,
            "point_id_start": ctx["point_id_start"],
            "dx_m": zeros,
            "dy_m": zeros,
        }


class ConstantVelocity:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        return self

    def predict(self, data):
        ctx = point_context(data)
        return {
            "prediction_kind": "deterministic",
            "geometry": LOCAL_GEOMETRY,
            "point_id_start": ctx["point_id_start"],
            "dx_m": ctx["prev_step_length_m"] * torch.sin(ctx["start_bearing_rad"]),
            "dy_m": ctx["prev_step_length_m"] * torch.cos(ctx["start_bearing_rad"]),
        }


class RandomWalk:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        _support, steps = training_steps(train)
        self.plan_ = train.inputs["plan"]
        self.sampler_state_ = ssf_grid._replay_sampler_state(
            train_manifest=train.inputs["point_stores"]["train"]["manifest"],
            train_steps=steps,
        )
        return self

    def predict(self, data):
        return ssf_grid.build_ssf_grid_artifact(
            grid_dataset=grid_dataset(data, plan=self.plan_, include_env=False),
            sampler_state=self.sampler_state_,
            batch_size=int(self.hparams.get("batch_size", 256)),
            num_workers=int(self.hparams.get("num_workers", 0) or 0),
            device=torch.device("cpu"),
            model=None,
            feature_adapter=None,
        )
