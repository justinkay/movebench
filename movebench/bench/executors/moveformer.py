"""Benchmark entry points for the MoveFormer fit/predict methods."""

from movebench.bench.run import _run_method
from movebench.models.moveformer import (
    MoveFormerMethod,
    SharedMoveFormerScorer,
    SharedMoveFormerTrainer,
)


def run(**kwargs):
    return _run_method(MoveFormerMethod, prediction_kind="grid", **kwargs)


def train_full(**kwargs):
    return _run_method(SharedMoveFormerTrainer, **kwargs)


def score_full(**kwargs):
    return _run_method(SharedMoveFormerScorer, prediction_kind="grid", **kwargs)
