"""Benchmark entry points for the baseline fit/predict methods."""

from movebench.bench.run import _run_method
from movebench.models.baselines import ConstantVelocity, NoMove, RandomWalk


def run_nomove(**kwargs):
    return _run_method(NoMove, reuse_predictions=False, **kwargs)


def run_constvel(**kwargs):
    return _run_method(ConstantVelocity, reuse_predictions=False, **kwargs)


def run_randomwalk(**kwargs):
    return _run_method(RandomWalk, reuse_predictions=False, **kwargs)
