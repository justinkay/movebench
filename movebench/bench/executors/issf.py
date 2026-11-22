"""Benchmark entry points for the iSSF fit/predict methods."""

from movebench.bench.run import _run_method
from movebench.models.ssf import linear_method, mlp_method


def run_linear(**kwargs):
    return _run_method(linear_method, **kwargs)


def run_mlp(**kwargs):
    return _run_method(mlp_method, **kwargs)
