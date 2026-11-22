"""Benchmark entry point for the deepSSF fit/predict method."""

from movebench.bench.run import _run_method
from movebench.models.deepssf import DeepSSFMethod


def run(**kwargs):
    return _run_method(DeepSSFMethod, **kwargs)
