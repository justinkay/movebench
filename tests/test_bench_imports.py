import subprocess
import sys
import unittest
from pathlib import Path


class BenchmarkImportsTest(unittest.TestCase):
    def test_method_modules_do_not_import_execution(self):
        self._check_imports(
            ["movebench.models.baselines", "movebench.models.ssf",
             "movebench.models.deepssf", "movebench.models.moveformer"],
            forbidden=["movebench.bench.run", "movebench.bench.executors.issf"],
        )

    def test_issf_import_does_not_change_multiprocessing_sharing(self):
        code = """
from unittest.mock import patch
import torch
with patch.object(torch.multiprocessing, 'set_sharing_strategy') as sharing:
    import movebench.models.ssf
sharing.assert_not_called()
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def _check_imports(self, modules, forbidden=()):
        code = (
            "import importlib, sys\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module(name)\n"
            f"assert not set({forbidden!r}).intersection(sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_input_records_do_not_import_dataset_implementations(self):
        self._check_imports(
            ["movebench.datasets"],
            ["torch", "movebench.datasets.ssf", "movebench.datasets.deepssf_gpu"],
        )

    def test_read_only_modules_do_not_import_execution(self):
        for name in ("run_state", "artifacts", "planning", "results", "report", "summary"):
            with self.subTest(module=name):
                forbidden = ["movebench.bench.run"]
                if name in {"run_state", "artifacts"}:
                    forbidden.append("movebench.bench.planning")
                self._check_imports([f"movebench.bench.{name}"], forbidden)

    def test_execution_and_reporting_import_in_either_order(self):
        modules = [
            "movebench.bench.results",
            "movebench.bench.run",
            "movebench.bench.report",
            "movebench.bench.summary",
            "movebench.bench.evaluation.runner",
            "movebench.datasets.chips",
            "movebench.datasets.ssf",
            "movebench.datasets.moveformer",
            "movebench.datasets.deepssf",
            "movebench.datasets.deepssf_gpu",
            "movebench.datasets.rasters",
            "movebench.datasets.points",
            "movebench.models.ssf",
            "movebench.models.moveformer",
            "movebench.models.deepssf",
            "movebench.ssf_grid",
            "movebench.training",
        ]
        execution_first = [modules[1], modules[0], *modules[2:]]
        for order in (modules, execution_first, list(reversed(modules))):
            with self.subTest(first=order[0]):
                self._check_imports(order)


if __name__ == "__main__":
    unittest.main()
