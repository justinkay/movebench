from __future__ import annotations

import builtins
import importlib
import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock


def _blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "mlflow" or name.startswith("mlflow."):
        raise ModuleNotFoundError(name="mlflow")
    if name == "ee" or name.startswith("ee."):
        raise ModuleNotFoundError(name="ee")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = builtins.__import__


class BenchPackagingTest(unittest.TestCase):
    def test_bench_extra_does_not_depend_on_mlflow(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        bench_extra = list(pyproject["project"]["optional-dependencies"]["bench"])
        self.assertNotIn("mlflow", bench_extra)

    def test_base_dependencies_do_not_depend_on_earthengine_api(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        dependencies = list(pyproject["project"]["dependencies"])
        self.assertNotIn("earthengine-api", dependencies)

    def test_gee_extra_depends_on_earthengine_api(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
        gee_extra = list(pyproject["project"]["optional-dependencies"]["gee"])
        self.assertIn("earthengine-api", gee_extra)

    def test_bench_modules_import_without_mlflow(self) -> None:
        for module_name in [
            "mlflow",
            "movebench.bench.mlflow",
            "movebench.bench.run",
            "movebench.bench.report",
            "movebench.bench.summary",
            "movebench.bench.evaluation.runner",
            "movebench.cli.main",
        ]:
            sys.modules.pop(module_name, None)

        with mock.patch("builtins.__import__", side_effect=_blocking_import):
            cli_main = importlib.import_module("movebench.cli.main")
            run_module = importlib.import_module("movebench.bench.run")
            report_module = importlib.import_module("movebench.bench.report")
            summary_module = importlib.import_module("movebench.bench.summary")
            runner_module = importlib.import_module("movebench.bench.evaluation.runner")
            parser = cli_main.build_parser()

        self.assertIsNotNone(parser)
        self.assertTrue(hasattr(run_module, "register"))
        self.assertTrue(hasattr(report_module, "write_report"))
        self.assertTrue(hasattr(summary_module, "build_current_summary_payload"))
        self.assertTrue(hasattr(runner_module, "evaluate_run_dir"))


if __name__ == "__main__":
    unittest.main()
