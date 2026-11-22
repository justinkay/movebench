from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from movebench.slurm import bench, launcher


class BenchConcurrencyTest(unittest.TestCase):
    @staticmethod
    def _args(*, max_cpu_jobs: int, max_gpu_jobs: int) -> SimpleNamespace:
        return SimpleNamespace(
            experiment_module="experiment.py",
            studies=None,
            methods=None,
            max_runs=None,
            rerun=False,
            dry_run=True,
            gpus_per_node=None,
            partition=None,
            exclude=None,
            qos=None,
            account=None,
            time=None,
            mem=None,
            cpus_per_task=1,
            log_dir=Path("logs"),
            polling_interval=1,
            max_cpu_jobs=max_cpu_jobs,
            max_gpu_jobs=max_gpu_jobs,
        )

    def _requests(
        self,
        *,
        cpu_runs: int,
        gpu_runs: int,
        max_cpu_jobs: int,
        max_gpu_jobs: int,
    ):
        pending_runs = [
            {"method": "baseline_nomove", "run_key": f"cpu-{index}"}
            for index in range(cpu_runs)
        ] + [
            {"method": "deepssf", "run_key": f"gpu-{index}"}
            for index in range(gpu_runs)
        ]
        resolved = {
            "counts": {
                "planned_runs": len(pending_runs),
                "pending_runs": len(pending_runs),
                "skipped_runs": 0,
                "blocked_runs": 0,
            },
            "pending_runs": pending_runs,
            "manifest_path": Path("manifest.json"),
            "manifest_payload": {},
        }
        bench_run = SimpleNamespace(
            resolve_pending_runs=lambda *_args, **_kwargs: resolved
        )
        args = self._args(
            max_cpu_jobs=max_cpu_jobs,
            max_gpu_jobs=max_gpu_jobs,
        )
        with (
            patch.object(bench, "bench_run", bench_run),
            patch.object(bench, "run_sbatch_queue", return_value=0) as queue,
        ):
            self.assertEqual(bench._run_launcher(args), 0)
        self.assertEqual(
            set(queue.call_args.kwargs),
            {"label", "requests", "polling_interval", "dry_run"},
        )
        return queue.call_args.kwargs["requests"]

    def test_cpu_and_gpu_limits_are_independent(self):
        requests = self._requests(
            cpu_runs=4,
            gpu_runs=6,
            max_cpu_jobs=3,
            max_gpu_jobs=5,
        )
        self.assertEqual(
            [
                (
                    request.resources.gpus_per_node,
                    request.array_concurrency,
                    len(request.worker_args_list),
                )
                for request in requests
            ],
            [(0, 3, 4), (1, 5, 6)],
        )

    def test_concurrency_is_capped_by_bucket_size(self):
        requests = self._requests(
            cpu_runs=2,
            gpu_runs=1,
            max_cpu_jobs=8,
            max_gpu_jobs=9,
        )
        self.assertEqual(
            [request.array_concurrency for request in requests],
            [2, 1],
        )

    def test_limits_must_be_positive(self):
        for name in ("max_cpu_jobs", "max_gpu_jobs"):
            with self.subTest(name=name):
                args = self._args(max_cpu_jobs=1, max_gpu_jobs=1)
                setattr(args, name, 0)
                with self.assertRaisesRegex(ValueError, name.replace("_", "-")):
                    bench._run_launcher(args)

    def test_bench_cli_replaces_global_concurrency_flag(self):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)
        bench.register(subparsers)
        args = parser.parse_args(
            [
                "bench",
                "run",
                "experiment.py",
                "--max-cpu-jobs",
                "3",
                "--max-gpu-jobs",
                "5",
            ]
        )
        self.assertEqual((args.max_cpu_jobs, args.max_gpu_jobs), (3, 5))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                ["bench", "run", "experiment.py", "--max-concurrent-jobs", "8"]
            )

    def test_slurm_config_is_independently_overridable(self):
        args = argparse.Namespace(
            slurm_config=Path("configs/slurm/beery.yaml"),
            max_cpu_jobs=None,
            max_gpu_jobs=7,
        )
        launcher.apply_slurm_config(
            args,
            command_name="bench",
            defaults=bench._SLURM_DEFAULTS,
        )
        self.assertEqual(args.max_cpu_jobs, 32)
        self.assertEqual(args.max_gpu_jobs, 7)


class SlurmStatusTest(unittest.TestCase):
    def test_pending_array_range_is_not_counted_as_active(self):
        result = SimpleNamespace(
            returncode=0,
            stdout=(
                "1547216_0|COMPLETED\n"
                "1547216_1|OUT_OF_MEMORY\n"
                "1547216_40|RUNNING\n"
                "1547216_[41-113%1]|PENDING\n"
            ),
        )
        with patch.object(launcher.subprocess, "run", return_value=result) as run:
            self.assertEqual(launcher._slurm_array_status_counts("1547216"), (1, 1, 1))
        self.assertIn("--format=JobID,State", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
