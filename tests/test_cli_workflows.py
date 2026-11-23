import io
import json
import os
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import movebench.extract as extraction
from movebench.cli import bench, download, extract, pack
from movebench.cli.main import main
from movebench.cli.slurm import extract as slurm_extract
from movebench.cli.slurm import pack as slurm_pack
from movebench.cli.slurm import worker


class CliWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.context = ExitStack()
        self.addCleanup(self.context.close)
        self.context.enter_context(redirect_stdout(io.StringIO()))
        self.context.enter_context(redirect_stderr(io.StringIO()))
        self.context.enter_context(patch.dict(os.environ))

    def test_download_parent_options_survive_gee_dispatch(self):
        study = Mock()
        study.download_rasters.return_value = {"chips": {"failed": 0}}
        with (
            patch.object(download, "Study", return_value=study) as study_type,
            patch.object(download, "resolve_study_selection", return_value=["study"]),
            patch.object(download, "load_layers_config"),
            patch.object(download.REGISTRY, "resolve", return_value=SimpleNamespace(rasters_hash="r")),
        ):
            study_type.list_available.return_value = ["study"]
            flags = ["--data-root", "/selected", "--paths-config", "/paths.yaml",
                     "--layers-config", "layers.yaml", "--n-workers", "3"]
            for argv in (["download", *flags, "gee"], ["download", "gee", *flags]):
                with self.subTest(argv=argv):
                    self.assertEqual(main(argv), 0)
                    self.assertEqual(os.environ["MOVEBENCH_DATA_ROOT"], "/selected")
                    self.assertEqual(os.environ["MOVEBENCH_PATHS_CONFIG"], "/paths.yaml")
                    self.assertEqual(study.download_rasters.call_args.kwargs["layers_config"], "layers.yaml")
                    self.assertEqual(study.download_rasters.call_args.kwargs["n_workers"], 3)
            self.assertEqual(main(["download", "--n-workers", "2", "gee", "--n-workers", "4"]), 0)
            self.assertEqual(study.download_rasters.call_args.kwargs["n_workers"], 4)

    def test_download_defaults_and_failure_exit(self):
        with patch.object(download, "download_public_studies", return_value=0) as public:
            self.assertEqual(main(["download"]), 0)
            self.assertEqual(public.call_args.kwargs["range_workers"], 1)
        study = Mock()
        with (
            patch.object(download, "Study", return_value=study) as study_type,
            patch.object(download, "resolve_study_selection", return_value=["study"]),
            patch.object(download, "load_layers_config"),
            patch.object(download.REGISTRY, "resolve", return_value=SimpleNamespace(rasters_hash="r")),
        ):
            study_type.list_available.return_value = ["study"]
            for result, code in (
                ({"chips": {"failed": 0}}, 0),
                ({"chips": {"failed": 1}}, 2),
                ({"tiles": {"layer": {"years": {"2020": {"failed": 1}}}}}, 2),
            ):
                with self.subTest(result=result):
                    study.download_rasters.return_value = result
                    self.assertEqual(main(["download", "gee"]), code)
                    self.assertEqual(study.download_rasters.call_args.kwargs["n_workers"], 32)
                    self.assertEqual(study.download_rasters.call_args.kwargs["layers_config"], "configs/layers/all.yaml")
            study.download_rasters.side_effect = RuntimeError("download failed")
            self.assertEqual(main(["download", "gee"]), 2)

    def test_pack_options_survive_submission_and_worker_execution(self):
        job = SimpleNamespace(study_id="study", timescale_key="12h", split_keys=("train", "test_temporal"))
        flags = [
            "--studies", "study", "--timescales", "12h", "--splits", "train", "test_temporal",
            "--layers-config", "custom.yaml", "--samples-per-shard", "128",
            "--spatial-dtype", "float32", "--target-dtype", "float16", "--compression", "gz",
            "--build-workers", "2", "--overwrite", "--no-resume", "--seed", "7",
        ]
        with (
            patch.object(pack.pack_runtime, "resolve_pack_jobs", return_value=[job]) as resolve,
            patch.object(pack.pack_runtime, "execute_pack_job") as execute,
            patch.object(slurm_pack, "run_sbatch_queue", return_value=0) as queue,
        ):
            self.assertEqual(main(["slurm", "pack", *flags]), 0)
            launch_options = resolve.call_args.kwargs.copy()
            self.assertEqual(launch_options["layers_config"], "custom.yaml")
            request = queue.call_args.kwargs["requests"][0]
            worker_args = list(request.worker_args_list[0])
            self.assertEqual(worker_args[0], "pack")
            self.assertEqual(main(worker_args), 0)
            self.assertEqual(resolve.call_args.kwargs, launch_options)
            execute.assert_called_once_with(job)

    def test_extraction_worker_rechecks_inputs_after_submission(self):
        run = SimpleNamespace(layers_config_path="layers.yaml", sampler_config_path="sampler.yaml")
        study = SimpleNamespace(study_dir=Path("study"), study_id="study")
        flags = ["--studies", "study", "--splits", "train", "--layers-config", "layers.yaml",
                 "--sampler-config", "sampler.yaml", "--n-workers", "3"]
        with (
            patch.object(extract, "build_extraction_runs", return_value=[run]),
            patch.object(extract, "resolve_study_selection", return_value=["study"]),
            patch.object(slurm_extract, "build_extraction_runs", return_value=[run]),
            patch.object(slurm_extract, "resolve_study_selection", return_value=["study"]),
            patch.object(slurm_extract, "Study", return_value=study),
            patch.object(slurm_extract, "check_extraction_inputs", return_value=(False, "s", None)),
            patch.object(slurm_extract, "run_sbatch_queue", return_value=0) as queue,
            patch.object(extraction, "Study", return_value=study),
            patch.object(extraction, "check_extraction_inputs", side_effect=ValueError("rasters removed")),
        ):
            self.assertEqual(main(["slurm", "extract", *flags]), 0)
            worker_args = list(queue.call_args.kwargs["requests"][0].worker_args_list[0])
            self.assertEqual(worker_args[0], "extract")
            self.assertEqual(main(worker_args), 2)

    def test_extraction_check_completion_missing_inputs_and_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = SimpleNamespace(study_id="study", study_dir=Path(tmp))
            run = SimpleNamespace(plan=object(), sampler=object(), layers_config_path="layers", sampler_config_path="sampler")
            kwargs = dict(study=study, run=run, splits=["train"], skip_existing=True,
                          allow_known_download_failures=False)
            with (
                patch.object(extraction, "resolve_sampler_hash", return_value="s"),
                patch.object(extraction, "is_extract_complete", return_value=(True, "s")) as complete,
                patch.object(extraction, "uses_gee_layers", return_value=True),
                patch.object(extraction, "uses_chip_downloads", return_value=True) as chips,
                patch.object(extraction, "rasters_root_for", return_value=Path(tmp) / "missing") as rasters,
                patch.object(extraction, "load_chip_request_status_for_extract", return_value={"status": "failed"}) as report,
            ):
                self.assertEqual(extraction.check_extraction_inputs(**kwargs), (True, "s", None))
                rasters.assert_not_called()
                complete.return_value = (False, "s")
                with self.assertRaisesRegex(ValueError, "missing raster cache root"):
                    extraction.check_extraction_inputs(**kwargs)
                rasters.return_value = Path(tmp)
                with self.assertRaisesRegex(ValueError, "known download failures"):
                    extraction.check_extraction_inputs(**kwargs)
                kwargs["allow_known_download_failures"] = True
                self.assertEqual(extraction.check_extraction_inputs(**kwargs), (False, "s", {"status": "failed"}))
                kwargs["allow_known_download_failures"] = False
                chips.return_value = False
                report.reset_mock()
                self.assertEqual(extraction.check_extraction_inputs(**kwargs), (False, "s", None))
                report.assert_not_called()

    def test_ingest_directory_keeps_splits_and_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "csvs"
            source.mkdir()
            for name in ("a", "b"):
                (source / f"{name}.csv").write_text(
                    "individual-local-identifier,timestamp,location-lat,location-long\n"
                    "animal,2020-01-01 00:00:00,1,2\n"
                    "animal,2020-01-02 00:00:00,1,2\n"
                )
            argv = ["ingest", str(source), "--data-root", str(root / "data")]
            for flags in ([], ["--force"]):
                self.assertEqual(main([*argv, *flags]), 0)
                studies = root / "data" / "studies"
                self.assertEqual(sorted(p.name for p in studies.iterdir()), ["a", "b"])
                for name in ("a", "b"):
                    manifest = json.loads((studies / name / "manifest.json").read_text())
                    self.assertEqual(manifest["study_id"], name)
                    self.assertEqual(manifest["trajectories"]["split_counts"], {"train": 1, "test_temporal": 1})

    def test_bench_failure_exit(self):
        for failures, code in ((0, 0), (1, 2)):
            summary = dict(planned_runs=1, executed_runs=1, skipped_runs=0, blocked_runs=0, failed_runs=failures)
            with patch.object(bench.bench_run, "run_experiment", return_value=summary):
                self.assertEqual(main(["bench", "run", "experiment.py"]), code)

    def test_array_worker_dispatches_argument_list_without_program_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "array.json"
            argv = ["pack", "--studies", "study", "--timescales", "12h"]
            manifest.write_text(json.dumps({"tasks": [{"worker_args": argv}]}))
            with patch.dict(os.environ, SLURM_ARRAY_TASK_ID="0"), patch.object(worker, "cli_main", return_value=2) as dispatch:
                self.assertEqual(worker.main(["--array-manifest", str(manifest)]), 2)
                dispatch.assert_called_once_with(argv)


if __name__ == "__main__":
    unittest.main()
