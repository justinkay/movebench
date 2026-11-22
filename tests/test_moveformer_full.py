import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

import movebench.bench.config as bench_config
import movebench.bench.executors.moveformer as moveformer_executor
import movebench.bench.moveformer_taxa as moveformer_taxa
import movebench.bench.artifacts as bench_artifacts
import movebench.bench.planning as bench_planning
import movebench.datasets.moveformer as moveformer_data
from movebench.datasets import DataSplit, SharedTrainingData, StudyTrainingData
from movebench.datasets.moveformer import MoveFormerPointStoreDataset, _StepRecord
from movebench.lib.moveformer.geo_transformers.trajectory import (
    SimpleTrajectoryProcessor,
)
from movebench.study import MINI_STUDIES


def _readiness():
    return {
        "available_point_splits": ["test_temporal"],
        "early_stop_point_split": "test_temporal",
        "train_point_ready": True,
        "point_ready": True,
        "point_blocked_reasons": [],
    }


def _blocked_readiness():
    return {
        "available_point_splits": [],
        "early_stop_point_split": None,
        "train_point_ready": False,
        "point_ready": False,
        "point_blocked_reasons": ["missing point store manifest for train"],
    }


def _train_only_readiness():
    return {
        "available_point_splits": [],
        "early_stop_point_split": None,
        "train_point_ready": True,
        "point_ready": False,
        "point_blocked_reasons": ["no held-out point stores available"],
    }


def _zero_step_readiness():
    return {
        "available_point_splits": [],
        "early_stop_point_split": None,
        "train_steps_empty": True,
        "train_point_ready": False,
        "point_ready": False,
        "point_blocked_reasons": ["train split has zero usable 1h steps"],
    }


def _timescale():
    return {
        "step_time_s": 3600,
        "alphaearth_layers_config": "alphaearth.yaml",
        "layers_config": "all.yaml",
        "rasters_hash": "rasters",
        "sampler_config": "sampler.yaml",
        "sampler_hash": "sampler",
        "geometry": "local-aeqd-v1",
    }


class MoveFormerFullPlannerTest(unittest.TestCase):
    def _expand(
        self,
        *,
        method,
        studies,
        shared_training_studies=None,
        training_studies=None,
        bands_by_study=None,
        resolve_artifacts=None,
        readiness_by_study=None,
    ):
        experiment = {
            "name": "full-test",
            "studies": list(studies),
            "timescales": ["1h"],
            "covariate_sets": ["alphaearth_only"],
            "methods": [method],
        }
        if shared_training_studies is not None:
            experiment["shared_training_studies"] = shared_training_studies

        def resolve_studies(_experiment, override=None):
            return list(studies if override is None else override)

        def band_ids(_timescale, _covariates, *, study_id):
            if bands_by_study is None:
                return ["alphaearth.A00"]
            return list(bands_by_study[study_id])

        def readiness(study_id, *_args, **_kwargs):
            if readiness_by_study is None:
                return _readiness()
            return readiness_by_study[study_id]

        if resolve_artifacts is None:
            resolve_artifacts = lambda _run: {"status": "resolved"}
        with (
            patch.object(bench_planning, "_resolve_studies", side_effect=resolve_studies),
            patch.object(
                bench_artifacts,
                "_study_timescale_readiness",
                side_effect=readiness,
            ),
            patch.object(
                bench_planning.bench_config,
                "resolve_covariate_band_ids",
                side_effect=band_ids,
            ),
            patch.object(
                bench_planning.bench_config,
                "resolve_timescale",
                return_value=_timescale(),
            ),
            patch.object(
                bench_artifacts,
                "_resolve_artifacts_for_run",
                side_effect=resolve_artifacts,
            ),
            patch.object(
                bench_planning,
                "resolve_study_selection",
                return_value=list(training_studies or studies),
            ),
        ):
            runs = bench_planning.expand_experiment(experiment)
        return experiment, runs

    def test_cli_all_and_mini_resolve_exact_training_cohorts(self):
        mini_study = MINI_STUDIES[0]
        available = [mini_study, "other-study"]
        with patch("movebench.study._list_available_studies", return_value=available):
            self.assertEqual(
                bench_planning._resolve_studies({"studies": "all"}, ["all"]),
                available,
            )
            self.assertEqual(
                bench_planning._resolve_studies({"studies": "all"}, ["mini"]),
                [mini_study],
            )

        _all_experiment, all_runs = self._expand(
            method="moveformer-full-trainonly",
            studies=available,
        )
        _mini_experiment, mini_runs = self._expand(
            method="moveformer-full-trainonly",
            studies=[mini_study],
        )
        self.assertEqual(all_runs[0]["shared_study_ids"], sorted(available))
        self.assertEqual(mini_runs[0]["shared_study_ids"], [mini_study])
        self.assertNotEqual(all_runs[0]["run_key"], mini_runs[0]["run_key"])

    def test_shared_training_groups_only_compatible_studies(self):
        _experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["land-a", "land-b", "marine"],
            bands_by_study={
                "land-a": ["alphaearth.A00"],
                "land-b": ["alphaearth.A00"],
                "marine": ["alphaearth.A00", "marine.sst"],
            },
        )
        self.assertEqual(
            {tuple(run["shared_study_ids"]) for run in runs},
            {("land-a", "land-b"), ("marine",)},
        )

    def test_resolver_accepts_planned_cohort_without_zero_step_studies(self):
        _experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["a", "b"],
        )
        self.assertEqual(len(runs), 1)
        payload = runs[0]
        self.assertNotIn("shared_zero_step_studies", payload)
        before = json.dumps(payload, sort_keys=True)

        def resolve_points(study_run, **_kwargs):
            return bench_artifacts._resolved_splits(study_run, {}, (101, 101))

        with (
            patch.object(bench_artifacts.bench_config, "resolve_timescale"),
            patch.object(
                bench_artifacts, "_resolve_point_artifacts", side_effect=resolve_points
            ),
        ):
            resolved = bench_artifacts._resolve_artifacts_for_run(payload)

        self.assertEqual(resolved["status"], "resolved")
        data = resolved["train"]
        self.assertEqual(data.zero_step_studies, ())
        self.assertEqual(data.requested_study_ids, ("a", "b"))
        self.assertEqual(
            [study.train.metadata["study_id"] for study in data.studies], ["a", "b"]
        )
        self.assertEqual(data.training_signature, payload["shared_training_signature"])
        self.assertEqual(json.dumps(payload, sort_keys=True), before)

    def test_shared_training_does_not_silently_drop_blocked_study(self):
        def resolve(run):
            if run["study_id"] == "blocked":
                return {"status": "blocked", "reason": "missing point store"}
            return {"status": "resolved"}

        experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["ready", "blocked"],
            resolve_artifacts=resolve,
        )
        self.assertEqual(runs, [])
        self.assertEqual(experiment["blocked"][0]["study_id"], "blocked")

    def test_shared_training_does_not_drop_readiness_blocked_study(self):
        experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["ready", "blocked"],
            readiness_by_study={
                "ready": _readiness(),
                "blocked": _blocked_readiness(),
            },
        )
        self.assertEqual(runs, [])
        self.assertEqual(experiment["blocked"][0]["study_id"], "blocked")

    def test_shared_training_records_zero_step_study_without_blocking_group(self):
        experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["ready", "empty"],
            readiness_by_study={
                "ready": _readiness(),
                "empty": _zero_step_readiness(),
            },
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["shared_study_ids"], ["empty", "ready"])
        self.assertEqual(runs[0]["shared_data_study_ids"], ["ready"])
        self.assertEqual(
            runs[0]["shared_zero_step_studies"],
            [
                {
                    "study_id": "empty",
                    "reason": "train split has zero usable 1h steps",
                }
            ],
        )
        self.assertEqual(experiment["blocked"][0]["study_id"], "empty")

    def test_shared_training_keeps_train_ready_study_without_heldout_store(self):
        experiment, runs = self._expand(
            method="moveformer-full-trainonly",
            studies=["with-eval", "train-only"],
            readiness_by_study={
                "with-eval": _readiness(),
                "train-only": _train_only_readiness(),
            },
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(
            runs[0]["shared_study_ids"],
            ["train-only", "with-eval"],
        )
        self.assertEqual(experiment["blocked"], [])

    def test_scoring_cohort_defaults_to_selection_and_can_be_explicit(self):
        _experiment, mini_runs = self._expand(
            method="moveformer-full",
            studies=["mini"],
        )
        self.assertEqual(mini_runs[0]["shared_training_study_ids"], ["mini"])

        _experiment, all_runs = self._expand(
            method="moveformer-full",
            studies=["mini"],
            shared_training_studies="all",
            training_studies=["mini", "other"],
        )
        self.assertEqual(
            all_runs[0]["shared_training_study_ids"],
            ["mini", "other"],
        )

    def test_scoring_requires_completed_matching_checkpoint(self):
        payload = {
            "shared_checkpoint_run_key": "train-key",
            "shared_training_signature": "signature",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.object(bench_planning.paths, "repo_root", return_value=root):
                blocked = bench_artifacts._resolve_shared_checkpoint(payload)
                self.assertEqual(blocked["status"], "blocked")

                run_dir = root / "results" / "bench" / "runs" / "train-key"
                run_dir.mkdir(parents=True)
                (run_dir / "resolved_run.json").write_text(
                    json.dumps({"shared_training_signature": "signature"})
                )
                (run_dir / "training_summary.json").write_text(
                    json.dumps({"status": "completed"})
                )
                (run_dir / "checkpoint_best.pt").write_bytes(b"checkpoint")
                resolved = bench_artifacts._resolve_shared_checkpoint(payload)
        self.assertEqual(resolved["status"], "resolved")


class MoveFormerLazyGeometryTest(unittest.TestCase):
    @staticmethod
    def _processor():
        return SimpleTrajectoryProcessor(
            use_target=True,
            use_candidates=True,
            add_dtime=True,
            add_mov_vec=False,
            mov_vec_scale=0.01,
            handle_nan=False,
            pass_through=["taxon", "cov", "cov_cand"],
        )

    @staticmethod
    def _payloads():
        return [
            (
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                torch.tensor([[True, True, False], [True, False, True]]),
                torch.tensor([10.01, 10.02, 10.03]),
                torch.tensor([75.01, 75.02, 75.03]),
                torch.tensor([1234.0, 11.0, 22.0]),
                torch.tensor([-432.0, 33.0, 44.0]),
            ),
            (
                torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
                torch.tensor([[True, True, True], [False, True, True]]),
                torch.tensor([10.11, 10.12, 10.13]),
                torch.tensor([75.11, 75.12, 75.13]),
                torch.tensor([-250.0, 55.0, 66.0]),
                torch.tensor([875.0, 77.0, 88.0]),
            ),
        ]

    def _dataset(self, *, lazy):
        payloads = self._payloads()
        records = []
        for idx, payload in enumerate(payloads):
            records.append(
                _StepRecord(
                    split_key="train",
                    row_idx=idx,
                    point_id_start=100 + idx,
                    point_id_end=101 + idx,
                    timestamp_ns=1_700_000_000_000_000_000
                    + idx * 3_600_000_000_000,
                    timestamp_end_ns=1_700_003_600_000_000_000
                    + idx * 3_600_000_000_000,
                    start_lon=10.0 + idx * 0.1,
                    start_lat=75.0 + idx * 0.1,
                    env=None if lazy else payload[0],
                    valid=None if lazy else payload[1],
                    candidate_lon=None if lazy else payload[2],
                    candidate_lat=None if lazy else payload[3],
                    candidate_x_m=None if lazy else payload[4],
                    candidate_y_m=None if lazy else payload[5],
                )
            )

        dataset = MoveFormerPointStoreDataset.__new__(MoveFormerPointStoreDataset)
        dataset._chains = [records]
        dataset._segments = [(0, 0, len(records))]
        dataset.split = "train"
        dataset.lazy_rows = lazy
        dataset.candidate_subset_size = None
        dataset.cov_dim = 2
        dataset._taxon_embedding = np.arange(4, dtype=np.float32)
        dataset._mov_vec_scale = 0.01
        dataset._processor = self._processor()

        def load_rows(_split_key, row_idxs):
            return tuple(
                torch.stack([payloads[row_idx][field_idx] for row_idx in row_idxs])
                for field_idx in range(6)
            )

        dataset._load_rows = load_rows
        return dataset

    def test_lazy_and_eager_use_identical_stored_local_geometry(self):
        eager = self._dataset(lazy=False)[0]
        lazy = self._dataset(lazy=True)[0]
        self.assertEqual(set(eager), set(lazy))
        for key in eager:
            np.testing.assert_allclose(eager[key], lazy[key], equal_nan=True)

        expected_target = np.array(
            [[1234.0, -432.0], [-250.0, 875.0]],
            dtype=np.float32,
        )
        expected_candidates = np.array(
            [
                [[1234.0, -432.0], [11.0, 33.0], [22.0, 44.0]],
                [[-250.0, 875.0], [55.0, 77.0], [66.0, 88.0]],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(eager["location_tgt_mov"], expected_target * 0.01)
        np.testing.assert_allclose(
            eager["location_cand_mov"],
            expected_candidates * 0.01,
        )
        np.testing.assert_allclose(eager["coords_m"], expected_candidates[:, 1:])

    def test_lazy_reader_loads_and_reorders_stored_metric_offsets(self):
        class Reader:
            closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True

            def load_env_valid_rows(self, **_kwargs):
                return (
                    torch.tensor([[[1.0, 2.0, 3.0]]]),
                    torch.tensor([[[True, False, True]]]),
                )

            def load_shared_rows(self, **_kwargs):
                return {
                    "candidate_lon": torch.tensor([[10.0, 20.0, 30.0]]),
                    "candidate_lat": torch.tensor([[40.0, 50.0, 60.0]]),
                    "candidate_x_m": torch.tensor([[100.0, 200.0, 300.0]]),
                    "candidate_y_m": torch.tensor([[400.0, 500.0, 600.0]]),
                }

        reader = Reader()
        dataset = MoveFormerPointStoreDataset.__new__(MoveFormerPointStoreDataset)
        dataset._reader_spec_by_split = {"train": (Path("store"), object())}
        dataset._band_ids = ["band"]
        dataset._cand_perm = torch.tensor([2, 0, 1])
        dataset.normalize_env = False
        with patch(
            "movebench.datasets.moveformer.PointStoreReader", return_value=reader
        ):
            env, _valid, _lon, _lat, x_m, y_m = dataset._load_rows("train", [0])

        self.assertTrue(reader.closed)
        torch.testing.assert_close(env, torch.tensor([[[3.0, 1.0, 2.0]]]))
        torch.testing.assert_close(x_m, torch.tensor([[300.0, 100.0, 200.0]]))
        torch.testing.assert_close(y_m, torch.tensor([[600.0, 400.0, 500.0]]))


class _FakeDataset(Dataset):
    calls = []

    def __init__(self, *, split, **kwargs):
        self.split = str(split)
        self.kwargs = kwargs
        self.cov_dim = 2
        self.num_candidates_total = 16
        self._mov_vec_scale = 0.5
        self.plan = object()
        self.calls.append(self)

    def __len__(self):
        return 3

    def __getitem__(self, index):
        raise AssertionError(index)


class _FakeLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return 3


class _FakeRuntime:
    def __init__(self, *, model, **kwargs):
        self.model = model
        self.kwargs = kwargs


class MoveFormerFullExecutorTest(unittest.TestCase):
    def setUp(self):
        _FakeDataset.calls = []

    @staticmethod
    def _hparams():
        return {
            "lr": 5e-5,
            "weight_decay": 0.0,
            "warmup_steps": 50,
            "lr_gamma": 1.0,
            "batch_size": 32,
            "max_len": 64,
            "min_len": 1,
            "max_epochs": 2,
            "evals_per_epoch": 2,
            "patience_evals": 20,
            "max_eval_items_per_study": 100,
            "candidate_subset_size": 16,
        }

    @staticmethod
    def _study_data(study_id, validation=("test_temporal",)):
        inputs = {
            "kind": "points",
            "study_dir": Path(study_id),
            "trajectories_path": Path(study_id) / "trajectories.csv",
            "plan": object(),
            "sampler": object(),
            "all_band_ids": ["alphaearth.A00"],
            "band_indices": [0],
            "normalization_stats_path": None,
            "point_stores": {},
        }
        metadata = {
            "study_id": study_id,
            "timescale": "1h",
            "band_ids": ("alphaearth.A00",),
            "geometry": "local_aeqd",
            "output_hw": (101, 101),
        }
        return StudyTrainingData(
            DataSplit("train", metadata, inputs),
            tuple(DataSplit(key, metadata, inputs) for key in validation),
        )

    def test_train_full_is_lazy_and_writes_only_shared_checkpoint(self):
        hparams = self._hparams()
        run_payload = {
            "method": "moveformer-full-trainonly",
            "method_hparams": hparams,
            "method_model": {},
            "seed": 0,
            "covariate_set": "alphaearth_only",
            "shared_training_signature": "signature",
            "shared_study_ids": ["a", "b"],
        }
        studies = tuple(self._study_data(study_id) for study_id in ("a", "b"))
        training_data = SharedTrainingData(studies, ("a", "b"), (), "signature")

        def taxon(study_id):
            return np.ones(4, dtype=np.float32), {"species": study_id}

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with (
                patch.object(
                    moveformer_data,
                    "MoveFormerPointStoreDataset",
                    _FakeDataset,
                ),
                patch.object(
                    moveformer_executor,
                    "moveformer_taxon_for_study",
                    side_effect=taxon,
                ),
                patch.object(
                    moveformer_executor,
                    "sequence_loader",
                    side_effect=lambda dataset, **_kwargs: _FakeLoader(dataset),
                ),
                patch.object(
                    moveformer_executor,
                    "build_moveformer_selection_model",
                    return_value=torch.nn.Linear(1, 1),
                ),
                patch.object(moveformer_executor, "_MoveFormerRuntime", _FakeRuntime),
                patch.object(
                    moveformer_executor,
                    "run_training_loop",
                    return_value={"completed_steps": 4, "best_step": 3},
                ),
                patch.object(
                    moveformer_executor,
                    "_eval_metrics",
                    return_value={"nll": 1.0, "nce": 1.0, "num_tokens": 1.0},
                ),
                patch.object(moveformer_executor, "_write_grid_predictions") as predict,
            ):
                summary = moveformer_executor.train_full(
                    run_payload=run_payload,
                    resolved_artifacts={"train": training_data},
                    run_dir=run_dir,
                    tracker=object(),
                )
            checkpoint = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu")

        self.assertEqual(summary["schedule"]["max_steps"], 6)
        self.assertEqual(summary["schedule"]["eval_every_steps"], 2)
        self.assertEqual(checkpoint["meta"]["shared_study_ids"], ["a", "b"])
        self.assertTrue(
            all(dataset.kwargs["lazy_rows"] for dataset in _FakeDataset.calls)
        )
        predict.assert_not_called()

    def test_shared_input_records_keep_validation_and_requested_membership(self):
        requested = ["a", "b", "zero"]
        zero_steps = [{"study_id": "zero", "reason": "zero training steps"}]
        payload = {
            "timescale": "1h",
            "band_ids": ["alphaearth.A00"],
            "shared_study_ids": requested,
            "shared_zero_step_studies": zero_steps,
            "shared_training_signature": "signature",
            "shared_studies": [
                {
                    "study_id": "a",
                    "early_stop_split_key": "test_individual",
                    "evaluation_split_keys": ["test_temporal", "test_individual"],
                },
                {
                    "study_id": "b",
                    "early_stop_split_key": None,
                    "evaluation_split_keys": [],
                },
            ],
        }

        def resolve(study_run, **_kwargs):
            return bench_artifacts._resolved_splits(
                study_run,
                self._study_data(study_run["study_id"]).train.inputs,
                (101, 101),
            )

        with (
            patch.object(bench_artifacts.bench_config, "resolve_timescale"),
            patch.object(bench_artifacts, "_resolve_point_artifacts", side_effect=resolve),
        ):
            data = bench_artifacts._resolve_shared_point_artifacts(payload, method={})[
                "train"
            ]

        self.assertEqual(data.requested_study_ids, tuple(requested))
        self.assertEqual(data.zero_step_studies, tuple(zero_steps))
        self.assertEqual(data.training_signature, "signature")
        self.assertEqual(
            [study.train.metadata["study_id"] for study in data.studies], ["a", "b"]
        )
        self.assertEqual(
            [split.split_key for split in data.studies[0].validation],
            ["test_individual", "test_temporal"],
        )
        self.assertEqual(data.studies[1].validation, ())
        for split in data.studies[0].validation:
            self.assertIs(split.inputs, data.studies[0].train.inputs)
            self.assertIs(split.metadata, data.studies[0].train.metadata)

    def test_full_validation_falls_back_from_empty_preferred_split(self):
        def dataset(*, split, **kwargs):
            if split == "test_temporal":
                raise ValueError(
                    "No segments found for split='test_temporal' (min_len=1, max_len=64)"
                )
            return _FakeDataset(split=split, **kwargs)

        with patch.object(
            moveformer_data,
            "MoveFormerPointStoreDataset",
            side_effect=dataset,
        ):
            result = moveformer_executor._full_eval_dataset(
                study=self._study_data("study", ("test_temporal", "test_individual")),
                max_len=64,
                min_len=1,
                taxon_embedding=np.ones(4, dtype=np.float32),
            )

        self.assertEqual(result.split, "test_individual")

    def test_score_full_only_consumes_completed_checkpoint(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "meta": {
                        "best_step": 7,
                        "shared_training_signature": "signature",
                    },
                },
                checkpoint_path,
            )
            run_payload = {
                "method_hparams": self._hparams(),
                "method_model": {},
                "study_id": "study",
                "band_ids": ["alphaearth.A00"],
                "seed": 0,
                "early_stop_split_key": "test_temporal",
                "shared_training_signature": "signature",
                "shared_checkpoint_run_key": "train-key",
            }
            study = self._study_data("study")
            artifacts = {
                "splits": {"train": study.train},
                "shared_training_summary": {"train_steps": 9},
            }
            with (
                patch.object(
                    moveformer_data,
                    "MoveFormerPointStoreDataset",
                    _FakeDataset,
                ),
                patch.object(
                    moveformer_executor,
                    "moveformer_taxon_for_study",
                    return_value=(
                        np.ones(4, dtype=np.float32),
                        {"species": "species"},
                    ),
                ),
                patch.object(
                    moveformer_executor,
                    "build_moveformer_selection_model",
                    return_value=model,
                ),
                patch.object(moveformer_executor, "_write_grid_predictions") as predict,
                patch.object(moveformer_executor, "run_training_loop") as train,
            ):
                summary = moveformer_executor.score_full(
                    run_payload=run_payload,
                    resolved_artifacts=artifacts,
                    restore_from={
                        "checkpoint_path": checkpoint_path,
                        "training_run_key": "train-key",
                        "training_signature": "signature",
                    },
                    run_dir=Path(tmpdir) / "score",
                    tracker=object(),
                )

        self.assertEqual(summary["train_steps"], 9)
        self.assertTrue(_FakeDataset.calls[0].kwargs["lazy_rows"])
        predict.assert_called_once()
        train.assert_not_called()


class MoveFormerTaxonAssetTest(unittest.TestCase):
    def test_safe_asset_loader_maps_studies_without_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            mapping = root / "taxa.csv"
            embeddings = root / "taxa.npz"
            with mapping.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "study_id",
                        "species",
                        "taxon_match_rank",
                        "taxon_match",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "study_id": "study",
                        "species": "Species name",
                        "taxon_match_rank": "species",
                        "taxon_match": "Species name",
                    }
                )
            np.savez(embeddings, vectors=np.arange(4, dtype=np.float32)[None, :])
            with (
                patch.object(moveformer_taxa, "_MAPPING_PATH", mapping),
                patch.object(moveformer_taxa, "_EMBEDDINGS_PATH", embeddings),
            ):
                moveformer_taxa._taxa.cache_clear()
                vector, metadata = moveformer_taxa.moveformer_taxon_for_study("study")
                with self.assertRaises(KeyError):
                    moveformer_taxa.moveformer_taxon_for_study("missing")
            moveformer_taxa._taxa.cache_clear()

        np.testing.assert_array_equal(vector, np.arange(4, dtype=np.float32))
        self.assertEqual(metadata["taxon_match_rank"], "species")


if __name__ == "__main__":
    unittest.main()
