import unittest

import numpy as np
import pandas as pd
from pandas.errors import MergeError

import movebench.bench.report as bench_report
import movebench.bench.results as bench_results


class BenchResultsTest(unittest.TestCase):
    def test_load_results_deduplicates_shared_runs_and_keeps_metrics(self):
        run = {
            "run_key": "shared",
            "study_id": "study-a",
            "timescale": "1d",
            "method": "issf_linear",
            "method_variant": "small",
            "covariate_set": "all",
            "seed": 3,
            "geometry": "local-v2",
            "evaluation_split_keys": ["test-temporal", "test_individual"],
        }
        expanded = [
            {"experiment": {"name": "main_results"}, "runs": [run]},
            {"experiment": {"name": "covariate_comparison"}, "runs": [run]},
        ]
        latest = {
            "shared": {
                "status": "completed",
                "run_id": "run-1",
                "metrics": {
                    "energy_score.test_temporal": 1.25,
                    "energy_score.accuracy.test_temporal": 2.0,
                    "energy_score.calibration.test_temporal": 0.5,
                    "n_steps_scored.test_temporal": 10,
                    "n_steps_excluded_out_of_support.test_temporal": 2,
                },
            }
        }

        result = bench_results.load_results(
            ["main_results", "covariate_comparison"],
            latest_run_records=latest,
            expanded_experiments=expanded,
        )

        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row["experiment_name"], "main_results")
        self.assertEqual(
            row["experiment_names"],
            ("main_results", "covariate_comparison"),
        )
        self.assertEqual(row["method_variant"], "small")
        self.assertEqual(row["geometry"], "local-v2")
        self.assertEqual(row["split_key"], "test-temporal")
        self.assertEqual(row["energy_score"], 1.25)
        self.assertEqual(row["accuracy"], 2.0)
        self.assertEqual(row["calibration"], 0.5)
        self.assertEqual(row["n_steps_scored"], 10)
        self.assertEqual(row["n_steps_excluded_out_of_support"], 2)

    def test_load_results_can_select_an_exact_experiment_module(self):
        main_run = {
            "run_key": "main",
            "study_id": "study-a",
            "timescale": "1d",
            "method": "issf_linear",
            "covariate_set": "all",
            "seed": 0,
            "geometry": "local-v2",
            "evaluation_split_keys": ["test_temporal"],
        }
        moveformer_run = {
            **main_run,
            "run_key": "moveformer_perstudy",
            "method": "moveformer_perstudy",
        }
        expanded = [
            {
                "module_path": "/repo/experiments/main_results.py",
                "experiment": {"name": "main_results"},
                "runs": [main_run],
            },
            {
                "module_path": "/repo/experiments/moveformer_perstudy.py",
                "experiment": {"name": "ablation"},
                "runs": [moveformer_run],
            },
        ]
        latest = {
            run_key: {
                "status": "completed",
                "metrics": {"energy_score.test_temporal": energy_score},
            }
            for run_key, energy_score in [("main", 1.0), ("moveformer_perstudy", 2.0)]
        }

        result = bench_results.load_results(
            "main_results",
            experiment_module_path="/repo/experiments/main_results.py",
            latest_run_records=latest,
            expanded_experiments=expanded,
        )

        self.assertEqual(result["run_key"].tolist(), ["main"])

    def test_report_loader_delegates_to_standard_results(self):
        expanded = [
            {
                "experiment": {"name": "main_results"},
                "runs": [
                    {
                        "run_key": "run-a",
                        "study_id": "study-a",
                        "timescale": "1d",
                        "method": "issf_linear",
                        "covariate_set": "all",
                        "seed": 0,
                        "evaluation_split_keys": ["test_temporal"],
                    }
                ],
            }
        ]
        result = bench_report.load_main_results(
            latest_run_records={
                "run-a": {
                    "status": "completed",
                    "run_id": "run-a",
                    "metrics": {"energy_score.test_temporal": 1.25},
                }
            },
            expanded_experiments=expanded,
        )

        self.assertEqual(result["run_key"].tolist(), ["run-a"])
        self.assertEqual(result["energy_score"].tolist(), [1.25])

    def test_report_loader_keeps_all_experiment_provenance(self):
        run = {
            "run_key": "run-a",
            "study_id": "study-a",
            "timescale": "1d",
            "method": "issf_linear",
            "covariate_set": "all",
            "evaluation_split_keys": ["test_temporal"],
        }
        result = bench_report.load_main_results(
            latest_run_records={
                "run-a": {
                    "status": "completed",
                    "metrics": {"energy_score.test_temporal": 1.25},
                }
            },
            expanded_experiments=[
                {"experiment": {"name": "main_results"}, "runs": [run]},
                {
                    "experiment": {"name": "covariate_comparison"},
                    "runs": [run],
                },
            ],
        )

        self.assertEqual(
            result.loc[0, "experiment_names"],
            ("main_results", "covariate_comparison"),
        )

    def test_normalize_results_pairs_by_study_before_aggregation(self):
        results = pd.DataFrame.from_records(
            [
                {
                    "study_id": "a",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "baseline_nomove",
                    "energy_score": 1.0,
                },
                {
                    "study_id": "a",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "model",
                    "energy_score": 2.0,
                },
                {
                    "study_id": "b",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "baseline_nomove",
                    "energy_score": 100.0,
                },
                {
                    "study_id": "b",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "model",
                    "energy_score": 100.0,
                },
                {
                    "study_id": "missing-baseline",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "model",
                    "energy_score": 3.0,
                },
                {
                    "study_id": "zero-baseline",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "baseline_nomove",
                    "energy_score": 0.0,
                },
                {
                    "study_id": "zero-baseline",
                    "timescale": "1d",
                    "split_key": "test_temporal",
                    "method": "model",
                    "energy_score": 3.0,
                },
            ]
        )

        normalized = bench_results.normalize_results(
            results,
            baseline_method="baseline_nomove",
            by=["study_id", "timescale", "split_key"],
        )
        aggregate = bench_results.aggregate_results(
            normalized,
            by=["method", "timescale", "split_key"],
        )

        self.assertEqual(normalized["study_id"].tolist(), ["a", "b"])
        self.assertEqual(normalized["normalized_energy_score"].tolist(), [2.0, 1.0])
        self.assertEqual(aggregate.loc[0, "normalized_energy_score_mean"], 1.5)
        self.assertEqual(aggregate.loc[0, "n_studies"], 2)

    def test_normalize_results_rejects_ambiguous_baselines(self):
        results = pd.DataFrame.from_records(
            [
                {"study_id": "a", "method": "baseline_nomove", "energy_score": 1.0},
                {"study_id": "a", "method": "baseline_nomove", "energy_score": 2.0},
                {"study_id": "a", "method": "model", "energy_score": 3.0},
            ]
        )

        with self.assertRaises(MergeError):
            bench_results.normalize_results(
                results,
                baseline_method="baseline_nomove",
                by=["study_id"],
            )

    def test_aggregate_results_weights_each_study_once(self):
        results = pd.DataFrame.from_records(
            [
                {"study_id": "a", "method": "model", "value": 2.0},
                {"study_id": "b", "method": "model", "value": 4.0},
                {"study_id": "c", "method": "other", "value": 5.0},
            ]
        )

        aggregate = bench_results.aggregate_results(
            results,
            by=["method"],
            value="value",
        ).set_index("method")

        self.assertEqual(aggregate.loc["model", "value_mean"], 3.0)
        self.assertAlmostEqual(aggregate.loc["model", "value_std"], np.sqrt(2.0))
        self.assertAlmostEqual(aggregate.loc["model", "value_sem"], 1.0)
        self.assertEqual(aggregate.loc["model", "n_studies"], 2)
        self.assertTrue(np.isnan(aggregate.loc["other", "value_std"]))
        self.assertEqual(aggregate.loc["other", "value_sem"], 0.0)

    def test_aggregate_results_rejects_duplicate_study_cells(self):
        results = pd.DataFrame.from_records(
            [
                {"study_id": "a", "method": "model", "value": 1.0},
                {"study_id": "a", "method": "model", "value": 3.0},
            ]
        )

        with self.assertRaisesRegex(ValueError, "Multiple result rows"):
            bench_results.aggregate_results(
                results,
                by=["method"],
                value="value",
            )


if __name__ == "__main__":
    unittest.main()
