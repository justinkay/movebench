import tempfile
import unittest
from pathlib import Path
from unittest import mock

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
import pandas as pd

from visualizations import plot_energy_scores_by_distance_group as energy_plot


class EnergyScoreDistancePlotTest(unittest.TestCase):
    def test_plot_data_cache_skips_expensive_loads(self):
        aggregate = pd.DataFrame({"value": [1.0]})
        individual_counts = pd.DataFrame({"n_individuals": [2]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "plot.cache.pkl"
            pd.to_pickle(
                {
                    "aggregate": aggregate,
                    "individual_counts": individual_counts,
                },
                cache_path,
            )
            with (
                mock.patch.object(energy_plot, "build_study_groups") as build,
                mock.patch.object(energy_plot, "load_plot_results") as load,
            ):
                cached_aggregate, cached_counts = (
                    energy_plot.load_or_build_plot_data(
                        Path("studies.csv"),
                        Path("traits.csv"),
                        cache_path,
                        refresh_cache=False,
                    )
                )

        pd.testing.assert_frame_equal(cached_aggregate, aggregate)
        pd.testing.assert_frame_equal(cached_counts, individual_counts)
        build.assert_not_called()
        load.assert_not_called()

    def test_plot_data_cache_can_be_refreshed(self):
        aggregate = pd.DataFrame({"value": [1.0]})
        individual_counts = pd.DataFrame({"n_individuals": [2]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "plot.cache.pkl"
            pd.to_pickle(
                {
                    "aggregate": pd.DataFrame({"value": [0.0]}),
                    "individual_counts": pd.DataFrame({"n_individuals": [0]}),
                },
                cache_path,
            )
            with (
                mock.patch.object(
                    energy_plot,
                    "build_study_groups",
                    return_value=pd.DataFrame(),
                ),
                mock.patch.object(
                    energy_plot,
                    "load_plot_results",
                    return_value=pd.DataFrame(),
                ),
                mock.patch.object(
                    energy_plot,
                    "aggregate_plot_results",
                    return_value=aggregate,
                ),
                mock.patch.object(
                    energy_plot,
                    "load_individual_results",
                    return_value=pd.DataFrame(),
                ),
                mock.patch.object(
                    energy_plot,
                    "count_plot_individuals",
                    return_value=individual_counts,
                ),
            ):
                energy_plot.load_or_build_plot_data(
                    Path("studies.csv"),
                    Path("traits.csv"),
                    cache_path,
                    refresh_cache=True,
                )

            cached = pd.read_pickle(cache_path)

        pd.testing.assert_frame_equal(cached["aggregate"], aggregate)
        pd.testing.assert_frame_equal(
            cached["individual_counts"],
            individual_counts,
        )

    def test_individual_counts_are_unique_within_study_and_panel(self):
        combinations = [
            ("a", "1h", "test_temporal", "shared"),
            ("a", "12h", "test_temporal", "shared"),
            ("b", "1h", "test_temporal", "shared"),
            ("c", "1h", "test_temporal", "other"),
            ("a", "1h", "test_individual", "held"),
        ]
        rows = []
        for study_id, timescale, split_key, individual_id in combinations:
            for method, energy_score in [
                ("baseline_nomove", 2.0),
                ("issf_linear", 1.0),
            ]:
                rows.append(
                    {
                        "study_id": study_id,
                        "timescale": timescale,
                        "split_key": split_key,
                        "individual_id": individual_id,
                        "method": method,
                        "energy_score": energy_score,
                    }
                )
        study_groups = pd.DataFrame.from_records(
            [
                {"study_id": "a", "panel": "panel-1"},
                {"study_id": "b", "panel": "panel-1"},
                {"study_id": "c", "panel": "panel-2"},
            ]
        )

        counts = energy_plot.count_plot_individuals(
            pd.DataFrame.from_records(rows),
            study_groups,
        )
        observed = {
            (row.split_key, row.panel): int(row.n_individuals)
            for row in counts.itertuples(index=False)
        }

        self.assertEqual(observed[("test_temporal", "panel-1")], 2)
        self.assertEqual(observed[("test_temporal", "panel-2")], 1)
        self.assertEqual(
            observed[("test_temporal", energy_plot.AVERAGE_PANEL)],
            3,
        )
        self.assertEqual(observed[("test_individual", "panel-1")], 1)
        self.assertEqual(
            observed[("test_individual", energy_plot.AVERAGE_PANEL)],
            1,
        )

    def test_average_panel_pools_studies_instead_of_panel_means(self):
        rows = []
        for study_id, model_score in [("a", 1.0), ("b", 3.0), ("c", 8.0)]:
            for method, energy_score in [
                ("baseline_nomove", 1.0),
                ("issf_linear", model_score),
            ]:
                rows.append(
                    {
                        "study_id": study_id,
                        "timescale": "1h",
                        "split_key": "test_temporal",
                        "method": method,
                        "energy_score": energy_score,
                    }
                )
        study_groups = pd.DataFrame.from_records(
            [
                {"study_id": "a", "panel": "panel-1"},
                {"study_id": "b", "panel": "panel-1"},
                {"study_id": "c", "panel": "panel-2"},
            ]
        )

        aggregate = energy_plot.aggregate_plot_results(
            pd.DataFrame.from_records(rows),
            study_groups,
        )
        average = aggregate[
            (aggregate["panel"] == energy_plot.AVERAGE_PANEL)
            & (aggregate["method"] == "issf_linear")
        ].iloc[0]

        self.assertEqual(average["normalized_energy_score_mean"], 4.0)
        self.assertEqual(average["n_studies"], 3)

    def test_renderer_matches_golden_plot_structure_and_style(self):
        rows = []
        for split_key in energy_plot.SPLIT_ORDER:
            for panel in energy_plot.PANEL_ORDER:
                for method in [*energy_plot.PLOT_METHODS, *energy_plot.OPTIONAL_PLOT_METHODS]:
                    for timescale in energy_plot.TIMESCALE_ORDER:
                        rows.append(
                            {
                                "split_key": split_key,
                                "split_label": energy_plot.SPLIT_LABELS[split_key],
                                "panel": panel,
                                "method": method,
                                "timescale": timescale,
                                "normalized_energy_score_mean": (
                                    0.9
                                    if split_key == "test_temporal"
                                    else 1.1
                                ),
                                "normalized_energy_score_sem": 0.01,
                            }
                        )
        counts = pd.DataFrame.from_records(
            [
                {
                    "split_key": split_key,
                    "panel": panel,
                    "n_individuals": (
                        70 if panel == energy_plot.AVERAGE_PANEL else 10
                    ),
                }
                for split_key in energy_plot.SPLIT_ORDER
                for panel in energy_plot.PANEL_ORDER
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_stem = Path(tmp_dir) / "energy"
            with (
                mock.patch.object(Figure, "savefig"),
                mock.patch.object(energy_plot.plt, "close"),
            ):
                energy_plot.write_outputs(
                    pd.DataFrame.from_records(rows),
                    counts,
                    output_stem,
                )
                figure = energy_plot.plt.gcf()

            axes = figure.axes[:14]
            self.assertEqual(tuple(figure.get_size_inches()), (17.5, 7.0))
            self.assertEqual(len(figure.axes), 15)
            self.assertEqual(
                [axis.get_title() for axis in axes[:7]],
                [energy_plot.PANEL_TITLES[panel] for panel in energy_plot.PANEL_ORDER],
            )
            self.assertEqual(axes[6].get_title(), "All")
            self.assertEqual(axes[0].get_ylim(), (0.65, 1.2))
            self.assertEqual(axes[7].get_ylim(), (0.7, 1.35))
            self.assertEqual(axes[0].get_yscale(), "log")
            self.assertEqual(axes[1].get_yticks().tolist(), [])
            self.assertTrue(all(len(axis.texts) == 1 for axis in axes))
            self.assertEqual(axes[0].texts[0].get_text(), "n=10")
            self.assertEqual(axes[6].texts[0].get_text(), "n=70")
            self.assertEqual(axes[0].lines[0].get_label(), "don't move")
            self.assertEqual(axes[0].lines[1].get_linestyle(), "--")
            self.assertEqual(axes[0].lines[1].get_marker(), "None")
            self.assertEqual(axes[0].lines[1].get_color(), "#363636")
            self.assertEqual(axes[0].lines[-1].get_color(), "#669bbc")
            self.assertEqual(
                [text.get_text() for text in figure.legends[0].get_texts()],
                [
                    "don't move",
                    " crw",
                    "issf linear",
                    "issf mlp",
                    "deepssf",
                    "moveformer",
                ],
            )
            self.assertEqual(
                figure._suptitle.get_text(),
                r"$\bf{Method~comparison:}$ Energy score (↓) on MoveBench",
            )
            self.assertEqual(figure.axes[-1].get_facecolor(), to_rgba("#f2f2f2"))
            energy_plot.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
