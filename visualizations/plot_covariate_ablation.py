from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from movebench import paths
import movebench.bench.results as bench_results
from movebench.geometry import LOCAL_GEOMETRY
from movebench.study import MINI_STUDIES

COVARIATE_COMPARISON_EXPERIMENT = (
    paths.repo_root() / "experiments" / "covariate_comparison.py"
)
MAIN_RESULTS_EXPERIMENT = paths.repo_root() / "experiments" / "main_results.py"
DEFAULT_OUTPUT_STEM = (
    paths.repo_root()
    / "results"
    / "bench"
    / "report"
    / "cov-1-withrw-average-2row"
)
NORMALIZE_BY = "baseline_nomove"
RANDOM_WALK = "baseline_randomwalk"
PLOT_METHODS = ["issf_linear", "issf_mlp", "deepssf", "moveformer"]
AVERAGE_METHOD = "all_average"
SPLIT_ORDER = ["test_temporal", "test_individual"]
SPLIT_LABELS = {
    "test_temporal": "Mini Temporal Split",
    "test_individual": "Mini Individual Split",
}
TIMESCALE_ORDER = bench_results.TIMESCALE_ORDER
COVARIATE_SET_ORDER = [
    "lite",
    "all_environmental",
    "alphaearth_only",
    "alphaearth_and_marine",
    "all",
]
COVARIATE_COLORS = {
    "lite": "#ADCFFF",
    "all_environmental": "#3a8cff",
    "alphaearth_only": "#288077",
    "alphaearth_and_marine": "#1b9aaa",
    "all": "#00409a",
}
AVERAGE_PANEL_BACKGROUND_COLOR = "#f2f2f2"
AVERAGE_PANEL_BACKGROUND_PAD = 0.01


def _method_label(method: str) -> str:
    return {
        "issf_linear": "iSSF Linear",
        "issf_mlp": "iSSF MLP",
        "deepssf": "deepSSF",
        "moveformer": "MoveFormer",
        AVERAGE_METHOD: "All",
    }[method]


def _covariate_label(covariate_set: str) -> str:
    return {
        "lite": "env lite",
        "all_environmental": "env all",
        "alphaearth_only": "alphaearth",
        "alphaearth_and_marine": "alphaearth + marine",
        "all": "alphaearth + env all",
    }[covariate_set]


def load_plot_results() -> pd.DataFrame:
    model_results = bench_results.load_results(
        "covariate_comparison",
        experiment_module_path=COVARIATE_COMPARISON_EXPERIMENT,
    )
    baseline_results = bench_results.load_results(
        "main_results",
        experiment_module_path=MAIN_RESULTS_EXPERIMENT,
    )
    results = pd.concat(
        [
            model_results[model_results["method"].isin(PLOT_METHODS)],
            baseline_results[
                baseline_results["method"].isin([NORMALIZE_BY, RANDOM_WALK])
            ],
        ],
        ignore_index=True,
    )
    return results[
        results["study_id"].isin(MINI_STUDIES)
        & (results["geometry"] == LOCAL_GEOMETRY)
        & results["split_key"].isin(SPLIT_ORDER)
        & results["covariate_set"].isin(COVARIATE_SET_ORDER)
    ].reset_index(drop=True)


def aggregate_plot_results(results: pd.DataFrame) -> pd.DataFrame:
    normalized = bench_results.normalize_results(
        results,
        baseline_method=NORMALIZE_BY,
        by=["study_id", "timescale", "split_key"],
    )
    aggregate = bench_results.aggregate_results(
        normalized,
        by=["split_key", "method", "covariate_set", "timescale"],
    )
    average = (
        aggregate[aggregate["method"].isin(PLOT_METHODS)]
        .groupby(["split_key", "covariate_set", "timescale"], as_index=False)
        .agg(
            normalized_energy_score_mean=(
                "normalized_energy_score_mean",
                "mean",
            )
        )
        .assign(method=AVERAGE_METHOD)
    )
    return pd.concat([aggregate, average], ignore_index=True)


def write_outputs(aggregate: pd.DataFrame, output_stem: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plot_methods = [*PLOT_METHODS, AVERAGE_METHOD]
    x_index = {timescale: index for index, timescale in enumerate(TIMESCALE_ORDER)}
    fig, axes = plt.subplots(
        len(SPLIT_ORDER),
        len(plot_methods),
        figsize=(4 * len(plot_methods), 4 * len(SPLIT_ORDER)),
        sharex=True,
        squeeze=False,
    )

    for row, split_key in enumerate(SPLIT_ORDER):
        split_results = aggregate[aggregate["split_key"] == split_key]
        random_walk = split_results[split_results["method"] == RANDOM_WALK].copy()
        random_walk["x"] = random_walk["timescale"].map(x_index)
        random_walk = random_walk.sort_values("x")

        for col, method in enumerate(plot_methods):
            ax = axes[row, col]
            method_results = split_results[split_results["method"] == method]

            if not random_walk.empty:
                ax.plot(
                    random_walk["x"],
                    random_walk["normalized_energy_score_mean"],
                    marker=None,
                    linewidth=2.0,
                    linestyle="--",
                    color="#6c757d",
                    label="crw",
                )

            for covariate_set in COVARIATE_SET_ORDER:
                covariate_rows = method_results[
                    method_results["covariate_set"] == covariate_set
                ].copy()
                covariate_rows["x"] = covariate_rows["timescale"].map(x_index)
                covariate_rows = covariate_rows.sort_values("x")
                if covariate_rows.empty:
                    continue
                ax.plot(
                    covariate_rows["x"],
                    covariate_rows["normalized_energy_score_mean"],
                    marker="o",
                    linewidth=2.0,
                    color=COVARIATE_COLORS[covariate_set],
                    label=_covariate_label(covariate_set),
                )

            ax.set_xticks(range(len(TIMESCALE_ORDER)), TIMESCALE_ORDER)
            ax.grid(True, linewidth=0.4, alpha=0.4)
            if row == 0:
                ax.set_title(_method_label(method))

        axes[row, 0].set_ylabel(f"Norm ES ↓\n{SPLIT_LABELS[split_key]}")

    handles, labels = [], []
    for row_axes in axes:
        for ax in row_axes:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=max(1, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.9),
    )
    fig.suptitle(
        r"$\bf{Covariate~comparison:}$ Energy score (↓) on MoveBench-Mini",
        y=0.94,
        fontsize=26,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.canvas.draw()

    for average_ax in axes[:, -1]:
        average_ax.set_facecolor("white")
    average_bboxes = [
        ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        for ax in axes[:, -1]
    ]
    background_left = max(
        0.0,
        min(box.x0 for box in average_bboxes) - AVERAGE_PANEL_BACKGROUND_PAD,
    )
    background_bottom = max(
        0.0,
        min(box.y0 for box in average_bboxes) - AVERAGE_PANEL_BACKGROUND_PAD,
    )
    background_right = 1.0
    background_top = min(
        1.0,
        max(box.y1 for box in average_bboxes) + AVERAGE_PANEL_BACKGROUND_PAD,
    )
    average_background_ax = fig.add_axes(
        [
            background_left,
            background_bottom,
            background_right - background_left,
            background_top - background_bottom,
        ],
        zorder=0,
    )
    average_background_ax.set_facecolor(AVERAGE_PANEL_BACKGROUND_COLOR)
    average_background_ax.set_xticks([])
    average_background_ax.set_yticks([])
    for spine in average_background_ax.spines.values():
        spine.set_visible(False)
    average_background_ax.set_in_layout(False)
    for row_axes in axes:
        for ax in row_axes:
            ax.set_zorder(1)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(output_stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the MoveBench-Mini covariate ablation."
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    args = parse_args()
    write_outputs(aggregate_plot_results(load_plot_results()), args.output_stem)
    print(args.output_stem.with_suffix(".pdf"))
    print(args.output_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
