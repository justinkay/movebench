from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from movebench import paths
import movebench.bench.results as bench_results
from movebench.geometry import LOCAL_GEOMETRY, geodesic_distance_m
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    TIME_COL,
    load_trajectories_csv,
)

MAIN_RESULTS_EXPERIMENT = paths.repo_root() / "experiments" / "main_results.py"
DEFAULT_STUDIES_CSV = paths.repo_root() / "MoveBench Studies V2.csv"
DEFAULT_TRAITS_CSV = (
    paths.repo_root().parent
    / "movebench-figures"
    / "outputs"
    / "traits"
    / "movebench_traits.csv"
)
DEFAULT_OUTPUT_STEM = (
    paths.repo_root()
    / "results"
    / "bench"
    / "report"
    / "energy_scores_by_distance_group"
)
NORMALIZE_BY = "baseline_nomove"
PLOT_METHODS = [
    "baseline_randomwalk",
    "issf_linear",
    "issf_mlp",
    "deepssf",
    "moveformer",
]
SPLIT_ORDER = ["test_temporal", "test_individual"]
SPLIT_LABELS = {
    "test_temporal": "Temporal Split",
    "test_individual": "Individual Split",
}
TIMESCALE_ORDER = bench_results.TIMESCALE_ORDER
SHORT_KM = 25.0
LONG_KM = 650.0
AVERAGE_PANEL = "all_average"
PANEL_ORDER = [
    "bird_short_medium",
    "bird_long",
    "mammal_short",
    "mammal_medium_long",
    "marine_all",
    "reptiles_all",
    AVERAGE_PANEL,
]
PANEL_TITLES = {
    "bird_short_medium": "Bird < 650km",
    "bird_long": "Bird > 650km",
    "mammal_short": "Mammal < 25km",
    "mammal_medium_long": "Mammal > 25km",
    "marine_all": "Marine",
    "reptiles_all": "Terrestrial Reptile",
    AVERAGE_PANEL: "All",
}
METHOD_COLORS = {
    "baseline_randomwalk": "#363636",
    "issf_linear": "#c1121f",
    "issf_mlp": "#f77f00",
    "deepssf": "#003049",
    "moveformer": "#669bbc",
}
STUDY_ID_OVERRIDES = {
    (
        "Antarctic Petrel",
        "Thalassoica antarctica",
    ): "klappstein2024_smoothSSF_data_petrel",
    (
        "Fiedler_0a3ec70a; Fiedler_520a64ae; Fiedler_5f8c5199",
        "Ciconia ciconia",
    ): "Fiedler_lifetrack_stork_all",
    (
        "Klzsch_22d42221; Klzsch_2e9dace9",
        "Anser albifrons",
    ): "Klzsch_22d42221_2e9dace9_all",
    ("AMT Deer", "Cervus elaphus"): "amtdeer_wgs84",
    ("Alaska Sea Lion", "Eumetopias jubatus"): "wilson2018_sea_lion_telemetry",
    ("Australia water buffalo", "Bubalus bubalis"): "forrest2024_buffalo",
    ("Botswana Hyena", "Crocuta crocuta"): "cazzi2016_hyena",
    ("Colorado Mountain Lion", "Puma concolor"): "hooten2018_mountain_lion_AM80",
    ("Desert Pronghorn", "Antilocapra americana"): "aikens2025_pronghorn",
    ("Elephant seal", "Mirounga angustirostris"): "costa2024_elephant_seals",
    ("Illinois Deer", "Odocoileus virginianus"): "egan2025_deer",
    ("Kenya Wildebeast", "Connochaetes taurinus"): "stabach2020_wildebeest",
    ("Mississippi Pig", "Sus scrofa"): "potts2022_pig",
    ("Mountain Pronghorn", "Antilocapra americana"): "aikens2024_pronghorn",
    ("Namibia Elephants", "Loxodonta africana"): "killian_elephants",
    ("Tsavo Lions", "Panthera leo"): "tsavo_lions",
}
DETAIL_GROUP_OVERRIDES = {
    ("Setyawan_0425c92c", "Mobula"): "Marine Sharks and Rays",
    ("Botswana Hyena", "Crocuta crocuta"): "Terrestrial Mammals",
}

def _method_label(method: str) -> str:
    return (
        str(method)
        .replace("baseline", "")
        .replace("_", " ")
        .replace("randomwalk", "crw")
    )


def _resolve_study_id(
    study_name: str,
    study_species: str,
    available_study_ids: set[str],
) -> str | None:
    override = STUDY_ID_OVERRIDES.get((study_name, study_species))
    if override:
        return override
    if study_name in available_study_ids:
        return study_name
    matches = [
        study_id
        for study_id in available_study_ids
        if study_id.startswith(study_name)
    ]
    if len(matches) == 1:
        return matches[0]
    species_suffix = study_species.replace(" ", "_")
    species_matches = [
        study_id for study_id in matches if species_suffix in study_id
    ]
    return species_matches[0] if len(species_matches) == 1 else None


def _group_lookup(traits_csv: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with traits_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            group = str(row["group"]).strip()
            if not group:
                continue
            if group == "Mammals":
                group = "Terrestrial Mammals"
            for key in ("study_species", "study_species_original"):
                species = str(row[key]).strip()
                if species:
                    lookup[species] = group
    lookup["Bubalus bubalis"] = "Freshwater Mammals"
    lookup["Mobula"] = "Marine Sharks and Rays"
    return lookup


def _high_level_group(detail_group: str | None) -> str | None:
    if not detail_group:
        return None
    if detail_group.startswith("Marine"):
        return "Marine"
    if "Bird" in detail_group:
        return "Birds"
    if "Mammal" in detail_group:
        return "Terrestrial mammals"
    if "Reptile" in detail_group:
        return "Reptiles"
    return None


def _distance_bin(distance_km: float) -> str:
    if distance_km < SHORT_KM:
        return "short"
    if distance_km < LONG_KM:
        return "medium"
    return "long"


def _panel(group: str, distance_bin: str) -> str | None:
    if group == "Birds":
        return "bird_long" if distance_bin == "long" else "bird_short_medium"
    if group == "Terrestrial mammals":
        return "mammal_short" if distance_bin == "short" else "mammal_medium_long"
    if group == "Marine":
        return "marine_all"
    if group == "Reptiles":
        return "reptiles_all"
    return None


def build_study_groups(studies_csv: Path, traits_csv: Path) -> pd.DataFrame:
    available_study_ids = {
        path.name for path in paths.studies_root().iterdir() if path.is_dir()
    }
    group_lookup = _group_lookup(traits_csv)
    rows: list[dict[str, object]] = []
    with studies_csv.open(newline="") as handle:
        for roster_row in csv.DictReader(handle):
            study_name = str(roster_row["study_name"]).strip()
            study_species = str(roster_row["study_species"]).strip()
            if not study_name or not study_species:
                continue
            study_id = _resolve_study_id(
                study_name,
                study_species,
                available_study_ids,
            )
            detail_group = DETAIL_GROUP_OVERRIDES.get(
                (study_name, study_species),
                group_lookup.get(study_species),
            )
            group = _high_level_group(detail_group)
            if not study_id or not group:
                continue

            trajectories = load_trajectories_csv(
                paths.study_dir(study_id) / "trajectories.csv"
            ).dropna(subset=[INDIVIDUAL_COL, LAT_COL, LON_COL, TIME_COL])
            trajectories = trajectories.sort_values([INDIVIDUAL_COL, TIME_COL])
            trajectories["_start_lat"] = trajectories.groupby(
                INDIVIDUAL_COL,
                sort=False,
            )[LAT_COL].transform("first")
            trajectories["_start_lon"] = trajectories.groupby(
                INDIVIDUAL_COL,
                sort=False,
            )[LON_COL].transform("first")
            trajectories["_net_displacement_m"] = geodesic_distance_m(
                trajectories["_start_lat"].to_numpy(),
                trajectories["_start_lon"].to_numpy(),
                trajectories[LAT_COL].to_numpy(),
                trajectories[LON_COL].to_numpy(),
            )
            individuals = (
                trajectories.groupby(INDIVIDUAL_COL, as_index=False, sort=False)
                .agg(
                    max_net_displacement_m=("_net_displacement_m", "max"),
                    n_points=(INDIVIDUAL_COL, "size"),
                )
                .query("n_points >= 2")
            )
            if individuals.empty:
                continue
            median_distance_km = float(
                individuals["max_net_displacement_m"].median() / 1000.0
            )
            distance_bin = _distance_bin(median_distance_km)
            panel = _panel(group, distance_bin)
            if panel:
                rows.append(
                    {
                        "study_id": study_id,
                        "study_species": study_species,
                        "high_level_group": group,
                        "median_max_displacement_km": median_distance_km,
                        "distance_bin": distance_bin,
                        "panel": panel,
                    }
                )
    return pd.DataFrame.from_records(
        rows,
        columns=[
            "study_id",
            "study_species",
            "high_level_group",
            "median_max_displacement_km",
            "distance_bin",
            "panel",
        ],
    )


def load_plot_results() -> pd.DataFrame:
    results = bench_results.load_results(
        "main_results",
        experiment_module_path=MAIN_RESULTS_EXPERIMENT,
    )
    return results[
        results["method"].isin([NORMALIZE_BY, *PLOT_METHODS])
        & (results["geometry"] == LOCAL_GEOMETRY)
        & results["split_key"].isin(SPLIT_ORDER)
    ].reset_index(drop=True)


def load_individual_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    trajectory_indexes: dict[str, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []

    for result in raw_results.itertuples(index=False):
        study_id = str(result.study_id)
        if study_id not in trajectory_indexes:
            trajectory_index = pd.read_csv(
                paths.study_dir(study_id) / "trajectories.csv",
                usecols=["point_id", INDIVIDUAL_COL],
            ).rename(
                columns={
                    "point_id": "point_id_start",
                    INDIVIDUAL_COL: "individual_id",
                }
            )
            trajectory_index["point_id_start"] = trajectory_index[
                "point_id_start"
            ].astype(int)
            trajectory_index["individual_id"] = trajectory_index[
                "individual_id"
            ].astype(str)
            trajectory_indexes[study_id] = trajectory_index

        energy = pd.read_csv(
            Path(result.run_dir)
            / "final_metrics"
            / str(result.split_key)
            / "energy_by_step.csv.gz",
            usecols=["point_id_start", "energy_score"],
        )
        energy["point_id_start"] = energy["point_id_start"].astype(int)
        grouped = (
            energy.merge(
                trajectory_indexes[study_id],
                on="point_id_start",
                how="left",
                validate="many_to_one",
            )
            .groupby("individual_id", as_index=False, sort=False)["energy_score"]
            .mean()
        )
        grouped["study_id"] = study_id
        grouped["timescale"] = result.timescale
        grouped["split_key"] = result.split_key
        grouped["method"] = result.method
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def count_plot_individuals(
    individual_results: pd.DataFrame,
    study_groups: pd.DataFrame,
) -> pd.DataFrame:
    normalized = bench_results.normalize_results(
        individual_results,
        baseline_method=NORMALIZE_BY,
        by=["study_id", "timescale", "split_key", "individual_id"],
    ).merge(
        study_groups[["study_id", "panel"]],
        on="study_id",
        how="inner",
        validate="many_to_one",
    )
    panel_counts = (
        normalized[["split_key", "panel", "study_id", "individual_id"]]
        .drop_duplicates()
        .groupby(["split_key", "panel"], as_index=False)
        .size()
        .rename(columns={"size": "n_individuals"})
    )
    average_counts = (
        normalized[["split_key", "study_id", "individual_id"]]
        .drop_duplicates()
        .groupby("split_key", as_index=False)
        .size()
        .rename(columns={"size": "n_individuals"})
        .assign(panel=AVERAGE_PANEL)
    )
    return pd.concat([panel_counts, average_counts], ignore_index=True)


def aggregate_plot_results(
    raw_results: pd.DataFrame,
    study_groups: pd.DataFrame,
) -> pd.DataFrame:
    normalized = bench_results.normalize_results(
        raw_results[raw_results["method"].isin([NORMALIZE_BY, *PLOT_METHODS])],
        baseline_method=NORMALIZE_BY,
        by=["study_id", "timescale", "split_key"],
    )
    normalized = normalized.merge(
        study_groups,
        on="study_id",
        how="inner",
        validate="many_to_one",
    )
    panel_results = bench_results.aggregate_results(
        normalized,
        by=["split_key", "panel", "method", "timescale"],
    )
    average_results = bench_results.aggregate_results(
        normalized,
        by=["split_key", "method", "timescale"],
    ).assign(panel=AVERAGE_PANEL)
    aggregate = pd.concat([panel_results, average_results], ignore_index=True)
    return aggregate.assign(
        split_label=lambda df: df["split_key"].map(SPLIT_LABELS),
        panel_title=lambda df: df["panel"].map(PANEL_TITLES),
        method_label=lambda df: df["method"].map(_method_label),
    ).sort_values(
        ["split_key", "panel", "method", "timescale"],
        key=lambda values: values.map(
            {
                **{key: index for index, key in enumerate(SPLIT_ORDER)},
                **{key: index for index, key in enumerate(PANEL_ORDER)},
                **{key: index for index, key in enumerate(PLOT_METHODS)},
                **{key: index for index, key in enumerate(TIMESCALE_ORDER)},
            }
        ),
    )


def load_or_build_plot_data(
    studies_csv: Path,
    traits_csv: Path,
    cache_path: Path,
    refresh_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_pickle(cache_path)
        print(f"Loaded cached plot data from {cache_path}")
        return cached["aggregate"], cached["individual_counts"]

    study_groups = build_study_groups(studies_csv, traits_csv)
    raw_results = load_plot_results()
    aggregate = aggregate_plot_results(raw_results, study_groups)
    individual_counts = count_plot_individuals(
        load_individual_results(raw_results),
        study_groups,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(
        {
            "aggregate": aggregate,
            "individual_counts": individual_counts,
        },
        cache_path,
    )
    print(f"Cached plot data at {cache_path}")
    return aggregate, individual_counts


def write_outputs(
    aggregate: pd.DataFrame,
    individual_counts: pd.DataFrame,
    output_stem: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    x_index = {timescale: index for index, timescale in enumerate(TIMESCALE_ORDER)}
    count_map = {
        (row.split_key, row.panel): int(row.n_individuals)
        for row in individual_counts.itertuples(index=False)
    }
    fig, axes = plt.subplots(
        len(SPLIT_ORDER),
        len(PANEL_ORDER),
        figsize=(17.5, 7),
        sharex=True,
        squeeze=False,
    )

    for row_index, split_key in enumerate(SPLIT_ORDER):
        for column_index, panel in enumerate(PANEL_ORDER):
            ax = axes[row_index, column_index]
            ax.axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="don't move",
            )
            subset = aggregate[
                (aggregate["split_key"] == split_key)
                & (aggregate["panel"] == panel)
            ]
            for method in PLOT_METHODS:
                method_rows = subset[subset["method"] == method].copy()
                if method_rows.empty:
                    continue
                method_rows["x"] = method_rows["timescale"].map(x_index)
                method_rows = method_rows.sort_values("x")
                x = method_rows["x"].to_numpy(dtype=float)
                mean = method_rows["normalized_energy_score_mean"].to_numpy(
                    dtype=float
                )
                sem = method_rows["normalized_energy_score_sem"].to_numpy(
                    dtype=float
                )
                ax.plot(
                    x,
                    mean,
                    marker=None if "baseline" in method else "o",
                    linewidth=2.0,
                    color=METHOD_COLORS[method],
                    label=_method_label(method),
                    linestyle="--" if "baseline" in method else None,
                )
                ax.fill_between(
                    x,
                    mean - sem,
                    mean + sem,
                    color=METHOD_COLORS[method],
                    alpha=0.1,
                )

            ax.set_xticks(range(len(TIMESCALE_ORDER)), TIMESCALE_ORDER)
            # if split_key == "test_temporal":
            #     ax.set_ylim(0.65, 1.1)
            # else:
            #     ax.set_ylim(0.7, 2)

            ax.set_ylim(0.65, 1.2)

            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator([0.7, 1, 2, 4]))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda value, _: f"{value:g}")
            )
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.grid(True, linewidth=0.4, alpha=0.4)

            if row_index == 0:
                ax.set_title(PANEL_TITLES[panel])
            if column_index != 0:
                ax.set_yticks([])
            ax.text(
                0.03,
                0.95,
                f"n={count_map.get((split_key, panel), 0)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            if column_index == 0:
                ax.set_ylabel(f"Norm ES ↓\n{SPLIT_LABELS[split_key]}")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(PLOT_METHODS) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
    )
    fig.suptitle(
        r"$\bf{Method~comparison:}$ Energy score (↓) on MoveBench", y=0.95
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.canvas.draw()

    for average_ax in axes[:, -1]:
        average_ax.set_facecolor("white")
    average_bboxes = [
        ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted()
        )
        for ax in axes[:, -1]
    ]
    pad = 0.01
    background_left = max(0.0, min(box.x0 for box in average_bboxes) - pad)
    background_bottom = max(0.0, min(box.y0 for box in average_bboxes) - pad)
    background_right = min(1.0, max(box.x1 for box in average_bboxes) + pad)
    background_top = min(1.0, max(box.y1 for box in average_bboxes) + pad)
    average_background_ax = fig.add_axes(
        [
            background_left,
            background_bottom,
            background_right - background_left,
            background_top - background_bottom,
        ],
        zorder=0,
    )
    average_background_ax.set_facecolor("#f2f2f2")
    average_background_ax.set_xticks([])
    average_background_ax.set_yticks([])
    for spine in average_background_ax.spines.values():
        spine.set_visible(False)
    average_background_ax.set_in_layout(False)
    for row_axes in axes:
        for ax in row_axes:
            ax.set_zorder(1)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    aggregate.to_csv(output_stem.with_suffix(".csv"), index=False)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot study-normalized energy scores by species and movement group."
    )
    parser.add_argument("--studies-csv", type=Path, default=DEFAULT_STUDIES_CSV)
    parser.add_argument("--traits-csv", type=Path, default=DEFAULT_TRAITS_CSV)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute the cached plot data from the raw results.",
    )
    return parser.parse_args()


def main() -> int:
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    args = parse_args()
    aggregate, individual_counts = load_or_build_plot_data(
        args.studies_csv,
        args.traits_csv,
        args.output_stem.with_suffix(".cache.pkl"),
        args.refresh_cache,
    )
    write_outputs(aggregate, individual_counts, args.output_stem)
    print(args.output_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
