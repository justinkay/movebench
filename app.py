#!/usr/bin/env python
# coding: utf-8

"""
Streamlit Trajectory Visualization and Review App

Features
--------
- Finds all CSVs in addtl_data/*.csv
- Tracks per-study review status in:
    - addtl_data/ok_files.txt
    - addtl_data/not_ok_files.txt
- Tracks per-individual exclusions in:
    - addtl_data/bad_individuals.txt
      (lines of the form: "<csv_basename>,<individual-id>")
- Tracks per-individual train/test split overrides in:
    - addtl_data/split_overrides.txt
      (lines of the form: "<csv_basename>,<individual-id>,YYYY-MM-DD")
- For each unreviewed CSV:
    - Saves ONE original PNG (no bad individuals removed, default mid-time split)
      in pngs/ named "<study_name>_summary_plots.png" at dpi=150, never overwritten.
    - Applies trajectory processing & visualization for the *current* view,
      excluding any individuals listed as bad and using any stored split overrides.
    - If there are >20 individuals, creates multiple map rows, each showing
      at most 20 individuals (train/test maps per row).
    - Shows metadata, per-individual counts, and plots.
    - Lets you:
        - Mark individual(s) as bad (removes them from plots on rerun)
        - Override train/test split date per individual (optional)
        - Mark study as "Okay" or "Not okay"
- Only shows files not already in ok_files/not_ok_files
"""

import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import streamlit as st


# -------------------------------------------------------------------
# Configuration: directories and list files
# -------------------------------------------------------------------

OUTPUT_DIR = "addtl_data"
PNG_DIR = "pngs"
OK_LIST_PATH = os.path.join(OUTPUT_DIR, "ok_files.txt")
NOT_OK_LIST_PATH = os.path.join(OUTPUT_DIR, "not_ok_files.txt")
BAD_INDIVIDUALS_PATH = os.path.join(OUTPUT_DIR, "bad_individuals.txt")
SPLIT_OVERRIDES_PATH = os.path.join(OUTPUT_DIR, "split_overrides.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PNG_DIR, exist_ok=True)


# -------------------------------------------------------------------
# Streamlit rerun helper (compatible with older/newer versions)
# -------------------------------------------------------------------


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# -------------------------------------------------------------------
# Helper functions: review list I/O
# -------------------------------------------------------------------


def _read_list(path: str) -> set:
    """Read a newline-separated list of filenames (basenames)."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return set(lines)


def _append_to_list(path: str, filename: str) -> None:
    """Append a single filename (basename) to the given list file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def get_review_status():
    ok = _read_list(OK_LIST_PATH)
    not_ok = _read_list(NOT_OK_LIST_PATH)
    return ok, not_ok


def get_remaining_files():
    """Return (remaining_files, ok_set, not_ok_set)."""
    all_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.csv")))
    ok_set, not_ok_set = get_review_status()
    reviewed_basenames = ok_set | not_ok_set

    remaining = [f for f in all_files if os.path.basename(f) not in reviewed_basenames]
    return remaining, ok_set, not_ok_set


# -------------------------------------------------------------------
# Helper functions: bad individuals I/O
# -------------------------------------------------------------------


def _read_bad_individuals() -> set:
    """
    Read bad individuals from BAD_INDIVIDUALS_PATH.

    Each line: <csv_basename>,<individual_local_identifier>
    Returns a set of (filename, individual_id) tuples.
    """
    if not os.path.exists(BAD_INDIVIDUALS_PATH):
        return set()

    bad = set()
    with open(BAD_INDIVIDUALS_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",", 1)
            if len(parts) == 2:
                fname, indiv = parts
                bad.add((fname, indiv))
    return bad


def _append_bad_individual(filename_basename: str, individual_id: str) -> None:
    """Append one bad individual entry to BAD_INDIVIDUALS_PATH."""
    with open(BAD_INDIVIDUALS_PATH, "a", encoding="utf-8") as f:
        f.write(f"{filename_basename},{individual_id}\n")


# -------------------------------------------------------------------
# Helper functions: split overrides I/O
# -------------------------------------------------------------------


def _read_split_overrides() -> dict:
    """
    Read split overrides from SPLIT_OVERRIDES_PATH.

    Each line: <csv_basename>,<individual_local_identifier>,YYYY-MM-DD
    Returns a dict mapping (filename, individual_id) -> pd.Timestamp
    (date at midnight).
    """
    overrides: dict[tuple[str, str], pd.Timestamp] = {}
    if not os.path.exists(SPLIT_OVERRIDES_PATH):
        return overrides

    with open(SPLIT_OVERRIDES_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split(",", 2)
            if len(parts) != 3:
                continue
            fname, indiv, date_str = parts
            try:
                ts = pd.to_datetime(date_str)
            except Exception:
                continue
            overrides[(fname, indiv)] = ts
    return overrides


def _append_split_override(
    filename_basename: str, individual_id: str, split_date
) -> None:
    """
    Append one split override entry to SPLIT_OVERRIDES_PATH.

    split_date is expected to be a datetime.date (from st.date_input).
    """
    with open(SPLIT_OVERRIDES_PATH, "a", encoding="utf-8") as f:
        f.write(f"{filename_basename},{individual_id},{split_date.isoformat()}\n")


# -------------------------------------------------------------------
# Core logic: processing & plotting from a DataFrame
# -------------------------------------------------------------------


def build_figure_from_df(
    df: pd.DataFrame,
    basename: str,
    study_name: str,
    save_path=None,
    split_times=None,
):
    """
    Given a DataFrame for a single CSV, run the full processing pipeline and return:
        fig, metadata_dict, individual_counts (Series)

    If save_path is provided, saves the figure as a PNG there (dpi=150).

    split_times: optional dict mapping individual_id -> pd.Timestamp
                 for train/test split. If None, use mid-time per individual.
    """
    if "timestamp" not in df.columns:
        raise RuntimeError("Expected a 'timestamp' column in the data.")

    df = df.copy()
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="mixed",  # let pandas infer per element
        errors="coerce",  # bad parses become NaT instead of raising
    )
    df = df[df["timestamp"].notna()].copy()
    df["individual_local_identifier"] = df["individual_local_identifier"].astype(str)

    duration = df["timestamp"].max() - df["timestamp"].min()
    num_individuals = df["individual_local_identifier"].nunique()
    num_points = len(df)

    if "individual_taxon_canonical_name" in df.columns:
        species = df["individual_taxon_canonical_name"].dropna().unique()
        species_str = ", ".join(map(str, species))
    else:
        species_str = "N/A (column 'individual_taxon_canonical_name' missing)"

    metadata = {
        "Study ID": basename.split("_")[0],
        "Filename": basename,
        "Species": species_str,
        "Number of individuals": f"{num_individuals:,}",
        "Number of data points": f"{num_points:,}",
        "Average points per individual": f"{num_points / num_individuals:.1f}"
        if num_individuals > 0
        else "N/A",
        "Time range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        "Duration (days)": duration.days,
    }

    # For the UI: counts
    individual_counts = df["individual_local_identifier"].value_counts().sort_index()

    # -------------------------------------------------------------------
    # 1) Compute and filter by time gap (per-individual quantile)
    # -------------------------------------------------------------------

    df = df.sort_values(["individual_local_identifier", "timestamp"])
    df["time_since_last"] = df.groupby("individual_local_identifier")[
        "timestamp"
    ].diff()

    gap_q = 0.999  # very high quantile for time gap per individual
    per_indiv_cutoff = df.groupby("individual_local_identifier")[
        "time_since_last"
    ].transform(lambda s: s.dropna().quantile(gap_q))

    df_filtered = df[
        df["time_since_last"].notna() & (df["time_since_last"] <= per_indiv_cutoff)
    ].copy()

    # -------------------------------------------------------------------
    # 2) Compute and filter by step length (Haversine)
    # -------------------------------------------------------------------

    zeros = df_filtered["location_lat"].values < 1.0
    if zeros.any():
        print(
            "Possible 0 latitude problem with",
            df_filtered["individual_local_identifier"].values[zeros],
        )

    lat_rad = np.radians(df_filtered["location_lat"])
    lon_rad = np.radians(df_filtered["location_long"])

    lat_prev = lat_rad.groupby(df_filtered["individual_local_identifier"]).shift()
    lon_prev = lon_rad.groupby(df_filtered["individual_local_identifier"]).shift()

    dlat = lat_rad - lat_prev
    dlon = lon_rad - lon_prev

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # meters
    df_filtered["step_length"] = R * c

    step_q = 0.95
    step_multiplier = 2

    q_vals = df_filtered.groupby("individual_local_identifier")[
        "step_length"
    ].transform(lambda s: s.quantile(step_q))

    df_filtered2 = df_filtered[
        df_filtered["step_length"] <= step_multiplier * q_vals
    ].copy()

    # -------------------------------------------------------------------
    # 3) Turn angles
    # -------------------------------------------------------------------

    df_filtered3 = df_filtered2.sort_values(
        ["individual_local_identifier", "timestamp"]
    ).copy()

    lat_rad2 = np.radians(df_filtered3["location_lat"])
    lon_rad2 = np.radians(df_filtered3["location_long"])

    lat_prev2 = lat_rad2.groupby(df_filtered3["individual_local_identifier"]).shift()
    lon_prev2 = lon_rad2.groupby(df_filtered3["individual_local_identifier"]).shift()

    dlon2 = lon_rad2 - lon_prev2
    y = np.sin(dlon2) * np.cos(lat_rad2)
    x = np.cos(lat_prev2) * np.sin(lat_rad2) - np.sin(lat_prev2) * np.cos(
        lat_rad2
    ) * np.cos(dlon2)

    bearing = np.arctan2(y, x)
    bearing_prev = bearing.groupby(df_filtered3["individual_local_identifier"]).shift()
    raw_turn = bearing - bearing_prev

    df_filtered3["turn_angle"] = np.arctan2(np.sin(raw_turn), np.cos(raw_turn))

    df_filtered3 = df_filtered3[df_filtered3["turn_angle"].notna()].copy()

    # -------------------------------------------------------------------
    # Prepare common pieces for plots
    # -------------------------------------------------------------------

    step_lengths_filtered = df_filtered3["step_length"]
    turn_angles_filtered = df_filtered3["turn_angle"]

    # Timeline prep
    df_timeline = df_filtered2.copy()
    df_timeline = df_timeline.sort_values(["individual_local_identifier", "timestamp"])
    gap_threshold = pd.Timedelta(days=3)

    span_time = df_timeline.groupby("individual_local_identifier")["timestamp"].agg(
        start="min", end="max"
    )
    span_time["default_mid"] = (
        span_time["start"] + (span_time["end"] - span_time["start"]) / 2
    )

    # Tracks & maps prep
    df_tracks = df_filtered2.copy()
    df_tracks = df_tracks.sort_values(["individual_local_identifier", "timestamp"])

    span_tracks = df_tracks.groupby("individual_local_identifier")["timestamp"].agg(
        start="min", end="max"
    )
    span_tracks["default_mid"] = (
        span_tracks["start"] + (span_tracks["end"] - span_tracks["start"]) / 2
    )

    # -------------------------------------------------------------------
    # Apply split overrides
    # -------------------------------------------------------------------
    # Default: use mid-time
    split_time_series = span_time["default_mid"].copy()
    split_time_tracks = span_tracks["default_mid"].copy()

    if split_times is not None:
        for indiv, ts in split_times.items():
            if indiv not in span_time.index:
                continue
            # Clip to individual range (timeline)
            start = span_time.loc[indiv, "start"]
            end = span_time.loc[indiv, "end"]
            clipped = min(max(ts, start), end)
            split_time_series.loc[indiv] = clipped

            if indiv in span_tracks.index:
                start_t = span_tracks.loc[indiv, "start"]
                end_t = span_tracks.loc[indiv, "end"]
                clipped_t = min(max(ts, start_t), end_t)
                split_time_tracks.loc[indiv] = clipped_t

    span_time["split"] = split_time_series
    span_tracks["split"] = split_time_tracks

    individuals_timeline = span_time.index.tolist()

    df_tracks["split_time"] = df_tracks["individual_local_identifier"].map(
        span_tracks["split"]
    )
    df_tracks["set"] = np.where(
        df_tracks["timestamp"] <= df_tracks["split_time"], "train", "test"
    )

    lat_min = df_tracks["location_lat"].min()
    lat_max = df_tracks["location_lat"].max()
    lon_min = df_tracks["location_long"].min()
    lon_max = df_tracks["location_long"].max()

    lat_pad = (lat_max - lat_min) * 0.05 if lat_max > lat_min else 0.01
    lon_pad = (lon_max - lon_min) * 0.05 if lon_max > lon_min else 0.01

    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    individuals_tracks = df_tracks["individual_local_identifier"].unique()
    cmap = cm.get_cmap("tab20", len(individuals_tracks))
    color_map = {ind: cmap(i) for i, ind in enumerate(individuals_tracks)}

    proj = ccrs.PlateCarree()

    # -------------------------------------------------------------------
    # Build one tall figure with 3+ stacked regions
    #   Row 0: histograms (2 columns)
    #   Row 1: timeline (full width)
    #   Rows 2..: one row per map group (train/test per row)
    # -------------------------------------------------------------------

    # Group individuals for maps in chunks of at most 20
    indiv_list = list(individuals_tracks)
    if len(indiv_list) == 0:
        raise RuntimeError("No individuals left after filtering; nothing to plot.")

    group_size = 20
    map_groups = [
        indiv_list[i : i + group_size] for i in range(0, len(indiv_list), group_size)
    ]
    n_map_groups = len(map_groups)

    total_rows = 2 + n_map_groups  # 1 hist row, 1 timeline row, rest maps
    height_ratios = [1.0, 1.0] + [1.4] * n_map_groups

    fig = plt.figure(figsize=(16, 8 + 6 * n_map_groups))
    gs = fig.add_gridspec(total_rows, 2, height_ratios=height_ratios)

    # Top row: histograms
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Middle row: timeline
    ax_timeline = fig.add_subplot(gs[1, :])

    # -------------------------------------------------------------------
    # 1) Histograms (top row)
    # -------------------------------------------------------------------

    ax1.hist(
        step_lengths_filtered, bins=50, alpha=0.7, color="skyblue", edgecolor="black"
    )
    ax1.set_xlabel("Step Length (meters)")
    ax1.set_ylabel("Frequency (log)")
    ax1.set_title(f"Step Length Distribution (log y)\n{study_name}")
    ax1.grid(True, alpha=0.3)

    mean_step = np.mean(step_lengths_filtered)
    median_step = np.median(step_lengths_filtered)
    ax1.axvline(
        median_step, color="orange", linestyle="--", label=f"Median: {median_step:.1f}m"
    )
    ax1.axvline(
        q_vals.mean(),
        color="purple",
        linestyle=":",
        label=f"{step_q * 100:.0f}th percentile: {q_vals.mean():.1f}m",
    )
    ax1.legend()
    ax1.set_yscale("log")

    turn_angles_filtered_deg = np.degrees(turn_angles_filtered)
    ax2.hist(
        turn_angles_filtered_deg,
        bins=50,
        alpha=0.7,
        color="lightcoral",
        edgecolor="black",
    )
    ax2.set_xlabel("Turn Angle (degrees)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Turn Angle\n{study_name}")
    ax2.grid(True, alpha=0.3)

    mean_turn = np.mean(turn_angles_filtered_deg)
    ax2.axvline(mean_turn, color="red", linestyle="--", label=f"Mean: {mean_turn:.1f}°")
    ax2.legend()

    # -------------------------------------------------------------------
    # 2) Timeline (middle row)
    # -------------------------------------------------------------------

    train_color = "skyblue"
    test_color = "lightcoral"
    bar_height = 0.75

    for i, individual in enumerate(individuals_timeline):
        ind = df_timeline[df_timeline["individual_local_identifier"] == individual]
        ts = ind["timestamp"].sort_values()

        start = span_time.loc[individual, "start"]
        end = span_time.loc[individual, "end"]
        split_time = span_time.loc[individual, "split"]

        gaps = ts.diff() > gap_threshold
        segment_id = gaps.cumsum()

        for seg, seg_ts in ts.groupby(segment_id):
            seg_start = seg_ts.min()
            seg_end = seg_ts.max()
            count_pts = len(seg_ts)
            count_lbl = f"{count_pts:,}"

            if seg_end <= split_time:
                seg_color = train_color
                ax_timeline.barh(
                    i,
                    seg_end - seg_start,
                    left=seg_start,
                    height=bar_height,
                    color=seg_color,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax_timeline.text(
                    seg_start + (seg_end - seg_start) / 2,
                    i,
                    count_lbl,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

            elif seg_start >= split_time:
                seg_color = test_color
                ax_timeline.barh(
                    i,
                    seg_end - seg_start,
                    left=seg_start,
                    height=bar_height,
                    color=seg_color,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax_timeline.text(
                    seg_start + (seg_end - seg_start) / 2,
                    i,
                    count_lbl,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

            else:
                # Segment crosses split → split into two
                pts_before = len(seg_ts[seg_ts <= split_time])
                pts_after = count_pts - pts_before
                lbl_before = f"{pts_before:,}"
                lbl_after = f"{pts_after:,}"

                # First part (train)
                ax_timeline.barh(
                    i,
                    split_time - seg_start,
                    left=seg_start,
                    height=bar_height,
                    color=train_color,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax_timeline.text(
                    seg_start + (split_time - seg_start) / 2,
                    i,
                    lbl_before,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

                # Second part (test)
                ax_timeline.barh(
                    i,
                    seg_end - split_time,
                    left=split_time,
                    height=bar_height,
                    color=test_color,
                    alpha=0.7,
                    edgecolor="black",
                )
                ax_timeline.text(
                    split_time + (seg_end - split_time) / 2,
                    i,
                    lbl_after,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=11,
                    fontweight="bold",
                )

        # Color-coded marker at the split time for each individual
        ax_timeline.scatter(
            split_time,
            i,
            s=40,
            color=color_map.get(individual, "black"),
            edgecolor="black",
            zorder=5,
        )

    ax_timeline.set_yticks(range(len(individuals_timeline)))
    ax_timeline.set_yticklabels(individuals_timeline)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax_timeline.xaxis.set_major_locator(locator)
    ax_timeline.xaxis.set_major_formatter(formatter)

    ax_timeline.set_xlabel("Time")
    ax_timeline.set_ylabel("Individual")
    ax_timeline.set_title(f"Data Availability and Split\n{study_name}")
    ax_timeline.grid(True, axis="x", alpha=0.3)

    legend_patches_tl = [
        Patch(
            facecolor=train_color,
            edgecolor="black",
            alpha=0.7,
            label="Train (before split)",
        ),
        Patch(
            facecolor=test_color,
            edgecolor="black",
            alpha=0.7,
            label="Test (after split)",
        ),
    ]
    ax_timeline.legend(handles=legend_patches_tl, loc="lower right")

    # -------------------------------------------------------------------
    # 3) Maps (rows 2..)
    # -------------------------------------------------------------------

    def setup_map_axis(ax, title):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
        ax.set_title(f"{title}\n{study_name}", fontsize=12)

    def plot_tracks(ax, subset_df):
        for indiv, ind_df in subset_df.groupby("individual_local_identifier"):
            ind_df = ind_df.sort_values("timestamp")
            if len(ind_df) < 2:
                continue

            col = color_map[indiv]

            # Black outline
            ax.plot(
                ind_df["location_long"].values,
                ind_df["location_lat"].values,
                "-",
                linewidth=3,
                alpha=0.9,
                color="black",
                transform=proj,
            )

            # Colored track
            ax.plot(
                ind_df["location_long"].values,
                ind_df["location_lat"].values,
                "-",
                linewidth=2.5,
                alpha=0.9,
                color=col,
                transform=proj,
            )

            # Start point
            ax.scatter(
                ind_df["location_long"].iloc[0],
                ind_df["location_lat"].iloc[0],
                s=40,
                color=col,
                edgecolor="black",
                transform=proj,
                zorder=3,
            )

            # End point
            ax.scatter(
                ind_df["location_long"].iloc[-1],
                ind_df["location_lat"].iloc[-1],
                s=60,
                facecolor=col,
                edgecolor="black",
                transform=proj,
                zorder=3,
            )

    train_df = df_tracks[df_tracks["set"] == "train"]
    test_df = df_tracks[df_tracks["set"] == "test"]

    for group_idx, group_indivs in enumerate(map_groups):
        row = 2 + group_idx
        ax_train = fig.add_subplot(gs[row, 0], projection=proj)
        ax_test = fig.add_subplot(gs[row, 1], projection=proj)

        setup_map_axis(
            ax_train, f"Train Tracks (individuals {group_idx + 1}/{n_map_groups})"
        )
        setup_map_axis(
            ax_test, f"Test Tracks (individuals {group_idx + 1}/{n_map_groups})"
        )

        subset_train = train_df[
            train_df["individual_local_identifier"].isin(group_indivs)
        ]
        subset_test = test_df[test_df["individual_local_identifier"].isin(group_indivs)]

        plot_tracks(ax_train, subset_train)
        plot_tracks(ax_test, subset_test)

        # Legend only for this group's individuals (on train map)
        legend_elements = [
            Line2D([0], [0], color=color_map[ind], lw=2.5, label=str(ind))
            for ind in group_indivs
        ]
        ax_train.legend(handles=legend_elements, title="Individuals", loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save PNG if requested (original or alternative view)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, metadata, individual_counts


# -------------------------------------------------------------------
# Wrapper: create figure for current file (handles originals vs filtered)
# -------------------------------------------------------------------


def create_trajectory_figure(csv_path: str):
    """
    For a given CSV file:
    - If original PNG does not yet exist, create it from the FULL data
      (no bad-individual filtering, default mid-time split).
    - Then create & return a figure based on the data with bad individuals
      removed and any per-individual split overrides applied.
    """
    df_full = pd.read_csv(csv_path)
    basename = os.path.basename(csv_path)
    study_name = basename.replace(".csv", "")

    # 1) Ensure ORIGINAL PNG exists (full data, default mid split)
    png_name = os.path.join(PNG_DIR, f"{study_name}_summary_plots.png")
    if not os.path.exists(png_name):
        fig_orig, _, _ = build_figure_from_df(
            df_full, basename, study_name, save_path=png_name, split_times=None
        )
        plt.close(fig_orig)

    # 2) Apply bad-individual filtering for the current view
    bad_indivs = _read_bad_individuals()
    bad_here = [ind for (fname, ind) in bad_indivs if fname == basename]
    print(
        "Bad here?",
        bad_here,
        df_full[df_full["individual_local_identifier"].astype(str).isin(bad_here)],
    )
    if bad_here:
        df_filtered = df_full[
            ~df_full["individual_local_identifier"].astype(str).isin(bad_here)
        ].copy()
    else:
        df_filtered = df_full.copy()

    if df_filtered.empty:
        raise RuntimeError(
            "All individuals in this file are marked bad (or no data left); nothing to plot."
        )

    # 3) Apply split overrides for this file
    all_split_overrides = _read_split_overrides()
    overrides_here: dict[str, pd.Timestamp] = {
        indiv: ts
        for (fname, indiv), ts in all_split_overrides.items()
        if fname == basename
    }

    fig, metadata, individual_counts = build_figure_from_df(
        df_filtered,
        basename,
        study_name,
        save_path=None,
        split_times=overrides_here if overrides_here else None,
    )
    return fig, metadata, individual_counts


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------


def main():
    st.set_page_config(page_title="Trajectory Review", layout="wide")

    st.title("Trajectory Visualization and Review")

    remaining_files, ok_set, not_ok_set = get_remaining_files()

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    total_files = len(remaining_files) + len(ok_set) + len(not_ok_set)

    st.sidebar.header("Progress")
    st.sidebar.write(f"Total files: {total_files}")
    st.sidebar.write(f"Reviewed (okay): {len(ok_set)}")
    st.sidebar.write(f"Reviewed (not okay): {len(not_ok_set)}")
    st.sidebar.write(f"Remaining: {len(remaining_files)}")

    if len(remaining_files) == 0:
        st.success("All files have been reviewed.")
        return

    # Ensure index is in range
    if st.session_state.current_index >= len(remaining_files):
        st.session_state.current_index = 0

    current_file = remaining_files[st.session_state.current_index]
    current_basename = os.path.basename(current_file)

    st.subheader(
        f"Reviewing file {st.session_state.current_index + 1} of {len(remaining_files)}"
    )
    st.code(current_basename)

    # Build figure and metadata for current file
    try:
        fig, metadata, individual_counts = create_trajectory_figure(current_file)
    except Exception as e:
        st.error(f"Error processing file {current_basename}: {e}")
        return

    # Metadata
    with st.expander("Study metadata", expanded=True):
        for k, v in metadata.items():
            st.write(f"**{k}:** {v}")

    # Per-individual controls (bad flags + split overrides)
    with st.expander(
        "Individuals in this study (mark bad / override split)", expanded=False
    ):
        bad_all = _read_bad_individuals()
        bad_here = {ind for (fname, ind) in bad_all if fname == current_basename}

        split_all = _read_split_overrides()
        split_here: dict[str, pd.Timestamp] = {
            indiv: ts
            for (fname, indiv), ts in split_all.items()
            if fname == current_basename
        }

        if bad_here:
            st.write("Already marked bad in this file:")
            st.write(", ".join(sorted(bad_here)))

        if split_here:
            st.write("Existing train/test split overrides in this file:")
            st.write(
                ", ".join(
                    f"{ind} → {ts.date().isoformat()}" for ind, ts in split_here.items()
                )
            )

        st.write(
            "You can mark individuals as bad (excluded from this file), "
            "and optionally override their train/test split date."
        )

        # individual_counts is a Series: index = id, value = count
        for indiv, count in individual_counts.items():
            cols = st.columns([3, 2, 3, 2])
            with cols[0]:
                st.write(f"**{indiv}**")
            with cols[1]:
                st.write(f"{count} points")

            # Split date override
            with cols[2]:
                existing_ts = split_here.get(indiv)
                default_date = existing_ts.date() if existing_ts is not None else None
                split_date = st.date_input(
                    "Split date",
                    value=default_date,
                    key=f"split_{current_basename}_{indiv}",
                )

            # Buttons (save split / mark bad)
            with cols[3]:
                # Save split override
                if st.button(
                    "Save split", key=f"save_split_{current_basename}_{indiv}"
                ):
                    if split_date is not None:
                        _append_split_override(current_basename, str(indiv), split_date)
                    _rerun()

                # Mark bad
                already_bad = indiv in bad_here
                if already_bad:
                    st.button(
                        "Marked bad",
                        key=f"bad_{current_basename}_{indiv}",
                        disabled=True,
                    )
                else:
                    if st.button("Mark bad", key=f"bad_{current_basename}_{indiv}"):
                        _append_bad_individual(current_basename, str(indiv))
                        _rerun()

    # Show the figure
    st.pyplot(fig)
    plt.close(fig)

    # Buttons for study-level classification
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Okay", type="primary", use_container_width=True):
            _append_to_list(OK_LIST_PATH, current_basename)
            st.session_state.current_index = 0
            _rerun()

    with col2:
        if st.button("Not okay", use_container_width=True):
            _append_to_list(NOT_OK_LIST_PATH, current_basename)
            st.session_state.current_index = 0
            _rerun()

    # Optional: navigation in remaining set (without changing lists)
    with st.expander("Navigation", expanded=False):
        st.write(
            "These controls only change which *unreviewed* file is shown; they do not mark files."
        )
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("Previous unreviewed file"):
                st.session_state.current_index = (
                    st.session_state.current_index - 1
                ) % len(remaining_files)
                _rerun()
        with nav_col2:
            if st.button("Next unreviewed file"):
                st.session_state.current_index = (
                    st.session_state.current_index + 1
                ) % len(remaining_files)
                _rerun()


if __name__ == "__main__":
    main()
