#!/usr/bin/env python
# coding: utf-8

"""
Streamlit Trajectory Visualization and Review App

- Finds all CSVs in mbr_studies_2year/*_2year.csv
- Skips any file whose basename is already listed in ok_files.txt or not_ok_files.txt
- For the current CSV:
    - Runs your original trajectory processing + visualization (single-page figure)
    - Shows the figure in the browser
- Lets you click "Okay" or "Not okay":
    - Appends the CSV basename to the corresponding text file
    - Moves on to the next unreviewed file
"""

import os
import glob
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import streamlit as st


# -------------------------------------------------------------------
# Configuration: directories and list files
# -------------------------------------------------------------------

OUTPUT_DIR = "mbr_studies_2year"
PDF_DIR = "pdfs"
OK_LIST_PATH = os.path.join(OUTPUT_DIR, "ok_files.txt")
NOT_OK_LIST_PATH = os.path.join(OUTPUT_DIR, "not_ok_files.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)


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
    all_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_2year.csv")))
    ok_set, not_ok_set = get_review_status()
    reviewed_basenames = ok_set | not_ok_set

    remaining = [f for f in all_files
                 if os.path.basename(f) not in reviewed_basenames]
    return remaining, ok_set, not_ok_set


# -------------------------------------------------------------------
# Core logic: build figure for a single file
# (mostly your original script, wrapped in a function)
# -------------------------------------------------------------------

def create_trajectory_figure(csv_path: str, save_pdf: bool = True):
    """
    Given a CSV path, run the full processing pipeline and return:
        fig, metadata_dict

    Optionally also save a PDF in PDF_DIR.
    """
    df = pd.read_csv(csv_path)

    # Study metadata
    study_id = os.path.basename(csv_path).split('_')[0]
    study_name = os.path.basename(csv_path).replace("_2year.csv", "")

    if 'timestamp' not in df.columns:
        raise RuntimeError("Expected a 'timestamp' column in the data.")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    duration = df['timestamp'].max() - df['timestamp'].min()
    num_individuals = df['individual-local-identifier'].nunique()
    num_points = len(df)

    if 'individual-taxon-canonical-name' in df.columns:
        species = df['individual-taxon-canonical-name'].dropna().unique()
        species_str = ", ".join(map(str, species))
    else:
        species_str = "N/A (column 'individual-taxon-canonical-name' missing)"

    metadata = {
        "Study ID": study_id,
        "Filename": os.path.basename(csv_path),
        "Species": species_str,
        "Number of individuals": f"{num_individuals:,}",
        "Number of data points": f"{num_points:,}",
        "Average points per individual": f"{num_points / num_individuals:.1f}" if num_individuals > 0 else "N/A",
        "Time range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        "Duration (days)": duration.days,
    }

    # 1) Compute and filter by time gap (per-individual quantile)
    df = df.sort_values(['individual-local-identifier', 'timestamp'])
    df['time_since_last'] = df.groupby('individual-local-identifier')['timestamp'].diff()

    gap_q = 0.999  # very high quantile for time gap per individual
    per_indiv_cutoff = (
        df.groupby('individual-local-identifier')['time_since_last']
        .transform(lambda s: s.dropna().quantile(gap_q))
    )

    df_filtered = df[
        df['time_since_last'].notna() &
        (df['time_since_last'] <= per_indiv_cutoff)
    ].copy()

    # 2) Step lengths (Haversine) and filtering
    lat_rad = np.radians(df_filtered['location-lat'])
    lon_rad = np.radians(df_filtered['location-long'])

    lat_prev = lat_rad.groupby(df_filtered['individual-local-identifier']).shift()
    lon_prev = lon_rad.groupby(df_filtered['individual-local-identifier']).shift()

    dlat = lat_rad - lat_prev
    dlon = lon_rad - lon_prev

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_prev) * np.cos(lat_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    R = 6371000  # meters
    df_filtered['step_length'] = R * c

    step_q = 0.95
    step_multiplier = 2

    q_vals = df_filtered.groupby('individual-local-identifier')['step_length'] \
        .transform(lambda s: s.quantile(step_q))

    df_filtered2 = df_filtered[df_filtered['step_length'] <= step_multiplier * q_vals].copy()

    # 3) Turn angles
    df_filtered3 = df_filtered2.sort_values(
        ['individual-local-identifier', 'timestamp']
    ).copy()

    lat_rad2 = np.radians(df_filtered3['location-lat'])
    lon_rad2 = np.radians(df_filtered3['location-long'])

    lat_prev2 = lat_rad2.groupby(df_filtered3['individual-local-identifier']).shift()
    lon_prev2 = lon_rad2.groupby(df_filtered3['individual-local-identifier']).shift()

    dlon2 = lon_rad2 - lon_prev2
    y = np.sin(dlon2) * np.cos(lat_rad2)
    x = (np.cos(lat_prev2) * np.sin(lat_rad2) -
         np.sin(lat_prev2) * np.cos(lat_rad2) * np.cos(dlon2))

    bearing = np.arctan2(y, x)
    bearing_prev = bearing.groupby(df_filtered3['individual-local-identifier']).shift()
    raw_turn = bearing - bearing_prev

    df_filtered3['turn_angle'] = np.arctan2(
        np.sin(raw_turn),
        np.cos(raw_turn)
    )

    df_filtered3 = df_filtered3[df_filtered3['turn_angle'].notna()].copy()

    # Prepare common pieces for plots
    step_lengths_filtered = df_filtered3['step_length']
    turn_angles_filtered = df_filtered3['turn_angle']

    # Timeline prep
    df_timeline = df_filtered2.copy()
    df_timeline = df_timeline.sort_values(['individual-local-identifier', 'timestamp'])
    gap_threshold = pd.Timedelta(days=3)

    span_time = (
        df_timeline.groupby('individual-local-identifier')['timestamp']
        .agg(start='min', end='max')
    )
    span_time['mid'] = span_time['start'] + (span_time['end'] - span_time['start']) / 2
    individuals_timeline = span_time.index.tolist()

    # Tracks & maps prep
    df_tracks = df_filtered2.copy()
    df_tracks = df_tracks.sort_values(['individual-local-identifier', 'timestamp'])

    span_tracks = (
        df_tracks.groupby('individual-local-identifier')['timestamp']
        .agg(start='min', end='max')
    )
    span_tracks['mid'] = span_tracks['start'] + (span_tracks['end'] - span_tracks['start']) / 2

    df_tracks['mid_time'] = df_tracks['individual-local-identifier'].map(span_tracks['mid'])
    df_tracks['set'] = np.where(
        df_tracks['timestamp'] <= df_tracks['mid_time'],
        'train',
        'test'
    )

    lat_min = df_tracks['location-lat'].min()
    lat_max = df_tracks['location-lat'].max()
    lon_min = df_tracks['location-long'].min()
    lon_max = df_tracks['location-long'].max()

    lat_pad = (lat_max - lat_min) * 0.05 if lat_max > lat_min else 0.01
    lon_pad = (lon_max - lon_min) * 0.05 if lon_max > lon_min else 0.01

    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad

    individuals_tracks = df_tracks['individual-local-identifier'].unique()
    cmap = cm.get_cmap('tab20', len(individuals_tracks))
    color_map = {ind: cmap(i) for i, ind in enumerate(individuals_tracks)}

    proj = ccrs.PlateCarree()

    # Build figure with 3 stacked regions
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(
        3, 2,  # 3 rows, 2 columns
        height_ratios=[1.0, 1.0, 1.4]
    )

    # Axes
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_timeline = fig.add_subplot(gs[1, :])
    ax_train = fig.add_subplot(gs[2, 0], projection=proj)
    ax_test = fig.add_subplot(gs[2, 1], projection=proj)

    # 1) Histograms
    ax1.hist(step_lengths_filtered, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Step Length (meters)')
    ax1.set_ylabel('Frequency (log)')
    ax1.set_title(f'Step Length Distribution (log y)\n{study_name}')
    ax1.grid(True, alpha=0.3)

    mean_step = np.mean(step_lengths_filtered)
    median_step = np.median(step_lengths_filtered)
    ax1.axvline(median_step, color='orange', linestyle='--', label=f'Median: {median_step:.1f}m')
    ax1.axvline(q_vals.mean(), color='purple', linestyle=':', label=f'{step_q * 100:.0f}th percentile: {q_vals.mean():.1f}m')
    ax1.legend()
    ax1.set_yscale('log')

    turn_angles_filtered_deg = np.degrees(turn_angles_filtered)
    ax2.hist(turn_angles_filtered_deg, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Turn Angle (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Turn Angle\n{study_name}')
    ax2.grid(True, alpha=0.3)

    mean_turn = np.mean(turn_angles_filtered_deg)
    ax2.axvline(mean_turn, color='red', linestyle='--', label=f'Mean: {mean_turn:.1f}°')
    ax2.legend()

    # 2) Timeline
    train_color = 'skyblue'
    test_color = 'lightcoral'
    bar_height = 0.75

    for i, individual in enumerate(individuals_timeline):
        ind = df_timeline[df_timeline['individual-local-identifier'] == individual]
        ts = ind['timestamp'].sort_values()

        start = span_time.loc[individual, 'start']
        end = span_time.loc[individual, 'end']
        mid = span_time.loc[individual, 'mid']

        gaps = ts.diff() > gap_threshold
        segment_id = gaps.cumsum()

        for seg, seg_ts in ts.groupby(segment_id):
            seg_start = seg_ts.min()
            seg_end = seg_ts.max()
            count_pts = len(seg_ts)
            count_lbl = f"{count_pts:,}"

            if seg_end <= mid:
                seg_color = train_color
                ax_timeline.barh(
                    i, seg_end - seg_start, left=seg_start,
                    height=bar_height, color=seg_color,
                    alpha=0.7, edgecolor='black'
                )
                ax_timeline.text(
                    seg_start + (seg_end - seg_start) / 2,
                    i,
                    count_lbl,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold'
                )

            elif seg_start >= mid:
                seg_color = test_color
                ax_timeline.barh(
                    i, seg_end - seg_start, left=seg_start,
                    height=bar_height, color=seg_color,
                    alpha=0.7, edgecolor='black'
                )
                ax_timeline.text(
                    seg_start + (seg_end - seg_start) / 2,
                    i,
                    count_lbl,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold'
                )

            else:
                # Segment crosses midpoint → split
                pts_before_mid = len(seg_ts[seg_ts <= mid])
                pts_after_mid = count_pts - pts_before_mid
                lbl_before = f"{pts_before_mid:,}"
                lbl_after = f"{pts_after_mid:,}"

                # First half (train)
                ax_timeline.barh(
                    i, mid - seg_start, left=seg_start,
                    height=bar_height, color=train_color,
                    alpha=0.7, edgecolor='black'
                )
                ax_timeline.text(
                    seg_start + (mid - seg_start) / 2,
                    i,
                    lbl_before,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold'
                )

                # Second half (test)
                ax_timeline.barh(
                    i, seg_end - mid, left=mid,
                    height=bar_height, color=test_color,
                    alpha=0.7, edgecolor='black'
                )
                ax_timeline.text(
                    mid + (seg_end - mid) / 2,
                    i,
                    lbl_after,
                    ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold'
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
    ax_timeline.grid(True, axis='x', alpha=0.3)

    legend_patches_tl = [
        Patch(facecolor=train_color, edgecolor='black', alpha=0.7, label='Train (first half)'),
        Patch(facecolor=test_color, edgecolor='black', alpha=0.7, label='Test (second half)')
    ]
    ax_timeline.legend(handles=legend_patches_tl, loc='lower right')

    # 3) Maps
    def setup_map_axis(ax, title):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
        ax.set_title(f"{title}\n{study_name}", fontsize=12)

    def plot_tracks(ax, subset_df):
        for indiv, ind_df in subset_df.groupby('individual-local-identifier'):
            ind_df = ind_df.sort_values('timestamp')
            if len(ind_df) < 2:
                continue

            col = color_map[indiv]

            # Black outline
            ax.plot(
                ind_df['location-long'].values,
                ind_df['location-lat'].values,
                '-', linewidth=3, alpha=0.9,
                color='black', transform=proj
            )

            # Colored track
            ax.plot(
                ind_df['location-long'].values,
                ind_df['location-lat'].values,
                '-', linewidth=2.5, alpha=0.9,
                color=col, transform=proj
            )

            # Start point
            ax.scatter(
                ind_df['location-long'].iloc[0],
                ind_df['location-lat'].iloc[0],
                s=40, color=col, edgecolor='black',
                transform=proj, zorder=3
            )

            # End point
            ax.scatter(
                ind_df['location-long'].iloc[-1],
                ind_df['location-lat'].iloc[-1],
                s=60, facecolor=col, edgecolor='black',
                transform=proj, zorder=3
            )

    train_df = df_tracks[df_tracks['set'] == 'train']
    test_df = df_tracks[df_tracks['set'] == 'test']

    setup_map_axis(ax_train, "Train Tracks")
    plot_tracks(ax_train, train_df)

    setup_map_axis(ax_test, "Test Tracks")
    plot_tracks(ax_test, test_df)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save to single-page PDF (optional)
    if save_pdf:
        pdf_name = os.path.join(PDF_DIR, f"{study_name}_summary_plots.pdf")
        with PdfPages(pdf_name) as pdf:
            pdf.savefig(fig)

    return fig, metadata


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

    st.subheader(f"Reviewing file {st.session_state.current_index + 1} of {len(remaining_files)}")
    st.code(current_basename)

    # Build figure and metadata for current file
    try:
        fig, metadata = create_trajectory_figure(current_file, save_pdf=True)
    except Exception as e:
        st.error(f"Error processing file {current_basename}: {e}")
        return

    # Show metadata
    with st.expander("Study metadata", expanded=True):
        for k, v in metadata.items():
            st.write(f"**{k}:** {v}")

    # Show the figure
    st.pyplot(fig)
    plt.close(fig)

    # Buttons for classification
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Okay", type="primary", use_container_width=True):
            _append_to_list(OK_LIST_PATH, current_basename)
            # Reset index so we always start at first remaining file after filtering
            st.session_state.current_index = 0
            st.experimental_rerun()

    with col2:
        if st.button("Not okay", use_container_width=True):
            _append_to_list(NOT_OK_LIST_PATH, current_basename)
            st.session_state.current_index = 0
            st.experimental_rerun()

    # Optional: navigation in remaining set (without changing lists)
    with st.expander("Navigation", expanded=False):
        st.write("These controls only change which *unreviewed* file is shown; they do not mark files.")
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("Previous unreviewed file"):
                st.session_state.current_index = (st.session_state.current_index - 1) % len(remaining_files)
                st.experimental_rerun()
        with nav_col2:
            if st.button("Next unreviewed file"):
                st.session_state.current_index = (st.session_state.current_index + 1) % len(remaining_files)
                st.experimental_rerun()


if __name__ == "__main__":
    main()
