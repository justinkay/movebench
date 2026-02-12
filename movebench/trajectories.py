"""Trajectories CSV schema and movement feature helpers.

All studies in this codebase are assumed to provide a `trajectories.csv` file
with a consistent Movebank-style schema.

`compute_movement_features(...)` is the single entrypoint to compute commonly
used movement quantities (step lengths, bearings, turn angles) and to flag
which rows can serve as the start of a step for stepwise algorithms (SSF/iSSF).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Movebank / trajectories.csv schema
LON_COL = "location-long"
LAT_COL = "location-lat"
TIME_COL = "timestamp"
INDIVIDUAL_COL = "individual-local-identifier"
SET_COL = "set"
POINT_ID_COL = "point_id"


def load_trajectories_csv(path: str) -> pd.DataFrame:
    """Load trajectories.csv with timestamp parsing and NaT rows dropped."""
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df.dropna(subset=[TIME_COL])


def compute_movement_features(
    df: pd.DataFrame,
    *,
    step_multiplier: float = 2.0,
    step_q: float = 0.95,
    gap_q: float = 0.999,
) -> pd.DataFrame:
    """Compute movement metrics + step-start flags for stepwise algorithms.

    This ports legacy logic from `movebench.utils.setup_df_for_ssf` but uses the
    more general name `can_be_step_start` instead of `can_be_ssf_start`.
    """
    df = df.copy()

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL])

    # Sort so diffs make sense; keep original index (point ids).
    df = df.sort_values([INDIVIDUAL_COL, TIME_COL])

    df["lat_rad"] = np.deg2rad(df[LAT_COL])
    df["lon_rad"] = np.deg2rad(df[LON_COL])

    g = df.groupby(INDIVIDUAL_COL)

    # ---- time gaps per individual ----
    df["time_since_last"] = g[TIME_COL].diff()
    per_indiv_cutoff = g["time_since_last"].transform(lambda s: s.dropna().quantile(gap_q))
    df["within_time"] = df["time_since_last"] <= per_indiv_cutoff

    # ---- step length per individual ----
    R = 6371000.0  # Earth radius in meters
    df["lat_prev"] = g["lat_rad"].shift()
    df["lon_prev"] = g["lon_rad"].shift()
    dlat = df["lat_rad"] - df["lat_prev"]
    dlon = df["lon_rad"] - df["lon_prev"]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(df["lat_prev"]) * np.cos(df["lat_rad"]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    df["step_length_m"] = R * c

    # Initial valid step mask (time + positive distance).
    df["valid_step"] = df["within_time"] & df["step_length_m"].notna() & (df["step_length_m"] > 0)

    # ---- step-length cutoff ----
    max_step_len = float(step_multiplier) * float(np.nanquantile(df.loc[df["valid_step"], "step_length_m"], q=step_q))
    df["max_step_len"] = max_step_len
    df["valid_step"] &= df["step_length_m"] <= max_step_len

    # ---- turn angles ----
    y = np.sin(dlon) * np.cos(df["lat_rad"])
    x = np.cos(df["lat_prev"]) * np.sin(df["lat_rad"]) - np.sin(df["lat_prev"]) * np.cos(df["lat_rad"]) * np.cos(dlon)
    df["bearing"] = np.arctan2(y, x)  # [-pi, pi], NaN at first row per individual

    df["bearing_prev"] = g["bearing"].shift()
    df["valid_step_prev"] = g["valid_step"].shift()

    turn = df["bearing"] - df["bearing_prev"]
    turn = (turn + np.pi) % (2.0 * np.pi) - np.pi  # wrap

    df["turn_angle_rad"] = np.where(df["valid_step"] & df["valid_step_prev"], turn, np.nan)

    # ---- bursts ----
    df["is_first_in_indiv"] = g.cumcount() == 0
    df["burst_start"] = df["is_first_in_indiv"] | (~df["valid_step"])
    df["burst_id"] = g["burst_start"].cumsum()

    # ---- step sample starts ----
    df["next_index"] = df.index.to_series().shift(-1).astype("Int64")

    same_indiv_next = df[INDIVIDUAL_COL].eq(df[INDIVIDUAL_COL].shift(-1))
    same_burst_next = df["burst_id"].eq(df["burst_id"].shift(-1))
    valid_step_next = df["valid_step"].shift(-1).fillna(False).astype(bool)

    df["can_be_step_start"] = (~df["is_first_in_indiv"]) & same_indiv_next & same_burst_next & valid_step_next

    df = df.drop(
        columns=[
            "lat_rad",
            "lon_rad",
            "lat_prev",
            "lon_prev",
            "burst_start",
            "is_first_in_indiv",
        ]
    )

    return df


def select_step_starts(
    df: pd.DataFrame,
    *,
    step_multiplier: float = 2.0,
    step_q: float = 0.95,
    gap_q: float = 0.999,
) -> pd.DataFrame:
    """Select valid step starts and attach observed endpoints.

    Returns a dataframe with one row per valid step start containing:
      - point_id_start, timestamp,
      - start_lon/start_lat/start_bearing,
      - end_lon/end_lat for the observed next location.

    This is the shared step selection helper used by both:
      - point-store extraction (SSF/iSSF), and
      - raster chip datasets for next-step prediction.
    """
    required = [INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL, POINT_ID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    d = compute_movement_features(df, step_multiplier=step_multiplier, step_q=step_q, gap_q=gap_q)
    starts = d[d["can_be_step_start"]].copy()
    if len(starts) == 0:
        return pd.DataFrame(
            {
                "point_id_start": pd.Series(dtype="int64"),
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "start_lon": pd.Series(dtype="float64"),
                "start_lat": pd.Series(dtype="float64"),
                "start_bearing": pd.Series(dtype="float64"),
                "end_lon": pd.Series(dtype="float64"),
                "end_lat": pd.Series(dtype="float64"),
            }
        )

    next_idx = starts["next_index"]
    starts = starts[next_idx.notna()].copy()
    next_idx_int = next_idx[next_idx.notna()].astype(int).to_numpy()

    # Align endpoints to starts by reindexing in the same order.
    ends = d.loc[next_idx_int].copy()
    starts = starts.reset_index(drop=True)
    ends = ends.reset_index(drop=True)
    if len(starts) != len(ends):
        raise RuntimeError(f"Internal error: starts/ends length mismatch ({len(starts)} vs {len(ends)})")

    out = pd.DataFrame(
        {
            "point_id_start": starts[POINT_ID_COL].astype(int),
            "timestamp": starts[TIME_COL],
            "start_lon": starts[LON_COL].astype(float),
            "start_lat": starts[LAT_COL].astype(float),
            "start_bearing": starts["bearing"].astype(float),
            "end_lon": ends[LON_COL].astype(float),
            "end_lat": ends[LAT_COL].astype(float),
        }
    )

    finite = np.isfinite(
        out[["start_lon", "start_lat", "start_bearing", "end_lon", "end_lat"]].to_numpy(dtype=np.float64, copy=False)
    ).all(axis=1)
    return out.loc[finite].reset_index(drop=True)
