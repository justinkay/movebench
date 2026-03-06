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
OUTLIER_COL = "manually-marked-outlier"


def _haversine_distance_m(
    start_lat_deg: np.ndarray,
    start_lon_deg: np.ndarray,
    end_lat_deg: np.ndarray,
    end_lon_deg: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in meters."""
    start_lat = np.deg2rad(np.asarray(start_lat_deg, dtype=np.float64))
    start_lon = np.deg2rad(np.asarray(start_lon_deg, dtype=np.float64))
    end_lat = np.deg2rad(np.asarray(end_lat_deg, dtype=np.float64))
    end_lon = np.deg2rad(np.asarray(end_lon_deg, dtype=np.float64))

    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    a = np.sin(dlat / 2.0) ** 2 + np.cos(start_lat) * np.cos(end_lat) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return 6371000.0 * c

def load_trajectories_csv(path: str) -> pd.DataFrame:
    """Load trajectories.csv with timestamp parsing and NaT rows dropped."""
    df = pd.read_csv(path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df.dropna(subset=[TIME_COL])


def prepare_interval_step_starts(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    max_time_error_s: float | None = None,
) -> pd.DataFrame:
    """Prepare burst-aware interval step starts for a requested step time.

    For each row, choose the later row in the same native burst whose timestamp
    is nearest to `timestamp + step_time_s`. Exact ties choose the later
    timestamp. A row is a valid interval step start only if a same-burst
    endpoint exists, the endpoint lands within the allowed time tolerance, and
    the resulting step length is finite and positive. The returned dataframe is
    restored to the cleaned input row order, and `step_end_index` refers to
    rows in that returned dataframe.
    """
    required = [INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    step_time_s = int(step_time_s)
    if step_time_s <= 0:
        raise ValueError(f"step_time_s must be > 0, got {step_time_s}")
    if not (0.0 <= float(gap_q) <= 1.0):
        raise ValueError(f"gap_q must be in [0,1], got {gap_q}")

    if max_time_error_s is None:
        max_time_error_s = float(step_time_s) / 2.0
    max_time_error_s = float(max_time_error_s)
    if max_time_error_s < 0:
        raise ValueError(f"max_time_error_s must be >= 0, got {max_time_error_s}")

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL]).reset_index(drop=True)
    d["_source_index"] = np.arange(len(d), dtype=np.int64)
    d = d.sort_values([INDIVIDUAL_COL, TIME_COL]).reset_index(drop=True)

    g = d.groupby(INDIVIDUAL_COL, sort=False)
    d["time_since_last"] = g[TIME_COL].diff()
    per_indiv_cutoff = g["time_since_last"].transform(lambda s: s.dropna().quantile(gap_q))
    d["within_time"] = d["time_since_last"] <= per_indiv_cutoff
    d["is_first_in_indiv"] = g.cumcount() == 0
    d["burst_start"] = d["is_first_in_indiv"] | (~d["within_time"])
    d["burst_id"] = g["burst_start"].cumsum()

    n_rows = len(d)
    source_index = d["_source_index"].to_numpy(dtype=np.int64, copy=False)
    time_values = d[TIME_COL].astype("int64").to_numpy(dtype=np.int64, copy=False)
    end_index = np.full(n_rows, -1, dtype=np.int64)
    actual_step_time_s = np.full(n_rows, np.nan, dtype=np.float64)
    step_time_error_s = np.full(n_rows, np.nan, dtype=np.float64)
    step_length_m = np.full(n_rows, np.nan, dtype=np.float64)
    valid_step_start = np.zeros(n_rows, dtype=bool)

    step_time_ns = int(step_time_s) * 1_000_000_000
    target_time_values = time_values + step_time_ns

    for _, burst in d.groupby([INDIVIDUAL_COL, "burst_id"], sort=False):
        positions = burst.index.to_numpy(dtype=np.int64, copy=False)
        if len(positions) < 2:
            continue

        burst_times = time_values[positions]
        burst_lat = d.loc[positions, LAT_COL].to_numpy(dtype=np.float64, copy=False)
        burst_lon = d.loc[positions, LON_COL].to_numpy(dtype=np.float64, copy=False)
        rel = np.arange(len(positions), dtype=np.int64)
        burst_targets = target_time_values[positions]
        insert = np.searchsorted(burst_times, burst_targets, side="left")
        cand_right = insert
        cand_left = insert - 1

        right_valid = (cand_right < len(positions)) & (cand_right > rel)
        left_valid = (cand_left >= 0) & (cand_left > rel)

        choose_right = right_valid & ~left_valid
        both_valid = right_valid & left_valid
        if both_valid.any():
            right_error = np.abs(burst_times[cand_right[both_valid]] - burst_targets[both_valid])
            left_error = np.abs(burst_times[cand_left[both_valid]] - burst_targets[both_valid])
            right_is_better = (right_error < left_error) | (
                (right_error == left_error) & (burst_times[cand_right[both_valid]] > burst_times[cand_left[both_valid]])
            )
            choose_right[both_valid] = right_is_better

        chosen_rel = np.full(len(positions), -1, dtype=np.int64)
        chosen_rel[choose_right] = cand_right[choose_right]
        choose_left = left_valid & (chosen_rel < 0)
        chosen_rel[choose_left] = cand_left[choose_left]

        selected = chosen_rel >= 0
        if not selected.any():
            continue

        start_pos = positions[selected]
        end_pos = positions[chosen_rel[selected]]
        start_times = burst_times[selected]
        end_times = burst_times[chosen_rel[selected]]
        actual_s = (end_times - start_times) / 1_000_000_000.0
        error_s = actual_s - float(step_time_s)
        lengths = _haversine_distance_m(
            burst_lat[selected],
            burst_lon[selected],
            burst_lat[chosen_rel[selected]],
            burst_lon[chosen_rel[selected]],
        )

        end_index[start_pos] = source_index[end_pos]
        actual_step_time_s[start_pos] = actual_s
        step_time_error_s[start_pos] = error_s
        step_length_m[start_pos] = lengths
        valid_step_start[start_pos] = (
            np.isfinite(lengths)
            & (lengths > 0.0)
            & np.isfinite(actual_s)
            & (np.abs(error_s) <= max_time_error_s)
        )

    d["step_target_time"] = d[TIME_COL] + pd.to_timedelta(step_time_s, unit="s")
    d["step_end_index"] = pd.array([None if i < 0 else int(i) for i in end_index], dtype="Int64")
    d["actual_step_time_s"] = actual_step_time_s
    d["step_time_error_s"] = step_time_error_s
    d["step_length_m"] = step_length_m
    d["valid_step_start"] = valid_step_start

    d = d.sort_values("_source_index").reset_index(drop=True)
    return d.drop(columns=["_source_index", "is_first_in_indiv", "burst_start"])


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
    lat_prev_deg = g[LAT_COL].shift()
    lon_prev_deg = g[LON_COL].shift()
    df["step_length_m"] = _haversine_distance_m(
        lat_prev_deg.to_numpy(dtype=np.float64, copy=False),
        lon_prev_deg.to_numpy(dtype=np.float64, copy=False),
        df[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        df[LON_COL].to_numpy(dtype=np.float64, copy=False),
    )

    df["lat_prev"] = g["lat_rad"].shift()
    df["lon_prev"] = g["lon_rad"].shift()
    dlat = df["lat_rad"] - df["lat_prev"]
    dlon = df["lon_rad"] - df["lon_prev"]

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
