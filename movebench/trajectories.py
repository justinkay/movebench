"""Trajectories CSV schema and movement feature helpers.

All studies in this codebase are assumed to provide a `trajectories.csv` file
with a consistent Movebank-style schema.

Fixed-interval helpers select step starts and fit movement support from training
trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from movebench.geometry import geodesic_distance_m, local_xy_m_from_lonlat

# Movebank / trajectories.csv schema
LON_COL = "location-long"
LAT_COL = "location-lat"
TIME_COL = "timestamp"
INDIVIDUAL_COL = "individual-local-identifier"
SET_COL = "set"
POINT_ID_COL = "point_id"
OUTLIER_COL = "manually-marked-outlier"


def canonicalize_split_key(split_key: object) -> str:
    return str(split_key).strip().replace("-", "_")


@dataclass(frozen=True)
class IntervalGapCutoffs:
    per_individual: dict[object, pd.Timedelta]
    fallback: pd.Timedelta | None = None


@dataclass(frozen=True)
class IntervalStepSupport:
    gap_cutoffs: IntervalGapCutoffs
    max_step_length_m: float


def bearing_rad(
    start_lat_deg: np.ndarray,
    start_lon_deg: np.ndarray,
    end_lat_deg: np.ndarray,
    end_lon_deg: np.ndarray,
) -> np.ndarray:
    """Vectorized WGS84 forward bearing in radians, wrapped to [-pi, pi]."""
    x_m, y_m = local_xy_m_from_lonlat(
        start_lon_deg,
        start_lat_deg,
        end_lon_deg,
        end_lat_deg,
    )
    out = np.arctan2(x_m, y_m)
    return (out + np.pi) % (2.0 * np.pi) - np.pi


def arrival_bearing_rad(
    start_lat_deg: np.ndarray,
    start_lon_deg: np.ndarray,
    end_lat_deg: np.ndarray,
    end_lon_deg: np.ndarray,
) -> np.ndarray:
    """Bearing of travel on arrival, expressed at the endpoint."""
    x_m, y_m = local_xy_m_from_lonlat(
        end_lon_deg,
        end_lat_deg,
        start_lon_deg,
        start_lat_deg,
    )
    out = np.arctan2(-x_m, -y_m)
    return (out + np.pi) % (2.0 * np.pi) - np.pi


def compute_step_length_extent_m(
    step_lengths_m,
    *,
    step_q: float,
    step_multiplier: float,
    max_step_q: float = 0.99,
) -> float:
    """Compute the extent cutoff from step lengths."""
    valid = np.asarray(step_lengths_m, dtype=np.float64)
    base_extent_m = float(step_multiplier) * float(
        np.nanquantile(valid, q=float(step_q))
    )
    max_extent_m = float(np.nanquantile(valid, q=float(max_step_q)))
    return min(base_extent_m, max_extent_m)


def _is_manual_outlier(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.isin({"true", "1", "t", "yes", "y"})


def load_trajectories_csv(
    path: str | Path,
    *,
    drop_manual_outliers: bool = True,
) -> pd.DataFrame:
    """Load trajectories.csv with early outlier filtering and timestamp parsing."""
    df = pd.read_csv(path)
    if SET_COL in df.columns:
        split_mask = df[SET_COL].notna()
        if bool(split_mask.any()):
            df.loc[split_mask, SET_COL] = df.loc[split_mask, SET_COL].map(
                canonicalize_split_key
            )
    if bool(drop_manual_outliers) and OUTLIER_COL in df.columns:
        df = df.loc[~_is_manual_outlier(df[OUTLIER_COL])].copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    return df.dropna(subset=[TIME_COL])


def fit_interval_gap_cutoffs(
    df: pd.DataFrame,
    *,
    gap_q: float = 0.999,
) -> IntervalGapCutoffs:
    """Fit per-individual burst gap cutoffs from training data only."""
    required = [INDIVIDUAL_COL, TIME_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")
    if not (0.0 <= float(gap_q) <= 1.0):
        raise ValueError(f"gap_q must be in [0,1], got {gap_q}")

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL])
    if len(d) == 0:
        return IntervalGapCutoffs(per_individual={}, fallback=None)

    d = d.sort_values([INDIVIDUAL_COL, TIME_COL]).reset_index(drop=True)
    time_since_last = d.groupby(INDIVIDUAL_COL, sort=False)[TIME_COL].diff()

    per_individual: dict[object, pd.Timedelta] = {}
    for indiv, series in time_since_last.groupby(d[INDIVIDUAL_COL], sort=False):
        valid = series.dropna()
        if len(valid) == 0:
            continue
        cutoff = valid.quantile(float(gap_q))
        if pd.isna(cutoff):
            continue
        per_individual[indiv] = pd.Timedelta(cutoff)

    valid_all = time_since_last.dropna()
    fallback = None
    if len(valid_all) > 0:
        cutoff_all = valid_all.quantile(float(gap_q))
        if not pd.isna(cutoff_all):
            fallback = pd.Timedelta(cutoff_all)

    return IntervalGapCutoffs(per_individual=per_individual, fallback=fallback)


def prepare_interval_step_starts(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    gap_cutoffs: IntervalGapCutoffs | None = None,
    max_time_error_s: float | None = None,
    max_step_length_m: float | None = None,
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
    if max_step_length_m is not None:
        max_step_length_m = float(max_step_length_m)
        if not np.isfinite(max_step_length_m) or max_step_length_m <= 0.0:
            raise ValueError(
                f"max_step_length_m must be finite and > 0, got {max_step_length_m}"
            )

    d = df.copy()
    d[TIME_COL] = pd.to_datetime(d[TIME_COL], errors="coerce")
    d = d.dropna(subset=[TIME_COL]).reset_index(drop=True)
    d["_source_index"] = np.arange(len(d), dtype=np.int64)
    d = d.sort_values([INDIVIDUAL_COL, TIME_COL]).reset_index(drop=True)

    g = d.groupby(INDIVIDUAL_COL, sort=False)
    d["time_since_last"] = g[TIME_COL].diff()
    if gap_cutoffs is None:
        per_indiv_cutoff = g["time_since_last"].transform(
            lambda s: s.dropna().quantile(gap_q)
        )
    else:
        per_indiv_cutoff = pd.to_timedelta(
            d[INDIVIDUAL_COL].map(gap_cutoffs.per_individual), errors="coerce"
        )
        if gap_cutoffs.fallback is not None:
            per_indiv_cutoff = per_indiv_cutoff.fillna(gap_cutoffs.fallback)
    d["within_time"] = d["time_since_last"] <= per_indiv_cutoff
    d["is_first_in_indiv"] = g.cumcount() == 0
    d["burst_start"] = d["is_first_in_indiv"] | (~d["within_time"])
    d["burst_id"] = g["burst_start"].cumsum()

    n_rows = len(d)
    source_index = d["_source_index"].to_numpy(dtype=np.int64, copy=False)
    # Pandas may preserve lower-resolution datetime64 units (for example `us`)
    # after parsing; normalize to nanoseconds before raw integer arithmetic.
    time_values = (
        d[TIME_COL]
        .to_numpy(dtype="datetime64[ns]", copy=False)
        .astype(np.int64, copy=False)
    )
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
            right_error = np.abs(
                burst_times[cand_right[both_valid]] - burst_targets[both_valid]
            )
            left_error = np.abs(
                burst_times[cand_left[both_valid]] - burst_targets[both_valid]
            )
            right_is_better = (right_error < left_error) | (
                (right_error == left_error)
                & (
                    burst_times[cand_right[both_valid]]
                    > burst_times[cand_left[both_valid]]
                )
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
        lengths = geodesic_distance_m(
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

    if max_step_length_m is not None:
        valid_step_start &= np.isfinite(step_length_m) & (
            step_length_m <= float(max_step_length_m)
        )

    d["step_target_time"] = d[TIME_COL] + pd.to_timedelta(step_time_s, unit="s")
    d["step_end_index"] = pd.array(
        [None if i < 0 else int(i) for i in end_index], dtype="Int64"
    )
    d["actual_step_time_s"] = actual_step_time_s
    d["step_time_error_s"] = step_time_error_s
    d["step_length_m"] = step_length_m
    d["valid_step_start"] = valid_step_start

    d = d.sort_values("_source_index").reset_index(drop=True)
    return d.drop(columns=["_source_index", "is_first_in_indiv", "burst_start"])


def fit_interval_step_length_cutoff_m(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    gap_cutoffs: IntervalGapCutoffs | None = None,
    step_q: float = 0.95,
    max_step_q: float = 0.99,
    step_multiplier: float = 2.0,
) -> float:
    """Fit the interval-step spatial cutoff from valid interval steps."""
    d = prepare_interval_step_starts(
        df,
        step_time_s=int(step_time_s),
        gap_q=float(gap_q),
        gap_cutoffs=gap_cutoffs,
    )
    valid = d.loc[d["valid_step_start"], "step_length_m"].to_numpy(
        dtype=np.float64, copy=False
    )
    if len(valid) == 0:
        raise ValueError("No valid interval step lengths computed")
    return compute_step_length_extent_m(
        valid,
        step_q=float(step_q),
        max_step_q=float(max_step_q),
        step_multiplier=float(step_multiplier),
    )


def fit_interval_step_support(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    step_q: float = 0.95,
    max_step_q: float = 0.99,
    step_multiplier: float = 2.0,
) -> IntervalStepSupport:
    """Fit burst gap cutoffs and spatial support from the provided rows."""
    gap_cutoffs = fit_interval_gap_cutoffs(df, gap_q=float(gap_q))
    max_step_length_m = fit_interval_step_length_cutoff_m(
        df,
        step_time_s=int(step_time_s),
        gap_q=float(gap_q),
        gap_cutoffs=gap_cutoffs,
        step_q=float(step_q),
        max_step_q=float(max_step_q),
        step_multiplier=float(step_multiplier),
    )
    return IntervalStepSupport(
        gap_cutoffs=gap_cutoffs,
        max_step_length_m=float(max_step_length_m),
    )


def select_step_starts(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    gap_cutoffs: IntervalGapCutoffs | None = None,
    max_step_length_m: float | None = None,
) -> pd.DataFrame:
    """Select valid fixed-interval step starts and attach observed endpoints."""
    required = [INDIVIDUAL_COL, TIME_COL, LON_COL, LAT_COL, POINT_ID_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectories df missing required columns: {missing}")

    null_dict = pd.DataFrame(
            {
                "point_id_start": pd.Series(dtype="int64"),
                "point_id_end": pd.Series(dtype="int64"),
                "timestamp": pd.Series(dtype="datetime64[ns]"),
                "timestamp_end": pd.Series(dtype="datetime64[ns]"),
                "start_lon": pd.Series(dtype="float64"),
                "start_lat": pd.Series(dtype="float64"),
                "end_lon": pd.Series(dtype="float64"),
                "end_lat": pd.Series(dtype="float64"),
                "start_bearing": pd.Series(dtype="float64"),
                "step_length_m": pd.Series(dtype="float64"),
                "turn_angle_rad": pd.Series(dtype="float64"),
                "actual_step_time_s": pd.Series(dtype="float64"),
                "step_time_error_s": pd.Series(dtype="float64"),
                "burst_id": pd.Series(dtype="int64"),
                SET_COL: pd.Series(dtype="object"),
                INDIVIDUAL_COL: pd.Series(dtype="object"),
            })

    d = prepare_interval_step_starts(
        df,
        step_time_s=int(step_time_s),
        gap_q=float(gap_q),
        gap_cutoffs=gap_cutoffs,
        max_step_length_m=max_step_length_m,
    )
    if len(d) == 0:
        return null_dict

    grouped = d.groupby([INDIVIDUAL_COL, "burst_id"], sort=False)
    prev_lon = grouped[LON_COL].shift()
    prev_lat = grouped[LAT_COL].shift()
    d["start_bearing"] = arrival_bearing_rad(
        prev_lat.to_numpy(dtype=np.float64, copy=False),
        prev_lon.to_numpy(dtype=np.float64, copy=False),
        d[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        d[LON_COL].to_numpy(dtype=np.float64, copy=False),
    )

    starts = d[d["valid_step_start"]].reset_index(drop=True)
    if len(starts) == 0:
        return null_dict
    ends = d.loc[starts["step_end_index"].astype(int)].reset_index(drop=True)

    observed_heading = bearing_rad(
        starts[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        starts[LON_COL].to_numpy(dtype=np.float64, copy=False),
        ends[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        ends[LON_COL].to_numpy(dtype=np.float64, copy=False),
    )
    turn = observed_heading - starts["start_bearing"].to_numpy(
        dtype=np.float64, copy=False
    )
    turn = (turn + np.pi) % (2.0 * np.pi) - np.pi

    out = pd.DataFrame(
        {
            "point_id_start": starts[POINT_ID_COL].astype(int),
            "point_id_end": ends[POINT_ID_COL].astype(int),
            "timestamp": starts[TIME_COL],
            "timestamp_end": ends[TIME_COL],
            "start_lon": starts[LON_COL].astype(float),
            "start_lat": starts[LAT_COL].astype(float),
            "start_bearing": starts["start_bearing"].astype(float),
            "end_lon": ends[LON_COL].astype(float),
            "end_lat": ends[LAT_COL].astype(float),
            "step_length_m": starts["step_length_m"].astype(float),
            "turn_angle_rad": turn.astype(np.float64, copy=False),
            "actual_step_time_s": starts["actual_step_time_s"].astype(float),
            "step_time_error_s": starts["step_time_error_s"].astype(float),
            "burst_id": starts["burst_id"].astype(int),
            SET_COL: starts[SET_COL].astype(object),
            INDIVIDUAL_COL: starts[INDIVIDUAL_COL].astype(object),
        }
    )

    finite = np.isfinite(
        out.select_dtypes(include="floating").to_numpy(dtype=np.float64, copy=False)
    ).all(axis=1)
    return out.loc[finite].reset_index(drop=True)


def select_step_starts_with_support(
    df: pd.DataFrame,
    *,
    step_time_s: int,
    gap_q: float = 0.999,
    support: IntervalStepSupport,
) -> pd.DataFrame:
    """Select fixed-interval step starts using pre-fit burst and spatial support."""
    return select_step_starts(
        df,
        step_time_s=int(step_time_s),
        gap_q=float(gap_q),
        gap_cutoffs=support.gap_cutoffs,
        max_step_length_m=float(support.max_step_length_m),
    )


def fit_train_step_support(
    trajectories_df,
    *,
    step_spec,
) -> tuple[object, object]:
    train_df = trajectories_df[trajectories_df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Study has no train split for SSF grid prediction support")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        step_q=float(step_spec.step_q),
        max_step_q=float(step_spec.max_step_q),
        step_multiplier=float(step_spec.step_multiplier),
    )
    train_steps = select_step_starts_with_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        support=support,
    )
    return support, train_steps
