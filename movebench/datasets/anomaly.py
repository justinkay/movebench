"""Step-based anomaly detection dataset for movebench.

Mirrors the structure of the iSSF point-store (``movebench/features/points.py``) but:
- Reads environmental covariates from ``values.parquet`` (produced by
  ``Study.download_point_values()``) instead of raster chips.
- Has no candidate points — each step has exactly one true endpoint.
- Exposes ``point_id_start`` (same naming convention as ``points.py``).

Step selection uses the same ``select_step_starts()`` helper from
``trajectories.py`` as the rest of the codebase, so step counts, movement
thresholds, and burst definitions all behave identically to the iSSF pipeline.

Writes nothing to disk; purely a read-path dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.layers.plan import LayerPlan
from movebench.movement_math import haversine_m, safe_speed_m_s_numpy
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    OUTLIER_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    compute_movement_features,
)

# Movement feature names appended after the env block (one scalar per step).
_MOVEMENT_FEATURES      = ["step_length_m",      "step_dt_s",      "speed_m_s",      "turn_angle_rad"]
_MOVEMENT_FEATURES_NEXT = ["step_length_m_next", "step_dt_s_next", "speed_m_s_next", "turn_angle_rad_next"]

def _point_values_parquet(study_dir: Path, config_hash: str) -> Path:
    return Path(study_dir) / "point_values" / config_hash / "values.parquet"


def _build_step_block(
    df: pd.DataFrame,
    step_multiplier: float,
    step_q: float,
    gap_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (point_id, timestamps_ns, movement_array) for all GPS fixes with a computed step.

    Uses ``compute_movement_features()`` from ``trajectories.py`` to compute
    step features (step_length_m, time_since_last, turn_angle_rad) for every
    GPS fix that has a preceding fix in the same individual.  The only rows
    excluded are the first fix per individual (no incoming step).

    Critically, the ``valid_step`` step-length cutoff is NOT applied: that
    cutoff removes extreme displacements, which are precisely the anomalies we
    want to retain.

    ``point_id`` identifies the GPS fix being characterised (the endpoint of the
    incoming step).  This is the key used to join env features from
    ``values.parquet``.

    Returns
    -------
    point_id      : int64 array, shape (N,)
    timestamps_ns : int64 array, shape (N,) — nanoseconds since epoch
    movement_in   : float32 array, shape (N, 4) — incoming step features
                    columns: step_length_m, step_dt_s, speed_m_s, turn_angle_rad
    movement_out  : float32 array, shape (N, 4) — outgoing step features
                    columns: step_length_m_next, step_dt_s_next, speed_m_s_next, turn_angle_rad_next
                    NaN for the last fix per individual (no outgoing step).
    """
    mov = compute_movement_features(
        df, step_multiplier=step_multiplier, step_q=step_q, gap_q=gap_q
    )
    # Keep every row that has a preceding fix in the same individual.
    # time_since_last is NaT *only* for the first fix per individual.
    # We intentionally do NOT use valid_step or step_length_m.notna():
    #   - valid_step applies a step-length cutoff that excludes extreme outliers.
    # Instead we recompute step lengths here using shared haversine math so this
    # path is explicitly decoupled from any future valid-step filtering changes.
    has_prev = mov["time_since_last"].notna()

    g_mov = mov.groupby(INDIVIDUAL_COL, sort=False)

    # ---- incoming step ----
    lat_prev = g_mov[LAT_COL].shift().to_numpy(dtype=np.float64)
    lon_prev = g_mov[LON_COL].shift().to_numpy(dtype=np.float64)
    step_length_all = haversine_m(
        lat_prev, lon_prev,
        mov[LAT_COL].to_numpy(dtype=np.float64),
        mov[LON_COL].to_numpy(dtype=np.float64),
    )

    # ---- outgoing step ----
    lat_next = g_mov[LAT_COL].shift(-1).to_numpy(dtype=np.float64)
    lon_next = g_mov[LON_COL].shift(-1).to_numpy(dtype=np.float64)
    step_length_next_all = haversine_m(
        mov[LAT_COL].to_numpy(dtype=np.float64),
        mov[LON_COL].to_numpy(dtype=np.float64),
        lat_next, lon_next,
    )
    # time_since_last of the next row = time from current to next fix
    step_dt_next_all = (
        g_mov["time_since_last"].shift(-1).dt.total_seconds().to_numpy(dtype=np.float64)
    )
    turn_next_all = g_mov["turn_angle_rad"].shift(-1).to_numpy(dtype=np.float64)

    steps = mov[has_prev].copy()

    if len(steps) == 0:
        empty4 = np.empty((0, 4), dtype=np.float32)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), empty4, empty4

    hp = has_prev.to_numpy()

    # incoming
    step_length = step_length_all[hp]
    step_dt     = steps["time_since_last"].dt.total_seconds().to_numpy(dtype=np.float64)
    speed = safe_speed_m_s_numpy(step_length, step_dt)
    turn = steps["turn_angle_rad"].to_numpy(dtype=np.float64)
    movement_in = np.stack([step_length, step_dt, speed, turn], axis=1).astype(np.float32)

    # outgoing
    sl_next  = step_length_next_all[hp]
    dt_next  = step_dt_next_all[hp]
    speed_next = safe_speed_m_s_numpy(sl_next, dt_next)
    turn_next = turn_next_all[hp]
    movement_out = np.stack([sl_next, dt_next, speed_next, turn_next], axis=1).astype(np.float32)

    ts_ns = (
        steps[TIME_COL]
        .astype("datetime64[ns]")
        .astype(np.int64)
        .to_numpy(dtype=np.int64)
    )

    return steps[POINT_ID_COL].to_numpy(dtype=np.int64), ts_ns, movement_in, movement_out


def _load_env(
    study_dir: Path, config_hash: str, point_id_start: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Load env features from values.parquet, aligned to point_id_start order."""
    parquet_path = _point_values_parquet(study_dir, config_hash)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"point_values parquet not found: {parquet_path}\n"
            "Run Study.download_point_values() first."
        )
    env_df = pd.read_parquet(parquet_path)
    band_cols = [c for c in env_df.columns if c != POINT_ID_COL]

    env_df = env_df.set_index(POINT_ID_COL).reindex(point_id_start)
    return env_df[band_cols].to_numpy(dtype=np.float32), band_cols


def _z_score(
    arr: np.ndarray, stats: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Column-wise z-score; NaNs ignored in stat computation, preserved in output."""
    if stats is None:
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        std = np.where(std == 0, 1.0, std)
        stats = {"mean": mean, "std": std}
    return (arr - stats["mean"]) / stats["std"], stats


class AnomalyPointDataset(Dataset):
    """Anomaly detection dataset using point-valued env covariates + movement features.

    Each item corresponds to one GPS fix that has a preceding fix in the same
    individual (i.e. ``step_length_m`` is not NaN after ``compute_movement_features()``).
    The iSSF step-length cutoff is intentionally NOT applied — extreme displacements
    are the anomalies we want to keep.

    - ``point_id_start`` is the ``point_id`` of the GPS fix being characterised
      (the endpoint of the incoming step).
    - Environmental features are sampled at that fix from ``values.parquet``.
    - Movement features describe the incoming step:
      ``step_length_m``, ``step_dt_s``, ``speed_m_s``, ``turn_angle_rad``.

    Usage::

        train_ds = study.anomaly_dataset(split="train", config=LAYER_CONFIG)
        X = train_ds.numpy()                    # (N, F)
        y = train_ds.labels()                   # bool (N,)

        test_ds = study.anomaly_dataset(
            split="test",
            normalization_stats=train_ds.normalization_stats,
        )

    Args:
        study_dir:            Path to the study directory.
        trajectories_path:    Path to trajectories.csv.
        plan:                 LayerPlan (defines which env layers to load).
        split:                "train", "test", or ``None`` for all splits.
        include_env:               Include GEE environmental features (requires
                                   ``download_point_values()`` to have been run).
        include_movement:          Include incoming step movement features.
        include_outgoing_movement: Include outgoing step movement features
                                   (``*_next`` suffix).  Useful for diagnosing
                                   speed-outlier labels which Movebank assigns to
                                   the *origin* of a fast leap.
        normalize:            Z-score normalize using this split's non-NaN stats.
        normalization_stats:  Pre-computed ``{"mean": ..., "std": ...}`` to apply
                              instead of computing (pass ``train_ds.normalization_stats``
                              when building the test dataset).
        step_multiplier:      Passed to ``compute_movement_features()`` — same
                              semantics as in the iSSF pipeline.
        step_q:               Step-length quantile threshold.
        gap_q:                Time-gap quantile threshold.
    """

    def __init__(
        self,
        study_dir: Path,
        trajectories_path: Path,
        plan: LayerPlan,
        split: str | None = "train",
        include_env: bool = True,
        include_movement: bool = True,
        include_outgoing_movement: bool = False,
        normalize: bool = True,
        normalization_stats: dict[str, Any] | None = None,
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
    ) -> None:
        if not include_env and not include_movement and not include_outgoing_movement:
            raise ValueError("At least one of include_env, include_movement, or include_outgoing_movement must be True.")

        df = pd.read_csv(trajectories_path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")

        if split is not None:
            df = df[df[SET_COL] == split].copy()

        if len(df) == 0:
            raise ValueError(
                f"No points found for split={split!r} in {trajectories_path}."
            )

        # All GPS fixes with a valid incoming step (superset of iSSF step starts)
        pid_start, ts_ns, movement_in, movement_out = _build_step_block(
            df, step_multiplier=step_multiplier, step_q=step_q, gap_q=gap_q
        )

        feature_blocks: list[np.ndarray] = []
        feature_names: list[str] = []

        if include_env:
            env_arr, env_cols = _load_env(Path(study_dir), plan.config_hash, pid_start)
            feature_blocks.append(env_arr)
            feature_names.extend(env_cols)

        if include_movement:
            feature_blocks.append(movement_in)
            feature_names.extend(_MOVEMENT_FEATURES)

        if include_outgoing_movement:
            feature_blocks.append(movement_out)
            feature_names.extend(_MOVEMENT_FEATURES_NEXT)

        features = (
            np.concatenate(feature_blocks, axis=1)
            if len(feature_blocks) > 1
            else feature_blocks[0]
        )

        if normalize:
            features, normalization_stats = _z_score(features, normalization_stats)
        else:
            if normalization_stats is None:
                normalization_stats = {}

        self._features = features.astype(np.float32)
        self._point_id_start = pid_start      # int64 (N,) — matches points.py naming
        self._timestamp_ns = ts_ns            # int64 (N,) — nanoseconds since epoch
        self.feature_names: list[str] = feature_names
        self.normalization_stats: dict[str, Any] = normalization_stats
        self.split = split
        self.plan = plan

        # Meta index: full trajectories (for label/individual lookup).
        # Indexed by point_id so we can look up attributes of the step-start fix.
        self._meta_df = pd.read_csv(trajectories_path).set_index(POINT_ID_COL)

    # ---- Dataset protocol ----

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return ``(feature_tensor, point_id_start)``.

        ``feature_tensor`` has shape ``(F,)``.  NaN values are kept as-is —
        impute (e.g. ``nan_to_num``) before passing to sklearn estimators.
        """
        return torch.from_numpy(self._features[idx]), int(self._point_id_start[idx])

    # ---- Named accessors (consistent with points.py tensor-store keys) ----

    @property
    def point_id_start(self) -> np.ndarray:
        """int64 array of GPS fix IDs (endpoint of each incoming step), shape (N,)."""
        return self._point_id_start

    @property
    def timestamp_ns(self) -> np.ndarray:
        """int64 array of GPS fix timestamps in nanoseconds, shape (N,)."""
        return self._timestamp_ns

    # ---- Convenience helpers ----

    def numpy(self) -> np.ndarray:
        """Return feature matrix as ``(N, F)`` float32 numpy array."""
        return self._features.copy()

    def labels(self, *, any_outlier: bool = True) -> np.ndarray:
        """Boolean outlier labels aligned to step order (at ``point_id_start``).

        Args:
            any_outlier: If True, OR all ``*outlier*`` columns in trajectories.csv.

        Returns:
            Boolean array of shape ``(N,)``.  False where label is unknown/missing.
        """
        outlier_cols = [c for c in self._meta_df.columns if "outlier" in c.lower()]
        if not outlier_cols:
            return np.zeros(len(self), dtype=bool)

        meta = self._meta_df.reindex(self._point_id_start)
        if any_outlier:
            return (
                pd.DataFrame({c: meta[c].fillna(False).astype(bool) for c in outlier_cols})
                .any(axis=1)
                .to_numpy(dtype=bool)
            )

        col = OUTLIER_COL if OUTLIER_COL in self._meta_df.columns else outlier_cols[0]
        return meta[col].fillna(False).astype(bool).to_numpy(dtype=bool)

    def individual_ids(self) -> np.ndarray:
        """Individual identifier for each step start, shape (N,)."""
        return self._meta_df.reindex(self._point_id_start)[INDIVIDUAL_COL].to_numpy()

    def to_dataframe(self) -> pd.DataFrame:
        """Return ``point_id_start`` + features as a pandas DataFrame."""
        df = pd.DataFrame(self._features, columns=self.feature_names)
        df.insert(0, "point_id_start", self._point_id_start)
        return df
