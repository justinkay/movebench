"""MoveFormer dataset wrapper backed by MoveBench point feature stores.

This integration reuses:
- MoveBench's extracted iSSF point stores for candidate sets + env values, and
- MoveFormer's upstream `SimpleTrajectoryProcessor.encode` for feature encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.datasets.ssf import _subset_plan_env_channels
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import select_point_store_dir
from movebench.layers.constants import WGS84_CRS
from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    SET_COL,
    TIME_COL,
    compute_movement_features,
    load_trajectories_csv,
)


@lru_cache(maxsize=1)
def _crs_transformers():
    from pyproj import Transformer

    to_3857 = Transformer.from_crs(WGS84_CRS, "EPSG:3857", always_xy=True)
    to_4326 = Transformer.from_crs("EPSG:3857", WGS84_CRS, always_xy=True)
    return to_3857, to_4326


def _timestamp_ns(ts: np.ndarray) -> np.ndarray:
    # Expect dtype datetime64[ns]; use astype('datetime64[ns]') to be robust.
    ts64 = ts.astype("datetime64[ns]", copy=False)
    return ts64.astype("int64", copy=False)


@dataclass(frozen=True)
class _StepStarts:
    point_id_start: np.ndarray  # (N,)
    start_ts: np.ndarray  # (N,) datetime64[ns]
    start_lon: np.ndarray  # (N,)
    start_lat: np.ndarray  # (N,)
    start_bearing: np.ndarray  # (N,)
    end_ts: np.ndarray  # (N,) datetime64[ns]
    end_lon: np.ndarray  # (N,)
    end_lat: np.ndarray  # (N,)
    individual: np.ndarray  # (N,) object/str
    burst_id: np.ndarray  # (N,) int


def _select_step_starts_with_targets(
    df_split: pd.DataFrame,
    *,
    step_multiplier: float,
    step_q: float,
    gap_q: float,
) -> _StepStarts:
    d = compute_movement_features(
        df_split,
        step_multiplier=float(step_multiplier),
        step_q=float(step_q),
        gap_q=float(gap_q),
    )
    starts = d[d["can_be_step_start"]].copy()
    starts = starts[starts["next_index"].notna()].copy()
    if len(starts) == 0:
        return _StepStarts(
            point_id_start=np.zeros((0,), dtype=np.int64),
            start_ts=np.zeros((0,), dtype="datetime64[ns]"),
            start_lon=np.zeros((0,), dtype=np.float64),
            start_lat=np.zeros((0,), dtype=np.float64),
            start_bearing=np.zeros((0,), dtype=np.float64),
            end_ts=np.zeros((0,), dtype="datetime64[ns]"),
            end_lon=np.zeros((0,), dtype=np.float64),
            end_lat=np.zeros((0,), dtype=np.float64),
            individual=np.zeros((0,), dtype=object),
            burst_id=np.zeros((0,), dtype=np.int64),
        )

    next_idx = starts["next_index"].astype(int).to_numpy()
    ends = d.loc[next_idx].copy()

    starts = starts.reset_index(drop=True)
    ends = ends.reset_index(drop=True)
    if len(starts) != len(ends):
        raise RuntimeError(f"Internal error: starts/ends length mismatch ({len(starts)} vs {len(ends)})")

    out = _StepStarts(
        point_id_start=starts[POINT_ID_COL].to_numpy(dtype=np.int64, copy=False),
        # Use microsecond resolution so numpy.datetime64.item() returns datetime.datetime
        # (MoveFormer's upstream encoder assumes that behavior).
        start_ts=starts[TIME_COL].to_numpy(dtype="datetime64[us]", copy=False),
        start_lon=starts[LON_COL].to_numpy(dtype=np.float64, copy=False),
        start_lat=starts[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        start_bearing=starts["bearing"].to_numpy(dtype=np.float64, copy=False),
        end_ts=ends[TIME_COL].to_numpy(dtype="datetime64[us]", copy=False),
        end_lon=ends[LON_COL].to_numpy(dtype=np.float64, copy=False),
        end_lat=ends[LAT_COL].to_numpy(dtype=np.float64, copy=False),
        individual=starts[INDIVIDUAL_COL].to_numpy(copy=False),
        burst_id=starts["burst_id"].to_numpy(dtype=np.int64, copy=False),
    )

    finite = np.isfinite(
        np.stack([out.start_lon, out.start_lat, out.start_bearing, out.end_lon, out.end_lat], axis=1)
    ).all(axis=1)
    if not finite.all():
        keep = finite
        out = _StepStarts(
            point_id_start=out.point_id_start[keep],
            start_ts=out.start_ts[keep],
            start_lon=out.start_lon[keep],
            start_lat=out.start_lat[keep],
            start_bearing=out.start_bearing[keep],
            end_ts=out.end_ts[keep],
            end_lon=out.end_lon[keep],
            end_lat=out.end_lat[keep],
            individual=out.individual[keep],
            burst_id=out.burst_id[keep],
        )
    return out


class MoveFormerPointStoreDataset(Dataset):
    """Sequence dataset for MoveFormer selection model, built from point stores.

    Each item is a variable-length segment (<= max_len) from one (individual, burst).
    """

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        split: str,
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        super().__init__()
        self.study_dir = Path(study_dir)
        self.trajectories_path = Path(trajectories_path)
        self.split = str(split)
        self.max_len = int(max_len)
        self.min_len = int(min_len)
        self.normalize_env = bool(normalize_env)
        self.require_normalization_stats = bool(require_normalization_stats)

        store_dir, m = select_point_store_dir(
            study_dir=self.study_dir,
            split=self.split,
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
        self.store_dir = store_dir
        self.manifest = m
        self.plan = m.layer_plan

        d = torch.load(store_dir / "features.pt", map_location="cpu")
        self.env: torch.Tensor = d["env"]
        self.valid: torch.Tensor = d["valid"]
        self.step_length_m: torch.Tensor = d["step_length_m"]
        self.turn_angle_rad: torch.Tensor = d["turn_angle_rad"]
        self.point_id_start: torch.Tensor = d["point_id_start"]
        self.timestamp_ns: torch.Tensor = d["timestamp_ns"]

        if self.env.ndim != 3:
            raise ValueError(f"Expected env shape (N,C,P); got {tuple(self.env.shape)}")
        if tuple(self.env.shape) != tuple(self.valid.shape):
            raise ValueError(f"env and valid must have identical shapes; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if int(self.env.shape[1]) != int(self.plan.num_channels()):
            raise ValueError(f"env.shape[1] ({self.env.shape[1]}) must match plan.num_channels() ({self.plan.num_channels()})")
        if self.step_length_m.shape[:2] != self.env.shape[:1] + self.env.shape[2:]:
            raise ValueError(
                "Expected step_length_m shape (N,P) matching env; "
                f"got step_length_m={tuple(self.step_length_m.shape)} env={tuple(self.env.shape)}"
            )

        idxs, view_plan = _subset_plan_env_channels(self.plan, subset_env_layers=None, subset_band_ids=subset_band_ids)
        if len(idxs) != int(self.plan.num_channels()):
            self.env = self.env[:, idxs, :]
            self.valid = self.valid[:, idxs, :]
            self.plan = view_plan

        self.num_candidates_total = int(self.env.shape[2])
        if self.num_candidates_total <= 1:
            raise ValueError(f"Expected P>=2 candidates; got P={self.num_candidates_total}")
        self.cov_dim = int(self.env.shape[1])
        self._cand_perm = torch.tensor(
            [self.num_candidates_total - 1] + list(range(self.num_candidates_total - 1)),
            dtype=torch.long,
        )

        from movebench.datasets.ssf import _compute_movement_stats

        if self.split == "train":
            stats = _compute_movement_stats(self.step_length_m, self.turn_angle_rad)
        else:
            train_dir = (
                self.study_dir
                / "features"
                / "points"
                / "train"
                / str(self.plan.config_hash)
                / str(self.manifest.sampler_hash)
            )
            train_pt = train_dir / "features.pt"
            if not train_pt.exists():
                raise FileNotFoundError(
                    "Train point feature store not found to compute mov_vec_scale. "
                    f"Expected: {train_pt}"
                )
            td = torch.load(train_pt, map_location="cpu")
            stats = _compute_movement_stats(td["step_length_m"], td["turn_angle_rad"])

        self._mov_vec_scale = 1.0 / float(stats["sl_std"])

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if self.normalize_env:
            try:
                self._mean, self._std = load_points_normalization_mean_std(
                    study_dir=self.study_dir,
                    plan=self.plan,
                    sampler_hash=str(self.manifest.sampler_hash),
                )
            except FileNotFoundError as e:
                if self.require_normalization_stats:
                    raise FileNotFoundError(
                        "MoveFormer covariate normalization stats not found. Run "
                        "Study.compute_normalization_vals(from_='points', ...) first, "
                        "or set require_normalization_stats=false."
                    ) from e
                warnings.warn(
                    "MoveFormer covariate normalization requested, but stats were not found; "
                    "proceeding with raw env values. "
                    "Run Study.compute_normalization_vals(from_='points', ...) to enable normalization.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.normalize_env = False

        # Load trajectories + compute step starts aligned to point store ordering.
        df = load_trajectories_csv(str(self.trajectories_path))
        df_split = df[df[SET_COL] == self.split].copy()
        if len(df_split) == 0:
            raise ValueError(f"No trajectories found for split={self.split!r} in {self.trajectories_path}")

        # Recompute movement features with the same params used by point-store extraction.
        sampler = self.manifest.sampler
        self._starts = _select_step_starts_with_targets(
            df_split,
            step_multiplier=float(sampler["step_multiplier"]),
            step_q=float(sampler["step_q"]),
            gap_q=float(sampler["gap_q"]),
        )

        n_steps = int(self.env.shape[0])
        if int(len(self._starts.point_id_start)) != n_steps:
            raise ValueError(f"Step-start count mismatch: point_store={n_steps} trajectories={len(self._starts.point_id_start)}")

        # Assert alignment by point_id_start + timestamp.
        pid_store = self.point_id_start.numpy()
        pid_traj = self._starts.point_id_start
        if not np.array_equal(pid_store, pid_traj):
            i = int(np.nonzero(pid_store != pid_traj)[0][0])
            raise ValueError(
                "Point-store step ordering mismatch at index "
                f"{i}: store_point_id_start={int(pid_store[i])} traj_point_id_start={int(pid_traj[i])}"
            )

        ts_store = self.timestamp_ns.numpy()
        ts_traj = _timestamp_ns(self._starts.start_ts)
        if not np.array_equal(ts_store, ts_traj):
            i = int(np.nonzero(ts_store != ts_traj)[0][0])
            raise ValueError(
                "Point-store timestamp mismatch at index "
                f"{i}: store_ts_ns={int(ts_store[i])} traj_ts_ns={int(ts_traj[i])}"
            )

        # Precompute previous-step index within (individual, burst_id).
        self._prev_idx = np.full((n_steps,), -1, dtype=np.int64)
        indiv = self._starts.individual
        burst = self._starts.burst_id
        for i in range(1, n_steps):
            if indiv[i] == indiv[i - 1] and int(burst[i]) == int(burst[i - 1]):
                self._prev_idx[i] = i - 1

        # Build segment slices (contiguous within burst).
        self._segments: list[tuple[int, int]] = []
        i = 0
        while i < n_steps:
            j = i + 1
            while j < n_steps and indiv[j] == indiv[i] and int(burst[j]) == int(burst[i]):
                j += 1

            group_len = j - i
            if group_len >= self.min_len:
                for k in range(i, j, self.max_len):
                    kk = min(k + self.max_len, j)
                    if (kk - k) >= self.min_len:
                        self._segments.append((k, kk))
            i = j

        if not self._segments:
            raise ValueError(f"No segments found for split={self.split!r} (min_len={self.min_len}, max_len={self.max_len})")

        from movebench.lib.moveformer.geo_transformers.trajectory import SimpleTrajectoryProcessor

        self._processor = SimpleTrajectoryProcessor(
            use_target=True,
            use_candidates=True,
            add_dtime=True,
            add_mov_vec=True,
            mov_vec_scale=float(self._mov_vec_scale),
            handle_nan=False,
            pass_through=["taxon", "cov", "cov_cand"],
        )

        # Light sanity check: reconstructed true candidate matches the observed endpoint.
        n_check = min(25, n_steps)
        if n_check > 0:
            lons_cand, lats_cand = self._reconstruct_candidates(0, n_check)  # (n_check,P)
            lons_true = lons_cand[:, 0]
            lats_true = lats_cand[:, 0]
            max_err = float(
                np.max(
                    np.abs(
                        np.stack(
                            [lons_true - self._starts.end_lon[:n_check], lats_true - self._starts.end_lat[:n_check]],
                            axis=1,
                        )
                    )
                )
            )
            if not np.isfinite(max_err) or max_err > 1e-4:
                raise ValueError(
                    "Candidate reconstruction sanity check failed: "
                    f"max_abs_deg_error={max_err:.3e} (expected <= 1e-4)."
                )

    def __len__(self) -> int:
        return len(self._segments)

    def _reconstruct_candidates(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (lons, lats) for candidates with MoveFormer ordering (true is index 0)."""
        to_3857, to_4326 = _crs_transformers()

        start_lon = self._starts.start_lon[start:end]
        start_lat = self._starts.start_lat[start:end]
        start_bearing = self._starts.start_bearing[start:end]

        sl = self.step_length_m[start:end].numpy().astype(np.float64, copy=False)  # (L,P)
        ta = self.turn_angle_rad[start:end].numpy().astype(np.float64, copy=False)  # (L,P)

        x0, y0 = to_3857.transform(start_lon, start_lat)
        x0 = np.asarray(x0, dtype=np.float64)
        y0 = np.asarray(y0, dtype=np.float64)

        heading = start_bearing[:, None] + ta
        dx = sl * np.sin(heading)
        dy = sl * np.cos(heading)

        xs = x0[:, None] + dx
        ys = y0[:, None] + dy

        lons, lats = to_4326.transform(xs, ys)
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)

        # Reorder: true endpoint first.
        perm = self._cand_perm.numpy()
        return lons[:, perm], lats[:, perm]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start, end = self._segments[int(idx)]
        L = int(end - start)
        P = int(self.num_candidates_total)

        # --- trajectory core fields ---
        traj: dict[str, Any] = {
            "timestamp": self._starts.start_ts[start:end],
            "timestamp_tgt": self._starts.end_ts[start:end],
            "location_lat": self._starts.start_lat[start:end],
            "location_long": self._starts.start_lon[start:end],
            "location_lat_tgt": self._starts.end_lat[start:end],
            "location_long_tgt": self._starts.end_lon[start:end],
        }

        lons_cand, lats_cand = self._reconstruct_candidates(start, end)  # (L,P)
        traj["location_long_cand"] = lons_cand
        traj["location_lat_cand"] = lats_cand

        # --- env features from point store ---
        env = self.env[start:end]  # (L,C,P)
        valid = self.valid[start:end]  # (L,C,P)

        if self.normalize_env:
            assert self._mean is not None
            assert self._std is not None
            mean = self._mean.view(1, -1, 1)
            std = self._std.view(1, -1, 1)
            env = (env - mean) / std

        # Index candidates into MoveFormer order.
        env = env.index_select(dim=2, index=self._cand_perm)
        valid = valid.index_select(dim=2, index=self._cand_perm)

        cov_cand = env.permute(0, 2, 1).contiguous().numpy().astype(np.float32, copy=False)  # (L,P,C)
        cov_v = valid.permute(0, 2, 1).contiguous().numpy()  # (L,P,C)
        cov_cand = np.where(cov_v, cov_cand, np.float32(np.nan)).astype(np.float32, copy=False)

        # Current-location env is previous step's target within the same burst.
        global_idxs = np.arange(start, end, dtype=np.int64)
        prev = self._prev_idx[global_idxs]
        has_prev = prev >= 0

        cov_cur = np.full((L, int(self.cov_dim)), np.float32(np.nan), dtype=np.float32)

        if bool(has_prev.any()):
            prev_idxs = prev[has_prev]
            prev_t = torch.from_numpy(prev_idxs.astype(np.int64, copy=False))
            env_prev = self.env.index_select(dim=0, index=prev_t)  # (K,C,P)
            valid_prev = self.valid.index_select(dim=0, index=prev_t)  # (K,C,P)

            # Previous target is the true endpoint (original last candidate).
            true_idx = int(P - 1)
            cov_prev = env_prev[:, :, true_idx]
            if self.normalize_env:
                assert self._mean is not None
                assert self._std is not None
                cov_prev = (cov_prev - self._mean.view(1, -1)) / self._std.view(1, -1)

            cov_prev = cov_prev.contiguous().numpy().astype(np.float32, copy=False)  # (K,C)
            cov_prev_v = valid_prev[:, :, true_idx].contiguous().numpy()  # (K,C)
            cov_prev = np.where(cov_prev_v, cov_prev, np.float32(np.nan)).astype(np.float32, copy=False)

            cov_cur[has_prev] = cov_prev

        traj.update(
            {
                "taxon": np.zeros((L, 100), dtype=np.float32),
                "cov": cov_cur,
                "cov_cand": cov_cand,
            }
        )

        encoded = self._processor.encode(traj)
        encoded["loss_mask"] = has_prev.astype(np.bool_, copy=False)
        return encoded
