"""MoveFormer dataset wrapper backed by MoveBench point feature stores."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.datasets.ssf import _compute_movement_stats, _subset_plan_env_channels
from movebench.features.config import SamplerConfig
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import (
    load_point_store_bands,
    load_point_store_manifest,
    load_point_store_shared,
    resolve_sampler_hash,
)
from movebench.layers.constants import WGS84_CRS
from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    INDIVIDUAL_COL,
    SET_COL,
    fit_interval_gap_cutoffs,
    fit_interval_step_length_cutoff_m,
    load_trajectories_csv,
    select_step_starts,
)

from pyproj import Transformer
from movebench.lib.moveformer.geo_transformers.trajectory import (
    SimpleTrajectoryProcessor,
)


@lru_cache(maxsize=1)
def _crs_transformers():
    to_3857 = Transformer.from_crs(WGS84_CRS, "EPSG:3857", always_xy=True)
    to_4326 = Transformer.from_crs("EPSG:3857", WGS84_CRS, always_xy=True)
    return to_3857, to_4326


def _timestamp_ns(ts: np.ndarray) -> np.ndarray:
    ts64 = ts.astype("datetime64[ns]", copy=False)
    return ts64.astype("int64", copy=False)


@dataclass(frozen=True)
class _StepStarts:
    point_id_start: np.ndarray
    start_ts: np.ndarray
    start_lon: np.ndarray
    start_lat: np.ndarray
    start_bearing: np.ndarray
    end_ts: np.ndarray
    end_lon: np.ndarray
    end_lat: np.ndarray
    individual: np.ndarray
    burst_id: np.ndarray


def _select_step_starts_with_targets(
    df_all: pd.DataFrame,
    *,
    plan: LayerPlan,
    selected_splits: list[str],
    split: str,
    train_gap_cutoffs,
    train_max_step_length_m: float,
) -> _StepStarts:
    df_selected = df_all[df_all[SET_COL].isin(selected_splits)].copy()
    starts = select_step_starts(
        df_selected,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        gap_cutoffs=train_gap_cutoffs,
        max_step_length_m=float(train_max_step_length_m),
    )
    starts = starts[starts[SET_COL] == str(split)].copy()
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
    return _StepStarts(
        point_id_start=starts["point_id_start"].to_numpy(dtype=np.int64, copy=False),
        start_ts=starts["timestamp"].to_numpy(dtype="datetime64[us]", copy=False),
        start_lon=starts["start_lon"].to_numpy(dtype=np.float64, copy=False),
        start_lat=starts["start_lat"].to_numpy(dtype=np.float64, copy=False),
        start_bearing=starts["start_bearing"].to_numpy(dtype=np.float64, copy=False),
        end_ts=starts["timestamp_end"].to_numpy(dtype="datetime64[us]", copy=False),
        end_lon=starts["end_lon"].to_numpy(dtype=np.float64, copy=False),
        end_lat=starts["end_lat"].to_numpy(dtype=np.float64, copy=False),
        individual=starts[INDIVIDUAL_COL].to_numpy(copy=False),
        burst_id=starts["burst_id"].to_numpy(dtype=np.int64, copy=False),
    )


class MoveFormerPointStoreDataset(Dataset):
    """Sequence dataset for MoveFormer selection model, built from point stores."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        sampler_config: SamplerConfig,
        split: str,
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
    ) -> None:
        super().__init__()
        if str(sampler_config.type) != "parametric":
            raise ValueError(
                f"MoveFormerPointStoreDataset requires a parametric sampler_config, got {sampler_config.type!r}"
            )

        self.study_dir = Path(study_dir)
        self.trajectories_path = Path(trajectories_path)
        self.split = str(split)
        self.max_len = int(max_len)
        self.min_len = int(min_len)
        self.normalize_env = bool(normalize_env)
        self.require_normalization_stats = bool(require_normalization_stats)

        requested_band_ids = plan.band_ids_in_order()
        sampler_hash = resolve_sampler_hash(sampler_config)
        store_dir, m = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )
        self.store_dir = store_dir
        self.manifest = m
        self.plan = plan

        self.env, self.valid = load_point_store_bands(
            store_dir=store_dir, manifest=m, band_ids=requested_band_ids
        )
        shared = load_point_store_shared(store_dir)
        self.step_length_m = shared["step_length_m"]
        self.turn_angle_rad = shared["turn_angle_rad"]
        self.point_id_start = shared["point_id_start"]
        self.timestamp_ns = shared["timestamp_ns"]

        if self.env.ndim != 3:
            raise ValueError(f"Expected env shape (N,C,P); got {tuple(self.env.shape)}")
        if tuple(self.env.shape) != tuple(self.valid.shape):
            raise ValueError(
                f"env and valid must have identical shapes; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}"
            )
        if int(self.env.shape[1]) != int(self.plan.num_channels()):
            raise ValueError(
                f"env.shape[1] ({self.env.shape[1]}) must match plan.num_channels() ({self.plan.num_channels()})"
            )
        if self.step_length_m.shape[:2] != self.env.shape[:1] + self.env.shape[2:]:
            raise ValueError(
                "Expected step_length_m shape (N,P) matching env; "
                f"got step_length_m={tuple(self.step_length_m.shape)} env={tuple(self.env.shape)}"
            )

        idxs, view_plan = _subset_plan_env_channels(
            self.plan, subset_env_layers=None, subset_band_ids=subset_band_ids
        )
        if len(idxs) != int(self.plan.num_channels()):
            self.env = self.env[:, idxs, :]
            self.valid = self.valid[:, idxs, :]
            self.plan = view_plan

        self.num_candidates_total = int(self.env.shape[2])
        if self.num_candidates_total <= 1:
            raise ValueError(
                f"Expected P>=2 candidates; got P={self.num_candidates_total}"
            )
        self.cov_dim = int(self.env.shape[1])
        self._cand_perm = torch.tensor(
            [self.num_candidates_total - 1]
            + list(range(self.num_candidates_total - 1)),
            dtype=torch.long,
        )

        if self.split == "train":
            stats = _compute_movement_stats(self.step_length_m, self.turn_angle_rad)
        else:
            train_dir = (
                self.study_dir
                / "features"
                / "points"
                / "train"
                / str(self.plan.rasters_hash)
                / str(self.manifest.sampler_hash)
            )
            train_pt = train_dir / "shared.pt"
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
                    sampler_config=sampler_config,
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

        df = load_trajectories_csv(str(self.trajectories_path))
        df_split = df[df[SET_COL] == self.split].copy()
        if len(df_split) == 0:
            raise ValueError(
                f"No trajectories found for split={self.split!r} in {self.trajectories_path}"
            )
        selected_splits = self.manifest.extraction.get("selected_splits")
        if not isinstance(selected_splits, list) or not selected_splits:
            selected_splits = sorted(df[SET_COL].dropna().astype(str).unique().tolist())
        train_df = df[df[SET_COL] == "train"].copy()
        if len(train_df) == 0:
            raise ValueError(
                f"No training trajectories found in {self.trajectories_path}"
            )
        train_gap_cutoffs = fit_interval_gap_cutoffs(
            train_df, gap_q=float(self.plan.step_spec.gap_q)
        )
        train_max_step_length_m = fit_interval_step_length_cutoff_m(
            train_df,
            step_time_s=int(self.plan.step_spec.step_time_s),
            gap_q=float(self.plan.step_spec.gap_q),
            gap_cutoffs=train_gap_cutoffs,
            step_q=float(self.plan.step_spec.step_q),
            max_step_q=float(self.plan.step_spec.max_step_q),
            step_multiplier=float(self.plan.step_spec.step_multiplier),
        )
        self._starts = _select_step_starts_with_targets(
            df,
            plan=self.plan,
            selected_splits=[str(x) for x in selected_splits],
            split=self.split,
            train_gap_cutoffs=train_gap_cutoffs,
            train_max_step_length_m=train_max_step_length_m,
        )

        n_steps = int(self.env.shape[0])
        if int(len(self._starts.point_id_start)) != n_steps:
            raise ValueError(
                f"Step-start count mismatch: point_store={n_steps} trajectories={len(self._starts.point_id_start)}"
            )

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

        self._prev_idx = np.full((n_steps,), -1, dtype=np.int64)
        indiv = self._starts.individual
        burst = self._starts.burst_id
        for i in range(1, n_steps):
            if indiv[i] == indiv[i - 1] and int(burst[i]) == int(burst[i - 1]):
                self._prev_idx[i] = i - 1

        self._segments: list[tuple[int, int]] = []
        i = 0
        while i < n_steps:
            j = i + 1
            while (
                j < n_steps and indiv[j] == indiv[i] and int(burst[j]) == int(burst[i])
            ):
                j += 1
            group_len = j - i
            if group_len >= self.min_len:
                for k in range(i, j, self.max_len):
                    kk = min(k + self.max_len, j)
                    if (kk - k) >= self.min_len:
                        self._segments.append((k, kk))
            i = j

        if not self._segments:
            raise ValueError(
                f"No segments found for split={self.split!r} (min_len={self.min_len}, max_len={self.max_len})"
            )

        self._processor = SimpleTrajectoryProcessor(
            use_target=True,
            use_candidates=True,
            add_dtime=True,
            add_mov_vec=True,
            mov_vec_scale=float(self._mov_vec_scale),
            handle_nan=False,
            pass_through=["taxon", "cov", "cov_cand"],
        )

        n_check = min(25, n_steps)
        if n_check > 0:
            lons_cand, lats_cand = self._reconstruct_candidates(0, n_check)
            lons_true = lons_cand[:, 0]
            lats_true = lats_cand[:, 0]
            max_err = float(
                np.max(
                    np.abs(
                        np.stack(
                            [
                                lons_true - self._starts.end_lon[:n_check],
                                lats_true - self._starts.end_lat[:n_check],
                            ],
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

    def _reconstruct_candidates(
        self, start: int, end: int
    ) -> tuple[np.ndarray, np.ndarray]:
        to_3857, to_4326 = _crs_transformers()

        start_lon = self._starts.start_lon[start:end]
        start_lat = self._starts.start_lat[start:end]
        start_bearing = self._starts.start_bearing[start:end]

        sl = self.step_length_m[start:end].numpy().astype(np.float64, copy=False)
        ta = self.turn_angle_rad[start:end].numpy().astype(np.float64, copy=False)

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

        perm = self._cand_perm.numpy()
        return lons[:, perm], lats[:, perm]

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        start, end = self._segments[int(idx)]
        segment_len = int(end - start)
        num_candidates = int(self.num_candidates_total)

        traj: dict[str, Any] = {
            "timestamp": self._starts.start_ts[start:end],
            "timestamp_tgt": self._starts.end_ts[start:end],
            "location_lat": self._starts.start_lat[start:end],
            "location_long": self._starts.start_lon[start:end],
            "location_lat_tgt": self._starts.end_lat[start:end],
            "location_long_tgt": self._starts.end_lon[start:end],
        }

        lons_cand, lats_cand = self._reconstruct_candidates(start, end)
        traj["location_long_cand"] = lons_cand
        traj["location_lat_cand"] = lats_cand

        env = self.env[start:end]
        valid = self.valid[start:end]
        if self.normalize_env:
            assert self._mean is not None
            assert self._std is not None
            mean = self._mean.view(1, -1, 1)
            std = self._std.view(1, -1, 1)
            env = (env - mean) / std

        env = env.index_select(dim=2, index=self._cand_perm)
        valid = valid.index_select(dim=2, index=self._cand_perm)
        cov_cand = env.permute(0, 2, 1).contiguous().numpy().astype(
            np.float32, copy=False
        )
        cov_v = valid.permute(0, 2, 1).contiguous().numpy()
        cov_cand = np.where(cov_v, cov_cand, np.float32(np.nan)).astype(
            np.float32, copy=False
        )

        global_idxs = np.arange(start, end, dtype=np.int64)
        prev = self._prev_idx[global_idxs]
        has_prev = prev >= 0

        cov_cur = np.full(
            (segment_len, int(self.cov_dim)), np.float32(np.nan), dtype=np.float32
        )
        if bool(has_prev.any()):
            prev_idxs = prev[has_prev]
            prev_t = torch.from_numpy(prev_idxs.astype(np.int64, copy=False))
            env_prev = self.env.index_select(dim=0, index=prev_t)
            valid_prev = self.valid.index_select(dim=0, index=prev_t)

            true_idx = int(num_candidates - 1)
            cov_prev = env_prev[:, :, true_idx]
            if self.normalize_env:
                assert self._mean is not None
                assert self._std is not None
                cov_prev = (cov_prev - self._mean.view(1, -1)) / self._std.view(1, -1)

            cov_prev = cov_prev.contiguous().numpy().astype(np.float32, copy=False)
            cov_prev_v = valid_prev[:, :, true_idx].contiguous().numpy()
            cov_prev = np.where(cov_prev_v, cov_prev, np.float32(np.nan)).astype(
                np.float32, copy=False
            )
            cov_cur[has_prev] = cov_prev

        traj.update(
            {
                "taxon": np.zeros((segment_len, 100), dtype=np.float32),
                "cov": cov_cur,
                "cov_cand": cov_cand,
            }
        )

        encoded = self._processor.encode(traj)
        encoded["loss_mask"] = has_prev.astype(np.bool_, copy=False)
        return encoded
