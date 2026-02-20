"""SSF/iSSF datasets backed by extracted point feature stores.

These datasets are intentionally thin wrappers over the on-disk point feature
stores written by `Study.extract_point_features(...)`.

They do not perform raster IO or candidate generation. They:
- load `store.pt` and `manifest.json`,
- optionally apply cached env normalization,
- (iSSF only) append movement/time/id features and interaction terms.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import PointStoreConfig, resolve_point_store_dir
from movebench.layers.plan import Channel, LayerPlan
from movebench.trajectories import INDIVIDUAL_COL, POINT_ID_COL, load_trajectories_csv


TARGET_CRS = "EPSG:3857" # meters


def _compute_movement_stats(step_length_m: torch.Tensor, turn_angle_rad: torch.Tensor) -> dict[str, float]:
    """Compute mean/std for sl, log_sl, cos_ta from true endpoints only."""
    sl = step_length_m[:, -1].to(dtype=torch.float64)
    ta = turn_angle_rad[:, -1].to(dtype=torch.float64)

    log_sl = torch.log(sl + 1e-8)
    cos_ta = torch.cos(ta)

    def ms(x: torch.Tensor) -> tuple[float, float]:
        m = float(x.mean().item())
        s = float(x.std(unbiased=False).item())
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        return m, s

    sl_mean, sl_std = ms(sl)
    log_sl_mean, log_sl_std = ms(log_sl)
    cos_ta_mean, cos_ta_std = ms(cos_ta)

    return {
        "sl_mean": sl_mean,
        "sl_std": sl_std,
        "log_sl_mean": log_sl_mean,
        "log_sl_std": log_sl_std,
        "cos_ta_mean": cos_ta_mean,
        "cos_ta_std": cos_ta_std,
    }


def _subset_plan_env_channels(
    plan: LayerPlan,
    *,
    subset_env_layers: list[str] | None,
    subset_band_ids: list[str] | None,
) -> tuple[list[int], LayerPlan]:
    if subset_env_layers is not None and subset_band_ids is not None:
        raise ValueError("Pass at most one of subset_env_layers or subset_band_ids")

    if subset_env_layers is None and subset_band_ids is None:
        return list(range(int(plan.num_channels()))), plan

    if subset_env_layers is not None:
        want = {str(x) for x in subset_env_layers}
        present_layers = {c.layer for c in plan.channels}
        missing = sorted(want - present_layers)
        if missing:
            raise ValueError(f"subset_env_layers contains unknown layer(s): {missing}. Available: {sorted(present_layers)}")
        indices = [i for i, ch in enumerate(plan.channels) if ch.layer in want]
    else:
        assert subset_band_ids is not None
        want = {str(x) for x in subset_band_ids}
        band_ids = plan.band_ids_in_order()
        present_band_ids = set(band_ids)
        missing = sorted(want - present_band_ids)
        if missing:
            raise ValueError(f"subset_band_ids contains unknown band_id(s): {missing}")
        indices = [i for i, bid in enumerate(band_ids) if bid in want]

    if len(indices) == int(plan.num_channels()):
        return indices, plan

    channels: list[Channel] = [plan.channels[i] for i in indices]
    layers_in_order: list[str] = []
    seen: set[str] = set()
    for ch in channels:
        if ch.layer in seen:
            continue
        seen.add(ch.layer)
        layers_in_order.append(ch.layer)

    chip_groups = {k: dict(v) for k, v in plan.chip_groups.items()}
    derived_dependencies = {k: list(v) for k, v in plan.derived_dependencies.items()}

    return indices, LayerPlan(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups=chip_groups,
        derived_dependencies=derived_dependencies,
    )


class SSFDataset(Dataset):
    """Empirical SSF dataset (env + valid mask) loaded from a point feature store."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        plan: LayerPlan,
        split: str,
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.split = str(split)

        store_cfg = PointStoreConfig(
            sampler_type="empirical",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
        store_dir, m = resolve_point_store_dir(
            study_dir=self.study_dir,
            split=self.split,
            config_hash=str(plan.config_hash),
            cfg=store_cfg,
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = m.layer_plan
        self.store_id = str(m.store_id)
        self.normalize = bool(normalize)

        d = torch.load(store_dir / "store.pt", map_location="cpu")
        self.env = d["env"]
        self.valid = d["valid"]
        self.step_length_m = d["step_length_m"]
        self.turn_angle_rad = d["turn_angle_rad"]
        self.point_id_start = d["point_id_start"]
        self.point_id_end = d["point_id_end"]
        self.timestamp_ns = d["timestamp_ns"]
        self.end_timestamp_ns = d["end_timestamp_ns"]
        self.start_lon = d["start_lon"]
        self.start_lat = d["start_lat"]
        self.start_bearing = d["start_bearing"]
        self.end_lon = d["end_lon"]
        self.end_lat = d["end_lat"]

        if self.env.ndim != 3 or self.valid.ndim != 3:
            raise ValueError(f"Expected env/valid rank-3 [N,C,P]; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if tuple(self.env.shape) != tuple(self.valid.shape):
            raise ValueError(f"env and valid shapes must match; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if int(self.env.shape[1]) != int(self.plan.num_channels()):
            raise ValueError(f"env.shape[1] ({self.env.shape[1]}) must match plan.num_channels() ({self.plan.num_channels()})")

        idxs, view_plan = _subset_plan_env_channels(self.plan, subset_env_layers=subset_env_layers, subset_band_ids=subset_band_ids)
        if len(idxs) != int(self.plan.num_channels()):
            self.env = self.env[:, idxs, :]
            self.valid = self.valid[:, idxs, :]
            self.plan = view_plan

        if self.normalize:
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                store_id=self.store_id,
            )
        else:
            self._mean = None
            self._std = None

    def __len__(self) -> int:
        return int(self.env.shape[0])

    @property
    def steps_df(self) -> pd.DataFrame:
        """Step metadata for each dataset index, as a pandas DataFrame.

        This is intended for linking dataset items back to rows in trajectories.csv via `point_id`.
        """
        cached = getattr(self, "_steps_df_cache", None)
        if cached is not None:
            return cached

        ts_ns = self.timestamp_ns.to(dtype=torch.int64).numpy()
        ts = pd.to_datetime(ts_ns, unit="ns", utc=True)
        end_ts_ns = self.end_timestamp_ns.to(dtype=torch.int64).numpy()
        end_ts = pd.to_datetime(end_ts_ns, unit="ns", utc=True)
        df = pd.DataFrame(
            {
                "point_id_start": self.point_id_start.to(dtype=torch.int64).numpy(),
                "point_id_end": self.point_id_end.to(dtype=torch.int64).numpy(),
                "timestamp_ns": ts_ns,
                "timestamp": ts,
                "end_timestamp_ns": end_ts_ns,
                "end_timestamp": end_ts,
                "start_lon": self.start_lon.to(dtype=torch.float32).numpy(),
                "start_lat": self.start_lat.to(dtype=torch.float32).numpy(),
                "start_bearing": self.start_bearing.to(dtype=torch.float32).numpy(),
                "end_lon": self.end_lon.to(dtype=torch.float32).numpy(),
                "end_lat": self.end_lat.to(dtype=torch.float32).numpy(),
            }
        )
        self._steps_df_cache = df
        return df

    def _get_env_valid(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env[int(idx)]
        valid = self.valid[int(idx)]

        if not self.normalize:
            return env, valid

        assert self._mean is not None
        assert self._std is not None
        mean = self._mean.view(-1, 1)
        std = self._std.view(-1, 1)
        env_norm = torch.where(valid, (env - mean) / std, torch.zeros_like(env))
        return env_norm, valid

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
        env, valid = self._get_env_valid(idx)
        return env, valid, {}, TARGET_CRS


class iSSFDataset(SSFDataset):
    """Parametric iSSF dataset (env + movement/time/id + interactions)."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        split: str,
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
        normalize_movement: bool = True,
        include_time: bool = False,
        include_individual_id: bool = False,
        movement_stats: dict[str, float] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        # Bypass SSFDataset selection: iSSF is always parametric.
        self.study_dir = Path(study_dir)
        self.split = str(split)
        self.trajectories_path = Path(trajectories_path)

        store_cfg = PointStoreConfig(
            sampler_type="parametric",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
        store_dir, m = resolve_point_store_dir(
            study_dir=self.study_dir,
            split=self.split,
            config_hash=str(plan.config_hash),
            cfg=store_cfg,
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = m.layer_plan
        self.store_id = str(m.store_id)
        self.normalize = bool(normalize)

        d = torch.load(store_dir / "store.pt", map_location="cpu")
        self.env = d["env"]
        self.valid = d["valid"]
        self.step_length_m = d["step_length_m"]
        self.turn_angle_rad = d["turn_angle_rad"]
        self.point_id_start = d["point_id_start"]
        self.point_id_end = d["point_id_end"]
        self.timestamp_ns = d["timestamp_ns"]
        self.end_timestamp_ns = d["end_timestamp_ns"]
        self.start_lon = d["start_lon"]
        self.start_lat = d["start_lat"]
        self.start_bearing = d["start_bearing"]
        self.end_lon = d["end_lon"]
        self.end_lat = d["end_lat"]

        if self.env.ndim != 3 or self.valid.ndim != 3:
            raise ValueError(f"Expected env/valid rank-3 [N,C,P]; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if tuple(self.env.shape) != tuple(self.valid.shape):
            raise ValueError(f"env and valid shapes must match; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if int(self.env.shape[1]) != int(self.plan.num_channels()):
            raise ValueError(f"env.shape[1] ({self.env.shape[1]}) must match plan.num_channels() ({self.plan.num_channels()})")

        idxs, view_plan = _subset_plan_env_channels(self.plan, subset_env_layers=subset_env_layers, subset_band_ids=subset_band_ids)
        if len(idxs) != int(self.plan.num_channels()):
            self.env = self.env[:, idxs, :]
            self.valid = self.valid[:, idxs, :]
            self.plan = view_plan

        if self.normalize:
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                store_id=self.store_id,
            )
        else:
            self._mean = None
            self._std = None

        self.include_time = bool(include_time)
        self.include_individual_id = bool(include_individual_id)
        self.normalize_movement = bool(normalize_movement)

        if self.normalize_movement:
            if movement_stats is not None:
                self._movement_stats = dict(movement_stats)
            elif self.split == "train":
                self._movement_stats = _compute_movement_stats(self.step_length_m, self.turn_angle_rad)
            else:
                train_dir = (
                    self.study_dir
                    / "features"
                    / "points"
                    / "train"
                    / str(self.plan.config_hash)
                    / str(self.store_id)
                )
                train_pt = train_dir / "store.pt"
                if not train_pt.exists():
                    raise FileNotFoundError(
                        "Train point feature store not found to compute movement_stats. "
                        f"Expected: {train_pt}"
                    )
                td = torch.load(train_pt, map_location="cpu")
                self._movement_stats = _compute_movement_stats(td["step_length_m"], td["turn_angle_rad"])
        else:
            self._movement_stats = None

        if self.include_individual_id:
            tdf = load_trajectories_csv(str(self.trajectories_path))
            if POINT_ID_COL not in tdf.columns or INDIVIDUAL_COL not in tdf.columns:
                raise ValueError(
                    f"trajectories.csv missing required columns: {POINT_ID_COL!r}, {INDIVIDUAL_COL!r}"
                )
            point_to_indiv: dict[int, str] = {
                int(pid): str(indiv) for pid, indiv in zip(tdf[POINT_ID_COL].to_numpy(), tdf[INDIVIDUAL_COL].to_numpy())
            }
            individuals = sorted(set(point_to_indiv.values()))
            self._point_to_indiv = point_to_indiv
            self._individual_to_idx = {ind: i for i, ind in enumerate(individuals)}
            self._n_individuals = int(len(individuals))
        else:
            self._point_to_indiv = {}
            self._individual_to_idx = {}
            self._n_individuals = 0

    @property
    def individual_mapping(self) -> dict[str, int]:
        return dict(self._individual_to_idx)

    @property
    def n_individuals(self) -> int:
        return int(self._n_individuals)

    def _movement_features(self, idx: int) -> torch.Tensor:
        sl = self.step_length_m[int(idx)].to(dtype=torch.float32)
        ta = self.turn_angle_rad[int(idx)].to(dtype=torch.float32)

        log_sl = torch.log(sl + 1e-8)
        cos_ta = torch.cos(ta)

        if self._movement_stats is not None:
            s = self._movement_stats
            sl = (sl - float(s["sl_mean"])) / float(s["sl_std"])
            log_sl = (log_sl - float(s["log_sl_mean"])) / float(s["log_sl_std"])
            cos_ta = (cos_ta - float(s["cos_ta_mean"])) / float(s["cos_ta_std"])

        return torch.stack([sl, log_sl, cos_ta], dim=0)

    def _time_features(self, idx: int, P: int) -> torch.Tensor:
        ts_ns = int(self.timestamp_ns[int(idx)].item())
        ts = pd.to_datetime(ts_ns, unit="ns", utc=True)

        doy = int(ts.dayofyear)
        hour_frac = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0

        sin_doy = math.sin(2.0 * math.pi * float(doy) / 365.25)
        cos_doy = math.cos(2.0 * math.pi * float(doy) / 365.25)
        sin_tod = math.sin(2.0 * math.pi * hour_frac / 24.0)
        cos_tod = math.cos(2.0 * math.pi * hour_frac / 24.0)

        feats = torch.tensor([sin_doy, cos_doy, sin_tod, cos_tod], dtype=torch.float32)
        return feats.view(4, 1).expand(4, P)

    def _individual_id_features(self, idx: int, P: int) -> torch.Tensor:
        pid = int(self.point_id_start[int(idx)].item())
        indiv = self._point_to_indiv.get(pid)
        if indiv is None:
            raise KeyError(f"point_id_start={pid} not found in trajectories.csv individual mapping")
        j = int(self._individual_to_idx[indiv])
        v = torch.zeros(self._n_individuals, dtype=torch.float32)
        v[j] = 1.0
        return v.view(self._n_individuals, 1).expand(self._n_individuals, P)

    @staticmethod
    def _interactions(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute row-wise interactions between (Ka,P) and (Kb,P) -> (Ka*Kb,P)."""
        Ka, P = a.shape
        Kb = b.shape[0]
        return (a.unsqueeze(1) * b.unsqueeze(0)).reshape(Ka * Kb, P)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
        env, env_valid = self._get_env_valid(idx)
        P = int(env.shape[1])
        C_env = int(env.shape[0])

        move = self._movement_features(idx)  # (3,P)
        move_valid = torch.ones((3, P), dtype=torch.bool)

        parts = [env, move]
        valids = [env_valid, move_valid]

        if self.include_time:
            time = self._time_features(idx, P)
            time_valid = torch.ones((4, P), dtype=torch.bool)
            parts.append(time)
            valids.append(time_valid)
        else:
            time = None

        if self.include_individual_id:
            indiv = self._individual_id_features(idx, P)
            indiv_valid = torch.ones((self._n_individuals, P), dtype=torch.bool)
            parts.append(indiv)
            valids.append(indiv_valid)

        move_env = self._interactions(move, env)
        move_env_valid = env_valid.unsqueeze(0).expand(3, -1, -1).reshape(3 * C_env, P)
        parts.append(move_env)
        valids.append(move_env_valid)

        if self.include_time and time is not None:
            time_env = self._interactions(time, env)
            time_env_valid = env_valid.unsqueeze(0).expand(4, -1, -1).reshape(4 * C_env, P)
            parts.append(time_env)
            valids.append(time_env_valid)

            time_move = self._interactions(time, move)
            time_move_valid = torch.ones((4 * 3, P), dtype=torch.bool)
            parts.append(time_move)
            valids.append(time_move_valid)

        step_pixels = torch.cat(parts, dim=0)
        valid_pixels = torch.cat(valids, dim=0)
        return step_pixels, valid_pixels, {}, TARGET_CRS
