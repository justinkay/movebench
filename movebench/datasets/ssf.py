"""SSF/iSSF datasets backed by extracted point feature stores.

These datasets are intentionally thin wrappers over the on-disk point feature
stores written by `Study.extract_point_features(...)`.

They do not perform raster IO or candidate generation. They:
- load band files, `shared.pt`, and `manifest.json`,
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
from movebench.features.points import (
    load_point_store_bands,
    load_point_store_manifest,
    load_point_store_shared,
    resolve_sampler_hash,
)
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
        max_step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.split = str(split)
        requested_band_ids = plan.band_ids_in_order()
        sampler_hash = resolve_sampler_hash(
            sampler_type="empirical",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            max_step_q=max_step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
        store_dir, m = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = plan
        self.sampler_hash = str(m.sampler_hash)
        self.normalize = bool(normalize)

        self.env, self.valid = load_point_store_bands(store_dir=store_dir, manifest=m, band_ids=requested_band_ids)
        shared = load_point_store_shared(store_dir)
        self.step_length_m = shared.get("step_length_m")
        self.turn_angle_rad = shared.get("turn_angle_rad")
        self.point_id_start = shared.get("point_id_start")
        self.timestamp_ns = shared.get("timestamp_ns")

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
                sampler_hash=self.sampler_hash,
            )
        else:
            self._mean = None
            self._std = None

    def __len__(self) -> int:
        return int(self.env.shape[0])

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
        max_step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        # Bypass SSFDataset selection: iSSF is always parametric.
        self.study_dir = Path(study_dir)
        self.split = str(split)
        self.trajectories_path = Path(trajectories_path)

        requested_band_ids = plan.band_ids_in_order()
        sampler_hash = resolve_sampler_hash(
            sampler_type="parametric",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            max_step_q=max_step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
        store_dir, m = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = plan
        self.sampler_hash = str(m.sampler_hash)
        self.normalize = bool(normalize)

        self.env, self.valid = load_point_store_bands(store_dir=store_dir, manifest=m, band_ids=requested_band_ids)
        shared = load_point_store_shared(store_dir)
        self.step_length_m = shared.get("step_length_m")
        self.turn_angle_rad = shared.get("turn_angle_rad")
        self.point_id_start = shared.get("point_id_start")
        self.timestamp_ns = shared.get("timestamp_ns")

        if self.step_length_m is None or self.turn_angle_rad is None or self.point_id_start is None or self.timestamp_ns is None:
            raise ValueError("shared.pt missing required iSSF keys: step_length_m, turn_angle_rad, point_id_start, timestamp_ns")

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
                sampler_hash=self.sampler_hash,
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
                    / str(self.plan.rasters_hash)
                    / str(self.sampler_hash)
                )
                train_pt = train_dir / "shared.pt"
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

    def build_expanded_x(
        self,
        *,
        cache_device: torch.device,
        dtype: torch.dtype,
        fallback_to_pinned_cpu: bool = True,
        chunk_size: int = 2048,
        validate: bool = False,
    ) -> torch.Tensor:
        """Materialize the full (N,C,P) iSSF feature tensor for fast training.

        This is a vectorized equivalent of `__getitem__` that constructs the
        entire expanded feature matrix once, instead of per-sample in Python.

        Returns:
            x: (N, C_total, P) tensor stored on `cache_device` when possible,
               otherwise on pinned CPU memory if CUDA OOM and
               `fallback_to_pinned_cpu=True`.

        Notes:
            - `valid` masks are not returned. Invalid env entries are zeroed
              during normalization, and downstream iSSF training ignores valid.
            - Output channel ordering matches `__getitem__`.
        """

        cache_device = torch.device(cache_device)
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be > 0; got {chunk_size}")

        N, C_env, P = self.env.shape
        n_indiv = int(self._n_individuals) if self.include_individual_id else 0
        c_time = 4 if self.include_time else 0
        c_total = int(C_env) + 3 + int(c_time) + int(n_indiv) + (3 * int(C_env)) + (0 if c_time == 0 else (4 * int(C_env) + 12))

        # Precompute time features on CPU once (small) to avoid repeated pandas work.
        time4_cpu: torch.Tensor | None
        if self.include_time:
            ts_ns = self.timestamp_ns.detach().cpu().numpy()
            ts = pd.to_datetime(ts_ns, unit="ns", utc=True)
            doy = torch.from_numpy(ts.dayofyear.to_numpy()).to(dtype=torch.float32)  # (N,)
            hour = torch.from_numpy(ts.hour.to_numpy()).to(dtype=torch.float32)
            minute = torch.from_numpy(ts.minute.to_numpy()).to(dtype=torch.float32)
            second = torch.from_numpy(ts.second.to_numpy()).to(dtype=torch.float32)
            hour_frac = hour + (minute / 60.0) + (second / 3600.0)

            two_pi = float(2.0 * math.pi)
            sin_doy = torch.sin(two_pi * doy / 365.25)
            cos_doy = torch.cos(two_pi * doy / 365.25)
            sin_tod = torch.sin(two_pi * hour_frac / 24.0)
            cos_tod = torch.cos(two_pi * hour_frac / 24.0)
            time4_cpu = torch.stack([sin_doy, cos_doy, sin_tod, cos_tod], dim=1).contiguous()  # (N,4)
        else:
            time4_cpu = None

        indiv_idx_cpu: torch.Tensor | None
        if self.include_individual_id:
            pids = self.point_id_start.detach().cpu().tolist()
            idxs: list[int] = []
            for pid in pids:
                indiv = self._point_to_indiv.get(int(pid))
                if indiv is None:
                    raise KeyError(f"point_id_start={pid} not found in trajectories.csv individual mapping")
                idxs.append(int(self._individual_to_idx[str(indiv)]))
            indiv_idx_cpu = torch.tensor(idxs, dtype=torch.long)
        else:
            indiv_idx_cpu = None

        def _build_into(*, out: torch.Tensor, op_device: torch.device) -> torch.Tensor:
            non_blocking = op_device.type == "cuda"

            mean = None
            std = None
            if self.normalize:
                assert self._mean is not None
                assert self._std is not None
                mean = self._mean.to(device=op_device, dtype=torch.float32, non_blocking=non_blocking).view(1, -1, 1)
                std = self._std.to(device=op_device, dtype=torch.float32, non_blocking=non_blocking).view(1, -1, 1)

            for s in range(0, int(N), int(chunk_size)):
                e = min(int(N), s + int(chunk_size))
                b = int(e - s)

                # --- env (normalized; invalid entries zeroed) ---
                env = self.env[s:e].to(device=op_device, dtype=torch.float32, non_blocking=non_blocking)
                valid = self.valid[s:e].to(device=op_device, non_blocking=non_blocking)
                if self.normalize:
                    assert mean is not None and std is not None
                    env = torch.where(valid, (env - mean) / std, torch.zeros_like(env))
                else:
                    env = torch.where(valid, env, torch.zeros_like(env))

                # --- movement (normalized using ds._movement_stats) ---
                sl = self.step_length_m[s:e].to(device=op_device, dtype=torch.float32, non_blocking=non_blocking)  # (B,P)
                ta = self.turn_angle_rad[s:e].to(device=op_device, dtype=torch.float32, non_blocking=non_blocking)  # (B,P)
                log_sl = torch.log(sl + 1e-8)
                cos_ta = torch.cos(ta)
                move = torch.stack([sl, log_sl, cos_ta], dim=1)  # (B,3,P)
                if self._movement_stats is not None:
                    st = self._movement_stats
                    move[:, 0, :] = (move[:, 0, :] - float(st["sl_mean"])) / float(st["sl_std"])
                    move[:, 1, :] = (move[:, 1, :] - float(st["log_sl_mean"])) / float(st["log_sl_std"])
                    move[:, 2, :] = (move[:, 2, :] - float(st["cos_ta_mean"])) / float(st["cos_ta_std"])

                # --- optional time + id features ---
                if time4_cpu is not None:
                    t4 = time4_cpu[s:e].to(device=op_device, dtype=torch.float32, non_blocking=non_blocking)  # (B,4)
                    time_feats = t4.unsqueeze(-1).expand(b, 4, int(P))  # (B,4,P)
                else:
                    time_feats = None

                if indiv_idx_cpu is not None:
                    idx = indiv_idx_cpu[s:e].to(device=op_device, non_blocking=non_blocking)  # (B,)
                    onehot = torch.zeros((b, int(n_indiv)), device=op_device, dtype=torch.float32)
                    onehot[torch.arange(b, device=op_device), idx] = 1.0
                    indiv_feats = onehot.unsqueeze(-1).expand(b, int(n_indiv), int(P))  # (B,n_indiv,P)
                else:
                    indiv_feats = None

                # --- interactions ---
                move_env = (move.unsqueeze(2) * env.unsqueeze(1)).reshape(b, 3 * int(C_env), int(P))
                if time_feats is not None:
                    time_env = (time_feats.unsqueeze(2) * env.unsqueeze(1)).reshape(b, 4 * int(C_env), int(P))
                    time_move = (time_feats.unsqueeze(2) * move.unsqueeze(1)).reshape(b, 12, int(P))
                else:
                    time_env = None
                    time_move = None

                # --- pack in canonical order ---
                o = 0
                out[s:e, o : o + int(C_env), :] = env.to(dtype=dtype)
                o += int(C_env)
                out[s:e, o : o + 3, :] = move.to(dtype=dtype)
                o += 3

                if time_feats is not None:
                    out[s:e, o : o + 4, :] = time_feats.to(dtype=dtype)
                    o += 4
                if indiv_feats is not None:
                    out[s:e, o : o + int(n_indiv), :] = indiv_feats.to(dtype=dtype)
                    o += int(n_indiv)

                out[s:e, o : o + (3 * int(C_env)), :] = move_env.to(dtype=dtype)
                o += 3 * int(C_env)

                if time_env is not None and time_move is not None:
                    out[s:e, o : o + (4 * int(C_env)), :] = time_env.to(dtype=dtype)
                    o += 4 * int(C_env)
                    out[s:e, o : o + 12, :] = time_move.to(dtype=dtype)
                    o += 12

                if o != int(c_total):
                    raise RuntimeError(f"Internal error: packed {o} channels, expected {c_total}")

            return out

        # Attempt to build on the requested cache device; on CUDA OOM, fall back to pinned CPU.
        if cache_device.type == "cuda":
            try:
                out = torch.empty((int(N), int(c_total), int(P)), device=cache_device, dtype=dtype)
                out = _build_into(out=out, op_device=cache_device)
            except RuntimeError as e:
                msg = str(e).lower()
                if (not fallback_to_pinned_cpu) or ("out of memory" not in msg):
                    raise
                torch.cuda.empty_cache()
                out = torch.empty((int(N), int(c_total), int(P)), dtype=dtype, pin_memory=True)
                out = _build_into(out=out, op_device=torch.device("cpu"))
        else:
            out = torch.empty((int(N), int(c_total), int(P)), device=cache_device, dtype=dtype)
            out = _build_into(out=out, op_device=cache_device)

        if validate:
            # Spot-check equivalence to __getitem__ on a few indices.
            idxs = [0, int(N) // 2, int(N) - 1] if int(N) >= 3 else list(range(int(N)))
            tol = 1e-4 if dtype == torch.float32 else 1e-1
            for i in idxs:
                slow_x, _slow_valid, _m, _crs = self[int(i)]
                fast_x = out[int(i)].detach().cpu().float()
                max_abs = (slow_x - fast_x).abs().max().item()
                if max_abs > float(tol):
                    raise AssertionError(f"build_expanded_x mismatch at idx={i}: max_abs_diff={max_abs} (tol={tol})")

        return out
