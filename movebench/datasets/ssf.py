"""SSF/iSSF datasets backed by extracted point feature stores."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench.features.config import SamplerConfig
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import (
    PointStoreReader,
    load_point_store_manifest,
    resolve_sampler_hash,
)
from movebench.layers.plan import Channel, LayerPlan

TARGET_CRS = "EPSG:3857"


def _compute_movement_stats(
    step_length_m: torch.Tensor, turn_angle_rad: torch.Tensor
) -> dict[str, float]:
    sl = step_length_m[:, -1].to(dtype=torch.float64)
    ta = turn_angle_rad[:, -1].to(dtype=torch.float64)
    log_sl = torch.log(sl + 1e-8)
    cos_ta = torch.cos(ta)

    def _mean_std(x: torch.Tensor) -> tuple[float, float]:
        mean = float(x.mean().item())
        std = float(x.std(unbiased=False).item())
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        return mean, std

    sl_mean, sl_std = _mean_std(sl)
    log_sl_mean, log_sl_std = _mean_std(log_sl)
    cos_ta_mean, cos_ta_std = _mean_std(cos_ta)
    return {
        "sl_mean": sl_mean,
        "sl_std": sl_std,
        "log_sl_mean": log_sl_mean,
        "log_sl_std": log_sl_std,
        "cos_ta_mean": cos_ta_mean,
        "cos_ta_std": cos_ta_std,
    }


def _movement_stats_from_reader(reader: PointStoreReader) -> dict[str, float]:
    sl_sum = 0.0
    sl_sumsq = 0.0
    log_sum = 0.0
    log_sumsq = 0.0
    cos_sum = 0.0
    cos_sumsq = 0.0
    count = 0
    for block in reader.iter_row_blocks(
        band_ids=None,
        shared_keys=["step_length_m", "turn_angle_rad"],
        block_size=2048,
    ):
        sl = block["step_length_m"][:, -1].to(dtype=torch.float64)
        ta = block["turn_angle_rad"][:, -1].to(dtype=torch.float64)
        log_sl = torch.log(sl + 1e-8)
        cos_ta = torch.cos(ta)
        sl_sum += float(sl.sum().item())
        sl_sumsq += float((sl * sl).sum().item())
        log_sum += float(log_sl.sum().item())
        log_sumsq += float((log_sl * log_sl).sum().item())
        cos_sum += float(cos_ta.sum().item())
        cos_sumsq += float((cos_ta * cos_ta).sum().item())
        count += int(sl.shape[0])
    if count == 0:
        raise ValueError("Cannot compute movement stats from an empty point store")

    def _finalize(total: float, total_sq: float) -> tuple[float, float]:
        mean = total / float(count)
        var = (total_sq / float(count)) - (mean * mean)
        std = math.sqrt(max(var, 0.0))
        if not math.isfinite(std) or std < 1e-8:
            std = 1.0
        return float(mean), float(std)

    sl_mean, sl_std = _finalize(sl_sum, sl_sumsq)
    log_sl_mean, log_sl_std = _finalize(log_sum, log_sumsq)
    cos_ta_mean, cos_ta_std = _finalize(cos_sum, cos_sumsq)
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
        present_layers = {channel.layer for channel in plan.channels}
        missing = sorted(want - present_layers)
        if missing:
            raise ValueError(
                f"subset_env_layers contains unknown layer(s): {missing}. Available: {sorted(present_layers)}"
            )
        indices = [i for i, channel in enumerate(plan.channels) if channel.layer in want]
    else:
        assert subset_band_ids is not None
        want = {str(x) for x in subset_band_ids}
        present_band_ids = set(plan.band_ids_in_order())
        missing = sorted(want - present_band_ids)
        if missing:
            raise ValueError(f"subset_band_ids contains unknown band_id(s): {missing}")
        indices = [
            i for i, band_id in enumerate(plan.band_ids_in_order()) if band_id in want
        ]

    if len(indices) == int(plan.num_channels()):
        return indices, plan

    channels: list[Channel] = [plan.channels[i] for i in indices]
    layers_in_order: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        if channel.layer in seen:
            continue
        seen.add(channel.layer)
        layers_in_order.append(channel.layer)
    return indices, LayerPlan(
        rasters_hash=str(plan.rasters_hash),
        step=plan.step,
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups={k: dict(v) for k, v in plan.chip_groups.items()},
        derived_dependencies={k: list(v) for k, v in plan.derived_dependencies.items()},
    )


class SSFDataset(Dataset):
    """Empirical SSF dataset (env + valid mask) loaded from a point feature store."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        plan: LayerPlan,
        sampler_config: SamplerConfig,
        split: str,
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
    ) -> None:
        if str(sampler_config.type) != "empirical":
            raise ValueError(
                f"SSFDataset requires an empirical sampler_config, got {sampler_config.type!r}"
            )
        self.study_dir = Path(study_dir)
        self.split = str(split)
        _indices, self.plan = _subset_plan_env_channels(
            plan,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
        )
        sampler_hash = resolve_sampler_hash(sampler_config)
        store_dir, manifest = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )
        self.store_dir = store_dir
        self.manifest = manifest
        self.sampler_hash = str(manifest.sampler_hash)
        self.normalize = bool(normalize)
        self.reader = PointStoreReader(store_dir=store_dir, manifest=manifest)
        self.c_env = int(self.plan.num_channels())

        all_rows = torch.arange(int(manifest.n_rows), dtype=torch.long)
        self.env, self.valid = self.reader.load_env_valid_rows(
            band_ids=self.plan.band_ids_in_order(),
            row_idxs=all_rows,
        )
        shared = self.reader.load_shared_rows(
            keys=["step_length_m", "turn_angle_rad", "point_id_start", "timestamp_ns"],
            row_idxs=all_rows,
        )
        self.step_length_m = shared["step_length_m"]
        self.turn_angle_rad = shared["turn_angle_rad"]
        self.point_id_start = shared["point_id_start"]
        self.timestamp_ns = shared["timestamp_ns"]

        if self.normalize:
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_config=sampler_config,
            )
        else:
            self._mean = None
            self._std = None

    def __len__(self) -> int:
        return int(self.manifest.n_rows)

    def _get_env_valid(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env[int(idx)]
        valid = self.valid[int(idx)]
        if not self.normalize:
            return env, valid
        assert self._mean is not None
        assert self._std is not None
        env = (env - self._mean.view(-1, 1)) / self._std.view(-1, 1)
        env = torch.where(valid, env, torch.zeros_like(env))
        return env, valid

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
        env, valid = self._get_env_valid(idx)
        return env, valid, {}, TARGET_CRS


class iSSFDataset(Dataset):
    """Parametric iSSF dataset that loads only the requested batch rows."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        sampler_config: SamplerConfig,
        split: str,
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
        normalize_movement: bool = True,
        include_time: bool = False,
        include_individual_id: bool = False,
        movement_stats: dict[str, float] | None = None,
    ) -> None:
        del trajectories_path
        if str(sampler_config.type) != "parametric":
            raise ValueError(
                f"iSSFDataset requires a parametric sampler_config, got {sampler_config.type!r}"
            )
        self.study_dir = Path(study_dir)
        self.split = str(split)
        _indices, self.plan = _subset_plan_env_channels(
            plan,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
        )
        sampler_hash = resolve_sampler_hash(sampler_config)
        store_dir, manifest = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )
        self.store_dir = store_dir
        self.manifest = manifest
        self.reader = PointStoreReader(store_dir=store_dir, manifest=manifest)
        self.sampler_hash = str(manifest.sampler_hash)
        self.normalize = bool(normalize)
        self.normalize_movement = bool(normalize_movement)
        self.include_time = bool(include_time)
        self.include_individual_id = bool(include_individual_id)
        self.c_env = int(self.plan.num_channels())
        self._band_ids = self.plan.band_ids_in_order()

        if self.normalize:
            self._mean, self._std = load_points_normalization_mean_std(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_config=sampler_config,
            )
        else:
            self._mean = None
            self._std = None

        if self.normalize_movement:
            if movement_stats is not None:
                self._movement_stats = dict(movement_stats)
            else:
                train_store_dir, train_manifest = load_point_store_manifest(
                    study_dir=self.study_dir,
                    split="train",
                    rasters_hash=str(plan.rasters_hash),
                    sampler_hash=str(sampler_hash),
                )
                self._movement_stats = _movement_stats_from_reader(
                    PointStoreReader(store_dir=train_store_dir, manifest=train_manifest)
                )
        else:
            self._movement_stats = None

        if self.include_individual_id:
            train_store_dir, train_manifest = load_point_store_manifest(
                study_dir=self.study_dir,
                split="train",
                rasters_hash=str(plan.rasters_hash),
                sampler_hash=str(sampler_hash),
            )
            train_reader = PointStoreReader(
                store_dir=train_store_dir,
                manifest=train_manifest,
            )
            train_ids = train_reader.load_shared_rows(
                keys=["individual_id"],
                row_idxs=torch.arange(train_manifest.n_rows, dtype=torch.long),
            )["individual_id"]
            individuals = sorted({str(individual_id) for individual_id in train_ids.tolist()})
            self._individual_to_idx = {
                individual_id: idx for idx, individual_id in enumerate(individuals)
            }
            self._n_individuals = int(len(individuals))
        else:
            self._individual_to_idx = {}
            self._n_individuals = 0

    @property
    def individual_mapping(self) -> dict[str, int]:
        return dict(self._individual_to_idx)

    @property
    def n_individuals(self) -> int:
        return int(self._n_individuals)

    def __len__(self) -> int:
        return int(self.manifest.n_rows)

    def _movement_tensor(
        self,
        *,
        step_length_m: torch.Tensor,
        turn_angle_rad: torch.Tensor,
    ) -> torch.Tensor:
        move = torch.stack(
            [
                step_length_m.to(dtype=torch.float32),
                torch.log(step_length_m.to(dtype=torch.float32) + 1e-8),
                torch.cos(turn_angle_rad.to(dtype=torch.float32)),
            ],
            dim=1,
        )
        if self._movement_stats is None:
            return move
        stats = self._movement_stats
        move[:, 0, :] = (move[:, 0, :] - float(stats["sl_mean"])) / float(
            stats["sl_std"]
        )
        move[:, 1, :] = (move[:, 1, :] - float(stats["log_sl_mean"])) / float(
            stats["log_sl_std"]
        )
        move[:, 2, :] = (move[:, 2, :] - float(stats["cos_ta_mean"])) / float(
            stats["cos_ta_std"]
        )
        return move

    def _time_tensor(self, timestamp_ns: torch.Tensor) -> torch.Tensor:
        ts = pd.to_datetime(timestamp_ns.detach().cpu().numpy(), unit="ns", utc=True)
        doy = torch.from_numpy(ts.dayofyear.to_numpy()).to(dtype=torch.float32)
        hour = torch.from_numpy(ts.hour.to_numpy()).to(dtype=torch.float32)
        minute = torch.from_numpy(ts.minute.to_numpy()).to(dtype=torch.float32)
        second = torch.from_numpy(ts.second.to_numpy()).to(dtype=torch.float32)
        hour_frac = hour + (minute / 60.0) + (second / 3600.0)
        two_pi = float(2.0 * math.pi)
        return torch.stack(
            [
                torch.sin(two_pi * doy / 365.25),
                torch.cos(two_pi * doy / 365.25),
                torch.sin(two_pi * hour_frac / 24.0),
                torch.cos(two_pi * hour_frac / 24.0),
            ],
            dim=1,
        ).contiguous()

    def _individual_idx_tensor(self, individual_ids: Any) -> torch.Tensor:
        indices: list[int] = []
        for individual_id in individual_ids.tolist():
            key = str(individual_id)
            if key not in self._individual_to_idx:
                raise KeyError(
                    f"individual_id={key!r} was not present in the training split mapping"
                )
            indices.append(int(self._individual_to_idx[key]))
        return torch.tensor(indices, dtype=torch.long)

    def load_batch(
        self,
        indices: list[int] | torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        idxs = torch.as_tensor(indices, dtype=torch.long)
        env, valid = self.reader.load_env_valid_rows(
            band_ids=self._band_ids,
            row_idxs=idxs,
        )
        if self.normalize:
            assert self._mean is not None
            assert self._std is not None
            env = (env - self._mean.view(1, -1, 1)) / self._std.view(1, -1, 1)
        env = torch.where(valid, env, torch.zeros_like(env))

        shared_keys = ["step_length_m", "turn_angle_rad"]
        if self.include_time:
            shared_keys.append("timestamp_ns")
        if self.include_individual_id:
            shared_keys.append("individual_id")
        shared = self.reader.load_shared_rows(keys=shared_keys, row_idxs=idxs)
        batch: dict[str, torch.Tensor | None] = {
            "env": env,
            "valid": valid,
            "move": self._movement_tensor(
                step_length_m=shared["step_length_m"],
                turn_angle_rad=shared["turn_angle_rad"],
            ),
            "time": None,
            "individual_idx": None,
        }
        if self.include_time:
            batch["time"] = self._time_tensor(shared["timestamp_ns"])
        if self.include_individual_id:
            batch["individual_idx"] = self._individual_idx_tensor(shared["individual_id"])
        return batch

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def collate(self, indices: list[int] | torch.Tensor) -> dict[str, torch.Tensor | None]:
        return self.load_batch(indices)
