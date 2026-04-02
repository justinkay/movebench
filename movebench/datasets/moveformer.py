"""MoveFormer dataset wrapper backed by MoveBench point feature stores."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from pyproj import Transformer

from movebench.datasets.ssf import (
    _movement_stats_from_reader,
    _subset_plan_env_channels,
)
from movebench.features.config import SamplerConfig
from movebench.features.normalization import load_points_normalization_mean_std
from movebench.features.points import (
    PointStoreReader,
    load_point_store_manifest,
    resolve_sampler_hash,
)
from movebench.layers.constants import WGS84_CRS
from movebench.layers.plan import LayerPlan
from movebench.lib.moveformer.geo_transformers.trajectory import (
    SimpleTrajectoryProcessor,
)


@lru_cache(maxsize=1)
def _crs_transformers():
    to_3857 = Transformer.from_crs(WGS84_CRS, "EPSG:3857", always_xy=True)
    return to_3857


@dataclass(frozen=True)
class _StepRecord:
    split_key: str
    point_id_start: int
    timestamp_ns: int
    timestamp_end_ns: int
    start_lon: float
    start_lat: float
    env: torch.Tensor
    valid: torch.Tensor
    candidate_lon: torch.Tensor
    candidate_lat: torch.Tensor


def _manifest_selected_splits(manifest) -> list[str]:
    selected_splits = manifest.extraction.get("selected_splits")
    if not isinstance(selected_splits, list) or not selected_splits:
        raise ValueError(
            "Point store manifest is missing extraction.selected_splits required "
            "for MoveFormer context resolution."
        )
    return [str(split_key) for split_key in selected_splits]


def _to_datetime64(ns_values: list[int]) -> np.ndarray:
    return np.asarray(ns_values, dtype=np.int64).astype("datetime64[ns]").astype(
        "datetime64[us]"
    )


def _coords_m(
    *,
    start_lon: np.ndarray,
    start_lat: np.ndarray,
    candidate_lon: torch.Tensor,
    candidate_lat: torch.Tensor,
) -> np.ndarray:
    to_3857 = _crs_transformers()
    start_x, start_y = to_3857.transform(start_lon, start_lat)
    cand_x, cand_y = to_3857.transform(
        candidate_lon.reshape(-1).numpy(),
        candidate_lat.reshape(-1).numpy(),
    )
    coords = np.stack([cand_x, cand_y], axis=-1).astype(np.float32, copy=False)
    coords = coords.reshape(int(candidate_lon.shape[0]), int(candidate_lon.shape[1]), 2)
    coords[:, :, 0] -= np.asarray(start_x, dtype=np.float32).reshape(-1, 1)
    coords[:, :, 1] -= np.asarray(start_y, dtype=np.float32).reshape(-1, 1)
    return coords[:, 1:, :]


def _split_mask(
    chain: list[_StepRecord],
    start: int,
    end: int,
    split_key: str,
    *,
    require_previous_token: bool,
) -> np.ndarray:
    mask = np.asarray(
        [record.split_key == split_key for record in chain[start:end]],
        dtype=np.bool_,
    )
    if len(mask) > 0 and (int(start) > 0 or require_previous_token):
        mask[0] = False
    return mask


def _loss_mask(
    chain: list[_StepRecord], start: int, end: int, split_key: str
) -> np.ndarray:
    return _split_mask(
        chain,
        start,
        end,
        split_key,
        require_previous_token=True,
    )


def _prediction_mask(
    chain: list[_StepRecord], start: int, end: int, split_key: str
) -> np.ndarray:
    return _split_mask(
        chain,
        start,
        end,
        split_key,
        require_previous_token=False,
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
        context_splits: list[str] | None = None,
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
    ) -> None:
        del trajectories_path
        super().__init__()
        if str(sampler_config.type) != "parametric":
            raise ValueError(
                f"MoveFormerPointStoreDataset requires a parametric sampler_config, got {sampler_config.type!r}"
            )

        self.study_dir = Path(study_dir)
        self.split = str(split)
        self.context_splits = list(
            dict.fromkeys(
                [self.split] if context_splits is None else [str(x) for x in context_splits]
            )
        )
        if self.split not in self.context_splits:
            raise ValueError(
                f"context_splits must include target split {self.split!r}, got {self.context_splits!r}"
            )
        self.max_len = int(max_len)
        self.min_len = int(min_len)
        self.normalize_env = bool(normalize_env)
        self.require_normalization_stats = bool(require_normalization_stats)

        _indices, self.plan = _subset_plan_env_channels(
            plan,
            subset_env_layers=None,
            subset_band_ids=subset_band_ids,
        )
        self.cov_dim = int(self.plan.num_channels())
        self._band_ids = self.plan.band_ids_in_order()
        sampler_hash = resolve_sampler_hash(sampler_config)

        target_store_dir, target_manifest = load_point_store_manifest(
            study_dir=self.study_dir,
            split=self.split,
            rasters_hash=str(self.plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )
        self.store_dir = target_store_dir
        self.manifest = target_manifest
        target_selected_splits = _manifest_selected_splits(target_manifest)

        train_store_dir, train_manifest = load_point_store_manifest(
            study_dir=self.study_dir,
            split="train",
            rasters_hash=str(self.plan.rasters_hash),
            sampler_hash=str(sampler_hash),
        )
        train_reader = PointStoreReader(store_dir=train_store_dir, manifest=train_manifest)
        train_stats = _movement_stats_from_reader(train_reader)
        self._mov_vec_scale = 1.0 / float(train_stats["sl_std"])

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if self.normalize_env:
            try:
                self._mean, self._std = load_points_normalization_mean_std(
                    study_dir=self.study_dir,
                    plan=self.plan,
                    sampler_config=sampler_config,
                )
            except FileNotFoundError as exc:
                if self.require_normalization_stats:
                    raise FileNotFoundError(
                        "MoveFormer covariate normalization stats not found. Run "
                        "Study.compute_normalization_vals(from_='points', ...) first, "
                        "or set require_normalization_stats=false."
                    ) from exc
                warnings.warn(
                    "MoveFormer covariate normalization requested, but stats were not found; "
                    "proceeding with raw env values. "
                    "Run Study.compute_normalization_vals(from_='points', ...) to enable normalization.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.normalize_env = False

        self._processor = SimpleTrajectoryProcessor(
            use_target=True,
            use_candidates=True,
            add_dtime=True,
            add_mov_vec=True,
            mov_vec_scale=float(self._mov_vec_scale),
            handle_nan=False,
            pass_through=["taxon", "cov", "cov_cand"],
        )

        self.num_candidates_total = 0
        chains_by_key: dict[tuple[str, int], list[_StepRecord]] = {}
        for split_key in self.context_splits:
            store_dir, manifest = load_point_store_manifest(
                study_dir=self.study_dir,
                split=split_key,
                rasters_hash=str(self.plan.rasters_hash),
                sampler_hash=str(sampler_hash),
            )
            selected_splits = _manifest_selected_splits(manifest)
            if selected_splits != target_selected_splits:
                raise ValueError(
                    "MoveFormer context_splits require point stores extracted from "
                    "the same selected_splits representation, but got "
                    f"{target_selected_splits} for split={self.split!r} and "
                    f"{selected_splits} for context split {split_key!r}."
                )

            reader = PointStoreReader(store_dir=store_dir, manifest=manifest)
            for block in reader.iter_row_blocks(
                band_ids=self._band_ids,
                shared_keys=[
                    "candidate_lon",
                    "candidate_lat",
                    "start_lon",
                    "start_lat",
                    "individual_id",
                    "burst_id",
                    "point_id_start",
                    "timestamp_ns",
                    "timestamp_end_ns",
                ],
                block_size=2048,
            ):
                env = block["env"]
                valid = block["valid"]
                if self.normalize_env:
                    assert self._mean is not None
                    assert self._std is not None
                    env = (env - self._mean.view(1, -1, 1)) / self._std.view(1, -1, 1)

                num_candidates = int(env.shape[2])
                if self.num_candidates_total == 0:
                    self.num_candidates_total = num_candidates
                    self._cand_perm = torch.tensor(
                        [num_candidates - 1] + list(range(num_candidates - 1)),
                        dtype=torch.long,
                    )
                elif num_candidates != self.num_candidates_total:
                    raise ValueError(
                        "All MoveFormer point stores must share the same number of candidates, "
                        f"got {self.num_candidates_total} and {num_candidates}."
                    )

                candidate_lon = block["candidate_lon"].index_select(1, self._cand_perm)
                candidate_lat = block["candidate_lat"].index_select(1, self._cand_perm)
                env = env.index_select(2, self._cand_perm)
                valid = valid.index_select(2, self._cand_perm)
                individual_ids = [str(x) for x in block["individual_id"].tolist()]
                burst_ids = block["burst_id"].to(dtype=torch.int64)

                for idx in range(int(env.shape[0])):
                    chain_key = (individual_ids[idx], int(burst_ids[idx].item()))
                    chains_by_key.setdefault(chain_key, []).append(
                        _StepRecord(
                            split_key=str(split_key),
                            point_id_start=int(block["point_id_start"][idx].item()),
                            timestamp_ns=int(block["timestamp_ns"][idx].item()),
                            timestamp_end_ns=int(block["timestamp_end_ns"][idx].item()),
                            start_lon=float(block["start_lon"][idx].item()),
                            start_lat=float(block["start_lat"][idx].item()),
                            env=env[idx].cpu(),
                            valid=valid[idx].cpu(),
                            candidate_lon=candidate_lon[idx].cpu(),
                            candidate_lat=candidate_lat[idx].cpu(),
                        )
                    )

        if self.num_candidates_total <= 1:
            raise ValueError(
                f"Expected P>=2 candidates; got P={self.num_candidates_total}"
            )

        self._chains: list[list[_StepRecord]] = []
        self._segments: list[tuple[int, int, int]] = []
        stride = max(1, self.max_len - 1)
        for chain in chains_by_key.values():
            chain.sort(key=lambda record: (record.timestamp_ns, record.point_id_start))
            if len(chain) < self.min_len:
                continue
            if not any(record.split_key == self.split for record in chain):
                continue
            chain_idx = len(self._chains)
            self._chains.append(chain)
            start = 0
            while start + self.min_len <= len(chain):
                end = min(len(chain), start + self.max_len)
                if bool(_loss_mask(chain, start, end, self.split).any()):
                    self._segments.append((chain_idx, start, end))
                if end == len(chain):
                    break
                start += stride

        if not self._segments:
            raise ValueError(
                f"No segments found for split={self.split!r} (min_len={self.min_len}, max_len={self.max_len})"
            )

    def __len__(self) -> int:
        return len(self._segments)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        chain_idx, start, end = self._segments[int(idx)]
        chain = self._chains[chain_idx]
        window = chain[start:end]
        seq_env = torch.stack([record.env for record in window], dim=0)
        seq_valid = torch.stack([record.valid for record in window], dim=0)
        cov_cand = seq_env.permute(0, 2, 1).numpy().astype(np.float32, copy=False)
        cov_cand_valid = seq_valid.permute(0, 2, 1).numpy()
        cov_cand = np.where(cov_cand_valid, cov_cand, np.float32(np.nan)).astype(
            np.float32,
            copy=False,
        )

        cov = np.full(
            (len(window), int(self.cov_dim)),
            np.float32(np.nan),
            dtype=np.float32,
        )
        for local_idx in range(len(window)):
            global_idx = start + local_idx
            if global_idx == 0:
                continue
            prev_record = chain[global_idx - 1]
            prev_env = prev_record.env[:, 0].numpy().astype(np.float32, copy=False)
            prev_valid = prev_record.valid[:, 0].numpy()
            cov[local_idx] = np.where(prev_valid, prev_env, np.float32(np.nan)).astype(
                np.float32,
                copy=False,
            )

        start_lon = np.asarray([record.start_lon for record in window], dtype=np.float64)
        start_lat = np.asarray([record.start_lat for record in window], dtype=np.float64)
        candidate_lon = torch.stack([record.candidate_lon for record in window], dim=0)
        candidate_lat = torch.stack([record.candidate_lat for record in window], dim=0)
        traj: dict[str, Any] = {
            "timestamp": _to_datetime64([record.timestamp_ns for record in window]),
            "timestamp_tgt": _to_datetime64(
                [record.timestamp_end_ns for record in window]
            ),
            "location_long": start_lon,
            "location_lat": start_lat,
            "location_long_tgt": candidate_lon[:, 0].numpy().astype(
                np.float64, copy=False
            ),
            "location_lat_tgt": candidate_lat[:, 0].numpy().astype(
                np.float64, copy=False
            ),
            "location_long_cand": candidate_lon.numpy().astype(np.float64, copy=False),
            "location_lat_cand": candidate_lat.numpy().astype(np.float64, copy=False),
            "taxon": np.zeros((len(window), 100), dtype=np.float32),
            "cov": cov,
            "cov_cand": cov_cand,
        }
        encoded = self._processor.encode(traj)
        encoded["loss_mask"] = _loss_mask(chain, start, end, self.split)
        encoded["prediction_mask"] = _prediction_mask(chain, start, end, self.split)
        encoded["point_id_start"] = np.asarray(
            [record.point_id_start for record in window],
            dtype=np.int64,
        )
        encoded["coords_m"] = _coords_m(
            start_lon=start_lon,
            start_lat=start_lat,
            candidate_lon=candidate_lon,
            candidate_lat=candidate_lat,
        ).astype(np.float32, copy=False)
        return encoded
