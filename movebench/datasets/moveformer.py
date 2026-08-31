"""MoveFormer dataset wrapper backed by MoveBench point feature stores."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

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
from movebench.geometry import local_xy_m_from_lonlat
from movebench.layers.plan import LayerPlan
from movebench.lib.moveformer.geo_transformers.trajectory import (
    SimpleTrajectoryProcessor,
)


@dataclass(frozen=True)
class _StepRecord:
    split_key: str
    row_idx: int
    point_id_start: int
    point_id_end: int
    timestamp_ns: int
    timestamp_end_ns: int
    start_lon: float
    start_lat: float
    env: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    candidate_lon: torch.Tensor | None = None
    candidate_lat: torch.Tensor | None = None
    candidate_x_m: torch.Tensor | None = None
    candidate_y_m: torch.Tensor | None = None


@dataclass(frozen=True)
class MoveFormerForecastSeed:
    records: list[_StepRecord]
    processor: SimpleTrajectoryProcessor
    cov_dim: int
    mov_vec_scale: float
    num_candidates_total: int
    point_id_start: int


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
    candidate_x_m: torch.Tensor,
    candidate_y_m: torch.Tensor,
) -> np.ndarray:
    coords = torch.stack([candidate_x_m, candidate_y_m], dim=-1)
    return coords[:, 1:, :].numpy().astype(np.float32, copy=False)


def _set_local_movement_vectors(
    encoded: dict[str, np.ndarray],
    *,
    start_lon: np.ndarray,
    start_lat: np.ndarray,
    candidate_x_m: torch.Tensor,
    candidate_y_m: torch.Tensor,
    mov_vec_scale: float,
) -> None:
    location_mov = np.zeros((len(start_lon), 2), dtype=np.float32)
    if len(start_lon) > 1:
        prev_x_m, prev_y_m = local_xy_m_from_lonlat(
            start_lon[1:],
            start_lat[1:],
            start_lon[:-1],
            start_lat[:-1],
        )
        location_mov[1:] = -np.stack([prev_x_m, prev_y_m], axis=-1)
    candidate_xy_m = torch.stack([candidate_x_m, candidate_y_m], dim=-1).numpy()
    encoded["location_mov"] = location_mov * float(mov_vec_scale)
    encoded["location_tgt_mov"] = (
        candidate_xy_m[:, 0, :] * float(mov_vec_scale)
    ).astype(np.float32, copy=False)
    encoded["location_cand_mov"] = (
        candidate_xy_m * float(mov_vec_scale)
    ).astype(np.float32, copy=False)


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


def _horizon_linked_chains(
    records: list[_StepRecord],
    *,
    step_time_ns: int,
) -> list[list[_StepRecord]]:
    if not records:
        return []

    ordered = sorted(
        records,
        key=lambda record: (record.timestamp_ns, record.point_id_start),
    )
    seen_start_ids: set[int] = set()
    preds_by_end_id: dict[int, list[int]] = {}
    for idx, record in enumerate(ordered):
        if record.point_id_start in seen_start_ids:
            raise ValueError(
                "Duplicate point_id_start "
                f"{record.point_id_start} encountered in MoveFormer context "
                "chain construction"
            )
        seen_start_ids.add(record.point_id_start)
        preds_by_end_id.setdefault(record.point_id_end, []).append(idx)

    predecessor_idx = [-1] * len(ordered)
    successor_idx = [-1] * len(ordered)
    for idx, record in enumerate(ordered):
        desired_prev_ts = int(record.timestamp_ns) - int(step_time_ns)
        candidates = [
            cand_idx
            for cand_idx in preds_by_end_id.get(record.point_id_start, [])
            if ordered[cand_idx].timestamp_ns < record.timestamp_ns
        ]
        if not candidates:
            continue
        best_pred_idx = min(
            candidates,
            key=lambda cand_idx: (
                abs(int(ordered[cand_idx].timestamp_ns) - desired_prev_ts),
                -int(ordered[cand_idx].timestamp_ns),
                int(ordered[cand_idx].point_id_start),
            ),
        )
        predecessor_idx[idx] = int(best_pred_idx)
        if successor_idx[best_pred_idx] != -1:
            raise ValueError(
                "MoveFormer context chain construction produced multiple "
                "successors for one horizon step"
            )
        successor_idx[best_pred_idx] = int(idx)

    chains: list[list[_StepRecord]] = []
    visited: set[int] = set()
    for root_idx, pred_idx in enumerate(predecessor_idx):
        if pred_idx != -1:
            continue
        chain: list[_StepRecord] = []
        current_idx = int(root_idx)
        while current_idx != -1 and current_idx not in visited:
            visited.add(current_idx)
            chain.append(ordered[current_idx])
            current_idx = successor_idx[current_idx]
        if chain:
            chains.append(chain)

    for start_idx in range(len(ordered)):
        if start_idx in visited:
            continue
        chain = []
        current_idx = int(start_idx)
        local_seen: set[int] = set()
        while current_idx != -1 and current_idx not in local_seen:
            local_seen.add(current_idx)
            visited.add(current_idx)
            chain.append(ordered[current_idx])
            current_idx = successor_idx[current_idx]
        if chain:
            chains.append(chain)

    return chains


def build_moveformer_forecast_seed(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_config: SamplerConfig,
    split_key: str,
    point_id_start: int,
    subset_band_ids: list[str],
    normalize_env: bool,
    max_len: int = 500,
) -> MoveFormerForecastSeed:
    dataset = MoveFormerPointStoreDataset(
        study_dir=study_dir,
        trajectories_path=Path(study_dir) / "trajectories.csv",
        plan=plan,
        sampler_config=sampler_config,
        split=str(split_key),
        context_splits=["train", str(split_key)],
        max_len=int(max_len),
        min_len=1,
        normalize_env=bool(normalize_env),
        require_normalization_stats=False,
        subset_band_ids=list(subset_band_ids),
        allow_prediction_only_segments=True,
    )
    for chain in dataset._chains:
        for idx, record in enumerate(chain):
            if (
                str(record.split_key) == str(split_key)
                and int(record.point_id_start) == int(point_id_start)
            ):
                return MoveFormerForecastSeed(
                    records=list(chain[: idx + 1]),
                    processor=dataset._processor,
                    cov_dim=int(dataset.cov_dim),
                    mov_vec_scale=float(dataset._mov_vec_scale),
                    num_candidates_total=int(dataset.num_candidates_total),
                    point_id_start=int(point_id_start),
                )
    raise ValueError(
        f"MoveFormer context did not contain point_id_start={int(point_id_start)} "
        f"for split={split_key!r}"
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
        min_len: int = 1,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
        candidate_subset_size: int | None = None,
        allow_prediction_only_segments: bool = False,
        taxon_embedding: np.ndarray | torch.Tensor | list[float] | None = None,
        lazy_rows: bool = False,
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
        self.candidate_subset_size = (
            None if candidate_subset_size is None else int(candidate_subset_size)
        )
        self.allow_prediction_only_segments = bool(allow_prediction_only_segments)
        self.lazy_rows = bool(lazy_rows)
        self._taxon_embedding = np.asarray(
            np.zeros(100, dtype=np.float32)
            if taxon_embedding is None
            else taxon_embedding,
            dtype=np.float32,
        )
        if self._taxon_embedding.ndim != 1:
            raise ValueError("taxon_embedding must be one-dimensional")

        _indices, self.plan = _subset_plan_env_channels(
            plan,
            subset_env_layers=None,
            subset_band_ids=subset_band_ids,
        )
        self.cov_dim = int(self.plan.num_channels())
        self._band_ids = self.plan.band_ids_in_order()
        self._step_time_ns = int(self.plan.step_spec.step_time_s) * 1_000_000_000
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
        with PointStoreReader(
            store_dir=train_store_dir, manifest=train_manifest
        ) as train_reader:
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
            add_mov_vec=False,
            mov_vec_scale=float(self._mov_vec_scale),
            handle_nan=False,
            pass_through=["taxon", "cov", "cov_cand"],
        )

        self.num_candidates_total = 0
        self._reader_spec_by_split: dict[str, tuple[Path, Any]] = {}
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

            self._reader_spec_by_split[str(split_key)] = (store_dir, manifest)
            with PointStoreReader(store_dir=store_dir, manifest=manifest) as reader:
                num_candidates = int(reader.points_per_step)
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

                shared_keys = [
                    "start_lon",
                    "start_lat",
                    "individual_id",
                    "burst_id",
                    "point_id_start",
                    "point_id_end",
                    "timestamp_ns",
                    "timestamp_end_ns",
                ]
                if not self.lazy_rows:
                    shared_keys = [
                        "candidate_lon",
                        "candidate_lat",
                        "candidate_x_m",
                        "candidate_y_m",
                        *shared_keys,
                    ]
                for block in reader.iter_row_blocks(
                    band_ids=None if self.lazy_rows else self._band_ids,
                    shared_keys=shared_keys,
                    block_size=2048,
                ):
                    env = None
                    valid = None
                    candidate_lon = None
                    candidate_lat = None
                    candidate_x_m = None
                    candidate_y_m = None
                    if not self.lazy_rows:
                        env = block["env"]
                        valid = block["valid"]
                        if self.normalize_env:
                            assert self._mean is not None
                            assert self._std is not None
                            env = (
                                env - self._mean.view(1, -1, 1)
                            ) / self._std.view(1, -1, 1)
                        candidate_lon = block["candidate_lon"].index_select(
                            1, self._cand_perm
                        )
                        candidate_lat = block["candidate_lat"].index_select(
                            1, self._cand_perm
                        )
                        candidate_x_m = block["candidate_x_m"].index_select(
                            1, self._cand_perm
                        )
                        candidate_y_m = block["candidate_y_m"].index_select(
                            1, self._cand_perm
                        )
                        env = env.index_select(2, self._cand_perm)
                        valid = valid.index_select(2, self._cand_perm)
                    individual_ids = [str(x) for x in block["individual_id"].tolist()]
                    burst_ids = block["burst_id"].to(dtype=torch.int64)

                    for idx in range(int(block["row_idxs"].shape[0])):
                        chain_key = (individual_ids[idx], int(burst_ids[idx].item()))
                        chains_by_key.setdefault(chain_key, []).append(
                            _StepRecord(
                                split_key=str(split_key),
                                row_idx=int(block["row_idxs"][idx].item()),
                                point_id_start=int(block["point_id_start"][idx].item()),
                                point_id_end=int(block["point_id_end"][idx].item()),
                                timestamp_ns=int(block["timestamp_ns"][idx].item()),
                                timestamp_end_ns=int(
                                    block["timestamp_end_ns"][idx].item()
                                ),
                                start_lon=float(block["start_lon"][idx].item()),
                                start_lat=float(block["start_lat"][idx].item()),
                                env=None if env is None else env[idx].cpu(),
                                valid=None if valid is None else valid[idx].cpu(),
                                candidate_lon=(
                                    None
                                    if candidate_lon is None
                                    else candidate_lon[idx].cpu()
                                ),
                                candidate_lat=(
                                    None
                                    if candidate_lat is None
                                    else candidate_lat[idx].cpu()
                                ),
                                candidate_x_m=(
                                    None
                                    if candidate_x_m is None
                                    else candidate_x_m[idx].cpu()
                                ),
                                candidate_y_m=(
                                    None
                                    if candidate_y_m is None
                                    else candidate_y_m[idx].cpu()
                                ),
                            )
                        )

        if self.num_candidates_total <= 1:
            raise ValueError(
                f"Expected P>=2 candidates; got P={self.num_candidates_total}"
            )
        if self.candidate_subset_size is not None:
            if self.candidate_subset_size < 2:
                raise ValueError(
                    "candidate_subset_size must be >= 2 to include the true endpoint "
                    f"plus at least one negative, got {self.candidate_subset_size}."
                )
            if self.candidate_subset_size > self.num_candidates_total:
                raise ValueError(
                    "candidate_subset_size cannot exceed stored candidate count, got "
                    f"{self.candidate_subset_size} > {self.num_candidates_total}."
                )
            self.num_candidates_total = int(self.candidate_subset_size)

        self._chains: list[list[_StepRecord]] = []
        self._segments: list[tuple[int, int, int]] = []
        stride = max(1, self.max_len - 1)
        for burst_records in chains_by_key.values():
            for chain in _horizon_linked_chains(
                burst_records,
                step_time_ns=self._step_time_ns,
            ):
                if len(chain) < self.min_len:
                    continue
                if not any(record.split_key == self.split for record in chain):
                    continue
                chain_idx = len(self._chains)
                self._chains.append(chain)
                start = 0
                while start + self.min_len <= len(chain):
                    end = min(len(chain), start + self.max_len)
                    has_split_rows = bool(
                        any(
                            record.split_key == self.split
                            for record in chain[start:end]
                        )
                    )
                    if has_split_rows:
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

    def _load_rows(
        self,
        split_key: str,
        row_idxs: list[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        store_dir, manifest = self._reader_spec_by_split[str(split_key)]
        with PointStoreReader(store_dir=store_dir, manifest=manifest) as reader:
            env, valid = reader.load_env_valid_rows(
                band_ids=self._band_ids,
                row_idxs=row_idxs,
            )
            shared = reader.load_shared_rows(
                keys=[
                    "candidate_lon",
                    "candidate_lat",
                    "candidate_x_m",
                    "candidate_y_m",
                ],
                row_idxs=row_idxs,
            )
        if self.normalize_env:
            assert self._mean is not None
            assert self._std is not None
            env = (env - self._mean.view(1, -1, 1)) / self._std.view(1, -1, 1)
        return (
            env.index_select(2, self._cand_perm),
            valid.index_select(2, self._cand_perm),
            torch.as_tensor(shared["candidate_lon"]).index_select(1, self._cand_perm),
            torch.as_tensor(shared["candidate_lat"]).index_select(1, self._cand_perm),
            torch.as_tensor(shared["candidate_x_m"]).index_select(1, self._cand_perm),
            torch.as_tensor(shared["candidate_y_m"]).index_select(1, self._cand_perm),
        )

    def _row_payload(
        self,
        record: _StepRecord,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        values = (
            record.env,
            record.valid,
            record.candidate_lon,
            record.candidate_lat,
            record.candidate_x_m,
            record.candidate_y_m,
        )
        if all(value is not None for value in values):
            return tuple(values)  # type: ignore[return-value]
        return tuple(
            value[0]
            for value in self._load_rows(
                str(record.split_key),
                [int(record.row_idx)],
            )
        )

    def _window_payload(
        self,
        window: list[_StepRecord],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if not self.lazy_rows:
            rows = [self._row_payload(record) for record in window]
            return tuple(
                torch.stack([row[field_idx] for row in rows], dim=0)
                for field_idx in range(6)
            )

        fields: list[list[torch.Tensor | None]] = [
            [None] * len(window) for _ in range(6)
        ]
        refs_by_split: dict[str, list[tuple[int, int]]] = {}
        for local_idx, record in enumerate(window):
            refs_by_split.setdefault(str(record.split_key), []).append(
                (local_idx, int(record.row_idx))
            )
        for split_key, refs in refs_by_split.items():
            batch = self._load_rows(
                split_key,
                [row_idx for _local_idx, row_idx in refs],
            )
            for field_idx, values in enumerate(batch):
                for batch_idx, (local_idx, _row_idx) in enumerate(refs):
                    fields[field_idx][local_idx] = values[batch_idx]
        if any(value is None for field in fields for value in field):
            raise ValueError("MoveFormer lazy row loading failed to populate a window")
        return tuple(
            torch.stack([value for value in field if value is not None], dim=0)
            for field in fields
        )

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        chain_idx, start, end = self._segments[int(idx)]
        chain = self._chains[chain_idx]
        window = chain[start:end]
        (
            seq_env,
            seq_valid,
            candidate_lon,
            candidate_lat,
            candidate_x_m,
            candidate_y_m,
        ) = self._window_payload(window)
        if self.candidate_subset_size is not None:
            num_negatives = int(self.candidate_subset_size) - 1
            negative_idx = torch.randperm(int(seq_env.shape[2]) - 1)[:num_negatives] + 1
            keep_idx = torch.cat(
                [torch.zeros(1, dtype=torch.long), negative_idx.to(dtype=torch.long)],
                dim=0,
            )
            seq_env = seq_env.index_select(2, keep_idx)
            seq_valid = seq_valid.index_select(2, keep_idx)
            candidate_lon = candidate_lon.index_select(1, keep_idx)
            candidate_lat = candidate_lat.index_select(1, keep_idx)
            candidate_x_m = candidate_x_m.index_select(1, keep_idx)
            candidate_y_m = candidate_y_m.index_select(1, keep_idx)
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
        if start > 0:
            prev_env, prev_valid, *_rest = self._row_payload(chain[start - 1])
            cov[0] = np.where(
                prev_valid[:, 0].numpy(),
                prev_env[:, 0].numpy(),
                np.float32(np.nan),
            ).astype(np.float32, copy=False)
        if len(window) > 1:
            prev_env = seq_env[:-1, :, 0].numpy()
            prev_valid = seq_valid[:-1, :, 0].numpy()
            cov[1:] = np.where(prev_valid, prev_env, np.float32(np.nan)).astype(
                np.float32,
                copy=False,
            )

        start_lon = np.asarray(
            [record.start_lon for record in window], dtype=np.float64
        )
        start_lat = np.asarray(
            [record.start_lat for record in window], dtype=np.float64
        )
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
            "taxon": np.tile(self._taxon_embedding, (len(window), 1)),
            "cov": cov,
            "cov_cand": cov_cand,
        }
        encoded = self._processor.encode(traj)
        _set_local_movement_vectors(
            encoded,
            start_lon=start_lon,
            start_lat=start_lat,
            candidate_x_m=candidate_x_m,
            candidate_y_m=candidate_y_m,
            mov_vec_scale=self._mov_vec_scale,
        )
        encoded["prediction_mask"] = _prediction_mask(chain, start, end, self.split)
        encoded["point_id_start"] = np.asarray(
            [record.point_id_start for record in window],
            dtype=np.int64,
        )
        encoded["coords_m"] = _coords_m(
            candidate_x_m=candidate_x_m,
            candidate_y_m=candidate_y_m,
        )
        return encoded
