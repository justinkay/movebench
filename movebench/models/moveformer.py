"""MoveBench wrapper for the vendored MoveFormer selection model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm
from x_transformers.x_transformers import AttentionLayers

import movebench.ssf_grid as ssf_grid
from movebench.bench.moveformer_taxa import moveformer_taxon_for_study
from movebench.datasets.chips import grid_dataset
from movebench.datasets.moveformer import (
    MoveFormerPointStoreDataset,
    sequence_dataset,
    sequence_loader,
)
from movebench.datasets.types import SharedTrainingData, StudyTrainingData
from movebench.features.points import PointStoreReader
from movebench.geometry import (
    LOCAL_GEOMETRY,
    centered_grid_coordinates_m,
    lonlat_from_local_xy_m,
)
from movebench.lib.moveformer.geo_transformers.geo_utils import gcs_to_n_vector
from movebench.lib.moveformer.geo_transformers.models.any_horizon_forecast_transformer import (
    AnyHorizonForecastTransformer,
    SelectionHead,
)
from movebench.training import Runtime, TrainerConfig, fit_or_restore, read_checkpoint


def build_moveformer_selection_model(
    *,
    cov_dim: int,
    depth: int = 6,
    heads: int = 8,
    var_len_training_v2: bool = True,
    max_len: int = 500,
    taxon_dim: int = 100,
) -> AnyHorizonForecastTransformer:
    """Instantiate MoveFormer VarCtx-style selection model over generic covariates.

    This matches `movebench/lib/moveformer/config/forecast_mbk+ung_12h_sel_nofut.yaml`
    as closely as possible, except that we treat MoveBench point-store env channels as a
    single continuous covariate vector (`cov` / `cov_cand`) so the baseline works with
    any layer plan.
    """
    if int(cov_dim) <= 0:
        raise ValueError(f"cov_dim must be > 0; got {cov_dim}")

    emb_dim = 128

    cont_feat_dims: Dict[str, int] = {
        "location": 3,
        "location_mov": 2,
        "location_cand": 3,
        "location_cand_mov": 2,
        "time_cont": 10,
        "time_cont_tgt": 10,
        "time_lmt": 2,
        "time_lmt_tgt": 2,
        "dtime_cont": 12,
        "taxon": int(taxon_dim),
        "cov": int(cov_dim),
        "cov_cand": int(cov_dim),
    }

    disc_feat_dims: Dict[str, int] = {
        "time_month": 12,
        "time_month_tgt": 12,
        "time_day": 31,
        "time_day_tgt": 31,
        "time_weekday": 7,
        "time_weekday_tgt": 7,
    }

    keys_in: List[str] = [
        "taxon",
        "location",
        "location_mov",
        "time_month",
        "time_day",
        "time_weekday",
        "time_cont",
        "time_lmt",
        "cov",
        "time_month_tgt",
        "time_day_tgt",
        "time_weekday_tgt",
        "time_cont_tgt",
        "time_lmt_tgt",
        "dtime_cont",
    ]

    encoder = AttentionLayers(
        dim=emb_dim,
        depth=int(depth),
        heads=int(heads),
        causal=True,
    )

    candidate_encoder_layers = [
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
    ]
    prediction_head = SelectionHead(
        candidate_encoder_layers=candidate_encoder_layers,
        embedding_in_keys=[
            "location_cand",
            "location_cand_mov",
            "cov_cand",
        ],
        in_features=128,
        hidden_features=128,
    )

    return AnyHorizonForecastTransformer(
        encoder=encoder,
        decoder=None,
        prediction_head=prediction_head,
        disc_feat_dims=disc_feat_dims,
        cont_feat_dims=cont_feat_dims,
        keys_in=keys_in,
        keys_future=None,
        emb_dim=emb_dim,
        learnable_nan=True,
        concat_enc_to_out=False,
        var_len_training=False,
        var_len_training_v2=bool(var_len_training_v2),
        max_len=int(max_len),
        optimizer_init=None,
        lr_warmup_init=None,
        lr_decay_init=None,
    )


def _batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _eval_metrics(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> dict[str, float]:
    metrics = _eval_moveformer(model, loader, device)
    return {
        "nll": float(metrics["nll"]),
        "nce": float(metrics["nce"]),
        "num_tokens": float(metrics["num_tokens"]),
    }


@lru_cache(maxsize=32)
def _grid_offsets_np(
    *,
    height: int,
    width: int,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_coords_m, y_coords_m = centered_grid_coordinates_m(
        (height, width),
        pixel_size_m,
    )
    x_hw, y_hw = np.meshgrid(x_coords_m, y_coords_m, indexing="xy")
    return x_hw, y_hw


@torch.no_grad()
def _eval_moveformer(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_items = 0
    num_classes = None

    for batch in loader:
        features = {key: value.to(device) for key, value in batch.items()}
        logits, _loss, _metrics = model(features)
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits with shape (B,L,P); got {tuple(logits.shape)}"
            )
        batch_size, _seq_len, num_classes = logits.shape
        if int(num_classes) <= 0:
            raise ValueError(f"P must be > 0; got {num_classes}")
        total_items += int(batch_size)

        mask = features.get("loss_mask", features["mask"]).to(torch.bool)
        target = torch.zeros(
            (int(batch_size), int(mask.shape[1])),
            dtype=torch.long,
            device=device,
        )
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
        valid = int(mask.sum().item())
        if valid == 0:
            continue
        total_nll += float((ce * mask.to(ce.dtype)).sum().item())
        total_tokens += valid

    if total_tokens == 0 or num_classes is None:
        raise ValueError("Loader yielded zero valid tokens; cannot evaluate")

    nll = float(total_nll / float(total_tokens))
    return {
        "nll": nll,
        "nce": float(nll / float(math.log(int(num_classes)))),
        "num_items": float(total_items),
        "num_tokens": float(total_tokens),
    }


def _grid_step_rows_by_point_id(
    grid_dataset,
) -> dict[int, tuple[int, Any]]:
    rows = {}
    for idx, row in enumerate(grid_dataset.chips.step_starts_df.itertuples(index=False)):
        point_id = int(row.point_id_start)
        if point_id in rows:
            raise ValueError(f"Duplicate grid step point_id_start {point_id}")
        rows[point_id] = (idx, row)
    return rows


def _train_sampler_state_from_point_store(
    *,
    train_store_dir: str | Path,
    train_manifest,
) -> dict[str, Any]:
    length_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
    with PointStoreReader(
        store_dir=Path(train_store_dir),
        manifest=train_manifest,
    ) as reader:
        for block in reader.iter_row_blocks(
            band_ids=None,
            shared_keys=["step_length_m", "turn_angle_rad"],
            block_size=2048,
        ):
            length_chunks.append(
                block["step_length_m"][:, -1].to(dtype=torch.float64).cpu().numpy()
            )
            turn_chunks.append(
                block["turn_angle_rad"][:, -1].to(dtype=torch.float64).cpu().numpy()
            )
    if not length_chunks or not turn_chunks:
        raise ValueError("MoveFormer grid prediction requires non-empty train point store")

    return ssf_grid._replay_sampler_state(
        train_manifest=train_manifest,
        step_lengths_m=np.concatenate(length_chunks),
        turn_angles_rad=np.concatenate(turn_chunks),
    )


@torch.no_grad()
def _prediction_queries(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    point_ids: list[torch.Tensor] = []
    queries: list[torch.Tensor] = []
    seen_point_ids: set[int] = set()
    with torch.no_grad():
        for batch in loader:
            features = _batch_to_device(batch, device)
            _embeddings, dec_out = model(features, run_prediction=False)
            prediction_mask = (
                batch["prediction_mask"]
                if "prediction_mask" in batch
                else batch.get("loss_mask", batch["mask"])
            ).to(torch.bool)
            if not bool(prediction_mask.any().item()):
                continue
            batch_point_ids = batch["point_id_start"][prediction_mask].to(torch.int64)
            batch_queries = dec_out[prediction_mask].detach().cpu().to(torch.float32)
            for idx, point_id in enumerate(batch_point_ids.tolist()):
                point_id_i = int(point_id)
                if point_id_i in seen_point_ids:
                    continue
                seen_point_ids.add(point_id_i)
                point_ids.append(torch.tensor([point_id_i], dtype=torch.int64))
                queries.append(batch_queries[idx : idx + 1])
    if not point_ids:
        return (
            torch.zeros((0,), dtype=torch.int64),
            torch.zeros((0, 0), dtype=torch.float32),
        )
    point_id_tensor = torch.cat(point_ids, dim=0)
    return point_id_tensor, torch.cat(queries, dim=0)


def _candidate_feature_embedding(
    model: torch.nn.Module,
    *,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if bool(getattr(model, "learnable_nan", False)):
        nan_fill = getattr(model, "nan_fill", None)
        if nan_fill is not None and key in nan_fill:
            value = torch.where(
                torch.isfinite(value),
                value,
                nan_fill[key].to(dtype=value.dtype),
            )
    return model.cont_proj[key](value)


def _selection_scores(
    model: torch.nn.Module,
    *,
    query_rows: torch.Tensor,
    candidate_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    head = model.prediction_head
    embeddings = [
        _candidate_feature_embedding(model, key=key, value=candidate_features[key])
        for key in head.embedding_in_keys
    ]
    cand_emb = head.combine_emb(torch.stack(embeddings, dim=-1)).squeeze(-1)
    cand_emb = torch.cat(
        [
            cand_emb,
            query_rows.unsqueeze(-2).expand(
                int(query_rows.shape[0]),
                int(cand_emb.shape[1]),
                -1,
            ),
        ],
        dim=-1,
    )
    cand_emb = head.candidate_encoder(cand_emb)
    query = head.query_proj(query_rows)
    scale = float(query.shape[-1]) ** -0.5
    return torch.einsum("ncd,nd->nc", cand_emb, query) * scale


def _grid_candidate_features(
    *,
    grid_dataset,
    rows_by_point_id,
    point_ids: torch.Tensor,
    env: torch.Tensor | None,
    valid: torch.Tensor | None,
    mov_vec_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if env is None or valid is None:
        raise ValueError("MoveFormer grid prediction requires environment chips")
    ordered_point_ids = [int(point_id) for point_id in point_ids.tolist()]
    start_lon = np.asarray(
        [float(rows_by_point_id[point_id][1].start_lon) for point_id in ordered_point_ids],
        dtype=np.float64,
    )
    start_lat = np.asarray(
        [float(rows_by_point_id[point_id][1].start_lat) for point_id in ordered_point_ids],
        dtype=np.float64,
    )

    height, width = (int(grid_dataset.output_hw[0]), int(grid_dataset.output_hw[1]))
    pixel_size_m = float(grid_dataset.pixel_size_m)
    x_hw, y_hw = _grid_offsets_np(
        height=height,
        width=width,
        pixel_size_m=pixel_size_m,
    )
    cand_lon, cand_lat = lonlat_from_local_xy_m(
        start_lon[:, None, None],
        start_lat[:, None, None],
        x_hw[None],
        y_hw[None],
    )
    cand_lon = np.asarray(cand_lon, dtype=np.float64).reshape(
        len(ordered_point_ids),
        height * width,
    )
    cand_lat = np.asarray(cand_lat, dtype=np.float64).reshape(
        len(ordered_point_ids),
        height * width,
    )

    loc_x, loc_y, loc_z = gcs_to_n_vector(cand_lat, cand_lon)
    location_cand = torch.from_numpy(
        np.stack([loc_x, loc_y, loc_z], axis=-1).astype(np.float32, copy=False)
    ).to(device=device)

    candidate_xy_m = np.broadcast_to(
        np.stack([x_hw, y_hw], axis=-1)[None],
        (len(ordered_point_ids), height, width, 2),
    ).reshape(len(ordered_point_ids), height * width, 2)
    location_cand_mov = torch.from_numpy(
        (candidate_xy_m * float(mov_vec_scale)).astype(np.float32, copy=False)
    ).to(device=device)

    env = env.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    valid = valid.to(device=device, dtype=torch.bool, non_blocking=device.type == "cuda")
    cov_cand = env.permute(0, 2, 3, 1).reshape(
        int(env.shape[0]),
        height * width,
        int(env.shape[1]),
    )
    cov_valid = valid.permute(0, 2, 3, 1).reshape_as(cov_cand)
    cov_cand = torch.where(cov_valid, cov_cand, torch.full_like(cov_cand, torch.nan))

    return {
        "location_cand": location_cand,
        "location_cand_mov": location_cand_mov,
        "cov_cand": cov_cand,
    }


def _grid_prediction_payload(
    model: torch.nn.Module,
    loader,
    *,
    grid_dataset,
    mov_vec_scale: float,
    step_batch_size: int,
    train_sampler_state: dict[str, Any],
    num_workers: int = 0,
    device: torch.device,
) -> dict[str, object]:
    point_ids, query_rows = _prediction_queries(model, loader, device=device)
    height, width = (int(grid_dataset.output_hw[0]), int(grid_dataset.output_hw[1]))
    pixel_size_m = float(grid_dataset.pixel_size_m)
    if int(point_ids.shape[0]) == 0:
        return {
            "prediction_kind": "grid",
            "geometry": LOCAL_GEOMETRY,
            "point_id_start": point_ids,
            "scores": torch.zeros((0, height, width), dtype=torch.float32),
            "pixel_size_m": pixel_size_m,
        }

    rows_by_point_id = _grid_step_rows_by_point_id(grid_dataset)
    ordered_point_ids = [int(point_id) for point_id in point_ids.tolist()]
    missing = [point_id for point_id in ordered_point_ids if point_id not in rows_by_point_id]
    if missing:
        raise ValueError(
            f"MoveFormer grid prediction is missing benchmark grid rows for point_id_start values {missing[:5]}"
        )
    if set(rows_by_point_id.keys()) != set(ordered_point_ids):
        raise ValueError("MoveFormer query point_id_start values do not match held-out grid steps")
    ordered_indices = [int(rows_by_point_id[point_id][0]) for point_id in ordered_point_ids]

    model.eval()
    grid_loader = DataLoader(
        Subset(grid_dataset, ordered_indices),
        batch_size=int(step_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=grid_dataset.collate,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(int(num_workers) > 0),
    )
    score_chunks: list[torch.Tensor] = []
    point_id_chunks: list[torch.Tensor] = []
    start = 0
    for grid_batch in tqdm(grid_loader):
        point_id_batch = grid_batch["point_id_start"].to(dtype=torch.int64)
        stop = start + int(point_id_batch.shape[0])
        expected = point_ids[start:stop]
        if not torch.equal(point_id_batch.cpu(), expected.cpu()):
            raise ValueError("MoveFormer grid batch point_id_start order does not match query order")
        log_q_hw, _off_grid_mass = ssf_grid.movement_log_prior_hw(
            start_bearing_rad=grid_batch["start_bearing_rad"],
            output_hw=grid_dataset.output_hw,
            pixel_size_m=float(grid_dataset.pixel_size_m),
            sampler_state=train_sampler_state,
        )
        candidate_features = _grid_candidate_features(
            grid_dataset=grid_dataset,
            rows_by_point_id=rows_by_point_id,
            point_ids=point_id_batch.cpu(),
            env=grid_batch["env"],
            valid=grid_batch["valid"],
            mov_vec_scale=float(mov_vec_scale),
            device=device,
        )
        scores_flat = _selection_scores(
            model,
            query_rows=query_rows[start:stop].to(
                device=device,
                dtype=torch.float32,
                non_blocking=device.type == "cuda",
            ),
            candidate_features=candidate_features,
        )
        scores_flat = scores_flat + log_q_hw.reshape(int(log_q_hw.shape[0]), -1).to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        score_chunks.append(scores_flat.detach().cpu().reshape(-1, height, width))
        point_id_chunks.append(point_id_batch.cpu())
        start = stop

    return {
        "prediction_kind": "grid",
        "geometry": LOCAL_GEOMETRY,
        "point_id_start": torch.cat(point_id_chunks, dim=0),
        "scores": torch.cat(score_chunks, dim=0),
        "pixel_size_m": pixel_size_m,
    }


@dataclass
class _MoveFormerRuntime(Runtime):
    model: torch.nn.Module
    train_loader: object
    eval_loaders: dict[str, object]
    early_stop_split_key: str
    device: torch.device
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_gamma: float

    def __post_init__(self) -> None:
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )
        self._base_lrs = [
            float(group["lr"]) for group in self.optimizer.param_groups
        ]
        self._steps_per_epoch = max(1, len(self.train_loader))
        self._steps_done = 0
        self._set_lr()
        self._train_iter = iter(self.train_loader)

    def _set_lr(self) -> None:
        if int(self.warmup_steps) > 0:
            warmup = min(1.0, float(self._steps_done) / float(self.warmup_steps))
        else:
            warmup = 1.0
        decay = float(self.lr_gamma) ** float(
            self._steps_done // self._steps_per_epoch
        )
        for group, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            group["lr"] = float(base_lr) * warmup * decay

    def _next_batch(self) -> dict[str, torch.Tensor]:
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def _attention_mask(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]] | None:
        seq_len = int(features["mask"].shape[1])
        max_len_cfg = getattr(self.model, "max_len", None)
        max_len = int(max_len_cfg) if max_len_cfg is not None else seq_len
        if bool(getattr(self.model, "var_len_training", False)):
            split_idx = torch.randint(
                low=1, high=max_len + 1, size=(), device=self.device
            )
            split_idx_i = int(split_idx.item())
            if split_idx_i >= seq_len:
                return None
            attn_mask = torch.zeros(
                (seq_len, seq_len),
                dtype=torch.bool,
                device=self.device,
            )
            attn_mask[:split_idx_i, :split_idx_i] = True
            attn_mask[split_idx_i:, split_idx_i:] = True
            return {
                "encoder_kwargs": {"attn_mask": attn_mask},
                "decoder_kwargs": {"attn_mask": attn_mask},
            }
        if bool(getattr(self.model, "var_len_training_v2", False)):
            ctx_len = torch.randint(
                low=1, high=max_len + 1, size=(), device=self.device
            )
            ctx_len_i = int(ctx_len.item())
            n_blocks = (seq_len + ctx_len_i - 1) // ctx_len_i
            attn_mask = torch.eye(n_blocks, dtype=torch.bool, device=self.device)
            attn_mask = attn_mask.repeat_interleave(ctx_len_i, dim=0)
            attn_mask = attn_mask.repeat_interleave(ctx_len_i, dim=1)
            attn_mask = attn_mask[:seq_len, :seq_len]
            return {
                "encoder_kwargs": {"attn_mask": attn_mask},
                "decoder_kwargs": {"attn_mask": attn_mask},
            }
        return None

    def train_step(self, step: int) -> None:
        batch = self._next_batch()
        features = _batch_to_device(batch, self.device)
        kwargs = self._attention_mask(features) or {}
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        _logits, loss, _metrics = self.model(features, **kwargs)
        loss.backward()
        self.optimizer.step()
        self._steps_done += 1
        self._set_lr()

    def evaluate(self, split_key: str) -> dict[str, float]:
        return _eval_metrics(
            self.model,
            self.eval_loaders[str(split_key)],
            device=self.device,
        )


def _limited_subset(
    dataset: Dataset,
    *,
    max_items: int,
    seed: int,
    key: str,
) -> Dataset:
    if len(dataset) <= int(max_items):
        return dataset
    key_seed = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(int(seed) + key_seed)
    indices = rng.choice(len(dataset), size=int(max_items), replace=False)
    return Subset(dataset, [int(index) for index in indices])


def _full_eval_dataset(
    *,
    study: StudyTrainingData,
    max_len: int,
    min_len: int,
    taxon_embedding: np.ndarray,
) -> MoveFormerPointStoreDataset | None:
    for data in study.validation:
        try:
            return sequence_dataset(
                data,
                max_len=max_len,
                min_len=min_len,
                context_splits=["train", data.split_key],
                taxon_embedding=taxon_embedding,
                lazy_rows=True,
            )
        except FileNotFoundError:
            continue
        except ValueError as exc:
            if not str(exc).startswith("No segments found for split="):
                raise
    return None


class MoveFormerMethod:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        method_model = self.model_config
        method_hparams = self.hparams
        batch_size = int(method_hparams["batch_size"])
        max_len = int(method_hparams["max_len"])
        seed = self.seed
        early_stop_split_key = validation.split_key
        train_dataset = sequence_dataset(
            train,
            max_len=max_len,
            min_len=int(method_hparams["min_len"]),
            context_splits=None,
            candidate_subset_size=16,
        )
        early_stop_dataset = sequence_dataset(
            validation,
            max_len=max_len,
            min_len=int(method_hparams["min_len"]),
            context_splits=["train", early_stop_split_key],
        )
        print("Moveformer train dataset length", len(train_dataset))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_moveformer_selection_model(
            cov_dim=int(train_dataset.cov_dim),
            max_len=max_len,
            **method_model,
        )
        train_loader = sequence_loader(
            train_dataset, batch_size=batch_size, shuffle=True, seed=seed
        )
        train_eval_loader = sequence_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        eval_loaders = {
            "train": train_eval_loader,
            early_stop_split_key: sequence_loader(
                early_stop_dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            ),
        }
        schedule = {
            "max_steps": int(method_hparams["max_steps"]),
            "eval_every_steps": int(method_hparams["eval_every_steps"]),
            "patience_evals": int(method_hparams["patience_evals"]),
        }
        runtime = _MoveFormerRuntime(
            model=model,
            train_loader=train_loader,
            eval_loaders=eval_loaders,
            early_stop_split_key=early_stop_split_key,
            device=device,
            lr=float(method_hparams["lr"]),
            weight_decay=float(method_hparams["weight_decay"]),
            warmup_steps=int(method_hparams["warmup_steps"]),
            lr_gamma=float(method_hparams["lr_gamma"]),
        )
        fit = fit_or_restore(
            runtime,
            TrainerConfig(
                max_steps=int(schedule["max_steps"]),
                eval_every_steps=int(schedule["eval_every_steps"]),
                patience_evals=int(schedule["patience_evals"]),
            ),
            work_dir=Path(work_dir),
            restore_from=restore_from,
            schedule=schedule,
        )

        self.model_ = runtime.model
        self.device_ = device
        self.train_ = train
        self.sampler_state_ = None
        self.mov_vec_scale_ = float(train_dataset._mov_vec_scale)
        self.taxon_ = None
        self.training_summary_ = {
            "train_steps": int(fit["completed_steps"]),
            "best_step": int(fit["best_step"]),
            "early_stop_split_key": early_stop_split_key,
            "train_metrics": dict(fit["train_metrics"]),
            "eval_metrics": dict(fit["eval_metrics"]),
            "schedule": dict(fit["schedule"]),
        }
        return self

    def predict(self, data):
        split_key = data.split_key
        hparams = self.hparams
        batch_size = int(hparams["batch_size"])
        if self.sampler_state_ is None:
            store = self.train_.inputs["point_stores"]["train"]
            self.sampler_state_ = _train_sampler_state_from_point_store(
                train_store_dir=store["store_dir"], train_manifest=store["manifest"]
            )
        prediction_dataset = sequence_dataset(
            data,
            max_len=int(hparams["max_len"]),
            min_len=1,
            context_splits=["train", split_key],
            taxon_embedding=self.taxon_,
            lazy_rows=False,
        )
        grid = grid_dataset(data, plan=prediction_dataset.plan, include_env=True)
        step_batch_size = ssf_grid.scaled_step_batch_size(
            batch_size=batch_size,
            train_points_per_step=max(1, int(prediction_dataset.num_candidates_total)),
            output_hw=data.metadata["output_hw"],
        )
        return _grid_prediction_payload(
            self.model_,
            sequence_loader(
                prediction_dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=self.seed,
            ),
            grid_dataset=grid,
            mov_vec_scale=self.mov_vec_scale_,
            step_batch_size=step_batch_size,
            train_sampler_state=self.sampler_state_,
            num_workers=int(hparams.get("prediction_num_workers", 4)),
            device=self.device_,
        )


class SharedMoveFormerTrainer:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        hparams = self.hparams
        seed = self.seed
        max_eval_items = int(hparams["max_eval_items_per_study"])
        candidate_subset_size = int(hparams["candidate_subset_size"])

        train_datasets: list[MoveFormerPointStoreDataset] = []
        train_eval_datasets: list[Dataset] = []
        eval_datasets: list[Dataset] = []
        taxon_rows: list[dict[str, str]] = []
        eval_split_by_study: dict[str, str] = {}
        skipped_eval_study_ids: list[str] = []
        cov_dim: int | None = None
        taxon_dim: int | None = None
        training_data: SharedTrainingData = train
        for study in training_data.studies:
            study_id = study.train.metadata["study_id"]
            taxon, taxon_meta = moveformer_taxon_for_study(study_id)
            train_dataset = sequence_dataset(
                study.train,
                max_len=int(hparams["max_len"]),
                min_len=int(hparams["min_len"]),
                context_splits=None,
                candidate_subset_size=candidate_subset_size,
                taxon_embedding=taxon,
                lazy_rows=True,
            )
            eval_dataset = _full_eval_dataset(
                study=study,
                max_len=int(hparams["max_len"]),
                min_len=int(hparams["min_len"]),
                taxon_embedding=taxon,
            )
            train_datasets.append(train_dataset)
            train_eval_datasets.append(
                _limited_subset(
                    train_dataset,
                    max_items=max_eval_items,
                    seed=seed,
                    key=f"{study_id}:train",
                )
            )
            if eval_dataset is None:
                skipped_eval_study_ids.append(study_id)
            else:
                eval_split_by_study[study_id] = str(eval_dataset.split)
                eval_datasets.append(
                    _limited_subset(
                        eval_dataset,
                        max_items=max_eval_items,
                        seed=seed,
                        key=f"{study_id}:eval",
                    )
                )
            taxon_rows.append({"study_id": study_id, **taxon_meta})
            cov_dim = int(train_dataset.cov_dim) if cov_dim is None else cov_dim
            taxon_dim = int(taxon.shape[0]) if taxon_dim is None else taxon_dim
            if (
                int(train_dataset.cov_dim) != cov_dim
                or int(taxon.shape[0]) != taxon_dim
            ):
                raise ValueError("Incompatible MoveFormer dimensions in shared training group")

        train_dataset = ConcatDataset(train_datasets)
        train_eval_dataset = ConcatDataset(train_eval_datasets)
        if not eval_datasets:
            raise ValueError("No usable validation split for shared MoveFormer training")
        eval_dataset = ConcatDataset(eval_datasets)
        batch_size = int(hparams["batch_size"])
        train_loader = sequence_loader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=0,
        )
        train_eval_loader = sequence_loader(
            train_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        eval_loader = sequence_loader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=0,
        )
        steps_per_epoch = max(1, len(train_loader))
        schedule = {
            "max_epochs": int(hparams["max_epochs"]),
            "evals_per_epoch": int(hparams["evals_per_epoch"]),
            "max_steps": int(hparams["max_epochs"]) * steps_per_epoch,
            "eval_every_steps": max(
                1,
                math.ceil(steps_per_epoch / int(hparams["evals_per_epoch"])),
            ),
            "patience_evals": int(hparams["patience_evals"]),
            "train_steps_per_epoch": steps_per_epoch,
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_moveformer_selection_model(
            cov_dim=int(cov_dim),
            taxon_dim=int(taxon_dim),
            max_len=int(hparams["max_len"]),
            **self.model_config,
        )
        runtime = _MoveFormerRuntime(
            model=model,
            train_loader=train_loader,
            eval_loaders={
                "train": train_eval_loader,
                "shared_eval": eval_loader,
            },
            early_stop_split_key="shared_eval",
            device=device,
            lr=float(hparams["lr"]),
            weight_decay=float(hparams["weight_decay"]),
            warmup_steps=int(hparams["warmup_steps"]),
            lr_gamma=float(hparams["lr_gamma"]),
        )
        fit = fit_or_restore(
            runtime,
            TrainerConfig(
                max_steps=int(schedule["max_steps"]),
                eval_every_steps=int(schedule["eval_every_steps"]),
                patience_evals=int(schedule["patience_evals"]),
            ),
            work_dir=Path(work_dir),
            restore_from=restore_from,
            schedule=schedule,
            model_metadata={
                "shared_training_signature": training_data.training_signature,
                "shared_study_ids": list(training_data.requested_study_ids),
                "shared_data_study_ids": [
                    study.train.metadata["study_id"] for study in training_data.studies
                ],
                "shared_zero_step_studies": list(training_data.zero_step_studies),
                "taxa": taxon_rows,
                "eval_split_by_study": eval_split_by_study,
                "skipped_eval_study_ids": skipped_eval_study_ids,
            },
        )
        self.model_ = runtime.model
        self.training_summary_ = {
            **fit["model_metadata"],
            "train_steps": int(fit["completed_steps"]),
            "best_step": int(fit["best_step"]),
            "early_stop_split_key": "shared_eval",
            "train_metrics": dict(fit["train_metrics"]),
            "eval_metrics": dict(fit["eval_metrics"]),
            "schedule": dict(fit["schedule"]),
        }
        return self


class SharedMoveFormerScorer(MoveFormerMethod):
    def fit(self, train, validation, *, work_dir, restore_from=None):
        if restore_from is None:
            raise ValueError("Shared MoveFormer scoring requires a restoration reference")
        study_id = train.metadata["study_id"]
        taxon, taxon_meta = moveformer_taxon_for_study(study_id)
        train_dataset = sequence_dataset(
            train,
            max_len=int(self.hparams["max_len"]),
            min_len=int(self.hparams["min_len"]),
            context_splits=None,
            candidate_subset_size=int(self.hparams["candidate_subset_size"]),
            taxon_embedding=taxon,
            lazy_rows=True,
        )

        checkpoint = read_checkpoint(restore_from)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_moveformer_selection_model(
            cov_dim=int(train_dataset.cov_dim),
            taxon_dim=int(taxon.shape[0]),
            max_len=int(self.hparams["max_len"]),
            **self.model_config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        self.model_ = model
        self.device_ = device
        self.train_ = train
        self.sampler_state_ = None
        self.mov_vec_scale_ = float(train_dataset._mov_vec_scale)
        self.taxon_ = taxon
        self.training_summary_ = {
            "train_steps": int(checkpoint["completed_steps"]),
            "best_step": int(checkpoint["best_step"]),
            "shared_checkpoint_run_key": restore_from["training_run_key"],
            "taxon": taxon_meta,
        }
        return self
