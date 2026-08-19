from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import movebench.bench.config as bench_config
import movebench.bench.evaluation.runner as evaluation_runner
import movebench.bench.executors.ssf_grid as ssf_grid
from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.moveformer import MoveFormerPointStoreDataset
from movebench.features.points import PointStoreReader
from movebench.geometry import (
    LOCAL_GEOMETRY,
    centered_grid_coordinates_m,
    lonlat_from_local_xy_m,
)
from movebench.lib.moveformer.geo_transformers.geo_utils import gcs_to_n_vector
from movebench.lib.moveformer.geo_transformers.training_utils import (
    collate_sequence_dicts,
)
from movebench.models.moveformer import build_moveformer_selection_model


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


def _collate_moveformer(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return collate_sequence_dicts(batch)


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
    reader = PointStoreReader(store_dir=Path(train_store_dir), manifest=train_manifest)
    length_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
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

    valid_lengths = np.concatenate(length_chunks).astype(np.float64, copy=False)
    valid_lengths = valid_lengths[np.isfinite(valid_lengths) & (valid_lengths > 0.0)]
    valid_turns = np.concatenate(turn_chunks).astype(np.float64, copy=False)
    valid_turns = valid_turns[np.isfinite(valid_turns)]
    if len(valid_lengths) == 0:
        raise ValueError("No valid train step lengths available for MoveFormer grid prediction")
    if len(valid_turns) == 0:
        raise ValueError("No valid train turn angles available for MoveFormer grid prediction")

    sampler = dict(getattr(train_manifest, "sampler", {}))
    gamma = (
        dict(sampler.get("gamma", {}))
        if isinstance(sampler.get("gamma"), dict)
        else {}
    )
    gamma_params = gamma.get("params")
    if not (bool(gamma.get("used", False)) and isinstance(gamma_params, list) and gamma_params):
        gamma_params = None
    else:
        gamma_params = tuple(float(x) for x in gamma_params)

    vonmises = (
        dict(sampler.get("vonmises", {}))
        if isinstance(sampler.get("vonmises"), dict)
        else {}
    )
    vonmises_params = vonmises.get("params")
    if not (
        bool(vonmises.get("used", False))
        and isinstance(vonmises_params, list)
        and vonmises_params
    ):
        vonmises_params = None
    else:
        vonmises_params = tuple(float(x) for x in vonmises_params)

    return {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(sampler["num_candidates"]),
        "random_seed": int(sampler["random_seed"]),
        "gamma_used": gamma_params is not None,
        "gamma_params": gamma_params,
        "vonmises_used": vonmises_params is not None,
        "vonmises_params": vonmises_params,
    }


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
    point_ids: torch.Tensor,
    env: torch.Tensor | None,
    valid: torch.Tensor | None,
    mov_vec_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if env is None or valid is None:
        raise ValueError("MoveFormer grid prediction requires environment chips")
    rows_by_point_id = _grid_step_rows_by_point_id(grid_dataset)
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
    split_key: str,
    grid_dataset,
    mov_vec_scale: float,
    step_batch_size: int,
    train_sampler_state: dict[str, Any] | None = None,
    train_store_dir: str | Path | None = None,
    train_manifest=None,
    num_workers: int = 0,
    device: torch.device,
) -> dict[str, object]:
    del split_key
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
    if train_sampler_state is None:
        if train_store_dir is None or train_manifest is None:
            raise ValueError(
                "MoveFormer grid prediction requires train_sampler_state or train_store_dir/train_manifest"
            )
        train_sampler_state = _train_sampler_state_from_point_store(
            train_store_dir=train_store_dir,
            train_manifest=train_manifest,
        )

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


def _completed_steps_from_history(run_dir: Path, *, fallback: int = 0) -> int:
    history_path = Path(run_dir) / "history.json"
    if not history_path.exists():
        return int(fallback)
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return int(fallback)
    steps = [
        int(float(entry["step"]))
        for entry in history_payload
        if isinstance(entry, dict) and "step" in entry
    ]
    return max(steps) if steps else int(fallback)


@dataclass
class _MoveFormerRuntime:
    model: torch.nn.Module
    train_loader: object
    eval_loaders: dict[str, object]
    early_stop_split_key: str
    device: torch.device
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_gamma: float

    early_stop_metric = "nll"

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

    def train_step(self, step: int) -> dict[str, float]:
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
        token_mask = features.get("loss_mask", features["mask"])
        return {
            "loss/train_step": float(loss.item()),
            "num_tokens/train_step": float(token_mask.sum().item()),
        }

    def evaluate(self, split_key: str) -> dict[str, float]:
        return _eval_metrics(
            self.model,
            self.eval_loaders[str(split_key)],
            device=self.device,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "model": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "steps_done": int(self._steps_done),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(dict(state["model"]))
        self.optimizer.load_state_dict(dict(state["optimizer"]))
        self._steps_done = int(state["steps_done"])
        self._set_lr()


def _loader(dataset: Dataset, *, batch_size: int, shuffle: bool, seed: int):
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": 4, #0,
        "collate_fn": _collate_moveformer,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return DataLoader(**kwargs)


def run(*, run_payload, resolved_artifacts, run_dir, tracker):
    run_dir = Path(run_dir)
    method_model = dict(run_payload.get("method_model", {}))
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams["batch_size"])
    prediction_num_workers = int(method_hparams.get("prediction_num_workers", 4))
    max_len = int(method_hparams["max_len"])
    min_len = int(method_hparams["min_len"])
    seed = int(run_payload["seed"])
    early_stop_split_key = str(run_payload["early_stop_split_key"])
    normalize_env = resolved_artifacts.get("normalization_stats_path") is not None
    point_stores = dict(resolved_artifacts["point_stores"])

    dataset_kwargs = {
        "study_dir": resolved_artifacts["study_dir"],
        "trajectories_path": resolved_artifacts["trajectories_path"],
        "plan": resolved_artifacts["plan"],
        "sampler_config": resolved_artifacts["sampler"],
        "max_len": max_len,
        "min_len": min_len,
        "normalize_env": bool(normalize_env),
        "require_normalization_stats": False,
        "subset_band_ids": [str(band_id) for band_id in run_payload["band_ids"]],
    }
    train_dataset = MoveFormerPointStoreDataset(
        split="train",
        context_splits=None,
        candidate_subset_size=16,
        **dataset_kwargs,
    )
    early_stop_dataset = MoveFormerPointStoreDataset(
        split=early_stop_split_key,
        context_splits=["train", early_stop_split_key],
        **dataset_kwargs,
    )
    print("Moveformer train dataset length", len(train_dataset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_moveformer_selection_model(
        cov_dim=int(train_dataset.cov_dim),
        max_len=max_len,
        **method_model,
    )
    train_loader = _loader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    train_eval_loader = _loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    eval_loaders = {
        "train": train_eval_loader,
        early_stop_split_key: _loader(
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
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runtime.model.load_state_dict(dict(checkpoint["state_dict"]))
        best_step = int(dict(checkpoint.get("meta", {})).get("best_step", 0))
        fit = {
            "completed_steps": _completed_steps_from_history(
                run_dir,
                fallback=best_step,
            ),
            "best_step": int(best_step),
        }
    else:
        fit = run_training_loop(
            runtime,
            TrainerConfig(
                max_steps=int(schedule["max_steps"]),
                eval_every_steps=int(schedule["eval_every_steps"]),
                patience_evals=int(schedule["patience_evals"]),
                seed=seed,
            ),
            tracker,
        )
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in runtime.model.state_dict().items()
                },
                "meta": {
                    "method": str(run_payload["method"]),
                    "best_step": int(fit["best_step"]),
                    "covariate_set": str(run_payload["covariate_set"]),
                    "cov_dim": int(train_dataset.cov_dim),
                    "num_candidates_total": int(train_dataset.num_candidates_total),
                },
            },
            checkpoint_path,
        )

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    timescale = bench_config.resolve_timescale(
        str(run_payload["timescale"]),
        study_id=str(run_payload["study_id"]),
    )
    output_hw = tuple(int(v) for v in timescale["output_hw"])
    prediction_dataset_kwargs = dict(dataset_kwargs)
    prediction_dataset_kwargs["min_len"] = 1
    prediction_dataset_kwargs["allow_prediction_only_segments"] = True
    for split_key in run_payload["evaluation_split_keys"]:
        split_key = str(split_key)
        prediction_dataset = MoveFormerPointStoreDataset(
            split=split_key,
            context_splits=["train", split_key],
            **prediction_dataset_kwargs,
        )
        pred_path = pred_dir / f"{split_key}.pt"
        if pred_path.exists():
            try:
                pred_artifact = dict(torch.load(pred_path, map_location="cpu"))
                if str(pred_artifact.get("prediction_kind")) == "grid":
                    evaluation_runner._score_split(
                        split_key=str(split_key),
                        pred_artifact=pred_artifact,
                        run_payload=run_payload,
                    )
                    continue
            except Exception:
                pass
        selected_splits = ssf_grid.selected_splits_from_manifest(
            point_stores[str(split_key)]["manifest"]
        )
        grid_dataset = ssf_grid.make_grid_step_dataset(
            study_dir=resolved_artifacts["study_dir"],
            trajectories_path=resolved_artifacts["trajectories_path"],
            split_key=str(split_key),
            plan=prediction_dataset.plan,
            sampler_config=resolved_artifacts["sampler"],
            output_hw=output_hw,
            selected_splits=selected_splits,
            include_env=True,
        )
        step_batch_size = ssf_grid.scaled_step_batch_size(
            batch_size=batch_size,
            train_points_per_step=max(1, int(prediction_dataset.num_candidates_total)),
            output_hw=output_hw,
        )
        torch.save(
            _grid_prediction_payload(
                runtime.model,
                _loader(
                    prediction_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    seed=seed,
                ),
                split_key=str(split_key),
                grid_dataset=grid_dataset,
                mov_vec_scale=float(train_dataset._mov_vec_scale),
                step_batch_size=step_batch_size,
                train_store_dir=point_stores["train"]["store_dir"],
                train_manifest=point_stores["train"]["manifest"],
                num_workers=prediction_num_workers,
                device=device,
            ),
            pred_path,
        )

    return {
        "status": "completed",
        "train_steps": int(fit["completed_steps"]),
        "best_step": int(fit["best_step"]),
        "early_stop_split_key": early_stop_split_key,
        "train_metrics": _eval_metrics(runtime.model, train_eval_loader, device=device),
        "eval_metrics": _eval_metrics(
            runtime.model,
            _loader(
                early_stop_dataset,
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            ),
            device=device,
        ),
        "schedule": dict(schedule),
    }
