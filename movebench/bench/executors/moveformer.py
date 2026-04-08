from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import movebench.bench.evaluation.runner as evaluation_runner
from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.moveformer import MoveFormerPointStoreDataset
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


def _prediction_payload(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    point_ids: list[torch.Tensor] = []
    coords: list[torch.Tensor] = []
    probs: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            features = _batch_to_device(batch, device)
            logits, _loss, _metrics = model(features)
            prediction_mask = batch.get("prediction_mask", batch["loss_mask"]).to(
                torch.bool
            )
            if not bool(prediction_mask.any().item()):
                continue
            pseudo_probs = F.softmax(logits[..., 1:], dim=-1).detach().cpu()
            point_ids.append(batch["point_id_start"][prediction_mask].to(torch.int64))
            coords.append(batch["coords_m"][prediction_mask].to(torch.float32))
            probs.append(pseudo_probs[prediction_mask].to(torch.float32))
    return {
        "prediction_kind": "discrete",
        "point_id_start": torch.cat(point_ids, dim=0),
        "coords_m": torch.cat(coords, dim=0),
        "probs": torch.cat(probs, dim=0),
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
        return {
            "loss/train_step": float(loss.item()),
            "num_tokens/train_step": float(features["loss_mask"].sum().item()),
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
        "num_workers": 0,
        "collate_fn": _collate_moveformer,
    }
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        kwargs["generator"] = generator
    return DataLoader(**kwargs)


def run(*, run_payload, resolved_artifacts, run_dir, tracker):
    run_dir = Path(run_dir)
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams["batch_size"])
    max_len = int(method_hparams["max_len"])
    min_len = int(method_hparams["min_len"])
    seed = int(run_payload["seed"])
    early_stop_split_key = str(run_payload["early_stop_split_key"])
    normalize_env = resolved_artifacts.get("normalization_stats_path") is not None

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
        **dataset_kwargs,
    )
    eval_datasets = {
        str(split_key): MoveFormerPointStoreDataset(
            split=str(split_key),
            context_splits=["train", str(split_key)],
            **dataset_kwargs,
        )
        for split_key in run_payload["evaluation_split_keys"]
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_moveformer_selection_model(cov_dim=int(train_dataset.cov_dim))
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
            eval_datasets[early_stop_split_key],
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
    for split_key, dataset in eval_datasets.items():
        pred_path = pred_dir / f"{split_key}.pt"
        if pred_path.exists():
            try:
                evaluation_runner._score_split(
                    split_key=str(split_key),
                    pred_artifact=dict(torch.load(pred_path, map_location="cpu")),
                    run_payload=run_payload,
                )
                continue
            except Exception:
                pass
        torch.save(
            _prediction_payload(
                runtime.model,
                _loader(dataset, batch_size=batch_size, shuffle=False, seed=seed),
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
                eval_datasets[early_stop_split_key],
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            ),
            device=device,
        ),
        "schedule": dict(schedule),
    }
