from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from movebench.bench_v2.train import TrainerConfig, run_training_loop
from movebench.datasets.deepssf_wds_cache import make_deepssf_wds_loader
from movebench.models.deepssf import DeepSSF

def _slice_spatial(
    spatial: torch.Tensor,
    *,
    channel_indices: torch.Tensor | None,
) -> torch.Tensor:
    if channel_indices is None:
        return spatial
    return spatial.index_select(1, channel_indices)


def _slice_batch_items(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(t[: int(num_items)] for t in batch)  # type: ignore[return-value]


def _grid_logits_and_targets(
    out: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if out.ndim != 4 or int(out.shape[-1]) != 2:
        raise ValueError(
            f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
        )
    scores = out[..., 0] + out[..., 1]
    batch_size, height, width = scores.shape
    logits = scores.view(int(batch_size), int(height) * int(width))
    target_flat = target.view(int(batch_size), int(height) * int(width))
    target_mass = target_flat.sum(dim=1)
    valid = (target_mass > 0.5) & (target_mass < 1.5)
    if not valid.any().item():
        return None
    return logits[valid], torch.argmax(target_flat[valid], dim=1)


@torch.no_grad()
def _evaluate_loader(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    channel_indices: torch.Tensor | None,
    max_items: int | None = None,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_valid = 0
    seen_items = 0
    item_limit = None if max_items is None else int(max_items)
    if item_limit is not None and item_limit <= 0:
        raise ValueError(f"max_items must be > 0 or None; got {max_items}")
    for spatial, scalars, bearing, target in loader:
        if item_limit is not None:
            remaining = item_limit - seen_items
            if remaining <= 0:
                break
            batch_size = int(spatial.shape[0])
            if batch_size > remaining:
                spatial, scalars, bearing, target = _slice_batch_items(
                    (spatial, scalars, bearing, target),
                    num_items=remaining,
                )
                batch_size = remaining
            seen_items += batch_size
        spatial = _slice_spatial(
            spatial.to(device=device, dtype=torch.float32),
            channel_indices=channel_indices,
        )
        scalars = scalars.to(device=device, dtype=torch.float32)
        bearing = bearing.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        packed = _grid_logits_and_targets(model((spatial, scalars, bearing)), target)
        if packed is None:
            continue
        logits, true_idx = packed
        total_nll += float(
            F.cross_entropy(logits, true_idx, reduction="sum").item()
        )
        total_valid += int(true_idx.shape[0])
    if total_valid <= 0:
        raise ValueError(
            "No valid deepSSF targets were found in the loader (all targets were out-of-bounds)."
        )
    return {
        "nll": float(total_nll / float(total_valid)),
        "n_valid": float(total_valid),
    }


@torch.no_grad()
def _prediction_payload(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    channel_indices: torch.Tensor | None,
    pixel_size_m: float,
) -> dict[str, object]:
    model.eval()
    scores_chunks: list[torch.Tensor] = []
    for spatial, scalars, bearing, _target in loader:
        spatial = _slice_spatial(
            spatial.to(device=device, dtype=torch.float32),
            channel_indices=channel_indices,
        )
        scalars = scalars.to(device=device, dtype=torch.float32)
        bearing = bearing.to(device=device, dtype=torch.float32)
        out = model((spatial, scalars, bearing))
        if out.ndim != 4 or int(out.shape[-1]) != 2:
            raise ValueError(
                f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
            )
        scores_chunks.append((out[..., 0] + out[..., 1]).detach().cpu())
    return {
        "prediction_kind": "grid",
        "scores": torch.cat(scores_chunks, dim=0),
        "pixel_size_m": float(pixel_size_m),
    }


def _make_loader(cache_dir: Path, *, batch_size: int, seed: int, shuffle: bool):
    return make_deepssf_wds_loader(
        cache_dir,
        batch_size=int(batch_size),
        num_workers=0,
        shardshuffle=100 if shuffle else 0,
        shuffle_buffer=int(batch_size) * 8 if shuffle else 0,
        seed=int(seed),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=None,
    )


class _DeepSSFRuntime:
    early_stop_metric = "nll"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader,
        eval_loaders: dict[str, object],
        early_stop_split_key: str,
        device: torch.device,
        lr_movement: float,
        lr_habitat: float,
        weight_decay: float,
        channel_indices: list[int],
        eval_max_items: int | None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loaders = dict(eval_loaders)
        self.early_stop_split_key = str(early_stop_split_key)
        self.device = device
        self.eval_max_items = eval_max_items
        self._train_iter = iter(self.train_loader)
        self._channel_indices = None
        if channel_indices:
            self._channel_indices = torch.as_tensor(
                list(channel_indices),
                dtype=torch.long,
                device=device,
            )

        groups = self.model.parameter_groups()  # type: ignore[attr-defined]
        self.optimizer = torch.optim.Adam(
            [
                {"params": list(groups["movement"]), "lr": float(lr_movement)},
                {"params": list(groups["habitat"]), "lr": float(lr_habitat)},
            ],
            weight_decay=float(weight_decay),
        )

    def _next_train_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def train_step(self, step: int) -> dict[str, float]:
        spatial, scalars, bearing, target = self._next_train_batch()
        spatial = _slice_spatial(
            spatial.to(device=self.device, dtype=torch.float32),
            channel_indices=self._channel_indices,
        )
        scalars = scalars.to(device=self.device, dtype=torch.float32)
        bearing = bearing.to(device=self.device, dtype=torch.float32)
        target = target.to(device=self.device, dtype=torch.float32)

        self.model.train()
        packed = _grid_logits_and_targets(self.model((spatial, scalars, bearing)), target)
        if packed is None:
            return {"loss/train_step": 0.0, "n_valid/train_step": 0.0}

        logits, true_idx = packed
        self.optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, true_idx, reduction="mean")
        loss.backward()
        self.optimizer.step()
        return {
            "loss/train_step": float(loss.item()),
            "n_valid/train_step": float(true_idx.shape[0]),
        }

    def evaluate(self, split_key: str) -> dict[str, float]:
        loader = self.eval_loaders[str(split_key)]
        return _evaluate_loader(
            self.model,
            loader,
            device=self.device,
            channel_indices=self._channel_indices,
            max_items=self.eval_max_items,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "model": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(dict(state["model"]))
        self.optimizer.load_state_dict(dict(state["optimizer"]))


def run(*, run_payload, resolved_artifacts, run_dir, tracker):
    run_dir = Path(run_dir)
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams["batch_size"])
    seed = int(run_payload["seed"])
    early_stop_split_key = str(run_payload["early_stop_split_key"])
    eval_max_items = int(method_hparams["max_eval_items"])

    train_manifest = dict(resolved_artifacts["cache_manifests"]["train"]["manifest"])
    output_hw = tuple(int(v) for v in train_manifest["output_hw"])
    if len(output_hw) != 2 or int(output_hw[0]) != int(output_hw[1]):
        raise ValueError(f"deepSSF expects square chips. Got output_hw={output_hw}.")
    image_dim = int(run_payload.get("method_model", {}).get("image_dim", output_hw[0]))
    if image_dim != int(output_hw[0]):
        raise ValueError(
            f"deepSSF model.image_dim ({image_dim}) must match dataset chip size ({output_hw[0]})."
        )

    pixel_size_m = float(train_manifest["pixel_size_m"])

    train_loader = _make_loader(
        Path(resolved_artifacts["cache_manifests"]["train"]["cache_dir"]),
        batch_size=batch_size,
        seed=seed,
        shuffle=True,
    )
    eval_loaders = {
        "train": _make_loader(
            Path(resolved_artifacts["cache_manifests"]["train"]["cache_dir"]),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        ),
        early_stop_split_key: _make_loader(
            Path(resolved_artifacts["cache_manifests"][early_stop_split_key]["cache_dir"]),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        ),
    }

    schedule = {
        "max_steps": int(method_hparams["max_steps"]),
        "eval_every_steps": int(method_hparams["eval_every_steps"]),
        "patience_evals": int(method_hparams["patience_evals"]),
    }
    channel_indices = list(resolved_artifacts["channel_indices"])
    model = DeepSSF(
        len(channel_indices),
        image_dim=image_dim,
        pixel_size_m=pixel_size_m,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    runtime = _DeepSSFRuntime(
        model=model,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        early_stop_split_key=early_stop_split_key,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lr_movement=float(method_hparams["lr_movement"]),
        lr_habitat=float(method_hparams["lr_habitat"]),
        weight_decay=float(method_hparams["weight_decay"]),
        channel_indices=channel_indices,
        eval_max_items=eval_max_items,
    )
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

    checkpoint_path = run_dir / "checkpoint_best.pt"
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
                "env_channels": len(channel_indices),
                "image_dim": image_dim,
                "pixel_size_m": pixel_size_m,
            },
        },
        checkpoint_path,
    )

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_key in run_payload["evaluation_split_keys"]:
        loader = _make_loader(
            Path(resolved_artifacts["cache_manifests"][split_key]["cache_dir"]),
            batch_size=batch_size,
            seed=seed,
            shuffle=False,
        )
        torch.save(
            _prediction_payload(
                runtime.model,
                loader,
                device=runtime.device,
                channel_indices=runtime._channel_indices,
                pixel_size_m=pixel_size_m,
            ),
            pred_dir / f"{split_key}.pt",
        )

    return {
        "status": "completed",
        "train_steps": int(fit["completed_steps"]),
        "best_step": int(fit["best_step"]),
        "early_stop_split_key": early_stop_split_key,
        "train_metrics": dict(fit["train_metrics"]),
        "eval_metrics": dict(fit["eval_metrics"]),
        "schedule": dict(schedule),
    }
