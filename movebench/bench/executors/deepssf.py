from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import movebench.bench.evaluation.runner as evaluation_runner
from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.deepssf_gpu import (
    AlphaEarthGPUCache,
    build_deepssf_gpu_datasets,
    make_deepssf_gpu_loader,
)
from movebench.datasets.deepssf_wds_cache import (
    make_deepssf_wds_loader,
    steps_per_epoch_from_manifest,
)
from movebench.geometry import LOCAL_GEOMETRY
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
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    num_items: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(t[: int(num_items)] for t in batch)  # type: ignore[return-value]


def _grid_logits_and_targets(
    out: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor] | None:
    if out.ndim != 4 or int(out.shape[-1]) != 2:
        raise ValueError(
            f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
        )
    batch_size, height, width, _components = out.shape
    target_flat = target.view(int(batch_size), int(height) * int(width))
    target_mass = target_flat.sum(dim=1)
    valid = (target_mass > 0.5) & (target_mass < 1.5)
    if not valid.any().item():
        return None
    habitat = out[..., 0].reshape(int(batch_size), int(height) * int(width))
    movement = out[..., 1].reshape(int(batch_size), int(height) * int(width))
    return {
        "nll": (habitat + movement)[valid],
        "nll_habitat": habitat[valid],
        "nll_movement": movement[valid],
    }, torch.argmax(target_flat[valid], dim=1)


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
    total_nll = {
        "nll": 0.0,
        "nll_habitat": 0.0,
        "nll_movement": 0.0,
    }
    total_valid = 0
    seen_items = 0
    item_limit = None if max_items is None else int(max_items)
    if item_limit is not None and item_limit <= 0:
        raise ValueError(f"max_items must be > 0 or None; got {max_items}")
    for spatial, scalars, bearing, target, _point_id_start in loader:
        if item_limit is not None:
            remaining = item_limit - seen_items
            if remaining <= 0:
                break
            batch_size = int(spatial.shape[0])
            if batch_size > remaining:
                spatial, scalars, bearing, target, _point_id_start = _slice_batch_items(
                    (spatial, scalars, bearing, target, _point_id_start),
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
        for key, values in logits.items():
            total_nll[key] += float(
                F.cross_entropy(values, true_idx, reduction="sum").item()
            )
        total_valid += int(true_idx.shape[0])
    if total_valid <= 0:
        raise ValueError(
            "No valid deepSSF targets were found in the loader (all targets were out-of-bounds)."
        )
    return {
        key: float(value / float(total_valid))
        for key, value in total_nll.items()
    } | {
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
    point_id_chunks: list[torch.Tensor] = []
    for spatial, scalars, bearing, _target, point_id_start in loader:
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
        point_id_chunks.append(point_id_start.to(dtype=torch.int64).cpu())
    return {
        "prediction_kind": "grid",
        "geometry": LOCAL_GEOMETRY,
        "point_id_start": torch.cat(point_id_chunks, dim=0),
        "scores": torch.cat(scores_chunks, dim=0),
        "pixel_size_m": float(pixel_size_m),
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


def _make_loader(
    cache_dir: Path,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
):
    return make_deepssf_wds_loader(
        cache_dir,
        batch_size=int(batch_size),
        num_workers=8,
        shardshuffle=100 if shuffle else 0,
        shuffle_buffer=int(batch_size) * 8 if shuffle else 0,
        seed=int(seed),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        prefetch_factor=None,
    )


def _device_for_backend(backend: str) -> torch.device:
    if str(backend) == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("deepssf_data='gpu' requires a CUDA device")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        steps_per_epoch: int,
        lr_scheduler_factor: float,
        lr_scheduler_patience_epochs: int,
        channel_indices: list[int],
        eval_max_items: int | None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loaders = dict(eval_loaders)
        self.early_stop_split_key = str(early_stop_split_key)
        self.device = device
        self.eval_max_items = eval_max_items
        self.steps_per_epoch = int(steps_per_epoch)
        self._train_iter = iter(self.train_loader)
        self._channel_indices = None
        if channel_indices:
            self._channel_indices = torch.as_tensor(
                list(channel_indices),
                dtype=torch.long,
                device=device,
            )

        groups = self.model.parameter_groups()  # type: ignore[attr-defined]
        self.optimizers = {
            "movement": torch.optim.Adam(
                list(groups["movement"]),
                lr=float(lr_movement),
                weight_decay=float(weight_decay),
            ),
            "habitat": torch.optim.Adam(
                list(groups["habitat"]),
                lr=float(lr_habitat),
                weight_decay=float(weight_decay),
            ),
        }
        self.lr_schedulers = {
            key: torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=float(lr_scheduler_factor),
                patience=int(lr_scheduler_patience_epochs),
            )
            for key, optimizer in self.optimizers.items()
        }

    def _next_train_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def train_step(self, step: int) -> dict[str, float]:
        spatial, scalars, bearing, target, _point_id_start = self._next_train_batch()
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
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits["nll"], true_idx, reduction="mean")
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"deepSSF loss became non-finite at step {step}")
        loss.backward()
        for optimizer in self.optimizers.values():
            optimizer.step()
        return {
            "loss/train_step": float(loss.item()),
            "n_valid/train_step": float(true_idx.shape[0]),
        }

    def lr_scheduler_due(self, step: int) -> bool:
        return int(step) % self.steps_per_epoch == 0

    def step_lr_schedulers(self, eval_metrics: dict[str, float]) -> dict[str, float]:
        self.lr_schedulers["movement"].step(float(eval_metrics["nll_movement"]))
        self.lr_schedulers["habitat"].step(float(eval_metrics["nll_habitat"]))
        return {
            f"lr_{key}": float(optimizer.param_groups[0]["lr"])
            for key, optimizer in self.optimizers.items()
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
            "optimizers": {
                key: optimizer.state_dict()
                for key, optimizer in self.optimizers.items()
            },
            "lr_schedulers": {
                key: scheduler.state_dict()
                for key, scheduler in self.lr_schedulers.items()
            },
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(dict(state["model"]))
        optimizer_states = dict(state["optimizers"])
        scheduler_states = dict(state["lr_schedulers"])
        for key, optimizer in self.optimizers.items():
            optimizer.load_state_dict(dict(optimizer_states[key]))
            self.lr_schedulers[key].load_state_dict(dict(scheduler_states[key]))


def run(*, run_payload, resolved_artifacts, run_dir, tracker):
    run_dir = Path(run_dir)
    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams["batch_size"])
    seed = int(run_payload["seed"])
    early_stop_split_key = str(run_payload["early_stop_split_key"])
    eval_max_items = int(method_hparams["max_eval_items"])
    backend = str(run_payload.get("deepssf_data", "wds"))
    device = _device_for_backend(backend)
    if backend == "gpu":
        selected_splits = [str(split) for split in run_payload["selected_splits"]]
        datasets = build_deepssf_gpu_datasets(
            study_id=str(run_payload["study_id"]),
            trajectories_path=resolved_artifacts["trajectories_path"],
            plan=resolved_artifacts["plan"],
            splits=selected_splits,
            catalog=resolved_artifacts["catalog"],
            output_hw=tuple(resolved_artifacts["output_hw"]),
        )
        output_hw = tuple(int(value) for value in resolved_artifacts["output_hw"])
        pixel_size_m = float(datasets["train"].pixel_size_m)
        cache = AlphaEarthGPUCache(resolved_artifacts["catalog"], device)

        def loader_for_split(split_key: str, *, shuffle: bool):
            return make_deepssf_gpu_loader(
                datasets[str(split_key)],
                cache=cache,
                batch_size=batch_size,
                num_workers=8,
                seed=seed,
                shuffle=shuffle,
            )

    elif backend == "wds":
        for split_key in ["train", *list(run_payload["evaluation_split_keys"])]:
            manifest = dict(
                resolved_artifacts["cache_manifests"][str(split_key)]["manifest"]
            )
            if str(manifest.get("sample_identity", "")) != "point_id_start":
                raise ValueError(
                    "deepSSF cache manifest must declare sample_identity='point_id_start'; rebuild the cache"
                )
        train_manifest = dict(
            resolved_artifacts["cache_manifests"]["train"]["manifest"]
        )
        output_hw = tuple(int(value) for value in train_manifest["output_hw"])
        pixel_size_m = float(train_manifest["pixel_size_m"])

        def loader_for_split(split_key: str, *, shuffle: bool):
            return _make_loader(
                Path(
                    resolved_artifacts["cache_manifests"][str(split_key)][
                        "cache_dir"
                    ]
                ),
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
            )

    else:
        raise ValueError(f"Unsupported deepssf_data={backend!r}")

    if len(output_hw) != 2 or int(output_hw[0]) != int(output_hw[1]):
        raise ValueError(f"deepSSF expects square chips. Got output_hw={output_hw}.")
    method_model = dict(run_payload.get("method_model", {}))
    image_dim = int(method_model.get("image_dim", output_hw[0]))
    if image_dim != int(output_hw[0]):
        raise ValueError(
            f"deepSSF model.image_dim={image_dim} is incompatible with output_hw={list(output_hw)}"
        )

    train_loader = loader_for_split("train", shuffle=True)
    steps_per_epoch = (
        len(train_loader)
        if backend == "gpu"
        else steps_per_epoch_from_manifest(train_manifest, batch_size=batch_size)
    )
    eval_loaders = {
        "train": loader_for_split("train", shuffle=False),
        early_stop_split_key: loader_for_split(
            early_stop_split_key,
            shuffle=False,
        ),
    }

    schedule = {
        "max_steps": int(method_hparams["max_steps"]),
        "eval_every_steps": int(method_hparams["eval_every_steps"]),
        "patience_evals": int(method_hparams["patience_evals"]),
        "steps_per_epoch": int(steps_per_epoch),
        "lr_scheduler_factor": float(method_hparams["lr_scheduler_factor"]),
        "lr_scheduler_patience_epochs": int(
            method_hparams["lr_scheduler_patience_epochs"]
        ),
    }
    channel_indices = list(resolved_artifacts["channel_indices"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = DeepSSF(
        len(channel_indices),
        image_dim=image_dim,
        output_channels=int(method_model["output_channels"]),
        dense_dim_hidden=int(method_model["dense_dim_hidden"]),
        dropout=float(method_model["dropout"]),
        pixel_size_m=pixel_size_m,
        device=device,
    )
    runtime = _DeepSSFRuntime(
        model=model,
        train_loader=train_loader,
        eval_loaders=eval_loaders,
        early_stop_split_key=early_stop_split_key,
        device=device,
        lr_movement=float(method_hparams["lr_movement"]),
        lr_habitat=float(method_hparams["lr_habitat"]),
        weight_decay=float(method_hparams["weight_decay"]),
        steps_per_epoch=int(schedule["steps_per_epoch"]),
        lr_scheduler_factor=float(schedule["lr_scheduler_factor"]),
        lr_scheduler_patience_epochs=int(
            schedule["lr_scheduler_patience_epochs"]
        ),
        channel_indices=channel_indices,
        eval_max_items=eval_max_items,
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
            "train_metrics": {},
            "eval_metrics": {},
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
        loader = loader_for_split(str(split_key), shuffle=False)
        torch.save(
            _prediction_payload(
                runtime.model,
                loader,
                device=runtime.device,
                channel_indices=runtime._channel_indices,
                pixel_size_m=pixel_size_m,
            ),
            pred_path,
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
