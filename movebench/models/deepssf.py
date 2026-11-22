"""MoveBench wrapper for the reference deepSSF implementation.

The underlying model lives in `movebench.lib.deepSSF.deepSSF_model`. This file
provides a MoveBench-friendly `nn.Module` wrapper that:
- exposes a stable constructor interface,
- auto-inferrs `dense_dim_in_all` (avoids notebook-only manual bookkeeping),
- provides parameter groups for separate learning rates.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from movebench.datasets.rasters import prepare_rasters
from movebench.geometry import LOCAL_GEOMETRY
from movebench.lib.deepSSF.deepSSF_model import (
    Conv2d_block_toFC,
    ConvJointModel,
    ModelParams,
)
from movebench.training import RunTracker, TrainerConfig, fit_or_restore


def _infer_dense_dim_in_all(
    *,
    image_dim: int,
    input_channels: int,
    output_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    kernel_size_mp: int,
    stride_mp: int,
) -> int:
    """Infer the flattened conv feature size entering the movement FCN."""
    params = ModelParams(
        {
            "batch_size": 1,
            "image_dim": int(image_dim),
            "pixel_size": 1.0,
            "dim_in_nonspatial_to_grid": 0,
            "dense_dim_in_nonspatial": 0,
            "dense_dim_hidden": 1,
            "dense_dim_in_all": 1,
            "input_channels": int(input_channels),
            "output_channels": int(output_channels),
            "kernel_size": int(kernel_size),
            "stride": int(stride),
            "kernel_size_mp": int(kernel_size_mp),
            "stride_mp": int(stride_mp),
            "padding": int(padding),
            "num_movement_params": 1,
            "dropout": 0.0,
            "device": torch.device("cpu"),
        }
    )

    block = Conv2d_block_toFC(params).to(torch.device("cpu"))
    x = torch.zeros(
        (1, int(input_channels), int(image_dim), int(image_dim)), dtype=torch.float32
    )
    with torch.no_grad():
        y = block(x)
    if y.ndim != 2 or int(y.shape[0]) != 1:
        raise RuntimeError(
            f"Unexpected Conv2d_block_toFC output shape: {tuple(y.shape)}"
        )
    return int(y.shape[1])


class DeepSSF(nn.Module):
    """deepSSF next-step prediction model (grid-based)."""

    def __init__(
        self,
        env_channels: int,
        *,
        image_dim: int = 101,
        pixel_size_m: float,
        time_channels: int = 4,
        output_channels: int = 4,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        kernel_size_mp: int = 2,
        stride_mp: int = 2,
        num_movement_params: int = 12,
        dense_dim_hidden: int = 128,
        dropout: float = 0.1,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if int(env_channels) <= 0:
            raise ValueError(f"env_channels must be > 0; got {env_channels}")
        if int(image_dim) <= 0:
            raise ValueError(f"image_dim must be > 0; got {image_dim}")
        if float(pixel_size_m) <= 0:
            raise ValueError(f"pixel_size_m must be > 0; got {pixel_size_m}")
        if int(time_channels) <= 0:
            raise ValueError(f"time_channels must be > 0; got {time_channels}")

        self.env_channels = int(env_channels)
        self.image_dim = int(image_dim)
        self.pixel_size_m = float(pixel_size_m)
        self.time_channels = int(time_channels)

        input_channels = int(env_channels) + int(time_channels)
        dense_dim_in_all = _infer_dense_dim_in_all(
            image_dim=int(image_dim),
            input_channels=int(input_channels),
            output_channels=int(output_channels),
            kernel_size=int(kernel_size),
            stride=int(stride),
            padding=int(padding),
            kernel_size_mp=int(kernel_size_mp),
            stride_mp=int(stride_mp),
        )

        dev = torch.device(device)
        params = ModelParams(
            {
                "batch_size": 1,
                "image_dim": int(image_dim),
                "pixel_size": float(pixel_size_m),
                "dim_in_nonspatial_to_grid": int(time_channels),
                "dense_dim_in_nonspatial": 1,
                "dense_dim_hidden": int(dense_dim_hidden),
                "dense_dim_in_all": int(dense_dim_in_all),
                "input_channels": int(input_channels),
                "output_channels": int(output_channels),
                "kernel_size": int(kernel_size),
                "stride": int(stride),
                "kernel_size_mp": int(kernel_size_mp),
                "stride_mp": int(stride_mp),
                "padding": int(padding),
                "num_movement_params": int(num_movement_params),
                "dropout": float(dropout),
                "device": dev,
            }
        )

        self.model = ConvJointModel(params).to(dev)
        self.model.conv_movement.double()
        self.model.fcn_movement_all.double()

        # Ensure deepSSF's internal device attributes and precomputed layers are on the model device.
        self._sync_internal_device(dev)

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        """Parameter groups for separate learning rates."""
        habitat = list(self.model.conv_habitat.parameters())
        movement = list(self.model.conv_movement.parameters()) + list(
            self.model.fcn_movement_all.parameters()
        )
        if not habitat:
            raise ValueError("deepSSF habitat parameter group is empty")
        if not movement:
            raise ValueError("deepSSF movement parameter group is empty")
        return {"movement": movement, "habitat": habitat}

    def _sync_internal_device(self, device: torch.device) -> None:
        """Keep deepSSF internal tensors/device attributes consistent."""
        mgo = getattr(self.model, "movement_grid_output", None)
        if mgo is not None:
            if hasattr(mgo, "device"):
                mgo.device = device
            for name in ("distance_layer", "bearing_layer"):
                t = getattr(mgo, name, None)
                if isinstance(t, torch.Tensor) and t.device != device:
                    setattr(mgo, name, t.to(device))

        # Other submodules store a `.device` attribute; keep it consistent even if unused.
        for sub_name in (
            "conv_habitat",
            "conv_movement",
            "fcn_movement_all",
            "scalar_grid_output",
        ):
            sub = getattr(self.model, sub_name, None)
            if sub is not None and hasattr(sub, "device"):
                sub.device = device

    def forward(
        self, x: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        spatial, scalars_to_grid, bearing = x
        dev = spatial.device
        self._sync_internal_device(dev)
        return self.model((spatial, scalars_to_grid, bearing))


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


class DeepSSFMethod:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        method_hparams = self.hparams
        batch_size = int(method_hparams["batch_size"])
        seed = self.seed
        early_stop_split_key = validation.split_key
        eval_max_items = int(method_hparams["max_eval_items"])
        backend = train.inputs["kind"]
        device = _device_for_backend(backend)
        rasters = prepare_rasters(
            train,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        output_hw = rasters.output_hw
        pixel_size_m = rasters.pixel_size_m
        if len(output_hw) != 2 or int(output_hw[0]) != int(output_hw[1]):
            raise ValueError(f"deepSSF expects square chips. Got output_hw={output_hw}.")
        method_model = self.model_config
        image_dim = int(method_model.get("image_dim", output_hw[0]))
        if image_dim != int(output_hw[0]):
            raise ValueError(
                f"deepSSF model.image_dim={image_dim} is incompatible with output_hw={list(output_hw)}"
            )

        train_loader = rasters.loader("train", shuffle=True)
        steps_per_epoch = rasters.steps_per_epoch
        eval_loaders = {
            "train": rasters.loader("train", shuffle=False),
            early_stop_split_key: rasters.loader(
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
        channel_indices = rasters.channel_indices
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
        fit = fit_or_restore(
            runtime,
            TrainerConfig(
                max_steps=int(schedule["max_steps"]),
                eval_every_steps=int(schedule["eval_every_steps"]),
                patience_evals=int(schedule["patience_evals"]),
            ),
            RunTracker(Path(work_dir)),
            work_dir=Path(work_dir),
            restore_from=restore_from,
            schedule=schedule,
        )

        self.model_ = runtime.model
        self.device_ = device
        self.rasters_ = rasters
        self.channel_indices_ = runtime._channel_indices
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
        return _prediction_payload(
            self.model_, self.rasters_.loader(data.split_key, shuffle=False),
            device=self.device_, channel_indices=self.channel_indices_,
            pixel_size_m=self.rasters_.pixel_size_m,
        )
