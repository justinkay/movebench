"""Step-selection models for iSSF benchmarks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import movebench.ssf_grid as ssf_grid
from movebench.datasets.chips import grid_dataset
from movebench.datasets.points import training_steps
from movebench.datasets.ssf import (
    candidate_dataset,
    candidate_eval_loader,
    candidate_loader,
)
from movebench.training import RunTracker, TrainerConfig, fit_or_restore


def _require_batch_tensor(
    batch: dict[str, torch.Tensor | None],
    key: str,
    *,
    ndim: int,
) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        raise ValueError(f"Missing required batch key: {key!r}")
    if value.ndim != ndim:
        raise ValueError(
            f"Expected batch[{key!r}] with rank {ndim}, got {tuple(value.shape)}"
        )
    return value


def _unpack_batch(
    batch: dict[str, torch.Tensor | None],
    *,
    include_time: bool,
    n_individuals: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    env = _require_batch_tensor(batch, "env", ndim=3)
    move = _require_batch_tensor(batch, "move", ndim=3)
    if int(move.shape[1]) != 3:
        raise ValueError(f"Expected move with shape (B,3,P), got {tuple(move.shape)}")
    if int(move.shape[0]) != int(env.shape[0]) or int(move.shape[2]) != int(env.shape[2]):
        raise ValueError(
            f"env/move batch shapes must agree on B,P; got env={tuple(env.shape)} move={tuple(move.shape)}"
        )

    time = batch.get("time")
    if include_time:
        if time is None:
            raise ValueError("Model expects time features but batch['time'] is None")
        if time.ndim != 2 or int(time.shape[0]) != int(env.shape[0]) or int(time.shape[1]) != 4:
            raise ValueError(
                f"Expected time with shape (B,4), got {tuple(time.shape)}"
            )
    elif time is not None:
        raise ValueError("Model does not expect time features")

    individual_idx = batch.get("individual_idx")
    if n_individuals > 0:
        if individual_idx is None:
            raise ValueError(
                "Model expects individual_idx but batch['individual_idx'] is None"
            )
        if individual_idx.ndim != 1 or int(individual_idx.shape[0]) != int(env.shape[0]):
            raise ValueError(
                f"Expected individual_idx with shape (B,), got {tuple(individual_idx.shape)}"
            )
    elif individual_idx is not None:
        raise ValueError("Model does not expect individual_idx")

    return env, move, time, individual_idx


class LinearStepSelect(nn.Module):
    """Exact linear iSSF scorer over batch base terms."""

    def __init__(
        self,
        c_env: int,
        *,
        include_time: bool = False,
        n_individuals: int = 0,
    ) -> None:
        super().__init__()
        if int(c_env) <= 0:
            raise ValueError(f"c_env must be > 0; got {c_env}")
        if int(n_individuals) < 0:
            raise ValueError(f"n_individuals must be >= 0; got {n_individuals}")
        self.c_env = int(c_env)
        self.include_time = bool(include_time)
        self.n_individuals = int(n_individuals)

        self.w_env = nn.Parameter(torch.empty(self.c_env))
        self.w_move = nn.Parameter(torch.empty(3))
        self.w_move_env = nn.Parameter(torch.empty(3, self.c_env))

        if self.include_time:
            self.w_time = nn.Parameter(torch.empty(4))
            self.w_time_env = nn.Parameter(torch.empty(4, self.c_env))
            self.w_time_move = nn.Parameter(torch.empty(4, 3))
        else:
            self.w_time = None
            self.w_time_env = None
            self.w_time_move = None

        if self.n_individuals > 0:
            self.individual_bias = nn.Parameter(torch.empty(self.n_individuals))
            self.w_individual_env = nn.Parameter(
                torch.empty(self.n_individuals, self.c_env)
            )
            self.w_individual_move = nn.Parameter(torch.empty(self.n_individuals, 3))
        else:
            self.individual_bias = None
            self.w_individual_env = None
            self.w_individual_move = None

        self.bias = nn.Parameter(torch.empty(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        in_dim = int(self.c_env) + 3 + (4 if self.include_time else 0) + self.n_individuals
        bound = 1.0 / math.sqrt(float(max(1, in_dim)))
        for param in (
            self.w_env,
            self.w_move,
            self.w_move_env,
            self.w_time,
            self.w_time_env,
            self.w_time_move,
            self.individual_bias,
            self.w_individual_env,
            self.w_individual_move,
            self.bias,
        ):
            if param is not None:
                nn.init.uniform_(param, -bound, bound)

    def forward(self, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        env, move, time, individual_idx = _unpack_batch(
            batch,
            include_time=self.include_time,
            n_individuals=self.n_individuals,
        )

        logits = torch.einsum("e,bep->bp", self.w_env, env)
        logits += torch.einsum("m,bmp->bp", self.w_move, move)
        logits += torch.einsum("me,bmp,bep->bp", self.w_move_env, move, env)

        if time is not None:
            assert self.w_time is not None
            assert self.w_time_env is not None
            assert self.w_time_move is not None
            logits += torch.einsum("t,bt->b", self.w_time, time).unsqueeze(1)
            logits += torch.einsum("te,bt,bep->bp", self.w_time_env, time, env)
            logits += torch.einsum("tm,bt,bmp->bp", self.w_time_move, time, move)

        if individual_idx is not None:
            assert self.individual_bias is not None
            assert self.w_individual_env is not None
            assert self.w_individual_move is not None
            logits += self.individual_bias[individual_idx].unsqueeze(1)
            logits += torch.einsum(
                "be,bep->bp", self.w_individual_env[individual_idx], env
            )
            logits += torch.einsum(
                "bm,bmp->bp", self.w_individual_move[individual_idx], move
            )

        return logits + self.bias


class MLPStepSelect(nn.Module):
    """MLP iSSF scorer over batch base terms."""

    def __init__(
        self,
        c_env: int,
        *,
        include_time: bool = False,
        n_individuals: int = 0,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.6,
        interactions_before_mlp: bool = False,
    ) -> None:
        super().__init__()
        if int(c_env) <= 0:
            raise ValueError(f"c_env must be > 0; got {c_env}")
        if int(n_individuals) < 0:
            raise ValueError(f"n_individuals must be >= 0; got {n_individuals}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1); got {dropout}")

        self.c_env = int(c_env)
        self.include_time = bool(include_time)
        self.n_individuals = int(n_individuals)
        self.interactions_before_mlp = bool(interactions_before_mlp)

        in_dim = self.c_env + 3 + (4 if self.include_time else 0) + self.n_individuals
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            h = int(h)
            if h <= 0:
                raise ValueError(f"hidden_dims must be all > 0; got {hidden_dims!r}")
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, batch: dict[str, torch.Tensor | None]) -> torch.Tensor:
        if self.interactions_before_mlp:
            raise NotImplementedError(
                "MLPStepSelect(interactions_before_mlp=True) is not implemented"
            )

        env, move, time, individual_idx = _unpack_batch(
            batch,
            include_time=self.include_time,
            n_individuals=self.n_individuals,
        )
        parts = [env, move]

        if time is not None:
            parts.append(time.unsqueeze(-1).expand(-1, -1, int(env.shape[2])))
        if individual_idx is not None:
            one_hot = F.one_hot(
                individual_idx.to(dtype=torch.long),
                num_classes=self.n_individuals,
            ).to(dtype=env.dtype)
            parts.append(one_hot.unsqueeze(-1).expand(-1, -1, int(env.shape[2])))

        x = torch.cat(parts, dim=1)
        logits = self.net(x.transpose(1, 2)).squeeze(-1)
        return logits


def random_baseline_metrics(num_choices: int, topk: Sequence[int]) -> dict:
    """Deterministic random-choice baseline metrics (no sampling noise)."""
    k = int(num_choices)
    if k <= 0:
        raise ValueError(f"num_choices must be > 0; got {k}")
    return {
        "nll": float(math.log(k)),
        "topk": {int(kk): float(min(int(kk), k)) / float(k) for kk in topk},
    }


def _loss_step_select(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B,P), got {tuple(logits.shape)}")
    target = torch.full(
        (int(logits.shape[0]),),
        int(logits.shape[1] - 1),
        dtype=torch.long,
        device=logits.device,
    )
    return F.cross_entropy(logits, target)


def _move_batch_to_device(
    batch: dict[str, torch.Tensor | None], device: torch.device
) -> dict[str, torch.Tensor | None]:
    out: dict[str, torch.Tensor | None] = {}
    for key, value in batch.items():
        if value is None:
            out[key] = None
        else:
            out[key] = value.to(device, non_blocking=device.type == "cuda")
    return out


@torch.no_grad()
def _eval_step_select_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch)
        loss = _loss_step_select(logits)
        total_nll += float(loss.item()) * int(logits.shape[0])
        total += int(logits.shape[0])
    if total == 0:
        raise ValueError("Loader yielded zero items")
    return {"nll": float(total_nll / float(total))}


class _ISSFRuntime:
    early_stop_metric = "nll"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_train_loader: DataLoader,
        eval_loader: DataLoader,
        early_stop_split_key: str,
        device: torch.device,
        lr: float,
        weight_decay: float,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_train_loader = eval_train_loader
        self.eval_loader = eval_loader
        self.early_stop_split_key = str(early_stop_split_key)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self._train_iter = iter(self.train_loader)

    def _next_batch(self) -> dict[str, torch.Tensor | None]:
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return _move_batch_to_device(batch, self.device)

    def train_step(self, step: int) -> dict[str, float]:
        del step
        batch = self._next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch)
        loss = _loss_step_select(logits)
        loss.backward()
        self.optimizer.step()
        return {"loss/train_step": float(loss.item())}

    def evaluate(self, split_key: str) -> dict[str, float]:
        if split_key == "train":
            return _eval_step_select_loader(
                self.model, self.eval_train_loader, self.device
            )
        if split_key != self.early_stop_split_key:
            raise KeyError(f"Unknown split for iSSF runtime: {split_key!r}")
        return _eval_step_select_loader(self.model, self.eval_loader, self.device)

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


class ISSFMethod:
    def __init__(self, *, model_config, hparams, seed, model_kind):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)
        self.model_kind = model_kind

    def _dataset(self, data):
        if data.split_key not in self.datasets_:
            self.datasets_[data.split_key] = candidate_dataset(
                data, include_time=bool(self.model_config.get("include_time", False))
            )
        return self.datasets_[data.split_key]

    def fit(self, train, validation, *, work_dir, restore_from=None):
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.device_ = torch.device("cpu")
        self.train_ = train
        self.datasets_ = {}
        self.sampler_state_ = None
        batch_size = int(self.hparams.get("batch_size", 256))
        num_workers = int(self.hparams.get("num_workers", 0) or 0)
        eval_max_items = self.hparams.get("max_eval_items")
        train_dataset = self._dataset(train)
        eval_dataset = self._dataset(validation)
        train_loader = candidate_loader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, device=self.device_,
        )
        eval_train_loader = candidate_eval_loader(
            train_dataset, batch_size=batch_size, seed=self.seed,
            max_items=eval_max_items, num_workers=num_workers, device=self.device_,
        )
        eval_loader = candidate_eval_loader(
            eval_dataset, batch_size=batch_size, seed=self.seed + 1,
            max_items=eval_max_items, num_workers=num_workers, device=self.device_,
        )
        if self.model_kind == "linear":
            model = LinearStepSelect(
                int(train_dataset.c_env),
                include_time=bool(train_dataset.include_time),
                n_individuals=int(train_dataset.n_individuals),
            )
        elif self.model_kind == "mlp":
            model = MLPStepSelect(
                int(train_dataset.c_env),
                include_time=bool(train_dataset.include_time),
                n_individuals=int(train_dataset.n_individuals),
                hidden_dims=tuple(self.model_config.get("hidden_dims", [128])),
                dropout=float(self.model_config.get("dropout", 0.6)),
                interactions_before_mlp=bool(
                    self.model_config.get("interactions_before_mlp", False)
                ),
            )
        else:
            raise ValueError(f"Unsupported iSSF model kind: {self.model_kind!r}")
        runtime = _ISSFRuntime(
            model=model, train_loader=train_loader,
            eval_train_loader=eval_train_loader, eval_loader=eval_loader,
            early_stop_split_key=validation.split_key, device=self.device_,
            lr=float(self.hparams["lr"]),
            weight_decay=float(self.hparams["weight_decay"]),
        )
        fit = fit_or_restore(
            runtime,
            TrainerConfig(
                max_steps=int(self.hparams["max_steps"]),
                eval_every_steps=int(self.hparams["eval_every_steps"]),
                patience_evals=int(self.hparams["patience_evals"]),
            ),
            RunTracker(Path(work_dir)),
            work_dir=Path(work_dir), restore_from=restore_from,
        )
        self.model_ = runtime.model
        self.training_summary_ = {
            "train_steps": int(fit["completed_steps"]),
            "best_step": int(fit["best_step"]),
            "early_stop_split_key": validation.split_key,
            "train_metrics": dict(fit["train_metrics"]),
            "eval_metrics": dict(fit["eval_metrics"]),
        }
        return self

    def predict(self, data):
        if self.sampler_state_ is None:
            _support, steps = training_steps(self.train_)
            self.sampler_state_ = ssf_grid._replay_sampler_state(
                train_manifest=self.train_.inputs["point_stores"]["train"]["manifest"],
                train_steps=steps,
            )
        dataset = self._dataset(data)
        return ssf_grid.build_ssf_grid_artifact(
            grid_dataset=grid_dataset(data, plan=dataset.plan, include_env=True),
            sampler_state=self.sampler_state_,
            batch_size=int(self.hparams.get("batch_size", 256)),
            num_workers=int(self.hparams.get("prediction_num_workers", 4) or 0),
            device=self.device_, model=self.model_, feature_adapter=dataset,
        )


def linear_method(*, model_config, hparams, seed):
    return ISSFMethod(
        model_config=model_config, hparams=hparams, seed=seed, model_kind="linear"
    )


def mlp_method(*, model_config, hparams, seed):
    return ISSFMethod(
        model_config=model_config, hparams=hparams, seed=seed, model_kind="mlp"
    )
