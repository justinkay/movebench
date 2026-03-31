"""Step-selection models for iSSF benchmarks."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


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
