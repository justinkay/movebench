"""PyTorch models for MoveBench2 step selection benchmarks.

All models in this module follow the same contract:
- input `x` has shape (B, C, P): batch, feature-channels, candidate points
- output `logits` has shape (B, P): one score per candidate point
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


class LinearStepSelect(nn.Module):
    """Linear step-selection: one logit per candidate."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0; got {in_dim}")
        self.linear = nn.Linear(int(in_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,C,P); got {tuple(x.shape)}")
        # (B,C,P) -> (B,P,C)
        x_pc = x.transpose(1, 2)
        logits = self.linear(x_pc).squeeze(-1)
        return logits


class MLPStepSelect(nn.Module):
    """MLP step-selection: one logit per candidate."""

    def __init__(
        self,
        in_dim: int,
        *,
        hidden_dims: Sequence[int] = (128,),
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0; got {in_dim}")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0,1); got {dropout}")

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape (B,C,P); got {tuple(x.shape)}")
        x_pc = x.transpose(1, 2)  # (B,P,C)
        logits = self.net(x_pc).squeeze(-1)  # (B,P)
        return logits


def random_baseline_metrics(num_choices: int, topk: Sequence[int]) -> dict:
    """Deterministic random-choice baseline metrics (no sampling noise)."""
    k = int(num_choices)
    if k <= 0:
        raise ValueError(f"num_choices must be > 0; got {num_choices}")
    return {
        "nll": float(math.log(k)),
        "topk": {int(kk): float(min(int(kk), k)) / float(k) for kk in topk},
    }
