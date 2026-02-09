"""Training utilities for MoveBench2 step-selection benchmarks."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as F


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_COLLATE_CHECKED = False


def collate_step_select(batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate iSSF/SSF dataset items into batched tensors.

    Returns:
        x: float32 (B, C, P)
        valid: bool (B, C, P)
    """
    xs: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for step_pixels, valid_pixels, _meta, _crs in batch:
        xs.append(step_pixels.to(dtype=torch.float32))
        valids.append(valid_pixels.to(dtype=torch.bool))

    x = torch.stack(xs, dim=0)
    valid = torch.stack(valids, dim=0)

    global _COLLATE_CHECKED
    if not _COLLATE_CHECKED:
        if not torch.isfinite(x).all().item():
            bad = (~torch.isfinite(x)).nonzero(as_tuple=False)
            raise ValueError(f"Non-finite values found in step features at indices (first 10): {bad[:10].tolist()}")
        if (x[~valid] != 0).any().item():
            bad = (x[~valid] != 0).nonzero(as_tuple=False)
            raise ValueError(
                "Invalid entries must be zero in the dataset feature tensors. "
                f"Found nonzero values at indices (first 10): {bad[:10].tolist()}"
            )
        _COLLATE_CHECKED = True

    return x, valid


def loss_step_select(logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss where the true candidate is always the last index (P-1)."""
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B,P); got {tuple(logits.shape)}")
    B, P = logits.shape
    if P <= 0:
        raise ValueError(f"P must be > 0; got {P}")
    target = torch.full((B,), int(P - 1), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


@torch.no_grad()
def eval_step_select(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    topk: Sequence[int] = (1, 5, 10, 25, 50),
) -> dict:
    model.eval()
    total_nll = 0.0
    total = 0
    correct_topk = {int(k): 0 for k in topk}

    for x, _valid in loader:
        x = x.to(device)
        logits = model(x)
        if logits.ndim != 2:
            raise ValueError(f"Model must return logits with shape (B,P); got {tuple(logits.shape)}")
        B, P = logits.shape
        loss = loss_step_select(logits)
        total_nll += float(loss.item()) * int(B)
        total += int(B)

        true_idx = int(P - 1)
        for k in topk:
            k = int(k)
            k_eff = min(k, int(P))
            topk_idx = torch.topk(logits, k=k_eff, dim=1).indices
            correct_topk[k] += int((topk_idx == true_idx).any(dim=1).sum().item())

    if total == 0:
        raise ValueError("Loader yielded zero batches/items; cannot evaluate")

    return {
        "nll": float(total_nll / float(total)),
        "topk": {int(k): float(v) / float(total) for k, v in correct_topk.items()},
    }


def fit_step_select(
    model: torch.nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    *,
    lr: float,
    weight_decay: float,
    max_steps: int,
    eval_interval: int,
    patience: int,
    seed: int,
) -> dict:
    """Step-based training with early stopping on validation NLL."""
    _seed_everything(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state: dict[str, torch.Tensor] | None = None
    best_val_nll = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    train_iter = iter(train_loader)
    for step in range(int(max_steps)):
        try:
            x, _valid = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _valid = next(train_iter)

        model.train()
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_step_select(logits)
        loss.backward()
        opt.step()

        if step % int(eval_interval) == 0:
            train_metrics = eval_step_select(model, train_loader, device, topk=())
            val_metrics = eval_step_select(model, val_loader, device, topk=())
            tr = float(train_metrics["nll"])
            va = float(val_metrics["nll"])
            history.append({"step": float(step), "train_nll": tr, "val_nll": va})

            if va < best_val_nll:
                best_val_nll = va
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(patience):
                    break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = eval_step_select(model, train_loader, device)
    final_val = eval_step_select(model, val_loader, device)

    return {
        "best_step": int(best_step),
        "train": final_train,
        "val": final_val,
        "history": history,
    }

