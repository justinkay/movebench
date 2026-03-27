"""Training utilities for MoveBench2 step-selection benchmarks."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from movebench.lib.moveformer.geo_transformers.training_utils import (
    collate_sequence_dicts,
)
from movebench.lib.moveformer.geo_transformers.training_utils import LinearWarmupLR


class LogCallback(Protocol):
    def __call__(self, metrics: dict[str, float], *, step: int) -> None: ...


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deterministic_eval_subset(loader: Any, *, max_items: int | None, seed: int) -> Any:
    """Return a deterministic, shuffled subset loader for evaluation.

    This is used to speed up frequent validation (early stopping) while keeping
    the evaluated items fixed across epochs/steps.
    """
    if max_items is None:
        return loader
    max_items = int(max_items)

    try:
        ds = loader.dataset
        n = int(len(ds))
        k = min(int(max_items), int(n))

        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(n, generator=g)[:k].tolist()

        subset = Subset(ds, idx)
        return DataLoader(
            subset,
            batch_size=loader.batch_size if loader.batch_size is not None else 1,
            shuffle=False,
            num_workers=loader.num_workers if loader.num_workers is not None else 0,
            collate_fn=loader.collate_fn,
            drop_last=False,
        )
    except Exception:
        return loader


_COLLATE_CHECKED = False


def collate_step_select(
    batch: list[tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]],
) -> tuple[torch.Tensor, torch.Tensor]:
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
            raise ValueError(
                f"Non-finite values found in step features at indices (first 10): {bad[:10].tolist()}"
            )
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
    max_items: int | None = None,
) -> dict:
    model.eval()
    total_nll = 0.0
    total = 0
    correct_topk = {int(k): 0 for k in topk}

    for x, _valid in loader:
        if max_items is not None and int(total) >= int(max_items):
            break
        x = x.to(device)
        logits = model(x)
        if logits.ndim != 2:
            raise ValueError(
                f"Model must return logits with shape (B,P); got {tuple(logits.shape)}"
            )
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


# deprecated
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
    max_eval_steps: int | None = None,
    log_callback: LogCallback | None = None,
) -> dict:
    """Step-based training with early stopping on validation NLL."""
    _seed_everything(seed)
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    eval_train_loader = _deterministic_eval_subset(
        train_loader, max_items=max_eval_steps, seed=int(seed) + 0
    )
    eval_val_loader = _deterministic_eval_subset(
        val_loader, max_items=max_eval_steps, seed=int(seed) + 1
    )

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

        if (step + 1) % int(eval_interval) == 0:
            train_metrics = eval_step_select(
                model, eval_train_loader, device, topk=(), max_items=max_eval_steps
            )
            val_metrics = eval_step_select(
                model, eval_val_loader, device, topk=(), max_items=max_eval_steps
            )
            tr = float(train_metrics["nll"])
            va = float(val_metrics["nll"])
            history.append({"step": float(step), "train_nll": tr, "val_nll": va})

            if log_callback is not None:
                log_callback({"loss/train": tr, "loss/test": va}, step=int(step))

            if va < best_val_nll:
                best_val_nll = va
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(patience):
                    break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = eval_step_select(model, train_loader, device)
    final_val = eval_step_select(model, val_loader, device)

    return {
        "best_step": int(best_step),
        "train": final_train,
        "val": final_val,
        "history": history,
    }


def _use_amp_for_x(x: torch.Tensor, device: torch.device) -> bool:
    return device.type == "cuda" and x.dtype in (torch.float16, torch.bfloat16)


@torch.no_grad()
def eval_step_select_tensor(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    topk: Sequence[int] = (1, 5, 10, 25, 50),
    max_items: int | None = None,
    batch_size: int = 1024,
    idxs: torch.Tensor | None = None,
) -> dict:
    """Evaluate step-selection on a pre-materialized feature tensor.

    Args:
        x: (N,C,P) tensor on CPU or GPU. When on CPU and device is CUDA, it
           should ideally be pinned for fast async copies.
        idxs: optional (K,) indices into x, evaluated in the given order.
        max_items: maximum number of samples to evaluate.
    """
    model.eval()

    if x.ndim != 3:
        raise ValueError(f"Expected x with shape (N,C,P); got {tuple(x.shape)}")
    N, _C, P = x.shape
    if P <= 0:
        raise ValueError(f"P must be > 0; got {P}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")

    if idxs is None:
        idxs = torch.arange(int(N), dtype=torch.long)
    else:
        if idxs.ndim != 1:
            raise ValueError(f"idxs must be 1D; got {tuple(idxs.shape)}")
        idxs = idxs.to(dtype=torch.long, device="cpu")

    if max_items is not None:
        k = min(int(max_items), int(idxs.shape[0]))
        idxs = idxs[:k]

    # Indices must live on the same device as x for index_select / advanced indexing.
    if idxs.device != x.device:
        idxs = idxs.to(device=x.device, non_blocking=False)

    total_nll = 0.0
    total = 0
    correct_topk = {int(k): 0 for k in topk}

    use_amp = _use_amp_for_x(x, device)
    true_idx = int(P - 1)

    for s in range(0, int(idxs.shape[0]), int(batch_size)):
        e = min(int(idxs.shape[0]), s + int(batch_size))
        idx_b = idxs[s:e]

        xb = x.index_select(0, idx_b)
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=xb.dtype):
                logits = model(xb)
                loss = loss_step_select(logits)
        else:
            logits = model(xb)
            loss = loss_step_select(logits)

        if logits.ndim != 2:
            raise ValueError(
                f"Model must return logits with shape (B,P); got {tuple(logits.shape)}"
            )
        B, P_logits = logits.shape
        if int(P_logits) != int(P):
            raise ValueError(f"Model returned P={P_logits}, expected P={P}")

        total_nll += float(loss.item()) * int(B)
        total += int(B)

        if correct_topk:
            for k in topk:
                k = int(k)
                k_eff = min(k, int(P))
                topk_idx = torch.topk(logits, k=k_eff, dim=1).indices
                correct_topk[k] += int((topk_idx == true_idx).any(dim=1).sum().item())

    if total == 0:
        raise ValueError("No items were evaluated (idxs was empty?)")

    return {
        "nll": float(total_nll / float(total)),
        "topk": {int(k): float(v) / float(total) for k, v in correct_topk.items()},
    }


def fit_step_select_tensor(
    model: torch.nn.Module,
    x_train: torch.Tensor,
    x_val: torch.Tensor,
    device: torch.device,
    *,
    lr: float,
    weight_decay: float,
    max_steps: int,
    eval_interval: int,
    patience: int,
    seed: int,
    batch_size: int,
    max_eval_items: int | None = None,
    log_callback: LogCallback | None = None,
) -> dict:
    """Step-based training on cached iSSF features with early stopping.

    This is the default iSSF training path: it avoids DataLoader overhead by
    pre-materializing the (N,C,P) feature tensor once and iterating directly.
    """
    _seed_everything(seed)
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )

    if x_train.ndim != 3 or x_val.ndim != 3:
        raise ValueError(
            f"Expected x_train/x_val with shape (N,C,P); got train={tuple(x_train.shape)} val={tuple(x_val.shape)}"
        )
    if tuple(x_train.shape[1:]) != tuple(x_val.shape[1:]):
        raise ValueError(
            f"x_train/x_val feature shapes must match (C,P). Got train={tuple(x_train.shape[1:])} val={tuple(x_val.shape[1:])}"
        )

    N_train = int(x_train.shape[0])
    N_val = int(x_val.shape[0])
    if N_train <= 0 or N_val <= 0:
        raise ValueError(f"Empty train/val tensors: N_train={N_train} N_val={N_val}")

    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    if int(max_steps) <= 0:
        raise ValueError(f"max_steps must be > 0; got {max_steps}")
    if int(eval_interval) <= 0:
        raise ValueError(f"eval_interval must be > 0; got {eval_interval}")
    if int(patience) <= 0:
        raise ValueError(f"patience must be > 0; got {patience}")

    # Fixed eval subsets for stable early stopping.
    def _eval_idxs(n: int, *, seed_off: int) -> torch.Tensor | None:
        if max_eval_items is None:
            return None
        k = min(int(max_eval_items), int(n))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(seed_off))
        return torch.randperm(int(n), generator=g)[:k].to(
            dtype=torch.long, device="cpu"
        )

    eval_train_idxs = _eval_idxs(N_train, seed_off=0)
    eval_val_idxs = _eval_idxs(N_val, seed_off=1)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_nll = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    use_amp = _use_amp_for_x(x_train, device)
    use_scaler = use_amp and x_train.dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    perm = torch.randperm(N_train, device=x_train.device)
    pos = 0

    for step in range(int(max_steps)):
        if pos >= N_train:
            perm = torch.randperm(N_train, device=x_train.device)
            pos = 0
        end = min(N_train, pos + int(batch_size))
        idx = perm[pos:end]
        pos = end

        xb = x_train.index_select(0, idx)
        if xb.device != device:
            xb = xb.to(device, non_blocking=True)

        model.train()
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=xb.dtype):
                logits = model(xb)
                loss = loss_step_select(logits)
            if use_scaler:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        else:
            logits = model(xb)
            loss = loss_step_select(logits)
            loss.backward()
            opt.step()

        if (step + 1) % int(eval_interval) == 0:
            train_metrics = eval_step_select_tensor(
                model,
                x_train,
                device,
                topk=(),
                max_items=max_eval_items,
                batch_size=int(batch_size),
                idxs=eval_train_idxs,
            )
            val_metrics = eval_step_select_tensor(
                model,
                x_val,
                device,
                topk=(),
                max_items=max_eval_items,
                batch_size=int(batch_size),
                idxs=eval_val_idxs,
            )

            tr = float(train_metrics["nll"])
            va = float(val_metrics["nll"])
            history.append({"step": float(step), "train_nll": tr, "val_nll": va})

            if log_callback is not None:
                log_callback({"loss/train": tr, "loss/test": va}, step=int(step))

            if va < best_val_nll:
                best_val_nll = va
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(patience):
                    break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = eval_step_select_tensor(
        model, x_train, device, batch_size=int(batch_size)
    )
    final_val = eval_step_select_tensor(
        model, x_val, device, batch_size=int(batch_size)
    )

    return {
        "best_step": int(best_step),
        "train": final_train,
        "val": final_val,
        "history": history,
    }


def collate_deepssf(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate DeepSSFDataset items.

    Returns:
        spatial: float32 (B, C, H, W)
        scalars: float32 (B, 4)
        bearing: float32 (B, 1)
        target:  float32 (B, H, W), one-hot when in-bounds, else all zeros
    """
    spatial, scalars, bearing, target = zip(*batch)
    x1 = torch.stack([t.to(dtype=torch.float32) for t in spatial], dim=0)
    x2 = torch.stack([t.to(dtype=torch.float32) for t in scalars], dim=0)
    x3 = torch.stack([t.to(dtype=torch.float32) for t in bearing], dim=0)
    y = torch.stack([t.to(dtype=torch.float32) for t in target], dim=0)
    return x1, x2, x3, y


def _unpack_deepssf_batch(
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) == 4:
        x1, x2, x3, y = batch
        return x1, x2, x3, y
    if len(batch) == 5:
        x1, x2, x3, y, _point_id_start = batch
        return x1, x2, x3, y
    raise ValueError(f"Expected deepSSF batch of length 4 or 5, got {len(batch)}")


def random_grid_baseline_metrics(num_pixels: int, topk: Sequence[int]) -> dict:
    """Deterministic uniform random baseline for grid-based next-step prediction."""
    n = int(num_pixels)
    if n <= 0:
        raise ValueError(f"num_pixels must be > 0; got {num_pixels}")
    return {
        "nll": float(math.log(n)),
        "topk": {int(k): float(min(int(k), n)) / float(n) for k in topk},
    }


@torch.no_grad()
def eval_deepssf(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, ...]],
    device: torch.device,
    *,
    topk: Sequence[int] = (1, 5, 10, 25, 50),
    max_items: int | None = None,
) -> dict:
    """Evaluate deepSSF by treating the (H,W) grid as a classification problem over H*W pixels."""
    model.eval()
    total_nll = 0.0
    total = 0
    correct_topk = {int(k): 0 for k in topk}

    for batch in loader:
        x1, x2, x3, y = _unpack_deepssf_batch(batch)
        if max_items is not None and int(total) >= int(max_items):
            break
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device)

        out = model((x1, x2, x3))
        if out.ndim != 4 or int(out.shape[-1]) != 2:
            raise ValueError(
                f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
            )

        scores = out[..., 0] + out[..., 1]  # (B,H,W)
        B, H, W = scores.shape
        scores_flat = scores.view(int(B), int(H) * int(W))
        y_flat = y.view(int(B), int(H) * int(W))

        t_sum = y_flat.sum(dim=1)
        valid = (t_sum > 0.5) & (t_sum < 1.5)
        if not valid.any().item():
            continue

        logits = scores_flat[valid]
        true_idx = torch.argmax(y_flat[valid], dim=1)

        loss = F.cross_entropy(logits, true_idx, reduction="sum")
        total_nll += float(loss.item())
        total += int(valid.sum().item())

        for k in topk:
            k = int(k)
            k_eff = min(k, int(logits.shape[1]))
            top_idx = torch.topk(logits, k=k_eff, dim=1).indices
            correct_topk[k] += int(
                (top_idx == true_idx.unsqueeze(1)).any(dim=1).sum().item()
            )

    if total == 0:
        raise ValueError(
            "No valid deepSSF targets were found in the loader (all targets were out-of-bounds)."
        )

    return {
        "nll": float(total_nll / float(total)),
        "n_valid": int(total),
        "topk": {int(k): float(v) / float(total) for k, v in correct_topk.items()},
    }


def fit_deepssf(
    model: torch.nn.Module,
    train_loader: Iterable[tuple[torch.Tensor, ...]],
    val_loader: Iterable[tuple[torch.Tensor, ...]],
    device: torch.device,
    *,
    lr_movement: float,
    lr_habitat: float,
    weight_decay: float,
    max_steps: int,
    eval_interval: int,
    patience: int,
    seed: int,
    max_eval_steps: int | None = None,
    log_callback: LogCallback | None = None,
) -> dict:
    """Step-based training with early stopping on validation NLL."""
    _seed_everything(seed)
    model.to(device)

    eval_train_loader = _deterministic_eval_subset(
        train_loader, max_items=max_eval_steps, seed=int(seed) + 0
    )
    eval_val_loader = _deterministic_eval_subset(
        val_loader, max_items=max_eval_steps, seed=int(seed) + 1
    )

    if not hasattr(model, "parameter_groups"):
        raise TypeError("fit_deepssf expects a model with a parameter_groups() method")
    groups = model.parameter_groups()  # type: ignore[attr-defined]
    if (
        not isinstance(groups, dict)
        or "movement" not in groups
        or "habitat" not in groups
    ):
        raise TypeError(
            "model.parameter_groups() must return a dict with keys {'movement','habitat'}"
        )

    opt = torch.optim.Adam(
        [
            {"params": list(groups["movement"]), "lr": float(lr_movement)},
            {"params": list(groups["habitat"]), "lr": float(lr_habitat)},
        ],
        weight_decay=float(weight_decay),
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_nll = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    train_iter = iter(train_loader)
    for step in range(int(max_steps)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        x1, x2, x3, y = _unpack_deepssf_batch(batch)

        model.train()
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device)

        out = model((x1, x2, x3))
        if out.ndim != 4 or int(out.shape[-1]) != 2:
            raise ValueError(
                f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
            )

        scores = out[..., 0] + out[..., 1]
        B, H, W = scores.shape
        scores_flat = scores.view(int(B), int(H) * int(W))
        y_flat = y.view(int(B), int(H) * int(W))

        t_sum = y_flat.sum(dim=1)
        valid = (t_sum > 0.5) & (t_sum < 1.5)
        if not valid.any().item():
            continue

        logits = scores_flat[valid]
        true_idx = torch.argmax(y_flat[valid], dim=1)

        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, true_idx, reduction="mean")
        loss.backward()
        opt.step()

        PRINT = True
        if (step + 1) % int(eval_interval) == 0:
            train_metrics = eval_deepssf(
                model, eval_train_loader, device, topk=(), max_items=max_eval_steps
            )
            val_metrics = eval_deepssf(
                model, eval_val_loader, device, topk=(), max_items=max_eval_steps
            )
            tr = float(train_metrics["nll"])
            va = float(val_metrics["nll"])
            history.append({"step": float(step), "train_nll": tr, "val_nll": va})

            if PRINT:
                print(history[-1])

            if log_callback is not None:
                log_callback({"loss/train": tr, "loss/test": va}, step=int(step))

            if va < best_val_nll:
                best_val_nll = va
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(patience):
                    break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = eval_deepssf(model, train_loader, device)
    final_val = eval_deepssf(model, val_loader, device)

    return {
        "best_step": int(best_step),
        "train": final_train,
        "val": final_val,
        "history": history,
    }


def collate_moveformer(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate MoveFormer segment dicts using upstream padding behavior."""

    return collate_sequence_dicts(batch)


@torch.no_grad()
def eval_moveformer(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    topk: Sequence[int] = (1, 5, 10, 25, 50),
    max_items: int | None = None,
) -> dict:
    """Evaluate MoveFormer selection head outputs.

    Reports standard cross-entropy (nll) and top-k accuracy with target index 0.
    Also reports normalized cross-entropy (nce = nll / log(P)).
    """
    model.eval()
    total_nll = 0.0
    total_items = 0
    total_tokens = 0
    correct_topk = {int(k): 0 for k in topk}

    for batch in loader:
        if max_items is not None and int(total_items) >= int(max_items):
            break

        features = {k: v.to(device) for k, v in batch.items()}
        logits, _loss, _metrics = model(features)
        if logits.ndim != 3:
            raise ValueError(
                f"Expected logits with shape (B,L,P); got {tuple(logits.shape)}"
            )
        B, L, P = logits.shape
        if P <= 0:
            raise ValueError(f"P must be > 0; got {P}")
        total_items += int(B)

        mask = features.get("loss_mask", features["mask"]).to(torch.bool)
        target = torch.zeros((B, L), dtype=torch.long, device=device)
        ce = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")  # (B,L)

        mask_f = mask.to(ce.dtype)
        n_valid = int(mask.sum().item())
        if n_valid == 0:
            continue

        total_nll += float((ce * mask_f).sum().item())
        total_tokens += n_valid

        for k in topk:
            k = int(k)
            k_eff = min(k, int(P))
            top_idx = logits.topk(k_eff, dim=-1).indices  # (B,L,k)
            correct = (top_idx == 0).any(dim=-1) & mask  # (B,L)
            correct_topk[k] += int(correct.sum().item())

    if total_tokens == 0:
        raise ValueError("Loader yielded zero valid tokens; cannot evaluate")

    nll = float(total_nll / float(total_tokens))
    return {
        "nll": nll,
        "nce": float(nll / float(math.log(int(P)))),
        "topk": {
            int(k): float(v) / float(total_tokens) for k, v in correct_topk.items()
        },
        "num_items": int(total_items),
        "num_tokens": int(total_tokens),
    }


def fit_moveformer(
    model: torch.nn.Module,
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    lr: float = 5e-5,
    weight_decay: float = 0.0,
    warmup_steps: int = 1000,
    lr_gamma: float = 0.98,
    max_steps: int,
    eval_interval: int,
    patience: int,
    seed: int,
    max_eval_steps: int | None = None,
    log_callback: LogCallback | None = None,
) -> dict:
    """Step-based training with early stopping on validation NLL."""

    _seed_everything(seed)
    model.to(device)

    opt = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    lr_decay = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=float(lr_gamma))
    lr_warmup = LinearWarmupLR(opt, warmup_steps=int(warmup_steps))

    eval_train_loader = _deterministic_eval_subset(
        train_loader, max_items=max_eval_steps, seed=int(seed) + 0
    )
    eval_val_loader = _deterministic_eval_subset(
        val_loader, max_items=max_eval_steps, seed=int(seed) + 1
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_nll = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    # Used to mimic MoveFormer's var_len_training_v2 attention mask behavior.
    var_len_training = bool(getattr(model, "var_len_training", False))
    var_len_training_v2 = bool(getattr(model, "var_len_training_v2", False))
    max_len_cfg = getattr(model, "max_len", None)

    train_iter = iter(train_loader)
    steps_per_epoch = None
    try:
        steps_per_epoch = len(train_loader)  # type: ignore[arg-type]
    except Exception:
        steps_per_epoch = None

    for step in range(int(max_steps)):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        features = {k: v.to(device) for k, v in batch.items()}
        seq_len = int(features["mask"].shape[1])
        max_len = int(max_len_cfg) if max_len_cfg is not None else seq_len

        attn_mask = None
        if var_len_training:
            split_idx = torch.randint(low=1, high=max_len + 1, size=(), device=device)
            if int(split_idx.item()) < seq_len:
                attn_mask = torch.zeros(
                    (seq_len, seq_len), dtype=torch.bool, device=device
                )
                attn_mask[: int(split_idx), : int(split_idx)] = True
                attn_mask[int(split_idx) :, int(split_idx) :] = True
        elif var_len_training_v2:
            ctx_len = torch.randint(low=1, high=max_len + 1, size=(), device=device)
            ctx_len_i = int(ctx_len.item())
            n_blocks = (seq_len + ctx_len_i - 1) // ctx_len_i
            attn_mask = torch.eye(n_blocks, dtype=torch.bool, device=device)
            attn_mask = attn_mask.repeat_interleave(ctx_len_i, dim=0)
            attn_mask = attn_mask.repeat_interleave(ctx_len_i, dim=1)
            attn_mask = attn_mask[:seq_len, :seq_len]

        model.train()
        opt.zero_grad(set_to_none=True)
        _logits, loss, _metrics = model(
            features,
            encoder_kwargs={"attn_mask": attn_mask} if attn_mask is not None else None,
            decoder_kwargs={"attn_mask": attn_mask} if attn_mask is not None else None,
        )
        loss.backward()
        opt.step()
        lr_warmup.step()

        if steps_per_epoch is not None and (step + 1) % int(steps_per_epoch) == 0:
            lr_decay.step()

        if (step + 1) % int(eval_interval) == 0:
            train_metrics = eval_moveformer(
                model, eval_train_loader, device, topk=(), max_items=max_eval_steps
            )
            val_metrics = eval_moveformer(
                model, eval_val_loader, device, topk=(), max_items=max_eval_steps
            )
            tr = float(train_metrics["nll"])
            va = float(val_metrics["nll"])
            history.append({"step": float(step), "train_nll": tr, "val_nll": va})

            if log_callback is not None:
                log_callback({"loss/train": tr, "loss/test": va}, step=int(step))

            if va < best_val_nll:
                best_val_nll = va
                best_step = step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(patience):
                    break

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    final_train = eval_moveformer(model, train_loader, device)
    final_val = eval_moveformer(model, val_loader, device)

    return {
        "best_step": int(best_step),
        "best_val_nll": float(best_val_nll),
        "train": final_train,
        "val": final_val,
        "history": history,
    }
