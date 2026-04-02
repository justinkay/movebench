from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _as_float64_tensor(value: Any) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _as_truth_xy(truth_xy_m: np.ndarray | torch.Tensor | list[float]) -> torch.Tensor:
    truth = _as_float64_tensor(truth_xy_m).reshape(-1)
    if int(truth.numel()) != 2:
        raise ValueError(f"truth_xy_m must contain exactly two values, got {tuple(truth.shape)}")
    return truth


def _scalar(value: Any, *, field: str) -> torch.Tensor:
    out = _as_float64_tensor(value).reshape(-1)
    if int(out.numel()) != 1:
        raise ValueError(f"{field} must be a scalar, got {tuple(out.shape)}")
    return out


def _validate_probs_1d(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.reshape(-1)
    if probs.ndim != 1:
        raise ValueError("probs must be rank-1")
    if int(probs.numel()) <= 0:
        raise ValueError("probs must be non-empty")
    if not bool(torch.isfinite(probs).all().item()):
        raise ValueError("probs must be finite")
    if not bool((probs >= 0.0).all().item()):
        raise ValueError("probs must be non-negative")
    total = probs.sum()
    if not bool(torch.isfinite(total).item()) or float(total) <= 0.0:
        raise ValueError("probs must have positive finite mass")
    return probs / total


def _validate_coords_2d(coords_m: torch.Tensor) -> torch.Tensor:
    coords_m = _as_float64_tensor(coords_m)
    if coords_m.ndim != 2 or int(coords_m.shape[1]) != 2:
        raise ValueError(f"coords_m must have shape (N,2), got {tuple(coords_m.shape)}")
    if int(coords_m.shape[0]) <= 0:
        raise ValueError("coords_m must be non-empty")
    if not bool(torch.isfinite(coords_m).all().item()):
        raise ValueError("coords_m must be finite")
    return coords_m


def _weighted_pairwise_l2(coords_m: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    n = int(coords_m.shape[0])
    total = torch.tensor(0.0, dtype=torch.float64)
    block_size = min(256, n)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        diff = coords_m[start:stop].unsqueeze(1) - coords_m.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff * diff, dim=2))
        weights = probs[start:stop].view(-1, 1) * probs.view(1, -1)
        total = total + torch.sum(distances * weights)
    return total


def _regular_step(coords: torch.Tensor, *, field: str) -> float:
    coords = _as_float64_tensor(coords).reshape(-1)
    if int(coords.numel()) <= 0:
        raise ValueError(f"{field} must be non-empty")
    if not bool(torch.isfinite(coords).all().item()):
        raise ValueError(f"{field} must be finite")
    if int(coords.numel()) == 1:
        return 0.0
    deltas = torch.diff(coords)
    if not bool(torch.isclose(deltas, deltas[0], atol=1e-4, rtol=1e-4).all().item()):
        raise ValueError(f"{field} must be evenly spaced")
    return float(deltas[0].abs().item())


def _energy_terms_deterministic(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> tuple[float, float, float]:
    truth = _as_truth_xy(truth_xy_m)
    dx_m = _scalar(pred["dx_m"], field="dx_m")
    dy_m = _scalar(pred["dy_m"], field="dy_m")
    point = torch.stack([dx_m[0], dy_m[0]])
    accuracy = torch.linalg.norm(point - truth)
    calibration = torch.tensor(0.0, dtype=torch.float64)
    score = accuracy
    return float(score), float(accuracy), float(calibration)


def _energy_terms_discrete(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> tuple[float, float, float]:
    truth = _as_truth_xy(truth_xy_m)
    coords_m = _validate_coords_2d(pred["coords_m"])
    probs = _validate_probs_1d(_as_float64_tensor(pred["probs"]))
    if int(coords_m.shape[0]) != int(probs.shape[0]):
        raise ValueError(
            f"coords_m and probs must agree on support size, got {tuple(coords_m.shape)} and {tuple(probs.shape)}"
        )

    accuracy = torch.matmul(torch.linalg.norm(coords_m - truth.view(1, 2), dim=1), probs)
    calibration = _weighted_pairwise_l2(coords_m, probs)
    score = accuracy - 0.5 * calibration
    return float(score), float(accuracy), float(calibration)


def _energy_terms_grid(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> tuple[float, float, float]:
    truth = _as_truth_xy(truth_xy_m)
    probs_hw = _as_float64_tensor(pred["probs_hw"])
    if probs_hw.ndim != 2 or int(probs_hw.shape[0]) <= 0 or int(probs_hw.shape[1]) <= 0:
        raise ValueError(f"probs_hw must have shape (H,W), got {tuple(probs_hw.shape)}")
    if not bool(torch.isfinite(probs_hw).all().item()):
        raise ValueError("probs_hw must be finite")
    if not bool((probs_hw >= 0.0).all().item()):
        raise ValueError("probs_hw must be non-negative")
    total = probs_hw.sum()
    if not bool(torch.isclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-8, rtol=1e-8).item()):
        raise ValueError(f"probs_hw must sum to 1, got {float(total):.12f}")

    height, width = int(probs_hw.shape[0]), int(probs_hw.shape[1])
    x_coords_m = _as_float64_tensor(pred["x_coords_m"]).reshape(-1)
    y_coords_m = _as_float64_tensor(pred["y_coords_m"]).reshape(-1)
    if int(x_coords_m.numel()) != width:
        raise ValueError(
            f"x_coords_m length must match probs_hw width, got {int(x_coords_m.numel())} and {width}"
        )
    if int(y_coords_m.numel()) != height:
        raise ValueError(
            f"y_coords_m length must match probs_hw height, got {int(y_coords_m.numel())} and {height}"
        )

    x_step_m = _regular_step(x_coords_m, field="x_coords_m")
    y_step_m = _regular_step(y_coords_m, field="y_coords_m")
    grid_x = x_coords_m.view(1, -1).expand(height, width)
    grid_y = y_coords_m.view(-1, 1).expand(height, width)
    accuracy = torch.sum(
        probs_hw * torch.sqrt((grid_x - truth[0]) ** 2 + (grid_y - truth[1]) ** 2)
    )

    row_offsets_m = torch.arange(
        -(height - 1),
        height,
        dtype=torch.float64,
    ) * y_step_m
    col_offsets_m = torch.arange(
        -(width - 1),
        width,
        dtype=torch.float64,
    ) * x_step_m
    kernel = torch.sqrt(
        row_offsets_m.view(-1, 1) ** 2 + col_offsets_m.view(1, -1) ** 2
    )
    padded_shape = (2 * height - 1, 2 * width - 1)
    corr = torch.fft.ifftn(
        torch.fft.fftn(probs_hw, s=padded_shape)
        * torch.fft.fftn(torch.flip(probs_hw, dims=(0, 1)), s=padded_shape)
    ).real
    corr = corr.clamp_min(0.0)
    calibration = torch.sum(corr * kernel)
    score = accuracy - 0.5 * calibration
    return float(score), float(accuracy), float(calibration)


def energy_score_deterministic(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> float:
    return _energy_terms_deterministic(pred, truth_xy_m)[0]


def energy_score_discrete(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> float:
    return _energy_terms_discrete(pred, truth_xy_m)[0]


def energy_score_grid(
    pred: dict[str, object],
    truth_xy_m: np.ndarray | torch.Tensor | list[float],
) -> float:
    return _energy_terms_grid(pred, truth_xy_m)[0]
