from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader, Dataset

from movebench.datasets.chips import ChipDataset
from movebench.trajectories import (
    SET_COL,
    canonicalize_split_key,
    fit_interval_step_support,
    select_step_starts_with_support,
)

GRID_QUADRATURE_SUBCELLS = 2
MIXED_MARGINAL_QUANTILES = 256


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def selected_splits_from_manifest(manifest) -> list[str] | None:
    raw_selected_splits = getattr(manifest, "extraction", {}).get("selected_splits")
    if not isinstance(raw_selected_splits, list) or not raw_selected_splits:
        return None
    selected_splits: list[str] = []
    seen: set[str] = set()
    for raw_split in raw_selected_splits:
        split = canonicalize_split_key(raw_split)
        if not split or split in seen:
            continue
        selected_splits.append(split)
        seen.add(split)
    return selected_splits or None


def fit_train_step_support(
    trajectories_df,
    *,
    step_spec,
) -> tuple[object, object]:
    train_df = trajectories_df[trajectories_df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Study has no train split for SSF grid prediction support")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        step_q=float(step_spec.step_q),
        max_step_q=float(step_spec.max_step_q),
        step_multiplier=float(step_spec.step_multiplier),
    )
    train_steps = select_step_starts_with_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        support=support,
    )
    return support, train_steps


class SSFGridStepDataset(Dataset):
    def __init__(self, chips: ChipDataset, *, include_env: bool) -> None:
        if getattr(chips, "mode", None) != "steps":
            raise ValueError("SSFGridStepDataset requires ChipDataset(mode='steps')")
        self.chips = chips
        self.include_env = bool(include_env)
        self.output_hw = tuple(int(v) for v in chips.output_hw)
        self.pixel_size_m = float(chips.pixel_size_m)
        steps = chips.step_starts_df.reset_index(drop=True)
        self._point_id_start = steps["point_id_start"].to_numpy(dtype=np.int64, copy=False)
        self._timestamp_ns = steps["timestamp"].to_numpy(dtype="datetime64[ns]").astype(
            np.int64, copy=False
        )
        self._start_bearing_rad = steps["start_bearing"].to_numpy(
            dtype=np.float32, copy=False
        )
        self._individual_id = steps["individual-local-identifier"].astype(str).tolist()

    def __len__(self) -> int:
        return int(len(self.chips))

    def __getitem__(self, idx: int) -> int:
        return int(idx)

    def collate(self, indices):
        idxs = [int(idx) for idx in indices]
        env = None
        valid = None
        if self.include_env:
            env_rows = []
            valid_rows = []
            for idx in idxs:
                env_i, valid_i, _target, _xfm, _crs = self.chips[idx]
                env_rows.append(env_i)
                valid_rows.append(valid_i)
            env = torch.stack(env_rows, dim=0)
            valid = torch.stack(valid_rows, dim=0)
        return {
            "env": env,
            "valid": valid,
            "point_id_start": torch.as_tensor(
                self._point_id_start[idxs],
                dtype=torch.int64,
            ),
            "timestamp_ns": torch.as_tensor(
                self._timestamp_ns[idxs],
                dtype=torch.int64,
            ),
            "start_bearing_rad": torch.as_tensor(
                self._start_bearing_rad[idxs],
                dtype=torch.float32,
            ),
            "individual_id": [self._individual_id[idx] for idx in idxs],
        }


def make_grid_step_dataset(
    *,
    study_dir: str | Path,
    trajectories_path: str | Path,
    split_key: str,
    plan,
    sampler_config,
    output_hw: tuple[int, int],
    selected_splits: list[str] | None,
    include_env: bool,
) -> SSFGridStepDataset:
    chips = ChipDataset(
        study_dir=study_dir,
        trajectories_path=trajectories_path,
        split=str(split_key),
        plan=plan,
        sampler_config=sampler_config,
        output_hw=output_hw,
        normalize=bool(include_env),
        mode="steps",
        selected_splits=selected_splits,
    )
    return SSFGridStepDataset(chips, include_env=include_env)


def make_grid_loader(
    dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=dataset.collate,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
    )


def scaled_step_batch_size(
    *,
    batch_size: int,
    train_points_per_step: int,
    output_hw: tuple[int, int],
) -> int:
    grid_points_per_step = int(output_hw[0]) * int(output_hw[1])
    scaled_batch = int(int(batch_size) * int(train_points_per_step) / grid_points_per_step)
    return max(1, min(int(batch_size), scaled_batch))


def _replay_sampler_state(
    *,
    train_manifest,
    train_steps,
) -> dict[str, Any]:
    valid_lengths = train_steps["step_length_m"].to_numpy(dtype=np.float64)
    valid_lengths = valid_lengths[np.isfinite(valid_lengths) & (valid_lengths > 0.0)]
    valid_turns = train_steps["turn_angle_rad"].to_numpy(dtype=np.float64)
    valid_turns = valid_turns[np.isfinite(valid_turns)]
    if len(valid_lengths) == 0:
        raise ValueError("No valid train step lengths available for SSF grid prediction")
    if len(valid_turns) == 0:
        raise ValueError("No valid train turn angles available for SSF grid prediction")

    sampler = dict(train_manifest.sampler)
    gamma = dict(sampler.get("gamma", {})) if isinstance(sampler.get("gamma"), dict) else {}
    gamma_params = gamma.get("params")
    if not (bool(gamma.get("used", False)) and isinstance(gamma_params, list) and gamma_params):
        gamma_params = None
    else:
        gamma_params = tuple(float(x) for x in gamma_params)

    vonmises = dict(sampler.get("vonmises", {})) if isinstance(sampler.get("vonmises"), dict) else {}
    vonmises_params = vonmises.get("params")
    if not (
        bool(vonmises.get("used", False))
        and isinstance(vonmises_params, list)
        and vonmises_params
    ):
        vonmises_params = None
    else:
        vonmises_params = tuple(float(x) for x in vonmises_params)

    return {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(sampler["num_candidates"]),
        "random_seed": int(sampler["random_seed"]),
        "gamma_used": gamma_params is not None,
        "gamma_params": gamma_params,
        "vonmises_used": vonmises_params is not None,
        "vonmises_params": vonmises_params,
    }


@lru_cache(maxsize=32)
def _grid_centers_np(
    height: int,
    width: int,
    pixel_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_coords = np.asarray(
        [(col - (width // 2)) * float(pixel_size_m) for col in range(width)],
        dtype=np.float64,
    )
    y_coords = np.asarray(
        [((height // 2) - row) * float(pixel_size_m) for row in range(height)],
        dtype=np.float64,
    )
    return np.meshgrid(x_coords, y_coords, indexing="xy")


@lru_cache(maxsize=32)
def _quadrature_geometry_np(
    height: int,
    width: int,
    pixel_size_m: float,
    subcells: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_hw, y_hw = _grid_centers_np(height, width, pixel_size_m)
    offsets = (
        (np.arange(int(subcells), dtype=np.float64) + 0.5) / float(subcells) - 0.5
    ) * float(pixel_size_m)
    x = x_hw[:, :, None, None] + offsets[None, None, :, None]
    y = y_hw[:, :, None, None] + offsets[None, None, None, :]
    return (
        np.sqrt((x * x) + (y * y)).reshape(-1),
        np.arctan2(x, y).reshape(-1),
    )


def movement_features_flat(
    *,
    start_bearing_rad: torch.Tensor,
    output_hw: tuple[int, int],
    pixel_size_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = (int(output_hw[0]), int(output_hw[1]))
    x_hw, y_hw = _grid_centers_np(height, width, float(pixel_size_m))
    r_flat = np.sqrt((x_hw * x_hw) + (y_hw * y_hw)).reshape(-1)
    phi_flat = np.arctan2(x_hw, y_hw).reshape(-1)
    start = torch.as_tensor(start_bearing_rad, dtype=torch.float64).reshape(-1)
    turn = _wrap_to_pi(phi_flat[None, :] - start.cpu().numpy()[:, None])
    step_length_m = torch.as_tensor(
        np.broadcast_to(r_flat, (int(start.shape[0]), int(r_flat.shape[0]))).copy(),
        dtype=torch.float32,
    )
    turn_angle_rad = torch.as_tensor(turn, dtype=torch.float32)
    return step_length_m, turn_angle_rad


def _parametric_raw_cell_mass(
    *,
    start_bearing_rad: np.ndarray,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    sampler_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(output_hw[0]), int(output_hw[1]))
    r_flat, phi_flat = _quadrature_geometry_np(
        height,
        width,
        float(pixel_size_m),
        GRID_QUADRATURE_SUBCELLS,
    )
    length_pdf = stats.gamma.pdf(r_flat, *sampler_state["gamma_params"])
    turn = _wrap_to_pi(phi_flat[None, :] - start_bearing_rad[:, None])
    turn_pdf = stats.vonmises.pdf(turn, *sampler_state["vonmises_params"])
    density = np.divide(
        length_pdf[None, :] * turn_pdf,
        r_flat[None, :],
        out=np.zeros_like(turn_pdf, dtype=np.float64),
        where=r_flat[None, :] > 0.0,
    )
    raw = density.reshape(
        int(start_bearing_rad.shape[0]),
        height,
        width,
        GRID_QUADRATURE_SUBCELLS * GRID_QUADRATURE_SUBCELLS,
    )
    raw = raw.mean(axis=3) * (float(pixel_size_m) ** 2)
    off_grid_mass = np.maximum(0.0, 1.0 - raw.sum(axis=(1, 2)))
    return raw, off_grid_mass


def _empirical_raw_cell_mass(
    *,
    start_bearing_rad: np.ndarray,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    sampler_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(output_hw[0]), int(output_hw[1]))
    buffer_m = (float(pixel_size_m) * float(width - 1)) / 2.0
    length_values, length_counts = np.unique(sampler_state["valid_lengths"], return_counts=True)
    turn_values, turn_counts = np.unique(sampler_state["valid_turns"], return_counts=True)
    return _weighted_endpoint_raw_cell_mass(
        start_bearing_rad=start_bearing_rad,
        output_hw=output_hw,
        pixel_size_m=pixel_size_m,
        length_values=length_values.astype(np.float64, copy=False),
        length_probs=length_counts.astype(np.float64) / float(length_counts.sum()),
        turn_values=turn_values.astype(np.float64, copy=False),
        turn_probs=turn_counts.astype(np.float64) / float(turn_counts.sum()),
        height=height,
        width=width,
        buffer_m=buffer_m,
    )


def _weighted_endpoint_raw_cell_mass(
    *,
    start_bearing_rad: np.ndarray,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    length_values: np.ndarray,
    length_probs: np.ndarray,
    turn_values: np.ndarray,
    turn_probs: np.ndarray,
    height: int | None = None,
    width: int | None = None,
    buffer_m: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if height is None or width is None:
        height, width = (int(output_hw[0]), int(output_hw[1]))
    else:
        height, width = (int(height), int(width))
    if buffer_m is None:
        buffer_m = (float(pixel_size_m) * float(width - 1)) / 2.0
    length_values = np.asarray(length_values, dtype=np.float64)
    length_probs = np.asarray(length_probs, dtype=np.float64)
    turn_values = np.asarray(turn_values, dtype=np.float64)
    turn_probs = np.asarray(turn_probs, dtype=np.float64)
    headings = _wrap_to_pi(start_bearing_rad[:, None] + turn_values[None, :])
    sin_heading = np.sin(headings)
    cos_heading = np.cos(headings)
    raw = np.zeros((int(start_bearing_rad.shape[0]), height, width), dtype=np.float64)
    off_grid_mass = np.zeros((int(start_bearing_rad.shape[0]),), dtype=np.float64)
    length_chunk = 256
    for start_idx in range(0, int(length_values.shape[0]), length_chunk):
        stop_idx = min(int(length_values.shape[0]), start_idx + length_chunk)
        lengths = length_values[start_idx:stop_idx]
        pair_weights = (
            length_probs[start_idx:stop_idx, None] * turn_probs[None, :]
        ).reshape(-1)
        for batch_idx in range(int(start_bearing_rad.shape[0])):
            x = (lengths[:, None] * sin_heading[batch_idx][None, :]).reshape(-1)
            y = (lengths[:, None] * cos_heading[batch_idx][None, :]).reshape(-1)
            rr = np.floor(
                (float(buffer_m) + 0.5 * float(pixel_size_m) - y)
                / float(pixel_size_m)
            ).astype(np.int64)
            cc = np.floor(
                (x + float(buffer_m) + 0.5 * float(pixel_size_m))
                / float(pixel_size_m)
            ).astype(np.int64)
            in_window = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            if in_window.any():
                flat = raw[batch_idx].reshape(-1)
                np.add.at(
                    flat,
                    rr[in_window] * width + cc[in_window],
                    pair_weights[in_window],
                )
            off_grid_mass[batch_idx] += float(pair_weights[~in_window].sum())
    return raw, off_grid_mass


def _continuous_quantile_values(
    *,
    distribution: str,
    params: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = (
        np.arange(int(MIXED_MARGINAL_QUANTILES), dtype=np.float64) + 0.5
    ) / float(MIXED_MARGINAL_QUANTILES)
    if distribution == "gamma":
        values = stats.gamma.ppf(quantiles, *params)
    elif distribution == "vonmises":
        values = stats.vonmises.ppf(quantiles, *params)
    else:
        raise ValueError(f"Unsupported mixed marginal distribution {distribution!r}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite quantiles for mixed {distribution} marginal")
    return (
        np.asarray(values, dtype=np.float64),
        np.full(int(MIXED_MARGINAL_QUANTILES), 1.0 / float(MIXED_MARGINAL_QUANTILES)),
    )


def _mixed_parametric_length_raw_cell_mass(
    *,
    start_bearing_rad: np.ndarray,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    sampler_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    length_values, length_probs = _continuous_quantile_values(
        distribution="gamma",
        params=tuple(sampler_state["gamma_params"]),
    )
    turn_values, turn_counts = np.unique(sampler_state["valid_turns"], return_counts=True)
    turn_probs = turn_counts.astype(np.float64) / float(turn_counts.sum())
    return _weighted_endpoint_raw_cell_mass(
        start_bearing_rad=start_bearing_rad,
        output_hw=output_hw,
        pixel_size_m=pixel_size_m,
        length_values=length_values,
        length_probs=length_probs,
        turn_values=turn_values.astype(np.float64, copy=False),
        turn_probs=turn_probs,
    )


def _mixed_parametric_turn_raw_cell_mass(
    *,
    start_bearing_rad: np.ndarray,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    sampler_state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    turn_values, turn_probs = _continuous_quantile_values(
        distribution="vonmises",
        params=tuple(sampler_state["vonmises_params"]),
    )
    length_values, length_counts = np.unique(sampler_state["valid_lengths"], return_counts=True)
    length_probs = length_counts.astype(np.float64) / float(length_counts.sum())
    return _weighted_endpoint_raw_cell_mass(
        start_bearing_rad=start_bearing_rad,
        output_hw=output_hw,
        pixel_size_m=pixel_size_m,
        length_values=length_values.astype(np.float64, copy=False),
        length_probs=length_probs,
        turn_values=turn_values,
        turn_probs=turn_probs,
    )


def movement_log_prior_hw(
    *,
    start_bearing_rad: torch.Tensor,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    sampler_state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    start = torch.as_tensor(start_bearing_rad, dtype=torch.float64).reshape(-1).cpu().numpy()
    if (
        bool(sampler_state.get("gamma_used", False))
        and sampler_state.get("gamma_params") is not None
        and bool(sampler_state.get("vonmises_used", False))
        and sampler_state.get("vonmises_params") is not None
    ):
        raw, off_grid_mass = _parametric_raw_cell_mass(
            start_bearing_rad=start,
            output_hw=output_hw,
            pixel_size_m=float(pixel_size_m),
            sampler_state=sampler_state,
        )
    elif bool(sampler_state.get("gamma_used", False)) and sampler_state.get(
        "gamma_params"
    ) is not None:
        raw, off_grid_mass = _mixed_parametric_length_raw_cell_mass(
            start_bearing_rad=start,
            output_hw=output_hw,
            pixel_size_m=float(pixel_size_m),
            sampler_state=sampler_state,
        )
    elif bool(sampler_state.get("vonmises_used", False)) and sampler_state.get(
        "vonmises_params"
    ) is not None:
        raw, off_grid_mass = _mixed_parametric_turn_raw_cell_mass(
            start_bearing_rad=start,
            output_hw=output_hw,
            pixel_size_m=float(pixel_size_m),
            sampler_state=sampler_state,
        )
    else:
        raw, off_grid_mass = _empirical_raw_cell_mass(
            start_bearing_rad=start,
            output_hw=output_hw,
            pixel_size_m=float(pixel_size_m),
            sampler_state=sampler_state,
        )

    in_window_mass = raw.sum(axis=(1, 2), keepdims=True)
    if np.any(in_window_mass <= 0.0):
        raise ValueError("Movement prior assigned zero mass to the benchmark grid")

    probs_hw = raw / in_window_mass
    log_q_hw = np.full_like(probs_hw, -np.inf, dtype=np.float64)
    positive = probs_hw > 0.0
    log_q_hw[positive] = np.log(probs_hw[positive])
    return (
        torch.as_tensor(log_q_hw, dtype=torch.float32),
        torch.as_tensor(off_grid_mass, dtype=torch.float32),
    )


def _move_batch_to_device(batch: dict[str, torch.Tensor | None], device: torch.device):
    out: dict[str, torch.Tensor | None] = {}
    for key, value in batch.items():
        if value is None:
            out[key] = None
        else:
            out[key] = value.to(device, non_blocking=device.type == "cuda")
    return out


@torch.no_grad()
def build_ssf_grid_artifact(
    *,
    grid_dataset,
    train_manifest,
    train_steps,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    model: torch.nn.Module | None,
    feature_adapter,
) -> dict[str, object]:
    height, width = (int(grid_dataset.output_hw[0]), int(grid_dataset.output_hw[1]))
    if len(grid_dataset) == 0:
        return {
            "prediction_kind": "grid",
            "point_id_start": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0, height, width), dtype=torch.float32),
            "pixel_size_m": float(grid_dataset.pixel_size_m),
        }

    sampler_state = _replay_sampler_state(
        train_manifest=train_manifest,
        train_steps=train_steps,
    )
    step_batch_size = scaled_step_batch_size(
        batch_size=batch_size,
        train_points_per_step=int(train_manifest.sampler["num_candidates"]) + 1,
        output_hw=grid_dataset.output_hw,
    )
    loader = make_grid_loader(
        grid_dataset,
        batch_size=step_batch_size,
        num_workers=num_workers,
        device=device,
    )

    scores_chunks: list[torch.Tensor] = []
    point_id_chunks: list[torch.Tensor] = []
    if model is not None:
        if feature_adapter is None:
            raise ValueError("feature_adapter is required when model is provided")
        model.eval()

    for batch in loader:
        point_id_start = batch["point_id_start"].to(dtype=torch.int64).cpu()
        log_q_hw, _off_grid_mass = movement_log_prior_hw(
            start_bearing_rad=batch["start_bearing_rad"],
            output_hw=grid_dataset.output_hw,
            pixel_size_m=float(grid_dataset.pixel_size_m),
            sampler_state=sampler_state,
        )
        scores_hw = log_q_hw
        if model is not None:
            env = batch["env"]
            valid = batch["valid"]
            if env is None or valid is None:
                raise ValueError("grid dataset is missing env/valid required for model scoring")
            n_steps = int(point_id_start.shape[0])
            flat_env = env.view(n_steps, int(env.shape[1]), height * width)
            flat_valid = valid.view(n_steps, int(valid.shape[1]), height * width)
            step_length_m, turn_angle_rad = movement_features_flat(
                start_bearing_rad=batch["start_bearing_rad"],
                output_hw=grid_dataset.output_hw,
                pixel_size_m=float(grid_dataset.pixel_size_m),
            )
            model_batch: dict[str, torch.Tensor | None] = {
                "env": flat_env.to(dtype=torch.float32),
                "valid": flat_valid.to(dtype=torch.bool),
                "move": feature_adapter._movement_tensor(
                    step_length_m=step_length_m,
                    turn_angle_rad=turn_angle_rad,
                ),
                "time": None,
                "individual_idx": None,
            }
            if bool(getattr(feature_adapter, "include_time", False)):
                model_batch["time"] = feature_adapter._time_tensor(batch["timestamp_ns"])
            if bool(getattr(feature_adapter, "include_individual_id", False)):
                model_batch["individual_idx"] = feature_adapter._individual_idx_tensor(
                    np.asarray(batch["individual_id"])
                )
            logits = model(_move_batch_to_device(model_batch, device))
            scores_hw = log_q_hw + logits.detach().cpu().view(n_steps, height, width)
        point_id_chunks.append(point_id_start)
        scores_chunks.append(scores_hw)

    return {
        "prediction_kind": "grid",
        "point_id_start": torch.cat(point_id_chunks, dim=0),
        "scores": torch.cat(scores_chunks, dim=0),
        "pixel_size_m": float(grid_dataset.pixel_size_m),
    }
