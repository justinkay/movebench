from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from movebench.features.points.config import PointStoreConfig


def _step_seed(random_seed: int, point_id_start: int) -> int:
    s = f"{int(random_seed)}|{int(point_id_start)}".encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:8], 16)


def fit_sampler(
    train_movement_df: pd.DataFrame,
    *,
    cfg: PointStoreConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit/train-only movement sampler for candidate generation.

    Expects `train_movement_df` to be the output of `compute_movement_features(...)`.
    Returns (sampler_summary_json, sampler_state).
    """
    if not cfg.is_fully_specified():
        raise ValueError("fit_sampler requires a fully specified PointStoreConfig.")

    if "valid_step" not in train_movement_df.columns or "step_length_m" not in train_movement_df.columns:
        raise ValueError("train_movement_df must include columns: valid_step, step_length_m")
    if "turn_angle_rad" not in train_movement_df.columns:
        raise ValueError("train_movement_df must include column: turn_angle_rad")

    valid_lengths = train_movement_df.loc[train_movement_df["valid_step"], "step_length_m"].to_numpy(dtype=np.float64)
    valid_turns = train_movement_df["turn_angle_rad"].dropna().to_numpy(dtype=np.float64)

    if len(valid_lengths) == 0:
        raise ValueError("No valid step lengths in train split; cannot build candidate sampler")
    if len(valid_turns) == 0:
        raise ValueError("No valid turn angles in train split; cannot build candidate sampler")

    max_step_len = float(train_movement_df["max_step_len"].iloc[0]) if len(train_movement_df) else float("nan")

    sampler: dict[str, Any] = {
        "type": str(cfg.sampler_type),
        "fit_source_split": "train",
        "num_candidates": int(cfg.num_candidates),
        "random_seed": int(cfg.random_seed),
        "step_multiplier": float(cfg.step_multiplier),
        "step_q": float(cfg.step_q),
        "gap_q": float(cfg.gap_q),
        "n_valid_lengths_train": int(len(valid_lengths)),
        "n_valid_turns_train": int(len(valid_turns)),
        "max_step_len_m_train": float(max_step_len),
    }

    state: dict[str, Any] = {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(cfg.num_candidates),
        "random_seed": int(cfg.random_seed),
        "gamma_used": False,
        "gamma_params": None,
        "vonmises_used": False,
        "vonmises_params": None,
    }

    if str(cfg.sampler_type) == "parametric":
        gamma_used = False
        vonmises_used = False
        gamma_params = None
        vonmises_params = None

        try:
            from scipy import stats  # type: ignore

            try:
                gamma_params = stats.gamma.fit(valid_lengths, floc=0)
                gamma_used = True
            except Exception:
                gamma_used = False
                gamma_params = None

            try:
                vonmises_params = stats.vonmises.fit(valid_turns)
                vonmises_used = True
            except Exception:
                vonmises_used = False
                vonmises_params = None

        except Exception:
            gamma_used = False
            vonmises_used = False
            gamma_params = None
            vonmises_params = None

        sampler["gamma"] = {
            "used": bool(gamma_used),
            "params": [float(x) for x in gamma_params] if gamma_used and gamma_params is not None else None,
        }
        sampler["vonmises"] = {
            "used": bool(vonmises_used),
            "params": [float(x) for x in vonmises_params] if vonmises_used and vonmises_params is not None else None,
        }

        state["gamma_used"] = bool(gamma_used)
        state["gamma_params"] = gamma_params
        state["vonmises_used"] = bool(vonmises_used)
        state["vonmises_params"] = vonmises_params

    return sampler, state


def sample_pseudo_steps(state: dict[str, Any], point_id_start: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample pseudo step lengths and turn angles for one step start (deterministic by point_id_start)."""
    seed = _step_seed(int(state["random_seed"]), int(point_id_start))
    rng = np.random.default_rng(seed)

    valid_lengths = state["valid_lengths"]
    valid_turns = state["valid_turns"]
    n = int(state["num_candidates"])

    pseudo_lens: np.ndarray
    pseudo_turns: np.ndarray

    if state.get("gamma_used", False) and state.get("gamma_params") is not None:
        from scipy import stats  # type: ignore

        pseudo_lens = stats.gamma.rvs(*state["gamma_params"], size=n, random_state=rng).astype(np.float64, copy=False)
    else:
        pseudo_lens = rng.choice(valid_lengths, size=n, replace=True).astype(np.float64, copy=False)

    if state.get("vonmises_used", False) and state.get("vonmises_params") is not None:
        from scipy import stats  # type: ignore

        pseudo_turns = stats.vonmises.rvs(*state["vonmises_params"], size=n, random_state=rng).astype(np.float64, copy=False)
    else:
        pseudo_turns = rng.choice(valid_turns, size=n, replace=True).astype(np.float64, copy=False)

    return pseudo_lens, pseudo_turns

