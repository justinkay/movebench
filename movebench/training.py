from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from tqdm import tqdm


class Runtime(ABC):
    """Training operations for the shared loop; lower metric values are better."""

    model: torch.nn.Module
    early_stop_split_key: str
    early_stop_metric: str = "nll"

    @abstractmethod
    def train_step(self, step: int) -> None:
        """Perform one training step, including method-specific scheduling."""
        ...

    @abstractmethod
    def evaluate(self, split_key: str) -> dict[str, float]:
        """Return split metrics without optimizer or scheduler updates.

        Early-stop split metrics must include early_stop_metric.
        """
        ...


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int
    eval_every_steps: int
    patience_evals: int


class RestorationReference(TypedDict):
    checkpoint_path: Path
    training_run_key: str | None
    training_signature: str | None


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def read_checkpoint(restore_from: RestorationReference) -> dict[str, object]:
    """Read a completed model on CPU, without training machinery or history."""
    checkpoint = torch.load(
        restore_from["checkpoint_path"], map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a completed checkpoint dictionary")
    for key in ("model_state", "train_metrics", "eval_metrics", "schedule", "model_metadata"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Checkpoint requires {key!r} as a dictionary")
    for key in ("best_step", "completed_steps"):
        if type(checkpoint.get(key)) is not int:
            raise ValueError(f"Checkpoint requires {key!r} as an integer")
    if not 0 <= checkpoint["best_step"] <= checkpoint["completed_steps"]:
        raise ValueError("Checkpoint has inconsistent best/completed steps")
    signature = restore_from["training_signature"]
    if signature is not None:
        metadata = checkpoint["model_metadata"]
        if metadata.get("shared_training_signature") != signature:
            raise ValueError("Shared checkpoint training signature mismatch")
    return checkpoint


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    fit: dict[str, object],
    schedule: dict[str, object],
    model_metadata: dict[str, object],
) -> dict[str, object]:
    checkpoint = {
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "best_step": int(fit["best_step"]),
        "completed_steps": int(fit["completed_steps"]),
        "train_metrics": dict(fit["train_metrics"]),
        "eval_metrics": dict(fit["eval_metrics"]),
        "schedule": dict(schedule),
        "model_metadata": dict(model_metadata),
    }
    torch.save(checkpoint, path)
    return checkpoint


def fit_or_restore(
    runtime: Runtime,
    config: TrainerConfig,
    *,
    work_dir: Path,
    restore_from: RestorationReference | None,
    schedule: dict[str, object] | None = None,
    model_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Honor the runner's fitting choice; a failed restoration never retrains."""
    if restore_from is not None:
        checkpoint = read_checkpoint(restore_from)
        runtime.model.load_state_dict(checkpoint["model_state"])
    else:
        fit = simple_training_loop(runtime, config)
        (Path(work_dir) / "history.json").write_text(
            json.dumps(fit["history"], indent=2, sort_keys=True), encoding="utf-8"
        )
        checkpoint = save_checkpoint(
            Path(work_dir) / "checkpoint_best.pt",
            model=runtime.model,
            fit=fit,
            schedule=schedule or {},
            model_metadata=model_metadata or {},
        )
    return {key: value for key, value in checkpoint.items() if key != "model_state"}


def simple_training_loop(
    runtime: Runtime,
    config: TrainerConfig,
) -> dict[str, object]:
    if int(config.max_steps) <= 0:
        raise ValueError(f"max_steps must be > 0, got {config.max_steps!r}")
    if int(config.eval_every_steps) <= 0:
        raise ValueError(
            f"eval_every_steps must be > 0, got {config.eval_every_steps!r}"
        )
    if int(config.patience_evals) <= 0:
        raise ValueError(
            f"patience_evals must be > 0, got {config.patience_evals!r}"
        )

    early_stop_split_key = runtime.early_stop_split_key
    early_stop_metric = runtime.early_stop_metric

    best_state: dict[str, object] | None = None
    best_metric = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []
    completed_steps = 0
    last_eval_step = 0
    stopped_early = False

    for step in tqdm(range(1, int(config.max_steps) + 1), desc=f"Training..."):
        completed_steps = int(step)
        runtime.train_step(step)

        should_eval = step % int(config.eval_every_steps) == 0
        if not should_eval and step != int(config.max_steps):
            continue

        train_metrics = {
            str(key): float(value)
            for key, value in dict(runtime.evaluate("train")).items()
        }
        eval_metrics = {
            str(key): float(value)
            for key, value in dict(runtime.evaluate(early_stop_split_key)).items()
        }
        last_eval_step = int(step)
        print("Train:", train_metrics)
        print("Eval:", eval_metrics)

        history_entry: dict[str, float] = {"step": float(step)}
        history_entry.update(
            {f"train/{key}": value for key, value in train_metrics.items()}
        )
        history_entry.update(
            {
                f"{early_stop_split_key}/{key}": value
                for key, value in eval_metrics.items()
            }
        )
        history.append(history_entry)

        metric_value = float(eval_metrics[early_stop_metric])
        if metric_value < best_metric:
            best_metric = metric_value
            best_step = int(step)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in runtime.model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(config.patience_evals):
                stopped_early = True
                break

    print("Training done; loading best state")
    if best_state is None:
        best_step = int(last_eval_step or completed_steps)
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in runtime.model.state_dict().items()
        }

    print("Computing final train and eval metrics")
    runtime.model.load_state_dict(best_state)
    final_train_metrics = {
        str(key): float(value)
        for key, value in dict(runtime.evaluate("train")).items()
    }
    final_eval_metrics = {
        str(key): float(value)
        for key, value in dict(runtime.evaluate(early_stop_split_key)).items()
    }
    print("Train final:", final_train_metrics)
    print("Eval final:", final_eval_metrics)
    return {
        "best_step": int(best_step),
        "completed_steps": int(completed_steps),
        "stopped_early": bool(stopped_early),
        "early_stop_split_key": early_stop_split_key,
        "train_metrics": final_train_metrics,
        "eval_metrics": final_eval_metrics,
        "history": history,
    }
