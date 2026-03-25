from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int
    eval_every_steps: int
    patience_evals: int
    seed: int


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _log_metrics(tracker: object, metrics: dict[str, float], *, step: int) -> None:
    log_fn = getattr(tracker, "log_metrics", None)
    if callable(log_fn):
        log_fn(dict(metrics), step=int(step))


def _write_history(tracker: object, history: list[dict[str, float]]) -> None:
    write_fn = getattr(tracker, "write_history", None)
    if callable(write_fn):
        write_fn([dict(entry) for entry in history])


def run_training_loop(
    runtime: object,
    config: TrainerConfig,
    tracker: object,
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

    _seed_everything(int(config.seed))

    early_stop_split_key = str(
        getattr(runtime, "early_stop_split_key", "test-temporal")
    )
    early_stop_metric = str(getattr(runtime, "early_stop_metric", "nll"))

    best_state: dict[str, object] | None = None
    best_metric = float("inf")
    best_step = 0
    patience_counter = 0
    history: list[dict[str, float]] = []
    completed_steps = 0
    last_eval_step = 0
    stopped_early = False

    try:
        for step in range(1, int(config.max_steps) + 1):
            completed_steps = int(step)
            train_step_metrics = {
                str(key): float(value)
                for key, value in dict(runtime.train_step(step)).items()
            }
            _log_metrics(tracker, train_step_metrics, step=step)

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
            _log_metrics(
                tracker,
                {
                    **{f"train/{key}": value for key, value in train_metrics.items()},
                    **{
                        f"{early_stop_split_key}/{key}": value
                        for key, value in eval_metrics.items()
                    },
                },
                step=step,
            )

            metric_value = float(eval_metrics[early_stop_metric])
            if metric_value < best_metric:
                best_metric = metric_value
                best_step = int(step)
                best_state = copy.deepcopy(dict(runtime.state_dict()))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(config.patience_evals):
                    stopped_early = True
                    break

        if best_state is None:
            best_step = int(last_eval_step or completed_steps)
            best_state = copy.deepcopy(dict(runtime.state_dict()))

        runtime.load_state_dict(best_state)
        final_train_metrics = {
            str(key): float(value)
            for key, value in dict(runtime.evaluate("train")).items()
        }
        final_eval_metrics = {
            str(key): float(value)
            for key, value in dict(runtime.evaluate(early_stop_split_key)).items()
        }
        _write_history(tracker, history)
        return {
            "best_step": int(best_step),
            "completed_steps": int(completed_steps),
            "stopped_early": bool(stopped_early),
            "early_stop_split_key": early_stop_split_key,
            "train_metrics": final_train_metrics,
            "eval_metrics": final_eval_metrics,
            "history": history,
        }
    finally:
        close_fn = getattr(runtime, "close", None)
        if callable(close_fn):
            close_fn()
