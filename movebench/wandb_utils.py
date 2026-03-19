"""Optional Weights & Biases (wandb) helpers.

This module is intentionally tiny and dependency-light:
- it lazily imports wandb only when enabled,
- it provides small adapters for logging metrics from training loops,
- and it is safe to use even when wandb is not installed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol


class LogCallback(Protocol):
    def __call__(self, metrics: dict[str, float], *, step: int) -> None: ...


_WARNED_IMPORT = False


def _as_str(val: Any) -> str:
    return str(val)


def _dedupe(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        s = str(s)
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _try_import_wandb() -> Any | None:
    global _WARNED_IMPORT
    try:
        import wandb  # type: ignore[import-not-found]

        return wandb
    except Exception as e:
        if not _WARNED_IMPORT:
            _WARNED_IMPORT = True
            print(f"[wandb] Disabled (import failed): {e}")
        return None


def _compact_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in d.items() if v is not None}


def make_wandb_log_callback(run: Any | None) -> LogCallback | None:
    if run is None:
        return None

    warned = False

    def _log(metrics: dict[str, float], *, step: int) -> None:
        nonlocal warned
        try:
            run.log(dict(metrics), step=int(step))
        except Exception as e:
            if not warned:
                warned = True
                print(f"[wandb] log failed (continuing without logging): {e}")

    return _log


@contextmanager
def maybe_wandb_run(
    wandb_cfg: dict[str, Any] | None,
    *,
    output_dir: str | Path | None,
    group: str | None,
    name: str | None,
    job_type: str | None,
    tags: Sequence[str] | None,
    config: dict[str, Any] | None,
    default_project: str = "movebench",
) -> Iterator[Any | None]:
    """Create a wandb run if enabled; otherwise yield None.

    Args:
        wandb_cfg: mapping with keys like enabled/project/entity/mode/dir/tags.
        output_dir: benchmark output directory used to default wandb dir.
        group/name/job_type/tags/config: forwarded to wandb.init when enabled.
    """
    cfg = wandb_cfg or {}
    if not isinstance(cfg, dict):
        raise ValueError("wandb config must be a mapping/dict when provided")

    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        yield None
        return

    # Resolve defaults with environment overrides.
    project = (
        _as_str(cfg["project"])
        if "project" in cfg
        else os.environ.get("WANDB_PROJECT", default_project)
    )
    entity = None
    if "entity" in cfg and cfg["entity"] is not None:
        entity = _as_str(cfg["entity"])
    elif os.environ.get("WANDB_ENTITY"):
        entity = os.environ.get("WANDB_ENTITY")

    mode = None
    if "mode" in cfg and cfg["mode"] is not None:
        mode = _as_str(cfg["mode"])
    elif os.environ.get("WANDB_MODE"):
        mode = os.environ.get("WANDB_MODE")
    else:
        mode = "online"

    if str(mode).strip().lower() == "disabled":
        yield None
        return

    run_dir = None
    if "dir" in cfg and cfg["dir"] is not None:
        run_dir = Path(_as_str(cfg["dir"]))
    elif os.environ.get("WANDB_DIR"):
        run_dir = Path(os.environ.get("WANDB_DIR", ""))
    elif output_dir is not None:
        run_dir = Path(output_dir) / "wandb"

    extra_tags = cfg.get("tags", None)
    tags_out: list[str] | None = None
    if tags or extra_tags:
        combined: list[str] = []
        if tags:
            combined.extend([_as_str(t) for t in tags])
        if extra_tags:
            if not isinstance(extra_tags, (list, tuple)):
                raise ValueError("wandb.tags must be a list of strings when provided")
            combined.extend([_as_str(t) for t in extra_tags])
        tags_out = _dedupe(combined)

    wandb = _try_import_wandb()
    if wandb is None:
        yield None
        return

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    run = None
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            mode=mode,
            dir=str(run_dir) if run_dir is not None else None,
            group=group,
            name=name,
            job_type=job_type,
            tags=tags_out,
            config=config,
            reinit=True,
        )
        yield run
    finally:
        try:
            if run is not None and hasattr(run, "finish"):
                run.finish()
        except Exception:
            pass


@contextmanager
def maybe_wandb_benchmark_log_callback(
    wandb_cfg: dict[str, Any] | None,
    *,
    output_dir: str | Path | None,
    study_id: str,
    benchmark_name: str,
    method_name: str,
    kind: str,
    layers_config: str | Path | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> Iterator[LogCallback | None]:
    """Create a W&B run (if enabled) and yield a metric log callback.

    This is the "batteries-included" helper used by scripts/benchmark runners to
    avoid duplicating wandb.init boilerplate and to keep the run metadata small.
    """
    config = _compact_dict(
        {
            "study_id": str(study_id),
            "method_name": str(method_name),
            "kind": str(kind),
            "benchmark": str(benchmark_name),
            "layers_config": str(layers_config) if layers_config is not None else None,
            "device": str(device) if device is not None else None,
            "seed": int(seed) if seed is not None else None,
        }
    )
    with maybe_wandb_run(
        wandb_cfg,
        output_dir=output_dir,
        group=str(study_id),
        name=f"{benchmark_name}/{method_name}",
        job_type=str(kind),
        tags=None,
        config=config,
    ) as run:
        yield make_wandb_log_callback(run)
