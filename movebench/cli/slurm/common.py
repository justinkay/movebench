from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from movebench.slurm.launcher import SlurmResources


def resources_from_args(args: argparse.Namespace) -> SlurmResources:
    return SlurmResources(
        partition=args.partition,
        exclude=args.exclude,
        qos=args.qos,
        account=args.account,
        time=args.time,
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        gpus_per_node=args.gpus_per_node,
    )


def register_common_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slurm-config",
        type=Path,
        default=None,
        help="Optional YAML file with shared Slurm argument defaults.",
    )
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--exclude", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--time", type=str, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--polling-interval", type=float, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")


_SLURM_CONFIG_FIELDS = {
    "partition": str,
    "exclude": str,
    "qos": str,
    "account": str,
    "time": str,
    "mem": str,
    "cpus_per_task": int,
    "gpus_per_node": int,
    "max_concurrent_jobs": int,
    "max_cpu_jobs": int,
    "max_gpu_jobs": int,
    "polling_interval": float,
    "log_dir": Path,
}


def _normalize_slurm_config_section(
    raw: Any,
    *,
    config_path: Path,
    section_name: str,
) -> dict[str, object]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} {section_name} must be a mapping")

    unknown = sorted(str(key) for key in raw.keys() if key not in _SLURM_CONFIG_FIELDS)
    if unknown:
        raise ValueError(
            f"{config_path} {section_name} has unknown key(s): {unknown}"
        )

    out: dict[str, object] = {}
    for key, converter in _SLURM_CONFIG_FIELDS.items():
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            continue
        if converter is Path:
            out[key] = Path(str(value))
        else:
            out[key] = converter(value)
    return out


def apply_slurm_config(
    args: argparse.Namespace,
    *,
    command_name: str,
    defaults: dict[str, object] | None = None,
) -> None:
    resolved = dict(defaults or {})
    config_path = getattr(args, "slurm_config", None)
    if config_path is not None:
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path} must be a mapping")

        defaults_raw = raw.get("defaults")
        commands_raw = raw.get("commands")
        unexpected = sorted(
            str(key)
            for key in raw.keys()
            if key not in {"defaults", "commands"}
        )
        if unexpected:
            raise ValueError(f"{config_path} has unknown top-level key(s): {unexpected}")

        resolved.update(
            _normalize_slurm_config_section(
                defaults_raw,
                config_path=config_path,
                section_name="defaults",
            )
        )

        if commands_raw is not None:
            if not isinstance(commands_raw, dict):
                raise ValueError(f"{config_path} commands must be a mapping")
            resolved.update(
                _normalize_slurm_config_section(
                    commands_raw.get(str(command_name)),
                    config_path=config_path,
                    section_name=f"commands.{command_name}",
                )
            )

    for key, value in resolved.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
