from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml


@dataclass(frozen=True)
class SlurmResources:
    partition: str | None = None
    qos: str | None = None
    account: str | None = None
    time: str | None = None
    mem: str | None = None
    cpus_per_task: int = 1
    gpus_per_node: int = 0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SlurmResources":
        return cls(
            partition=args.partition,
            qos=args.qos,
            account=args.account,
            time=args.time,
            mem=args.mem,
            cpus_per_task=(
                int(args.cpus_per_task)
                if getattr(args, "cpus_per_task", None) is not None
                else 1
            ),
            gpus_per_node=(
                int(args.gpus_per_node)
                if getattr(args, "gpus_per_node", None) is not None
                else 0
            ),
        )


def register_common_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slurm-config",
        type=Path,
        default=None,
        help="Optional YAML file with shared Slurm argument defaults.",
    )
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--time", type=str, default=None)
    parser.add_argument("--mem", type=str, default=None)
    parser.add_argument("--cpus-per-task", type=int, default=None)
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument("--max-concurrent-jobs", type=int, default=None)
    parser.add_argument("--polling-interval", type=float, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")


_SLURM_CONFIG_FIELDS = {
    "partition": str,
    "qos": str,
    "account": str,
    "time": str,
    "mem": str,
    "cpus_per_task": int,
    "gpus_per_node": int,
    "max_concurrent_jobs": int,
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


def build_srun_command(
    *,
    resources: SlurmResources,
    worker_args: Sequence[str],
    log_path: str | Path | None = None,
) -> list[str]:
    command = ["srun", "--export=ALL"]
    if resources.partition:
        command.extend(["--partition", str(resources.partition)])
    if resources.qos:
        command.extend(["--qos", str(resources.qos)])
    if resources.account:
        command.extend(["--account", str(resources.account)])
    if resources.time:
        command.extend(["--time", str(resources.time)])
    if resources.mem:
        command.extend(["--mem", str(resources.mem)])
    if resources.cpus_per_task:
        command.extend(["--cpus-per-task", str(resources.cpus_per_task)])
    if resources.gpus_per_node:
        command.extend(["--gpus-per-node", str(resources.gpus_per_node)])
    if log_path is not None:
        log_path = Path(log_path)
        command.extend(["--output", str(log_path), "--error", str(log_path)])
    command.extend([sys.executable, "-m", "movebench.cli.main", *worker_args])
    return command


def run_srun_queue(
    *,
    label: str,
    commands: Sequence[Sequence[str]],
    max_concurrent_jobs: int,
    polling_interval: float,
    dry_run: bool,
) -> int:
    if int(max_concurrent_jobs) <= 0:
        raise ValueError("--max-concurrent-jobs must be > 0")

    command_list = [list(command) for command in commands]
    if dry_run:
        print(
            f"[{label}] Dry-run: would launch {len(command_list)} jobs "
            f"(max_concurrent={max_concurrent_jobs})."
        )
        for command in command_list:
            print(f"[{label}] {shlex.join(command)}")
        return 0

    pending = list(command_list)
    running: list[tuple[list[str], subprocess.Popen[bytes]]] = []
    failures: list[tuple[list[str], int]] = []
    completed = 0
    total = len(pending)
    try:
        while pending or running:
            while pending and len(running) < int(max_concurrent_jobs):
                command = pending.pop(0)
                print(f"[{label}] Launching {completed + len(running) + 1}/{total}.")
                print(f"[{label}] {shlex.join(command)}")
                running.append((command, subprocess.Popen(command)))

            if not running:
                continue

            time.sleep(float(polling_interval))
            next_running: list[tuple[list[str], subprocess.Popen[bytes]]] = []
            progress_changed = False
            for command, process in running:
                return_code = process.poll()
                if return_code is None:
                    next_running.append((command, process))
                    continue
                completed += 1
                progress_changed = True
                if int(return_code) != 0:
                    failures.append((command, int(return_code)))
            if progress_changed:
                print(
                    f"[{label}] Progress: completed={completed} "
                    f"running={len(next_running)} pending={len(pending)} "
                    f"failed={len(failures)}"
                )
            running = next_running
    except KeyboardInterrupt:
        print(f"[{label}] Interrupted; {len(running)} jobs may still be running.")
        return 130

    return 0 if not failures else 2
