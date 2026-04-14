from __future__ import annotations

import argparse
import hashlib
import json
import os
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


@dataclass(frozen=True)
class ArrayRequest:
    resources: SlurmResources
    array_concurrency: int
    log_path: Path
    worker_args_list: tuple[tuple[str, ...], ...]
    manifest_dir: Path
    manifest_stem: str
    name: str | None = None


@dataclass
class _ActiveArray:
    request_name: str
    process: subprocess.Popen[str]
    task_count: int
    job_id: str = ""
    last_status_counts: tuple[int, int, int] | None = None


_ACTIVE_STATE_PREFIXES = (
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
)
_ARRAY_TASK_STAGGER_SECONDS = 0.1


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


def _slurm_resource_flags(resources: SlurmResources) -> list[str]:
    command: list[str] = []
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
    return command


def _worker_command(worker_args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "movebench.cli.main", *worker_args]


def _array_manifest_payload(request: ArrayRequest) -> dict[str, object]:
    return {
        "tasks": [
            {"worker_args": list(worker_args)}
            for worker_args in request.worker_args_list
        ]
    }


def _array_manifest_path(request: ArrayRequest) -> Path:
    payload = json.dumps(
        _array_manifest_payload(request),
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return request.manifest_dir.resolve() / f"{request.manifest_stem}__{digest}.json"


def _write_array_manifest(*, request: ArrayRequest, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(_array_manifest_payload(request), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_sbatch_command(*, request: ArrayRequest, manifest_path: Path) -> list[str]:
    task_count = len(request.worker_args_list)
    if task_count <= 0:
        raise ValueError("Array requests must contain at least one task")
    if int(request.array_concurrency) <= 0:
        raise ValueError("Array requests must set array_concurrency > 0")

    wrapped_command = shlex.join(
        [
            sys.executable,
            "-m",
            "movebench.slurm.launcher",
            "--array-manifest",
            str(Path(manifest_path).resolve()),
        ]
    )
    return [
        "sbatch",
        "--wait",
        "--parsable",
        "--requeue",
        "--export=ALL",
        f"--array=0-{task_count - 1}%{int(request.array_concurrency)}",
        *_slurm_resource_flags(request.resources),
        "--output",
        str(request.log_path),
        "--error",
        str(request.log_path),
        "--wrap",
        wrapped_command,
    ]


def _request_name(request: ArrayRequest, index: int) -> str:
    if request.name:
        return str(request.name)
    return f"array={index}"


def _slurm_array_status_counts(job_id: str) -> tuple[int, int, int] | None:
    job_id = str(job_id).strip()
    if not job_id:
        return None
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "-n",
                "-P",
                "-X",
                "--format=State",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if int(result.returncode) != 0:
        return None

    running = 0
    finished = 0
    failed = 0
    saw_state = False
    for raw_line in str(result.stdout).splitlines():
        state = raw_line.split("|", 1)[0].strip()
        if not state:
            continue
        saw_state = True
        if state.startswith("COMPLETED"):
            finished += 1
        elif state.startswith(_ACTIVE_STATE_PREFIXES):
            running += 1
        else:
            failed += 1
    if not saw_state:
        return None
    return running, finished, failed


def run_sbatch_queue(
    *,
    label: str,
    requests: Sequence[ArrayRequest],
    polling_interval: float,
    dry_run: bool,
    max_concurrent_jobs: int | None = None,
) -> int:
    request_list = list(requests)
    if dry_run:
        summary = f"[{label}] Dry-run: would launch {len(request_list)} array jobs"
        if max_concurrent_jobs is not None:
            summary += f" (max_concurrent={int(max_concurrent_jobs)})"
        print(summary + ".")
        for index, request in enumerate(request_list):
            request_name = _request_name(request, index)
            print(
                f"[{label}] ARRAY {request_name} "
                f"tasks={len(request.worker_args_list)} "
                f"concurrency={int(request.array_concurrency)}"
            )
            manifest_path = _array_manifest_path(request)
            print(
                f"[{label}] "
                f"{shlex.join(build_sbatch_command(request=request, manifest_path=manifest_path))}"
            )
            for task_index, worker_args in enumerate(request.worker_args_list):
                print(
                    f"[{label}] task[{task_index}] "
                    f"{shlex.join(_worker_command(worker_args))}"
                )
        return 0

    if not request_list:
        return 0

    running: list[_ActiveArray] = []
    failures: list[tuple[str, int]] = []
    completed = 0
    try:
        for index, request in enumerate(request_list):
            request_name = _request_name(request, index)
            manifest_path = _array_manifest_path(request)
            _write_array_manifest(request=request, manifest_path=manifest_path)
            request.log_path.parent.mkdir(parents=True, exist_ok=True)
            command = build_sbatch_command(request=request, manifest_path=manifest_path)
            print(
                f"[{label}] ARRAY {request_name} "
                f"tasks={len(request.worker_args_list)} "
                f"concurrency={int(request.array_concurrency)}"
            )
            print(f"[{label}] {shlex.join(command)}")
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                text=True,
            )
            job_id = ""
            if process.stdout is not None:
                job_id = process.stdout.readline().strip()
            if job_id:
                print(f"[{label}] Submitted {request_name} job_id={job_id}")
            running.append(
                _ActiveArray(
                    request_name=request_name,
                    process=process,
                    task_count=len(request.worker_args_list),
                    job_id=job_id,
                )
            )

        while running:
            time.sleep(float(polling_interval))
            next_running: list[_ActiveArray] = []
            progress_changed = False
            for active in running:
                status_counts = _slurm_array_status_counts(active.job_id)
                if (
                    status_counts is not None
                    and status_counts != active.last_status_counts
                ):
                    running_count, finished_count, failed_count = status_counts
                    reported_count = running_count + finished_count + failed_count
                    print(
                        f"[{label}] Status {active.request_name}: "
                        f"tasks_reported={reported_count}/{active.task_count} "
                        f"tasks_active={running_count} "
                        f"tasks_finished={finished_count} "
                        f"tasks_failed={failed_count}"
                    )
                    active.last_status_counts = status_counts

                return_code = active.process.poll()
                if return_code is None:
                    next_running.append(active)
                    continue
                completed += 1
                progress_changed = True
                if int(return_code) != 0:
                    failures.append((active.request_name, int(return_code)))
            if progress_changed:
                print(
                    f"[{label}] Progress: arrays_completed={completed} "
                    f"arrays_running={len(next_running)} "
                    f"arrays_failed={len(failures)}"
                )
            running = next_running
    except KeyboardInterrupt:
        print(f"[{label}] Interrupted; {len(running)} arrays may still be running.")
        return 130

    return 0 if not failures else 2


def _load_array_manifest(manifest_path: Path) -> list[list[str]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = raw.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError(f"{manifest_path} must contain a tasks list")
    worker_args_list: list[list[str]] = []
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("worker_args"), list):
            raise ValueError(f"{manifest_path} has an invalid task entry")
        worker_args_list.append([str(arg) for arg in task["worker_args"]])
    return worker_args_list


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="python -m movebench.slurm.launcher",
        description="Internal runtime for MoveBench Slurm job arrays.",
    )
    parser.add_argument("--array-manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    time.sleep(_ARRAY_TASK_STAGGER_SECONDS * float(task_id))
    worker_args_list = _load_array_manifest(Path(args.array_manifest))
    try:
        worker_args = worker_args_list[task_id]
    except IndexError as exc:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={task_id} is out of range for {args.array_manifest}"
        ) from exc

    from movebench.cli.main import main as cli_main

    return int(cli_main(["movebench", *worker_args]))


if __name__ == "__main__":
    raise SystemExit(main())
