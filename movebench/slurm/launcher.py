from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class SlurmResources:
    partition: str | None = None
    exclude: str | None = None
    qos: str | None = None
    account: str | None = None
    time: str | None = None
    mem: str | None = None
    cpus_per_task: int = 1
    gpus_per_node: int = 0


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


def _slurm_resource_flags(resources: SlurmResources) -> list[str]:
    command: list[str] = []
    if resources.partition:
        command.extend(["--partition", str(resources.partition)])
    if resources.exclude:
        command.extend(["--exclude", str(resources.exclude)])
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
            "movebench.cli.slurm.worker",
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
                "--format=JobID,State",
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
        fields = raw_line.split("|")
        if len(fields) < 2:
            continue
        record_id, state = (field.strip() for field in fields[:2])
        if "[" in record_id or not state:
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
) -> int:
    request_list = list(requests)
    if dry_run:
        print(f"[{label}] Dry-run: would launch {len(request_list)} array jobs.")
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
