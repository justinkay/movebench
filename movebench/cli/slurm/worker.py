from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from movebench.cli.main import main as cli_main

_ARRAY_TASK_STAGGER_SECONDS = 0.1


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
    parser = argparse.ArgumentParser(
        prog="python -m movebench.cli.slurm.worker",
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

    return int(cli_main(worker_args))


if __name__ == "__main__":
    raise SystemExit(main())
