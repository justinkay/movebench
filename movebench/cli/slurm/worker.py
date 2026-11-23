from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import movebench.bench.run as bench_run
from movebench.cli.main import main as cli_main

_ARRAY_TASK_STAGGER_SECONDS = 0.1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m movebench.cli.slurm.worker",
        description="Internal runtime for MoveBench Slurm job arrays.",
    )
    parser.add_argument("--array-manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    time.sleep(_ARRAY_TASK_STAGGER_SECONDS * float(task_id))
    task = json.loads(args.array_manifest.read_text(encoding="utf-8"))["tasks"][
        task_id
    ]

    if "run_payload" in task:
        summary = bench_run.execute_run_payload(task["run_payload"], resume=task["resume"])
        return 0 if str(summary.get("status", "completed")) == "completed" else 2
    return int(cli_main(task["worker_args"]))


if __name__ == "__main__":
    raise SystemExit(main())
