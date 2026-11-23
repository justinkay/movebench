from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
from contextlib import contextmanager
from pathlib import Path

import movebench.paths as paths

_ACTIVE_RUN_FILENAME = "active_run.json"
_RESULTS_ROOT = Path("results") / "bench" / "runs"


class RunInterrupted(BaseException):
    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


def interrupt_reason(exc: BaseException) -> str:
    return exc.reason if isinstance(exc, RunInterrupted) else exc.__class__.__name__


def _install_interrupt_handlers() -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handle_signal(signum, frame) -> None:
        del frame
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = f"SIG{int(signum)}"
        raise RunInterrupted(name)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except (ValueError, OSError, RuntimeError):
            continue
    return previous


@contextmanager
def active_run(run_dir: Path):
    write_active_run(run_dir)
    previous = _install_interrupt_handlers()
    try:
        yield
    finally:
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, RuntimeError):
                continue
        remove_active_run(run_dir)


def load_training_summary(run_dir: Path) -> dict[str, object]:
    try:
        return _load_json(run_dir / "training_summary.json")
    except (OSError, ValueError):
        return {}


def write_training_summary(
    run_dir: Path,
    summary: dict[str, object],
    *,
    status: str,
    reason: str | None = None,
) -> dict[str, object]:
    summary = {**summary, "status": status}
    summary.pop("reason", None)
    if reason is not None:
        summary["reason"] = reason
    _write_json(run_dir / "training_summary.json", summary)
    return summary


def completion_error(
    run_dir: Path, run_payload: dict[str, object], summary: dict[str, object]
) -> str | None:
    if summary.get("fit_completed") is not True:
        return "missing successful fitting information"
    if not run_payload.get("evaluates", True):
        required = [run_dir / "checkpoint_best.pt"]
    else:
        required = [
            path
            for split in run_payload["evaluation_split_keys"]
            for path in (
                run_dir / "predictions" / f"{split}.pt",
                run_dir / "final_metrics" / split / "energy_by_step.csv.gz",
                run_dir / "final_metrics" / split / "summary.json",
            )
        ]
    for path in required:
        if not path.is_file():
            return f"missing {path.relative_to(run_dir)}"
    return None


def metric_repair_eligible(
    run_dir: Path, run_payload: dict[str, object], summary: dict[str, object]
) -> bool:
    splits = run_payload.get("evaluation_split_keys", [])
    return bool(
        run_payload.get("evaluates", True)
        and splits
        and summary.get("fit_completed") is True
        and not active_run_is_live(run_dir)
        and all((run_dir / "predictions" / f"{split}.pt").is_file() for split in splits)
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return dict(payload)


def run_dir(run_key: str) -> Path:
    return paths.repo_root() / _RESULTS_ROOT / str(run_key)


def _active_run_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / _ACTIVE_RUN_FILENAME


def _load_active_run_state(run_dir: str | Path) -> dict[str, object] | None:
    path = _active_run_path(run_dir)
    if not path.exists():
        return None
    try:
        return _load_json(path)
    except Exception:
        return None


def write_active_run(run_dir: Path) -> None:
    _write_json(
        _active_run_path(run_dir),
        {
            "pid": int(os.getpid()),
            "hostname": socket.gethostname(),
            "slurm_job_id": str(os.environ.get("SLURM_JOB_ID", "")).strip(),
        },
    )


def remove_active_run(run_dir: str | Path) -> None:
    path = _active_run_path(run_dir)
    if path.exists():
        path.unlink()


def _pid_is_live(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _slurm_job_is_live(job_id: str) -> bool | None:
    job_id = str(job_id).strip()
    if not job_id:
        return None
    if shutil.which("squeue") is None:
        return None
    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%i"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if int(result.returncode) not in {0, 1}:
        return None
    return bool(str(result.stdout).strip())


def active_run_is_live(run_dir: str | Path) -> bool:
    state = _load_active_run_state(run_dir)
    if not state:
        return False
    job_live = _slurm_job_is_live(str(state.get("slurm_job_id", "")))
    if job_live is not None:
        return bool(job_live)
    hostname = str(state.get("hostname", "")).strip()
    if hostname and hostname != socket.gethostname():
        return False
    try:
        pid = int(state.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return False
    return _pid_is_live(pid)


def _metric_split_key(split_key: str) -> str:
    return str(split_key).replace("-", "_")


def _flatten_final_metrics(run_dir: Path, split_keys: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_key in split_keys:
        path = run_dir / "final_metrics" / str(split_key) / "summary.json"
        if not path.exists():
            continue
        try:
            summary = _load_json(path)
        except Exception:
            continue
        metric_split = _metric_split_key(str(split_key))
        energy_score = summary.get("energy_score")
        if energy_score is not None:
            metrics[f"energy_score.{metric_split}"] = float(energy_score)
            metrics[f"energy_score.accuracy.{metric_split}"] = float(summary["accuracy"])
            metrics[f"energy_score.calibration.{metric_split}"] = float(
                summary["calibration"]
            )
        if "n_steps_scored" in summary:
            metrics[f"n_steps_scored.{metric_split}"] = float(summary["n_steps_scored"])
        if "n_steps_excluded_out_of_support" in summary:
            metrics[f"n_steps_excluded_out_of_support.{metric_split}"] = float(
                summary["n_steps_excluded_out_of_support"]
            )
    return metrics


def _incomplete_reason(run_dir: Path) -> str | None:
    if _load_active_run_state(run_dir) is not None:
        return "stale interrupted run"
    if (run_dir / "resolved_run.json").exists():
        return "incomplete run missing training_summary"
    return None


def _load_run_record(run_key: str) -> dict[str, object] | None:
    current_run_dir = run_dir(str(run_key))
    resolved_path = current_run_dir / "resolved_run.json"
    if not resolved_path.exists():
        return None
    try:
        run_payload = _load_json(resolved_path)
    except Exception:
        return None

    failure_reason = None
    if active_run_is_live(current_run_dir):
        status = "running"
    elif (current_run_dir / "training_summary.json").exists():
        try:
            training_summary = _load_json(current_run_dir / "training_summary.json")
        except Exception:
            training_summary = {}
        status = str(training_summary.get("status", "")).strip()
        if status == "completed":
            failure_reason = completion_error(current_run_dir, run_payload, training_summary)
            if failure_reason:
                status = "failed"
        elif status in {"", "running"}:
            failure_reason = (
                "stale interrupted run" if status == "running"
                else "invalid training_summary.json"
            )
            status = "failed"
        elif status in {"failed", "interrupted"}:
            failure_reason = str(training_summary.get("reason", "")).strip()
            status = "failed"
    else:
        failure_reason = _incomplete_reason(current_run_dir)
        if failure_reason is None:
            return None
        status = "failed"

    return {
        "run_id": str(run_key),
        "status": status,
        "tags": {"failure_reason": failure_reason} if failure_reason else {},
        "metrics": _flatten_final_metrics(
            current_run_dir,
            [
                str(split_key)
                for split_key in list(run_payload.get("evaluation_split_keys", []) or [])
            ],
        ),
    }


def search_latest_run_records_by_key(
    run_keys: list[str],
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for run_key in [str(value) for value in run_keys]:
        record = _load_run_record(run_key)
        if record is None:
            continue
        records[run_key] = record
    return records

