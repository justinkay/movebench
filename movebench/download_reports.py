from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    POINT_ID_COL,
    SET_COL,
    canonicalize_split_key,
    fit_interval_step_support,
    load_trajectories_csv,
)


def chip_request_hash(point_ids: list[int]) -> str:
    payload = {"point_ids": sorted({int(pid) for pid in point_ids})}
    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def download_reports_dir(*, study_dir: Path, rasters_hash: str) -> Path:
    return Path(study_dir) / "rasters" / str(rasters_hash) / "download_reports"


def latest_download_report_path(*, study_dir: Path, rasters_hash: str) -> Path:
    return download_reports_dir(study_dir=study_dir, rasters_hash=rasters_hash) / "latest.json"


def _timestamp_slug(ts: str) -> str:
    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def timestamped_download_report_path(
    *, study_dir: Path, rasters_hash: str, request_hash: str, completed_at: str
) -> Path:
    stamp = _timestamp_slug(completed_at)
    return (
        download_reports_dir(study_dir=study_dir, rasters_hash=rasters_hash)
        / f"{stamp}__{str(request_hash)}.json"
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=str(path.parent),
        prefix=f"{path.name}.",
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        Path(tmp.name).replace(path)
    os.chmod(path, 0o664)


def write_chip_download_report(
    *,
    study_dir: Path,
    rasters_hash: str,
    request_hash: str,
    requested_splits: list[str],
    requested_point_count: int,
    started_at: str,
    completed_at: str,
    chip_groups: dict[str, dict[str, Any]],
    status: str,
) -> Path:
    payload = {
        "schema_version": "1",
        "study_id": Path(study_dir).name,
        "study_dir": str(Path(study_dir)),
        "rasters_hash": str(rasters_hash),
        "request_hash": str(request_hash),
        "requested_splits": list(requested_splits),
        "requested_point_count": int(requested_point_count),
        "started_at": str(started_at),
        "completed_at": str(completed_at),
        "status": str(status),
        "chip_groups": chip_groups,
    }
    report_path = timestamped_download_report_path(
        study_dir=study_dir,
        rasters_hash=rasters_hash,
        request_hash=request_hash,
        completed_at=completed_at,
    )
    _write_json_atomic(report_path, payload)
    _write_json_atomic(
        latest_download_report_path(study_dir=study_dir, rasters_hash=rasters_hash),
        payload,
    )
    return report_path


def read_request_status(
    *, study_dir: Path, rasters_hash: str, request_hash: str
) -> dict[str, Any] | None:
    meta_path = Path(study_dir) / "rasters" / str(rasters_hash) / "chips_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    request_status = meta.get("request_status")
    if not isinstance(request_status, dict):
        return None
    status = request_status.get(str(request_hash))
    if not isinstance(status, dict):
        return None
    return dict(status)


def update_request_status(
    meta: dict[str, Any],
    *,
    request_hash: str,
    status: str,
    report_path: Path,
    completed_at: str,
) -> None:
    request_status = meta.get("request_status")
    if not isinstance(request_status, dict):
        request_status = {}
        meta["request_status"] = request_status
    request_status[str(request_hash)] = {
        "status": str(status),
        "report_path": str(Path(report_path)),
        "completed_at": str(completed_at),
    }


def resolve_chip_request_hash(
    *, study, plan: LayerPlan, splits: list[str]
) -> tuple[str | None, int]:
    from movebench.study import _rows_for_raster_downloads

    splits = [canonicalize_split_key(split) for split in splits]
    if not splits:
        return None, 0

    df = load_trajectories_csv(study.trajectories_path)
    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        return None, 0

    support = fit_interval_step_support(
        train_df,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        step_q=float(plan.step_spec.step_q),
        max_step_q=float(plan.step_spec.max_step_q),
        step_multiplier=float(plan.step_spec.step_multiplier),
    )
    download_df = _rows_for_raster_downloads(df, splits=splits, plan=plan, support=support)
    point_ids = sorted({int(x) for x in download_df[POINT_ID_COL].tolist()})
    if not point_ids:
        return None, 0
    return chip_request_hash(point_ids), len(point_ids)


def load_chip_request_status_for_extract(
    *, study, plan: LayerPlan, splits: list[str]
) -> dict[str, Any] | None:
    request_hash, requested_point_count = resolve_chip_request_hash(
        study=study, plan=plan, splits=splits
    )
    if request_hash is None:
        return None
    status = read_request_status(
        study_dir=study.study_dir,
        rasters_hash=str(plan.rasters_hash),
        request_hash=request_hash,
    )
    if status is None:
        return None
    out = {
        "request_hash": str(request_hash),
        "requested_point_count": int(requested_point_count),
        **status,
    }
    report_path = Path(status["report_path"])
    if report_path.exists():
        out["report"] = json.loads(report_path.read_text(encoding="utf-8"))
    return out


def summarize_failed_chip_report(report: dict[str, Any], *, max_examples: int = 3) -> str:
    parts: list[str] = []
    chip_groups = report.get("chip_groups", {})
    if not isinstance(chip_groups, dict):
        return ""
    for group_name, group_stats in chip_groups.items():
        if not isinstance(group_stats, dict):
            continue
        failed = int(group_stats.get("failed", 0))
        if failed <= 0:
            continue
        failures = group_stats.get("failures", [])
        examples: list[str] = []
        if isinstance(failures, list):
            for item in failures[: max(int(max_examples), 0)]:
                if not isinstance(item, dict):
                    continue
                point_id = item.get("point_id")
                if point_id is None:
                    continue
                examples.append(str(point_id))
        example_text = ""
        if examples:
            example_text = f" example_point_ids={','.join(examples)}"
        parts.append(f"{group_name} failed={failed}{example_text}")
    return "; ".join(parts)
