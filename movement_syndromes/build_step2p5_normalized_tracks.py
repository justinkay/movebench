#!/usr/bin/env python3
"""Normalize candidate study trajectories (Step 2.5)."""

from __future__ import annotations

import argparse
import csv
import io
import logging
import math
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

INDIVIDUAL_COLUMNS = [
    "individual-local-identifier",
    "individual_local_identifier",
    "individual_id",
]
TIMESTAMP_COLUMNS = [
    "timestamp",
    "timestamp_start",
    "timestamp_end",
    "time",
]
LATITUDE_COLUMNS = [
    "location-lat",
    "location_lat",
    "latitude",
    "lat",
]
LONGITUDE_COLUMNS = [
    "location-long",
    "location_long",
    "longitude",
    "lon",
    "long",
]

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_ROWS_PER_STUDY = 500_000
MAX_INTERVALS_PER_INDIVIDUAL = 5_000
MAX_GAP_FACTOR = 6  # break interpolation if gap exceeds factor * target interval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize per-individual trajectories for candidate studies."
    )
    parser.add_argument(
        "--metadata",
        default="movement_syndromes/mbr_studies_metadata_step2_coarse_filter.csv",
        help="Step-2 metadata CSV with filter_status column (default: %(default)s).",
    )
    parser.add_argument(
        "--study-root",
        default="mbr_studies",
        help="Directory containing per-study raw CSV/ZIP files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="movement_syndromes/normalized_tracks",
        help="Directory to store normalized per-individual trajectories (default: %(default)s).",
    )
    parser.add_argument(
        "--summary",
        default="movement_syndromes/normalized_tracks_index_step2p5.csv",
        help="Output CSV summarizing processed individuals (default: %(default)s).",
    )
    parser.add_argument(
        "--study-id",
        action="append",
        default=[],
        help="Process only the specified study_id(s). Can be provided multiple times.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of studies to process.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def parse_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Metadata CSV missing header")
        rows = [row for row in reader]
    return rows


def select_studies(rows: Iterable[Dict[str, str]], study_ids: List[str]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    study_id_set = {sid for sid in study_ids if sid}
    for row in rows:
        status = (row.get("filter_status") or "").strip().lower()
        if status != "candidate":
            continue
        if study_id_set and (row.get("study_id") not in study_id_set):
            continue
        filtered.append(row)
    return filtered


def detect_field(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    return None


def parse_iso_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def timestamp_to_epoch_seconds(ts: datetime) -> float:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return (ts - EPOCH).total_seconds()


def open_csv_reader(path: Path) -> Tuple[csv.DictReader, io.TextIOBase]:
    if not path.exists():
        raise FileNotFoundError(path)
    if zipfile.is_zipfile(path):
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist() if not name.endswith("/")]
        target = None
        for name in members:
            if name.lower().endswith(".csv"):
                target = name
                break
        if target is None and members:
            target = members[0]
        if target is None:
            raise RuntimeError(f"{path} contains no files")
        raw = archive.open(target)
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)

        class _ArchiveHandle(io.TextIOBase):
            def close(self_nonlocal) -> None:  # type: ignore[override]
                text.close()
                raw.close()
                archive.close()

        return reader, _ArchiveHandle()
    handle = path.open("r", encoding="utf-8", errors="replace", newline="")
    reader = csv.DictReader(handle)
    return reader, handle


class MissingColumnsError(RuntimeError):
    """Raised when required columns are absent in a study file."""


def load_individual_records(csv_path: Path) -> Dict[str, List[Tuple[float, float, float]]]:
    reader, handle = open_csv_reader(csv_path)
    try:
        fieldnames = reader.fieldnames or []
        id_col = detect_field(fieldnames, INDIVIDUAL_COLUMNS)
        ts_col = detect_field(fieldnames, TIMESTAMP_COLUMNS)
        lat_col = detect_field(fieldnames, LATITUDE_COLUMNS)
        lon_col = detect_field(fieldnames, LONGITUDE_COLUMNS)
        missing: List[str] = []
        if not id_col:
            missing.append("individual_id")
        if not ts_col:
            missing.append("timestamp")
        if not lat_col:
            missing.append("latitude")
        if not lon_col:
            missing.append("longitude")
        if missing:
            raise MissingColumnsError(
                f"Missing columns: {', '.join(missing)} in {csv_path.name}"
            )

        per_individual: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        processed = 0
        for row in reader:
            processed += 1
            ident = (row.get(id_col) or "").strip()
            if not ident:
                continue
            ts = parse_iso_datetime(row.get(ts_col, ""))
            if not ts:
                continue
            try:
                lat = float(row.get(lat_col, ""))
                lon = float(row.get(lon_col, ""))
            except (TypeError, ValueError):
                continue
            per_individual[ident].append((timestamp_to_epoch_seconds(ts), lat, lon))
            if processed >= MAX_ROWS_PER_STUDY:
                logging.debug("Reached row cap (%s) for %s", MAX_ROWS_PER_STUDY, csv_path)
                break
        return per_individual
    finally:
        handle.close()


def compute_median_interval_seconds(times: List[float]) -> Optional[float]:
    if len(times) < 2:
        return None
    deltas = [
        times[idx + 1] - times[idx]
        for idx in range(len(times) - 1)
        if times[idx + 1] > times[idx]
    ]
    if not deltas:
        return None
    return median(deltas)


def resample_track(
    records: List[Tuple[float, float, float]], target_interval: float
) -> List[Tuple[float, float, float]]:
    if not records or target_interval <= 0:
        return []
    records.sort(key=lambda item: item[0])
    times = [r[0] for r in records]
    lats = [r[1] for r in records]
    lons = [r[2] for r in records]
    resampled: List[Tuple[float, float, float]] = [(times[0], lats[0], lons[0])]
    idx = 0
    target = times[0] + target_interval
    last_time = times[-1]

    while target < last_time:
        while idx + 1 < len(times) and times[idx + 1] < target:
            idx += 1
        if idx + 1 >= len(times):
            break
        t0, t1 = times[idx], times[idx + 1]
        lat0, lat1 = lats[idx], lats[idx + 1]
        lon0, lon1 = lons[idx], lons[idx + 1]
        gap = t1 - t0
        if gap <= 0:
            idx += 1
            continue
        if gap > MAX_GAP_FACTOR * target_interval:
            # Large gap — anchor at the next real observation and realign target grid.
            resampled.append((t1, lat1, lon1))
            target = t1 + target_interval
            idx += 1
            continue
        ratio = (target - t0) / gap
        lat = lat0 + ratio * (lat1 - lat0)
        lon = lon0 + ratio * (lon1 - lon0)
        resampled.append((target, lat, lon))
        target += target_interval

    if resampled[-1][0] != last_time:
        resampled.append((last_time, lats[-1], lons[-1]))
    return resampled


def sanitize(value: str, default: str = "unknown") -> str:
    value = value.strip() if value else ""
    if not value:
        return default
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_normalized_csv(
    output_path: Path,
    records: List[Tuple[float, float, float]],
) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_iso", "latitude", "longitude"])
        for epoch_seconds, lat, lon in records:
            ts = datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()
            writer.writerow([ts, f"{lat:.9f}", f"{lon:.9f}"])


def process_study(
    row: Dict[str, str],
    study_root: Path,
    output_dir: Path,
) -> List[Dict[str, str]]:
    study_id = row.get("study_id") or ""
    study_name = row.get("study_name") or ""
    csv_name = row.get("main_csv_file") or ""
    study_dir = study_root / study_name
    csv_path = study_dir / csv_name

    if not csv_name:
        logging.warning("Study %s missing main_csv_file", study_id)
        return []
    if not csv_path.exists():
        logging.warning("CSV path not found for study %s: %s", study_id, csv_path)
        return []

    try:
        individual_records = load_individual_records(csv_path)
    except MissingColumnsError as exc:
        logging.warning("Skipping study %s: %s", study_id, exc)
        return []

    logging.info("Normalizing study %s (%s)", study_id, study_name)
    summaries: List[Dict[str, str]] = []

    for individual_id, records in individual_records.items():
        if len(records) < 2:
            continue
        records.sort(key=lambda item: item[0])
        interval_seconds = compute_median_interval_seconds([r[0] for r in records])
        if not interval_seconds or interval_seconds <= 0:
            continue
        resampled = resample_track(records, interval_seconds)
        if len(resampled) < 2:
            continue

        study_dir_name = sanitize(study_name or study_id)
        indiv_name = sanitize(individual_id)
        output_path = output_dir / study_dir_name / f"individual_{indiv_name}.csv"
        write_normalized_csv(output_path, resampled)

        summaries.append(
            {
                "study_id": study_id,
                "study_name": study_name,
                "individual_id": individual_id,
                "target_interval_hours": f"{interval_seconds / 3600.0:.3f}",
                "num_points_original": str(len(records)),
                "num_points_resampled": str(len(resampled)),
                "start_timestamp": datetime.fromtimestamp(
                    records[0][0], tz=timezone.utc
                ).isoformat(),
                "end_timestamp": datetime.fromtimestamp(
                    records[-1][0], tz=timezone.utc
                ).isoformat(),
                "output_file": str(output_path),
            }
        )
    return summaries


def merge_summary(existing_path: Path, new_entries: List[Dict[str, str]]) -> None:
    existing: Dict[Tuple[str, str], Dict[str, str]] = {}
    if existing_path.exists():
        with existing_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = (row.get("study_id") or "", row.get("individual_id") or "")
                existing[key] = row
    for entry in new_entries:
        key = (entry.get("study_id") or "", entry.get("individual_id") or "")
        existing[key] = entry

    if not existing:
        return

    fieldnames = [
        "study_id",
        "study_name",
        "individual_id",
        "target_interval_hours",
        "num_points_original",
        "num_points_resampled",
        "start_timestamp",
        "end_timestamp",
        "output_file",
    ]
    with existing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing.values())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    metadata_path = Path(args.metadata)
    study_root = Path(args.study_root)
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    if not study_root.exists():
        raise FileNotFoundError(f"Study root not found: {study_root}")
    ensure_dir(output_dir)
    ensure_dir(summary_path.parent)

    rows = parse_metadata(metadata_path)
    selected = select_studies(rows, args.study_id)

    if args.limit is not None:
        selected = selected[: args.limit]

    all_summaries: List[Dict[str, str]] = []
    for row in selected:
        summaries = process_study(row, study_root=study_root, output_dir=output_dir)
        all_summaries.extend(summaries)

    merge_summary(summary_path, all_summaries)
    logging.info("Wrote %s normalized individual tracks", len(all_summaries))
    logging.info("Summary index: %s", summary_path)


if __name__ == "__main__":
    main()
