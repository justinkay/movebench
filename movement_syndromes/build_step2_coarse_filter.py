#!/usr/bin/env python3
"""Coarse filter for movement-syndrome candidates with fix-interval checks."""

from __future__ import annotations

import argparse
import csv
import io
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import zipfile

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

TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d",
)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MAX_ROWS_PER_STUDY = 500_000
MAX_INTERVALS_PER_INDIVIDUAL = 5_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter metadata rows to those that plausibly meet the Abrahms et al. "
            "movement-syndrome requirements, including duration and fix cadence."
        )
    )
    parser.add_argument(
        "--input",
        default="movement_syndromes/mbr_studies_metadata_step1.csv",
        help="Path to the step1 metadata snapshot (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="movement_syndromes/mbr_studies_metadata_step2_coarse_filter.csv",
        help="Path for the filtered metadata CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--study-root",
        default="mbr_studies",
        help="Directory containing per-study raw CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--min-duration-days",
        type=float,
        default=45.0,
        help="Minimum study duration to consider (default: %(default)s days).",
    )
    parser.add_argument(
        "--max-median-interval-hours",
        type=float,
        default=3.0,
        help="Maximum allowable median fix interval (default: %(default)s hours).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity (default: %(default)s).",
    )
    return parser


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
        for fmt in TIMESTAMP_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def parse_duration_days(start: str, end: str) -> Optional[float]:
    start_dt = parse_iso_datetime(start)
    end_dt = parse_iso_datetime(end)
    if not start_dt or not end_dt:
        return None
    delta = end_dt - start_dt
    return max(delta.total_seconds() / 86400.0, 0.0)


def safe_int(value: str) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def detect_field(fieldnames: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    names_lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        match = names_lower.get(candidate.lower())
        if match:
            return match
    return None


def timestamp_to_epoch_seconds(ts: datetime) -> Optional[float]:
    if not ts:
        return None
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
        members = [m for m in archive.namelist() if not m.endswith("/")]
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


def compute_median_fix_hours(csv_path: Path) -> Tuple[Optional[float], int, Optional[str]]:
    try:
        reader, handle = open_csv_reader(csv_path)
    except FileNotFoundError:
        return None, 0, "file_missing"
    except RuntimeError as exc:
        logging.warning("Unable to read %s: %s", csv_path, exc)
        return None, 0, "invalid_archive"

    try:
        fieldnames = reader.fieldnames or []
        id_col = detect_field(fieldnames, INDIVIDUAL_COLUMNS)
        ts_col = detect_field(fieldnames, TIMESTAMP_COLUMNS)
        if not id_col or not ts_col:
            return None, 0, "missing_id_or_timestamp"

        prev_timestamps: Dict[str, float] = {}
        interval_map: Dict[str, List[float]] = defaultdict(list)
        processed = 0
        truncated = False
        for row in reader:
            processed += 1
            ident = (row.get(id_col) or "").strip()
            if not ident:
                continue
            ts_value = row.get(ts_col)
            ts = parse_iso_datetime(ts_value or "")
            if not ts:
                continue
            epoch_seconds = timestamp_to_epoch_seconds(ts)
            if epoch_seconds is None:
                continue

            prev_value = prev_timestamps.get(ident)
            if prev_value is not None:
                delta = epoch_seconds - prev_value
                if delta > 0:
                    intervals = interval_map[ident]
                    if len(intervals) < MAX_INTERVALS_PER_INDIVIDUAL:
                        intervals.append(delta)
            prev_timestamps[ident] = epoch_seconds

            if processed >= MAX_ROWS_PER_STUDY:
                truncated = True
                break

        if processed == 0:
            return None, 0, "empty_file"

        individual_medians: List[float] = []
        usable_individuals = 0
        for intervals in interval_map.values():
            if not intervals:
                continue
            individual_medians.append(median(intervals) / 3600.0)
            usable_individuals += 1

        if not individual_medians:
            return None, usable_individuals, "insufficient_intervals"

        study_median = median(individual_medians)
        note = "truncated_sample" if truncated else None
        return study_median, usable_individuals, note
    finally:
        handle.close()


def analyze_row(
    row: Dict[str, str],
    study_root: Path,
    min_duration_days: float,
    max_interval_hours: float,
) -> Dict[str, object]:
    duration_days = parse_duration_days(row.get("start_date", ""), row.get("end_date", ""))
    num_individuals = safe_int(row.get("num_individuals", ""))
    study_species = (row.get("study_species") or "").strip()
    csv_name = (row.get("main_csv_file") or "").strip()
    study_dir = study_root / (row.get("study_name") or "")
    csv_path = study_dir / csv_name if csv_name else None

    issues: List[str] = []
    non_blocking: List[str] = []

    if not study_species or study_species.lower() == "unknown":
        issues.append("unknown_species")

    if num_individuals is None or num_individuals <= 0:
        issues.append("no_individuals")

    if duration_days is None:
        issues.append("missing_duration")
    elif duration_days < min_duration_days:
        issues.append("short_duration")

    if not csv_name:
        issues.append("missing_main_csv")
    elif not study_dir.exists():
        issues.append("missing_study_directory")
    elif csv_path and not csv_path.exists():
        issues.append("main_csv_not_found")

    median_hours: Optional[float] = None
    cadence_issue: Optional[str] = None
    usable_individuals = 0

    if not issues and csv_path:
        median_hours, usable_individuals, cadence_issue = compute_median_fix_hours(csv_path)
        if cadence_issue:
            non_blocking.append(cadence_issue)

    status: str
    if issues:
        status = "excluded"
    elif median_hours is None:
        status = "needs_interval_review"
    elif median_hours <= max_interval_hours:
        status = "candidate"
    else:
        status = "needs_resampling"

    notes = ";".join(issues + non_blocking)

    return {
        "duration_days": f"{duration_days:.2f}" if duration_days is not None else "",
        "median_fix_hours": f"{median_hours:.3f}" if median_hours is not None else "",
        "median_fix_individuals": str(usable_individuals) if usable_individuals else "",
        "filter_status": status,
        "filter_notes": notes,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    input_path = Path(args.input)
    output_path = Path(args.output)
    study_root = Path(args.study_root)

    if not input_path.exists():
        raise FileNotFoundError(f"Input metadata not found: {input_path}")
    if not study_root.exists():
        raise FileNotFoundError(f"Study root not found: {study_root}")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Input CSV missing header")
        rows: List[Dict[str, str]] = []
        statuses = Counter()
        for row in reader:
            logging.debug("Processing study %s (%s)", row.get("study_id"), row.get("study_name"))
            summary = analyze_row(
                row,
                study_root=study_root,
                min_duration_days=args.min_duration_days,
                max_interval_hours=args.max_median_interval_hours,
            )
            row.update({k: str(v) for k, v in summary.items()})
            statuses[row["filter_status"]] += 1
            rows.append(row)

    fieldnames = list(reader.fieldnames) + [
        "duration_days",
        "median_fix_hours",
        "median_fix_individuals",
        "filter_status",
        "filter_notes",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Wrote %s rows to %s", len(rows), output_path)
    for status, count in statuses.items():
        logging.info("  %s: %s", status, count)


if __name__ == "__main__":
    main()
