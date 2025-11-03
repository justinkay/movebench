#!/usr/bin/env python3
"""Download Movebank repository studies listed in a CSV."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.downloader import MovebankStudyDownloader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Movebank repository datasets from CSV."
    )
    parser.add_argument(
        "--csv",
        default="movebank_links.csv",
        help="Path to CSV produced by scrape_movebank_repository.py",
    )
    parser.add_argument(
        "--output-dir",
        default="mbr_studies",
        help="Directory where studies will be saved.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip studies whose directory already exists.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between file downloads in seconds.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Disable automatic archive extraction.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Maximum number of download attempts (retries) to run.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Initial request timeout in seconds.",
    )
    parser.add_argument(
        "--timeout-step",
        type=float,
        default=30.0,
        help="Additional timeout seconds to add before each retry attempt.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=5.0,
        help="Delay between attempts when retries are needed.",
    )
    parser.add_argument(
        "--report",
        default="download_results.json",
        help="JSON file path where the detailed results will be written.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional JSON report path; if provided, only failed study IDs from that report will be retried.",
    )
    return parser


def load_failed_ids(report_path: Optional[str]) -> Set[str]:
    if not report_path:
        return set()
    path = Path(report_path)
    if not path.exists():
        logging.warning("Resume report %s not found; starting fresh.", report_path)
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse resume report %s: %s", report_path, exc)
        return set()

    failed_entries = data.get("failed", [])
    if isinstance(failed_entries, dict):
        failed_entries = failed_entries.values()
    failed_ids = {
        entry.get("study_id")
        for entry in failed_entries
        if isinstance(entry, dict) and entry.get("study_id")
    }

    if not failed_ids and "attempts" in data:
        for attempt in data.get("attempts", []):
            for result in attempt.get("results", []):
                if result.get("status") == "failed" and result.get("study_id"):
                    failed_ids.add(result["study_id"])

    logging.info("Loaded %d failed study IDs from %s", len(failed_ids), report_path)
    return failed_ids


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    remaining_ids: Optional[Set[str]] = load_failed_ids(args.resume_from) or None

    attempts_data = []
    downloaded_map: dict[str, dict] = {}
    failure_map: dict[str, dict] = {}
    total_attempts = max(1, args.attempts)
    timeout = args.timeout

    for attempt_num in range(1, total_attempts + 1):
        logging.info(
            "Starting attempt %s/%s (timeout=%ss, remaining=%s)",
            attempt_num,
            total_attempts,
            timeout,
            "all" if remaining_ids is None else len(remaining_ids),
        )
        downloader = MovebankStudyDownloader(
            base_dir=Path(args.output_dir),
            extraction_enabled=not args.no_extract,
            skip_existing=args.skip_existing,
            request_timeout=timeout,
            delay_between_requests=args.delay,
        )

        results = downloader.download_all(Path(args.csv), study_ids=remaining_ids)

        for res in results:
            sid = res.get("study_id")
            if not sid:
                continue
            res_with_attempt = {**res, "attempt": attempt_num, "timeout": timeout}
            if res["status"] in {"downloaded", "skipped"}:
                downloaded_map[sid] = res_with_attempt
                failure_map.pop(sid, None)
            elif res["status"] == "failed":
                failure_map[sid] = res_with_attempt

        attempts_data.append(
            {
                "attempt": attempt_num,
                "timeout": timeout,
                "study_ids": None if remaining_ids is None else sorted(remaining_ids),
                "results": results,
            }
        )

        if not failure_map:
            logging.info("All studies downloaded successfully after attempt %s.", attempt_num)
            break

        if attempt_num < total_attempts:
            remaining_ids = set(failure_map.keys())
            timeout += args.timeout_step
            logging.info(
                "%d studies still failing; retrying with timeout=%ss after %.1fs delay.",
                len(remaining_ids),
                timeout,
                args.retry_delay,
            )
            time.sleep(args.retry_delay)
        else:
            logging.info("Reached maximum attempts (%s).", total_attempts)

    report = {
        "summary": {
            "attempts_run": len(attempts_data),
            "downloads_completed": len(downloaded_map),
            "failures_remaining": len(failure_map),
        },
        "downloaded": sorted(downloaded_map.values(), key=lambda x: x.get("study_id", "")),
        "failed": sorted(failure_map.values(), key=lambda x: x.get("study_id", "")),
        "attempts": attempts_data,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Results written to {report_path} (failures remaining: {len(failure_map)})")

    if failure_map:
        sys.exit(1)


if __name__ == "__main__":
    main()

