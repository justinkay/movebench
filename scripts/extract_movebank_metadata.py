#!/usr/bin/env python3
"""CLI entrypoint for extracting Movebank study metadata."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.metadata import extract_metadata  # noqa: E402
from movebench.downloader import MovebankStudyDownloader, StudyRecord  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract metadata from downloaded Movebank studies."
    )
    parser.add_argument(
        "--studies-dir",
        default="mbr_studies",
        help="Directory containing downloaded study folders.",
    )
    parser.add_argument(
        "--links-csv",
        default="movebank_links.csv",
        help="CSV listing expected study IDs (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="mbr_studies_metadata.csv",
        help="Destination CSV for extracted metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    downloader = MovebankStudyDownloader(
        base_dir=Path(args.studies_dir),
        skip_existing=False,
    )

    def redownload(metadata: dict) -> bool:
        study_id = metadata.get("study_id")
        source_url = metadata.get("source_url") or metadata.get("raw_row", {}).get("url")
        if not study_id or not source_url:
            logging.warning("Cannot redownload study without study_id and source_url: %s", metadata)
            return False
        raw_row = metadata.get("raw_row", {})
        record = StudyRecord(
            study_id=study_id,
            url=source_url,
            raw_title=raw_row.get("text", metadata.get("title", "")),
            csv_row=raw_row,
        )
        try:
            downloader.download_study(record)
            return True
        except Exception as exc:
            logging.error("Re-download failed for %s: %s", study_id, exc)
            return False

    summary = extract_metadata(
        studies_dir=Path(args.studies_dir),
        links_csv=Path(args.links_csv),
        output_csv=Path(args.output),
        redownload_callback=redownload,
    )

    print(json.dumps(summary, indent=2))

    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
