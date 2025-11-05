#!/usr/bin/env python3
"""Populate real article DOIs for Movebank studies by rescraping dataset pages."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.downloader import MovebankStudyDownloader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch DOI links from Movebank dataset pages and update metadata CSV + JSON."
    )
    parser.add_argument(
        "--metadata",
        default="mbr_studies_metadata.csv",
        help="Path to metadata CSV to update (default: %(default)s).",
    )
    parser.add_argument(
        "--studies-dir",
        default="mbr_studies",
        help="Directory containing study downloads (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV. Overwrites input when omitted after backing up to *.bak.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between HTTP requests in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint JSON to resume progress.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    metadata_path = Path(args.metadata)
    studies_dir = Path(args.studies_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if "study_name" not in df.columns:
        raise ValueError("Metadata CSV must include 'study_name'.")
    if "paper_url" not in df.columns:
        df["paper_url"] = ""

    processed_ids: set[str] = set()
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            try:
                resume_data = json.loads(resume_path.read_text(encoding="utf-8"))
                processed_ids = set(resume_data.get("processed", []))
                logging.info("Loaded %s processed study IDs from %s", len(processed_ids), resume_path)
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning("Failed to read resume file %s: %s", resume_path, exc)

    downloader = MovebankStudyDownloader(
        base_dir=studies_dir,
        skip_existing=True,
        extraction_enabled=False,
        delay_between_requests=args.delay,
    )

    updates = 0
    failures = []
    new_urls: list[str] = []

    for study_name, current_url in zip(df["study_name"], df["paper_url"]):
        study_dir = studies_dir / str(study_name)
        study_id = str(study_name).split("_")[-1] if "_" in str(study_name) else ""
        if study_id in processed_ids:
            new_urls.append(current_url)
            continue

        meta_path = study_dir / "study_metadata.json"
        if not meta_path.exists():
            logging.warning("Missing study_metadata.json for %s; skipping", study_dir)
            new_urls.append(current_url)
            continue

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Failed to parse %s: %s", meta_path, exc)
            new_urls.append(current_url)
            continue

        if isinstance(current_url, str) and "doi.org" in current_url:
            new_urls.append(current_url)
            processed_ids.add(study_id)
            continue

        source_url = meta.get("source_url") or meta.get("paper_url") or current_url
        if not isinstance(source_url, str) or not source_url.strip():
            logging.warning("No source URL for %s; skipping", study_dir)
            new_urls.append(current_url)
            processed_ids.add(study_id)
            continue

        try:
            soup = downloader._get_soup(source_url.strip())
            metadata = downloader._extract_metadata(soup)
            doi = metadata.get("doi_url")
            if not doi:
                citation = metadata.get("citation")
                if citation:
                    from movebench.metadata import extract_doi_url  # imported lazily

                    doi = extract_doi_url(citation)
            if doi:
                if doi != current_url:
                    updates += 1
                new_urls.append(doi)
                meta["paper_url"] = doi
                meta["doi_url"] = doi
                if metadata.get("citation"):
                    meta["citation"] = metadata["citation"]
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            else:
                logging.warning("No DOI found for %s", source_url)
                new_urls.append(current_url)
                failures.append({"study_name": study_name, "url": source_url})
            processed_ids.add(study_id)
            time.sleep(args.delay)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to fetch DOI for %s: %s", study_name, exc)
            failures.append({"study_name": study_name, "url": source_url, "error": str(exc)})
            new_urls.append(current_url)

        if args.resume:
            Path(args.resume).write_text(json.dumps({"processed": sorted(processed_ids)}, indent=2))

    df["paper_url"] = new_urls

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
        metadata_path.replace(backup_path)
        logging.info("Backed up original CSV to %s", backup_path)
        output_path = metadata_path

    df.to_csv(output_path, index=False)
    logging.info("Updated paper URLs in %s (changed %s rows)", output_path, updates)
    if failures:
        logging.warning("Could not resolve DOI for %s studies", len(failures))
        logging.debug("Failures: %s", failures)


if __name__ == "__main__":
    main()

