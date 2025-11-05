#!/usr/bin/env python3
"""Backfill paper URLs in an existing metadata CSV."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.metadata import extract_doi_url  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update paper_url column using citation files and study_metadata.json."
    )
    parser.add_argument(
        "--metadata",
        default="mbr_studies_metadata.csv",
        help="Path to the existing metadata CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--studies-dir",
        default="mbr_studies",
        help="Directory containing study folders (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file. If omitted, overwrite the input after creating a .bak backup.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def _load_citation_texts(study_dir: Path) -> list[str]:
    texts: list[str] = []
    for name in ("citation.txt", "citation.bib"):
        path = study_dir / name
        if path.exists():
            try:
                texts.append(path.read_text(encoding="utf-8"))
            except OSError:
                continue
    return texts


def _derive_paper_url(study_dir: Path) -> Optional[str]:
    meta_path = study_dir / "study_metadata.json"
    download_meta = {}
    if meta_path.exists():
        try:
            download_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            download_meta = {}

    texts = _load_citation_texts(study_dir)
    citation = download_meta.get("citation")
    if isinstance(citation, str) and citation:
        texts.append(citation)

    for text in texts:
        doi_url = extract_doi_url(text)
        if doi_url:
            return doi_url

    doi_url = download_meta.get("doi_url")
    if isinstance(doi_url, str) and doi_url:
        candidate = extract_doi_url(doi_url) or doi_url.strip()
        if candidate:
            return candidate

    for key in ("paper_url", "source_url"):
        value = download_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raw_row = download_meta.get("raw_row")
    if isinstance(raw_row, dict):
        value = raw_row.get("url")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    studies_dir = Path(args.studies_dir)
    df = pd.read_csv(metadata_path)
    if "study_name" not in df.columns:
        raise ValueError("Metadata CSV must contain a 'study_name' column.")
    if "paper_url" not in df.columns:
        df["paper_url"] = ""

    updates = 0
    new_urls: list[str] = []

    for study_name, current_url in zip(df["study_name"], df["paper_url"]):
        study_dir = studies_dir / str(study_name)
        paper_url = _derive_paper_url(study_dir)
        if paper_url and paper_url != current_url:
            updates += 1
            new_urls.append(paper_url)
        else:
            new_urls.append(current_url or (paper_url or ""))  # ensure non-null

    df["paper_url"] = new_urls

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = metadata_path.with_suffix(metadata_path.suffix + ".bak")
        metadata_path.replace(backup_path)
        logging.info("Backed up original file to %s", backup_path)
        output_path = metadata_path

    df.to_csv(output_path, index=False)
    logging.info("Updated paper URLs in %s (changed %s rows)", output_path, updates)


if __name__ == "__main__":
    main()

