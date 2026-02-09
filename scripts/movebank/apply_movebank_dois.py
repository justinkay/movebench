#!/usr/bin/env python3
"""Apply DOI URLs from movebank_links.csv to study metadata."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update metadata CSV and per-study JSONs using DOI info gathered during scraping."
    )
    parser.add_argument(
        "--links",
        default="movebank_links.csv",
        help="CSV produced by scrape_movebank_repository.py (default: %(default)s).",
    )
    parser.add_argument(
        "--metadata",
        default="mbr_studies_metadata.csv",
        help="Metadata CSV to update (default: %(default)s).",
    )
    parser.add_argument(
        "--studies-dir",
        default="mbr_studies",
        help="Directory containing per-study JSON metadata (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV. When omitted, overwrites metadata after creating a .bak backup.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def load_doi_mapping(links_csv: Path) -> Dict[str, Dict[str, str]]:
    df = pd.read_csv(links_csv)
    mapping: Dict[str, Dict[str, str]] = {}
    if "study_id" not in df.columns:
        raise ValueError("Links CSV must include a study_id column.")

    doi_col = "doi_url" if "doi_url" in df.columns else None
    title_col = "paper_title" if "paper_title" in df.columns else None

    for _, row in df.iterrows():
        study_id = str(row["study_id"]).strip()
        if not study_id:
            continue
        values: Dict[str, str] = {}
        if doi_col and isinstance(row.get(doi_col), str) and row[doi_col].strip():
            values["doi_url"] = row[doi_col].strip()
        if title_col and isinstance(row.get(title_col), str) and row[title_col].strip():
            values["paper_title"] = row[title_col].strip()
        mapping[study_id] = values
    return mapping


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    links_csv = Path(args.links)
    metadata_csv = Path(args.metadata)
    studies_dir = Path(args.studies_dir)

    if not links_csv.exists():
        raise FileNotFoundError(f"Links CSV not found: {links_csv}")
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    doi_mapping = load_doi_mapping(links_csv)
    logging.info("Loaded DOI info for %s studies", len(doi_mapping))

    df = pd.read_csv(metadata_csv)
    if "study_id" not in df.columns:
        raise ValueError("Metadata CSV must include a study_id column.")
    if "paper_url" not in df.columns:
        df["paper_url"] = ""

    updates = 0
    for idx, row in df.iterrows():
        study_id = str(row["study_id"]).strip()
        if not study_id:
            continue
        info = doi_mapping.get(study_id)
        if not info:
            continue
        doi_url = info.get("doi_url")
        if doi_url and doi_url != row.get("paper_url"):
            df.at[idx, "paper_url"] = doi_url
            updates += 1
        paper_title = info.get("paper_title")
        if paper_title and "paper_title" in df.columns and paper_title != row.get("paper_title"):
            df.at[idx, "paper_title"] = paper_title

        study_dir = studies_dir / str(row.get("study_name", ""))
        meta_path = study_dir / "study_metadata.json"
        if doi_url and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            changed = False
            if meta.get("paper_url") != doi_url:
                meta["paper_url"] = doi_url
                changed = True
            if meta.get("doi_url") != doi_url:
                meta["doi_url"] = doi_url
                changed = True
            if paper_title and meta.get("title") and meta.get("title").lower() == "no thumbnail available":
                meta["title"] = paper_title
                changed = True
            if paper_title and meta.get("paper_title") != paper_title:
                meta["paper_title"] = paper_title
                changed = True
            if changed:
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.output:
        output_path = Path(args.output)
    else:
        backup_path = metadata_csv.with_suffix(metadata_csv.suffix + ".bak")
        metadata_csv.replace(backup_path)
        logging.info("Backed up original metadata CSV to %s", backup_path)
        output_path = metadata_csv

    df.to_csv(output_path, index=False)
    logging.info("Updated %s metadata rows with DOI URLs", updates)


if __name__ == "__main__":
    main()

