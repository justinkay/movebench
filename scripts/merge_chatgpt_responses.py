#!/usr/bin/env python3
"""Merge ChatGPT responses into the study metadata CSV.

Reads the JSONL log produced by ``run_chatgpt_requests.py`` and adds a new
column with structured responses for each matching study in
``mbr_studies_metadata.csv``. A backup of the original CSV is created before
any modifications are applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping

LOGGER = logging.getLogger("merge_chatgpt_responses")

DEFAULT_RESPONSES_PATH = Path("papers/chatgpt_responses.jsonl")
DEFAULT_METADATA_PATH = Path("mbr_studies_metadata.csv")
DEFAULT_COLUMN_NAME = "chatgpt_responses"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append parsed ChatGPT responses to the study metadata CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=DEFAULT_RESPONSES_PATH,
        help="Path to the chat responses JSONL file.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to the study metadata CSV to patch.",
    )
    parser.add_argument(
        "--column",
        default=DEFAULT_COLUMN_NAME,
        help="Column name to store serialized responses.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def load_responses(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    responses: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        LOGGER.warning("Responses file does not exist: %s", path)
        return responses

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.error("Skipping invalid JSONL entry on line %s: %s", line_no, exc)
                continue

            study_id = payload.get("study_id")
            if not study_id:
                LOGGER.warning("Missing study_id in response line %s; skipping.", line_no)
                continue

            message = (
                payload.get("response", {})
                .get("choices", [{}])[0]
                .get("message", {})
            )
            content = message.get("content")

            responses[study_id].append(
                {
                    "row_index": payload.get("row_index"),
                    "publication_url": payload.get("publication_url"),
                    "response": content,
                    "error": payload.get("error"),
                    "received_at": payload.get("received_at"),
                }
            )
    LOGGER.info("Loaded responses for %s study_id entries.", len(responses))
    return responses


def create_backup(path: Path) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    backup = path.with_name(f"{path.name}.{timestamp}.bak")
    shutil.copy2(path, backup)
    LOGGER.info("Created backup at %s", backup)
    return backup


def update_metadata(
    metadata_path: Path,
    responses: Dict[str, List[Dict[str, Any]]],
    column_name: str,
) -> None:
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if column_name not in fieldnames:
            fieldnames.append(column_name)

        rows: List[MutableMapping[str, Any]] = []
        for row in reader:
            study_id = row.get("study_id")
            records = responses.get(study_id, [])
            row[column_name] = json.dumps(records, ensure_ascii=False) if records else ""
            rows.append(row)

    fd, temp_path = tempfile.mkstemp(
        prefix=metadata_path.stem + "_", suffix=".csv", dir=str(metadata_path.parent)
    )
    os.close(fd)
    temp_file = Path(temp_path)
    try:
        with temp_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temp_file.replace(metadata_path)
    finally:
        if temp_file.exists() and temp_file != metadata_path:
            temp_file.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.metadata.exists():
        raise SystemExit(f"Metadata file not found: {args.metadata}")

    responses = load_responses(args.responses)
    create_backup(args.metadata)
    update_metadata(args.metadata, responses, args.column)
    LOGGER.info("Metadata file updated with ChatGPT responses.")


if __name__ == "__main__":
    main()
