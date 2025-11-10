#!/usr/bin/env python3
"""Build Responses API payloads with Sci-Hub sourced publication PDFs.

Given the JSONL file produced by ``create_gpt_prompts_for_papers.py`` this script:

1. Uses the unofficial Sci-Hub client bundled in ``scihub.py`` to download each
   referenced publication (identified via DOI or direct URL) as a PDF.
2. Stores every downloaded document under a configurable directory, grouping by
   study identifier when available so multiple papers per dataset remain
   discoverable later.
3. Emits a new JSONL file whose Requests payloads include an ``input_file``
   reference pointing at the cached PDF, ready for upload to the OpenAI API.

If Sci-Hub fails (captcha, missing DOI, etc.), the script keeps the original
payload and records the error so the row can be retried manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import unquote, urlparse

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure the bundled scihub client is importable.
SCI_HUB_SRC = REPO_ROOT / "scihub.py" / "scihub" / "scihub.py"
if not SCI_HUB_SRC.exists():
    raise SystemExit(f"Cannot locate Sci-Hub client at {SCI_HUB_SRC}")

import importlib.util

spec = importlib.util.spec_from_file_location("scihub_client", SCI_HUB_SRC)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)  # type: ignore[misc]

SciHub = module.SciHub
CaptchaNeedException = getattr(module, "CaptchaNeedException", Exception)

LOGGER = logging.getLogger("build_requests_with_scihub")

DEFAULT_INPUT = REPO_ROOT / "papers" / "chatgpt_requests.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "papers" / "chatgpt_requests_with_pdfs.jsonl"
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "papers" / "scihub_downloads"


@dataclass
class RequestEntry:
    raw: Dict[str, Any]

    @property
    def row_index(self) -> Optional[int]:
        value = self.raw.get("row_index")
        return int(value) if isinstance(value, int) else None

    @property
    def study_id(self) -> Optional[str]:
        value = self.raw.get("study_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        metadata = (
            self.raw.get("chat_request", {}).get("metadata")
            if isinstance(self.raw.get("chat_request"), dict)
            else None
        )
        if isinstance(metadata, dict):
            meta_id = metadata.get("study_id")
            if isinstance(meta_id, str) and meta_id.strip():
                return meta_id.strip()
        return None

    @property
    def publication_url(self) -> Optional[str]:
        value = self.raw.get("publication_url")
        if isinstance(value, str) and value.strip():
            return value.strip()
        metadata = (
            self.raw.get("chat_request", {}).get("metadata")
            if isinstance(self.raw.get("chat_request"), dict)
            else None
        )
        if isinstance(metadata, dict):
            meta_url = metadata.get("publication_url")
            if isinstance(meta_url, str) and meta_url.strip():
                return meta_url.strip()
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download publications with Sci-Hub and rebuild Requests payloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source JSONL file with original chat_request entries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file with PDF-aware payloads.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Directory where PDFs will be cached.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout (seconds) for Sci-Hub HTTP calls.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_entries(path: Path) -> Iterator[RequestEntry]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.error("Skipping malformed JSON on line %s: %s", idx, exc)
                continue
            if not isinstance(data, dict):
                LOGGER.error("Skipping non-object entry on line %s.", idx)
                continue
            yield RequestEntry(raw=data)


def slugify(value: str, allow_period: bool = False) -> str:
    value = value.strip().lower()
    if not allow_period:
        value = value.replace(".", "-")
        pattern = r"[^a-z0-9\-]+"
    else:
        pattern = r"[^a-z0-9\-.]+"
    value = re.sub(pattern, "-", value)
    value = re.sub(r"-{2,}", "-", value)
    value = value.strip("-")
    return value or "item"


def extract_identifier(url_or_doi: str) -> Tuple[str, str]:
    """Return (identifier_for_fetch, file_label) for Sci-Hub lookup and storage."""
    url_or_doi = url_or_doi.strip()
    parsed = urlparse(url_or_doi)
    if parsed.scheme and parsed.netloc:
        netloc = parsed.netloc.lower()
        path = unquote(parsed.path or "").strip("/")
        label_base = slugify(path or netloc, allow_period=True)
        if "doi.org" in netloc and path:
            # Use the full DOI URL when contacting Sci-Hub as requested.
            return url_or_doi, label_base or slugify(path, allow_period=True)
        return url_or_doi, (slugify(netloc) + "-" + slugify(path or "doc"))
    return url_or_doi, slugify(url_or_doi, allow_period=True)


def clean_existing_file_blocks(request: Dict[str, Any]) -> None:
    payload_input = request.get("input")
    if not isinstance(payload_input, list):
        return
    cleaned: list[Any] = []
    for block in payload_input:
        if not isinstance(block, dict):
            cleaned.append(block)
            continue
        block_copy = dict(block)
        contents = block_copy.get("content")
        if isinstance(contents, list):
            block_copy["content"] = [
                item
                for item in contents
                if not (isinstance(item, dict) and item.get("type") == "input_file")
            ]
            if not block_copy["content"]:
                block_copy.pop("content")
        cleaned.append(block_copy)
    request["input"] = cleaned


def append_file_block(request: Dict[str, Any], file_path: Path) -> None:
    payload_input = request.setdefault("input", [])
    if not isinstance(payload_input, list):
        raise TypeError("chat_request.input must be a list to append file context.")
    attachment_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "Reference paper attached. Use it to extract validated "
                    "environmental covariates as requested."
                ),
            },
            {
                "type": "input_file",
                "file_path": str(file_path),
                "description": "Publication PDF downloaded via Sci-Hub",
            },
        ],
    }
    payload_input.append(attachment_message)


def download_pdf(sh: SciHub, identifier: str, destination: Path, *, timeout: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sh.sess.timeout = timeout  # type: ignore[attr-defined]
    data = sh.fetch(identifier)
    if not data or "pdf" not in data or not data["pdf"]:
        raise RuntimeError(f"Sci-Hub did not return a PDF for identifier {identifier}")
    with destination.open("wb") as handle:
        handle.write(data["pdf"])
    return destination


def process_entries(
    entries: Iterable[RequestEntry],
    *,
    download_root: Path,
    timeout: float,
) -> Iterator[Dict[str, Any]]:
    sh = SciHub()
    cache: Dict[str, Path] = {}
    for entry in entries:
        identifier = entry.publication_url
        base_payload = {k: v for k, v in entry.raw.items() if k != "chat_request"}
        cloned_request = entry.raw.get("chat_request")
        if not isinstance(cloned_request, dict):
            LOGGER.warning("Row %s missing chat_request; emitting original.", entry.row_index)
            yield entry.raw
            continue

        augmented_request = json.loads(json.dumps(cloned_request))

        if not identifier:
            LOGGER.warning("Row %s missing publication URL; skipping download.", entry.row_index)
            yield {
                **base_payload,
                "chat_request": augmented_request,
                "download_error": "missing_publication_url",
            }
            continue

        normalized_id, label = extract_identifier(identifier)
        study_id = entry.study_id or f"row-{entry.row_index or 'unknown'}"
        study_slug = slugify(study_id)
        target_dir = download_root / study_slug
        pdf_path = target_dir / f"{label}.pdf"

        download_error: Optional[str] = None
        if normalized_id in cache:
            pdf_path = cache[normalized_id]
            LOGGER.debug("Reusing cached PDF for %s -> %s", normalized_id, pdf_path)
        else:
            try:
                pdf_path = download_pdf(sh, normalized_id, pdf_path, timeout=timeout)
                cache[normalized_id] = pdf_path
                LOGGER.info("Downloaded %s to %s", normalized_id, pdf_path)
            except CaptchaNeedException as exc:
                download_error = f"captcha_required: {exc}"
            except Exception as exc:
                download_error = f"download_failed: {exc}"

        clean_existing_file_blocks(augmented_request)

        if download_error or not pdf_path.exists():
            output_entry = {
                **base_payload,
                "chat_request": augmented_request,
                "download_error": download_error or "file_missing",
            }
            yield output_entry
            continue

        append_file_block(augmented_request, pdf_path)
        output_entry = {
            **base_payload,
            "chat_request": augmented_request,
            "context_files": [
                {
                    "path": str(pdf_path),
                    "study_id": study_id,
                    "identifier": normalized_id,
                }
            ],
            "download_error": None,
        }
        yield output_entry


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input JSONL not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.download_root.mkdir(parents=True, exist_ok=True)

    entries = list(iter_entries(args.input))
    if not entries:
        LOGGER.warning("No records found in %s; exiting.", args.input)
        return

    LOGGER.info("Processing %s records via Sci-Hub.", len(entries))
    with args.output.open("w", encoding="utf-8") as sink:
        for result in process_entries(
            entries,
            download_root=args.download_root,
            timeout=args.timeout,
        ):
            sink.write(json.dumps(result, ensure_ascii=False) + "\n")

    LOGGER.info("Completed. Output written to %s", args.output)


if __name__ == "__main__":
    main()
