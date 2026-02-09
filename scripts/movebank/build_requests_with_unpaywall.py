#!/usr/bin/env python3
"""Augment request payloads with PDFs discovered via the Unpaywall API.

Workflow:
1. Read the JSONL payload emitted by ``create_gpt_prompts_for_papers.py``.
2. For every entry, resolve the DOI (if available) and query Unpaywall.
3. When Unpaywall provides an open-access PDF, download it to a local cache.
4. Append an ``input_file`` block referencing the cached PDF so downstream
   callers can upload it prior to invoking the OpenAI Responses API.

Unpaywall attribution email defaults to kayj@mit.edu, per user request.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger("build_requests_with_unpaywall")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = REPO_ROOT / "papers" / "chatgpt_requests.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "papers" / "chatgpt_requests_with_unpaywall.jsonl"
DEFAULT_DOWNLOAD_ROOT = REPO_ROOT / "papers" / "unpaywall_downloads"
DEFAULT_EMAIL = "kayj@mit.edu"

UNPAYWALL_API = "https://api.unpaywall.org/v2/{doi}"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


@dataclass
class RequestEntry:
    raw: Dict[str, Any]

    def row_index(self) -> Optional[int]:
        value = self.raw.get("row_index")
        return int(value) if isinstance(value, int) else None

    def study_id(self) -> Optional[str]:
        value = self.raw.get("study_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
        chat_request = self.raw.get("chat_request")
        if isinstance(chat_request, dict):
            metadata = chat_request.get("metadata")
            if isinstance(metadata, dict):
                meta_id = metadata.get("study_id")
                if isinstance(meta_id, str) and meta_id.strip():
                    return meta_id.strip()
        return None

    def publication_url(self) -> Optional[str]:
        value = self.raw.get("publication_url")
        if isinstance(value, str) and value.strip():
            return value.strip()
        chat_request = self.raw.get("chat_request")
        if isinstance(chat_request, dict):
            metadata = chat_request.get("metadata")
            if isinstance(metadata, dict):
                meta_url = metadata.get("publication_url")
                if isinstance(meta_url, str) and meta_url.strip():
                    return meta_url.strip()
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OA PDFs via Unpaywall and rebuild request payloads.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Source JSONL file produced by create_gpt_prompts_for_papers.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file with augmented payloads.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Directory where downloaded PDFs will be cached.",
    )
    parser.add_argument(
        "--email",
        default=DEFAULT_EMAIL,
        help="Email address required by the Unpaywall API.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Timeout (seconds) for HTTP requests.",
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
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                LOGGER.error("Skipping malformed JSON on line %s: %s", line_number, exc)
                continue
            if not isinstance(payload, dict):
                LOGGER.error("Skipping non-object entry on line %s.", line_number)
                continue
            yield RequestEntry(raw=payload)


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


def extract_doi(url_or_identifier: str) -> Optional[str]:
    candidate = url_or_identifier.strip()
    if not candidate:
        return None

    parsed = urlparse(candidate)
    if parsed.scheme and parsed.netloc:
        netloc = parsed.netloc.lower()
        if "doi.org" in netloc and parsed.path:
            return parsed.path.strip("/ ")
        if re.match(r"10\.\d{4,9}/.+", parsed.path or "", re.IGNORECASE):
            return parsed.path.strip("/ ")
        # Not a DOI URL; nothing to extract.
        return None

    if re.match(r"^10\.\d{4,9}/.+", candidate, re.IGNORECASE):
        return candidate

    return None


def normalize_chat_request(chat_request: Dict[str, Any]) -> Dict[str, Any]:
    copy = json.loads(json.dumps(chat_request))
    content = copy.get("input")
    if isinstance(content, list):
        cleaned = []
        for block in content:
            if not isinstance(block, dict):
                cleaned.append(block)
                continue
            block_copy = dict(block)
            block_content = block_copy.get("content")
            if isinstance(block_content, list):
                block_copy["content"] = [
                    item
                    for item in block_content
                    if not (isinstance(item, dict) and item.get("type") == "input_file")
                ]
                if not block_copy["content"]:
                    block_copy.pop("content")
            cleaned.append(block_copy)
        copy["input"] = cleaned
    return copy


def append_file_context(request: Dict[str, Any], file_path: Path, source_url: str) -> None:
    payload_input = request.setdefault("input", [])
    if not isinstance(payload_input, list):
        raise TypeError("chat_request.input must be a list.")
    payload_input.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "The attached file contains the full publication text "
                        "retrieved via Unpaywall. Use it to extract the requested "
                        "environmental covariates."
                    ),
                },
                {
                    "type": "input_file",
                    "file_path": str(file_path),
                    "description": f"Unpaywall-sourced PDF ({source_url})",
                },
            ],
        }
    )


def pick_best_location(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(record, dict):
        return None
    for key in ("best_oa_location", "first_oa_location"):
        location = record.get(key)
        if isinstance(location, dict):
            return location
    locations = record.get("oa_locations")
    if isinstance(locations, list):
        for location in locations:
            if isinstance(location, dict):
                return location
    return None


def determine_download_url(location: Dict[str, Any]) -> Optional[str]:
    if not isinstance(location, dict):
        return None
    for key in ("url_for_pdf", "url"):
        value = location.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def download_file(session: requests.Session, url: str, destination: Path, *, timeout: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(url, headers=HTTP_HEADERS, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "application/pdf" not in content_type:
            destination = destination.with_suffix(".pdf")
        first_chunk: bytes | None = None
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = chunk
                    if b"%PDF" not in first_chunk[:1024]:
                        handle.close()
                        destination.unlink(missing_ok=True)
                        raise ValueError(
                            f"Downloaded content is not a PDF (content-type={content_type or 'unknown'})"
                        )
                    handle.write(first_chunk)
                    continue
                handle.write(chunk)
        if first_chunk is None:
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded file was empty.")
    return destination


def process_entries(
    entries: Iterable[RequestEntry],
    *,
    email: str,
    download_root: Path,
    timeout: float,
) -> Iterator[Dict[str, Any]]:
    session = requests.Session()
    cache: Dict[str, Path] = {}
    unpaywall_cache: Dict[str, Dict[str, Any]] = {}

    try:
        for entry in entries:
            chat_request = entry.raw.get("chat_request")
            if not isinstance(chat_request, dict):
                LOGGER.warning(
                    "Row %s missing chat_request field; skipping augmentation.",
                    entry.row_index(),
                )
                yield entry.raw
                continue

            doi = None
            publication_url = entry.publication_url()
            if publication_url:
                doi = extract_doi(publication_url)
            if not doi:
                LOGGER.info(
                    "Row %s has no resolvable DOI (publication_url=%r); skipping.",
                    entry.row_index(),
                    publication_url,
                )
                yield {
                    **entry.raw,
                    "download_error": "missing_doi",
                }
                continue

            doi_key = doi.lower()
            try:
                if doi_key not in unpaywall_cache:
                    api_url = UNPAYWALL_API.format(doi=doi)
                    resp = session.get(api_url, params={"email": email}, timeout=timeout)
                    if resp.status_code == 404:
                        LOGGER.warning("Unpaywall has no entry for DOI %s", doi)
                        unpaywall_cache[doi_key] = {"error": "not_found"}
                    else:
                        resp.raise_for_status()
                        unpaywall_cache[doi_key] = resp.json()

                record = unpaywall_cache.get(doi_key, {})
            except Exception as exc:
                LOGGER.error("Failed to query Unpaywall for DOI %s: %s", doi, exc)
                yield {
                    **entry.raw,
                    "download_error": f"unpaywall_request_failed: {exc}",
                }
                continue

            if record.get("error") == "not_found":
                LOGGER.info("DOI %s not found in Unpaywall.", doi)
                yield {
                    **entry.raw,
                    "download_error": "unpaywall_not_found",
                }
                continue

            location = pick_best_location(record)
            download_url = determine_download_url(location or {})
            if not download_url:
                LOGGER.info("No OA location available for DOI %s", doi)
                yield {
                    **entry.raw,
                    "unpaywall_record": record,
                    "download_error": "no_open_access_url",
                }
                continue

            study_id = entry.study_id() or f"row-{entry.row_index() or 'unknown'}"
            study_slug = slugify(study_id)
            doi_slug = slugify(doi, allow_period=True)
            target_dir = download_root / study_slug
            target_path = target_dir / f"{doi_slug}.pdf"

            download_error: Optional[str] = None
            if doi_key in cache and cache[doi_key].exists():
                target_path = cache[doi_key]
            else:
                try:
                    target_path = download_file(session, download_url, target_path, timeout=timeout)
                    cache[doi_key] = target_path
                    LOGGER.info("Downloaded DOI %s to %s", doi, target_path)
                except Exception as exc:
                    download_error = f"download_failed: {exc}"
                    LOGGER.error("Failed to download %s: %s", download_url, exc)

            normalized_request = normalize_chat_request(chat_request)

            if download_error or not target_path.exists():
                yield {
                    **{k: v for k, v in entry.raw.items() if k != "chat_request"},
                    "chat_request": normalized_request,
                    "unpaywall_record": record,
                    "download_error": download_error or "file_missing",
                }
                continue

            append_file_context(normalized_request, target_path, download_url)

            context_files = [
                {
                    "path": str(target_path),
                    "doi": doi,
                    "study_id": study_id,
                    "download_url": download_url,
                    "host_type": (location or {}).get("host_type"),
                    "license": (location or {}).get("license"),
                }
            ]

            yield {
                **{k: v for k, v in entry.raw.items() if k != "chat_request"},
                "chat_request": normalized_request,
                "unpaywall_record": record,
                "context_files": context_files,
                "download_error": None,
            }
    finally:
        session.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.download_root.mkdir(parents=True, exist_ok=True)

    entries = list(iter_entries(args.input))
    LOGGER.info("Loaded %s records from %s", len(entries), args.input)

    with args.output.open("w", encoding="utf-8") as handle:
        for payload in process_entries(
            entries,
            email=args.email,
            download_root=args.download_root,
            timeout=args.timeout,
        ):
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    LOGGER.info("Finished. Augmented payloads written to %s", args.output)


if __name__ == "__main__":
    main()
