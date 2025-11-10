#!/usr/bin/env python3
"""Download publication content and rebuild Responses API payloads with local files.

This script:
1. Reads the JSONL payload produced by ``create_gpt_prompts_for_papers.py``.
2. Downloads each publication URL (one download per unique URL) into a local cache.
3. Generates a new JSONL file that augments every chat request with a reference to the
   downloaded file so it can later be uploaded to the OpenAI API as context.

The emitted requests keep the existing model/prompts and append an extra user message
containing both a reminder that the publication is attached and a placeholder
``input_file`` block whose ``file_path`` points to the cached file. Downstream code
should replace that placeholder with an uploaded file_id before invoking the API.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional
from urllib.parse import urlparse

import requests

LOGGER = logging.getLogger("download_publications")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "papers" / "chatgpt_requests.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "papers" / "chatgpt_requests_with_files.jsonl"
DEFAULT_DOWNLOAD_DIR = REPO_ROOT / "papers" / "publications"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf, text/html;q=0.9, */*;q=0.8",
}


@dataclass
class DownloadInfo:
    url: str
    path: Path
    content_type: str
    status_code: int
    from_cache: bool = False


@dataclass
class RequestRecord:
    raw: Dict[str, Any]

    @property
    def publication_url(self) -> Optional[str]:
        url = self.raw.get("publication_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        chat_request = self.raw.get("chat_request")
        if isinstance(chat_request, dict):
            meta = chat_request.get("metadata")
            if isinstance(meta, dict):
                meta_url = meta.get("publication_url")
                if isinstance(meta_url, str) and meta_url.strip():
                    return meta_url.strip()
        return None

    @property
    def row_index(self) -> Optional[int]:
        value = self.raw.get("row_index")
        return int(value) if isinstance(value, int) else None

    @property
    def study_id(self) -> Optional[str]:
        study_id = self.raw.get("study_id")
        if isinstance(study_id, str) and study_id.strip():
            return study_id.strip()
        chat_request = self.raw.get("chat_request")
        if isinstance(chat_request, dict):
            meta = chat_request.get("metadata")
            if isinstance(meta, dict):
                meta_study = meta.get("study_id")
                if isinstance(meta_study, str) and meta_study.strip():
                    return meta_study.strip()
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download publication URLs and add local file context to requests.",
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
        help="Destination JSONL file containing augmented requests.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory where downloaded publications will be stored.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="HTTP timeout per request (seconds).",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=30 * 1024 * 1024,
        help="Maximum size (bytes) to download for a single file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_records(path: Path) -> Iterator[RequestRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.error("Skipping invalid JSON on line %s: %s", line_number, exc)
                continue
            if not isinstance(payload, dict):
                LOGGER.error("Skipping non-object entry on line %s.", line_number)
                continue
            yield RequestRecord(raw=payload)


def slugify(value: str, fallback: str = "file") -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or fallback


def guess_extension(content_type: str, url: str) -> str:
    if not content_type:
        content_type = ""
    content_type = content_type.lower()
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    if "plain" in content_type:
        return ".txt"

    parsed = urlparse(url)
    path = parsed.path or ""
    if path.endswith(".pdf"):
        return ".pdf"
    if path.endswith(".html"):
        return ".html"
    if path.endswith(".txt"):
        return ".txt"
    suffix = Path(path).suffix
    if suffix:
        return suffix
    return ".bin"


def download_publication(
    session: requests.Session,
    url: str,
    download_dir: Path,
    *,
    timeout: float,
    max_size: int,
) -> DownloadInfo:
    response = session.get(
        url,
        headers=DEFAULT_HEADERS,
        allow_redirects=True,
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").split(";")[0].strip()
    extension = guess_extension(content_type, response.url or url)

    parsed = urlparse(url)
    hostname = parsed.hostname or "publication"

    slug_parts = [
        slugify(hostname, "publication"),
        slugify(Path(parsed.path or "document").stem or "doc"),
    ]
    filename = "-".join(filter(None, slug_parts)) + extension

    destination = download_dir / filename

    total = 0
    download_dir.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                total += len(chunk)
                if total > max_size:
                    response.close()
                    destination.unlink(missing_ok=True)
                    raise ValueError(
                        f"File exceeds max-size limit of {max_size} bytes (saw > {total})."
                    )
                handle.write(chunk)

    LOGGER.info("Downloaded %s -> %s (%s)", url, destination, content_type or "unknown type")
    return DownloadInfo(
        url=url,
        path=destination,
        content_type=content_type or "application/octet-stream",
        status_code=response.status_code,
    )


def augment_request_with_file(request: Dict[str, Any], file_info: DownloadInfo) -> Dict[str, Any]:
    """Append a placeholder file attachment block to the Responses API input."""
    if not isinstance(request, dict):
        raise TypeError("chat_request entry must be a JSON object.")

    request_copy = json.loads(json.dumps(request))  # Deep copy via JSON

    input_blocks = request_copy.setdefault("input", [])
    if not isinstance(input_blocks, list):
        raise ValueError("chat_request.input must be a list.")

    attachment_block = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": (
                    "The complete publication has been attached. "
                    "Use it to extract the requested environmental covariates."
                ),
            },
            {
                "type": "input_file",
                "file_path": str(file_info.path),
                "mime_type": file_info.content_type,
                "description": "Downloaded publication content",
            },
        ],
    }
    input_blocks.append(attachment_block)
    return request_copy


def process_records(
    records: Iterable[RequestRecord],
    *,
    download_dir: Path,
    timeout: float,
    max_size: int,
) -> Iterator[Dict[str, Any]]:
    session = requests.Session()
    download_cache: Dict[str, DownloadInfo] = {}
    try:
        for record in records:
            publication_url = record.publication_url
            if not publication_url:
                LOGGER.warning("Row %s missing publication_url; skipping download.", record.row_index)
                yield {
                    **record.raw,
                    "download_error": "missing_publication_url",
                }
                continue

            download_error = None
            download_info: Optional[DownloadInfo] = None

            if publication_url in download_cache:
                cached = download_cache[publication_url]
                download_info = DownloadInfo(
                    url=cached.url,
                    path=cached.path,
                    content_type=cached.content_type,
                    status_code=cached.status_code,
                    from_cache=True,
                )
                LOGGER.debug(
                    "Reusing cached download for %s -> %s",
                    publication_url,
                    cached.path,
                )
            else:
                try:
                    download_info = download_publication(
                        session,
                        publication_url,
                        download_dir,
                        timeout=timeout,
                        max_size=max_size,
                    )
                    download_cache[publication_url] = download_info
                except Exception as exc:
                    download_error = str(exc)
                    LOGGER.error("Failed to download %s: %s", publication_url, exc)

            base_payload = {k: v for k, v in record.raw.items() if k != "chat_request"}

            if download_info:
                try:
                    augmented_request = augment_request_with_file(
                        record.raw.get("chat_request", {}),
                        download_info,
                    )
                except Exception as exc:
                    download_error = f"failed_to_augment_request: {exc}"
                    augmented_request = record.raw.get("chat_request")
            else:
                augmented_request = record.raw.get("chat_request")

            output_entry = {
                **base_payload,
                "chat_request": augmented_request,
            }
            if download_info:
                output_entry["context_files"] = [
                    {
                        "path": str(download_info.path),
                        "content_type": download_info.content_type,
                        "status_code": download_info.status_code,
                        "from_cache": download_info.from_cache,
                    }
                ]
            else:
                output_entry["context_files"] = []

            if download_error:
                output_entry["download_error"] = download_error

            yield output_entry
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

    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading requests from %s", args.input)
    records = list(iter_records(args.input))
    LOGGER.info("Loaded %s records.", len(records))

    processed = list(
        process_records(
            records,
            download_dir=args.download_dir,
            timeout=args.timeout,
            max_size=args.max_size,
        )
    )

    if not processed:
        LOGGER.warning("No records to write. Exiting without creating output.")
        return

    LOGGER.info("Writing augmented requests to %s", args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        for entry in processed:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    LOGGER.info(
        "Done. %s entries written. Download cache stored under %s.",
        len(processed),
        args.download_dir,
    )


if __name__ == "__main__":
    main()
