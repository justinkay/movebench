#!/usr/bin/env python3
"""Submit stored GPT requests to the OpenAI Responses API and save outputs.

This utility consumes the JSONL file produced by
``scripts/create_gpt_prompts_for_papers.py`` and streams each request to the
Responses endpoint. Results (or errors) are appended to an output JSONL
file so the process can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - import error message
    raise SystemExit(
        "The `openai` package is required. Install it with `pip install openai`."
    ) from exc

try:  # pragma: no cover - exception imports differ between SDK versions
    from openai import APIError, APIStatusError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover
    APIError = APIStatusError = APITimeoutError = RateLimitError = Exception  # type: ignore

LOGGER = logging.getLogger("run_chatgpt_requests")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "papers" / "chatgpt_requests.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "papers" / "chatgpt_responses.jsonl"
DEFAULT_FILE_CACHE = REPO_ROOT / "papers" / "uploaded_files.json"


@dataclass(frozen=True)
class RequestRecord:
    """Container for a single request payload and metadata."""

    raw: Dict[str, Any]

    @property
    def row_index(self) -> Optional[int]:
        value = self.raw.get("row_index")
        return int(value) if value is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit stored GPT requests to the Responses API and capture outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the JSONL file containing chat_request payloads.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSONL file for responses.",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="Optional custom API base URL.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Process at most this many new requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Number of retries for rate limits or transient errors.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to pause between successful requests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip entries that already exist in the output file.",
    )
    parser.add_argument(
        "--file-cache",
        type=Path,
        default=DEFAULT_FILE_CACHE,
        help="JSON file storing cached uploads (maps file paths to file_ids).",
    )
    parser.add_argument(
        "--file-purpose",
        default="assistants",
        help="Purpose to assign when uploading files to OpenAI (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_requests(path: Path) -> Iterator[RequestRecord]:
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
            yield RequestRecord(raw=payload)


def load_completed_indices(path: Path) -> Set[int]:
    completed: Set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_index = payload.get("row_index")
            if isinstance(row_index, int):
                completed.add(row_index)
    return completed


def load_file_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            cache: Dict[str, str] = {}
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, str):
                    cache[key] = value
            return cache
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to read file cache %s: %s", path, exc)
    return {}


def save_file_cache(path: Path, cache: Dict[str, str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to write file cache %s: %s", path, exc)


def serialize_response(response: Any) -> Any:
    """Convert SDK response objects to JSON-serializable structures."""
    if response is None:
        return None

    for attr in ("model_dump", "to_dict", "dict"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                continue
    for attr in ("model_dump_json", "json"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return json.loads(method())
            except (TypeError, json.JSONDecodeError):
                continue

    try:
        json.dumps(response)
        return response
    except TypeError:
        return str(response)


def extract_text_output(response: Any) -> Optional[str]:
    """Best-effort extraction of the textual output from a Responses API result."""
    if response is None:
        return None

    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value:
            return value

    output = getattr(response, "output", None)
    if not output and isinstance(response, dict):
        output = response.get("output")

    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            content = None
            if isinstance(item, dict):
                content = item.get("content")
            else:
                content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    text = None
                    if isinstance(block, dict):
                        text = block.get("text")
                    else:
                        text = getattr(block, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            elif isinstance(content, str):
                texts.append(content)
        if texts:
            return "\n".join(texts).strip() or None

    return None


def normalize_request_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure the request payload matches the Responses API schema."""
    normalized = dict(payload)
    raw_input = normalized.get("input")
    if isinstance(raw_input, list):
        normalized_input = []
        for entry in raw_input:
            if not isinstance(entry, dict):
                normalized_input.append(entry)
                continue
            entry_copy = dict(entry)
            content = entry_copy.get("content")
            if isinstance(content, list):
                normalized_content = []
                for block in content:
                    if isinstance(block, dict):
                        block_copy = dict(block)
                        if block_copy.get("type") == "text":
                            block_copy["type"] = "input_text"
                        normalized_content.append(block_copy)
                    else:
                        normalized_content.append(block)
                entry_copy["content"] = normalized_content
            normalized_input.append(entry_copy)
        normalized["input"] = normalized_input
    return normalized


def resolve_file_path(file_path: str) -> str:
    return str(Path(file_path).expanduser().resolve())


def upload_file(client: OpenAI, file_path: str, purpose: str) -> str:
    LOGGER.info("Uploading file for context: %s", file_path)
    with open(file_path, "rb") as handle:
        uploaded = client.files.create(file=handle, purpose=purpose)
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError(f"Upload returned unexpected response for {file_path}: {uploaded}")
    return file_id


def replace_file_paths_with_ids(
    client: OpenAI,
    request: Dict[str, Any],
    *,
    file_cache: Dict[str, str],
    file_purpose: str,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    working = json.loads(json.dumps(request))
    uploads: List[Dict[str, str]] = []
    raw_input = working.get("input")
    if isinstance(raw_input, list):
        for entry in raw_input:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "input_file":
                    continue
                if block.get("file_id"):
                    block.pop("file_path", None)
                    block.pop("description", None)
                    continue
                file_path = block.get("file_path")
                if not isinstance(file_path, str) or not file_path.strip():
                    raise ValueError("input_file block requires a file_path or file_id.")
                resolved = resolve_file_path(file_path)
                if not Path(resolved).exists():
                    raise FileNotFoundError(f"Context file not found: {resolved}")
                file_id = file_cache.get(resolved)
                if not file_id:
                    file_id = upload_file(client, resolved, file_purpose)
                    file_cache[resolved] = file_id
                    uploads.append({"path": resolved, "file_id": file_id})
                block["file_id"] = file_id
                block.pop("file_path", None)
                block.pop("description", None)
    return working, uploads


def prepare_request_for_sending(
    client: OpenAI,
    request: Dict[str, Any],
    *,
    file_cache: Dict[str, str],
    file_purpose: str,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    working_with_ids, uploads = replace_file_paths_with_ids(
        client,
        request,
        file_cache=file_cache,
        file_purpose=file_purpose,
    )
    normalized = normalize_request_payload(working_with_ids)
    return normalized, uploads


def send_request(client: OpenAI, request: Dict[str, Any], max_retries: int, pause_seconds: float) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            normalized_request = normalize_request_payload(request)
            response = client.responses.create(**normalized_request)
            return {
                "response": serialize_response(response),
                "response_text": extract_text_output(response),
                "error": None,
                "attempts": attempt,
            }
        except (RateLimitError, APITimeoutError) as exc:
            if attempt > max_retries:
                LOGGER.error("Exceeded retries for retriable request: %s", exc)
                return {"response": None, "response_text": None, "error": str(exc), "attempts": attempt}
            wait_time = pause_seconds or min(2 ** attempt, 30)
            LOGGER.warning(
                "Rate limit or timeout encountered (attempt %s/%s). Sleeping %.1fs.",
                attempt,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 429:
                if attempt > max_retries:
                    LOGGER.error("Exceeded retries for rate-limited request: %s", exc)
                    return {"response": None, "response_text": None, "error": str(exc), "attempts": attempt}
                wait_time = pause_seconds or min(2 ** attempt, 30)
                LOGGER.warning(
                    "Rate limit encountered (attempt %s/%s). Sleeping %.1fs.",
                    attempt,
                    max_retries,
                    wait_time,
                )
                time.sleep(wait_time)
                continue

            details = None
            if hasattr(exc, "response") and exc.response is not None:
                try:
                    details = exc.response.json()
                except Exception:
                    details = getattr(exc.response, "text", None)
            LOGGER.error(
                "OpenAI API returned status %s: %s",
                status,
                details or str(exc),
            )
            return {
                "response": None,
                "response_text": None,
                "error": str(details or exc),
                "attempts": attempt,
            }
        except APIError as exc:
            LOGGER.error("OpenAI API returned an error: %s", exc)
            return {"response": None, "response_text": None, "error": str(exc), "attempts": attempt}
        except Exception as exc:  # pragma: no cover - guard against SDK regressions
            LOGGER.exception("Unexpected error while calling the API: %s", exc)
            return {"response": None, "response_text": None, "error": str(exc), "attempts": attempt}


def ensure_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No OpenAI API key found. Set the OPENAI_API_KEY environment variable."
        )
    return api_key


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    client_kwargs: Dict[str, Any] = {"api_key": ensure_api_key()}
    if args.api_base:
        client_kwargs["base_url"] = args.api_base
    client = OpenAI(**client_kwargs)

    completed = load_completed_indices(args.output) if args.resume else set()
    if completed:
        LOGGER.info("Loaded %s completed row(s) from %s", len(completed), args.output)

    file_cache = load_file_cache(args.file_cache)
    if file_cache:
        LOGGER.info("Loaded %s cached file upload(s) from %s", len(file_cache), args.file_cache)

    processed = 0
    with args.output.open("a", encoding="utf-8") as sink:
        for record in iter_requests(args.input):
            row_index = record.row_index
            if row_index is not None and row_index in completed:
                LOGGER.debug("Skipping row_index %s (already processed).", row_index)
                continue
            chat_request = record.raw.get("chat_request")
            if not isinstance(chat_request, dict):
                LOGGER.error("Missing chat_request payload for row %s", row_index)
                continue

            try:
                prepared_request, upload_events = prepare_request_for_sending(
                    client,
                    chat_request,
                    file_cache=file_cache,
                    file_purpose=args.file_purpose,
                )
                if upload_events:
                    save_file_cache(args.file_cache, file_cache)
            except Exception as exc:
                LOGGER.error(
                    "Failed to prepare request for row %s: %s",
                    row_index if row_index is not None else "<unknown>",
                    exc,
                )
                failure_payload = {
                    **{k: v for k, v in record.raw.items() if k != "chat_request"},
                    "row_index": row_index,
                    "chat_request": chat_request,
                    "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "response": None,
                    "response_text": None,
                    "error": str(exc),
                    "attempts": 0,
                    "uploaded_files": [],
                }
                sink.write(json.dumps(failure_payload, ensure_ascii=False) + "\n")
                sink.flush()
                continue

            LOGGER.info(
                "Processing row %s (%s)",
                row_index if row_index is not None else "<unknown>",
                record.raw.get("publication_url", record.raw.get("study_id", "")),
            )
            result = send_request(client, prepared_request, args.max_retries, args.sleep)

            output_payload = {
                **{k: v for k, v in record.raw.items() if k != "chat_request"},
                "row_index": row_index,
                "chat_request": prepared_request,
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "uploaded_files": upload_events,
                **result,
            }
            sink.write(json.dumps(output_payload, ensure_ascii=False) + "\n")
            sink.flush()

            if row_index is not None:
                completed.add(row_index)

            processed += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.max_requests and processed >= args.max_requests:
                LOGGER.info("Reached max_requests=%s limit.", args.max_requests)
                break

    LOGGER.info("Finished sending %s request(s).", processed)


if __name__ == "__main__":
    main()
