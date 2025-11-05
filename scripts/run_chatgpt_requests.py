#!/usr/bin/env python3
"""Submit stored ChatGPT requests to the OpenAI API and save responses.

This utility consumes the JSONL file produced by
``scripts/create_gpt_prompts_for_papers.py`` and streams each request to the
Chat Completions endpoint. Results (or errors) are appended to an output JSONL
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
from typing import Any, Dict, Iterator, Optional, Set

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
        description="Submit stored ChatGPT requests and capture responses.",
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


def send_request(client: OpenAI, request: Dict[str, Any], max_retries: int, pause_seconds: float) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        try:
            completion = client.chat.completions.create(**request)
            return {
                "response": serialize_response(completion),
                "error": None,
                "attempts": attempt,
            }
        except (RateLimitError, APIStatusError, APITimeoutError) as exc:
            print(exc)
            if attempt > max_retries:
                LOGGER.error("Exceeded retries for rate-limited request: %s", exc)
                return {"response": None, "error": str(exc), "attempts": attempt}
            wait_time = pause_seconds or min(2 ** attempt, 30)
            LOGGER.warning(
                "Rate limit or timeout encountered (attempt %s/%s). Sleeping %.1fs.",
                attempt,
                max_retries,
                wait_time,
            )
            time.sleep(wait_time)
        except APIError as exc:
            LOGGER.error("OpenAI API returned an error: %s", exc)
            return {"response": None, "error": str(exc), "attempts": attempt}
        except Exception as exc:  # pragma: no cover - guard against SDK regressions
            LOGGER.exception("Unexpected error while calling the API: %s", exc)
            return {"response": None, "error": str(exc), "attempts": attempt}


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

            LOGGER.info(
                "Processing row %s (%s)",
                row_index if row_index is not None else "<unknown>",
                record.raw.get("publication_url", record.raw.get("study_id", "")),
            )
            result = send_request(client, chat_request, args.max_retries, args.sleep)

            output_payload = {
                **{k: v for k, v in record.raw.items() if k != "chat_request"},
                "row_index": row_index,
                "chat_request": chat_request,
                "received_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
