#!/usr/bin/env python3
"""Update stored ChatGPT request payloads to use the GPT‑5 model.

The JSONL produced by ``create_gpt_prompts_for_papers.py`` still targets
``gpt-4o`` and includes a ``temperature`` parameter. GPT-5 does not accept that
parameter, so this script rewrites each request accordingly.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator

LOGGER = logging.getLogger("patch_chatgpt_requests")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite chat_request entries to target GPT-5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("papers/chatgpt_requests.jsonl"),
        help="Source JSONL file to patch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting the input file.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Model name to set on each chat_request.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args()


def iter_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield line


def update_payload(raw_line: str, model: str) -> str:
    payload = json.loads(raw_line)
    chat_request = payload.get("chat_request")
    if isinstance(chat_request, dict):
        chat_request["model"] = model
        chat_request.pop("temperature", None)
    else:
        LOGGER.warning("Skipping record without chat_request: %s", payload.get("row_index"))
    return json.dumps(payload, ensure_ascii=False)


def write_output(lines: Iterator[str], destination: Path, model: str) -> None:
    with destination.open("w", encoding="utf-8") as handle:
        for line in lines:
            patched = update_payload(line, model)
            handle.write(patched + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    output_path = args.output
    if output_path is None:
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=args.input.stem + "_", suffix=".jsonl", dir=str(args.input.parent)
        )
        os.close(temp_fd)
        Path(temp_path).unlink()  # Will rewrite via Path API below.
        output_path = Path(temp_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write_output(iter_lines(args.input), output_path, args.model)
        if args.output is None:
            output_path.replace(args.input)
            LOGGER.info("Patched %s in place (model=%s).", args.input, args.model)
        else:
            LOGGER.info("Wrote patched records to %s (model=%s).", args.output, args.model)
    finally:
        if args.output is None and output_path.exists() and output_path != args.input:
            # Safety cleanup in case replace() failed.
            try:
                output_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
