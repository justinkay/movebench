#!/usr/bin/env python3
"""Locate related publication links on Movebank repository dataset pages."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.repository_scraper import _build_session  # noqa: E402

logger = logging.getLogger(__name__)
TARGET_SECTION_TITLE = "dois of related publications"

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant who analyses scientific publications "
    "to extract key details relevant to animal movement datasets."
)

DEFAULT_USER_PROMPT = """\
Visit this url and read the paper in its entirety {publication_url}  The goal of reading the text of the publication (if accessible) is to find which environmental covariates, if any,  were used to analyze the data points in the study. This needs to be 100% precise:  you must be absolutely certain that that environmental feature was used in the study in order to report it to me.   Return the following: one row per covariate used, with comma separated values: covariate_name,spatial_resolution,temporal_resolution. DO NOT make any assumptions or hallucinate variables that were not used. Each variable must be backed  up by explicit text in the manuscript saying it was used.  If you cannot access the publication, say so. If you read the paper exhaustively and find no covariates were used, say so.
"""


@dataclass
class PublicationLink:
    """Representation of a related publication anchor discovered on a page."""

    url: str
    text: str
    raw: Dict[str, str] | None = None


def iter_section_candidates(soup: BeautifulSoup) -> Iterable[Tag]:
    """Yield elements whose text matches (or contains) the target section title."""
    heading_tags = ("h1", "h2", "h3", "h4", "h5", "h6", "span", "dt", "strong", "div")
    for tag_name in heading_tags:
        for tag in soup.find_all(tag_name):
            text = tag.get_text(" ", strip=True)
            if not text:
                continue
            normalized = text.lower()
            if normalized == TARGET_SECTION_TITLE or TARGET_SECTION_TITLE in normalized:
                yield tag


def extract_links_from_section(section: Tag) -> List[PublicationLink]:
    """Collect anchors belonging to the identified section."""
    links: List[PublicationLink] = []

    def gather_from(node: Tag) -> None:
        if not isinstance(node, Tag):
            return
        for anchor in node.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href:
                continue
            text = anchor.get_text(" ", strip=True) or href
            links.append(PublicationLink(url=href, text=text))

    # Anchors might sit directly under the heading.
    gather_from(section)
    if links:
        return links

    # Otherwise consume siblings until another heading-like element appears.
    for sibling in section.next_siblings:
        if isinstance(sibling, NavigableString):
            continue
        if not isinstance(sibling, Tag):
            continue
        if sibling.name and sibling.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6", "dt"}:
            break
        gather_from(sibling)
        if links:
            return links

    # Fall back to the parent container.
    parent = section.parent
    if isinstance(parent, Tag):
        gather_from(parent)

    # Deduplicate while preserving order.
    unique: dict[str, PublicationLink] = {}
    for link in links:
        if link.url not in unique:
            unique[link.url] = link
    return list(unique.values())


def find_related_publication_links(soup: BeautifulSoup) -> List[PublicationLink]:
    """Locate the related publication section and return the anchors it contains."""
    for section in iter_section_candidates(soup):
        links = extract_links_from_section(section)
        if links:
            return links
    return []


def fetch_publication_urls(
    session: requests.Session,
    dataset_url: str,
    *,
    timeout: float,
) -> List[PublicationLink]:
    """Fetch the dataset page and return absolute URLs for related publications."""
    response = session.get(dataset_url, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = find_related_publication_links(soup)
    base_url = response.url or dataset_url
    absolute_links: List[PublicationLink] = []
    for link in links:
        absolute_url = urljoin(base_url, link.url)
        absolute_links.append(PublicationLink(url=absolute_url, text=link.text))
    return absolute_links


def load_prompt_template(path: Path | None) -> str:
    if path is None:
        return DEFAULT_USER_PROMPT
    return path.read_text(encoding="utf-8")


def build_chat_request(
    *,
    model: str,
    system_prompt: str,
    user_prompt_template: str,
    row_index: int,
    dataset_url: str,
    study_id: str,
    study_title: str,
    publication: PublicationLink,
    temperature: float,
) -> Dict[str, object]:
    try:
        user_prompt = user_prompt_template.format(
            dataset_url=dataset_url,
            study_id=study_id,
            study_title=study_title,
            publication_url=publication.url,
            publication_label=publication.text,
        )
    except KeyError as exc:
        missing = exc.args[0]
        raise ValueError(
            f"Prompt template is missing a placeholder for {{{missing}}}. "
            "Update the template or supply the correct placeholder."
        ) from exc

    return {
        "row_index": row_index,
        "study_id": study_id,
        "dataset_url": dataset_url,
        "study_title": study_title,
        "publication_url": publication.url,
        "publication_label": publication.text,
        "chat_request": {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print the links from the 'DOIs of related Publications' section for each "
            "dataset listed in the provided CSV."
        )
    )
    parser.add_argument(
        "--links-csv",
        default="mbr_studies_metadata.csv",
        help="CSV file containing a paper_url column (default: %(default)s).",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=90.0,
        help="HTTP timeout in seconds for dataset requests.",
    )
    parser.add_argument(
        "--dataset-delay",
        type=float,
        default=3.0,
        help="Delay in seconds after processing each dataset page (default: %(default)s).",
    )
    parser.add_argument(
        "--output-jsonl",
        default="papers/chatgpt_requests.jsonl",
        help="Destination JSONL file with chat completion payloads (default: %(default)s).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="ChatGPT model name to include in each payload (default: %(default)s).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for each request (default text tunes the assistant role).",
    )
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Path to a text file containing the user prompt template. "
        "Available placeholders: {dataset_url}, {study_id}, {study_title}, "
        "{publication_url}, {publication_label}.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature value to include with each chat request (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    links_csv = Path(args.links_csv)
    if not links_csv.exists():
        parser.error(f"CSV file not found: {links_csv}")

    session = _build_session()

    prompt_path = Path(args.prompt_template) if args.prompt_template else None
    if prompt_path and not prompt_path.exists():
        parser.error(f"Prompt template file not found: {prompt_path}")
    user_prompt_template = load_prompt_template(prompt_path)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[str] = []

    with open(links_csv, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "paper_url" not in reader.fieldnames:
            parser.error("Input CSV must include a 'paper_url' column.")

        for idx, row in enumerate(reader, start=1):
            dataset_url = (row.get("paper_url") or "").strip()
            if not dataset_url:
                logger.debug("Row %s missing paper_url; skipping.", idx)
                continue

            study_id = (row.get("study_id") or "").strip()
            label = study_id or f"row-{idx}"

            try:
                links = fetch_publication_urls(
                    session,
                    dataset_url,
                    timeout=args.request_timeout,
                )
            except requests.RequestException as exc:
                logger.warning("Failed to load dataset page %s: %s", dataset_url, exc)
                continue

            if not links:
                logger.info("No related publication links found for %s", dataset_url)
                continue

            for link in links:
                print(f"{label}\t{dataset_url}\t{link.url}\t{link.text}")
                request_payload = build_chat_request(
                    model=args.model,
                    system_prompt=args.system_prompt,
                    user_prompt_template=user_prompt_template,
                    row_index=idx,
                    dataset_url=dataset_url,
                    study_id=study_id,
                    study_title=(row.get("title") or row.get("study_name") or "").strip(),
                    publication=link,
                    temperature=args.temperature,
                )
                records.append(json.dumps(request_payload))
                print("Payload for", link)
                print(request_payload)

                output_path.write_text("\n".join(records) + "\n", encoding="utf-8")

            time.sleep(args.dataset_delay)

    if records:
        output_path.write_text("\n".join(records) + "\n", encoding="utf-8")
        logger.info("Wrote %s requests to %s", len(records), output_path)
    else:
        logger.info("No chat requests generated.")


if __name__ == "__main__":
    main()
