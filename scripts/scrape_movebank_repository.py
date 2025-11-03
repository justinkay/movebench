#!/usr/bin/env python3
"""CLI for scraping Movebank Data Repository study links."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from movebench.repository_scraper import MovebankRepositoryScraper, scrape_to_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape Movebank repository for GPS dataset links."
    )
    parser.add_argument(
        "--output",
        default="movebank_links.csv",
        help="Destination CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=50,
        help="Maximum number of paginated search pages to crawl.",
    )
    parser.add_argument(
        "--page-delay",
        type=float,
        default=1.0,
        help="Delay between page requests in seconds.",
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scraper = MovebankRepositoryScraper(
        page_delay=args.page_delay,
        max_pages=args.max_pages,
    )

    links, dest = scrape_to_csv(str(output_path), scraper=scraper)
    print(f"Scraped {len(links)} studies -> {dest}")


if __name__ == "__main__":
    main()
