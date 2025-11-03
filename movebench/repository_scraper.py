"""
Utilities for scraping dataset links from the Movebank Data Repository.

The repository does not offer an API, so we fall back to scraping the paginated
search UI that lists dataset (datapackage) entities.  Each dataset appears on a
search page inside several anchor tags that contain a `/entities/datapackage/`
fragment in their href.  We collect the unique datapackage URLs, normalise a
title, and persist the results in a CSV that mirrors the historical
`movebank_links.csv` output while adding helpful fields (study_id, title).
"""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


DEFAULT_SEARCH_TEMPLATE = (
    "https://datarepository.movebank.org/search"
    "?f.has_content_in_original_bundle=true,equals"
    "&f.entityType=Datapackage,equals"
    "&spc.sf=dc.date.accessioned"
    "&spc.sd=DESC"
    "&spc.page={page}"
    "&f.sensorName=GPS,equals"
)


@dataclass
class StudyLink:
    """Simple container representing a repository dataset listing."""

    study_id: str
    url: str
    title: str
    page: int
    raw_text: str


def _default_retry() -> Retry:
    # Retry on common transient problems (timeouts, throttling, gateway errors).
    return Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET", "OPTIONS"),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_default_retry())
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
    )
    return session


class MovebankRepositoryScraper:
    """Scrape Movebank Data Repository search pages for dataset links."""

    def __init__(
        self,
        search_template: str = DEFAULT_SEARCH_TEMPLATE,
        *,
        base_url: str = "https://datarepository.movebank.org",
        session: Optional[requests.Session] = None,
        request_timeout: float = 30.0,
        page_delay: float = 1.0,
        max_pages: int = 50,
        consecutive_empty_limit: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.search_template = search_template
        self.session = session or _build_session()
        self.request_timeout = request_timeout
        self.page_delay = page_delay
        self.max_pages = max_pages
        self.consecutive_empty_limit = consecutive_empty_limit

    def _fetch_page(self, page_number: int) -> str:
        url = self.search_template.format(page=page_number)
        logger.debug("Fetching search page %s -> %s", page_number, url)
        response = self.session.get(url, timeout=self.request_timeout)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _normalise_title(text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.lower().startswith("data from:"):
            cleaned = cleaned[len("data from:") :].strip()
        return cleaned

    @staticmethod
    def _extract_study_id(url: str) -> str:
        path = urlparse(url).path.rstrip("/")
        return path.split("/")[-1]

    def _parse_links(self, html: str, page_number: int) -> List[StudyLink]:
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        links: List[StudyLink] = []

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if "/entities/datapackage/" not in href:
                continue

            full_url = urljoin(self.base_url + "/", href)
            if full_url in seen_urls:
                continue

            text = anchor.get_text(strip=True)
            title = self._normalise_title(text)
            study_id = self._extract_study_id(full_url)
            seen_urls.add(full_url)
            links.append(
                StudyLink(
                    study_id=study_id,
                    url=full_url,
                    title=title,
                    page=page_number,
                    raw_text=text,
                )
            )

        return links

    def scrape(self) -> List[StudyLink]:
        """Scrape search pages until exhaustion or max_pages is reached."""
        results: List[StudyLink] = []
        seen_ids: set[str] = set()
        empty_pages = 0

        for page_num in range(1, self.max_pages + 1):
            try:
                html = self._fetch_page(page_num)
            except requests.RequestException as exc:  # pragma: no cover - network issues
                logger.warning("Failed to fetch page %s: %s", page_num, exc)
                empty_pages += 1
                if empty_pages >= self.consecutive_empty_limit:
                    break
                continue

            page_links = self._parse_links(html, page_num)

            new_links = [
                link for link in page_links if link.study_id not in seen_ids
            ]
            logger.info(
                "Page %s: %s links (%s new, %s duplicate)",
                page_num,
                len(page_links),
                len(new_links),
                len(page_links) - len(new_links),
            )

            if not page_links:
                empty_pages += 1
            else:
                empty_pages = 0

            if not new_links and empty_pages >= self.consecutive_empty_limit:
                logger.info("Stopping scrape after %s consecutive empty pages", empty_pages)
                break

            for link in new_links:
                results.append(link)
                seen_ids.add(link.study_id)

            time.sleep(self.page_delay)

        return results

    @staticmethod
    def write_csv(
        links: Sequence[StudyLink],
        dest_path: str,
        *,
        include_header: bool = True,
    ) -> None:
        """Persist scraped links to CSV."""
        fieldnames = ["study_id", "url", "title", "raw_text", "page"]
        with open(dest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if include_header:
                writer.writeheader()
            for link in links:
                writer.writerow(
                    {
                        "study_id": link.study_id,
                        "url": link.url,
                        "title": link.title,
                        "raw_text": link.raw_text,
                        "page": link.page,
                    }
                )


def scrape_to_csv(
    dest_path: str,
    *,
    scraper: Optional[MovebankRepositoryScraper] = None,
) -> Tuple[List[StudyLink], str]:
    """Helper that runs the scraper and writes the CSV."""
    scraper = scraper or MovebankRepositoryScraper()
    links = scraper.scrape()
    scraper.write_csv(links, dest_path)
    return links, dest_path

