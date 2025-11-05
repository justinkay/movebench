"""
Robust downloader for Movebank Data Repository studies scraped from search pages.

This module reads the CSV produced by :mod:`movebench.repository_scraper`,
visits each study page, finds bitstream download links, saves the associated
files, and optionally extracts compressed archives.  Directories are named
according to the LastName_StudyID convention introduced in the historical
``rename_directories_fixed.py`` script so that the downstream metadata pipeline
can operate without an additional rename pass.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Set
from urllib.parse import urljoin, urlparse
import zipfile

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DOI_PATTERNS = (
    r"https?://(?:www\.)?doi\.org/[\w./-]+",
    r"https?://dx\.doi\.org/[\w./-]+",
    r"doi:\s*([\w./-]+)",
    r"DOI:\s*([\w./-]+)",
)


def _build_retry() -> Retry:
    return Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        allowed_methods=("GET", "HEAD"),
        status_forcelist=(429, 500, 502, 503, 504),
        respect_retry_after_header=True,
        raise_on_status=False,
    )


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_build_retry())
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


def sanitize_filename(filename: str) -> str:
    """Return a filename safe for most filesystems."""
    clean = re.sub(r'[<>:"/\\|?*]', "_", filename)
    clean = re.sub(r"\s+", " ", clean).strip()
    if len(clean) > 200:
        clean = clean[:200].rstrip()
    return clean or "unnamed"


def _normalize_doi(content: str) -> Optional[str]:
    if not content:
        return None
    doi = content.strip().strip(' ,.;')
    if not doi:
        return None
    if doi.lower().startswith("doi:"):
        doi = doi.split(":", 1)[1].strip()
    if doi.startswith("http"):
        if "doi.org" in doi:
            return doi.rstrip(' .,;')
        return None
    return f"https://doi.org/{doi.rstrip(' .,;')}"


def _extract_doi_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in DOI_PATTERNS:
        match = re.search(pattern, text)
        if not match:
            continue
        group = match.group(0 if "http" in pattern else 1)
        doi_url = _normalize_doi(group)
        if doi_url:
            return doi_url
    return None


def extract_first_author_lastname(author_string: str) -> str:
    """Return the first author's surname from a string."""
    if not author_string:
        return "Unknown"
    author_string = author_string.strip()
    if not author_string or author_string.lower() == "unknown":
        return "Unknown"

    first_author = author_string.split(" and ")[0].strip()
    if "," in first_author:
        last_name = first_author.split(",")[0].strip()
    else:
        parts = first_author.split()
        last_name = parts[-1].strip() if parts else first_author
    last_name = re.sub(r"[^A-Za-z]", "", last_name)
    return last_name or "Unknown"


def extract_study_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    segment = Path(parsed.path).name or "unknown"
    return segment


def compose_directory_name(author_list: str, study_id: str, fallback_title: str) -> str:
    """Create LastName_ShortID directory identifier."""
    last_name = extract_first_author_lastname(author_list)
    short_id = study_id[:8] if study_id else "unknown"
    if not short_id or short_id == "entities":
        short_id = re.sub(r"[^0-9a-fA-F]", "", fallback_title)[:8] or "unknown"
    return f"{last_name}_{short_id}"


def ensure_unique_path(base_path: Path) -> Path:
    """Append a numerical suffix until the path is unique."""
    candidate = base_path
    counter = 1
    while candidate.exists():
        candidate = base_path.parent / f"{base_path.name}_{counter}"
        counter += 1
    return candidate


@dataclass
class DownloadLink:
    url: str
    filename: str
    size_hint: Optional[int] = None


@dataclass
class StudyRecord:
    study_id: str
    url: str
    raw_title: str
    csv_row: Dict[str, str] = field(default_factory=dict)


def _iter_study_records(csv_path: Path, study_ids: Optional[Set[str]] = None) -> Iterator[StudyRecord]:
    study_ids = {sid for sid in (study_ids or set()) if sid}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = row.get("url") or row.get("study_url")
            if not url:
                logger.warning("Skipping row without study URL: %s", row)
                continue
            study_id = row.get("study_id") or extract_study_id_from_url(url)
            if study_ids and study_id not in study_ids:
                continue
            raw_title = row.get("title") or row.get("text") or ""
            yield StudyRecord(study_id=study_id, url=url, raw_title=raw_title, csv_row=row)


class MovebankStudyDownloader:
    def __init__(
        self,
        base_dir: Path = Path("mbr_studies"),
        *,
        session: Optional[requests.Session] = None,
        request_timeout: float = 60.0,
        delay_between_requests: float = 1.0,
        extraction_enabled: bool = True,
        skip_existing: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or _build_session()
        self.request_timeout = request_timeout
        self.delay_between_requests = delay_between_requests
        self.extraction_enabled = extraction_enabled
        self.skip_existing = skip_existing

    # ---- network helpers -------------------------------------------------
    def _get(self, url: str, **kwargs) -> requests.Response:
        response = self.session.get(url, timeout=self.request_timeout, **kwargs)
        response.raise_for_status()
        return response

    def _get_soup(self, url: str) -> BeautifulSoup:
        response = self._get(url)
        return BeautifulSoup(response.text, "html.parser")

    # ---- parsing ---------------------------------------------------------
    @staticmethod
    def _extract_metadata(soup: BeautifulSoup) -> Dict[str, str]:
        title = ""
        creators: List[str] = []
        citation_text = ""
        doi_url = None

        title_tag = soup.find("h1")
        if title_tag:
            title = title_tag.get_text(strip=True)
        if title.lower().startswith("data from:"):
            title = title[len("data from:") :].strip()

        # DSpace often renders authors in meta tags
        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            if name in {"citation_author", "dc.creator"}:
                content = meta.get("content")
                if content:
                    creators.append(content.strip())
            if name in {"dcterms.bibliographiccitation", "citation_reference"}:
                content = meta.get("content")
                if content:
                    citation_text = content.strip()
            if name in {"citation_doi", "dc.identifier"}:
                content = meta.get("content")
                candidate = _normalize_doi(content) if content else None
                if candidate:
                    doi_url = candidate

        if not doi_url and citation_text:
            doi_url = _extract_doi_from_text(citation_text)

        if not doi_url:
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if "doi.org" in href:
                    candidate = _normalize_doi(href)
                    if candidate:
                        doi_url = candidate
                        break

        # Fallback: authors listed in page content
        if not creators:
            author_spans = soup.find_all("span", class_=re.compile("author", re.I))
            for span in author_spans:
                text = span.get_text(strip=True)
                if text:
                    creators.append(text)

        author_list = " and ".join(dict.fromkeys(creators))  # dedupe maintaining order
        return {
            "title": title,
            "author_list": author_list,
            "citation": citation_text,
            "doi_url": doi_url,
        }

    @staticmethod
    def _find_download_links(soup: BeautifulSoup, base_url: str) -> List[DownloadLink]:
        links: List[DownloadLink] = []
        seen_urls: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if "/bitstreams/" not in href or "license" in href.lower():
                continue
            full_url = urljoin(base_url + "/", href)
            if full_url in seen_urls:
                continue

            text = anchor.get_text(strip=True)
            filename = text or anchor.get("title") or ""
            if not filename:
                # try parent element text
                parent = anchor.parent
                if parent:
                    filename = parent.get_text(strip=True)

            if not filename:
                bits = urlparse(full_url).path.rstrip("/").split("/")
                filename = bits[-2] if len(bits) >= 2 else "downloaded_file"

            filename = sanitize_filename(filename)

            # Attempt to infer extension
            if not re.search(r"\.[A-Za-z0-9]{2,6}$", filename):
                if "csv" in filename.lower():
                    filename += ".csv"
                elif "readme" in filename.lower():
                    filename += ".txt"
                elif "pdf" in filename.lower():
                    filename += ".pdf"

            size_hint = None
            size_span = anchor.find_next("span", class_=re.compile("file-size", re.I))
            if size_span:
                # e.g. "(12.3 MB)"
                numbers = re.findall(r"[\d\.]+", size_span.get_text())
                if numbers:
                    try:
                        size_hint = float(numbers[0])
                    except ValueError:
                        size_hint = None

            links.append(DownloadLink(url=full_url, filename=filename, size_hint=size_hint))
            seen_urls.add(full_url)
        return links

    # ---- file handling ---------------------------------------------------
    def _attempt_download(self, url: str, destination: Path) -> None:
        try:
            self._download_to_path(url, destination)
            return
        except RuntimeError as exc:
            fallback_url = self._bitstream_fallback_url(url)
            if not fallback_url:
                raise
            logger.info("Retrying download via API endpoint: %s", fallback_url)
            self._download_to_path(fallback_url, destination)

    def _download_to_path(self, url: str, destination: Path) -> None:
        temp_path = destination.with_suffix(destination.suffix + ".part")
        response = self._get(url, stream=True)
        bytes_downloaded = 0
        first_chunk_checked = False
        with open(temp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                if not first_chunk_checked:
                    first_chunk_checked = True
                    if self._chunk_looks_like_html(chunk, response.headers):
                        temp_path.unlink(missing_ok=True)
                        raise RuntimeError("Received HTML instead of dataset content")
                handle.write(chunk)
                bytes_downloaded += len(chunk)
        temp_path.replace(destination)
        logger.info("Downloaded %s (%d bytes)", destination.name, bytes_downloaded)

    def _extract_archive(self, archive_path: Path, dest_dir: Path) -> None:
        if not self.extraction_enabled:
            return
        suffix = archive_path.suffix.lower()
        if suffix != ".zip":
            return

        extract_dir = dest_dir / f"{archive_path.stem}_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir()
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extracted %s to %s", archive_path.name, extract_dir)

    @staticmethod
    def _looks_like_html(path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                prefix = handle.read(1024).lstrip().lower()
        except OSError:
            return False
        return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html")

    @staticmethod
    def _chunk_looks_like_html(chunk: bytes, headers: Dict[str, str]) -> bool:
        if not chunk:
            return False
        content_type = headers.get("Content-Type") or headers.get("content-type")
        if content_type and "text/html" in content_type.lower():
            return True
        stripped = chunk.lstrip().lower()
        return stripped.startswith(b"<!doctype html") or stripped.startswith(b"<html")

    @staticmethod
    def _bitstream_fallback_url(url: str) -> Optional[str]:
        parsed = urlparse(url)
        if "bitstreams" not in parsed.path:
            return None
        parts = parsed.path.strip("/").split("/")
        try:
            idx = parts.index("bitstreams")
            bitstream_id = parts[idx + 1]
        except (ValueError, IndexError):
            return None
        return f"https://datarepository.movebank.org/server/api/core/bitstreams/{bitstream_id}/content"

    # ---- public API ------------------------------------------------------
    def download_study(self, record: StudyRecord) -> Dict[str, object]:
        logger.info("Processing %s", record.url)
        soup = self._get_soup(record.url)
        metadata = self._extract_metadata(soup)
        title = metadata.get("title") or record.raw_title or record.study_id
        author_list = metadata.get("author_list") or record.csv_row.get("author_list", "")

        directory_name = compose_directory_name(author_list, record.study_id, title)
        study_dir = self.base_dir / sanitize_filename(directory_name)
        if study_dir.exists():
            if self.skip_existing:
                logger.info("Skipping %s (already exists)", study_dir.name)
                return {
                    "status": "skipped",
                    "study_id": record.study_id,
                    "study_dir": str(study_dir),
                    "record": record.csv_row,
                }
        study_dir.mkdir(parents=True, exist_ok=True)

        download_links = self._find_download_links(soup, base_url="https://datarepository.movebank.org")
        if not download_links:
            raise RuntimeError(f"No downloadable files discovered for {record.url}")

        manifest: Dict[str, Dict[str, object]] = {}
        for link in download_links:
            filepath = study_dir / link.filename
            try:
                needs_download = True
                if filepath.exists():
                    if self._looks_like_html(filepath):
                        logger.warning("Existing file looks like HTML, re-downloading: %s", filepath.name)
                        filepath.unlink()
                    elif self.skip_existing:
                        logger.info("File already exists, skipping: %s", filepath.name)
                        needs_download = False
                if needs_download:
                    self._attempt_download(link.url, filepath)
                    time.sleep(self.delay_between_requests)
                self._extract_archive(filepath, study_dir)
                manifest[link.filename] = {
                    "url": link.url,
                    "size_hint": link.size_hint,
                }
            except Exception as exc:
                logger.error("Failed to download %s: %s", link.url, exc)
                raise

        if metadata.get("citation"):
            citation_file = study_dir / "citation.txt"
            citation_file.write_text(metadata["citation"], encoding="utf-8")

        metadata_path = study_dir / "study_metadata.json"
        metadata_payload = {
            "study_id": record.study_id,
            "source_url": record.url,
            "title": title,
            "author_list": author_list,
            "raw_row": record.csv_row,
            "files": manifest,
        }
        if metadata.get("citation"):
            metadata_payload["citation"] = metadata.get("citation")
        if metadata.get("doi_url"):
            metadata_payload["doi_url"] = metadata.get("doi_url")
        metadata_payload["paper_url"] = metadata.get("doi_url") or record.url
        metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")

        return {
            "status": "downloaded",
            "study_id": record.study_id,
            "study_dir": str(study_dir),
            "record": record.csv_row,
            "files": list(manifest.keys()),
        }

    def download_all(self, csv_path: Path, study_ids: Optional[Set[str]] = None) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for record in _iter_study_records(csv_path, study_ids):
            try:
                result = self.download_study(record)
                results.append(result)
            except Exception as exc:
                logger.error("Study %s failed: %s", record.url, exc)
                results.append(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "study_id": record.study_id,
                        "url": record.url,
                    }
                )
        return results
