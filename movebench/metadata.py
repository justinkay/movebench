"""
Metadata extraction utilities for Movebank repository studies.

The extractor reads the metadata emitted by :mod:`movebench.downloader`,
derives useful summary statistics from the downloaded CSV files, and writes a
summary table compatible with the historical `mbr_studies_metadata.csv`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
import zipfile
from functools import lru_cache

logger = logging.getLogger(__name__)

DEFAULT_ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

SPECIES_COLUMNS = [
    "individual-taxon-canonical-name",
    "taxon_canonical_name",
    "species",
]

INDIVIDUAL_COLUMNS = [
    "individual-local-identifier",
    "individual_local_identifier",
    "individual_id",
]

TIMESTAMP_COLUMNS = ["timestamp", "timestamp_start", "timestamp_end", "time"]

LATITUDE_COLUMNS = ["location-lat", "location_lat", "latitude", "lat"]
LONGITUDE_COLUMNS = ["location-long", "location_long", "longitude", "lon", "long"]

EXCLUDE_PATTERNS = (
    "reference",
    "readme",
    "metadata",
    "license",
    "licence",
    "citation",
)

COMMON_NAME_MAPPING = {
    "Tyto furcata": "American Barn Owl",
    "Panthera leo": "Lion",
    "Elephas maximus": "Asian Elephant",
    "Loxodonta africana": "African Elephant",
    "Ursus americanus": "American Black Bear",
    "Canis lupus": "Gray Wolf",
    "Vulpes vulpes": "Red Fox",
    "Aquila chrysaetos": "Golden Eagle",
    "Falco peregrinus": "Peregrine Falcon",
    "Corvus corax": "Common Raven",
    "Grus grus": "Common Crane",
    "Chelonia mydas": "Green Sea Turtle",
}

try:
    import taxoniq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    taxoniq = None


class MetadataExtractionError(RuntimeError):
    """Base exception for metadata processing failures."""


class AuthorizationRequiredError(MetadataExtractionError):
    """Raised when the downloaded file is an HTML login page rather than data."""


class CSVParsingError(MetadataExtractionError):
    """Raised when a CSV file cannot be parsed with the supported encodings."""


def parse_timespan(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
    """Represent the duration between two timestamps."""
    if pd.isna(start_date) or pd.isna(end_date):
        return "Unknown"

    delta = end_date - start_date
    days = delta.days
    if days < 0:
        return "Unknown"
    if days < 30:
        return f"{days} days"
    if days < 365:
        months = round(days / 30.44)
        return f"{months} month{'s' if months != 1 else ''}"
    years = round(days / 365.25, 1)
    if years == int(years):
        years = int(years)
    return f"{years} year{'s' if years != 1 else ''}"


def get_continent_from_coords(lat: float, lon: float) -> str:
    """Estimate a continent label from latitude/longitude."""
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return "Unknown"

    if math.isnan(lat) or math.isnan(lon):
        return "Unknown"

    if lat >= 35 and -10 <= lon <= 70:
        return "Europe"
    if -35 <= lat <= 37 and -20 <= lon <= 55:
        return "Africa"
    if lat >= 10 and 60 <= lon <= 180:
        return "Asia"
    if -10 <= lat <= 30 and 60 <= lon <= 180:
        return "Asia"
    if -50 <= lat <= -10 and 110 <= lon <= 180:
        return "Australia/Oceania"
    if -180 <= lon <= -30:
        return "North America" if lat >= 25 else "South America"
    return "Unknown"


@lru_cache(maxsize=2048)
def _taxoniq_lookup(scientific_name: str) -> Optional[str]:
    if not taxoniq:
        return None
    try:
        taxon = taxoniq.Taxon(scientific_name=scientific_name)
    except Exception:
        return None
    try:
        common = getattr(taxon, "common_name", None)
    except Exception:
        common = None
    if not common and hasattr(taxon, "common_names"):
        names = getattr(taxon, "common_names")
        if isinstance(names, (list, tuple)):
            common = names[0] if names else None
    if isinstance(common, str) and common.strip():
        return common.strip()
    return None


def get_common_names(scientific_names_str: str) -> str:
    """Map scientific species names to user-friendly common names."""
    if not scientific_names_str or scientific_names_str == "Unknown":
        return "Unknown"
    names = []
    for sci_name in (name.strip() for name in scientific_names_str.split(";")):
        if not sci_name:
            continue
        common = COMMON_NAME_MAPPING.get(sci_name)
        if not common:
            common = _taxoniq_lookup(sci_name)
        names.append(common or sci_name)
    return "; ".join(names) if names else "Unknown"


def extract_doi_url(citation_text: str) -> str:
    """Pull a DOI URL out of a citation string if present."""
    if not citation_text:
        return ""
    doi_patterns = [
        r"https?://(?:www\.)?doi\.org/[^\s<>\"']+",
        r"https?://dx\.doi\.org/[^\s<>\"']+",
        r"doi:\s*([^\s<>\"']+)",
        r"DOI:\s*([^\s<>\"']+)",
    ]
    for pattern in doi_patterns:
        match = re.search(pattern, citation_text)
        if not match:
            continue
        if "doi.org" in match.group(0):
            return match.group(0).rstrip(".,")
        return f"https://doi.org/{match.group(1).rstrip('.,')}"
    return ""


def _normalize_doi_candidate(value: str) -> Optional[str]:
    if not value:
        return None
    candidate = value.strip().strip(".,; ")
    if not candidate:
        return None
    if candidate.lower().startswith("doi:"):
        candidate = candidate.split(":", 1)[1].strip()
    if candidate.startswith("http"):
        return candidate if "doi.org" in candidate else None
    return f"https://doi.org/{candidate}"


def _is_html_document(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            snippet = handle.read(1024).lstrip().lower()
    except OSError:
        return False
    return snippet.startswith(b"<!doctype html") or snippet.startswith(b"<html")


def _candidate_csv_paths(study_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for csv_path in study_dir.rglob("*.csv"):
        name = csv_path.name.lower()
        if any(keyword in name for keyword in EXCLUDE_PATTERNS):
            continue
        paths.append(csv_path)
    paths.sort(key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    return paths


def _accumulate_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _parse_csv_metadata(path: Path, encodings: Sequence[str] = DEFAULT_ENCODINGS) -> Dict[str, object]:
    if _is_html_document(path):
        raise AuthorizationRequiredError(f"{path.name} looks like an HTML login page")

    if path.stat().st_size == 0:
        raise CSVParsingError(f"{path.name} is empty")

    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
    except OSError as exc:
        raise CSVParsingError(f"Failed to read {path.name}: {exc}") from exc

    if signature.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as archive:
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                target_member = members[0] if members else (archive.namelist()[0] if archive.namelist() else None)
                if not target_member:
                    raise CSVParsingError(f"{path.name} zip archive contains no files")
                with tempfile.TemporaryDirectory() as tmpdir:
                    extracted_path = Path(archive.extract(target_member, tmpdir))
                    return _parse_csv_metadata(extracted_path, encodings=encodings)
        except zipfile.BadZipFile as exc:
            raise CSVParsingError(f"{path.name} is not a valid zip archive") from exc

    for encoding in encodings:
        try:
            chunks = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                chunksize=50000,
            )
            return _compute_metadata_from_chunks(chunks)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise CSVParsingError(f"Unable to parse {path.name} with supported encodings")


def _compute_metadata_from_chunks(chunks: Iterable[pd.DataFrame]) -> Dict[str, object]:
    total_rows = 0
    species: Set[str] = set()
    individuals: Set[str] = set()
    lat_samples: List[float] = []
    lon_samples: List[float] = []
    min_timestamp: Optional[pd.Timestamp] = None
    max_timestamp: Optional[pd.Timestamp] = None

    for chunk in chunks:
        if chunk.empty:
            continue
        total_rows += len(chunk)

        for col in SPECIES_COLUMNS:
            if col in chunk:
                values = chunk[col].dropna().astype(str)
                species.update(values.tolist())

        for col in INDIVIDUAL_COLUMNS:
            if col in chunk:
                values = chunk[col].dropna().astype(str)
                individuals.update(values.tolist())

        ts_col = next((c for c in TIMESTAMP_COLUMNS if c in chunk), None)
        if ts_col:
            ts_values = pd.to_datetime(chunk[ts_col], errors="coerce")
            if not ts_values.empty:
                chunk_min = ts_values.min(skipna=True)
                chunk_max = ts_values.max(skipna=True)
                if pd.notna(chunk_min):
                    min_timestamp = (
                        chunk_min
                        if min_timestamp is None or chunk_min < min_timestamp
                        else min_timestamp
                    )
                if pd.notna(chunk_max):
                    max_timestamp = (
                        chunk_max
                        if max_timestamp is None or chunk_max > max_timestamp
                        else max_timestamp
                    )

        lat_col = next((c for c in LATITUDE_COLUMNS if c in chunk), None)
        lon_col = next((c for c in LONGITUDE_COLUMNS if c in chunk), None)
        if lat_col and lon_col:
            lat_values = _accumulate_numeric(chunk[lat_col])
            lon_values = _accumulate_numeric(chunk[lon_col])
            limit = 500
            lat_samples.extend(lat_values.head(limit).tolist())
            lon_samples.extend(lon_values.head(limit).tolist())
            if len(lat_samples) > 4000:
                lat_samples = lat_samples[:4000]
            if len(lon_samples) > 4000:
                lon_samples = lon_samples[:4000]

    if total_rows == 0:
        raise CSVParsingError("CSV contains no data rows")

    continent = "Unknown"
    if lat_samples and lon_samples:
        lat_med = median(lat_samples)
        lon_med = median(lon_samples)
        continent = get_continent_from_coords(lat_med, lon_med)

    return {
        "num_gps_points": total_rows,
        "species": "; ".join(sorted(species)) if species else "Unknown",
        "num_individuals": len(individuals),
        "start_date": min_timestamp.strftime("%Y-%m-%d") if min_timestamp is not None else "Unknown",
        "end_date": max_timestamp.strftime("%Y-%m-%d") if max_timestamp is not None else "Unknown",
        "timespan": parse_timespan(min_timestamp, max_timestamp),
        "continent": continent,
    }


@dataclass
class StudyMetadata:
    study_id: str
    study_name: str
    paper_title: str
    author_list: str
    study_species: str
    study_species_common: str
    num_individuals: int
    num_gps_points: int
    timespan: str
    start_date: str
    end_date: str
    continent: str
    paper_url: str
    main_csv_file: str

    def to_row(self) -> Dict[str, object]:
        return {
            "study_id": self.study_id,
            "study_name": self.study_name,
            "paper_title": self.paper_title,
            "author_list": self.author_list,
            "study_species": self.study_species,
            "study_species_common": self.study_species_common,
            "num_individuals": self.num_individuals,
            "num_gps_points": self.num_gps_points,
            "timespan": self.timespan,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "continent": self.continent,
            "paper_url": self.paper_url,
            "main_csv_file": self.main_csv_file,
        }


class StudyMetadataExtractor:
    """Coordinate metadata extraction for a set of Movebank studies."""

    def __init__(
        self,
        *,
        studies_dir: Path = Path("mbr_studies"),
        links_csv: Path = Path("movebank_links.csv"),
        output_csv: Path = Path("mbr_studies_metadata.csv"),
        redownload_callback: Optional[Callable[[Dict[str, object]], bool]] = None,
    ) -> None:
        self.studies_dir = studies_dir
        self.links_csv = links_csv
        self.output_csv = output_csv
        self.redownload_callback = redownload_callback

    def _load_expected_ids(self) -> List[str]:
        if not self.links_csv.exists():
            logger.warning("Links CSV %s not found; inferring study IDs from directories", self.links_csv)
            return sorted(self._load_available_directories().keys())

        expected: List[str] = []
        with self.links_csv.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                study_id = row.get("study_id")
                if not study_id:
                    url = row.get("url") or ""
                    study_id = url.rstrip("/").split("/")[-1] if url else ""
                if study_id:
                    expected.append(study_id)
        return expected

    def _load_available_directories(self) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for meta_file in self.studies_dir.glob("*/study_metadata.json"):
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", meta_file, exc)
                continue
            study_id = data.get("study_id")
            if not study_id:
                continue
            mapping[study_id] = meta_file.parent
        return mapping

    def _read_download_metadata(self, study_dir: Path) -> Dict[str, object]:
        meta_path = study_dir / "study_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", meta_path, exc)
            return {}

    def _candidate_citation_texts(self, study_dir: Path) -> List[str]:
        texts: List[str] = []
        for filename in ("citation.txt", "citation.bib"):
            path = study_dir / filename
            if path.exists():
                try:
                    texts.append(path.read_text(encoding="utf-8"))
                except OSError:
                    continue
        return texts

    def _derive_paper_title(self, download_meta: Dict[str, object], citation_texts: List[str], study_name: str) -> str:
        title = str(download_meta.get("title", "")).strip()
        if title and title.lower() != "no thumbnail available":
            return title
        raw_title = download_meta.get("raw_row", {}).get("text")
        if raw_title and raw_title.lower() != "no thumbnail available":
            return str(raw_title).strip()
        for citation in citation_texts:
            match = re.search(r"title\s*=\s*{([^}]+)}", citation)
            if match:
                return match.group(1).strip()
        return study_name

    def _derive_author_list(self, download_meta: Dict[str, object], citation_texts: List[str]) -> str:
        author_list = str(download_meta.get("author_list", "")).strip()
        if author_list and author_list.lower() != "unknown":
            return author_list
        for citation in citation_texts:
            match = re.search(r"author\s*=\s*{([^}]+)}", citation)
            if match:
                return match.group(1).strip()
        return "Unknown"

    def _derive_paper_url(self, citation_texts: List[str], download_meta: Dict[str, object]) -> str:
        texts = list(citation_texts)
        download_citation = str(download_meta.get("citation", "") or "")
        if download_citation:
            texts.append(download_citation)

        for citation in texts:
            doi_url = extract_doi_url(citation)
            if doi_url:
                return doi_url

        doi_candidate = download_meta.get("doi_url")
        if isinstance(doi_candidate, str):
            doi_url = extract_doi_url(doi_candidate) or _normalize_doi_candidate(doi_candidate)
            if doi_url:
                return doi_url

        for key in ("paper_url", "source_url"):
            value = download_meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        raw_row = download_meta.get("raw_row", {})
        if isinstance(raw_row, dict):
            raw_url = raw_row.get("url")
            if isinstance(raw_url, str) and raw_url.strip():
                return raw_url.strip()

        return ""

    def _extract_study_metadata(self, study_dir: Path) -> StudyMetadata:
        download_meta = self._read_download_metadata(study_dir)
        study_id = download_meta.get("study_id") or study_dir.name
        study_name = study_dir.name

        citation_texts = self._candidate_citation_texts(study_dir)
        paper_title = self._derive_paper_title(download_meta, citation_texts, study_name)
        author_list = self._derive_author_list(download_meta, citation_texts)
        paper_url = self._derive_paper_url(citation_texts, download_meta)
        csv_metadata: Optional[Dict[str, object]] = None
        chosen_file: Optional[Path] = None
        retry_attempted = False

        while True:
            csv_paths = _candidate_csv_paths(study_dir)
            last_error: Optional[Exception] = None

            for csv_path in csv_paths:
                try:
                    csv_metadata = _parse_csv_metadata(csv_path)
                    chosen_file = csv_path
                    logger.debug("Selected main CSV %s for %s", csv_path.name, study_name)
                    break
                except MetadataExtractionError as exc:
                    logger.warning("Skipping %s: %s", csv_path, exc)
                    last_error = exc
                    continue

            if csv_metadata is not None and chosen_file is not None:
                break

            if not retry_attempted and self.redownload_callback:
                retry_attempted = self.redownload_callback(download_meta)
                if retry_attempted:
                    logger.info("Re-download triggered for %s, retrying metadata extraction", study_name)
                    continue

            raise MetadataExtractionError(
                f"Could not derive metadata for {study_name}: {last_error or 'no suitable CSV files'}"
            )

        species = csv_metadata.get("species", "Unknown")
        common_names = get_common_names(species)

        return StudyMetadata(
            study_id=str(study_id),
            study_name=study_name,
            paper_title=paper_title,
            author_list=author_list,
            study_species=species,
            study_species_common=common_names,
            num_individuals=int(csv_metadata.get("num_individuals", 0)),
            num_gps_points=int(csv_metadata.get("num_gps_points", 0)),
            timespan=str(csv_metadata.get("timespan", "Unknown")),
            start_date=str(csv_metadata.get("start_date", "Unknown")),
            end_date=str(csv_metadata.get("end_date", "Unknown")),
            continent=str(csv_metadata.get("continent", "Unknown")),
            paper_url=paper_url,
            main_csv_file=chosen_file.name,
        )

    def extract(self) -> Dict[str, object]:
        expected_ids = self._load_expected_ids()
        available_dirs = self._load_available_directories()

        results: List[StudyMetadata] = []
        failures: List[Dict[str, str]] = []

        for study_id in expected_ids:
            study_dir = available_dirs.get(study_id)
            if not study_dir:
                failures.append(
                    {
                        "study_id": study_id,
                        "reason": "study directory not found",
                    }
                )
                logger.error("Missing study directory for %s", study_id)
                continue
            try:
                metadata = self._extract_study_metadata(study_dir)
                results.append(metadata)
            except MetadataExtractionError as exc:
                failures.append(
                    {
                        "study_id": study_id,
                        "reason": str(exc),
                        "study_dir": str(study_dir),
                    }
                )
                logger.error("Failed to process %s: %s", study_dir, exc)

        if results:
            fieldnames = list(results[0].to_row().keys())
            with self.output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows([result.to_row() for result in results])

        summary = {
            "expected": len(expected_ids),
            "succeeded": len(results),
            "failed": failures,
            "output": str(self.output_csv),
        }
        logger.info(
            "Metadata extraction complete: %s succeeded, %s failed (expected %s)",
            summary["succeeded"],
            len(failures),
            summary["expected"],
        )
        return summary


def extract_metadata(
    *,
    studies_dir: Path = Path("mbr_studies"),
    links_csv: Path = Path("movebank_links.csv"),
    output_csv: Path = Path("mbr_studies_metadata.csv"),
    redownload_callback: Optional[Callable[[Dict[str, object]], bool]] = None,
) -> Dict[str, object]:
    extractor = StudyMetadataExtractor(
        studies_dir=studies_dir,
        links_csv=links_csv,
        output_csv=output_csv,
        redownload_callback=redownload_callback,
    )
    return extractor.extract()
