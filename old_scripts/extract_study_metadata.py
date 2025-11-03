#!/usr/bin/env python3
"""
Extract high-level metadata from downloaded Movebank studies.
Adapted from download_all_mb/generate_study_info.py to work with mbr_studies/ structure.
"""

import csv
import os
import re
from pathlib import Path
import pandas as pd
from datetime import datetime
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_timespan(start_date, end_date):
    """Parse start and end dates to calculate timespan."""
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        delta = end - start

        days = delta.days
        if days < 30:
            return f"{days} days"
        elif days < 365:
            months = round(days / 30.44)
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            years = round(days / 365.25, 1)
            if years == int(years):
                years = int(years)
            return f"{years} year{'s' if years != 1 else ''}"
    except:
        return "Unknown"

def get_continent_from_coords(lat, lon):
    """Estimate continent from coordinates."""
    try:
        lat, lon = float(lat), float(lon)

        # Simple continent boundaries (approximate)
        if lat >= 35 and lon >= -10 and lon <= 70:  # Europe
            return "Europe"
        elif lat >= -35 and lat <= 37 and lon >= -20 and lon <= 55:  # Africa
            return "Africa"
        elif lat >= 10 and lon >= 60 and lon <= 180:  # Asia (northern part)
            return "Asia"
        elif lat >= -10 and lat <= 30 and lon >= 60 and lon <= 180:  # Asia (southern part)
            return "Asia"
        elif lat >= -50 and lat <= -10 and lon >= 110 and lon <= 180:  # Australia/Oceania
            return "Australia/Oceania"
        elif lon >= -180 and lon <= -30:  # Americas
            if lat >= 25:
                return "North America"
            else:
                return "South America"
        else:
            return "Unknown"
    except:
        return "Unknown"

def get_common_names(scientific_names_str):
    """Convert scientific names to common names (simplified version without taxoniq)."""
    if not scientific_names_str or scientific_names_str == 'Unknown':
        return 'Unknown'

    # Simple mapping for common animal names (can be expanded)
    common_name_mapping = {
        'Tyto furcata': 'American Barn Owl',
        'Panthera leo': 'Lion',
        'Elephas maximus': 'Asian Elephant',
        'Loxodonta africana': 'African Elephant',
        'Ursus americanus': 'American Black Bear',
        'Canis lupus': 'Gray Wolf',
        'Vulpes vulpes': 'Red Fox',
        'Aquila chrysaetos': 'Golden Eagle',
        'Falco peregrinus': 'Peregrine Falcon',
        'Corvus corax': 'Common Raven',
        'Grus grus': 'Common Crane',
        'Chelonia mydas': 'Green Sea Turtle'
    }

    scientific_names = [name.strip() for name in scientific_names_str.split(';')]
    common_names = []

    for sci_name in scientific_names:
        common_name = common_name_mapping.get(sci_name, sci_name)
        common_names.append(common_name)

    return '; '.join(common_names)

def extract_doi_url(citation_text):
    """Extract DOI URL from citation text."""
    if not citation_text or pd.isna(citation_text):
        return ""

    # Look for DOI patterns
    doi_patterns = [
        r'https?://(?:www\.)?doi\.org/[^\s<>"]+',
        r'https?://dx\.doi\.org/[^\s<>"]+',
        r'doi:\s*([^\s<>"]+)',
        r'DOI:\s*([^\s<>"]+)'
    ]

    for pattern in doi_patterns:
        match = re.search(pattern, citation_text)
        if match:
            if 'doi.org' in match.group(0):
                return match.group(0).rstrip('.,')
            else:
                # For doi: prefix, construct the URL
                return f"https://doi.org/{match.group(1).rstrip('.,')}"

    return ""

def extract_bibtex_metadata(bibtex_content):
    """Extract metadata from BibTeX content."""
    metadata = {}

    if not bibtex_content:
        return metadata

    # Extract title
    title_match = re.search(r'title\s*=\s*{([^}]+)}', bibtex_content)
    if title_match:
        metadata['title'] = title_match.group(1).strip()

    # Extract authors
    author_match = re.search(r'author\s*=\s*{([^}]+)}', bibtex_content)
    if author_match:
        metadata['authors'] = author_match.group(1).strip()

    # Extract DOI
    doi_match = re.search(r'doi\s*=\s*{([^}]+)}', bibtex_content)
    if doi_match:
        metadata['doi'] = doi_match.group(1).strip()

    # Extract URL
    url_match = re.search(r'URL\s*=\s*{([^}]+)}', bibtex_content)
    if url_match:
        metadata['url'] = url_match.group(1).strip()

    return metadata

def find_main_csv_file(study_dir):
    """Find the main data CSV file (largest one, excluding reference data)."""
    csv_files = list(study_dir.glob('*.csv'))

    if not csv_files:
        return None

    # Filter out reference data files and compressed files that can't be read directly
    main_files = [f for f in csv_files if 'reference-data' not in f.name and not f.name.endswith('.zip.csv')]

    if not main_files:
        # If no uncompressed main files, try any CSV (but prefer uncompressed)
        uncompressed_files = [f for f in csv_files if not f.name.endswith('.zip.csv')]
        main_files = uncompressed_files if uncompressed_files else csv_files

    # Return the largest file
    return max(main_files, key=lambda f: f.stat().st_size)

def extract_metadata_from_csv(csv_file):
    """Extract metadata from the main CSV file."""
    try:
        # Try different encodings for problematic files
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df_sample = None
        df_full = None

        for encoding in encodings:
            try:
                # Read just a sample to understand structure
                df_sample = pd.read_csv(csv_file, nrows=1000, encoding=encoding, low_memory=False)
                df_full = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue

        if df_sample is None or df_full is None:
            raise ValueError("Could not read CSV with any encoding")

        metadata = {}

        # Number of GPS points
        metadata['num_gps_points'] = len(df_full)

        # Species information
        species_columns = ['individual-taxon-canonical-name', 'taxon_canonical_name', 'species']
        species_set = set()

        for col in species_columns:
            if col in df_full.columns:
                species_values = df_full[col].dropna().unique()
                species_set.update(species_values)

        metadata['species'] = '; '.join(sorted(species_set)) if species_set else 'Unknown'

        # Number of individuals
        individual_columns = ['individual-local-identifier', 'individual_local_identifier', 'individual_id']
        num_individuals = 0

        for col in individual_columns:
            if col in df_full.columns:
                num_individuals = df_full[col].nunique()
                break

        metadata['num_individuals'] = num_individuals

        # Time span
        timestamp_columns = ['timestamp', 'timestamp_start', 'time']
        timestamps = []

        for col in timestamp_columns:
            if col in df_full.columns:
                timestamps = pd.to_datetime(df_full[col], errors='coerce').dropna()
                break

        if len(timestamps) > 0:
            start_date = timestamps.min()
            end_date = timestamps.max()
            metadata['timespan'] = parse_timespan(start_date, end_date)
            metadata['start_date'] = start_date.strftime('%Y-%m-%d')
            metadata['end_date'] = end_date.strftime('%Y-%m-%d')
        else:
            metadata['timespan'] = 'Unknown'
            metadata['start_date'] = 'Unknown'
            metadata['end_date'] = 'Unknown'

        # Location information for continent detection
        lat_columns = ['location-lat', 'location_lat', 'latitude', 'lat']
        lon_columns = ['location-long', 'location_long', 'longitude', 'lon', 'long']

        lat_col = None
        lon_col = None

        for col in lat_columns:
            if col in df_sample.columns:
                lat_col = col
                break

        for col in lon_columns:
            if col in df_sample.columns:
                lon_col = col
                break

        if lat_col and lon_col:
            # Use median coordinates for continent detection
            lat_median = df_sample[lat_col].median()
            lon_median = df_sample[lon_col].median()
            metadata['continent'] = get_continent_from_coords(lat_median, lon_median)
        else:
            metadata['continent'] = 'Unknown'

        return metadata

    except Exception as e:
        logger.error(f"Error processing CSV {csv_file}: {e}")
        return {}

def process_study(study_dir):
    """Process a single study directory and extract metadata."""
    logger.info(f"Processing study: {study_dir.name}")

    try:
        # Find the main CSV file
        main_csv = find_main_csv_file(study_dir)
        if not main_csv:
            logger.warning(f"No CSV files found in {study_dir.name}")
            return None

        # Extract metadata from CSV
        csv_metadata = extract_metadata_from_csv(main_csv)

        # Read BibTeX citation if available
        citation_file = study_dir / 'citation.bib'
        bibtex_metadata = {}
        if citation_file.exists():
            with open(citation_file, 'r', encoding='utf-8') as f:
                bibtex_content = f.read()
                bibtex_metadata = extract_bibtex_metadata(bibtex_content)

        # Combine metadata
        study_metadata = {
            'study_name': study_dir.name,
            'paper_title': bibtex_metadata.get('title', study_dir.name),
            'author_list': bibtex_metadata.get('authors', 'Unknown'),
            'study_species': csv_metadata.get('species', 'Unknown'),
            'study_species_common': get_common_names(csv_metadata.get('species', 'Unknown')),
            'num_individuals': csv_metadata.get('num_individuals', 0),
            'num_gps_points': csv_metadata.get('num_gps_points', 0),
            'timespan': csv_metadata.get('timespan', 'Unknown'),
            'start_date': csv_metadata.get('start_date', 'Unknown'),
            'end_date': csv_metadata.get('end_date', 'Unknown'),
            'continent': csv_metadata.get('continent', 'Unknown'),
            'paper_url': bibtex_metadata.get('url', '') or extract_doi_url(bibtex_metadata.get('doi', '')),
            'main_csv_file': main_csv.name
        }

        logger.info(f"  ✓ {study_metadata['num_individuals']} individuals, {study_metadata['num_gps_points']} GPS points, {study_metadata['timespan']}")

        return study_metadata

    except Exception as e:
        logger.error(f"Error processing study {study_dir.name}: {e}")
        return None

def main():
    # Set up directories
    studies_dir = Path("mbr_studies")
    if not studies_dir.exists():
        logger.error("Error: mbr_studies directory not found!")
        return

    # Get list of study directories
    study_dirs = [d for d in studies_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(study_dirs)} study directories")

    # Process each study
    output_rows = []
    processed_count = 0

    for study_dir in study_dirs:
        metadata = process_study(study_dir)
        if metadata:
            output_rows.append(metadata)
            processed_count += 1

    # Write output CSV
    output_file = "mbr_studies_metadata.csv"
    if output_rows:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_rows[0].keys())
            writer.writeheader()
            writer.writerows(output_rows)

    logger.info(f"\n✓ Generated {output_file} with {len(output_rows)} studies")
    print(f"Processed {processed_count} studies successfully")
    print(f"Generated metadata file: {output_file}")

if __name__ == "__main__":
    main()