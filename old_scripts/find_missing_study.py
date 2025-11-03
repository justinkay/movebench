#!/usr/bin/env python3
"""
Find the missing study by comparing CSV entries with downloaded directories
"""

import csv
import os
from pathlib import Path
import re

def sanitize_filename(filename):
    """Sanitize filename for filesystem compatibility"""
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove extra spaces and truncate if too long
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    if len(sanitized) > 200:
        sanitized = sanitized[:200].strip()
    return sanitized

def extract_study_name_from_text(study_text):
    """Extract a study name from the CSV text"""
    if study_text.startswith('Data from:'):
        return sanitize_filename(study_text)
    elif study_text == "Loading...":
        return "Loading..."
    else:
        return sanitize_filename(f"Data from_ {study_text}")

def main():
    # Read the original CSV
    csv_studies = []
    with open('movebank_links.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            study_name = extract_study_name_from_text(row['text'])
            csv_studies.append({
                'url': row['url'],
                'text': row['text'],
                'expected_dir': study_name
            })

    # Get actual directories
    actual_dirs = set(os.listdir('mbr_studies'))

    print(f"CSV studies: {len(csv_studies)}")
    print(f"Actual directories: {len(actual_dirs)}")
    print()

    # Find missing studies
    missing_studies = []
    matched_studies = []
    
    for study in csv_studies:
        expected_dir = study['expected_dir']

        # Check if exact match exists
        if expected_dir in actual_dirs:
            matched_studies.append(study)
            continue

        # Check for partial matches (in case of slight naming differences)
        partial_matches = [d for d in actual_dirs if expected_dir.lower() in d.lower() or d.lower() in expected_dir.lower()]

        if not partial_matches:
            missing_studies.append(study)
        else:
            matched_studies.append(study)

    print(f"Matched studies: {len(matched_studies)}")
    print(f"Missing studies: {len(missing_studies)}")
    print()

    if missing_studies:
        print("MISSING STUDIES:")
        for study in missing_studies:
            print(f"- {study['text']}")
            print(f"  URL: {study['url']}")
            print(f"  Expected dir: {study['expected_dir']}")
            print()

    # Also check for directories that don't match any CSV entry
    csv_expected_dirs = set(study['expected_dir'] for study in csv_studies)
    extra_dirs = actual_dirs - csv_expected_dirs

    if extra_dirs:
        print(f"Extra directories not in CSV ({len(extra_dirs)}):")
        for extra_dir in sorted(extra_dirs):
            print(f"  {extra_dir}")

if __name__ == "__main__":
    main()
