#!/usr/bin/env python3

import csv
import os
import re

# Read the CSV file to get all study URLs and titles
csv_file = '/data/vision/beery/scratch/justin/movebench/movebank_links.csv'
mbr_studies_dir = '/data/vision/beery/scratch/justin/movebench/mbr_studies'

def normalize_for_comparison(text):
    """Normalize text for comparison by removing punctuation, quotes, and normalizing whitespace"""
    # Remove quotes and other punctuation
    normalized = text.replace('"', '').replace("'", "").replace(':', '').replace('?', '')
    normalized = normalized.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    normalized = normalized.replace('/', ' ').replace('-', ' ').replace('_', ' ')
    normalized = normalized.replace(',', ' ').replace('.', ' ').replace(';', ' ')

    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = normalized.strip().lower()

    return normalized

# Read all studies from CSV
csv_studies = []
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        url = row['url']
        text = row['text']

        # Extract the study ID from URL
        study_id = url.split('/')[-1]

        # Skip the problematic "Loading..." entry
        if text == "Loading...":
            continue

        csv_studies.append({
            'line': i + 2,  # +2 because CSV reader skips header and we want 1-based line numbers
            'url': url,
            'study_id': study_id,
            'original_text': text,
            'normalized_text': normalize_for_comparison(text)
        })

print(f"Found {len(csv_studies)} valid studies in CSV")

# Get existing directories and normalize their names
existing_dirs = []
for dirname in os.listdir(mbr_studies_dir):
    if os.path.isdir(os.path.join(mbr_studies_dir, dirname)):
        existing_dirs.append({
            'dirname': dirname,
            'normalized': normalize_for_comparison(dirname)
        })

print(f"Found {len(existing_dirs)} directories in mbr_studies")

# Find missing studies by comparing normalized text
missing_studies = []
matched_studies = []

for study in csv_studies:
    found_match = False

    for existing_dir in existing_dirs:
        # Check if the normalized texts are similar enough
        study_words = set(study['normalized_text'].split())
        dir_words = set(existing_dir['normalized'].split())

        # Calculate overlap - if most words from the study title are in the directory name
        if len(study_words) > 0:
            overlap = len(study_words.intersection(dir_words)) / len(study_words)
            if overlap > 0.7:  # 70% of words match
                found_match = True
                matched_studies.append({
                    'study': study,
                    'directory': existing_dir['dirname'],
                    'overlap': overlap
                })
                break

    if not found_match:
        missing_studies.append(study)

print(f"\nMatched studies: {len(matched_studies)}")
print(f"Missing studies: {len(missing_studies)}")

if missing_studies:
    print(f"\nMissing studies:")
    for study in missing_studies:
        print(f"Line {study['line']}: {study['study_id']}")
        print(f"  URL: {study['url']}")
        print(f"  Title: {study['original_text']}")
        print()

# Show a few examples of matched studies for verification
if matched_studies:
    print(f"\nExample matches (first 5):")
    for i, match in enumerate(matched_studies[:5]):
        print(f"  CSV: {match['study']['original_text'][:80]}...")
        print(f"  Dir: {match['directory'][:80]}...")
        print(f"  Overlap: {match['overlap']:.2f}")
        print()