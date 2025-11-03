#!/usr/bin/env python3
"""
Comprehensive analysis to find missing studies and naming issues
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

def main():
    # Read the original CSV
    csv_studies = []
    with open('movebank_links.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            csv_studies.append({
                'line': i + 1,  # +1 for header
                'url': row['url'],
                'text': row['text'],
                'page': row['page']
            })

    # Get actual directories
    actual_dirs = os.listdir('mbr_studies')

    print(f"=== COMPREHENSIVE STUDY ANALYSIS ===")
    print(f"CSV studies: {len(csv_studies)}")
    print(f"Actual directories: {len(actual_dirs)}")
    print()

    # Group studies by ID from URL
    csv_by_id = {}
    for study in csv_studies:
        study_id = study['url'].split('/')[-1]
        csv_by_id[study_id] = study

    # For each directory, try to match it back to a CSV entry
    matched_ids = set()
    unmatched_dirs = []

    for dir_name in actual_dirs:
        found_match = False
        
        # Try to find the study ID by checking citation files or other methods
        study_path = Path('mbr_studies') / dir_name
        
        # Check if there's a citation.bib or study_page.html that might contain the URL
        for file_name in ['citation.bib', 'study_page.html']:
            file_path = study_path / file_name
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        
                    # Look for the study ID in the content
                    for study_id, study_info in csv_by_id.items():
                        if study_id in content:
                            matched_ids.add(study_id)
                            print(f"✓ MATCHED: {dir_name[:60]}...")
                            print(f"  -> Line {study_info['line']}: {study_info['text'][:60]}...")
                            print(f"  -> ID: {study_id}")
                            print()
                            found_match = True
                            break
                    
                    if found_match:
                        break
                        
                except Exception as e:
                    continue
        
        if not found_match:
            unmatched_dirs.append(dir_name)

    # Find CSV entries that don't have matching directories
    unmatched_csv = []
    for study_id, study_info in csv_by_id.items():
        if study_id not in matched_ids:
            unmatched_csv.append(study_info)

    print(f"✓ Matched studies: {len(matched_ids)}")
    print(f"❌ Unmatched directories: {len(unmatched_dirs)}")
    print(f"❌ Unmatched CSV entries: {len(unmatched_csv)}")
    print()

    if unmatched_dirs:
        print("UNMATCHED DIRECTORIES:")
        for dir_name in unmatched_dirs:
            print(f"  - {dir_name}")
        print()

    if unmatched_csv:
        print("MISSING STUDIES (in CSV but no matching directory found):")
        for study in unmatched_csv:
            print(f"  - Line {study['line']}: {study['text']}")
            print(f"    URL: {study['url']}")
            print()

if __name__ == "__main__":
    main()
