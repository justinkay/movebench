#!/usr/bin/env python3
"""
Handle studies that don't have CSV files but should still be in metadata
"""

import pandas as pd
import os
from pathlib import Path
import re

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

def process_no_csv_studies():
    """Process studies that have no CSV files"""
    
    # Read current metadata
    df = pd.read_csv('mbr_studies_metadata.csv')
    processed_studies = set(df.study_name)
    
    all_dirs = set(os.listdir('mbr_studies'))
    missing_studies = all_dirs - processed_studies
    
    print(f"Processing {len(missing_studies)} studies missing from metadata")
    
    new_rows = []
    
    for study_name in missing_studies:
        print(f"Processing: {study_name}")
        
        study_dir = Path('mbr_studies') / study_name
        
        # Check what files are available
        files = list(study_dir.iterdir()) if study_dir.exists() else []
        csv_files = [f for f in files if f.suffix == '.csv']
        
        print(f"  Files found: {len(files)}, CSV files: {len(csv_files)}")
        
        # Read BibTeX if available
        citation_file = study_dir / 'citation.bib'
        bibtex_metadata = {}
        if citation_file.exists():
            try:
                with open(citation_file, 'r', encoding='utf-8') as f:
                    bibtex_content = f.read()
                    bibtex_metadata = extract_bibtex_metadata(bibtex_content)
                    print(f"  ✓ Found BibTeX citation")
            except Exception as e:
                print(f"  ❌ Error reading citation: {e}")
        
        # Create metadata entry for this study
        study_metadata = {
            'study_name': study_name,
            'paper_title': bibtex_metadata.get('title', study_name),
            'author_list': bibtex_metadata.get('authors', 'Unknown'),
            'study_species': 'Unknown',
            'study_species_common': 'Unknown',
            'num_individuals': 0,
            'num_gps_points': 0,
            'timespan': 'Unknown',
            'start_date': 'Unknown',
            'end_date': 'Unknown',
            'continent': 'Unknown',
            'paper_url': bibtex_metadata.get('url', ''),
            'main_csv_file': 'No CSV file available'
        }
        
        new_rows.append(study_metadata)
        print(f"  ✓ Added to metadata")
    
    if new_rows:
        # Add new rows to existing DataFrame
        new_df = pd.DataFrame(new_rows)
        combined_df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated metadata
        combined_df.to_csv('mbr_studies_metadata.csv', index=False)
        print(f"\n✅ Added {len(new_rows)} studies to metadata")
    else:
        print("\n✅ No new studies to add")
    
    return len(new_rows)

if __name__ == "__main__":
    added_count = process_no_csv_studies()
    
    # Print final summary
    df = pd.read_csv('mbr_studies_metadata.csv')
    total_studies = len(df)
    studies_with_data = len(df[df.num_gps_points > 0])
    studies_no_data = len(df[df.num_gps_points == 0])
    
    print(f"\n=== FINAL METADATA SUMMARY ===")
    print(f"Total studies in metadata: {total_studies}")
    print(f"Studies with GPS data: {studies_with_data}")
    print(f"Studies without GPS data: {studies_no_data}")
    print(f"Coverage: {total_studies}/214 directories")
