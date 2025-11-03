#!/usr/bin/env python3

import csv
import os
from pathlib import Path

csv_file = '/data/vision/beery/scratch/justin/movebench/movebank_links.csv'
mbr_studies_dir = '/data/vision/beery/scratch/justin/movebench/mbr_studies'

# Read the CSV and identify the "Loading..." entry
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    loading_study = None
    total_studies = 0

    for i, row in enumerate(reader):
        total_studies += 1
        if row['text'] == 'Loading...':
            loading_study = {
                'line': i + 2,
                'url': row['url'],
                'study_id': row['url'].split('/')[-1]
            }
            print(f"Found 'Loading...' study at line {loading_study['line']}:")
            print(f"  URL: {loading_study['url']}")
            print(f"  Study ID: {loading_study['study_id']}")

print(f"\nTotal studies in CSV: {total_studies}")

# Count directories
dir_count = len([d for d in os.listdir(mbr_studies_dir) if os.path.isdir(os.path.join(mbr_studies_dir, d))])
print(f"Total directories in mbr_studies: {dir_count}")

# Check if the Reindeer study directory exists and has the Loading study ID
reindeer_dir = Path(mbr_studies_dir) / "Data from_ Study _Reindeer movement in East Iceland_"
if reindeer_dir.exists():
    print(f"\nReindeer study directory exists: {reindeer_dir}")

    # Check the study URL in the HTML file if it exists
    html_file = reindeer_dir / "study_page.html"
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
            if loading_study and loading_study['study_id'] in html_content:
                print(f"✓ Confirmed: Reindeer study corresponds to the 'Loading...' entry")
                print(f"  Study ID {loading_study['study_id']} found in HTML content")
            else:
                print(f"✗ Could not confirm Reindeer study matches 'Loading...' entry")

# Final status
if dir_count == total_studies:
    print(f"\n✓ SUCCESS: All {total_studies} studies from CSV have been downloaded!")
elif dir_count == total_studies - 1:
    print(f"\n⚠ We have {dir_count} directories but {total_studies} studies in CSV")
    print(f"  This likely means the 'Loading...' entry was problematic and the Reindeer study was already partially downloaded")
else:
    print(f"\n✗ MISMATCH: {dir_count} directories vs {total_studies} studies in CSV")