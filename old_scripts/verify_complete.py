#!/usr/bin/env python3
"""
Simple verification that we have all studies
"""

import csv
import os

def main():
    # Read CSV
    csv_count = 0
    with open('movebank_links.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_count += 1

    # Count directories
    dir_count = len(os.listdir('mbr_studies'))

    print(f"CSV studies: {csv_count}")
    print(f"Downloaded directories: {dir_count}")
    print(f"Difference: {csv_count - dir_count}")
    print()

    if csv_count == dir_count:
        print("✅ SUCCESS: We have all studies downloaded!")
    elif csv_count == 215 and dir_count == 214:
        print("❌ We are missing exactly 1 study")
        print("This could be:")
        print("1. The 'Loading...' study that failed to download")
        print("2. A study that got renamed/truncated during download")
        print("3. A duplicate that overwrote another study")
    else:
        print(f"❌ Unexpected difference: {csv_count - dir_count}")

    # Let's also check for very obvious matches
    with open('movebank_links.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        loading_studies = []
        for i, row in enumerate(reader, 1):
            if row['text'] == "Loading...":
                loading_studies.append((i+1, row['url']))

    if loading_studies:
        print(f"\nFound {len(loading_studies)} 'Loading...' entries:")
        for line_num, url in loading_studies:
            study_id = url.split('/')[-1]
            print(f"  Line {line_num}: {study_id}")
            
        # Check if we have the reindeer study for the Loading entry
        reindeer_dir = "Data from_ Study _Reindeer movement in East Iceland_"
        if reindeer_dir in os.listdir('mbr_studies'):
            print(f"\n✅ The 'Loading...' study was downloaded as: {reindeer_dir}")
        else:
            print(f"\n❌ The 'Loading...' study is actually missing")

if __name__ == "__main__":
    main()
