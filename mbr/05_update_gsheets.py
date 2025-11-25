#!/usr/bin/env python3
"""
Update after_broad_filtering.csv with actual values from mbr_studies_2year_filtered_traintest/
Updates: num_individuals, num_gps_points, start_date, end_date
Preserves section headers and removes rows for studies that don't have corresponding files.
Outputs: after_manual2yr_filtering.csv
"""

import pandas as pd
import glob
import os
from datetime import datetime

def main():
    # Read the original file
    input_file = "../after_broad_filtering.csv"
    output_file = "../after_manual2yr_filtering.csv"
    filtered_dir = "../mbr_studies_2year_filtered_traintest"

    print(f"Reading {input_file}...")
    df_orig = pd.read_csv(input_file)
    print(f"Original file has {len(df_orig)} rows")

    # Get all available filtered files
    filtered_files = glob.glob(os.path.join(filtered_dir, "*_filtered_traintest.csv"))

    # Extract study IDs from filenames
    available_studies = set()
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        study_id = filename.split('_', 1)[0]
        available_studies.add(study_id)

    print(f"Found {len(available_studies)} filtered studies")

    # Identify which rows are actual studies vs section headers/empty rows
    # Study rows have valid study_id values that match our filtered files
    study_col = 'study_id'
    print(f"Using '{study_col}' as study identifier column")

    # Create a mask for rows that are actual studies (have study IDs in our available set)
    study_mask = df_orig[study_col].isin(available_studies)

    # Create a mask for rows that should be kept as section headers
    # These are rows where study_id is empty/NaN OR contains section headers like "Reptiles", "terrestrial mammals", etc.
    section_headers = ['Reptiles', 'Italics:', 'terrestrial mammals', 'avian', 'humans/domestic']
    header_mask = (df_orig[study_col].isna() |
                   (df_orig[study_col] == '') |
                   df_orig[study_col].str.contains('|'.join(section_headers), na=False, case=False))

    # Keep both study rows that we have data for AND header/separator rows
    keep_mask = study_mask | header_mask
    df_filtered = df_orig[keep_mask].copy()

    print(f"Keeping {study_mask.sum()} study rows and {header_mask.sum()} header/separator rows")
    print(f"Total rows after filtering: {len(df_filtered)}")

    # Process each filtered file to get updated stats
    updated_stats = {}

    print("Processing filtered files...")
    for file_path in filtered_files:
        filename = os.path.basename(file_path)
        study_id = filename.split('_', 1)[0]

        try:
            # Read the filtered file
            df_study = pd.read_csv(file_path)

            # Calculate updated statistics
            num_individuals = df_study['individual-local-identifier'].nunique()
            num_gps_points = len(df_study)

            # Get date range
            df_study['timestamp'] = pd.to_datetime(df_study['timestamp'], errors='coerce')
            start_date = df_study['timestamp'].min()
            end_date = df_study['timestamp'].max()

            updated_stats[study_id] = {
                'num_individuals': num_individuals,
                'num_gps_points': num_gps_points,
                'start_date': start_date.strftime('%Y-%m-%d') if pd.notna(start_date) else None,
                'end_date': end_date.strftime('%Y-%m-%d') if pd.notna(end_date) else None
            }

            print(f"  {study_id}: {num_individuals} individuals, {num_gps_points:,} GPS points")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print(f"Successfully processed {len(updated_stats)} studies")

    # Update the dataframe with new statistics (only for actual study rows)
    for idx, row in df_filtered.iterrows():
        study_id = row[study_col]
        # Only update if this is an actual study row (not a header/separator)
        if pd.notna(study_id) and study_id != '' and study_id in updated_stats:
            stats = updated_stats[study_id]

            # Update the columns
            df_filtered.loc[idx, 'num_individuals'] = stats['num_individuals']
            df_filtered.loc[idx, 'num_gps_points'] = stats['num_gps_points']

            if stats['start_date']:
                df_filtered.loc[idx, 'start_date'] = stats['start_date']
            if stats['end_date']:
                df_filtered.loc[idx, 'end_date'] = stats['end_date']

    # Save the updated file
    df_filtered.to_csv(output_file, index=False)
    print(f"\nSaved updated file to {output_file}")
    print(f"Final file has {len(df_filtered)} rows (should be 37)")

    # Summary statistics (only for actual study rows)
    study_rows = df_filtered[df_filtered[study_col].notna() & (df_filtered[study_col] != '')]
    total_individuals = study_rows['num_individuals'].sum()
    total_gps = study_rows['num_gps_points'].sum()
    print(f"Total individuals: {total_individuals:,}")
    print(f"Total GPS points: {total_gps:,}")
    print(f"Study rows: {len(study_rows)}, Header/separator rows: {len(df_filtered) - len(study_rows)}")

if __name__ == "__main__":
    main()