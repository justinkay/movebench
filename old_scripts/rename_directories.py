#!/usr/bin/env python3
"""
Rename all study directories to LastName_StudyID format
"""

import pandas as pd
import os
import re
from pathlib import Path
import shutil

def extract_study_id_from_url(url):
    """Extract study ID from URL"""
    return url.split('/')[-1]

def extract_first_author_lastname(author_string):
    """Extract first author's last name from author string"""
    if not author_string or author_string == 'Unknown':
        return 'Unknown'
    
    # Handle different author formats
    # Format: "LastName, FirstName and LastName2, FirstName2"
    # Format: "LastName FirstName and LastName2 FirstName2"
    # Format: "LastName F and LastName2 F2"
    
    # Split by 'and' to get first author
    first_author = author_string.split(' and ')[0].strip()
    
    # Handle comma-separated format (LastName, FirstName)
    if ',' in first_author:
        lastname = first_author.split(',')[0].strip()
    else:
        # Handle space-separated format, take last word as lastname
        parts = first_author.split()
        if parts:
            lastname = parts[-1].strip()
        else:
            lastname = first_author
    
    # Clean lastname
    lastname = re.sub(r'[^a-zA-Z]', '', lastname)
    return lastname if lastname else 'Unknown'

def create_new_directory_name(author_string, url):
    """Create new directory name in format LastName_StudyID"""
    lastname = extract_first_author_lastname(author_string)
    study_id = extract_study_id_from_url(url)
    
    # Take first 8 chars of study ID for brevity
    short_id = study_id[:8]
    
    return f"{lastname}_{short_id}"

def main():
    # Read metadata and original CSV to get author info and URLs
    df_meta = pd.read_csv('mbr_studies_metadata.csv')
    df_links = pd.read_csv('movebank_links.csv')
    
    # Create mapping from study names to URLs
    url_mapping = {}
    for _, row in df_links.iterrows():
        # Try to match study names to URLs (this might be imperfect due to naming differences)
        url_mapping[row['url']] = row['text']
    
    print(f"Processing {len(df_meta)} studies for renaming...")
    
    rename_mapping = {}
    failed_renames = []
    
    for _, row in df_meta.iterrows():
        old_name = row['study_name']
        author_list = row['author_list']
        paper_url = row['paper_url']
        
        # Extract study ID from URL
        if paper_url and 'datarepository.movebank.org' in paper_url:
            study_id = extract_study_id_from_url(paper_url)
        else:
            # Try to find matching URL from links CSV
            study_id = 'unknown'
            # This is a fallback - we'll use a counter for unknown IDs
        
        new_name = create_new_directory_name(author_list, paper_url or f"unknown/{study_id}")
        
        # Handle duplicates by adding suffix
        original_new_name = new_name
        counter = 1
        while new_name in rename_mapping.values():
            new_name = f"{original_new_name}_{counter}"
            counter += 1
        
        rename_mapping[old_name] = new_name
        
        print(f"{old_name[:50]}... -> {new_name}")
    
    print(f"\nCreated mapping for {len(rename_mapping)} directories")
    
    # Perform the actual renaming
    base_dir = Path('mbr_studies')
    success_count = 0
    
    # First pass: rename to temporary names to avoid conflicts
    temp_mapping = {}
    for old_name, new_name in rename_mapping.items():
        old_path = base_dir / old_name
        temp_name = f"TEMP_{new_name}"
        temp_path = base_dir / temp_name
        
        if old_path.exists():
            try:
                shutil.move(str(old_path), str(temp_path))
                temp_mapping[temp_name] = new_name
                print(f"✓ Temp renamed: {old_name} -> {temp_name}")
            except Exception as e:
                print(f"❌ Failed to temp rename {old_name}: {e}")
                failed_renames.append((old_name, new_name, str(e)))
        else:
            print(f"⚠️  Directory not found: {old_name}")
            failed_renames.append((old_name, new_name, "Directory not found"))
    
    # Second pass: rename from temporary to final names
    for temp_name, new_name in temp_mapping.items():
        temp_path = base_dir / temp_name
        new_path = base_dir / new_name
        
        try:
            shutil.move(str(temp_path), str(new_path))
            success_count += 1
            print(f"✓ Final renamed: {temp_name} -> {new_name}")
        except Exception as e:
            print(f"❌ Failed final rename {temp_name} -> {new_name}: {e}")
            failed_renames.append((temp_name, new_name, str(e)))
    
    # Update metadata file with new directory names
    df_meta_updated = df_meta.copy()
    for old_name, new_name in rename_mapping.items():
        df_meta_updated.loc[df_meta_updated.study_name == old_name, 'study_name'] = new_name
    
    # Save updated metadata
    df_meta_updated.to_csv('mbr_studies_metadata.csv', index=False)
    
    print(f"\n=== RENAMING SUMMARY ===")
    print(f"✅ Successfully renamed: {success_count}")
    print(f"❌ Failed renames: {len(failed_renames)}")
    
    if failed_renames:
        print(f"\nFailed renames:")
        for old, new, error in failed_renames[:5]:
            print(f"  {old} -> {new}: {error}")
    
    print(f"\n✅ Updated metadata file with new directory names")
    
    # Verify final state
    actual_dirs = set(os.listdir('mbr_studies'))
    expected_dirs = set(rename_mapping.values())
    
    print(f"\nVerification:")
    print(f"Expected directories: {len(expected_dirs)}")
    print(f"Actual directories: {len(actual_dirs)}")
    print(f"Missing: {len(expected_dirs - actual_dirs)}")

if __name__ == "__main__":
    main()
