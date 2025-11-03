#!/usr/bin/env python3
"""
Rename all study directories to LastName_StudyID format (fixed version)
"""

import pandas as pd
import os
import re
from pathlib import Path
import shutil

def extract_study_id_from_url(url):
    """Extract study ID from URL"""
    if pd.isna(url) or not url:
        return 'unknown'
    return url.split('/')[-1]

def extract_first_author_lastname(author_string):
    """Extract first author's last name from author string"""
    if not author_string or author_string == 'Unknown' or pd.isna(author_string):
        return 'Unknown'
    
    # Split by 'and' to get first author
    first_author = str(author_string).split(' and ')[0].strip()
    
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
    
    # Clean lastname - remove non-alphabetic characters
    lastname = re.sub(r'[^a-zA-Z]', '', lastname)
    return lastname if lastname else 'Unknown'

def create_new_directory_name(author_string, paper_url, study_name):
    """Create new directory name in format LastName_StudyID"""
    lastname = extract_first_author_lastname(author_string)
    
    # Try to get study ID from URL, fallback to hash of study name
    if pd.notna(paper_url) and paper_url and 'datarepository.movebank.org' in str(paper_url):
        study_id = extract_study_id_from_url(paper_url)
        short_id = study_id[:8]
    else:
        # Create a short hash from study name
        import hashlib
        hash_obj = hashlib.md5(study_name.encode())
        short_id = hash_obj.hexdigest()[:8]
    
    return f"{lastname}_{short_id}"

def main():
    # Read metadata
    df_meta = pd.read_csv('mbr_studies_metadata.csv')
    
    print(f"Processing {len(df_meta)} studies for renaming...")
    
    rename_mapping = {}
    
    for _, row in df_meta.iterrows():
        old_name = row['study_name']
        author_list = row['author_list']
        paper_url = row['paper_url']
        
        new_name = create_new_directory_name(author_list, paper_url, old_name)
        
        # Handle duplicates by adding suffix
        original_new_name = new_name
        counter = 1
        while new_name in rename_mapping.values():
            new_name = f"{original_new_name}_{counter}"
            counter += 1
        
        rename_mapping[old_name] = new_name
        
    print(f"Created mapping for {len(rename_mapping)} directories")
    
    # Show a sample of the mapping
    print("\nSample mappings:")
    for i, (old_name, new_name) in enumerate(list(rename_mapping.items())[:10]):
        print(f"  {old_name[:40]}... -> {new_name}")
    
    # Check for any potential issues before renaming
    base_dir = Path('mbr_studies')
    existing_dirs = set(os.listdir(base_dir))
    mapped_dirs = set(rename_mapping.keys())
    
    missing_dirs = mapped_dirs - existing_dirs
    if missing_dirs:
        print(f"\n⚠️  Warning: {len(missing_dirs)} directories in metadata not found on disk:")
        for dir_name in list(missing_dirs)[:5]:
            print(f"  - {dir_name}")
    
    # Perform the actual renaming in two passes to avoid conflicts
    print(f"\nStarting rename process...")
    
    # First pass: rename to temporary names
    temp_mapping = {}
    for old_name, new_name in rename_mapping.items():
        old_path = base_dir / old_name
        if not old_path.exists():
            print(f"⚠️  Skipping {old_name} (not found)")
            continue
            
        temp_name = f"TEMP_{hash(old_name) % 100000}_{new_name}"
        temp_path = base_dir / temp_name
        
        try:
            shutil.move(str(old_path), str(temp_path))
            temp_mapping[temp_name] = new_name
        except Exception as e:
            print(f"❌ Failed to rename {old_name}: {e}")
    
    print(f"✅ First pass complete: {len(temp_mapping)} directories renamed to temp names")
    
    # Second pass: rename from temporary to final names
    success_count = 0
    for temp_name, new_name in temp_mapping.items():
        temp_path = base_dir / temp_name
        new_path = base_dir / new_name
        
        try:
            shutil.move(str(temp_path), str(new_path))
            success_count += 1
        except Exception as e:
            print(f"❌ Failed final rename {temp_name} -> {new_name}: {e}")
    
    print(f"✅ Second pass complete: {success_count} directories renamed to final names")
    
    # Update metadata file with new directory names
    df_meta_updated = df_meta.copy()
    for old_name, new_name in rename_mapping.items():
        df_meta_updated.loc[df_meta_updated.study_name == old_name, 'study_name'] = new_name
    
    # Save updated metadata
    df_meta_updated.to_csv('mbr_studies_metadata.csv', index=False)
    print(f"✅ Updated metadata file with new directory names")
    
    # Final verification
    final_dirs = set(os.listdir(base_dir))
    print(f"\nFinal verification:")
    print(f"  Expected: {len(rename_mapping)} directories")
    print(f"  Actual: {len(final_dirs)} directories")
    print(f"  Success rate: {success_count}/{len(rename_mapping)} ({success_count/len(rename_mapping)*100:.1f}%)")

if __name__ == "__main__":
    main()
