#!/usr/bin/env python3
"""
Handle compressed CSV files to improve metadata extraction
"""

import pandas as pd
import os
from pathlib import Path
import zipfile
import gzip
import shutil
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_compressed_file(file_path):
    """Check if a file is compressed based on name and content"""
    file_path = Path(file_path)
    
    # Check filename patterns
    if '.zip' in file_path.name.lower():
        return True
        
    # Check if file starts with zip/gzip magic bytes
    try:
        with open(file_path, 'rb') as f:
            header = f.read(10)
            # ZIP file magic bytes
            if header.startswith(b'PK\x03\x04') or header.startswith(b'PK\x05\x06'):
                return True
            # GZIP magic bytes
            if header.startswith(b'\x1f\x8b'):
                return True
    except:
        pass
        
    return False

def decompress_file(file_path):
    """Decompress a file and return path to decompressed content"""
    file_path = Path(file_path)
    
    try:
        # Try ZIP first
        if '.zip' in file_path.name.lower():
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get the first CSV file in the archive
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    # Extract to temporary file
                    temp_dir = tempfile.mkdtemp()
                    extracted_path = zip_ref.extract(csv_files[0], temp_dir)
                    return extracted_path
        
        # Try GZIP
        try:
            with gzip.open(file_path, 'rb') as gz_file:
                temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
                with open(temp_fd, 'wb') as temp_file:
                    shutil.copyfileobj(gz_file, temp_file)
                return temp_path
        except:
            pass
            
    except Exception as e:
        logger.error(f"Failed to decompress {file_path}: {e}")
        
    return None

def try_read_csv_with_decompression(file_path):
    """Try to read a CSV file, decompressing if necessary"""
    file_path = Path(file_path)
    
    # First try reading directly
    try:
        return pd.read_csv(file_path, nrows=1000, low_memory=False)
    except:
        pass
    
    # If that fails and it looks compressed, try decompressing
    if is_compressed_file(file_path):
        logger.info(f"Attempting to decompress: {file_path.name}")
        decompressed_path = decompress_file(file_path)
        
        if decompressed_path:
            try:
                df = pd.read_csv(decompressed_path, nrows=1000, low_memory=False)
                # Clean up temp file
                os.unlink(decompressed_path)
                return df
            except Exception as e:
                logger.error(f"Failed to read decompressed file: {e}")
                # Clean up temp file
                if os.path.exists(decompressed_path):
                    os.unlink(decompressed_path)
    
    return None

def process_failed_studies():
    """Process studies that failed due to compression issues"""
    
    # Read current metadata
    df = pd.read_csv('mbr_studies_metadata.csv')
    failed_studies = df[df.num_gps_points == 0].copy()
    
    logger.info(f"Processing {len(failed_studies)} studies with compression issues")
    
    success_count = 0
    
    for idx, row in failed_studies.iterrows():
        study_name = row['study_name']
        main_csv_file = row['main_csv_file']
        
        logger.info(f"Processing: {study_name[:50]}...")
        
        # Find the actual CSV file
        study_dir = Path('mbr_studies') / study_name
        csv_file_path = study_dir / main_csv_file
        
        if not csv_file_path.exists():
            logger.warning(f"CSV file not found: {csv_file_path}")
            continue
        
        # Try to read with decompression
        df_sample = try_read_csv_with_decompression(csv_file_path)
        
        if df_sample is not None:
            logger.info(f"✓ Successfully read compressed file: {len(df_sample)} rows sampled")
            
            # Now read full file to get accurate counts
            try:
                if is_compressed_file(csv_file_path):
                    decompressed_path = decompress_file(csv_file_path)
                    if decompressed_path:
                        df_full = pd.read_csv(decompressed_path, low_memory=False)
                        os.unlink(decompressed_path)
                    else:
                        continue
                else:
                    df_full = pd.read_csv(csv_file_path, low_memory=False)
                
                # Update metadata
                num_gps_points = len(df_full)
                
                # Get species info
                species_columns = ['individual-taxon-canonical-name', 'taxon_canonical_name', 'species']
                species_set = set()
                for col in species_columns:
                    if col in df_full.columns:
                        species_values = df_full[col].dropna().unique()
                        species_set.update(species_values)
                
                # Get individual count
                individual_columns = ['individual-local-identifier', 'individual_local_identifier', 'individual_id']
                num_individuals = 0
                for col in individual_columns:
                    if col in df_full.columns:
                        num_individuals = df_full[col].nunique()
                        break
                
                # Update the metadata DataFrame
                df.loc[df.study_name == study_name, 'num_gps_points'] = num_gps_points
                df.loc[df.study_name == study_name, 'num_individuals'] = num_individuals
                if species_set:
                    df.loc[df.study_name == study_name, 'study_species'] = '; '.join(sorted(species_set))
                
                logger.info(f"  ✓ Updated: {num_individuals} individuals, {num_gps_points} GPS points")
                success_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process full file {csv_file_path}: {e}")
        else:
            logger.warning(f"❌ Could not read file: {csv_file_path}")
    
    # Save updated metadata
    df.to_csv('mbr_studies_metadata.csv', index=False)
    logger.info(f"✅ Successfully processed {success_count}/{len(failed_studies)} compressed files")
    
    return success_count

if __name__ == "__main__":
    success_count = process_failed_studies()
    
    # Print summary
    df = pd.read_csv('mbr_studies_metadata.csv')
    readable_studies = len(df[df.num_gps_points > 0])
    total_studies = len(df)
    
    print(f"\n=== COMPRESSION HANDLING RESULTS ===")
    print(f"Successfully processed: {success_count} compressed files")
    print(f"Studies with readable data: {readable_studies}/{total_studies}")
    print(f"Studies still with issues: {total_studies - readable_studies}")
