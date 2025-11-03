#!/usr/bin/env python3
"""
Directory Renaming and Download Fixes
=====================================

This script handles post-download organization and fixes:
1. Renames directories to LastName_StudyID format for filesystem compatibility
2. Identifies and re-downloads failed studies
3. Handles special cases and missing studies

Author: Claude Code Assistant
Date: November 2024
"""

import pandas as pd
import re
import hashlib
from pathlib import Path
import logging
import requests
from bs4 import BeautifulSoup
import time
import shutil
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_first_author_lastname(author_string):
    """Extract the last name of the first author"""
    if pd.isna(author_string) or author_string == 'Unknown':
        return 'Unknown'

    # Clean up the author string
    author_string = str(author_string).strip()

    # Split by common delimiters
    authors = re.split(r'[,;]', author_string)
    if authors:
        first_author = authors[0].strip()

        # Handle "Last, First" format
        if ',' in first_author:
            parts = first_author.split(',')
            return parts[0].strip()

        # Handle "First Last" format
        parts = first_author.split()
        if len(parts) >= 2:
            return parts[-1]  # Last word is usually the last name
        elif len(parts) == 1:
            return parts[0]  # Single name

    return 'Unknown'

def extract_study_id_from_url(url):
    """Extract the study ID from a Movebank URL"""
    if pd.isna(url):
        return None

    url_str = str(url)
    # Look for the UUID pattern in the URL
    uuid_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
    match = re.search(uuid_pattern, url_str)

    if match:
        return match.group(1)
    return None

def create_new_directory_name(author_string, paper_url, study_name):
    """Create a new directory name in the format LastName_StudyID"""
    lastname = extract_first_author_lastname(author_string)

    if pd.notna(paper_url) and 'datarepository.movebank.org' in str(paper_url):
        study_id = extract_study_id_from_url(paper_url)
        if study_id:
            short_id = study_id[:8]
        else:
            # Fallback to hash if no ID found
            hash_obj = hashlib.md5(study_name.encode())
            short_id = hash_obj.hexdigest()[:8]
    else:
        # Create hash from study name for studies without proper URLs
        hash_obj = hashlib.md5(study_name.encode())
        short_id = hash_obj.hexdigest()[:8]

    return f"{lastname}_{short_id}"

def rename_directories():
    """Rename all study directories to LastName_StudyID format"""
    logger.info("Starting directory renaming process...")

    # Read the metadata file to get author and URL information
    try:
        df = pd.read_csv('mbr_studies_metadata_updated.csv')
    except FileNotFoundError:
        logger.error("Metadata file not found. Please run metadata extraction first.")
        return False

    studies_dir = Path("mbr_studies")
    if not studies_dir.exists():
        logger.error("mbr_studies directory not found!")
        return False

    # Get current directories
    current_dirs = [d for d in studies_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(current_dirs)} directories to potentially rename")

    # Create mapping of old names to new names
    renaming_map = {}
    new_names_used = set()

    for _, row in df.iterrows():
        study_name = row['study_name']
        author_list = row['author_list']
        paper_url = row['paper_url']

        # Find corresponding directory
        current_dir = None
        for dir_path in current_dirs:
            if dir_path.name == study_name:
                current_dir = dir_path
                break

        if current_dir is None:
            logger.warning(f"Directory not found for study: {study_name}")
            continue

        # Generate new name
        new_name = create_new_directory_name(author_list, paper_url, study_name)

        # Handle duplicates
        original_new_name = new_name
        counter = 1
        while new_name in new_names_used:
            new_name = f"{original_new_name}_{counter}"
            counter += 1

        new_names_used.add(new_name)

        if current_dir.name != new_name:
            renaming_map[current_dir] = studies_dir / new_name

    # Perform renaming in two passes to avoid conflicts
    logger.info(f"Will rename {len(renaming_map)} directories")

    # Pass 1: Rename to temporary names
    temp_mappings = {}
    for old_path, new_path in renaming_map.items():
        temp_name = f"temp_rename_{old_path.name}_{int(time.time())}"
        temp_path = studies_dir / temp_name

        try:
            old_path.rename(temp_path)
            temp_mappings[temp_path] = new_path
            logger.info(f"  Temp renamed: {old_path.name} -> {temp_name}")
        except Exception as e:
            logger.error(f"  Failed to temp rename {old_path.name}: {e}")

    # Pass 2: Rename from temporary to final names
    success_count = 0
    for temp_path, final_path in temp_mappings.items():
        try:
            temp_path.rename(final_path)
            success_count += 1
            logger.info(f"  Final renamed: {temp_path.name} -> {final_path.name}")
        except Exception as e:
            logger.error(f"  Failed to final rename {temp_path.name}: {e}")

    logger.info(f"Successfully renamed {success_count} directories")
    return True

class FailedStudyDownloader:
    """Download studies that previously failed"""

    def __init__(self, base_dir='mbr_studies'):
        self.base_dir = Path(base_dir)
        self.base_url = "https://datarepository.movebank.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if len(sanitized) > 200:
            sanitized = sanitized[:200].strip()
        return sanitized

    def extract_bibtex(self, soup):
        """Extract BibTeX citation from the page"""
        for tag in soup.find_all(['pre', 'code', 'div', 'textarea']):
            text = tag.get_text()
            if '@misc{' in text or '@article{' in text or '@dataset{' in text:
                return text.strip()
        return None

    def find_download_links(self, soup):
        """Find all downloadable files"""
        download_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/bitstreams/' in href and '/download' in href:
                filename = link.get_text(strip=True)
                if not filename or filename.lower() in ['download', 'view', '']:
                    parent = link.parent
                    if parent:
                        filename = parent.get_text(strip=True)

                if filename:
                    filename = self.sanitize_filename(filename)
                    if not filename.endswith(('.csv', '.txt', '.pdf', '.zip', '.xlsx', '.json')):
                        if 'csv' in filename.lower():
                            filename += '.csv'
                        elif 'readme' in filename.lower():
                            filename += '.txt'
                        else:
                            filename += '.dat'
                else:
                    url_parts = href.split('/')
                    filename = f"file_{url_parts[-2][:8]}.dat" if len(url_parts) > 2 else "unknown_file.dat"

                full_url = urljoin(self.base_url, href)
                download_links.append((full_url, filename))
        return download_links

    def download_file(self, url, filepath):
        """Download a single file"""
        try:
            logger.info(f"    Downloading: {filepath.name}")
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            file_size = filepath.stat().st_size
            logger.info(f"    ✓ Downloaded: {filepath.name} ({file_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"    ❌ Failed to download {filepath.name}: {e}")
            return False

    def download_study(self, study_url, current_dir_name):
        """Download a complete study"""
        try:
            logger.info(f"\n=== Re-downloading {current_dir_name} ===")
            logger.info(f"URL: {study_url}")

            # Get the study page
            response = self.session.get(study_url, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Use current directory (already exists with proper name)
            study_dir = self.base_dir / current_dir_name

            # Clear existing content if any
            if study_dir.exists():
                for item in study_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            else:
                study_dir.mkdir(exist_ok=True)

            files_downloaded = 0

            # Find and download all files
            download_links = self.find_download_links(soup)
            logger.info(f"  Found {len(download_links)} files to download")

            if not download_links:
                logger.warning("  ❌ No download links found!")
                return False

            for file_url, filename in download_links:
                filepath = study_dir / filename
                if self.download_file(file_url, filepath):
                    files_downloaded += 1
                time.sleep(1)  # Be respectful

            # Save BibTeX citation
            bibtex = self.extract_bibtex(soup)
            if bibtex:
                bibtex_file = study_dir / "citation.bib"
                with open(bibtex_file, 'w', encoding='utf-8') as f:
                    f.write(bibtex)
                logger.info(f"    ✓ Saved BibTeX citation")
                files_downloaded += 1

            # Save the HTML page for reference
            html_file = study_dir / "study_page.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            files_downloaded += 1

            logger.info(f"  ✅ Successfully downloaded {files_downloaded} files")
            return files_downloaded > 0

        except Exception as e:
            logger.error(f"  ❌ Error downloading study: {e}")
            return False

def retry_failed_studies():
    """Retry downloading studies that previously failed"""

    # These are the known failed studies with their URLs
    failed_studies = [
        ('Unknown_e6e86353', 'https://datarepository.movebank.org/entities/datapackage/cd62a27e-79b2-4d8e-92de-8c5643104bfd'),
        ('Unknown_13f86496', 'https://datarepository.movebank.org/entities/datapackage/12a063ce-17f5-485c-9e93-13fa7cad4d5d'),
        ('Unknown_a3a3c141', 'https://datarepository.movebank.org/entities/datapackage/8b91498b-27ce-4c9b-9f28-104204b1ed08'),
        ('Unknown_44611e27', 'https://datarepository.movebank.org/entities/datapackage/29312801-d75d-418a-b4ce-8c5d0ad5b045'),
        ('Unknown_8d0f4088', 'https://datarepository.movebank.org/entities/datapackage/6360747f-37f5-40f1-9731-f4926387a21e')
    ]

    downloader = FailedStudyDownloader()
    success_count = 0

    logger.info(f"Re-downloading {len(failed_studies)} failed studies...")

    for study_name, study_url in failed_studies:
        if downloader.download_study(study_url, study_name):
            success_count += 1

        time.sleep(3)  # Be extra respectful between downloads

    logger.info(f"Re-download complete! Successfully downloaded: {success_count}/{len(failed_studies)} studies")
    return success_count

def main():
    """Main execution function"""
    print("=== Directory Renaming and Download Fixes ===")
    print("This script will:")
    print("1. Rename directories to LastName_StudyID format")
    print("2. Re-download previously failed studies")
    print()

    # Step 1: Rename directories
    logger.info("Step 1: Renaming directories...")
    if not rename_directories():
        logger.error("Directory renaming failed!")
        return

    # Step 2: Retry failed downloads
    logger.info("\nStep 2: Retrying failed downloads...")
    success_count = retry_failed_studies()

    print(f"\n{'='*60}")
    print(f"Directory organization completed!")
    print(f"Successfully re-downloaded: {success_count}/5 failed studies")
    print(f"Next step: Run metadata extraction script")

if __name__ == "__main__":
    main()