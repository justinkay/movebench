#!/usr/bin/env python3
"""
Movebank Study File Downloader
Downloads all files from each study listed in the movebank_links.csv file
"""

import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import re
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StudyDownloader:
    def __init__(self, csv_file='movebank_links.csv', base_dir='mbr_studies'):
        self.csv_file = csv_file
        self.base_dir = Path(base_dir)
        self.base_url = "https://datarepository.movebank.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create base directory
        self.base_dir.mkdir(exist_ok=True)

        # Track progress
        self.completed_studies = 0
        self.total_studies = 0

    def sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove extra spaces and truncate if too long
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if len(sanitized) > 200:
            sanitized = sanitized[:200].strip()
        return sanitized

    def get_study_name_from_page(self, soup):
        """Extract a good study name from the page"""
        # Try to get the main title
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title.startswith('Data from:'):
                title = title[10:].strip()
            return self.sanitize_filename(title)

        # Fallback to any title-like element
        for tag in ['title', 'h2', 'h3']:
            elem = soup.find(tag)
            if elem:
                text = elem.get_text(strip=True)
                if len(text) > 10 and not text.startswith('Movebank'):
                    return self.sanitize_filename(text)

        return "unknown_study"

    def extract_bibtex(self, soup):
        """Extract BibTeX citation from the page"""
        # Look for BibTeX in various possible locations
        bibtex_text = None

        # Try to find BibTeX in pre tags or code blocks
        for tag in soup.find_all(['pre', 'code', 'div', 'textarea']):
            text = tag.get_text()
            if '@misc{' in text or '@article{' in text or '@dataset{' in text:
                bibtex_text = text.strip()
                break

        # If not found, look for text containing bibtex patterns
        if not bibtex_text:
            page_text = soup.get_text()
            lines = page_text.split('\n')
            for i, line in enumerate(lines):
                if '@misc{' in line or '@article{' in line or '@dataset{' in line:
                    # Collect the full bibtex entry
                    bibtex_lines = []
                    brace_count = 0
                    for j in range(i, len(lines)):
                        bibtex_lines.append(lines[j])
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count <= 0 and j > i:
                            break
                    bibtex_text = '\n'.join(bibtex_lines).strip()
                    break

        return bibtex_text

    def find_download_links(self, soup):
        """Find all downloadable files in the Files section"""
        download_links = []

        # Look for links that point to /bitstreams/.../download
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            if href and '/bitstreams/' in href and '/download' in href:
                # Get the filename from the link text or nearby elements
                filename = link.get_text(strip=True)

                # If the link text is not descriptive, look for nearby text
                if not filename or filename.lower() in ['download', 'view', '']:
                    parent = link.parent
                    if parent:
                        filename = parent.get_text(strip=True)

                # Clean up filename
                if filename:
                    filename = self.sanitize_filename(filename)
                    if not filename.endswith(('.csv', '.txt', '.pdf', '.zip', '.xlsx', '.json')):
                        # Try to infer extension from context or add .dat
                        if 'csv' in filename.lower():
                            filename += '.csv'
                        elif 'readme' in filename.lower():
                            filename += '.txt'
                        else:
                            filename += '.dat'
                else:
                    # Generate filename from URL
                    url_parts = href.split('/')
                    if len(url_parts) > 2:
                        filename = f"file_{url_parts[-2][:8]}.dat"
                    else:
                        filename = "unknown_file.dat"

                full_url = urljoin(self.base_url, href)
                download_links.append((full_url, filename))

        return download_links

    def download_file(self, url, filepath):
        """Download a single file"""
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Downloaded: {filepath}")
            return True

        except requests.RequestException as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    def process_study(self, study_url, study_text):
        """Process a single study: create directory and download files"""
        try:
            logger.info(f"Processing study: {study_url}")

            # Get the study page
            response = self.session.get(study_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Get study name for directory
            study_name = self.get_study_name_from_page(soup)
            if not study_name or study_name == "unknown_study":
                # Fallback to using the study text from CSV
                study_name = self.sanitize_filename(study_text[:100])

            # Create study directory
            study_dir = self.base_dir / study_name
            study_dir.mkdir(exist_ok=True)

            files_downloaded = 0

            # Find and download files
            download_links = self.find_download_links(soup)
            for file_url, filename in download_links:
                filepath = study_dir / filename
                if not filepath.exists():  # Skip if already downloaded
                    if self.download_file(file_url, filepath):
                        files_downloaded += 1
                    time.sleep(0.5)  # Be respectful to the server
                else:
                    logger.info(f"File already exists: {filepath}")

            # Extract and save BibTeX
            bibtex = self.extract_bibtex(soup)
            if bibtex:
                bibtex_file = study_dir / "citation.bib"
                with open(bibtex_file, 'w', encoding='utf-8') as f:
                    f.write(bibtex)
                logger.info(f"Saved BibTeX: {bibtex_file}")
                files_downloaded += 1
            else:
                logger.warning(f"No BibTeX found for: {study_url}")

            self.completed_studies += 1
            logger.info(f"Completed study {self.completed_studies}/{self.total_studies}: {study_name} ({files_downloaded} files)")

            return True

        except requests.RequestException as e:
            logger.error(f"Error processing study {study_url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error processing study {study_url}: {e}")
            return False

    def download_all_studies(self):
        """Download files from all studies in the CSV"""
        # Check if CSV file exists
        if not os.path.exists(self.csv_file):
            logger.error(f"CSV file not found: {self.csv_file}")
            return

        # Read the CSV file
        studies = []
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                studies.append((row['url'], row['text']))

        self.total_studies = len(studies)
        logger.info(f"Found {self.total_studies} studies to process")

        # Process each study
        for study_url, study_text in studies:
            self.process_study(study_url, study_text)
            time.sleep(1)  # Be respectful to the server

        logger.info(f"Completed downloading from {self.completed_studies} studies")

def main():
    downloader = StudyDownloader()
    downloader.download_all_studies()

    print(f"Download process completed!")
    print(f"Studies processed: {downloader.completed_studies}")
    print(f"Files saved to: {downloader.base_dir}")

if __name__ == "__main__":
    main()