#!/usr/bin/env python3
"""
Retry downloading the 8 failed studies with longer timeouts and retry logic
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetryDownloader:
    def __init__(self, base_dir='mbr_studies'):
        self.base_dir = Path(base_dir)
        self.base_url = "https://datarepository.movebank.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True)

        # Failed studies to retry
        self.failed_studies = [
            ('https://datarepository.movebank.org/entities/datapackage/26c82518-d17c-470c-8c9a-557d1b7fbe61', 'Data from: Study "LifeTrack Mongolia Demoiselle cranes"'),
            ('https://datarepository.movebank.org/entities/datapackage/88c61e4c-e277-46ae-9762-2e2d9efbf451', 'Data from: Study "1000 Cranes. Russia. Altai."'),
            ('https://datarepository.movebank.org/entities/datapackage/dfcf5eac-b599-41dc-87e0-5646f6a68f40', 'Data from: Remote estimation of overwintering home ranges in an elusive migratory nocturnal bird'),
            ('https://datarepository.movebank.org/entities/datapackage/e19ad165-36bb-4c9c-b18f-610fc3eadafd', 'Data from: Black kites of different age and sex show similar avoidance responses to wind turbines during migration'),
            ('https://datarepository.movebank.org/entities/datapackage/b18ff4b6-6bc4-4970-9d80-52d50ea807b5', 'Data from: The homing pigeons\' olfactory map is affected by geographical barriers'),
            ('https://datarepository.movebank.org/entities/datapackage/e3343be6-d586-4c65-8b0a-cd2a063f24c9', 'Data from: Pteropus lylei primarily forages in residential areas in Kandal, Cambodia'),
            ('https://datarepository.movebank.org/entities/datapackage/dbf2f016-6a3c-4951-a738-0096df24493c', 'Data from: Commuting fruit bats beneficially modulate their flight in relation to wind'),
            ('https://datarepository.movebank.org/entities/datapackage/316cff40-b957-4ad6-9ca5-0f50253ab56a', 'Data from: Flexibility of continental navigation and migration in European mallards')
        ]

        self.completed_studies = 0

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

    def download_file_with_retry(self, url, filepath, max_retries=3):
        """Download a single file with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading (attempt {attempt+1}/{max_retries}): {url}")

                # Use longer timeout and allow redirects
                response = self.session.get(url, stream=True, timeout=120, allow_redirects=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                logger.info(f"Downloaded: {filepath}")
                return True

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # Progressive backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {url}: {e}")

        return False

    def process_study_with_retry(self, study_url, study_text, max_retries=3):
        """Process a single study with retry logic"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing study (attempt {attempt+1}/{max_retries}): {study_url}")

                # Get the study page with longer timeout
                response = self.session.get(study_url, timeout=60)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Get study name for directory
                study_name = self.get_study_name_from_page(soup)
                if not study_name or study_name == "unknown_study":
                    # Fallback to using the study text
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
                        if self.download_file_with_retry(file_url, filepath):
                            files_downloaded += 1
                        time.sleep(2)  # Be extra respectful to the server
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
                logger.info(f"✓ Retry completed: {study_name} ({files_downloaded} files)")
                return True

            except requests.RequestException as e:
                logger.warning(f"Attempt {attempt+1} failed for {study_url}: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 15  # Progressive backoff
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {study_url}: {e}")

            except Exception as e:
                logger.error(f"Unexpected error processing study {study_url}: {e}")
                break

        return False

    def retry_failed_studies(self):
        """Retry downloading all failed studies"""
        logger.info(f"Starting retry process for {len(self.failed_studies)} failed studies...")

        success_count = 0
        for study_url, study_text in self.failed_studies:
            logger.info(f"\n--- Retrying study {success_count + 1}/{len(self.failed_studies)} ---")
            if self.process_study_with_retry(study_url, study_text):
                success_count += 1

            # Extra wait between studies
            time.sleep(5)

        logger.info(f"\nRetry process completed!")
        logger.info(f"Successfully downloaded: {success_count}/{len(self.failed_studies)} studies")
        return success_count

def main():
    downloader = RetryDownloader()
    success_count = downloader.retry_failed_studies()

    print(f"Retry process completed!")
    print(f"Successfully downloaded: {success_count}/8 failed studies")
    print(f"Files saved to: {downloader.base_dir}")

if __name__ == "__main__":
    main()