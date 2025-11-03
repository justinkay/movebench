#!/usr/bin/env python3
"""
Movebank Data Repository Scraper and Downloader
===============================================

This script performs the complete scraping and downloading process:
1. Scrapes study links from the Movebank data repository
2. Downloads all files from each study
3. Saves files in organized directories with proper naming

Author: Claude Code Assistant
Date: November 2024
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import re
from urllib.parse import urljoin, urlparse
from pathlib import Path
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovebankScraper:
    def __init__(self):
        self.base_url = "https://datarepository.movebank.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def scrape_study_links(self, start_url, max_pages=25):
        """Scrape all study links from paginated search results"""
        all_links = []
        page = 1
        current_url = start_url

        while page <= max_pages:
            logger.info(f"Scraping page {page}...")

            try:
                response = self.session.get(current_url, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')

                # Find study links
                page_links = []
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href and '/entities/datapackage/' in href:
                        full_url = urljoin(self.base_url, href)
                        link_text = link.get_text(strip=True)

                        # Skip generic links
                        if link_text and link_text not in ['No Thumbnail Available', 'Download', 'View']:
                            page_links.append((full_url, link_text))

                logger.info(f"  Found {len(page_links)} study links on page {page}")
                all_links.extend(page_links)

                # Find next page link
                next_link = None
                for link in soup.find_all('a', href=True):
                    if 'next' in link.get_text().lower() or '>' in link.get_text():
                        next_link = urljoin(self.base_url, link.get('href'))
                        break

                if not next_link:
                    logger.info("No more pages found")
                    break

                current_url = next_link
                page += 1
                time.sleep(1)  # Be respectful

            except Exception as e:
                logger.error(f"Error scraping page {page}: {e}")
                break

        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for url, text in all_links:
            if url not in seen:
                seen.add(url)
                unique_links.append((url, text))

        logger.info(f"Total unique study links found: {len(unique_links)}")
        return unique_links

    def save_links_to_csv(self, links, filename="movebank_links.csv"):
        """Save scraped links to CSV file"""
        df = pd.DataFrame(links, columns=['url', 'text'])
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(links)} links to {filename}")

class StudyDownloader:
    def __init__(self, base_dir='mbr_studies'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
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
        """Download a single file with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
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
                logger.warning(f"    Attempt {attempt + 1} failed for {filepath.name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"    ❌ Failed to download {filepath.name} after {max_retries} attempts")
                    return False

    def download_study(self, study_url, study_text):
        """Download a complete study"""
        try:
            logger.info(f"\n=== Downloading Study ===")
            logger.info(f"URL: {study_url}")
            logger.info(f"Title: {study_text}")

            # Get the study page
            response = self.session.get(study_url, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Create directory name from study text
            dir_name = self.sanitize_filename(study_text)
            study_dir = self.base_dir / dir_name
            study_dir.mkdir(exist_ok=True)

            # Clear existing content
            for item in study_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

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

def main():
    """Main execution function"""
    print("=== Movebank Data Repository Scraper & Downloader ===")
    print("This script will:")
    print("1. Scrape study links from Movebank data repository")
    print("2. Download all files from each study")
    print("3. Organize files in subdirectories")
    print()

    # Step 1: Scrape links
    scraper = MovebankScraper()
    start_url = "https://datarepository.movebank.org/discover?filtertype=*&filter=&sort_by=score&order=desc"

    logger.info("Step 1: Scraping study links...")
    links = scraper.scrape_study_links(start_url)
    scraper.save_links_to_csv(links)

    if not links:
        logger.error("No links found! Exiting.")
        return

    # Step 2: Download studies
    downloader = StudyDownloader()
    logger.info(f"\nStep 2: Downloading {len(links)} studies...")

    success_count = 0
    for i, (url, text) in enumerate(links, 1):
        logger.info(f"\n--- Processing study {i}/{len(links)} ---")
        if downloader.download_study(url, text):
            success_count += 1

        # Be respectful - pause between downloads
        if i < len(links):
            time.sleep(3)

    # Final summary
    print(f"\n{'='*80}")
    print(f"Download process completed!")
    print(f"Successfully downloaded: {success_count}/{len(links)} studies")
    print(f"Files saved to: {downloader.base_dir}")
    print(f"Metadata saved to: movebank_links.csv")

if __name__ == "__main__":
    main()