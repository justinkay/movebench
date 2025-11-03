#!/usr/bin/env python3
"""
Download any actually missing study to reach 215/215
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin
import csv
from pathlib import Path

class MissingStudyDownloader:
    def __init__(self, base_dir='mbr_studies'):
        self.base_dir = Path(base_dir)
        self.base_url = "https://datarepository.movebank.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def sanitize_filename(self, filename):
        """Sanitize filename for filesystem compatibility"""
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        if len(sanitized) > 200:
            sanitized = sanitized[:200].strip()
        return sanitized

    def get_study_name_from_page(self, soup):
        """Extract a good study name from the page"""
        title_elem = soup.find('h1')
        if title_elem:
            title = title_elem.get_text(strip=True)
            if title.startswith('Data from:'):
                title = title[10:].strip()
            return self.sanitize_filename(title)
        return "unknown_study"

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
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {filepath}")
            return True
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return False

    def process_study(self, study_url):
        """Process a single study"""
        try:
            print(f"Processing: {study_url}")
            response = self.session.get(study_url, timeout=60)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            study_name = self.get_study_name_from_page(soup)
            
            # Create study directory
            study_dir = self.base_dir / study_name
            if study_dir.exists():
                print(f"Directory already exists: {study_dir}")
                return False
                
            study_dir.mkdir(exist_ok=True)
            files_downloaded = 0

            # Download files
            download_links = self.find_download_links(soup)
            for file_url, filename in download_links:
                filepath = study_dir / filename
                if self.download_file(file_url, filepath):
                    files_downloaded += 1
                time.sleep(1)

            # Save BibTeX
            bibtex = self.extract_bibtex(soup)
            if bibtex:
                bibtex_file = study_dir / "citation.bib"
                with open(bibtex_file, 'w', encoding='utf-8') as f:
                    f.write(bibtex)
                print(f"Saved BibTeX: {bibtex_file}")
                files_downloaded += 1

            print(f"✓ Downloaded: {study_name} ({files_downloaded} files)")
            return True

        except Exception as e:
            print(f"Error processing {study_url}: {e}")
            return False

def main():
    # Find the first study that doesn't have a directory
    with open('movebank_links.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        downloader = MissingStudyDownloader()
        
        for row in reader:
            study_url = row['url']
            study_text = row['text']
            
            # Skip the Loading... study since we know it exists as Reindeer
            if study_text == "Loading...":
                continue
                
            # Try to find a matching directory for this study
            study_id = study_url.split('/')[-1]
            
            # Check if any directory contains files that reference this study ID
            found_match = False
            for dir_name in os.listdir('mbr_studies'):
                dir_path = Path('mbr_studies') / dir_name
                
                # Check citation.bib for study ID
                citation_file = dir_path / 'citation.bib'
                if citation_file.exists():
                    try:
                        with open(citation_file, 'r', encoding='utf-8', errors='ignore') as f:
                            if study_id in f.read():
                                found_match = True
                                break
                    except:
                        pass
                        
                # Check study_page.html for study ID
                html_file = dir_path / 'study_page.html'
                if html_file.exists():
                    try:
                        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                            if study_id in f.read():
                                found_match = True
                                break
                    except:
                        pass
            
            if not found_match:
                print(f"\n🔍 Found missing study: {study_text}")
                print(f"URL: {study_url}")
                
                if downloader.process_study(study_url):
                    print(f"\n✅ Successfully downloaded missing study!")
                    break
                else:
                    print(f"\n❌ Failed to download missing study")
    
    # Check final count
    final_count = len(os.listdir('mbr_studies'))
    print(f"\nFinal directory count: {final_count}")
    if final_count == 215:
        print("🎉 SUCCESS! Reached 215/215 studies!")
    else:
        print(f"Still at {final_count}/215 studies")

if __name__ == "__main__":
    main()
