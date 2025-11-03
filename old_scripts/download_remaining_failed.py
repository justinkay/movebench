#!/usr/bin/env python3
"""
Download the 5 remaining failed studies using their found URLs
"""

import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin
from pathlib import Path
import shutil

class StudyDownloader:
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
            print(f"    Downloading: {filepath.name}")
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = filepath.stat().st_size
            print(f"    ✓ Downloaded: {filepath.name} ({file_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"    ❌ Failed to download {filepath.name}: {e}")
            return False

    def download_study(self, study_url, current_dir_name):
        """Download a complete study"""
        try:
            print(f"\n=== Downloading {current_dir_name} ===")
            print(f"URL: {study_url}")
            
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
            print(f"  Found {len(download_links)} files to download")
            
            if not download_links:
                print("  ❌ No download links found!")
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
                print(f"    ✓ Saved BibTeX citation")
                files_downloaded += 1
            
            # Save the HTML page for reference
            html_file = study_dir / "study_page.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            files_downloaded += 1
            
            print(f"  ✅ Successfully downloaded {files_downloaded} files")
            return files_downloaded > 0
            
        except Exception as e:
            print(f"  ❌ Error downloading study: {e}")
            return False

def main():
    # Define the failed studies with their found URLs
    failed_studies = [
        ('Unknown_e6e86353', 'https://datarepository.movebank.org/entities/datapackage/cd62a27e-79b2-4d8e-92de-8c5643104bfd'),
        ('Unknown_13f86496', 'https://datarepository.movebank.org/entities/datapackage/12a063ce-17f5-485c-9e93-13fa7cad4d5d'),
        ('Unknown_a3a3c141', 'https://datarepository.movebank.org/entities/datapackage/8b91498b-27ce-4c9b-9f28-104204b1ed08'),
        ('Unknown_44611e27', 'https://datarepository.movebank.org/entities/datapackage/29312801-d75d-418a-b4ce-8c5d0ad5b045'),
        ('Unknown_8d0f4088', 'https://datarepository.movebank.org/entities/datapackage/6360747f-37f5-40f1-9731-f4926387a21e')
    ]
    
    downloader = StudyDownloader()
    success_count = 0
    
    print(f"Re-downloading {len(failed_studies)} failed studies...")
    
    for study_name, study_url in failed_studies:
        if downloader.download_study(study_url, study_name):
            success_count += 1
        
        time.sleep(3)  # Be extra respectful between downloads
    
    print(f"\n{'='*80}")
    print(f"Re-download complete!")
    print(f"Successfully downloaded: {success_count}/{len(failed_studies)} studies")
    
    return success_count

if __name__ == "__main__":
    main()
