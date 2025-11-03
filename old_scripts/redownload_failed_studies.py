#!/usr/bin/env python3
"""
Re-download studies that completely failed during initial download
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin
from pathlib import Path
import shutil

class StudyRedownloader:
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
        """Extract study name from the page"""
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
            print(f"  Downloading: {url}")
            response = self.session.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = filepath.stat().st_size
            print(f"  ✓ Downloaded: {filepath.name} ({file_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"  ❌ Failed to download {url}: {e}")
            return False

    def download_study(self, study_url, current_dir_name):
        """Download a complete study"""
        try:
            print(f"\n=== Re-downloading study ===")
            print(f"URL: {study_url}")
            print(f"Current directory: {current_dir_name}")
            
            # Get the study page
            response = self.session.get(study_url, timeout=60)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get proper study name
            study_name = self.get_study_name_from_page(soup)
            print(f"Extracted study name: {study_name}")
            
            # Use current directory (already renamed)
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
            print(f"Found {len(download_links)} files to download")
            
            if not download_links:
                print("  ❌ No download links found!")
                return False
            
            for file_url, filename in download_links:
                filepath = study_dir / filename
                if self.download_file(file_url, filepath):
                    files_downloaded += 1
                time.sleep(1)  # Be respectful to server
            
            # Save BibTeX citation
            bibtex = self.extract_bibtex(soup)
            if bibtex:
                bibtex_file = study_dir / "citation.bib"
                with open(bibtex_file, 'w', encoding='utf-8') as f:
                    f.write(bibtex)
                print(f"  ✓ Saved BibTeX: {bibtex_file}")
                files_downloaded += 1
            
            # Save the HTML page for reference
            html_file = study_dir / "study_page.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
            files_downloaded += 1
            
            print(f"✅ Successfully downloaded {files_downloaded} files")
            return files_downloaded > 0
            
        except Exception as e:
            print(f"❌ Error downloading study: {e}")
            return False

def main():
    # Find studies that need re-downloading
    df_meta = pd.read_csv('mbr_studies_metadata.csv')
    df_links = pd.read_csv('movebank_links.csv')
    
    # Get failed studies (no CSV files or empty directories)
    failed_studies = df_meta[df_meta.main_csv_file == 'No CSV file available']
    
    print(f"Found {len(failed_studies)} studies that need re-downloading")
    
    # Create URL mapping
    url_to_text = dict(zip(df_links['url'], df_links['text']))
    
    downloader = StudyRedownloader()
    success_count = 0
    
    for _, row in failed_studies.iterrows():
        study_name = row['study_name']
        paper_url = row['paper_url']
        paper_title = row['paper_title']
        
        print(f"\n{'='*80}")
        print(f"Re-downloading study: {study_name}")
        print(f"Title: {paper_title}")
        
        # Try to find the URL
        download_url = None
        
        if pd.notna(paper_url) and paper_url:
            download_url = paper_url
        else:
            # Try to find URL by matching title
            for url, text in url_to_text.items():
                if text and paper_title and text.lower() in paper_title.lower():
                    download_url = url
                    break
        
        if not download_url:
            print(f"❌ Could not find download URL for {study_name}")
            continue
        
        if downloader.download_study(download_url, study_name):
            success_count += 1
        
        time.sleep(2)  # Be extra respectful between downloads
    
    print(f"\n{'='*80}")
    print(f"Re-download complete!")
    print(f"Successfully re-downloaded: {success_count}/{len(failed_studies)} studies")
    
    return success_count

if __name__ == "__main__":
    main()
