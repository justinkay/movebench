#!/usr/bin/env python3
"""
Movebank Data Repository Link Scraper
Scrapes all dataset links from the Movebank data repository GPS datasets
"""

import requests
from bs4 import BeautifulSoup
import csv
import time
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovebankScraper:
    def __init__(self):
        self.base_url = "https://datarepository.movebank.org"
        self.search_url_template = "https://datarepository.movebank.org/search?f.has_content_in_original_bundle=true,equals&f.entityType=Datapackage,equals&spc.sf=dc.date.accessioned&spc.sd=DESC&spc.page={}&f.sensorName=GPS,equals"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.links = []

    def scrape_page(self, page_num):
        """Scrape links from a single page"""
        url = self.search_url_template.format(page_num)
        logger.info(f"Scraping page {page_num}: {url}")

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all dataset links (typically in <a> tags with specific patterns)
            # We'll look for links that point to dataset pages
            page_links = []

            # Look for dataset links that contain '/entities/datapackage/'
            url_to_link = {}  # Track URLs to avoid duplicates and keep the better text
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href and '/entities/datapackage/' in href:
                    full_url = urljoin(self.base_url, href)
                    text = link.get_text(strip=True)

                    # Skip "No Thumbnail Available" links if we already have this URL with better text
                    if full_url in url_to_link:
                        if text != "No Thumbnail Available" and url_to_link[full_url]['text'] == "No Thumbnail Available":
                            url_to_link[full_url]['text'] = text
                    else:
                        url_to_link[full_url] = {
                            'url': full_url,
                            'text': text,
                            'page': page_num
                        }

            # Convert to list and filter out URLs we already have globally
            for url, link_data in url_to_link.items():
                if url not in [l['url'] for l in self.links]:
                    page_links.append(link_data)

            logger.info(f"Found {len(page_links)} links on page {page_num}")
            self.links.extend(page_links)

            return len(page_links)

        except requests.RequestException as e:
            logger.error(f"Error scraping page {page_num}: {e}")
            return 0

    def scrape_all_pages(self, total_pages=22):
        """Scrape all pages"""
        logger.info(f"Starting to scrape {total_pages} pages...")

        for page in range(1, total_pages + 1):
            links_found = self.scrape_page(page)

            # Add a small delay to be respectful to the server
            time.sleep(1)

            if links_found == 0:
                logger.warning(f"No links found on page {page}")

        logger.info(f"Completed scraping. Total links found: {len(self.links)}")
        return self.links

    def save_to_csv(self, filename='movebank_links.csv'):
        """Save collected links to CSV file"""
        logger.info(f"Saving {len(self.links)} links to {filename}")

        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['url', 'text', 'page']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for link in self.links:
                writer.writerow(link)

        logger.info(f"Links saved to {filename}")

def main():
    scraper = MovebankScraper()

    # Scrape all 22 pages
    links = scraper.scrape_all_pages(22)

    # Save to CSV
    scraper.save_to_csv('movebank_links.csv')

    # Print summary
    print(f"Scraping completed!")
    print(f"Total links found: {len(links)}")
    print(f"Expected: 215 links")
    print(f"Links saved to: movebank_links.csv")

if __name__ == "__main__":
    main()