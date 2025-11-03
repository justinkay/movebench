#!/usr/bin/env python3
"""
Find the original URLs for failed studies by matching titles
"""

import pandas as pd
import re

def normalize_title(title):
    """Normalize title for comparison"""
    # Remove common prefixes and clean up
    title = str(title).lower()
    title = re.sub(r'^data from[:\s]*', '', title)
    title = re.sub(r'^study[:\s]*', '', title)
    title = re.sub(r'[^a-z0-9\s]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title

def main():
    # Read both files
    df_meta = pd.read_csv('mbr_studies_metadata.csv')
    df_links = pd.read_csv('movebank_links.csv')
    
    # Get failed studies with Unknown authors
    failed_studies = df_meta[df_meta.author_list == 'Unknown']
    
    print("=== FINDING URLS FOR FAILED STUDIES ===")
    print(f"Studies with Unknown authors: {len(failed_studies)}")
    print()
    
    for _, row in failed_studies.iterrows():
        study_name = row['study_name']
        paper_title = row['paper_title']
        
        print(f"Study: {study_name}")
        print(f"Title: {paper_title}")
        
        # Normalize the title for matching
        normalized_target = normalize_title(paper_title)
        
        # Find best match in links CSV
        best_match = None
        best_score = 0
        
        for _, link_row in df_links.iterrows():
            link_text = link_row['text']
            link_url = link_row['url']
            
            normalized_link = normalize_title(link_text)
            
            # Calculate similarity (simple word overlap)
            target_words = set(normalized_target.split())
            link_words = set(normalized_link.split())
            
            if target_words and link_words:
                overlap = len(target_words & link_words)
                score = overlap / max(len(target_words), len(link_words))
                
                if score > best_score and score > 0.3:  # At least 30% similarity
                    best_match = (link_url, link_text, score)
                    best_score = score
        
        if best_match:
            url, text, score = best_match
            print(f"  ✓ Found match (score: {score:.2f})")
            print(f"    URL: {url}")
            print(f"    Original text: {text}")
        else:
            print(f"  ❌ No match found")
        
        print()

if __name__ == "__main__":
    main()
