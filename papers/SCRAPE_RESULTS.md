# MoveBench Paper Scraping Results

## Overview
This document tracks the complete process of extracting environmental covariates from animal movement studies, starting from the original Movebank dataset through PDF acquisition to final covariate extraction.

## Step-by-Step Results

### Step 1: Initial Dataset
- **Starting point**: 215 studies with publications (from `mbr_studies_metadata.csv`)

### Step 2: PDF Download Attempts
- **Unpaywall download attempts**: 156 directories created in `papers/unpaywall_downloads/`
- **Successful PDF downloads**: 92 files with `.pdf` extensions
- **Download failure rate**: ~41% (64 directories empty or failed)
- **Additional sources**: 4 PDFs obtained via other methods (scihub_downloads)
- **Total PDFs acquired**: 96 files across all sources

**Why only 92 PDFs from unpaywall?**
- Many studies' publications are not open access
- Some DOIs may be invalid or not indexed by unpaywall
- Network/server issues during download process
- Some papers may be behind paywalls that unpaywall couldn't access

### Step 3: PDF Quality Assessment
- **Files claiming to be PDFs**: 92 files in unpaywall_downloads
- **Actual valid PDF documents**: 61 files (66% success rate)
- **Invalid files (HTML error pages)**: 31 files (34% failure rate)
- **Common failure**: HTML error pages or redirect pages saved with `.pdf` extensions

**File type breakdown**:
- 61 legitimate PDF documents
- 27 HTML documents (error/redirect pages)
- 3 files with various corruption issues
- 1 file too large for processing

### Step 4: Environmental Covariate Extraction
- **Successfully processed studies**: 51 out of 61 valid PDFs
- **Processing success rate**: 83.6% of valid PDFs
- **Studies with environmental covariates**: ~30-35 studies (exact count TBD)
- **Studies with no environmental covariates**: ~15-20 studies (marked as "NO_COVARIATES")

**Processing failures (10 studies)**:
- 2 files failed due to API size limits (PDFs too large)
- 3 files had PDF validation issues despite being valid PDFs
- 5 files encountered other processing errors

### Step 5: Final Results Summary
- **Total studies analyzed**: 51/215 (23.7% of original dataset)
- **Successfully extracted covariates**: Available for 51 studies
- **Coverage by source**:
  - Unpaywall downloads: 50 studies processed
  - Other sources: 1 study processed

## Detailed Breakdown by Category

### Environmental Covariates Found
Examples of extracted covariates include:
- **Weather variables**: wind speed/direction, temperature, precipitation, atmospheric pressure
- **Vegetation indices**: NDVI (various temporal resolutions)
- **Terrain features**: elevation, slope, terrain ruggedness
- **Atmospheric conditions**: solar radiation, thermal uplift, cloud cover
- **Habitat classifications**: land cover types, forest cover percentages
- **Oceanographic**: sea surface temperature, chlorophyll-a concentration
- **Human infrastructure**: distance to roads, land use types

### Data Quality Issues Encountered

1. **Download Issues (64/156 = 41% failure)**:
   - Papers not open access
   - Invalid/broken DOIs
   - Network timeouts
   - Publisher restrictions

2. **File Format Issues (31/92 = 34% of downloads)**:
   - HTML error pages with `.pdf` extensions
   - Redirect pages instead of actual papers
   - Corrupted downloads

3. **Processing Issues (10/61 = 16% of valid PDFs)**:
   - Files too large for API processing
   - PDF validation failures
   - Encoding or formatting issues

## Recommendations for Future Work

1. **Improve PDF acquisition**:
   - Try multiple sources (PubMed, arXiv, institutional repositories)
   - Implement retry mechanisms for failed downloads
   - Add file validation before saving

2. **Enhance processing robustness**:
   - Implement file size checks and chunking for large PDFs
   - Add fallback processing methods for edge cases
   - Improve error handling and recovery

3. **Scale coverage**:
   - Process the remaining ~160 studies with alternative download methods
   - Consider manual acquisition for high-value papers
   - Expand to additional movement databases beyond Movebank

## Files Generated
- `covariates.csv`: 51 files containing extracted environmental covariates
- `SCRAPE_RESULTS.md`: This summary document
- Individual CSV files in each processed study directory

## Data Accessibility
- **Final coverage**: 23.7% of target studies successfully processed
- **Data location**: `papers/unpaywall_downloads/*/covariates.csv`
- **Format**: CSV files with columns: `covariate_name,spatial_resolution,temporal_resolution`
- **Special values**: "NO_COVARIATES" for studies without environmental variables, "N/A" for missing resolution information