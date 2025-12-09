import os
import json
from pathlib import Path

import pandas as pd
import ee

from gee_common import (
    compute_group_scale, square_region_for_point, download_tiff, get_buffer_m_for_study,
    union_regions,
    LON_COL, LAT_COL, TIME_COL
)


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

# Initialize Earth Engine (high-volume endpoint)
# Run ee.Authenticate() once in another step if needed.
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

# Output dir for DEM rasters
OUTPUT_DIR = Path("dem_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DEM scale (meters) - Copernicus GLO-30 is 30m native resolution
DEM_SCALE_M = 30.0


# -------------------------------------------------------------------------
# DEM IMAGE CREATION
# -------------------------------------------------------------------------

def make_dem_image():
    """
    Create DEM image from Copernicus GLO-30.

    Returns:
        ee.Image: DEM image with single band 'dem_glo30'
    """
    dem_ic = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    dem_img = dem_ic.mosaic().select("DEM").rename("dem_glo30")
    return dem_img.toFloat()


# -------------------------------------------------------------------------
# STUDY PROCESSING
# -------------------------------------------------------------------------

def process_study_dem(df, study_id, px_per_side=500, crs='EPSG:4326'):
    """
    Process a study to create DEM download regions.
    Instead of downloading individual chips, create union region(s) for DEM.

    Args:
        df: DataFrame with movement data containing LON_COL, LAT_COL columns
        study_id: Study identifier string

    Returns:
        dict: Study processing information including regions and metadata
    """
    if LON_COL not in df.columns or LAT_COL not in df.columns:
        raise ValueError(f"DataFrame must have '{LON_COL}', '{LAT_COL}' columns.")

    print(f"[INFO] Processing study {study_id}: {len(df)} points")

    # Compute buffer (meters per step) and scale (meters per pixel) for this study
    buffer_km = get_buffer_m_for_study(df) / 1000
    print(f"[INFO] Buffer km for study {study_id}: {buffer_km:.3f}")
    dem_scale = compute_group_scale(buffer_km, DEM_SCALE_M)
    print(f"[INFO] DEM scale for study {study_id}: {dem_scale:.1f}m")

    # Create one big region for all points
    print(f"[INFO] Creating regions for {len(df)} points...")
    regions = []
    for i, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        region = square_region_for_point(lon, lat, buffer_km)
        regions.append(region)
    print(f"[INFO] Creating union of {len(regions)} regions...")
    union_region = union_regions(regions)

    # get a grid of px_per_side**2 over the whole region 
    grid_size_m = dem_scale * px_per_side
    bounds = union_region.bounds(**{
        'proj': crs, 'maxError': 10
    })
    grid = bounds.coveringGrid(**{
        'proj':crs, 'scale': grid_size_m
    })
    filtered_grid = grid.filter(ee.Filter.intersects('.geo', union_region))
    tile_ids = filtered_grid.aggregate_array('system:index').getInfo()
    download_regions = [ ee.Feature(filtered_grid.toList(1, i).get(0)).geometry() for i, _ in enumerate(tile_ids) ]

    return {
        "study_id": study_id,
        "buffer_km": buffer_km,
        "dem_scale": dem_scale,
        "num_chunks": len(download_regions),
        "download_regions": download_regions,
    }


def download_dem_for_study(study_info):
    """
    Download DEM chunks for a study.

    Args:
        study_info: Dictionary from process_study_dem()

    Returns:
        dict: Download results with manifest information
    """
    study_id = study_info["study_id"]
    dem_scale = study_info["dem_scale"]
    download_regions = study_info["download_regions"]

    study_output_dir = OUTPUT_DIR / study_id
    study_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = study_output_dir / "dem_manifest.json"
    if manifest_path.exists():
        print(f"[INFO] {study_id}: DEM already downloaded (manifest exists)")
        with open(manifest_path, 'r') as f:
            return json.load(f)

    print(f"[INFO] {study_id}: Starting DEM download...")

    dem_img = make_dem_image()

    # make sure each tile has same dims; 
    # see https://spatialthoughts.com/2024/10/23/large-image-exports-gee/
    noDataValue = 0
    dem_img = dem_img.unmask(**{
        'value':noDataValue,
        'sameFootprint': False
    })

    downloaded_files = []
    chunk_info = []

    for i, region in enumerate(download_regions):
        chunk_filename = f"{study_id}_dem_chunk_{i:03d}.tif"
        chunk_path = study_output_dir / chunk_filename

        info = region.getInfo()
        coords = info['coordinates'][0]
        min_lon = min(coord[0] for coord in coords)
        max_lon = max(coord[0] for coord in coords)
        min_lat = min(coord[1] for coord in coords)
        max_lat = max(coord[1] for coord in coords)
        print("Downloading chunk at", coords)

        chunk_metadata = {
            "chunk_id": i,
            "filename": chunk_filename,
            "bounds": {
                "min_lon": min_lon,
                "max_lon": max_lon,
                "min_lat": min_lat,
                "max_lat": max_lat
            },
        }

        if chunk_path.exists():
            print(f"[INFO] {study_id}: Chunk {i:03d} already exists, skipping")
            downloaded_files.append(str(chunk_path))
            chunk_metadata["status"] = "existing"
            chunk_info.append(chunk_metadata)
            continue

        try:
            print(f"[INFO] {study_id}: Downloading DEM chunk {i+1:03d}/{len(download_regions):03d}")
            dem_params = {
                "scale": dem_scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            dem_url = dem_img.getDownloadURL(dem_params)
            download_tiff(dem_url, chunk_path)
            downloaded_files.append(str(chunk_path))
            chunk_metadata["status"] = "downloaded"
            print(f"[INFO] {study_id}: Successfully downloaded chunk {i:03d}")
        except Exception as e:
            print(f"[ERROR] {study_id}: Failed to download chunk {i:03d}: {e}")
            downloaded_files.append(None)
            chunk_metadata["status"] = "failed"
            chunk_metadata["error"] = str(e)

        chunk_info.append(chunk_metadata)

    # Save manifest
    manifest_data = {
        "study_id": study_id,
        "buffer_km": study_info["buffer_km"],
        "dem_scale": dem_scale,
        "num_chunks": len(download_regions),
        "downloaded_files": downloaded_files,
        "successful_chunks": sum(1 for f in downloaded_files if f is not None),
        "chunk_info": chunk_info,
        "processing_complete": True
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    success_rate = manifest_data["successful_chunks"] / manifest_data["num_chunks"]
    print(f"[INFO] {study_id}: DEM download complete. "
          f"{manifest_data['successful_chunks']}/{manifest_data['num_chunks']} chunks successful "
          f"({success_rate:.1%})")

    return manifest_data


# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------

def main(df, study_id):
    """
    Main function to process DEM for a study.

    Args:
        df: DataFrame with movement data
        study_id: Study identifier

    Returns:
        dict: Processing and download results
    """
    print(f"\n[INFO] ========== Processing DEM for study: {study_id} ==========")

    try:
        study_info = process_study_dem(df, study_id)
        result = download_dem_for_study(study_info)
        print(f"[INFO] {study_id}: Processing complete!")
        return result
    except Exception as e:
        print(f"[ERROR] {study_id}: Failed to process study: {e}")
        return {"study_id": study_id, "error": str(e), "processing_complete": False}


if __name__ == "__main__":
    print("[INFO] Starting DEM processing for mammal studies...")
    all_df = pd.read_csv('after_manual2yr_filtering_with_covs.csv')
    mammals_df = all_df[all_df.broad_taxonomic_group.str.lower() == 'terrestrial mammals']
    csv_loc = 'mbr_studies_2year_filtered_traintest/'
    print(f"[INFO] Found {len(mammals_df)} mammal studies to process")

    results = []
    for idx, (study_id, study_name) in enumerate(zip(mammals_df.study_id.values, mammals_df.study_name.values)):
        try:
            print(f"\n[INFO] Processing study {idx+1}/{len(mammals_df)}: {study_id}")

            study_file = os.path.join(csv_loc, f"{study_id}_{study_name}_2year_filtered_traintest.csv")
            study_df = pd.read_csv(study_file)

            # Use training data only
            train_df = study_df[study_df.set == 'train']
            train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors='coerce')
            train_df = train_df.dropna(subset=[TIME_COL])

            if len(train_df) == 0:
                print(f"[WARN] No valid training data for study {study_id}, skipping")
                results.append({"study_id": study_id, "error": "No valid training data", "processing_complete": False})
                continue

            result = main(train_df, study_id)
            results.append(result)

        except Exception as e:
            print(f"[ERROR] Failed to process study {study_id}: {e}")
            results.append({"study_id": study_id, "error": str(e), "processing_complete": False})
            continue

    # Save summary results
    summary_file = OUTPUT_DIR / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    successful_studies = sum(1 for r in results if r.get("processing_complete", False))
    print(f"\n[INFO] ========== PROCESSING COMPLETE ==========")
    print(f"[INFO] Successfully processed {successful_studies}/{len(results)} studies")
    print(f"[INFO] Summary saved to {summary_file}")