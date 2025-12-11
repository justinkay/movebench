# known issue
# compute_group_scale different for train and test

import os
import json
from pathlib import Path
import multiprocessing as mp

import pandas as pd
import ee

# Import shared components
from gee_common import (
    compute_group_scale, square_region_for_point, download_tiff,
    get_buffer_m_for_study, LON_COL, LAT_COL, TIME_COL
)


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------

# Initialize Earth Engine (high-volume endpoint)
# Run ee.Authenticate() once in another step if needed.
ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")

# Output dir
OUTPUT_DIR = Path("chips_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Defaults moved to gee_common.py
N_PROCS = 16

# Group nominal scales (meters)
BIG_GROUP_SCALE_M    = 11132.0   # Big: ERA5-Land (~11km), ERA5 (~28km upsampled a bit)
MEDIUM_GROUP_SCALE_M = 250.0     # Medium: MODIS/WorldClim ~250m target
SMALL_GROUP_SCALE_M  = 30.0      # Small: DEM 30m, DynamicWorld 10m → sampled at ~30m

# TODO: change to cmd line args
SET = 'test'
REGION_MULTIPLIER = 1.0 
if SET == 'train':
    REGION_MULTIPLIER = 1.0     # 2x 95th quantile step size
elif SET == 'test':
    REGION_MULTIPLIER = 15.0    # 30x 95th quantile
else:
    print("Set must be 'train' or 'test'")
    exit()


# -------------------------------------------------------------------------
# TEMPORAL HELPERS (nearest in time)
# -------------------------------------------------------------------------

def _nearest_hourly_image(collection_id, timestamp, band_list):
    ts = ee.Date(timestamp)
    start = ts.advance(-1, "hour")
    end   = ts.advance(1, "hour")

    col = (ee.ImageCollection(collection_id)
           .select(band_list)
           .filterDate(start, end)
           .sort("system:time_start"))
    return ee.Image(col.first())


def _nearest_modis_collection(collection_id, timestamp, days_window, select_bands):
    ts = ee.Date(timestamp)
    start = ts.advance(-days_window, "day")
    end   = ts.advance(days_window, "day")

    col = (ee.ImageCollection(collection_id)
           .select(select_bands)
           .filterDate(start, end)
           .sort("system:time_start"))
    return ee.Image(col.first())



# -------------------------------------------------------------------------
# BIG GROUP STACK: ERA5-Land + ERA5 (hourly)
# -------------------------------------------------------------------------

def make_big_stack(timestamp):
    # ERA5-Land hourly
    era5l_bands = [
        "temperature_2m",
        "dewpoint_temperature_2m",
        "total_precipitation",
        "surface_pressure",
        "u_component_of_wind_10m",
        "v_component_of_wind_10m",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
        "surface_latent_heat_flux",
        "surface_sensible_heat_flux",
    ]
    era5l_img = _nearest_hourly_image(
        "ECMWF/ERA5_LAND/HOURLY",
        timestamp,
        era5l_bands
    ).rename(["era5l_" + b for b in era5l_bands])

    # ERA5 hourly
    era5_bands = [
        "boundary_layer_height",
        "total_cloud_cover",
        "mean_sea_level_pressure",
    ]
    era5_img = _nearest_hourly_image(
        "ECMWF/ERA5/HOURLY",
        timestamp,
        era5_bands
    ).rename(["era5_" + b for b in era5_bands])

    return era5l_img.addBands(era5_img).toFloat()


# -------------------------------------------------------------------------
# MEDIUM GROUP STACK (~250m)
#   - MODIS LST (MOD11A2)
#   - MODIS NDVI, EVI (MOD13Q1)
#   - MODIS NDSI
#   - MODIS BAI
# -------------------------------------------------------------------------

def make_medium_stack(timestamp):
    ts = ee.Date(timestamp)

    # MOD11A2 LST: nearest 8-day composite
    mod11 = _nearest_modis_collection(
        "MODIS/061/MOD11A2",
        ts,
        days_window=4,
        select_bands=["LST_Day_1km", "LST_Night_1km"]
    )
    lst_day = mod11.select("LST_Day_1km").multiply(0.02)   # Kelvin
    lst_night = mod11.select("LST_Night_1km").multiply(0.02)
    lst_img = lst_day.addBands(lst_night).rename(
        ["mod11a2_LST_Day_K", "mod11a2_LST_Night_K"]
    )

    # MOD13Q1 NDVI & EVI: nearest 16-day composite
    mod13 = _nearest_modis_collection(
        "MODIS/061/MOD13Q1",
        ts,
        days_window=10,
        select_bands=["NDVI", "EVI"]
    )
    ndvi = mod13.select("NDVI").multiply(0.0001)
    evi  = mod13.select("EVI").multiply(0.0001)
    vi_img = ndvi.addBands(evi).rename(["mod13q1_NDVI", "mod13q1_EVI"])

    # MCD43A4-derived NDSI: nearest composite
    ndsi_img = _nearest_modis_collection(
        "MODIS/MCD43A4_006_NDSI",
        ts,
        days_window=10,
        select_bands=["NDSI"]
    ).select("NDSI").rename("mcd43a4_NDSI")

    # MCD43A4-derived BAI: nearest composite
    bai_img = _nearest_modis_collection(
        "MODIS/MCD43A4_006_BAI",
        ts,
        days_window=10,
        select_bands=["BAI"]
    ).select("BAI").rename("mcd43a4_BAI")

    medium_stack = (lst_img
                    .addBands(vi_img)
                    .addBands(ndsi_img)
                    .addBands(bai_img)
                    )

    return medium_stack.toFloat()


# -------------------------------------------------------------------------
# SMALL GROUP STACK (~30m)
#   - Dynamic World (nearest image)
# -------------------------------------------------------------------------

def make_small_stack(timestamp):
    ts = ee.Date(timestamp)

    dw_col = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
              .filterDate(ts.advance(-3, "day"), ts.advance(3, "day"))
              .sort("system:time_start"))
    dw_img = ee.Image(dw_col.first())

    dw_bands = [
        "label", # probably don't need this -- it's the argmax
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    dw_sel = dw_img.select(dw_bands).rename(
        ["dw_" + b for b in dw_bands]
    )

    return dw_sel.toFloat()

# -------------------------------------------------------------------------
# REQUESTS FROM PANDAS DATAFRAME
# -------------------------------------------------------------------------

def getRequests(df: pd.DataFrame, study_id):
    """
    Build a list of work items from a DataFrame with:
      - location-long
      - location-lat
      - timestamp
    Optional:
      - buffer_km
    """
    if LON_COL not in df.columns or LAT_COL not in df.columns or TIME_COL not in df.columns:
        raise ValueError(f"DataFrame must have '{LON_COL}', '{LAT_COL}', '{TIME_COL}' columns.")

    buffer_km = get_buffer_m_for_study(df) / 1000
    print("Buffer km is", buffer_km)

    items = []
    for i, row in df.iterrows():
        lon = float(row[LON_COL])
        lat = float(row[LAT_COL])
        ts = row[TIME_COL]
        
        if hasattr(ts, "isoformat"):
            ts_iso = ts.isoformat()
        else:
            ts_iso = str(ts)

        items.append({
            "point_id": int(i),
            "lon": lon,
            "lat": lat,
            "buffer_km": buffer_km,
            "timestamp": ts_iso,
            "study": study_id
        })

    return items

# -------------------------------------------------------------------------
# WORKER
# -------------------------------------------------------------------------

def getResult(idx, req):
    point_id  = req["point_id"]
    lon       = req["lon"]
    lat       = req["lat"]
    buffer_km = float(req["buffer_km"])
    ts        = req["timestamp"]
    study     = req["study"]

    print(f"[INFO] idx={idx}, point_id={point_id}, buffer_km={buffer_km}, ts={ts}")

    region = square_region_for_point(lon, lat, 
                                     REGION_MULTIPLIER*buffer_km) # scale unaffected

    # ---- BIG GROUP (11km) ----
    big_out = OUTPUT_DIR / f"{study}/point_{point_id}_big.tif"
    if not os.path.exists(big_out):
        try:
            big_img = make_big_stack(ts)
            big_scale = compute_group_scale(buffer_km, BIG_GROUP_SCALE_M)
            big_params = {
                "scale": big_scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            big_url = big_img.getDownloadURL(big_params)
            download_tiff(big_url, big_out)
        except:
            big_out = None # error

    # ---- MEDIUM GROUP (250m) ----
    med_out = OUTPUT_DIR / f"{study}/point_{point_id}_medium.tif"
    if not os.path.exists(med_out):
        try:
            med_img = make_medium_stack(ts)
            med_scale = compute_group_scale(buffer_km, MEDIUM_GROUP_SCALE_M)
            med_params = {
                "scale": med_scale,
                "region": region,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
            med_url = med_img.getDownloadURL(med_params)
            download_tiff(med_url, med_out)
        except:
            med_out = None # error

    # ---- SMALL GROUP (30m) ----
    # sml_out = OUTPUT_DIR / f"{study}/point_{point_id}_small.tif"
    # if not os.path.exists(sml_out):
    #     try:
    #         sml_img = make_small_stack(ts)
    #         sml_scale = compute_group_scale(buffer_km, SMALL_GROUP_SCALE_M)
    #         sml_params = {
    #             "scale": sml_scale,
    #             "region": region,
    #             "format": "GEO_TIFF",
    #             "crs": "EPSG:4326",
    #         }
    #         sml_url = sml_img.getDownloadURL(sml_params)
    #         download_tiff(sml_url, sml_out)
    #     except:
    #         sml_out = None # error
    sml_out = None # TODO not working

    return {
        "point_id": point_id,
        "big_file": str(big_out),
        "medium_file": str(med_out),
        "small_file": str(sml_out),
    }

# -------------------------------------------------------------------------
# MAIN - MULTITHREAD
# -------------------------------------------------------------------------

def main(df, study_id):
    manifest_path = OUTPUT_DIR / f"{study_id}_manifest_{SET}.json"
    if os.path.exists(manifest_path):
        print(study_id, "already complete (manifest present). Skipping.")
        return

    items = getRequests(df, study_id)
    print(f"[INFO] {study_id}: Total points: {len(items)}")

    with mp.Pool(N_PROCS) as pool:
        results = pool.starmap(getResult, enumerate(items))
    
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] {study_id} Done. Manifest at {manifest_path}")

if __name__ == "__main__":
    all_df = pd.read_csv('after_manual2yr_filtering_with_covs.csv')
    mammals_df = all_df[all_df.broad_taxonomic_group.str.lower() == 'terrestrial mammals']
    csv_loc = 'mbr_studies_2year_filtered_traintest/'

    # do succesful ones first
    studies = [ d for d in os.listdir('dem_out') if os.path.isdir('dem_out/'+d) ]
    studies.remove('c73c7698-2c6b-44d4-8879-fe10d7536935') # this one failed

    for study_id, study_name in zip(mammals_df.study_id.values, mammals_df.study_name.values):
        if study_id in studies:
            print("Running on", study_id, SET, REGION_MULTIPLIER)

            study_df = pd.read_csv(os.path.join(csv_loc, study_id +"_"+ study_name + "_2year_filtered_traintest.csv"))
            df = study_df[study_df.set == SET]
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            os.makedirs(os.path.join(OUTPUT_DIR, study_id), exist_ok=True)
            main(df, study_id)