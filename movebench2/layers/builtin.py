"""Built-in layer registry for movebench2.

This is the canonical, greenfield set of layer specs keyed by stable layer names.
Layer configs (YAML) refer to these keys; band lists live here in the registry.
"""

from __future__ import annotations

from movebench2.layers.spec import LayerSpec


LAYERS: dict[str, LayerSpec] = {
    "modis_lst": LayerSpec(
        name="modis_lst",
        source="gee_chip",
        bands=["LST_Day_1km", "LST_Night_1km"],
        scale_factor=0.02,
    ),
    "modis_vi": LayerSpec(
        name="modis_vi",
        source="gee_chip",
        bands=["NDVI", "EVI"],
        scale_factor=0.0001,
    ),
    "modis_ndsi": LayerSpec(
        name="modis_ndsi",
        source="gee_chip",
        bands=["NDSI"],
    ),
    "modis_bai": LayerSpec(
        name="modis_bai",
        source="gee_chip",
        bands=["BAI"],
    ),
    "era5_land": LayerSpec(
        name="era5_land",
        source="gee_chip",
        bands=[
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
            "snow_depth",
        ],
    ),
    "era5": LayerSpec(
        name="era5",
        source="gee_chip",
        bands=["boundary_layer_height", "total_cloud_cover", "mean_sea_level_pressure"],
    ),
    "era5_land_daily": LayerSpec(
        name="era5_land_daily",
        source="gee_chip",
        bands=[
            "temperature_2m",
            "dewpoint_temperature_2m",
            "total_precipitation_sum",
            "surface_pressure",
            "u_component_of_wind_10m",
            "v_component_of_wind_10m",
            "surface_solar_radiation_downwards_sum",
            "surface_thermal_radiation_downwards_sum",
            "surface_net_solar_radiation_sum",
            "surface_net_thermal_radiation_sum",
            "surface_latent_heat_flux_sum",
            "surface_sensible_heat_flux_sum",
            "snow_depth",
        ],
    ),
    "dem": LayerSpec(
        name="dem",
        source="gee_tile",
        bands=["elevation"],
    ),
    "dist_water": LayerSpec(
        name="dist_water",
        source="gee_tile",
        bands=["dist_water"],
    ),
    "slope": LayerSpec(
        name="slope",
        source="derived",
        bands=["slope"],
        derived_from=["dem"],
    ),
    "aspect": LayerSpec(
        name="aspect",
        source="derived",
        bands=["aspect_sin", "aspect_cos"],
        derived_from=["dem"],
    ),
    "rugosity": LayerSpec(
        name="rugosity",
        source="derived",
        bands=["tri"],
        derived_from=["dem"],
    ),
    "orographic_uplift": LayerSpec(
        name="orographic_uplift",
        source="derived",
        bands=["orographic_uplift"],
        derived_from=["slope", "aspect", "era5_land"],
    ),
    "thermal_uplift": LayerSpec(
        name="thermal_uplift",
        source="derived",
        bands=["thermal_uplift"],
        derived_from=["era5_land", "era5"],
    ),
    "modis_lc": LayerSpec(
        name="modis_lc",
        source="local_hdf",
        bands=["LC_Type1"],
        categorical=True,
        num_classes=18,
        normalize=False,
    ),
    "modis_cvf_tree": LayerSpec(
        name="modis_cvf_tree",
        source="local_hdf",
        bands=["Percent_Tree_Cover"],
        valid_range=(0.0, 100.0),
    ),
    "modis_cvf_nontree": LayerSpec(
        name="modis_cvf_nontree",
        source="local_hdf",
        bands=["Percent_NonTree_Vegetation"],
        valid_range=(0.0, 100.0),
    ),
    "modis_cvf_nonveg": LayerSpec(
        name="modis_cvf_nonveg",
        source="local_hdf",
        bands=["Percent_NonVegetated"],
        valid_range=(0.0, 100.0),
    ),
    "hii": LayerSpec(
        name="hii",
        source="local_tiff",
        bands=["hii"],
    ),
    "bioclim": LayerSpec(
        name="bioclim",
        source="local_tiff",
        bands=[f"bio_{i}" for i in range(1, 20)],
    ),
    "dynamic_world": LayerSpec(
        name="dynamic_world",
        source="gee_chip",
        bands=[
            "water",
            "trees",
            "grass",
            "flooded_vegetation",
            "crops",
            "shrub_and_scrub",
            "built",
            "bare",
            "snow_and_ice",
        ],
        normalize=False,
    ),
    "landsat_ndwi": LayerSpec(
        name="landsat_ndwi",
        source="gee_chip",
        bands=["NDWI"],
    ),
}

