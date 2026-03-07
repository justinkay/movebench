"""Layer registry.

`LayerSpec` describes one layer (data source + bands + light metadata).

`LAYERS` is the built-in mapping of layer key -> `LayerSpec`.

`LayerRegistry` provides lookup and resolves a `LayersConfig` into a `LayerPlan`.

`REGISTRY` is the globally available registry preloaded with `LAYERS`, so users
can call `REGISTRY.get(...)` / `REGISTRY.resolve(...)` without any setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from movebench.layers.config import LayersConfig, config_hash, rasters_hash
from movebench.layers.plan import Channel, LayerPlan


LayerSource = Literal["gee_chip", "gee_tile", "local_hdf", "local_tiff", "derived"]


@dataclass(frozen=True)
class LayerSpec:
    """Specification for a single layer in the layer registry."""

    name: str
    source: LayerSource
    bands: list[str]
    native_scale_m: float | None = None

    # Data access metadata. These are populated for built-in layers but are not
    # required for all registries (e.g., tests).
    gee_collection: str = ""
    temporal: str = ""
    path_pattern: str = ""
    hdf_path_suffix: str = ""
    year_bounds: tuple[int, int] | None = None

    categorical: bool = False
    num_classes: int = 0
    normalize: bool = True

    derived_from: list[str] | None = None

    scale_factor: float = 1.0
    valid_range: tuple[float, float] | None = None
    clip_max: float | None = None


LAYERS: dict[str, LayerSpec] = {
    # ERA5 - weather reanalysis data
    # Low spatial resolution, high temporal resolution
    # Include hourly and daily layers
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
        native_scale_m=11132.0,
        gee_collection="ECMWF/ERA5_LAND/HOURLY",
        temporal="hourly",
    ),
    "era5": LayerSpec(
        name="era5",
        source="gee_chip",
        bands=["boundary_layer_height", "total_cloud_cover", "mean_sea_level_pressure"],
        native_scale_m=27830.0,
        gee_collection="ECMWF/ERA5/HOURLY",
        temporal="hourly",
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
        native_scale_m=11132.0,
        gee_collection="ECMWF/ERA5_LAND/DAILY_AGGR",
        temporal="daily",
    ),

    # MODIS
    "modis_lst": LayerSpec( # replaced by landsat by default
        name="modis_lst",
        source="gee_chip",
        bands=["LST_Day_1km", "LST_Night_1km"],
        native_scale_m=1000.0,
        gee_collection="MODIS/061/MOD11A2",
        temporal="8day",
        scale_factor=0.02,
    ),
    "modis_vi": LayerSpec( # replaced by landsat by default
        name="modis_vi",
        source="gee_chip",
        bands=["NDVI", "EVI"],
        native_scale_m=250.0,
        gee_collection="MODIS/061/MOD13Q1",
        temporal="16day",
        scale_factor=0.0001,
    ),
    "modis_ndsi": LayerSpec( # replaced by landsat by default
        name="modis_ndsi",
        source="gee_chip",
        bands=["NDSI"],
        native_scale_m=500.0,
        gee_collection="MODIS/MCD43A4_006_NDSI",
        temporal="daily",
    ),
    "modis_bai": LayerSpec( # replaced by landsat by default
        name="modis_bai",
        source="gee_chip",
        bands=["BAI"],
        native_scale_m=500.0,
        gee_collection="MODIS/MCD43A4_006_BAI",
        temporal="daily",
    ),
    # "modis_lc": LayerSpec( # note: implemented as a local HDF file
    #     name="modis_lc",
    #     source="local_hdf",
    #     bands=["LC_Type1"],
    #     native_scale_m=500.0,
    #     temporal="annual",
    #     path_pattern="{modis_lc}/MCD12Q1.A{year}001.h{h:02d}v{v:02d}.061.*.hdf",
    #     hdf_path_suffix=":MCD12Q1:LC_Type1",
    #     year_bounds=(2015, 2025),
    #     categorical=True,
    #     num_classes=18,
    #     normalize=False,
    # ),
    # "modis_cvf_tree": LayerSpec( # note: implemented as a local HDF file
    #     name="modis_cvf_tree",
    #     source="local_hdf",
    #     bands=["Percent_Tree_Cover"],
    #     native_scale_m=250.0,
    #     temporal="annual",
    #     path_pattern="{modis_cvf}/MOD44B.A{year}065.h{h:02d}v{v:02d}.061.*.hdf",
    #     hdf_path_suffix=":MOD44B_250m_GRID:Percent_Tree_Cover",
    #     year_bounds=(2015, 2025),
    #     valid_range=(0.0, 100.0),
    # ),
    # "modis_cvf_nontree": LayerSpec( # note: implemented as a local HDF file
    #     name="modis_cvf_nontree",
    #     source="local_hdf",
    #     bands=["Percent_NonTree_Vegetation"],
    #     native_scale_m=250.0,
    #     temporal="annual",
    #     path_pattern="{modis_cvf}/MOD44B.A{year}065.h{h:02d}v{v:02d}.061.*.hdf",
    #     hdf_path_suffix=":MOD44B_250m_GRID:Percent_NonTree_Vegetation",
    #     year_bounds=(2015, 2025),
    #     valid_range=(0.0, 100.0),
    # ),
    # "modis_cvf_nonveg": LayerSpec( # note: implemented as a local HDF file
    #     name="modis_cvf_nonveg",
    #     source="local_hdf",
    #     bands=["Percent_NonVegetated"],
    #     native_scale_m=250.0,
    #     temporal="annual",
    #     path_pattern="{modis_cvf}/MOD44B.A{year}065.h{h:02d}v{v:02d}.061.*.hdf",
    #     hdf_path_suffix=":MOD44B_250m_GRID:Percent_NonVegetated",
    #     year_bounds=(2015, 2025),
    #     valid_range=(0.0, 100.0),
    # ),
    "modis_lc": LayerSpec(
        name="modis_lc",
        source="gee_tile",
        bands=["LC_Type1"],
        native_scale_m=500.0,
        gee_collection="MODIS/061/MCD12Q1",
        temporal="annual",
        year_bounds=(2001, 2024),
        categorical=True,
        num_classes=18,
        normalize=False,
    ),
    "modis_cvf_tree": LayerSpec(
        name="modis_cvf_tree",
        source="gee_tile",
        bands=["Percent_Tree_Cover"],
        native_scale_m=250.0,
        gee_collection="MODIS/061/MOD44B",
        temporal="annual",
        year_bounds=(2000, 2024),
        valid_range=(0.0, 100.0),
    ),
    "modis_cvf_nontree": LayerSpec(
        name="modis_cvf_nontree",
        source="gee_tile",
        bands=["Percent_NonTree_Vegetation"],
        native_scale_m=250.0,
        gee_collection="MODIS/061/MOD44B",
        temporal="annual",
        year_bounds=(2000, 2024),
        valid_range=(0.0, 100.0),
    ),
    "modis_cvf_nonveg": LayerSpec(
        name="modis_cvf_nonveg",
        source="gee_tile",
        bands=["Percent_NonVegetated"],
        native_scale_m=250.0,
        gee_collection="MODIS/061/MOD44B",
        temporal="annual",
        year_bounds=(2000, 2024),
        valid_range=(0.0, 100.0),
    ),

    # Landsat and derivatives
    "landsat": LayerSpec(
        name="landsat",
        source="gee_chip",
        bands=["blue", "green", "red", "nir", "swir1", "swir2", "thermal"],
        native_scale_m=30.0,
        gee_collection="LANDSAT/COMPOSITES/C02/T1_L2_8DAY",
        temporal="8day",
    ),
    "landsat_ndwi": LayerSpec(
        name="landsat_ndwi",
        source="derived",
        bands=["NDWI"],
        native_scale_m=30.0,
        temporal="8day",
        derived_from=["landsat"],
    ),
    "landsat_ndvi": LayerSpec(
        name="landsat_ndvi",
        source="derived",
        bands=["NDVI"],
        native_scale_m=30.0,
        temporal="8day",
        derived_from=["landsat"],
    ),
    "landsat_evi": LayerSpec(
        name="landsat_evi",
        source="derived",
        bands=["EVI"],
        native_scale_m=30.0,
        temporal="8day",
        derived_from=["landsat"],
    ),
    "landsat_ndsi": LayerSpec(
        name="landsat_ndsi",
        source="derived",
        bands=["NDSI"],
        native_scale_m=30.0,
        temporal="8day",
        derived_from=["landsat"],
    ),
    "landsat_bai": LayerSpec(
        name="landsat_bai",
        source="derived",
        bands=["BAI"],
        native_scale_m=30.0,
        temporal="8day",
        derived_from=["landsat"],
    ),
    # "landsat_ndwi": LayerSpec( # Compute ourselves from the bands instead
    #     name="landsat_ndwi",
    #     source="gee_chip",
    #     bands=["NDWI"],
    #     native_scale_m=30.0,
    #     gee_collection="LANDSAT/COMPOSITES/C02/T1_L2_32DAY_NDWI",
    #     temporal="32day",
    # ),

    # Elevation and derivatives
    "dem": LayerSpec(
        name="dem",
        source="gee_tile",
        bands=["elevation"],
        native_scale_m=30.0,
        gee_collection="COPERNICUS/DEM/GLO30",
        temporal="static",
    ),
    "slope": LayerSpec(
        name="slope",
        source="derived",
        bands=["slope"],
        native_scale_m=30.0,
        temporal="static",
        derived_from=["dem"],
    ),
    "aspect": LayerSpec(
        name="aspect",
        source="derived",
        bands=["aspect_sin", "aspect_cos"],
        native_scale_m=30.0,
        temporal="static",
        derived_from=["dem"],
    ),
    "rugosity": LayerSpec(
        name="rugosity",
        source="derived",
        bands=["tri"],
        native_scale_m=30.0,
        temporal="static",
        derived_from=["dem"],
    ),
    "orographic_uplift": LayerSpec(
        name="orographic_uplift",
        source="derived",
        bands=["orographic_uplift"],
        native_scale_m=30.0,
        temporal="hourly",
        derived_from=["slope", "aspect", "era5_land"],
    ),
    "thermal_uplift": LayerSpec(
        name="thermal_uplift",
        source="derived",
        bands=["thermal_uplift"],
        native_scale_m=11132.0,
        temporal="hourly",
        derived_from=["era5_land", "era5"],
    ),

    # Other GEE features
    "dist_water": LayerSpec(
        name="dist_water",
        source="gee_tile",
        bands=["dist_water"],
        native_scale_m=30.0,
        gee_collection="JRC/GSW1_4/GlobalSurfaceWater",
        temporal="static",
        clip_max=30000.0,
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
        native_scale_m=10.0,
        gee_collection="GOOGLE/DYNAMICWORLD/V1",
        temporal="daily",
        year_bounds=(2015, 2025),
        normalize=False,
    ),
    "alphaearth_embeddings": LayerSpec(
        name="alphaearth_embeddings",
        source="gee_tile",
        bands=[f"A{i:02d}" for i in range(64)],
        native_scale_m=10.0,
        gee_collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        temporal="annual",
        normalize=False,
    ),

    # Other non-GEE features -- must download locally first
    "hii": LayerSpec(
        name="hii",
        source="local_tiff",
        bands=["hii"],
        native_scale_m=1000.0,
        temporal="annual",
        path_pattern="{hii}/hii_{year}-01-01.tif",
        year_bounds=(2015, 2020),
    ),
    "bioclim": LayerSpec(
        name="bioclim",
        source="local_tiff",
        bands=[f"bio_{i}" for i in range(1, 20)],
        native_scale_m=1000.0,
        temporal="static",
        path_pattern="{bioclim}/wc2.1_30s_bio_{band_num}.tif",
    ),
    
}


class LayerRegistry:
    """Registry of all known layers and their specifications."""

    def __init__(self, layers: dict[str, LayerSpec] | None = None):
        self._layers = dict(LAYERS if layers is None else layers)

    def names(self) -> list[str]:
        return sorted(self._layers.keys())

    def get(self, name: str) -> LayerSpec:
        try:
            return self._layers[name]
        except KeyError as e:
            raise KeyError(f"Unknown layer: {name!r}") from e

    def resolve(self, cfg: LayersConfig) -> LayerPlan:
        """Resolve a layers config into a concrete LayerPlan."""
        if len(cfg.layers) != len(set(cfg.layers)):
            dupes = [x for x in cfg.layers if cfg.layers.count(x) > 1]
            raise ValueError(f"Duplicate layer(s) in config.layers: {sorted(set(dupes))}")

        for name in cfg.layers:
            _ = self.get(name)  # validate existence

        required_layers, derived_dependencies = self._resolve_required_layers(cfg.layers)

        chip_groups_plan = self._validate_and_plan_chip_groups(cfg, required_layers)

        channels = self._build_channels(cfg.layers)

        return LayerPlan(
            config_hash=config_hash(cfg),
            rasters_hash=rasters_hash(cfg),
            layers_in_order=list(cfg.layers),
            channels=channels,
            chip_groups=chip_groups_plan,
            derived_dependencies=derived_dependencies,
        )

    def _resolve_required_layers(
        self, output_layers: list[str]
    ) -> tuple[set[str], dict[str, list[str]]]:
        required: set[str] = set(output_layers)
        derived_deps: dict[str, list[str]] = {}

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Cycle detected in derived layer dependencies at {name!r}")

            spec = self.get(name)
            if spec.source != "derived":
                visited.add(name)
                return

            if not spec.derived_from:
                raise ValueError(f"Derived layer {name!r} must define derived_from")

            visiting.add(name)
            parents = list(spec.derived_from)
            derived_deps[name] = parents
            for p in parents:
                _ = self.get(p)  # validate existence
                required.add(p)
                visit(p)
            visiting.remove(name)
            visited.add(name)

        for name in list(required):
            visit(name)

        return required, derived_deps

    def _validate_and_plan_chip_groups(
        self, cfg: LayersConfig, required_layers: set[str]
    ) -> dict[str, dict[str, object]]:
        chip_layer_to_group: dict[str, str] = {}
        for group_name, group in cfg.chip_groups.items():
            for layer in group["layers"]:
                spec = self.get(layer)
                if spec.source != "gee_chip":
                    raise ValueError(
                        f"chip_groups.{group_name} includes {layer!r} which is not a gee_chip layer"
                    )
                prev = chip_layer_to_group.get(layer)
                if prev is not None:
                    raise ValueError(
                        f"Layer {layer!r} appears in multiple chip groups: {prev!r}, {group_name!r}"
                    )
                chip_layer_to_group[layer] = group_name

        required_chip_layers = {n for n in required_layers if self.get(n).source == "gee_chip"}
        missing = sorted(required_chip_layers - set(chip_layer_to_group.keys()))
        if missing:
            raise ValueError(
                "Missing required gee_chip layers from chip_groups: "
                + ", ".join(missing)
            )

        return {
            name: {"layers": list(g["layers"]), "target_scale_m": float(g["target_scale_m"])}
            for name, g in cfg.chip_groups.items()
        }

    def _build_channels(self, layers_in_order: list[str]) -> list[Channel]:
        channels: list[Channel] = []
        seen_band_ids: set[str] = set()

        for layer_name in layers_in_order:
            spec = self.get(layer_name)

            if spec.categorical:
                if spec.num_classes <= 0:
                    raise ValueError(f"Categorical layer {layer_name!r} must set num_classes > 0")
                for i in range(spec.num_classes):
                    band_id = f"{layer_name}.class_{i}"
                    if band_id in seen_band_ids:
                        raise ValueError(f"Duplicate band_id generated: {band_id!r}")
                    seen_band_ids.add(band_id)
                    channels.append(
                        Channel(
                            band_id=band_id,
                            layer=layer_name,
                            categorical=True,
                            normalize=False,
                        )
                    )
                continue

            if not spec.bands:
                raise ValueError(f"Layer {layer_name!r} must define at least one band")
            for band in spec.bands:
                band_id = f"{layer_name}.{band}"
                if band_id in seen_band_ids:
                    raise ValueError(f"Duplicate band_id generated: {band_id!r}")
                seen_band_ids.add(band_id)
                channels.append(
                    Channel(
                        band_id=band_id,
                        layer=layer_name,
                        categorical=False,
                        normalize=bool(spec.normalize),
                    )
                )

        return channels


REGISTRY = LayerRegistry()
