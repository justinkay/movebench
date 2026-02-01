"""Layer stack builder for loading pipeline."""

from __future__ import annotations

import numpy as np

from movebench.config import CHIP_GROUPS, LAYERS, LAYER_ORDER
from movebench.loaders import get_loader
from movebench.utils import modis_tiles_for_bounds, make_placeholder


PLACEHOLDER_LAYERS = {
    "dist_water",
    "dynamic_world",
    "landsat_ndwi",
}


class LayerTask:
    """Resolved load task for a single layer."""

    def __init__(self, name: str, source: str, chip_group: str | None, dependencies: list[str]):
        self.name = name
        self.source = source
        self.chip_group = chip_group
        self.dependencies = dependencies


class LayerPipeline:
    """Builds and executes a layer loading pipeline."""

    def __init__(self, study_dir, chip_dir):
        self.layer_order = list(LAYER_ORDER)
        self._chip_group_for_layer = self._build_chip_group_map()
        self._tasks = self._build_tasks()
        self._loaders = self._init_loaders(study_dir, chip_dir)

    def _init_loaders(self, study_dir, chip_dir) -> dict:
        loaders = {"chip_groups": {}, "layers": {}}

        for group_name, group in CHIP_GROUPS.items():
            layer_name = group["layers"][0]
            loaders["chip_groups"][group_name] = get_loader(
                layer_name,
                chip_dir=str(chip_dir),
                chip_group=group_name,
            )

        for layer_name, config in LAYERS.items():
            if config.source == "gee_tile":
                tile_dir = study_dir / "tiles" / layer_name
                index_path = tile_dir / "index.json"
                if index_path.exists():
                    loaders["layers"][layer_name] = get_loader(
                        layer_name,
                        tile_dir=str(tile_dir),
                    )
            elif config.source in {"local_hdf", "local_tiff", "derived"}:
                loaders["layers"][layer_name] = get_loader(layer_name)

        return loaders

    def _build_chip_group_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for group_name, group in CHIP_GROUPS.items():
            for layer_name in group["layers"]:
                mapping[layer_name] = group_name
        return mapping

    def _build_tasks(self) -> list[LayerTask]:
        tasks: list[LayerTask] = []
        for name in self.layer_order:
            config = LAYERS[name]
            tasks.append(
                LayerTask(
                    name=name,
                    source=config.source,
                    chip_group=self._chip_group_for_layer.get(name),
                    dependencies=config.derived_from or [],
                )
            )
        return tasks

    @property
    def tasks(self) -> list[LayerTask]:
        return list(self._tasks)

    def make_placeholder(self, layer_name: str, reference: dict) -> dict:
        config = LAYERS[layer_name]
        _, height, width = reference["array"].shape
        return make_placeholder(
            config,
            shape=(height, width),
            transform=reference["transform"],
            crs=reference["crs"],
            bounds=reference["bounds"],
        )

    def _split_chip_group(self, group_name: str, chip: dict) -> dict[str, dict]:
        results: dict[str, dict] = {}
        idx = 0
        for layer_name in CHIP_GROUPS[group_name]["layers"]:
            config = LAYERS[layer_name]
            band_count = len(config.bands)
            if chip["array"].shape[0] < idx + band_count:
                raise ValueError(
                    f"{group_name} chip has {chip['array'].shape[0]} bands, "
                    f"expected at least {idx + band_count}."
                )
            layer_array = chip["array"][idx:idx + band_count]
            results[layer_name] = {
                **chip,
                "array": layer_array,
                "bands": config.bands,
            }
            idx += band_count
        return results

    def _load_bioclim(self, loader, bounds, timestamp) -> dict:
        band_arrays = []
        band_dict = None
        for band_num in range(1, len(LAYERS["bioclim"].bands) + 1):
            band_dict = loader.load(bounds=bounds, timestamp=timestamp, band_num=band_num)
            band_arrays.append(band_dict["array"])
        stacked = np.concatenate(band_arrays, axis=0)
        return {
            **band_dict,
            "array": stacked,
            "bands": LAYERS["bioclim"].bands,
        }

    def load_all(self, context: dict) -> tuple[list[dict], list[str]]:
        """Load all layers in canonical order."""
        point_id = context["point_id"]
        timestamp = context["timestamp"]

        layers: list[dict] = []
        layer_names: list[str] = []
        loaded: dict[str, dict] = {}
        chip_cache: dict[str, dict | None] = {}
        split_cache: dict[str, dict[str, dict]] = {}

        bounds = None
        reference = None

        for task in self._tasks:
            config = LAYERS[task.name]
            layer_data: dict | None = None

            if task.source == "gee_chip":
                group_name = task.chip_group
                if group_name not in chip_cache:
                    try:
                        chip_cache[group_name] = self._loaders["chip_groups"][group_name].load(
                            point_id=point_id
                        )
                    except (FileNotFoundError, ValueError):
                        chip_cache[group_name] = None

                chip = chip_cache[group_name]
                if chip is not None:
                    if group_name not in split_cache:
                        split_cache[group_name] = self._split_chip_group(group_name, chip)
                    layer_data = split_cache[group_name].get(task.name)

            elif task.source == "gee_tile":
                loader = self._loaders["layers"].get(task.name)
                if loader is not None:
                    try:
                        layer_data = loader.load(bounds=bounds)
                    except (ValueError, FileNotFoundError):
                        layer_data = None

            elif task.source == "local_hdf":
                loader = self._loaders["layers"][task.name]
                hvs = modis_tiles_for_bounds(bounds)
                layer_data = loader.load(bounds=bounds, timestamp=timestamp, hvs=hvs)

            elif task.source == "local_tiff":
                loader = self._loaders["layers"][task.name]
                if task.name == "bioclim":
                    layer_data = self._load_bioclim(loader, bounds, timestamp)
                else:
                    layer_data = loader.load(bounds=bounds, timestamp=timestamp)

            elif task.source == "derived":
                parent_layers: dict[str, dict] = {}
                for dep_name in task.dependencies:
                    parent = loaded.get(dep_name)
                    if parent is None:
                        raise KeyError(
                            f"Missing dependency {dep_name} required for {task.name}"
                        )
                    parent_layers[dep_name] = parent
                layer_data = self._loaders["layers"][task.name].compute(parent_layers)

            if layer_data is None:
                if task.name in PLACEHOLDER_LAYERS and reference is not None:
                    layer_data = self.make_placeholder(task.name, reference)
                else:
                    raise FileNotFoundError(f"Missing data for layer {task.name}")

            layers.append(layer_data)
            layer_names.append(task.name)
            loaded[task.name] = layer_data

            if bounds is None:
                bounds = layer_data["bounds"]
            if reference is None:
                reference = layer_data

        return layers, layer_names
