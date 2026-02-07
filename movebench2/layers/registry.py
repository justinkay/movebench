"""Built-in layer registry for movebench2.

This is the canonical set of layer specs keyed by stable layer names.
Layer configs (YAML) refer to these keys; band lists live here in the registry.
"""

from __future__ import annotations

from movebench2.layers.config import LayersConfig, config_hash
from movebench2.layers.plan import Channel, LayerPlan
from movebench2.layers.spec import LayerSpec


class LayerRegistry:
    """Registry of all known layers and their specifications."""

    def __init__(self, layers: dict[str, LayerSpec]):
        self._layers = dict(layers)

    def names(self) -> list[str]:
        return sorted(self._layers.keys())

    def get(self, name: str) -> LayerSpec:
        try:
            return self._layers[name]
        except KeyError as e:
            raise KeyError(f"Unknown layer: {name!r}") from e

    def resolve(self, cfg: LayersConfig, *, config_path: str | None = None) -> LayerPlan:
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
            config_path=config_path,
            config_hash=config_hash(cfg),
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
