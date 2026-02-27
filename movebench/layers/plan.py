"""Layer planning primitives.

A `LayerPlan` is a compact, serializable description of how a layers config maps
into model-ready feature channels.

The plan is produced once (typically by `LayerRegistry.resolve(...)`) and then
shared by:
- raster downloading code (to understand chip groups),
- feature extraction code (to build tensors with a stable channel layout),
- normalization code (to compute/apply stats by channel identity),
- dataset code (to interpret tensors on disk).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Channel:
    """Metadata for one output channel in a feature tensor.

    `band_id` must be stable across configs and ordering changes.
    Examples:
      - "era5_land.temperature_2m"
      - "dem.elevation"
      - "modis_lc.class_7"
    """

    band_id: str
    layer: str
    categorical: bool
    normalize: bool


@dataclass(frozen=True)
class LayerPlan:
    """Resolved channel layout for a layers config."""

    config_hash: str
    rasters_hash: str

    # Ordered list of layer names included in the output tensor.
    layers_in_order: list[str]

    # Expanded channels for the output tensor, in order.
    channels: list[Channel]

    # Chip group definitions used for download and chip-file layout.
    chip_groups: dict[str, dict[str, Any]]

    # Derived layers required by this plan (including intermediate derived layers
    # needed to compute output derived layers).
    derived_dependencies: dict[str, list[str]]

    def num_channels(self) -> int:
        return len(self.channels)

    def band_ids_in_order(self) -> list[str]:
        return [c.band_id for c in self.channels]

    def layer_slice(self, layer: str) -> tuple[int, int]:
        """Return (start, end) channel indices for a layer in the output tensor."""
        start = None
        end = None
        for i, ch in enumerate(self.channels):
            if ch.layer != layer:
                continue
            if start is None:
                start = i
            end = i + 1
        if start is None:
            raise KeyError(f"Layer not present in plan: {layer!r}")
        return start, end

    def band_index(self, band_id: str) -> int:
        """Return the channel index for a band_id."""
        for i, ch in enumerate(self.channels):
            if ch.band_id == band_id:
                return i
        raise KeyError(f"band_id not present in plan: {band_id!r}")

    def chip_group_for_layer(self, layer: str) -> str | None:
        """Return chip group name for a layer, or None if not in any chip group."""
        for group_name, group in self.chip_groups.items():
            if layer in group["layers"]:
                return group_name
        return None

    def categorical_layers(self) -> list[str]:
        """Return layer names that contribute categorical channels."""
        seen: set[str] = set()
        out: list[str] = []
        for ch in self.channels:
            if not ch.categorical:
                continue
            if ch.layer in seen:
                continue
            seen.add(ch.layer)
            out.append(ch.layer)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "rasters_hash": self.rasters_hash,
            "layers_in_order": list(self.layers_in_order),
            "channels": [
                {
                    "band_id": c.band_id,
                    "layer": c.layer,
                    "categorical": c.categorical,
                    "normalize": c.normalize,
                }
                for c in self.channels
            ],
            "chip_groups": {
                name: {
                    "layers": list(cg["layers"]),
                    "target_scale_m": float(cg["target_scale_m"]),
                }
                for name, cg in self.chip_groups.items()
            },
            "derived_dependencies": {k: list(v) for k, v in self.derived_dependencies.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LayerPlan":
        channels = [
            Channel(
                band_id=cd["band_id"],
                layer=cd["layer"],
                categorical=bool(cd["categorical"]),
                normalize=bool(cd["normalize"]),
            )
            for cd in d["channels"]
        ]

        chip_groups: dict[str, dict[str, Any]] = {}
        for name, cgd in d.get("chip_groups", {}).items():
            chip_groups[name] = {"layers": list(cgd["layers"]), "target_scale_m": float(cgd["target_scale_m"])}

        return cls(
            config_hash=d["config_hash"],
            rasters_hash=d["rasters_hash"],
            layers_in_order=list(d["layers_in_order"]),
            channels=channels,
            chip_groups=chip_groups,
            derived_dependencies={k: list(v) for k, v in d.get("derived_dependencies", {}).items()},
        )
