from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LayersConfig:
    layers: list[str]
    chip_groups: dict[str, dict[str, Any]]


def _canonical_dict(cfg: LayersConfig) -> dict[str, Any]:
    return {
        "layers": list(cfg.layers),
        "chip_groups": {
            name: {
                "layers": list(g["layers"]),
                "target_scale_m": float(g["target_scale_m"]),
            }
            for name, g in sorted(cfg.chip_groups.items(), key=lambda kv: kv[0])
        },
    }


def config_hash(cfg: LayersConfig) -> str:
    """Stable hash for a layers config."""
    payload = json.dumps(_canonical_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_layers_config(data: dict[str, Any]) -> LayersConfig:
    if "layers" not in data:
        raise ValueError("layers config missing required key: 'layers'")
    layers = data["layers"]
    if not isinstance(layers, list) or not all(isinstance(x, str) for x in layers):
        raise ValueError("'layers' must be a list of strings")

    chip_groups_raw = data.get("chip_groups", {})
    if not isinstance(chip_groups_raw, dict):
        raise ValueError("'chip_groups' must be a mapping")

    chip_groups: dict[str, dict[str, Any]] = {}
    for name, g in chip_groups_raw.items():
        if not isinstance(name, str):
            raise ValueError("chip_groups keys must be strings")
        if not isinstance(g, dict):
            raise ValueError(f"chip_groups.{name} must be a mapping")
        if "layers" not in g:
            raise ValueError(f"chip_groups.{name} missing required key: 'layers'")
        if "target_scale_m" not in g:
            raise ValueError(f"chip_groups.{name} missing required key: 'target_scale_m'")
        glayers = g["layers"]
        if not isinstance(glayers, list) or not all(isinstance(x, str) for x in glayers):
            raise ValueError(f"chip_groups.{name}.layers must be a list of strings")
        chip_groups[name] = {"layers": glayers, "target_scale_m": float(g["target_scale_m"])}

    return LayersConfig(layers=layers, chip_groups=chip_groups)


def load_layers_config(path: str | Path) -> LayersConfig:
    """Load a layers YAML config from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyYAML is required to load YAML configs. Install with: pip install PyYAML"
        ) from e

    with open(p, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top-level in {p}")
    return parse_layers_config(data)
