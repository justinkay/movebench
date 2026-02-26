"""YAML config loading."""

from __future__ import annotations

import hashlib
import json
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RASTER_PARAMS: dict[str, Any] = {
    "buffer": {
        "step_multiplier": 2.0,
        "step_q": 0.95,
        "gap_q": 0.999,
    },
    "scale_clamp": {
        "min_pixels": 3,
        "max_pixels": 100,
    },
    "tiles": {
        "px_per_side": 500,
    },
    "region_multiplier": {
        "train": 1.0,
        "test": 1.0,
    },
}


@dataclass(frozen=True)
class LayersConfig:
    layers: list[str]
    chip_groups: dict[str, dict[str, Any]]
    raster_params: dict[str, Any]

def _canonical_chip_groups(cfg: LayersConfig) -> dict[str, Any]:
    return {
        name: {
            "layers": list(g["layers"]),
            "target_scale_m": float(g["target_scale_m"]),
        }
        for name, g in sorted(cfg.chip_groups.items(), key=lambda kv: kv[0])
    }

def _canonical_dict(cfg: LayersConfig) -> dict[str, Any]:
    return {
        "layers": list(cfg.layers),
        "chip_groups": _canonical_chip_groups(cfg),
    }

def _canonical_raster_params(cfg: LayersConfig) -> dict[str, Any]:
    rp = cfg.raster_params
    region_multiplier = {
        str(k): float(v) for k, v in sorted(dict(rp["region_multiplier"]).items(), key=lambda kv: str(kv[0]))
    }
    return {
        "buffer": {
            "step_multiplier": float(rp["buffer"]["step_multiplier"]),
            "step_q": float(rp["buffer"]["step_q"]),
            "gap_q": float(rp["buffer"]["gap_q"]),
        },
        "scale_clamp": {
            "min_pixels": int(rp["scale_clamp"]["min_pixels"]),
            "max_pixels": int(rp["scale_clamp"]["max_pixels"]),
        },
        "tiles": {
            "px_per_side": int(rp["tiles"]["px_per_side"]),
        },
        "region_multiplier": region_multiplier,
    }


def config_hash(cfg: LayersConfig) -> str:
    """Stable hash for a layers config."""
    payload = json.dumps(_canonical_dict(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def rasters_hash(cfg: LayersConfig) -> str:
    """Stable hash for the on-disk rasters cache.

    This is intentionally computed from only the config fields that affect
    transferability of downloaded rasters across layer configs (e.g. between a
    superset config and a strict subset config).
    """
    payload = json.dumps(_canonical_raster_params(cfg), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def _parse_raster_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("'raster_params' must be a mapping")

    def _get_section(name: str) -> dict[str, Any]:
        v = raw.get(name, {})
        if v is None:
            v = {}
        if not isinstance(v, dict):
            raise ValueError(f"raster_params.{name} must be a mapping")
        return v

    buf = {**DEFAULT_RASTER_PARAMS["buffer"], **_get_section("buffer")}
    step_multiplier = float(buf["step_multiplier"])
    step_q = float(buf["step_q"])
    gap_q = float(buf["gap_q"])
    if step_multiplier <= 0:
        raise ValueError(f"raster_params.buffer.step_multiplier must be > 0, got {step_multiplier}")
    if not (0.0 <= step_q <= 1.0):
        raise ValueError(f"raster_params.buffer.step_q must be in [0,1], got {step_q}")
    if not (0.0 <= gap_q <= 1.0):
        raise ValueError(f"raster_params.buffer.gap_q must be in [0,1], got {gap_q}")

    clamp = {**DEFAULT_RASTER_PARAMS["scale_clamp"], **_get_section("scale_clamp")}
    min_pixels = int(clamp["min_pixels"])
    max_pixels = int(clamp["max_pixels"])
    if min_pixels <= 0:
        raise ValueError(f"raster_params.scale_clamp.min_pixels must be >= 1, got {min_pixels}")
    if max_pixels < min_pixels:
        raise ValueError(
            f"raster_params.scale_clamp.max_pixels must be >= min_pixels ({min_pixels}), got {max_pixels}"
        )

    tiles = {**DEFAULT_RASTER_PARAMS["tiles"], **_get_section("tiles")}
    px_per_side = int(tiles["px_per_side"])
    if px_per_side <= 0:
        raise ValueError(f"raster_params.tiles.px_per_side must be >= 1, got {px_per_side}")

    rm_raw = raw.get("region_multiplier", DEFAULT_RASTER_PARAMS["region_multiplier"])
    if rm_raw is None:
        rm_raw = {}
    if not isinstance(rm_raw, dict):
        raise ValueError("raster_params.region_multiplier must be a mapping")
    region_multiplier: dict[str, float] = {**DEFAULT_RASTER_PARAMS["region_multiplier"]}
    for k, v in rm_raw.items():
        if not isinstance(k, str):
            raise ValueError("raster_params.region_multiplier keys must be strings")
        region_multiplier[str(k)] = float(v)

    return {
        "buffer": {"step_multiplier": step_multiplier, "step_q": step_q, "gap_q": gap_q},
        "scale_clamp": {"min_pixels": min_pixels, "max_pixels": max_pixels},
        "tiles": {"px_per_side": px_per_side},
        "region_multiplier": region_multiplier,
    }


def parse_layers_config(data: dict[str, Any]) -> LayersConfig:
    raster_params = _parse_raster_params(data.get("raster_params"))

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

    return LayersConfig(layers=layers, chip_groups=chip_groups, raster_params=raster_params)


def load_layers_config(path: str | Path) -> LayersConfig:
    """Load a layers YAML config from disk."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with open(p, "r") as f:
        data = yaml.safe_load(f.read())

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top-level in {p}")
    return parse_layers_config(data)
