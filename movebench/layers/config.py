"""YAML config loading."""

from __future__ import annotations

import hashlib
import json
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RASTER_PARAMS: dict[str, Any] = {
    "buffer": {
        "step_multiplier": 2.0,
        "step_q": 0.95,
        "gap_q": 0.999,
        "min_step_time_s": None,
        "max_step_time_s": None,
    },
    "scale_clamp": {
        "min_pixels": 3,
        "max_pixels": 100,
    },
    "tiles": {
        "px_per_side": 500,
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
    buffer = {
        "step_multiplier": float(rp["buffer"]["step_multiplier"]),
        "step_q": float(rp["buffer"]["step_q"]),
        "gap_q": float(rp["buffer"]["gap_q"]),
    }
    if rp["buffer"].get("min_step_time_s") is not None:
        buffer["min_step_time_s"] = int(rp["buffer"]["min_step_time_s"])
    if rp["buffer"].get("max_step_time_s") is not None:
        buffer["max_step_time_s"] = int(rp["buffer"]["max_step_time_s"])
    return {
        "buffer": buffer,
        "scale_clamp": {
            "min_pixels": int(rp["scale_clamp"]["min_pixels"]),
            "max_pixels": int(rp["scale_clamp"]["max_pixels"]),
        },
        "tiles": {
            "px_per_side": int(rp["tiles"]["px_per_side"]),
        },
        "region_multiplier": {
            "train": 1.0,
            "test": 1.0,
        },
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


_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")


def _parse_duration_seconds(raw: Any, *, field_name: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string duration like '1h' or null")
    match = _DURATION_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(f"{field_name} must match <int><unit> with unit in {{s,m,h,d}}, got {raw!r}")

    value = int(match.group("value"))
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0, got {raw!r}")

    unit = match.group("unit")
    multiplier = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return value * multiplier

def _parse_raster_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("'raster_params' must be a mapping")

    if "region_multiplier" in raw:
        raise ValueError("raster_params.region_multiplier has been removed; use max_step_time to control extent")

    allowed_keys = {"buffer", "scale_clamp", "tiles"}
    unexpected_keys = sorted(str(k) for k in raw.keys() if k not in allowed_keys)
    if unexpected_keys:
        raise ValueError(
            f"raster_params has unexpected key(s): {unexpected_keys}; expected only {sorted(allowed_keys)}"
        )

    def _get_section(name: str) -> dict[str, Any]:
        v = raw.get(name, {})
        if v is None:
            v = {}
        if not isinstance(v, dict):
            raise ValueError(f"raster_params.{name} must be a mapping")
        return v

    buf_raw = _get_section("buffer")
    buf = {**DEFAULT_RASTER_PARAMS["buffer"], **buf_raw}
    step_multiplier = float(buf["step_multiplier"])
    step_q = float(buf["step_q"])
    gap_q = float(buf["gap_q"])
    min_step_time_s = _parse_duration_seconds(
        buf_raw.get("min_step_time"),
        field_name="raster_params.buffer.min_step_time",
    )
    max_step_time_s = _parse_duration_seconds(
        buf_raw.get("max_step_time"),
        field_name="raster_params.buffer.max_step_time",
    )
    if min_step_time_s is not None and max_step_time_s is not None and min_step_time_s > max_step_time_s:
        raise ValueError("raster_params.buffer.min_step_time must be <= raster_params.buffer.max_step_time")
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

    return {
        "buffer": {
            "step_multiplier": step_multiplier,
            "step_q": step_q,
            "gap_q": gap_q,
            "min_step_time_s": min_step_time_s,
            "max_step_time_s": max_step_time_s,
        },
        "scale_clamp": {"min_pixels": min_pixels, "max_pixels": max_pixels},
        "tiles": {"px_per_side": px_per_side},
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
