"""YAML config loading."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_RASTER_PARAMS: dict[str, Any] = {
    "buffer": {
        "step_time_s": None,
        "step_multiplier": 2.0,
        "step_q": 0.95,
        "max_step_q": 0.99,
        "gap_q": 0.999,
    },
    "scale_clamp": {
        "min_pixels": 3,
        "max_pixels": 100,
    },
    "tiles": {
        "px_per_side": 350,
    },
}

_MAX_ARTIFACT_KEY_LEN = 96

# TODO refactor this to a separate ssf preparation step - not actually related to layers
@dataclass(frozen=True)
class FixedStepSpec:
    step_time_s: int
    step_multiplier: float
    step_q: float
    max_step_q: float
    gap_q: float


@dataclass(frozen=True)
class LayersConfig:
    layers: list[str]
    chip_groups: dict[str, dict[str, Any]]
    raster_params: dict[str, Any]
    step: FixedStepSpec


def _compact_name(name: str) -> str:
    parts = re.findall(r"[a-z0-9]+", str(name).lower())
    if not parts:
        return "x"
    return "".join(part[:4] for part in parts)


def _format_slug_number(value: int | float) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    value = float(value)
    if value.is_integer():
        whole = int(value)
        if -10 < whole < 10:
            return f"{whole}0"
        return str(whole)
    text = f"{value:g}"
    if "." not in text:
        return text
    whole, frac = text.split(".", 1)
    frac = frac.rstrip("0")
    if not frac:
        return whole
    if whole in {"0", "-0"}:
        return ("n" if whole.startswith("-") else "") + "0" + frac
    return f"{whole}p{frac}"


def _format_slug_duration(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _truncate_key(
    base: str, *, payload: bytes, max_len: int = _MAX_ARTIFACT_KEY_LEN
) -> str:
    if len(base) <= max_len:
        return base
    suffix = hashlib.sha256(payload).hexdigest()[:8]
    keep = max_len - len("-k") - len(suffix)
    return f"{base[:keep].rstrip('-')}-k{suffix}"


def _canonical_chip_groups(cfg: LayersConfig) -> dict[str, Any]:
    return {
        name: {
            "layers": list(g["layers"]),
            "target_scale_m": float(g["target_scale_m"]),
        }
        for name, g in sorted(cfg.chip_groups.items(), key=lambda kv: kv[0])
    }


def _canonical_raster_params(cfg: LayersConfig) -> dict[str, Any]:
    rp = cfg.raster_params
    buffer = {
        "step_time_s": int(cfg.step.step_time_s),
        "step_multiplier": float(rp["buffer"]["step_multiplier"]),
        "step_q": float(rp["buffer"]["step_q"]),
        "max_step_q": float(
            rp["buffer"].get(
                "max_step_q", DEFAULT_RASTER_PARAMS["buffer"]["max_step_q"]
            )
        ),
        "gap_q": float(rp["buffer"]["gap_q"]),
    }
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


def rasters_hash(cfg: LayersConfig) -> str:
    """Readable, bounded key for the on-disk rasters cache.

    This is intentionally computed from only the config fields that affect
    transferability of downloaded rasters across layer configs (e.g. between a
    superset config and a strict subset config).
    """
    raster_params = _canonical_raster_params(cfg)
    payload = json.dumps(raster_params, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    buffer = raster_params["buffer"]
    parts = [
        f"st{_format_slug_duration(int(buffer['step_time_s']))}",
        f"sm{_format_slug_number(buffer['step_multiplier'])}",
        f"sq{_format_slug_number(buffer['step_q'])}",
        f"msq{_format_slug_number(buffer['max_step_q'])}",
        f"gq{_format_slug_number(buffer['gap_q'])}",
    ]
    parts.extend(
        [
            f"mnp{int(raster_params['scale_clamp']['min_pixels'])}",
            f"mxp{int(raster_params['scale_clamp']['max_pixels'])}",
            f"pps{int(raster_params['tiles']['px_per_side'])}",
        ]
    )
    return _truncate_key("-".join(parts), payload=payload)


_DURATION_RE = re.compile(r"^(?P<value>\d+)(?P<unit>[smhd])$")


def _parse_duration_seconds(raw: Any, *, field_name: str) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be a string duration like '1h' or null")
    match = _DURATION_RE.fullmatch(raw.strip())
    if match is None:
        raise ValueError(
            f"{field_name} must match <int><unit> with unit in {{s,m,h,d}}, got {raw!r}"
        )

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
        raise ValueError(
            "raster_params.region_multiplier has been removed; "
            "use raster_params.buffer.step_time with step_multiplier/step_q/max_step_q"
        )

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
    max_step_q = float(buf["max_step_q"])
    gap_q = float(buf["gap_q"])
    if "min_step_time" in buf_raw or "max_step_time" in buf_raw:
        raise ValueError(
            "raster_params.buffer.min_step_time/max_step_time are no longer supported; "
            "use raster_params.buffer.step_time"
        )
    step_time_s = _parse_duration_seconds(
        buf_raw.get("step_time"),
        field_name="raster_params.buffer.step_time",
    )
    if step_time_s is None:
        raise ValueError("raster_params.buffer.step_time is required")
    if step_multiplier <= 0:
        raise ValueError(
            f"raster_params.buffer.step_multiplier must be > 0, got {step_multiplier}"
        )
    if not (0.0 <= step_q <= 1.0):
        raise ValueError(f"raster_params.buffer.step_q must be in [0,1], got {step_q}")
    if not (0.0 <= max_step_q <= 1.0):
        raise ValueError(
            f"raster_params.buffer.max_step_q must be in [0,1], got {max_step_q}"
        )
    if not (0.0 <= gap_q <= 1.0):
        raise ValueError(f"raster_params.buffer.gap_q must be in [0,1], got {gap_q}")

    clamp = {**DEFAULT_RASTER_PARAMS["scale_clamp"], **_get_section("scale_clamp")}
    min_pixels = int(clamp["min_pixels"])
    max_pixels = int(clamp["max_pixels"])
    if min_pixels <= 0:
        raise ValueError(
            f"raster_params.scale_clamp.min_pixels must be >= 1, got {min_pixels}"
        )
    if max_pixels < min_pixels:
        raise ValueError(
            f"raster_params.scale_clamp.max_pixels must be >= min_pixels ({min_pixels}), got {max_pixels}"
        )

    tiles = {**DEFAULT_RASTER_PARAMS["tiles"], **_get_section("tiles")}
    px_per_side = int(tiles["px_per_side"])
    if px_per_side <= 0:
        raise ValueError(
            f"raster_params.tiles.px_per_side must be >= 1, got {px_per_side}"
        )

    return {
        "buffer": {
            "step_time_s": int(step_time_s),
            "step_multiplier": step_multiplier,
            "step_q": step_q,
            "max_step_q": max_step_q,
            "gap_q": gap_q,
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
            raise ValueError(
                f"chip_groups.{name} missing required key: 'target_scale_m'"
            )
        glayers = g["layers"]
        if not isinstance(glayers, list) or not all(
            isinstance(x, str) for x in glayers
        ):
            raise ValueError(f"chip_groups.{name}.layers must be a list of strings")
        chip_groups[name] = {
            "layers": glayers,
            "target_scale_m": float(g["target_scale_m"]),
        }

    return LayersConfig(
        layers=layers,
        chip_groups=chip_groups,
        raster_params=raster_params,
        step=FixedStepSpec(
            step_time_s=int(raster_params["buffer"]["step_time_s"]),
            step_multiplier=float(raster_params["buffer"]["step_multiplier"]),
            step_q=float(raster_params["buffer"]["step_q"]),
            max_step_q=float(raster_params["buffer"]["max_step_q"]),
            gap_q=float(raster_params["buffer"]["gap_q"]),
        ),
    )


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
