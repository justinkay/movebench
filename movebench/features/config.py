from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SamplerConfig:
    type: str
    num_candidates: int
    random_seed: int


def load_sampler_config(path: str | Path) -> SamplerConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sampler config not found: {p}")

    with open(p, "r") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"Sampler config must be a mapping/dict at the top level; got {type(data).__name__}"
        )
    if int(data.get("version", -1)) != 1:
        raise ValueError(
            f"Unsupported sampler config version={data.get('version')!r} (expected 1)"
        )

    sampler_type = str(data.get("type", "")).strip()
    if sampler_type not in {"parametric", "empirical"}:
        raise ValueError(
            f"Unsupported sampler config type={sampler_type!r}; expected 'parametric' or 'empirical'"
        )

    required = ("num_candidates", "random_seed")
    for key in required:
        if key not in data:
            raise ValueError(f"Sampler config missing required key: {key!r}")

    allowed = {"version", "type", "num_candidates", "random_seed"}
    unexpected = sorted(str(key) for key in data.keys() if key not in allowed)
    if unexpected:
        raise ValueError(
            "Sampler config has unexpected key(s): "
            f"{unexpected}. Step semantics now live in the layers config."
        )

    return SamplerConfig(
        type=sampler_type,
        num_candidates=int(data["num_candidates"]),
        random_seed=int(data["random_seed"]),
    )
