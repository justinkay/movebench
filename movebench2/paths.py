from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def repo_root() -> Path:
    # movebench2/paths.py -> repo root
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    env = os.environ.get("MOVEBENCH2_DATA_ROOT")
    if env:
        return Path(env)
    return repo_root() / "data" / "movebench2"


def studies_root() -> Path:
    return data_root() / "studies"


def study_dir(study_id: str) -> Path:
    return studies_root() / study_id


def paths_config_path() -> Path:
    env = os.environ.get("MOVEBENCH2_PATHS_CONFIG")
    if env:
        return Path(env)
    return repo_root() / "configs" / "paths.yaml"


@lru_cache(maxsize=1)
def paths_config() -> dict[str, str]:
    p = paths_config_path()
    if not p.exists():
        raise FileNotFoundError(
            f"paths config not found: {p}. Set MOVEBENCH2_PATHS_CONFIG to override."
        )
    
    import yaml

    with open(p, "r") as f:
        data = yaml.safe_load(f.read())

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at top-level in {p}")

    cfg: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, (str, Path)):
            raise ValueError(f"paths config values must be strings. Got {k!r}: {v!r}")
        cfg[k] = str(v)

    required = {"modis_grid", "modis_lc", "modis_cvf", "hii", "bioclim"}
    missing = sorted(required - set(cfg.keys()))
    if missing:
        raise ValueError(f"paths config missing required key(s): {missing}. File: {p}")

    return cfg


def get_path(key: str) -> Path:
    cfg = paths_config()
    try:
        return Path(cfg[key])
    except KeyError as e:
        available = sorted(cfg.keys())
        raise KeyError(
            f"Unknown paths key: {key!r}. Available keys: {available}. File: {paths_config_path()}"
        ) from e
