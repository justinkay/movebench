from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    # movebench2/paths.py -> repo root
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    env = os.environ.get("MOVEBENCH2_DATA_ROOT")
    if env:
        return Path(env)
    return '/archive/vision/beery/movebench/' # or: repo_root() / "data" / "movebench2"


def studies_root() -> Path:
    return data_root() / "studies"


def study_dir(study_id: str) -> Path:
    return studies_root() / study_id

