from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np


_ASSET_DIR = Path(__file__).with_name("assets")
_MAPPING_PATH = _ASSET_DIR / "moveformer_taxa.csv"
_EMBEDDINGS_PATH = _ASSET_DIR / "moveformer_taxa.npz"


@lru_cache(maxsize=1)
def _taxa() -> dict[str, tuple[np.ndarray, dict[str, str]]]:
    with _MAPPING_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(_EMBEDDINGS_PATH, allow_pickle=False) as data:
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if len(rows) != len(vectors):
        raise ValueError(
            "MoveFormer taxon mapping and embeddings have different lengths"
        )
    return {
        str(row["study_id"]): (
            vectors[idx],
            {
                "species": str(row["species"]),
                "taxon_match_rank": str(row["taxon_match_rank"]),
                "taxon_match": str(row["taxon_match"]),
            },
        )
        for idx, row in enumerate(rows)
    }


def moveformer_taxon_for_study(
    study_id: str,
) -> tuple[np.ndarray, dict[str, str]]:
    try:
        vector, metadata = _taxa()[str(study_id)]
    except KeyError as exc:
        raise KeyError(f"No MoveFormer taxon embedding for study {study_id!r}") from exc
    return vector.copy(), dict(metadata)
