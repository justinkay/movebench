"""Feature store manifests.

Manifests make on-disk feature stores self-describing and reproducible. They
record the layer/channel mapping (via an embedded `LayerPlan`) and the sampling
procedure used to generate candidate points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from movebench2.layers.plan import LayerPlan


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PointFeatureManifest:
    """Manifest for a point feature store (SSF/iSSF candidate features).

    Fields with structured dicts:
    - `sampler`: {"type": "empirical"|"parametric", "num_candidates": int, "random_seed": int,
                 "fit_params": {...} (optional)}
    - `extraction`: {"points_per_step": int, "bounds_buffer_deg": float,
                    "choice_policy": str (optional)}
    - `tensors`: {name: {"shape": [..], "dtype": str, "description": str (optional)}}
    """

    schema_version: str
    study_id: str
    split: str
    created_at: str

    config_path: str | None
    config_hash: str
    sampler_hash: str

    layer_plan: LayerPlan
    sampler: dict[str, Any]
    extraction: dict[str, Any]

    tensors: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "split": self.split,
            "created_at": self.created_at,
            "config_path": self.config_path,
            "config_hash": self.config_hash,
            "sampler_hash": self.sampler_hash,
            "layer_plan": self.layer_plan.to_dict(),
            "sampler": self.sampler,
            "extraction": self.extraction,
            "tensors": self.tensors,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PointFeatureManifest":
        return cls(
            schema_version=d["schema_version"],
            study_id=d["study_id"],
            split=d["split"],
            created_at=d["created_at"],
            config_path=d.get("config_path"),
            config_hash=d["config_hash"],
            sampler_hash=d["sampler_hash"],
            layer_plan=LayerPlan.from_dict(d["layer_plan"]),
            sampler=dict(d["sampler"]),
            extraction=dict(d["extraction"]),
            tensors={k: dict(v) for k, v in d["tensors"].items()},
        )

    def validate_basic(self) -> None:
        """Lightweight sanity checks to fail fast on obviously bad manifests."""
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if self.layer_plan.config_hash != self.config_hash:
            raise ValueError("Manifest config_hash disagrees with embedded layer_plan.config_hash")
        band_ids = self.layer_plan.band_ids_in_order()
        if len(band_ids) != len(set(band_ids)):
            raise ValueError("Duplicate band_ids detected in layer_plan")
        if "env" in self.tensors:
            env = self.tensors["env"]
            env_shape = list(env.get("shape", []))
            if len(env_shape) != 3:
                raise ValueError(f"env tensor must be rank-3 [N,C,P], got shape={env_shape}")
            if int(env_shape[1]) != self.layer_plan.num_channels():
                raise ValueError(
                    f"env.shape[1] ({env_shape[1]}) must match num_channels ({self.layer_plan.num_channels()})"
                )
        if "type" not in self.sampler:
            raise ValueError("sampler.type is required")
        if self.sampler["type"] not in ("empirical", "parametric"):
            raise ValueError(f"Unknown sampler.type: {self.sampler['type']!r}")
        if "num_candidates" not in self.sampler:
            raise ValueError("sampler.num_candidates is required")
        if "random_seed" not in self.sampler:
            raise ValueError("sampler.random_seed is required")
        if "points_per_step" not in self.extraction:
            raise ValueError("extraction.points_per_step is required")
        if "bounds_buffer_deg" not in self.extraction:
            raise ValueError("extraction.bounds_buffer_deg is required")


def write_manifest(path: Path, manifest: PointFeatureManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic()
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def read_manifest(path: Path) -> PointFeatureManifest:
    with open(path, "r") as f:
        d = json.load(f)
    m = PointFeatureManifest.from_dict(d)
    m.validate_basic()
    return m
