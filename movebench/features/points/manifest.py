from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from movebench.features.points.config import PointStoreConfig, stable_hash_dict
from movebench.layers.plan import LayerPlan


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PointStoreManifest:
    """Manifest for a point feature store (SSF/iSSF candidate features)."""

    schema_version: str
    study_id: str
    split: str
    created_at: str

    config_path: str | None
    config_hash: str
    store_id: str

    layer_plan: LayerPlan
    store_config: dict[str, Any]
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
            "store_id": self.store_id,
            "layer_plan": self.layer_plan.to_dict(),
            "store_config": self.store_config,
            "sampler": self.sampler,
            "extraction": self.extraction,
            "tensors": self.tensors,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PointStoreManifest":
        return cls(
            schema_version=str(d["schema_version"]),
            study_id=str(d["study_id"]),
            split=str(d["split"]),
            created_at=str(d["created_at"]),
            config_path=d.get("config_path"),
            config_hash=str(d["config_hash"]),
            store_id=str(d["store_id"]),
            layer_plan=LayerPlan.from_dict(d["layer_plan"]),
            store_config=dict(d["store_config"]),
            sampler=dict(d.get("sampler", {})),
            extraction=dict(d.get("extraction", {})),
            tensors={k: dict(v) for k, v in dict(d.get("tensors", {})).items()},
        )

    def validate_basic(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if str(self.layer_plan.config_hash) != str(self.config_hash):
            raise ValueError("Manifest config_hash disagrees with embedded layer_plan.config_hash")

        # Ensure store_config is complete enough to define identity and matches store_id.
        cfg = PointStoreConfig(
            sampler_type=self.store_config.get("sampler_type"),
            num_candidates=self.store_config.get("num_candidates"),
            random_seed=self.store_config.get("random_seed"),
            step_multiplier=self.store_config.get("step_multiplier"),
            step_q=self.store_config.get("step_q"),
            gap_q=self.store_config.get("gap_q"),
            bounds_buffer_deg=self.store_config.get("bounds_buffer_deg"),
        )
        if stable_hash_dict(cfg.to_dict()) != str(self.store_id):
            raise ValueError("Manifest store_id does not match hash(store_config)")

        if "points_per_step" not in self.extraction:
            raise ValueError("extraction.points_per_step is required")
        if "bounds_buffer_deg" not in self.extraction:
            raise ValueError("extraction.bounds_buffer_deg is required")


def write_manifest(path: Path, manifest: PointStoreManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic()
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def read_manifest(path: Path) -> PointStoreManifest:
    with open(path, "r") as f:
        d = json.load(f)
    m = PointStoreManifest.from_dict(d)
    m.validate_basic()
    return m
