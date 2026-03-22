"""Feature store manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class PointFeatureManifest:
    schema_version: str
    study_id: str
    split: str
    created_at: str
    rasters_hash: str
    sampler_hash: str
    sampler: dict[str, Any]
    extraction: dict[str, Any]
    shared: dict[str, dict[str, Any]]
    bands: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "split": self.split,
            "created_at": self.created_at,
            "rasters_hash": self.rasters_hash,
            "sampler_hash": self.sampler_hash,
            "sampler": self.sampler,
            "extraction": self.extraction,
            "shared": self.shared,
            "bands": self.bands,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PointFeatureManifest":
        return cls(
            schema_version=str(d["schema_version"]),
            study_id=str(d["study_id"]),
            split=str(d["split"]),
            created_at=str(d["created_at"]),
            rasters_hash=str(d["rasters_hash"]),
            sampler_hash=str(d["sampler_hash"]),
            sampler=dict(d["sampler"]),
            extraction=dict(d["extraction"]),
            shared={str(k): dict(v) for k, v in d["shared"].items()},
            bands={str(k): dict(v) for k, v in d["bands"].items()},
        )

    def validate_basic(self, *, root_dir: Path | None = None) -> None:
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if self.schema_version != "3":
            raise ValueError(
                f"Unsupported point feature manifest schema_version={self.schema_version!r} (expected '3')"
            )
        if not self.rasters_hash:
            raise ValueError("rasters_hash is required")
        if not self.sampler_hash:
            raise ValueError("sampler_hash is required")
        if "type" not in self.sampler:
            raise ValueError("sampler.type is required")
        if self.sampler["type"] not in ("empirical", "parametric"):
            raise ValueError(f"Unknown sampler.type: {self.sampler['type']!r}")
        for key in ("num_candidates", "random_seed"):
            if key not in self.sampler:
                raise ValueError(f"sampler.{key} is required")
        if self.extraction.get("step_selector") != "interval":
            raise ValueError("extraction.step_selector must be 'interval'")
        for key in ("step_time_s", "gap_q", "points_per_step", "choice_policy", "target_crs"):
            if key not in self.extraction:
                raise ValueError(f"extraction.{key} is required")
        if "points_per_step" not in self.extraction:
            raise ValueError("extraction.points_per_step is required")
        if "shared.pt" not in self.shared:
            raise ValueError("shared.pt metadata is required")

        shared_shape = self.shared["shared.pt"].get("shape")
        if not isinstance(shared_shape, list) or len(shared_shape) != 2:
            raise ValueError("shared.pt.shape must be [N, P]")

        n_steps = int(shared_shape[0])
        points_per_step = int(shared_shape[1])
        if points_per_step != int(self.extraction["points_per_step"]):
            raise ValueError("shared.pt.shape[1] must match extraction.points_per_step")

        for band_id, meta in self.bands.items():
            shape = list(meta.get("shape", []))
            if len(shape) != 2:
                raise ValueError(
                    f"Band {band_id!r} must have rank-2 shape [N,P], got {shape}"
                )
            if int(shape[0]) != n_steps or int(shape[1]) != points_per_step:
                raise ValueError(
                    f"Band {band_id!r} shape {shape} does not match shared shape [{n_steps},{points_per_step}]"
                )
            if "path" not in meta:
                raise ValueError(f"Band {band_id!r} missing path")

        if root_dir is not None:
            shared_path = root_dir / "shared.pt"
            if not shared_path.exists():
                raise FileNotFoundError(f"Missing shared.pt under {root_dir}")
            for band_id, meta in self.bands.items():
                band_path = root_dir / str(meta["path"])
                if not band_path.exists():
                    raise FileNotFoundError(
                        f"Missing band file for {band_id!r}: {band_path}"
                    )


def write_manifest(path: Path, manifest: PointFeatureManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic(root_dir=path.parent)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def read_manifest(path: Path) -> PointFeatureManifest:
    with open(path, "r") as f:
        d = json.load(f)
    m = PointFeatureManifest.from_dict(d)
    m.validate_basic(root_dir=path.parent)
    return m
