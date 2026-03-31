"""Feature store manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_SHARED_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "candidate_lon": {"dtype": "float32", "shape": "points"},
    "candidate_lat": {"dtype": "float32", "shape": "points"},
    "step_length_m": {"dtype": "float32", "shape": "points"},
    "turn_angle_rad": {"dtype": "float32", "shape": "points"},
    "start_lon": {"dtype": "float32", "shape": []},
    "start_lat": {"dtype": "float32", "shape": []},
    "burst_id": {"dtype": "int64", "shape": []},
    "individual_id": {"dtype": "unicode", "shape": []},
    "point_id_start": {"dtype": "int64", "shape": []},
    "point_id_end": {"dtype": "int64", "shape": []},
    "timestamp_ns": {"dtype": "int64", "shape": []},
    "timestamp_end_ns": {"dtype": "int64", "shape": []},
    "actual_step_time_s": {"dtype": "float32", "shape": []},
    "step_time_error_s": {"dtype": "float32", "shape": []},
}


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
    shared: dict[str, Any]
    layers: dict[str, dict[str, Any]]
    bands: dict[str, dict[str, Any]]

    @property
    def n_rows(self) -> int:
        return int(self.shared["shape"][0])

    @property
    def points_per_step(self) -> int:
        return int(self.extraction["points_per_step"])

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
            "layers": self.layers,
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
            shared=dict(d["shared"]),
            layers={str(k): dict(v) for k, v in d["layers"].items()},
            bands={str(k): dict(v) for k, v in d["bands"].items()},
        )

    def validate_basic(self, *, root_dir: Path | None = None) -> None:
        if not self.schema_version:
            raise ValueError("schema_version is required")
        if self.schema_version != "4":
            raise ValueError(
                f"Unsupported point feature manifest schema_version={self.schema_version!r} (expected '4')"
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
        for key in (
            "step_time_s",
            "gap_q",
            "points_per_step",
            "choice_policy",
            "target_crs",
        ):
            if key not in self.extraction:
                raise ValueError(f"extraction.{key} is required")

        points_per_step = int(self.extraction["points_per_step"])
        shared_path = str(self.shared.get("path", "")).strip()
        if not shared_path:
            raise ValueError("shared.path is required")
        shared_shape = list(self.shared.get("shape", []))
        if len(shared_shape) != 1:
            raise ValueError("shared.shape must be [N]")
        if str(self.shared.get("dtype")) != "structured":
            raise ValueError("shared.dtype must be 'structured'")
        shared_fields = dict(self.shared.get("fields", {}))
        for key, spec in _SHARED_FIELD_SPECS.items():
            meta = shared_fields.get(key)
            if meta is None:
                raise ValueError(f"shared.fields missing required key {key!r}")
            if str(meta.get("dtype")) != str(spec["dtype"]):
                raise ValueError(
                    f"shared.fields[{key!r}].dtype must be {spec['dtype']!r}"
                )
            expected_shape = (
                [points_per_step] if spec["shape"] == "points" else list(spec["shape"])
            )
            if list(meta.get("shape", [])) != expected_shape:
                raise ValueError(
                    f"shared.fields[{key!r}].shape must be {expected_shape}"
                )
            if key == "individual_id" and int(meta.get("max_chars", 0)) <= 0:
                raise ValueError("shared.fields['individual_id'].max_chars must be > 0")

        if not self.layers:
            raise ValueError("layers metadata is required")
        layer_band_ids_seen: dict[str, list[str]] = {}
        for layer_name, meta in self.layers.items():
            env_path = str(meta.get("env_path", "")).strip()
            valid_path = str(meta.get("valid_path", "")).strip()
            if not env_path or not valid_path:
                raise ValueError(f"Layer {layer_name!r} missing env_path/valid_path")
            shape = list(meta.get("shape", []))
            if len(shape) != 3:
                raise ValueError(
                    f"Layer {layer_name!r} must have shape [N,K,P], got {shape}"
                )
            if int(shape[0]) != self.n_rows or int(shape[2]) != points_per_step:
                raise ValueError(
                    f"Layer {layer_name!r} shape {shape} does not match shared/extraction metadata"
                )
            if str(meta.get("env_dtype")) != "float32":
                raise ValueError(f"Layer {layer_name!r} env_dtype must be 'float32'")
            if str(meta.get("valid_dtype")) != "bool":
                raise ValueError(f"Layer {layer_name!r} valid_dtype must be 'bool'")
            band_ids = [str(band_id) for band_id in meta.get("band_ids", [])]
            if len(band_ids) != int(shape[1]):
                raise ValueError(
                    f"Layer {layer_name!r} band_ids length must match shape[1]"
                )
            layer_band_ids_seen[str(layer_name)] = band_ids
            if root_dir is not None:
                if not (root_dir / env_path).exists():
                    raise FileNotFoundError(
                        f"Missing layer env file for {layer_name!r}: {root_dir / env_path}"
                    )
                if not (root_dir / valid_path).exists():
                    raise FileNotFoundError(
                        f"Missing layer valid file for {layer_name!r}: {root_dir / valid_path}"
                    )

        if not self.bands:
            raise ValueError("bands metadata is required")
        for band_id, meta in self.bands.items():
            layer_name = str(meta.get("layer", "")).strip()
            if layer_name not in layer_band_ids_seen:
                raise ValueError(f"Band {band_id!r} references unknown layer {layer_name!r}")
            channel_idx = int(meta.get("channel_idx", -1))
            layer_band_ids = layer_band_ids_seen[layer_name]
            if channel_idx < 0 or channel_idx >= len(layer_band_ids):
                raise ValueError(
                    f"Band {band_id!r} channel_idx {channel_idx} is out of range for layer {layer_name!r}"
                )
            if layer_band_ids[channel_idx] != band_id:
                raise ValueError(
                    f"Band {band_id!r} does not match layer band_ids order for {layer_name!r}"
                )
            if "categorical" not in meta:
                raise ValueError(f"Band {band_id!r} missing categorical flag")
            if "normalize" not in meta:
                raise ValueError(f"Band {band_id!r} missing normalize flag")

        flattened_layer_bands = [
            band_id for band_ids in layer_band_ids_seen.values() for band_id in band_ids
        ]
        if set(flattened_layer_bands) != set(self.bands.keys()):
            raise ValueError("layers.band_ids and bands keys must contain the same band_ids")

        if root_dir is not None:
            if not (root_dir / shared_path).exists():
                raise FileNotFoundError(f"Missing shared payload under {root_dir}: {shared_path}")


def write_manifest(path: Path, manifest: PointFeatureManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic(root_dir=path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True)


def read_manifest(path: Path) -> PointFeatureManifest:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest = PointFeatureManifest.from_dict(payload)
    manifest.validate_basic(root_dir=path.parent)
    return manifest
