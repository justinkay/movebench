"""Point feature stores (SSF/iSSF) for MoveBench.

This package handles:
- fitting a movement sampler on the train split,
- generating candidate endpoints per step,
- sampling environmental rasters at candidate endpoints, and
- writing/reading disk-backed point stores used by datasets/models.
"""

from movebench.features.points.config import PointStoreConfig, SamplerType, stable_hash, stable_hash_dict
from movebench.features.points.candidates import CandidateSet, StepRequest, compute_candidates
from movebench.features.points.extract import extract_point_feature_stores
from movebench.features.points.manifest import PointStoreManifest, read_manifest, write_manifest
from movebench.features.points.sampler import fit_sampler, sample_pseudo_steps
from movebench.features.points.store import PointStore, load_point_store, point_store_dir, point_stores_dir, resolve_point_store_dir

__all__ = [
    "CandidateSet",
    "PointStore",
    "PointStoreConfig",
    "PointStoreManifest",
    "SamplerType",
    "StepRequest",
    "compute_candidates",
    "extract_point_feature_stores",
    "fit_sampler",
    "load_point_store",
    "point_store_dir",
    "point_stores_dir",
    "read_manifest",
    "resolve_point_store_dir",
    "sample_pseudo_steps",
    "stable_hash",
    "stable_hash_dict",
    "write_manifest",
]
