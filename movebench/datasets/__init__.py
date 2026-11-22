"""Read-only descriptions of prepared inputs; constructing them reads no samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Mapping, TypedDict

if TYPE_CHECKING:
    from movebench.datasets.deepssf_gpu import AlphaEarthCatalog
    from movebench.features.config import SamplerConfig
    from movebench.features.manifest import PointFeatureManifest
    from movebench.layers.plan import LayerPlan


class PointStoreInput(TypedDict):
    store_dir: Path
    manifest_path: Path
    manifest: PointFeatureManifest


class WDSCacheInput(TypedDict):
    cache_dir: Path
    manifest_path: Path
    manifest: dict[str, object]


class PointInputs(TypedDict):
    kind: Literal["points"]
    study_dir: Path
    trajectories_path: Path
    plan: LayerPlan
    sampler: SamplerConfig
    normalization_stats_path: Path | None
    point_stores: Mapping[str, PointStoreInput]


class WDSInputs(TypedDict):
    kind: Literal["wds"]
    study_dir: Path
    channel_indices: list[int]
    cache_manifests: Mapping[str, WDSCacheInput]


class GPUInputs(TypedDict):
    kind: Literal["gpu"]
    study_dir: Path
    trajectories_path: Path
    plan: LayerPlan
    catalog: AlphaEarthCatalog
    channel_indices: list[int]
    selected_splits: tuple[str, ...]


class SplitMetadata(TypedDict):
    study_id: str
    timescale: str
    band_ids: tuple[str, ...]
    geometry: str
    output_hw: tuple[int, int]


@dataclass(frozen=True)
class DataSplit:
    split_key: str
    metadata: SplitMetadata
    inputs: PointInputs | WDSInputs | GPUInputs


@dataclass(frozen=True)
class StudyTrainingData:
    train: DataSplit
    validation: tuple[DataSplit, ...]


class ZeroStepStudy(TypedDict):
    study_id: str
    reason: str


@dataclass(frozen=True)
class SharedTrainingData:
    studies: tuple[StudyTrainingData, ...]
    requested_study_ids: tuple[str, ...]
    zero_step_studies: tuple[ZeroStepStudy, ...]
    training_signature: str
