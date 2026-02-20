"""WebDataset-backed cache for deepSSF training.

This cache materializes `DeepSSFDataset` samples into tar shards to avoid
per-sample raster loading/reprojection during training.

Each sample is stored as a single tar member: `__key__.npz`, containing:
  - spatial: (C,H,W) float16/float32
  - scalars: (4,)    float32
  - bearing: (1,)    float32
  - target:  (H,W)   uint8/float16/float32

On read, values are returned as float32 torch tensors in the tuple order:
  (spatial, scalars_to_grid, bearing, target)
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import webdataset as wds

from movebench.features.manifest import utc_now_iso
from movebench.layers.plan import LayerPlan


Compression = Literal["none", "gz"]
SpatialDType = Literal["float16", "float32"]
TargetDType = Literal["uint8", "float16", "float32"]


def _stable_hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def deepssf_wds_cache_hash(
    *,
    config_hash: str,
    rasters_hash: str,
    split: str,
    output_hw: tuple[int, int],
    normalize: bool,
    normalization_sampler_hash: str | None,
    step_multiplier: float,
    step_q: float,
    gap_q: float,
    samples_per_shard: int,
    spatial_dtype: SpatialDType,
    target_dtype: TargetDType,
    compression: Compression,
) -> str:
    return _stable_hash_dict(
        {
            "kind": "deepssf_wds_cache",
            "schema_version": "1",
            "config_hash": str(config_hash),
            "rasters_hash": str(rasters_hash),
            "split": str(split),
            "output_hw": [int(output_hw[0]), int(output_hw[1])],
            "normalize": bool(normalize),
            "normalization_sampler_hash": str(normalization_sampler_hash) if normalization_sampler_hash is not None else None,
            "step_multiplier": float(step_multiplier),
            "step_q": float(step_q),
            "gap_q": float(gap_q),
            "samples_per_shard": int(samples_per_shard),
            "spatial_dtype": str(spatial_dtype),
            "target_dtype": str(target_dtype),
            "compression": str(compression),
        }
    )


@dataclass(frozen=True)
class DeepSSFWDSCacheManifest:
    schema_version: str
    created_at: str

    study_id: str
    split: str
    cache_hash: str

    config_path: str | None
    layer_plan: LayerPlan

    rasters_hash: str
    normalization_sampler_hash: str | None

    output_hw: tuple[int, int]
    normalize: bool
    step_multiplier: float
    step_q: float
    gap_q: float

    pixel_size_m: float

    num_samples: int
    samples_per_shard: int
    shards: list[str]

    spatial_dtype: str
    target_dtype: str
    compression: str

    tensors: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "created_at": str(self.created_at),
            "study_id": str(self.study_id),
            "split": str(self.split),
            "cache_hash": str(self.cache_hash),
            "config_path": self.config_path,
            "config_hash": str(self.layer_plan.config_hash),
            "layer_plan": self.layer_plan.to_dict(),
            "rasters_hash": str(self.rasters_hash),
            "normalization_sampler_hash": self.normalization_sampler_hash,
            "output_hw": [int(self.output_hw[0]), int(self.output_hw[1])],
            "normalize": bool(self.normalize),
            "step_multiplier": float(self.step_multiplier),
            "step_q": float(self.step_q),
            "gap_q": float(self.gap_q),
            "pixel_size_m": float(self.pixel_size_m),
            "num_samples": int(self.num_samples),
            "samples_per_shard": int(self.samples_per_shard),
            "shards": list(self.shards),
            "spatial_dtype": str(self.spatial_dtype),
            "target_dtype": str(self.target_dtype),
            "compression": str(self.compression),
            "tensors": dict(self.tensors),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeepSSFWDSCacheManifest":
        ohw = d.get("output_hw", None)
        if not (isinstance(ohw, (list, tuple)) and len(ohw) == 2):
            raise ValueError(f"manifest.output_hw must be a 2-item list/tuple; got {ohw!r}")
        return cls(
            schema_version=str(d["schema_version"]),
            created_at=str(d["created_at"]),
            study_id=str(d["study_id"]),
            split=str(d["split"]),
            cache_hash=str(d["cache_hash"]),
            config_path=d.get("config_path"),
            layer_plan=LayerPlan.from_dict(d["layer_plan"]),
            rasters_hash=str(d["rasters_hash"]),
            normalization_sampler_hash=d.get("normalization_sampler_hash"),
            output_hw=(int(ohw[0]), int(ohw[1])),
            normalize=bool(d["normalize"]),
            step_multiplier=float(d["step_multiplier"]),
            step_q=float(d["step_q"]),
            gap_q=float(d["gap_q"]),
            pixel_size_m=float(d["pixel_size_m"]),
            num_samples=int(d["num_samples"]),
            samples_per_shard=int(d["samples_per_shard"]),
            shards=[str(s) for s in d["shards"]],
            spatial_dtype=str(d["spatial_dtype"]),
            target_dtype=str(d["target_dtype"]),
            compression=str(d["compression"]),
            tensors={str(k): dict(v) for k, v in dict(d.get("tensors", {})).items()},
        )

    def validate_basic(self) -> None:
        if str(self.schema_version) != "1":
            raise ValueError(f"Unsupported schema_version={self.schema_version!r} (expected '1').")
        if self.layer_plan.rasters_hash != str(self.rasters_hash):
            raise ValueError("Manifest rasters_hash disagrees with embedded layer_plan.rasters_hash")
        if int(self.num_samples) <= 0:
            raise ValueError(f"num_samples must be > 0; got {self.num_samples}")
        if int(self.samples_per_shard) <= 0:
            raise ValueError(f"samples_per_shard must be > 0; got {self.samples_per_shard}")
        if not self.shards:
            raise ValueError("shards list is empty")
        if float(self.pixel_size_m) <= 0:
            raise ValueError(f"pixel_size_m must be > 0; got {self.pixel_size_m}")
        C = int(self.layer_plan.num_channels())
        spatial = self.tensors.get("spatial", {})
        sshape = list(spatial.get("shape", []))
        if len(sshape) != 4 or int(sshape[1]) != int(C):
            raise ValueError(
                "spatial tensor shape must be [N,C,H,W] and match layer_plan.num_channels(). "
                f"Got shape={sshape}, num_channels={C}."
            )


def write_deepssf_wds_manifest(path: Path, manifest: DeepSSFWDSCacheManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic()
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def read_deepssf_wds_manifest(path: Path) -> DeepSSFWDSCacheManifest:
    with open(path, "r") as f:
        d = json.load(f)
    m = DeepSSFWDSCacheManifest.from_dict(d)
    m.validate_basic()
    return m


def _tensor_to_numpy(x: torch.Tensor, *, dtype: np.dtype) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    x_cpu = x.detach().to("cpu")
    if not x_cpu.is_contiguous():
        x_cpu = x_cpu.contiguous()
    arr = x_cpu.numpy()
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _encode_npz_bytes(
    *,
    spatial: np.ndarray,
    scalars: np.ndarray,
    bearing: np.ndarray,
    target: np.ndarray,
) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, spatial=spatial, scalars=scalars, bearing=bearing, target=target)
    return buf.getvalue()


def build_deepssf_wds_cache(
    dataset: Any,
    out_dir: str | Path,
    *,
    samples_per_shard: int = 256,
    spatial_dtype: SpatialDType = "float16",
    target_dtype: TargetDType = "uint8",
    compression: Compression = "none",
    seed: int = 0,
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
    config_path: str | None = None,
    normalization_sampler_hash: str | None = None,
) -> DeepSSFWDSCacheManifest:
    """Build a WebDataset cache from a DeepSSFDataset-like dataset."""
    out_dir = Path(out_dir)
    shards_dir = out_dir / "shards"
    manifest_path = out_dir / "manifest.json"

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir exists (pass overwrite=True to replace): {out_dir}")
        import shutil

        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(dataset, "chips"):
        raise TypeError("dataset must have a .chips attribute (expected DeepSSFDataset)")
    chips = dataset.chips
    plan = getattr(chips, "plan", None)
    if not isinstance(plan, LayerPlan):
        raise TypeError("dataset.chips.plan must be a LayerPlan")

    split = str(getattr(chips, "split", ""))
    if not split:
        raise ValueError("Could not determine dataset split from dataset.chips.split")

    study_dir = Path(getattr(chips, "study_dir", ""))
    study_id = study_dir.name if study_dir.name else "unknown"

    ohw = getattr(chips, "output_hw", None)
    if not (isinstance(ohw, (list, tuple)) and len(ohw) == 2):
        raise TypeError("dataset.chips.output_hw must be a (H,W) tuple")
    output_hw = (int(ohw[0]), int(ohw[1]))

    normalize = bool(getattr(chips, "normalize", False))
    step_multiplier = float(getattr(chips, "step_multiplier", 2.0))
    step_q = float(getattr(chips, "step_q", 0.95))
    gap_q = float(getattr(chips, "gap_q", 0.999))

    pixel_size_m = float(getattr(chips, "pixel_size_m"))
    if pixel_size_m <= 0:
        raise ValueError(f"Invalid pixel_size_m={pixel_size_m}")

    n = int(len(dataset))
    if n <= 0:
        raise ValueError(f"Dataset is empty (len={n}); nothing to cache")

    cache_hash = deepssf_wds_cache_hash(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        split=split,
        output_hw=output_hw,
        normalize=normalize,
        normalization_sampler_hash=normalization_sampler_hash,
        step_multiplier=step_multiplier,
        step_q=step_q,
        gap_q=gap_q,
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,
        target_dtype=target_dtype,
        compression=compression,
    )

    spatial_np_dtype = np.float16 if spatial_dtype == "float16" else np.float32
    if target_dtype == "uint8":
        target_np_dtype = np.uint8
    elif target_dtype == "float16":
        target_np_dtype = np.float16
    else:
        target_np_dtype = np.float32

    def _identity_collate(batch: list[Any]) -> Any:
        if len(batch) != 1:
            raise ValueError(f"Expected batch_size=1; got {len(batch)}")
        return batch[0]

    g = torch.Generator()
    g.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(num_workers),
        generator=g,
        collate_fn=_identity_collate,
        drop_last=False,
    )

    ext = ".tar.gz" if compression == "gz" else ".tar"
    pattern = str(shards_dir / f"shard-%06d{ext}")
    sink = wds.ShardWriter(pattern, maxcount=int(samples_per_shard))

    it: Iterable[Any] = loader
    if progress:
        it = tqdm(loader, total=n, desc=f"build_deepssf_wds_cache[{split}]", unit="samples")

    try:
        for idx, sample in enumerate(it):
            try:
                spatial, scalars, bearing, target = sample
            except Exception as e:
                raise ValueError(
                    "Dataset samples must be a 4-tuple: (spatial, scalars_to_grid, bearing, target)"
                ) from e

            spatial_arr = _tensor_to_numpy(spatial, dtype=spatial_np_dtype)
            scalars_arr = _tensor_to_numpy(scalars, dtype=np.float32)
            bearing_arr = _tensor_to_numpy(bearing, dtype=np.float32)

            target_f32 = _tensor_to_numpy(target, dtype=np.float32)
            if target_np_dtype == np.uint8:
                target_arr = (target_f32 > 0.5).astype(np.uint8, copy=False)
            else:
                target_arr = target_f32.astype(target_np_dtype, copy=False)

            npz_bytes = _encode_npz_bytes(
                spatial=spatial_arr,
                scalars=scalars_arr,
                bearing=bearing_arr,
                target=target_arr,
            )
            key = f"{int(idx):09d}"
            sink.write({"__key__": key, "npz": npz_bytes})
    finally:
        sink.close()

    shards = sorted([p for p in shards_dir.glob(f"shard-*{ext}") if p.is_file()])
    if not shards:
        raise RuntimeError(f"No shards were written under: {shards_dir}")
    rel_shards = [str(Path("shards") / p.name) for p in shards]

    C = int(plan.num_channels())
    H, W = output_hw
    manifest = DeepSSFWDSCacheManifest(
        schema_version="1",
        created_at=utc_now_iso(),
        study_id=str(study_id),
        split=str(split),
        cache_hash=str(cache_hash),
        config_path=str(config_path) if config_path is not None else None,
        layer_plan=plan,
        rasters_hash=str(plan.rasters_hash),
        normalization_sampler_hash=str(normalization_sampler_hash) if normalization_sampler_hash is not None else None,
        output_hw=(int(H), int(W)),
        normalize=bool(normalize),
        step_multiplier=float(step_multiplier),
        step_q=float(step_q),
        gap_q=float(gap_q),
        pixel_size_m=float(pixel_size_m),
        num_samples=int(n),
        samples_per_shard=int(samples_per_shard),
        shards=rel_shards,
        spatial_dtype=str(spatial_dtype),
        target_dtype=str(target_dtype),
        compression=str(compression),
        tensors={
            "spatial": {"shape": [int(n), int(C), int(H), int(W)], "dtype": str(spatial_dtype)},
            "scalars": {"shape": [int(n), 4], "dtype": "float32"},
            "bearing": {"shape": [int(n), 1], "dtype": "float32"},
            "target": {"shape": [int(n), int(H), int(W)], "dtype": str(target_dtype)},
        },
    )
    write_deepssf_wds_manifest(manifest_path, manifest)
    return manifest


def _decode_npz_sample(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = sample.get("npz")
    if b is None:
        raise KeyError("WebDataset sample missing required field 'npz'")
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected sample['npz'] to be bytes-like, got {type(b).__name__}")

    with np.load(io.BytesIO(bytes(b)), allow_pickle=False) as z:
        spatial = z["spatial"]
        scalars = z["scalars"]
        bearing = z["bearing"]
        target = z["target"]

    x1 = torch.from_numpy(spatial).to(dtype=torch.float32)
    x2 = torch.from_numpy(scalars).to(dtype=torch.float32)
    x3 = torch.from_numpy(bearing).to(dtype=torch.float32)
    y = torch.from_numpy(target).to(dtype=torch.float32)
    return x1, x2, x3, y


def open_deepssf_wds_dataset(
    cache_dir: str | Path,
    *,
    shardshuffle: int,
    shuffle_buffer: int,
    seed: int,
) -> Any:
    """Open a WebDataset pipeline yielding deepSSF sample tuples."""
    cache_dir = Path(cache_dir)
    manifest = read_deepssf_wds_manifest(cache_dir / "manifest.json")
    urls = [str(cache_dir / s) for s in manifest.shards]

    ds = wds.WebDataset(
        urls,
        shardshuffle=int(shardshuffle),
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    )
    if int(shuffle_buffer) > 0:
        ds = ds.shuffle(int(shuffle_buffer), rng=random.Random(int(seed)))
    ds = ds.map(_decode_npz_sample)
    return ds


def make_deepssf_wds_loader(
    cache_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int,
    shardshuffle: int,
    shuffle_buffer: int,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
) -> Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    ds = open_deepssf_wds_dataset(
        cache_dir,
        shardshuffle=int(shardshuffle),
        shuffle_buffer=0,  # shuffle after worker merge below for better mixing
        seed=int(seed),
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "batch_size": None,
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = wds.WebLoader(ds, **loader_kwargs)

    if int(shuffle_buffer) > 0:
        loader = loader.unbatched().shuffle(int(shuffle_buffer), rng=random.Random(int(seed))).batched(int(batch_size))
    else:
        loader = loader.batched(int(batch_size))

    return loader


def cache_dir_for_study(
    *,
    study_dir: str | Path,
    split: str,
    config_hash: str,
    cache_hash: str,
) -> Path:
    return Path(study_dir) / "cache" / "deepssf_wds" / str(split) / str(config_hash) / str(cache_hash)


def cache_size_bytes(cache_dir: str | Path) -> int:
    """Compute the on-disk size of a cache directory."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            try:
                total += int((Path(root) / name).stat().st_size)
            except FileNotFoundError:
                continue
    return int(total)


def steps_per_epoch_from_manifest(manifest: DeepSSFWDSCacheManifest, *, batch_size: int) -> int:
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    return int(math.ceil(float(manifest.num_samples) / float(b)))
