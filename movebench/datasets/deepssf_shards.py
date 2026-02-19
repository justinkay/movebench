"""Shard-cache support for deepSSF training.

This module provides a minimal "WebDataset-like" tar-sharded cache format for
`DeepSSFDataset` samples:
  - spatial: (C,H,W) float16/float32
  - scalars: (4,)    float32
  - bearing: (1,)    float32
  - target:  (H,W)   uint8/float16/float32 (loaded as float32)

The goal is to avoid per-sample raster loading/reprojection during training by
materializing model-ready tensors once.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from movebench.features.manifest import utc_now_iso
from movebench.layers.plan import LayerPlan


Compression = Literal["none", "gz"]
SpatialDType = Literal["float16", "float32"]
TargetDType = Literal["uint8", "float16", "float32"]


def _stable_hash_dict(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def deepssf_cache_hash(
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
    """Stable hash for a deepSSF shard cache configuration."""
    return _stable_hash_dict(
        {
            "kind": "deepssf_shards",
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
            "schema_version": "1",
        }
    )


@dataclass(frozen=True)
class DeepSSFShardManifest:
    """Manifest for a deepSSF shard cache directory."""

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
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "study_id": self.study_id,
            "split": self.split,
            "cache_hash": self.cache_hash,
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
    def from_dict(cls, d: dict[str, Any]) -> "DeepSSFShardManifest":
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


def write_deepssf_manifest(path: Path, manifest: DeepSSFShardManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.validate_basic()
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def read_deepssf_manifest(path: Path) -> DeepSSFShardManifest:
    with open(path, "r") as f:
        d = json.load(f)
    m = DeepSSFShardManifest.from_dict(d)
    m.validate_basic()
    return m


def _as_numpy(x: torch.Tensor, *, dtype: np.dtype) -> np.ndarray:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    x_cpu = x.detach().to("cpu")
    if not x_cpu.is_contiguous():
        x_cpu = x_cpu.contiguous()
    arr = x_cpu.numpy()
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _write_npy(tar: tarfile.TarFile, name: str, arr: np.ndarray) -> None:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    data = buf.getvalue()
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = int(time.time())
    tar.addfile(ti, io.BytesIO(data))


def build_deepssf_shards(
    *,
    dataset: Any,
    out_dir: str | Path,
    config_path: str | None,
    normalization_sampler_hash: str | None,
    samples_per_shard: int = 256,
    spatial_dtype: SpatialDType = "float16",
    target_dtype: TargetDType = "uint8",
    compression: Compression = "none",
    num_workers: int = 0,
    overwrite: bool = False,
    progress: bool = True,
) -> DeepSSFShardManifest:
    """Materialize a DeepSSFDataset-like dataset into tar shards on disk.

    The `dataset` is expected to yield `(spatial, scalars, bearing, target)`.
    """
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
        raise ValueError(f"Dataset is empty (len={n}); nothing to shard")

    spatial_np_dtype = np.float16 if spatial_dtype == "float16" else np.float32
    if target_dtype == "uint8":
        target_np_dtype = np.uint8
    elif target_dtype == "float16":
        target_np_dtype = np.float16
    else:
        target_np_dtype = np.float32

    # Build a stable cache hash so the directory name can be derived from args.
    cache_hash = deepssf_cache_hash(
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

    def _identity_collate(batch: list[Any]) -> Any:
        if len(batch) != 1:
            raise ValueError(f"Expected batch_size=1; got {len(batch)}")
        return batch[0]

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_identity_collate,
        drop_last=False,
    )

    ext = ".tar.gz" if compression == "gz" else ".tar"
    tar_mode = "w:gz" if compression == "gz" else "w"

    shards: list[str] = []
    current_shard_idx = -1
    tar: tarfile.TarFile | None = None

    it = loader
    if progress:
        it = tqdm(loader, total=n, desc=f"build_deepssf_shards[{split}]", unit="samples")

    for idx, sample in enumerate(it):
        shard_idx = int(idx) // int(samples_per_shard)
        if shard_idx != current_shard_idx:
            if tar is not None:
                tar.close()
                tar = None
            shard_name = f"shard-{int(shard_idx):06d}{ext}"
            shard_path = shards_dir / shard_name
            tar = tarfile.open(shard_path, mode=tar_mode)
            shards.append(str(Path("shards") / shard_name))
            current_shard_idx = shard_idx

        if tar is None:
            raise RuntimeError("Internal error: tar writer not initialized")

        try:
            spatial, scalars, bearing, target = sample
        except Exception as e:
            raise ValueError(
                "Dataset samples must be a 4-tuple: (spatial, scalars_to_grid, bearing, target)"
            ) from e

        key = f"{int(idx):09d}"
        spatial_arr = _as_numpy(spatial, dtype=spatial_np_dtype)
        scalars_arr = _as_numpy(scalars, dtype=np.float32)
        bearing_arr = _as_numpy(bearing, dtype=np.float32)
        target_arr = _as_numpy(target, dtype=np.float32)
        if target_arr.dtype != target_np_dtype:
            target_arr = target_arr.astype(target_np_dtype, copy=False)

        _write_npy(tar, f"{key}.spatial.npy", spatial_arr)
        _write_npy(tar, f"{key}.scalars.npy", scalars_arr)
        _write_npy(tar, f"{key}.bearing.npy", bearing_arr)
        _write_npy(tar, f"{key}.target.npy", target_arr)

    if tar is not None:
        tar.close()

    C = int(plan.num_channels())
    H, W = output_hw
    manifest = DeepSSFShardManifest(
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
        shards=shards,
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
    write_deepssf_manifest(manifest_path, manifest)
    return manifest


def _read_npy(fileobj: Any) -> np.ndarray:
    data = fileobj.read()
    return np.load(io.BytesIO(data), allow_pickle=False)


def _iter_tar_samples(tar: tarfile.TarFile) -> Iterator[dict[str, np.ndarray]]:
    current_key: str | None = None
    fields: dict[str, np.ndarray] = {}

    def _emit() -> dict[str, np.ndarray] | None:
        nonlocal current_key, fields
        if current_key is None:
            return None
        if fields:
            out = fields
            current_key = None
            fields = {}
            return out
        current_key = None
        fields = {}
        return None

    for ti in tar:
        if not ti.isfile():
            continue
        name = str(ti.name)
        if not name.endswith(".npy"):
            continue
        base = name[:-4]
        if "." not in base:
            continue
        key, field = base.split(".", 1)
        if current_key is None:
            current_key = key
        if key != current_key:
            out = _emit()
            if out is not None:
                yield out
            current_key = key
        f = tar.extractfile(ti)
        if f is None:
            raise IOError(f"Failed to extract member: {name}")
        fields[field] = _read_npy(f)

    out = _emit()
    if out is not None:
        yield out


def _shuffle_buffer(it: Iterator[Any], *, bufsize: int, rng: random.Random) -> Iterator[Any]:
    bufsize = int(bufsize)
    if bufsize <= 0:
        yield from it
        return

    buf: list[Any] = []
    for x in it:
        if len(buf) < bufsize:
            buf.append(x)
            continue
        j = rng.randrange(len(buf))
        yield buf[j]
        buf[j] = x

    rng.shuffle(buf)
    yield from buf


class DeepSSFShardDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """IterableDataset over deepSSF shard tar files.

    Yields the same tuple as `DeepSSFDataset`:
      (spatial, scalars_to_grid, bearing, target)
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        shuffle_shards: bool = True,
        shuffle_buffer: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.manifest = read_deepssf_manifest(self.cache_dir / "manifest.json")
        self._shards = [self.cache_dir / s for s in self.manifest.shards]
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)

    def __len__(self) -> int:  # best-effort; DataLoader may ignore for IterableDataset
        return int(self.manifest.num_samples)

    @property
    def shards(self) -> list[Path]:
        return list(self._shards)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers, worker_seed = 0, 1, None
        else:
            worker_id, num_workers, worker_seed = int(worker_info.id), int(worker_info.num_workers), int(worker_info.seed)

        shards = list(self._shards)
        rng = random.Random(int(self.seed) + (worker_seed if worker_seed is not None else 0) + int(worker_id))
        if self.shuffle_shards:
            rng.shuffle(shards)

        # Static split of shards across workers.
        shards = shards[int(worker_id) :: int(num_workers)]

        def _samples() -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            required = ("spatial", "scalars", "bearing", "target")
            for shard_path in shards:
                with tarfile.open(shard_path, mode="r:*") as tar:
                    for rec in _iter_tar_samples(tar):
                        missing = [k for k in required if k not in rec]
                        if missing:
                            raise ValueError(f"Shard sample missing fields {missing} in {shard_path}")
                        spatial = torch.from_numpy(rec["spatial"]).to(dtype=torch.float32)
                        scalars = torch.from_numpy(rec["scalars"]).to(dtype=torch.float32)
                        bearing = torch.from_numpy(rec["bearing"]).to(dtype=torch.float32)
                        target = torch.from_numpy(rec["target"]).to(dtype=torch.float32)
                        yield spatial, scalars, bearing, target

        return _shuffle_buffer(_samples(), bufsize=int(self.shuffle_buffer), rng=rng)
