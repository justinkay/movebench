"""WebDataset-backed cache for deepSSF training.

This cache materializes `DeepSSFDataset` samples into tar shards to avoid
per-sample raster loading/reprojection during training.

Each sample is stored as a single tar member: `__key__.pth`, produced via
`webdataset.torch_dumps` and decoded via `webdataset.torch_loads`. The stored
object is a 4-tuple:
  - spatial: (C,H,W) torch.float16/torch.float32
  - scalars: (4,)    torch.float32
  - bearing: (1,)    torch.float32
  - target:  (H,W)   torch.uint8/torch.float16/torch.float32

On read, values are returned as float32 torch tensors in the tuple order:
  (spatial, scalars_to_grid, bearing, target)

Security note: `torch_loads` ultimately uses `torch.load` (pickle). Only open
caches you built locally; do not load untrusted shard files.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import webdataset as wds

from movebench.features.manifest import utc_now_iso


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
            "schema_version": "2",
            "payload_format": "pth",
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
    config_hash: str

    rasters_hash: str
    normalization_sampler_hash: str | None

    output_hw: tuple[int, int]
    normalize: bool
    step_multiplier: float
    step_q: float
    gap_q: float

    pixel_size_m: float
    env_channels: int

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
            "config_hash": str(self.config_hash),
            "rasters_hash": str(self.rasters_hash),
            "normalization_sampler_hash": self.normalization_sampler_hash,
            "output_hw": [int(self.output_hw[0]), int(self.output_hw[1])],
            "normalize": bool(self.normalize),
            "step_multiplier": float(self.step_multiplier),
            "step_q": float(self.step_q),
            "gap_q": float(self.gap_q),
            "pixel_size_m": float(self.pixel_size_m),
            "env_channels": int(self.env_channels),
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
            config_hash=str(d["config_hash"]),
            rasters_hash=str(d["rasters_hash"]),
            normalization_sampler_hash=d.get("normalization_sampler_hash"),
            output_hw=(int(ohw[0]), int(ohw[1])),
            normalize=bool(d["normalize"]),
            step_multiplier=float(d["step_multiplier"]),
            step_q=float(d["step_q"]),
            gap_q=float(d["gap_q"]),
            pixel_size_m=float(d["pixel_size_m"]),
            env_channels=int(d["env_channels"]),
            num_samples=int(d["num_samples"]),
            samples_per_shard=int(d["samples_per_shard"]),
            shards=[str(s) for s in d["shards"]],
            spatial_dtype=str(d["spatial_dtype"]),
            target_dtype=str(d["target_dtype"]),
            compression=str(d["compression"]),
            tensors={str(k): dict(v) for k, v in dict(d.get("tensors", {})).items()},
        )

    def validate_basic(self) -> None:
        if str(self.schema_version) != "2":
            raise ValueError(f"Unsupported schema_version={self.schema_version!r} (expected '2').")
        if int(self.num_samples) <= 0:
            raise ValueError(f"num_samples must be > 0; got {self.num_samples}")
        if int(self.samples_per_shard) <= 0:
            raise ValueError(f"samples_per_shard must be > 0; got {self.samples_per_shard}")
        if not self.shards:
            raise ValueError("shards list is empty")
        if float(self.pixel_size_m) <= 0:
            raise ValueError(f"pixel_size_m must be > 0; got {self.pixel_size_m}")
        if int(self.env_channels) <= 0:
            raise ValueError(f"env_channels must be > 0; got {self.env_channels}")

        spatial = self.tensors.get("spatial", {})
        sshape = list(spatial.get("shape", []))
        if sshape:
            if len(sshape) != 4:
                raise ValueError(f"spatial tensor shape must be [N,C,H,W]; got shape={sshape}")
            if int(sshape[0]) != int(self.num_samples):
                raise ValueError(f"spatial.shape[0] must match num_samples; got {sshape[0]} vs {self.num_samples}")
            if int(sshape[1]) != int(self.env_channels):
                raise ValueError(
                    f"spatial.shape[1] must match env_channels; got {sshape[1]} vs {self.env_channels}"
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
    from movebench.layers.plan import LayerPlan

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

    spatial_torch_dtype = torch.float16 if spatial_dtype == "float16" else torch.float32
    if target_dtype == "uint8":
        target_torch_dtype = torch.uint8
    elif target_dtype == "float16":
        target_torch_dtype = torch.float16
    else:
        target_torch_dtype = torch.float32

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

            if not isinstance(spatial, torch.Tensor):
                raise TypeError(f"Expected spatial to be torch.Tensor, got {type(spatial)!r}")
            if not isinstance(scalars, torch.Tensor):
                raise TypeError(f"Expected scalars to be torch.Tensor, got {type(scalars)!r}")
            if not isinstance(bearing, torch.Tensor):
                raise TypeError(f"Expected bearing to be torch.Tensor, got {type(bearing)!r}")
            if not isinstance(target, torch.Tensor):
                raise TypeError(f"Expected target to be torch.Tensor, got {type(target)!r}")

            spatial_t = spatial.detach().to(device="cpu", dtype=spatial_torch_dtype)
            scalars_t = scalars.detach().to(device="cpu", dtype=torch.float32)
            bearing_t = bearing.detach().to(device="cpu", dtype=torch.float32)

            if target_torch_dtype == torch.uint8:
                target_t = (target.detach().to(device="cpu", dtype=torch.float32) > 0.5).to(dtype=torch.uint8)
            else:
                target_t = target.detach().to(device="cpu", dtype=target_torch_dtype)

            if not spatial_t.is_contiguous():
                spatial_t = spatial_t.contiguous()
            if not scalars_t.is_contiguous():
                scalars_t = scalars_t.contiguous()
            if not bearing_t.is_contiguous():
                bearing_t = bearing_t.contiguous()
            if not target_t.is_contiguous():
                target_t = target_t.contiguous()

            pth_bytes = wds.torch_dumps((spatial_t, scalars_t, bearing_t, target_t))
            key = f"{int(idx):09d}"
            sink.write({"__key__": key, "pth": pth_bytes})
    finally:
        sink.close()

    shards = sorted([p for p in shards_dir.glob(f"shard-*{ext}") if p.is_file()])
    if not shards:
        raise RuntimeError(f"No shards were written under: {shards_dir}")
    rel_shards = [str(Path("shards") / p.name) for p in shards]

    env_channels = int(plan.num_channels())
    H, W = output_hw
    manifest = DeepSSFWDSCacheManifest(
        schema_version="2",
        created_at=utc_now_iso(),
        study_id=str(study_id),
        split=str(split),
        cache_hash=str(cache_hash),
        config_path=str(config_path) if config_path is not None else None,
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        normalization_sampler_hash=str(normalization_sampler_hash) if normalization_sampler_hash is not None else None,
        output_hw=(int(H), int(W)),
        normalize=bool(normalize),
        step_multiplier=float(step_multiplier),
        step_q=float(step_q),
        gap_q=float(gap_q),
        pixel_size_m=float(pixel_size_m),
        env_channels=int(env_channels),
        num_samples=int(n),
        samples_per_shard=int(samples_per_shard),
        shards=rel_shards,
        spatial_dtype=str(spatial_dtype),
        target_dtype=str(target_dtype),
        compression=str(compression),
        tensors={
            "spatial": {"shape": [int(n), int(env_channels), int(H), int(W)], "dtype": str(spatial_dtype)},
            "scalars": {"shape": [int(n), 4], "dtype": "float32"},
            "bearing": {"shape": [int(n), 1], "dtype": "float32"},
            "target": {"shape": [int(n), int(H), int(W)], "dtype": str(target_dtype)},
        },
    )
    write_deepssf_wds_manifest(manifest_path, manifest)
    return manifest


def _decode_pth_sample(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b = sample.get("pth")
    if b is None:
        raise KeyError("WebDataset sample missing required field 'pth'")
    if not isinstance(b, (bytes, bytearray, memoryview)):
        raise TypeError(f"Expected sample['pth'] to be bytes-like, got {type(b).__name__}")

    obj = wds.torch_loads(bytes(b))
    try:
        spatial, scalars, bearing, target = obj
    except Exception as e:
        raise ValueError("Expected .pth payload to be a 4-tuple: (spatial, scalars, bearing, target)") from e

    if not isinstance(spatial, torch.Tensor):
        raise TypeError(f"Expected spatial to be torch.Tensor, got {type(spatial)!r}")
    if not isinstance(scalars, torch.Tensor):
        raise TypeError(f"Expected scalars to be torch.Tensor, got {type(scalars)!r}")
    if not isinstance(bearing, torch.Tensor):
        raise TypeError(f"Expected bearing to be torch.Tensor, got {type(bearing)!r}")
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"Expected target to be torch.Tensor, got {type(target)!r}")

    x1 = spatial.to(dtype=torch.float32)
    x2 = scalars.to(dtype=torch.float32)
    x3 = bearing.to(dtype=torch.float32)
    y = target.to(dtype=torch.float32)
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
    ds = ds.map(_decode_pth_sample)
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
        shuffle_buffer=int(shuffle_buffer),
        seed=int(seed),
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "batch_size": int(batch_size),
        "pin_memory": bool(pin_memory),
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    loader = wds.WebLoader(ds, **loader_kwargs)
    return loader


def cache_dir_for_study(
    *,
    study_dir: str | Path,
    split: str,
    config_hash: str,
    cache_hash: str,
) -> Path:
    return Path(study_dir) / "cache" / "deepssf_wds" / str(split) / str(config_hash) / str(cache_hash)


def steps_per_epoch_from_manifest(manifest: DeepSSFWDSCacheManifest, *, batch_size: int) -> int:
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    return int(math.ceil(float(manifest.num_samples) / float(b)))


def ensure_deepssf_wds_cache_split(
    *,
    study: Any,
    split: str,
    layers_config: str,
    ds_kwargs: dict[str, Any],
    samples_per_shard: int,
    spatial_dtype: SpatialDType,
    target_dtype: TargetDType,
    compression: Compression,
    seed: int,
    build_workers: int,
    build_if_missing: bool,
) -> tuple[Path, DeepSSFWDSCacheManifest]:
    """Ensure a deepSSF WebDataset cache exists for a study+split and return (cache_dir, manifest)."""
    split = str(split)
    layers_config = str(layers_config)
    plan = study.layer_plan(layers_config)

    ds_kwargs = dict(ds_kwargs)
    ds_kwargs["split"] = split
    ds_kwargs["config"] = layers_config

    output_hw = ds_kwargs.get("output_hw", (101, 101))
    if not (isinstance(output_hw, (list, tuple)) and len(output_hw) == 2):
        raise ValueError("datasets.deepssf.output_hw must be a 2-item sequence like [101,101].")
    output_hw = (int(output_hw[0]), int(output_hw[1]))

    normalize = bool(ds_kwargs.get("normalize", True))
    norm_hash_raw = ds_kwargs.get("normalization_sampler_hash", None)
    norm_hash = str(norm_hash_raw) if (normalize and norm_hash_raw is not None) else None

    step_multiplier = float(ds_kwargs.get("step_multiplier", 2.0))
    step_q = float(ds_kwargs.get("step_q", 0.95))
    gap_q = float(ds_kwargs.get("gap_q", 0.999))

    cache_hash = deepssf_wds_cache_hash(
        config_hash=str(plan.config_hash),
        rasters_hash=str(plan.rasters_hash),
        split=split,
        output_hw=output_hw,
        normalize=normalize,
        normalization_sampler_hash=norm_hash,
        step_multiplier=step_multiplier,
        step_q=step_q,
        gap_q=gap_q,
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,
        target_dtype=target_dtype,
        compression=compression,
    )

    cache_dir = cache_dir_for_study(
        study_dir=study.study_dir,
        split=split,
        config_hash=str(plan.config_hash),
        cache_hash=str(cache_hash),
    )
    manifest_path = cache_dir / "manifest.json"

    if not manifest_path.exists():
        if not build_if_missing:
            raise FileNotFoundError(
                f"deepSSF cache missing: {manifest_path} (set datasets.deepssf.cache.build_if_missing=true)"
            )
        ds = study.deepssf_dataset(**ds_kwargs)
        build_deepssf_wds_cache(
            ds,
            cache_dir,
            samples_per_shard=int(samples_per_shard),
            spatial_dtype=spatial_dtype,
            target_dtype=target_dtype,
            compression=compression,
            seed=int(seed),
            num_workers=int(build_workers),
            overwrite=False,
            progress=True,
            config_path=str(layers_config),
            normalization_sampler_hash=norm_hash,
        )

    manifest = read_deepssf_wds_manifest(manifest_path)
    return cache_dir, manifest


def make_deepssf_wds_train_test_loaders(
    *,
    study: Any,
    layers_config: str,
    ds_kwargs_train: dict[str, Any],
    ds_kwargs_test: dict[str, Any],
    cache_cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    runtime_seed: int,
) -> tuple[
    Any,  # train_loader
    Any,  # test_loader
    DeepSSFWDSCacheManifest,  # train_manifest
    DeepSSFWDSCacheManifest,  # test_manifest
    int,  # steps_per_epoch
]:
    if not isinstance(cache_cfg, dict):
        raise ValueError("datasets.deepssf.cache must be a mapping/dict when provided")

    samples_per_shard = int(cache_cfg.get("samples_per_shard", 256))
    spatial_dtype = str(cache_cfg.get("spatial_dtype", "float16"))
    target_dtype = str(cache_cfg.get("target_dtype", "uint8"))
    compression = str(cache_cfg.get("compression", "none"))
    build_workers = int(cache_cfg.get("build_workers", 0))
    shardshuffle = int(cache_cfg.get("shardshuffle", 100))
    shuffle_buffer = int(cache_cfg.get("shuffle_buffer", 0))
    seed = int(cache_cfg.get("seed", int(runtime_seed)))
    build_if_missing = bool(cache_cfg.get("build_if_missing", True))

    if samples_per_shard <= 0:
        raise ValueError(f"datasets.deepssf.cache.samples_per_shard must be > 0; got {samples_per_shard}")
    if spatial_dtype not in {"float16", "float32"}:
        raise ValueError("datasets.deepssf.cache.spatial_dtype must be one of {'float16','float32'}")
    if target_dtype not in {"uint8", "float16", "float32"}:
        raise ValueError("datasets.deepssf.cache.target_dtype must be one of {'uint8','float16','float32'}")
    if compression not in {"none", "gz"}:
        raise ValueError("datasets.deepssf.cache.compression must be one of {'none','gz'}")
    if build_workers < 0:
        raise ValueError(f"datasets.deepssf.cache.build_workers must be >= 0; got {build_workers}")
    if shardshuffle < 0:
        raise ValueError(f"datasets.deepssf.cache.shardshuffle must be >= 0; got {shardshuffle}")
    if shuffle_buffer < 0:
        raise ValueError(f"datasets.deepssf.cache.shuffle_buffer must be >= 0; got {shuffle_buffer}")

    bs = int(batch_size)
    if bs <= 0:
        raise ValueError(f"datasets.deepssf.batch_size must be > 0; got {batch_size!r}")

    nw = int(num_workers)
    if nw < 0:
        raise ValueError(f"datasets.deepssf.num_workers must be >= 0; got {num_workers!r}")

    cache_dir_train, m_train = ensure_deepssf_wds_cache_split(
        study=study,
        split="train",
        layers_config=str(layers_config),
        ds_kwargs=ds_kwargs_train,
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,  # type: ignore[arg-type]
        target_dtype=target_dtype,  # type: ignore[arg-type]
        compression=compression,  # type: ignore[arg-type]
        seed=int(seed),
        build_workers=int(build_workers),
        build_if_missing=bool(build_if_missing),
    )
    cache_dir_test, m_test = ensure_deepssf_wds_cache_split(
        study=study,
        split="test",
        layers_config=str(layers_config),
        ds_kwargs=ds_kwargs_test,
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,  # type: ignore[arg-type]
        target_dtype=target_dtype,  # type: ignore[arg-type]
        compression=compression,  # type: ignore[arg-type]
        seed=int(seed),
        build_workers=int(build_workers),
        build_if_missing=bool(build_if_missing),
    )

    train_loader = make_deepssf_wds_loader(
        cache_dir_train,
        batch_size=int(bs),
        num_workers=int(nw),
        shardshuffle=int(shardshuffle),
        shuffle_buffer=int(shuffle_buffer),
        seed=int(seed),
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    test_loader = make_deepssf_wds_loader(
        cache_dir_test,
        batch_size=int(bs),
        num_workers=int(nw),
        shardshuffle=0,
        shuffle_buffer=0,
        seed=int(seed),
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )

    steps_per_epoch = steps_per_epoch_from_manifest(m_train, batch_size=int(bs))
    return train_loader, test_loader, m_train, m_test, int(steps_per_epoch)
