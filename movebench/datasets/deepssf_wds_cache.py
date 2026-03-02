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
import tarfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import webdataset as wds
import movebench.paths as paths


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

_MANIFEST_SCHEMA_VERSION = "2"
_PAYLOAD_FORMAT = "pth"


def write_deepssf_wds_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_deepssf_wds_manifest(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        d = json.load(f)
    if str(d.get("schema_version", "")) != _MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version={d.get('schema_version')!r} (expected {_MANIFEST_SCHEMA_VERSION!r}).")
    payload = str(d.get("payload_format", _PAYLOAD_FORMAT))
    if payload != _PAYLOAD_FORMAT:
        raise ValueError(f"Unsupported payload_format={payload!r} (expected {_PAYLOAD_FORMAT!r}).")
    if int(d.get("num_samples", 0)) <= 0:
        raise ValueError(f"manifest.num_samples must be > 0; got {d.get('num_samples')!r}")
    if float(d.get("pixel_size_m", 0.0)) <= 0:
        raise ValueError(f"manifest.pixel_size_m must be > 0; got {d.get('pixel_size_m')!r}")
    return dict(d)


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
    resume: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Build a WebDataset cache from a DeepSSFDataset-like dataset."""
    out_dir = Path(out_dir)
    shards_dir = out_dir / "shards"
    manifest_path = out_dir / "manifest.json"
    resuming = bool(resume) and out_dir.exists() and not bool(overwrite)

    if out_dir.exists():
        if not overwrite and not resume:
            raise FileExistsError(
                f"Output dir exists (pass overwrite=True to replace, or resume=True to continue): {out_dir}"
            )
        import shutil

        if overwrite:
            shutil.rmtree(out_dir)

    if manifest_path.exists() and not overwrite:
        return read_deepssf_wds_manifest(manifest_path)

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

    ohw = getattr(chips, "output_hw", None)
    if not (isinstance(ohw, (list, tuple)) and len(ohw) == 2):
        raise TypeError("dataset.chips.output_hw must be a (H,W) tuple")
    output_hw = (int(ohw[0]), int(ohw[1]))

    pixel_size_m = float(getattr(chips, "pixel_size_m"))
    if pixel_size_m <= 0:
        raise ValueError(f"Invalid pixel_size_m={pixel_size_m}")

    n = int(len(dataset))
    if n <= 0:
        raise ValueError(f"Dataset is empty (len={n}); nothing to cache")

    if int(samples_per_shard) <= 0:
        raise ValueError(f"samples_per_shard must be > 0; got {samples_per_shard}")

    spatial_torch_dtype = torch.float16 if spatial_dtype == "float16" else torch.float32
    if target_dtype == "uint8":
        target_torch_dtype = torch.uint8
    elif target_dtype == "float16":
        target_torch_dtype = torch.float16
    else:
        target_torch_dtype = torch.float32

    def _resume_offset() -> int:
        if not shards_dir.exists():
            return 0
        shard_paths = sorted([p for p in shards_dir.glob("shard-*.tar*") if p.is_file()])
        if not shard_paths:
            return 0

        existing = 0
        for i, p in enumerate(shard_paths):
            try:
                pth_count = 0
                with tarfile.open(p, mode="r:*") as tf:
                    for m in tf:
                        if m.isfile() and str(m.name).endswith(".pth"):
                            pth_count += 1
                existing += int(pth_count)
            except (tarfile.TarError, EOFError, OSError) as e:
                warnings.warn(
                    f"Corrupt shard found; deleting and resuming from here: {p} ({e})",
                    RuntimeWarning,
                )
                try:
                    # Delete this shard and any that come after it. We can't safely
                    # resume from later shards if there's a gap/corruption.
                    for q in shard_paths[i:]:
                        try:
                            q.unlink()
                        except FileNotFoundError:
                            pass
                except Exception:
                    # Best-effort cleanup; resume will still proceed from the
                    # last known-good offset.
                    pass
                break

            # Empty trailing shard is usually just a leftover from a crash; remove it.
            if pth_count == 0 and i == (len(shard_paths) - 1):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        return int(existing)

    start_idx = 0
    if resuming:
        start_idx = _resume_offset()
        if start_idx < 0:
            start_idx = 0
        if start_idx > n:
            raise RuntimeError(f"Cache already has {start_idx} samples but dataset has len={n}; rebuild with overwrite=True.")
        if start_idx == n:
            # Shards already contain the full dataset; just write the manifest.
            env_channels = int(plan.num_channels())
            H, W = output_hw
            manifest: dict[str, Any] = {
                "schema_version": _MANIFEST_SCHEMA_VERSION,
                "payload_format": _PAYLOAD_FORMAT,
                "cache_hash": str(out_dir.name),
                "output_hw": [int(H), int(W)],
                "pixel_size_m": float(pixel_size_m),
                "env_channels": int(env_channels),
                "num_samples": int(n),
            }
            write_deepssf_wds_manifest(manifest_path, manifest)
            return dict(manifest)

    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    remaining = perm[int(start_idx) :].tolist()
    if not remaining:
        raise RuntimeError("No remaining samples to write (resume offset consumed dataset).")

    from torch.utils.data import Subset

    subset = Subset(dataset, remaining)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=lambda batch: batch[0],
        drop_last=False,
    )

    ext = ".tar.gz" if compression == "gz" else ".tar"
    if resuming:
        tag = 0
        while any(shards_dir.glob(f"shard-resume-{int(start_idx):09d}-{int(tag):02d}-*{ext}")):
            tag += 1
        pattern = str(shards_dir / f"shard-resume-{int(start_idx):09d}-{int(tag):02d}-%06d{ext}")
    else:
        pattern = str(shards_dir / f"shard-%06d{ext}")
    sink = wds.ShardWriter(pattern, maxcount=int(samples_per_shard))

    it: Iterable[Any] = loader
    if progress:
        total = int(n - int(start_idx))
        prefix = "resume_" if int(start_idx) > 0 else ""
        study_id = str(out_dir).replace(str(paths.studies_root()), "").split("/")[1]
        it = tqdm(loader, total=total, desc=f"{study_id} | {prefix}build_deepssf_wds_cache[{split}]", unit="samples")

    try:
        for j, (spatial, scalars, bearing, target) in enumerate(it):
            idx = int(start_idx) + int(j)
            spatial_t = spatial.detach().to(device="cpu", dtype=spatial_torch_dtype).contiguous()
            scalars_t = scalars.detach().to(device="cpu", dtype=torch.float32).contiguous()
            bearing_t = bearing.detach().to(device="cpu", dtype=torch.float32).contiguous()

            if target_torch_dtype == torch.uint8:
                target_t = (target.detach().to(device="cpu", dtype=torch.float32) > 0.5).to(dtype=torch.uint8)
            else:
                target_t = target.detach().to(device="cpu", dtype=target_torch_dtype)
            target_t = target_t.contiguous()

            sink.write({"__key__": f"{int(idx):09d}", "pth": wds.torch_dumps((spatial_t, scalars_t, bearing_t, target_t))})
    finally:
        sink.close()

    num_shards = sum(1 for p in shards_dir.glob(f"shard-*{ext}") if p.is_file())
    if num_shards <= 0:
        raise RuntimeError(f"No shards were written under: {shards_dir}")

    env_channels = int(plan.num_channels())
    H, W = output_hw
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "payload_format": _PAYLOAD_FORMAT,
        "cache_hash": str(out_dir.name),
        "output_hw": [int(H), int(W)],
        "pixel_size_m": float(pixel_size_m),
        "env_channels": int(env_channels),
        "num_samples": int(n),
    }
    write_deepssf_wds_manifest(manifest_path, manifest)
    return dict(manifest)


def _decode_pth_sample(sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial, scalars, bearing, target = wds.torch_loads(bytes(sample["pth"]))
    return (
        spatial.to(dtype=torch.float32),
        scalars.to(dtype=torch.float32),
        bearing.to(dtype=torch.float32),
        target.to(dtype=torch.float32),
    )


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
    cache_dir = Path(cache_dir)
    _ = read_deepssf_wds_manifest(cache_dir / "manifest.json")
    shards_dir = cache_dir / "shards"
    urls = sorted([str(p) for p in shards_dir.glob("shard-*.tar*") if p.is_file()])
    if not urls:
        raise FileNotFoundError(f"No shards found under: {shards_dir}")

    # WebDataset splits shards across DataLoader workers. If num_workers exceeds
    # the number of shards, some workers would receive no shards and can error
    # (or hang) depending on the loader pipeline. Cap workers to shards.
    nw = int(num_workers or 0)
    if nw > len(urls):
        nw = int(len(urls))

    ds = wds.WebDataset(
        urls,
        shardshuffle=int(shardshuffle),
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
    )
    if int(shuffle_buffer) > 0:
        ds = ds.shuffle(int(shuffle_buffer), rng=random.Random(int(seed)))
    ds = ds.map(_decode_pth_sample)

    loader_kwargs: dict[str, Any] = {
        "num_workers": int(nw),
        "batch_size": int(batch_size),
        "pin_memory": bool(pin_memory),
    }
    if int(nw) > 0:
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


def steps_per_epoch_from_manifest(manifest: dict[str, Any], *, batch_size: int) -> int:
    b = int(batch_size)
    if b <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    return int(math.ceil(float(int(manifest["num_samples"])) / float(b)))


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
    overwrite: bool = False,
    resume: bool = False,
) -> tuple[Path, dict[str, Any]]:
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

    if bool(overwrite) and cache_dir.exists():
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
            overwrite=True,
            resume=bool(resume),
            progress=True,
        )
    elif not manifest_path.exists():
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
            resume=bool(resume),
            progress=True,
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
    dict[str, Any],  # train_manifest
    dict[str, Any],  # test_manifest
    int,  # steps_per_epoch
]:
    """Build/load deepSSF WebDataset caches and return train/test loaders + manifests.

    This helper centralizes defaults and build-if-missing behavior so call sites
    (e.g., benchmark.py) stay minimal.
    """
    if not isinstance(cache_cfg, dict):
        raise ValueError("cache_cfg must be a mapping/dict")

    bs = int(batch_size) if batch_size is not None else 1
    nw = int(num_workers or 0)

    samples_per_shard = int(cache_cfg.get("samples_per_shard", 256))
    spatial_dtype = str(cache_cfg.get("spatial_dtype", "float16"))
    target_dtype = str(cache_cfg.get("target_dtype", "uint8"))
    compression = str(cache_cfg.get("compression", "none"))
    build_workers = int(cache_cfg.get("build_workers", 0))
    shardshuffle = int(cache_cfg.get("shardshuffle", 100))
    shuffle_buffer = int(cache_cfg.get("shuffle_buffer", 0))
    seed = int(cache_cfg.get("seed", int(runtime_seed)))
    build_if_missing = bool(cache_cfg.get("build_if_missing", True))
    resume = bool(cache_cfg.get("resume", False))
    overwrite = bool(cache_cfg.get("overwrite", False))

    cache_dir_train, m_train = ensure_deepssf_wds_cache_split(
        study=study,
        split="train",
        layers_config=str(layers_config),
        ds_kwargs=dict(ds_kwargs_train),
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,  # type: ignore[arg-type]
        target_dtype=target_dtype,  # type: ignore[arg-type]
        compression=compression,  # type: ignore[arg-type]
        seed=int(seed),
        build_workers=int(build_workers),
        build_if_missing=bool(build_if_missing),
        overwrite=bool(overwrite),
        resume=bool(resume),
    )
    cache_dir_test, m_test = ensure_deepssf_wds_cache_split(
        study=study,
        split="test",
        layers_config=str(layers_config),
        ds_kwargs=dict(ds_kwargs_test),
        samples_per_shard=int(samples_per_shard),
        spatial_dtype=spatial_dtype,  # type: ignore[arg-type]
        target_dtype=target_dtype,  # type: ignore[arg-type]
        compression=compression,  # type: ignore[arg-type]
        seed=int(seed),
        build_workers=int(build_workers),
        build_if_missing=bool(build_if_missing),
        overwrite=bool(overwrite),
        resume=bool(resume),
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
