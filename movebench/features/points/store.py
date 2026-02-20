from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from movebench.features.points.config import PointStoreConfig
from movebench.features.points.manifest import PointStoreManifest, read_manifest, write_manifest


def point_stores_dir(*, study_dir: Path, split: str, config_hash: str) -> Path:
    return Path(study_dir) / "features" / "points" / str(split) / str(config_hash)


def point_store_dir(*, study_dir: Path, split: str, config_hash: str, store_id: str) -> Path:
    return point_stores_dir(study_dir=study_dir, split=split, config_hash=config_hash) / str(store_id)


@dataclass(frozen=True)
class PointStore:
    store_dir: Path
    manifest: PointStoreManifest
    tensors: dict[str, Any]

    @property
    def env(self) -> torch.Tensor:
        return self.tensors["env"]

    @property
    def valid(self) -> torch.Tensor:
        return self.tensors["valid"]

    @property
    def step_length_m(self) -> torch.Tensor:
        return self.tensors["step_length_m"]

    @property
    def turn_angle_rad(self) -> torch.Tensor:
        return self.tensors["turn_angle_rad"]

    @property
    def point_id_start(self) -> torch.Tensor:
        return self.tensors["point_id_start"]

    @property
    def point_id_end(self) -> torch.Tensor:
        return self.tensors["point_id_end"]

    @property
    def timestamp_ns(self) -> torch.Tensor:
        return self.tensors["timestamp_ns"]

    @property
    def end_timestamp_ns(self) -> torch.Tensor:
        return self.tensors["end_timestamp_ns"]

    @property
    def start_lon(self) -> torch.Tensor:
        return self.tensors["start_lon"]

    @property
    def start_lat(self) -> torch.Tensor:
        return self.tensors["start_lat"]

    @property
    def start_bearing(self) -> torch.Tensor:
        return self.tensors["start_bearing"]

    @property
    def end_lon(self) -> torch.Tensor:
        return self.tensors["end_lon"]

    @property
    def end_lat(self) -> torch.Tensor:
        return self.tensors["end_lat"]


def load_point_store(store_dir: Path) -> PointStore:
    store_dir = Path(store_dir)
    manifest = read_manifest(store_dir / "manifest.json")
    d = torch.load(store_dir / "store.pt", map_location="cpu")
    if not isinstance(d, dict):
        raise ValueError(f"store.pt must contain a dict, got {type(d).__name__}")
    return PointStore(store_dir=store_dir, manifest=manifest, tensors=d)


def resolve_point_store_dir(
    *,
    study_dir: Path,
    split: str,
    config_hash: str,
    cfg: PointStoreConfig,
) -> tuple[Path, PointStoreManifest]:
    """Resolve a point store directory from config_hash + PointStoreConfig.

    If cfg is fully specified, this is deterministic (compute store_id, check one directory).
    If cfg is partially specified, scan directories and require a unique match.
    """
    base_dir = point_stores_dir(study_dir=study_dir, split=split, config_hash=config_hash)

    def _has_store(dir_: Path) -> bool:
        return (dir_ / "store.pt").exists() and (dir_ / "manifest.json").exists()

    if cfg.is_fully_specified():
        store_id = cfg.store_id()
        store_dir = base_dir / store_id
        if not _has_store(store_dir):
            raise FileNotFoundError(
                "Point store not found. Run Study.extract_point_features(...) first. "
                f"Expected: {store_dir}"
            )
        m = read_manifest(store_dir / "manifest.json")
        if str(m.config_hash) != str(config_hash):
            raise ValueError("Point store manifest config_hash does not match requested config_hash")
        if str(m.store_id) != str(store_id):
            raise ValueError("Point store manifest store_id does not match requested store_id")
        return store_dir, m

    if not base_dir.exists():
        raise FileNotFoundError(f"No point stores directory found: {base_dir}")

    matches: list[tuple[str, PointStoreManifest]] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        if not _has_store(p):
            continue
        try:
            m = read_manifest(p / "manifest.json")
        except Exception:
            continue
        if str(m.config_hash) != str(config_hash):
            continue
        ok = True
        for field in (
            "sampler_type",
            "num_candidates",
            "random_seed",
            "step_multiplier",
            "step_q",
            "gap_q",
            "bounds_buffer_deg",
        ):
            want = getattr(cfg, field)
            if want is None:
                continue
            got = m.store_config.get(field)
            if got is None:
                ok = False
                break
            if field in {"num_candidates", "random_seed"}:
                ok = int(got) == int(want)
            elif field == "sampler_type":
                ok = str(got) == str(want)
            else:
                ok = float(got) == float(want)
            if not ok:
                break
        if ok:
            matches.append((p.name, m))

    if not matches:
        raise FileNotFoundError(
            "No matching point store found under "
            f"{base_dir}. Run Study.extract_point_features(...) first."
        )
    if len(matches) != 1:
        lines = [
            "Multiple matching point stores found. Disambiguate by passing all PointStoreConfig fields.",
            "Found:",
        ]
        for store_id, m in matches:
            lines.append(
                f"- store_id={store_id!r} created_at={m.created_at!r} store_config={m.store_config!r}"
            )
        raise ValueError("\n".join(lines))

    store_id, m = matches[0]
    store_dir = base_dir / store_id
    if not _has_store(store_dir):
        raise FileNotFoundError(f"store.pt/manifest.json not found under: {store_dir}")
    return store_dir, m


def save_point_store(
    *,
    store_dir: Path,
    tensors: dict[str, Any],
    manifest: PointStoreManifest,
    overwrite: bool = False,
) -> None:
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    out_pt = store_dir / "store.pt"
    out_manifest = store_dir / "manifest.json"
    if not overwrite and (out_pt.exists() or out_manifest.exists()):
        raise FileExistsError(f"Refusing to overwrite existing store under: {store_dir}")

    torch.save(tensors, out_pt)
    write_manifest(out_manifest, manifest)
