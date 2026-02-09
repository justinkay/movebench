"""SSF/iSSF datasets backed by extracted point feature stores.

These datasets are intentionally thin wrappers over the on-disk point feature
stores written by `Study.extract_point_features(...)`.

They do not perform raster IO or candidate generation. They:
- load `features.pt` and `manifest.json`,
- optionally apply cached env normalization,
- (iSSF only) append movement/time/id features and interaction terms.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from torch.utils.data import Dataset

from movebench2.features.manifest import PointFeatureManifest, read_manifest
from movebench2.layers.plan import LayerPlan
from movebench2.trajectories import INDIVIDUAL_COL, POINT_ID_COL, load_trajectories_csv


TARGET_CRS = "EPSG:3857" # meters
SamplerType = Literal["empirical", "parametric"]


def _point_store_base_dir(*, study_dir: Path, split: str, config_hash: str) -> Path:
    return Path(study_dir) / "features" / "points" / str(split) / str(config_hash)


def _iter_point_store_manifests(base_dir: Path) -> list[tuple[str, PointFeatureManifest]]:
    if not base_dir.exists():
        return []
    out: list[tuple[str, PointFeatureManifest]] = []
    for p in sorted(base_dir.iterdir()):
        if not p.is_dir():
            continue
        mp = p / "manifest.json"
        if not mp.exists():
            continue
        out.append((p.name, read_manifest(mp)))
    return out


def _manifest_matches(
    m: PointFeatureManifest,
    *,
    expected_type: SamplerType,
    config_hash: str,
    num_candidates: int | None,
    random_seed: int | None,
    step_multiplier: float | None,
    step_q: float | None,
    gap_q: float | None,
    bounds_buffer_deg: float | None,
) -> bool:
    if str(m.config_hash) != str(config_hash):
        return False
    if str(m.sampler.get("type")) != str(expected_type):
        return False
    if num_candidates is not None and int(m.sampler.get("num_candidates")) != int(num_candidates):
        return False
    if random_seed is not None and int(m.sampler.get("random_seed")) != int(random_seed):
        return False
    if step_multiplier is not None and float(m.sampler.get("step_multiplier")) != float(step_multiplier):
        return False
    if step_q is not None and float(m.sampler.get("step_q")) != float(step_q):
        return False
    if gap_q is not None and float(m.sampler.get("gap_q")) != float(gap_q):
        return False
    if bounds_buffer_deg is not None and float(m.extraction.get("bounds_buffer_deg")) != float(bounds_buffer_deg):
        return False
    return True


def _select_point_store(
    *,
    study_dir: Path,
    split: str,
    config_hash: str,
    expected_type: SamplerType,
    num_candidates: int | None,
    random_seed: int | None,
    step_multiplier: float | None,
    step_q: float | None,
    gap_q: float | None,
    bounds_buffer_deg: float | None,
) -> tuple[Path, PointFeatureManifest]:
    base_dir = _point_store_base_dir(study_dir=study_dir, split=split, config_hash=config_hash)
    candidates = _iter_point_store_manifests(base_dir)
    matches = [
        (sampler_hash, m)
        for sampler_hash, m in candidates
        if _manifest_matches(
            m,
            expected_type=expected_type,
            config_hash=config_hash,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )
    ]

    if not matches:
        raise FileNotFoundError(
            "No extracted point feature store found for "
            f"split={split!r}, config_hash={config_hash!r}, sampler_type={expected_type!r}. "
            "Run Study.extract_point_features(...) first."
        )

    if len(matches) != 1:
        lines = [
            "Multiple matching point feature stores found. Disambiguate by passing one or more of:",
            "num_candidates, random_seed, step_multiplier, step_q, gap_q, bounds_buffer_deg.",
            "Found:",
        ]
        for sampler_hash, m in matches:
            s = m.sampler
            e = m.extraction
            lines.append(
                f"- store_dir={sampler_hash!r} created_at={m.created_at!r} "
                f"num_candidates={s.get('num_candidates')!r} random_seed={s.get('random_seed')!r} "
                f"step_multiplier={s.get('step_multiplier')!r} step_q={s.get('step_q')!r} gap_q={s.get('gap_q')!r} "
                f"bounds_buffer_deg={e.get('bounds_buffer_deg')!r}"
            )
        raise ValueError("\n".join(lines))

    sampler_hash, m = matches[0]
    store_dir = base_dir / sampler_hash
    if not (store_dir / "features.pt").exists():
        raise FileNotFoundError(f"features.pt not found under: {store_dir}")
    return store_dir, m


def _load_env_normalizer(
    *,
    study_dir: Path,
    plan: LayerPlan,
    sampler_hash: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats_path = (
        Path(study_dir)
        / "normalization"
        / "points"
        / str(plan.config_hash)
        / str(sampler_hash)
        / "stats.json"
    )
    if not stats_path.exists():
        raise FileNotFoundError(
            "Normalization stats not found. Run "
            "Study.compute_normalization_vals(from_='points', ...) first. "
            f"Expected: {stats_path}"
        )

    with open(stats_path, "r") as f:
        d = json.load(f)
    band_stats: dict[str, Any] = dict(d.get("band_stats", {}))

    band_ids = plan.band_ids_in_order()
    C = int(plan.num_channels())
    mean = torch.zeros(C, dtype=torch.float32)
    std = torch.ones(C, dtype=torch.float32)

    for i, ch in enumerate(plan.channels):
        if not ch.normalize:
            continue
        band_id = band_ids[i]
        st = band_stats.get(band_id)
        if st is None:
            raise ValueError(f"Missing normalization stats for band_id={band_id!r} in {stats_path}")
        mean[i] = float(st["mean"])
        s = float(st["std"])
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        std[i] = s

    return mean, std


def _compute_movement_stats(step_length_m: torch.Tensor, turn_angle_rad: torch.Tensor) -> dict[str, float]:
    """Compute mean/std for sl, log_sl, cos_ta from true endpoints only."""
    sl = step_length_m[:, -1].to(dtype=torch.float64)
    ta = turn_angle_rad[:, -1].to(dtype=torch.float64)

    log_sl = torch.log(sl + 1e-8)
    cos_ta = torch.cos(ta)

    def ms(x: torch.Tensor) -> tuple[float, float]:
        m = float(x.mean().item())
        s = float(x.std(unbiased=False).item())
        if not math.isfinite(s) or s < 1e-8:
            s = 1.0
        return m, s

    sl_mean, sl_std = ms(sl)
    log_sl_mean, log_sl_std = ms(log_sl)
    cos_ta_mean, cos_ta_std = ms(cos_ta)

    return {
        "sl_mean": sl_mean,
        "sl_std": sl_std,
        "log_sl_mean": log_sl_mean,
        "log_sl_std": log_sl_std,
        "cos_ta_mean": cos_ta_mean,
        "cos_ta_std": cos_ta_std,
    }


class SSFDataset(Dataset):
    """Empirical SSF dataset (env + valid mask) loaded from a point feature store."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        plan: LayerPlan,
        split: str,
        normalize: bool = True,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        self.study_dir = Path(study_dir)
        self.split = str(split)

        store_dir, m = _select_point_store(
            study_dir=self.study_dir,
            split=self.split,
            config_hash=str(plan.config_hash),
            expected_type="empirical",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = m.layer_plan
        self.sampler_hash = str(m.sampler_hash)
        self.normalize = bool(normalize)

        d = torch.load(store_dir / "features.pt", map_location="cpu")
        self.env = d["env"]
        self.valid = d["valid"]
        self.step_length_m = d.get("step_length_m")
        self.turn_angle_rad = d.get("turn_angle_rad")
        self.point_id_start = d.get("point_id_start")
        self.timestamp_ns = d.get("timestamp_ns")

        if self.env.ndim != 3 or self.valid.ndim != 3:
            raise ValueError(f"Expected env/valid rank-3 [N,C,P]; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if tuple(self.env.shape) != tuple(self.valid.shape):
            raise ValueError(f"env and valid shapes must match; got env={tuple(self.env.shape)} valid={tuple(self.valid.shape)}")
        if int(self.env.shape[1]) != int(self.plan.num_channels()):
            raise ValueError(f"env.shape[1] ({self.env.shape[1]}) must match plan.num_channels() ({self.plan.num_channels()})")

        if self.normalize:
            self._mean, self._std = _load_env_normalizer(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_hash=self.sampler_hash,
            )
        else:
            self._mean = None
            self._std = None

    def __len__(self) -> int:
        return int(self.env.shape[0])

    def _get_env_valid(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        env = self.env[int(idx)]
        valid = self.valid[int(idx)]

        if not self.normalize:
            return env, valid

        assert self._mean is not None
        assert self._std is not None
        mean = self._mean.view(-1, 1)
        std = self._std.view(-1, 1)
        env_norm = torch.where(valid, (env - mean) / std, torch.zeros_like(env))
        return env_norm, valid

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
        env, valid = self._get_env_valid(idx)
        return env, valid, {}, TARGET_CRS


class iSSFDataset(SSFDataset):
    """Parametric iSSF dataset (env + movement/time/id + interactions)."""

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        split: str,
        normalize: bool = True,
        normalize_movement: bool = True,
        include_time: bool = False,
        include_individual_id: bool = False,
        movement_stats: dict[str, float] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ) -> None:
        # Bypass SSFDataset selection: iSSF is always parametric.
        self.study_dir = Path(study_dir)
        self.split = str(split)
        self.trajectories_path = Path(trajectories_path)

        store_dir, m = _select_point_store(
            study_dir=self.study_dir,
            split=self.split,
            config_hash=str(plan.config_hash),
            expected_type="parametric",
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

        self.store_dir = store_dir
        self.manifest = m
        self.plan = m.layer_plan
        self.sampler_hash = str(m.sampler_hash)
        self.normalize = bool(normalize)

        d = torch.load(store_dir / "features.pt", map_location="cpu")
        self.env = d["env"]
        self.valid = d["valid"]
        self.step_length_m = d.get("step_length_m")
        self.turn_angle_rad = d.get("turn_angle_rad")
        self.point_id_start = d.get("point_id_start")
        self.timestamp_ns = d.get("timestamp_ns")

        if self.step_length_m is None or self.turn_angle_rad is None or self.point_id_start is None or self.timestamp_ns is None:
            raise ValueError("features.pt missing required iSSF keys: step_length_m, turn_angle_rad, point_id_start, timestamp_ns")

        if self.normalize:
            self._mean, self._std = _load_env_normalizer(
                study_dir=self.study_dir,
                plan=self.plan,
                sampler_hash=self.sampler_hash,
            )
        else:
            self._mean = None
            self._std = None

        self.include_time = bool(include_time)
        self.include_individual_id = bool(include_individual_id)
        self.normalize_movement = bool(normalize_movement)

        if self.normalize_movement:
            if movement_stats is not None:
                self._movement_stats = dict(movement_stats)
            elif self.split == "train":
                self._movement_stats = _compute_movement_stats(self.step_length_m, self.turn_angle_rad)
            else:
                train_dir = (
                    self.study_dir
                    / "features"
                    / "points"
                    / "train"
                    / str(self.plan.config_hash)
                    / str(self.sampler_hash)
                )
                train_pt = train_dir / "features.pt"
                if not train_pt.exists():
                    raise FileNotFoundError(
                        "Train point feature store not found to compute movement_stats. "
                        f"Expected: {train_pt}"
                    )
                td = torch.load(train_pt, map_location="cpu")
                self._movement_stats = _compute_movement_stats(td["step_length_m"], td["turn_angle_rad"])
        else:
            self._movement_stats = None

        if self.include_individual_id:
            tdf = load_trajectories_csv(str(self.trajectories_path))
            if POINT_ID_COL not in tdf.columns or INDIVIDUAL_COL not in tdf.columns:
                raise ValueError(
                    f"trajectories.csv missing required columns: {POINT_ID_COL!r}, {INDIVIDUAL_COL!r}"
                )
            point_to_indiv: dict[int, str] = {
                int(pid): str(indiv) for pid, indiv in zip(tdf[POINT_ID_COL].to_numpy(), tdf[INDIVIDUAL_COL].to_numpy())
            }
            individuals = sorted(set(point_to_indiv.values()))
            self._point_to_indiv = point_to_indiv
            self._individual_to_idx = {ind: i for i, ind in enumerate(individuals)}
            self._n_individuals = int(len(individuals))
        else:
            self._point_to_indiv = {}
            self._individual_to_idx = {}
            self._n_individuals = 0

    @property
    def individual_mapping(self) -> dict[str, int]:
        return dict(self._individual_to_idx)

    @property
    def n_individuals(self) -> int:
        return int(self._n_individuals)

    def _movement_features(self, idx: int) -> torch.Tensor:
        sl = self.step_length_m[int(idx)].to(dtype=torch.float32)
        ta = self.turn_angle_rad[int(idx)].to(dtype=torch.float32)

        log_sl = torch.log(sl + 1e-8)
        cos_ta = torch.cos(ta)

        if self._movement_stats is not None:
            s = self._movement_stats
            sl = (sl - float(s["sl_mean"])) / float(s["sl_std"])
            log_sl = (log_sl - float(s["log_sl_mean"])) / float(s["log_sl_std"])
            cos_ta = (cos_ta - float(s["cos_ta_mean"])) / float(s["cos_ta_std"])

        return torch.stack([sl, log_sl, cos_ta], dim=0)

    def _time_features(self, idx: int, P: int) -> torch.Tensor:
        ts_ns = int(self.timestamp_ns[int(idx)].item())
        ts = pd.to_datetime(ts_ns, unit="ns", utc=True)

        doy = int(ts.dayofyear)
        hour_frac = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0

        sin_doy = math.sin(2.0 * math.pi * float(doy) / 365.25)
        cos_doy = math.cos(2.0 * math.pi * float(doy) / 365.25)
        sin_tod = math.sin(2.0 * math.pi * hour_frac / 24.0)
        cos_tod = math.cos(2.0 * math.pi * hour_frac / 24.0)

        feats = torch.tensor([sin_doy, cos_doy, sin_tod, cos_tod], dtype=torch.float32)
        return feats.view(4, 1).expand(4, P)

    def _individual_id_features(self, idx: int, P: int) -> torch.Tensor:
        pid = int(self.point_id_start[int(idx)].item())
        indiv = self._point_to_indiv.get(pid)
        if indiv is None:
            raise KeyError(f"point_id_start={pid} not found in trajectories.csv individual mapping")
        j = int(self._individual_to_idx[indiv])
        v = torch.zeros(self._n_individuals, dtype=torch.float32)
        v[j] = 1.0
        return v.view(self._n_individuals, 1).expand(self._n_individuals, P)

    @staticmethod
    def _interactions(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute row-wise interactions between (Ka,P) and (Kb,P) -> (Ka*Kb,P)."""
        Ka, P = a.shape
        Kb = b.shape[0]
        return (a.unsqueeze(1) * b.unsqueeze(0)).reshape(Ka * Kb, P)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], str]:
        env, env_valid = self._get_env_valid(idx)
        P = int(env.shape[1])
        C_env = int(env.shape[0])

        move = self._movement_features(idx)  # (3,P)
        move_valid = torch.ones((3, P), dtype=torch.bool)

        parts = [env, move]
        valids = [env_valid, move_valid]

        if self.include_time:
            time = self._time_features(idx, P)
            time_valid = torch.ones((4, P), dtype=torch.bool)
            parts.append(time)
            valids.append(time_valid)
        else:
            time = None

        if self.include_individual_id:
            indiv = self._individual_id_features(idx, P)
            indiv_valid = torch.ones((self._n_individuals, P), dtype=torch.bool)
            parts.append(indiv)
            valids.append(indiv_valid)

        move_env = self._interactions(move, env)
        move_env_valid = env_valid.unsqueeze(0).expand(3, -1, -1).reshape(3 * C_env, P)
        parts.append(move_env)
        valids.append(move_env_valid)

        if self.include_time and time is not None:
            time_env = self._interactions(time, env)
            time_env_valid = env_valid.unsqueeze(0).expand(4, -1, -1).reshape(4 * C_env, P)
            parts.append(time_env)
            valids.append(time_env_valid)

            time_move = self._interactions(time, move)
            time_move_valid = torch.ones((4 * 3, P), dtype=torch.bool)
            parts.append(time_move)
            valids.append(time_move_valid)

        step_pixels = torch.cat(parts, dim=0)
        valid_pixels = torch.cat(valids, dim=0)
        return step_pixels, valid_pixels, {}, TARGET_CRS

