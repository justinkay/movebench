from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from movebench import paths
from movebench.features.config import SamplerConfig, load_sampler_config
from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    POINT_ID_COL,
    SET_COL,
    IntervalStepSupport,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)


def _load_sampler(sampler_config: str | Path | SamplerConfig) -> SamplerConfig:
    if isinstance(sampler_config, SamplerConfig):
        return sampler_config
    return load_sampler_config(sampler_config)


def _rows_for_raster_downloads(
    df: pd.DataFrame,
    *,
    splits: list[str],
    plan: LayerPlan,
    support: IntervalStepSupport,
) -> pd.DataFrame:
    sub = df[df[SET_COL].isin(splits)].copy()
    if len(sub) == 0:
        return sub
    starts = select_step_starts_with_support(
        sub,
        step_time_s=int(plan.step_spec.step_time_s),
        gap_q=float(plan.step_spec.gap_q),
        support=support,
    )
    if len(starts) == 0:
        return sub.iloc[0:0].copy()
    wanted = set(starts["point_id_start"].astype(int).tolist())
    return sub[sub[POINT_ID_COL].astype(int).isin(wanted)].copy()


@dataclass(frozen=True)
class Study:
    study_id: str

    def __post_init__(self) -> None:
        sd = paths.study_dir(self.study_id)
        if not sd.exists():
            raise FileNotFoundError(f"Study directory not found: {sd}")
        traj = sd / "trajectories.csv"
        if not traj.exists():
            raise FileNotFoundError(f"Study trajectories.csv not found: {traj}")

    @property
    def study_dir(self) -> Path:
        return paths.study_dir(self.study_id)

    @property
    def trajectories_path(self) -> Path:
        return self.study_dir / "trajectories.csv"

    @classmethod
    def list_available(cls) -> list[str]:
        root = paths.studies_root()
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def layer_plan(
        self, layers_config: str | Path = "configs/layers/all.yaml"
    ) -> LayerPlan:
        cfg = load_layers_config(layers_config)
        return REGISTRY.resolve(cfg)

    def layer_io(self, layers_config: str | Path = "configs/layers/all.yaml"):
        from movebench.layers.io import LayerIO

        plan = self.layer_plan(layers_config)
        return LayerIO(study_dir=self.study_dir, plan=plan)

    def download_rasters(
        self,
        *,
        layers_config: str | Path = "configs/layers/all.yaml",
        splits: list[str] | None = None,
        n_workers: int = 16,
        skip_existing: bool = True,
    ) -> dict:
        from movebench.layers import download as mb2dl

        cfg = load_layers_config(layers_config)
        plan = REGISTRY.resolve(cfg)

        df = load_trajectories_csv(self.trajectories_path)
        train_df = df[df[SET_COL] == "train"].copy()
        if len(train_df) == 0:
            raise ValueError(
                "No training data found (set=='train'); cannot compute buffer"
            )
        step = plan.step_spec
        support = fit_interval_step_support(
            train_df,
            step_time_s=int(step.step_time_s),
            gap_q=float(step.gap_q),
            step_q=float(step.step_q),
            max_step_q=float(step.max_step_q),
            step_multiplier=float(step.step_multiplier),
        )
        extent_buffer_m = float(support.max_step_length_m)
        resolution_buffer_m = float(extent_buffer_m)

        if splits is None:
            splits = sorted(df[SET_COL].dropna().unique().tolist())
        download_df = _rows_for_raster_downloads(
            df,
            splits=splits,
            plan=plan,
            support=support,
        )

        clamp = cfg.raster_params["scale_clamp"]
        min_pixels = int(clamp["min_pixels"])
        max_pixels = int(clamp["max_pixels"])
        px_per_side = int(cfg.raster_params["tiles"]["px_per_side"])

        if len(download_df) == 0:
            return {
                "rasters_hash": plan.rasters_hash,
                "buffer_m": extent_buffer_m,
                "resolution_buffer_m": resolution_buffer_m,
                "extent_buffer_m": extent_buffer_m,
                "chips": {
                    "total_points": 0,
                    "downloaded": 0,
                    "existing": 0,
                    "failed": 0,
                    "failures": [],
                },
                "tiles": {},
            }

        chip_stats = mb2dl.download_chips(
            download_df,
            study_dir=self.study_dir,
            plan=plan,
            resolution_buffer_m=resolution_buffer_m,
            extent_buffer_m=extent_buffer_m,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            splits=splits,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )
        tile_stats = mb2dl.download_tiles(
            download_df,
            study_dir=self.study_dir,
            plan=plan,
            resolution_buffer_m=resolution_buffer_m,
            extent_buffer_m=extent_buffer_m,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            px_per_side=px_per_side,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

        return {
            "rasters_hash": plan.rasters_hash,
            "buffer_m": extent_buffer_m,
            "resolution_buffer_m": resolution_buffer_m,
            "extent_buffer_m": extent_buffer_m,
            "chips": chip_stats,
            "tiles": tile_stats,
        }

    def extract_point_features(
        self,
        *,
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        splits: list[str] | None = None,
        n_workers: int = 16,
        skip_existing: bool = True,
    ) -> dict:
        from movebench.features.points import extract_point_feature_stores

        plan = self.layer_plan(layers_config)
        sampler = _load_sampler(sampler_config)
        return extract_point_feature_stores(
            study_id=self.study_id,
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            sampler_config=sampler,
            splits=splits,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

    def chip_dataset(
        self,
        *,
        split: str = "train",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig | None = None,
        mode: str = "points",
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
    ):
        from movebench.datasets.chips import ChipDataset

        plan = self.layer_plan(layers_config)
        sampler = None if sampler_config is None else _load_sampler(sampler_config)
        return ChipDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            split=split,
            plan=plan,
            sampler_config=sampler,
            mode=mode,
            output_hw=output_hw,
            normalize=normalize,
        )

    def deepssf_dataset(
        self,
        *,
        split: str = "train",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
    ):
        from movebench.datasets.deepssf import DeepSSFDataset

        chips = self.chip_dataset(
            split=split,
            layers_config=layers_config,
            sampler_config=sampler_config,
            mode="steps",
            output_hw=output_hw,
            normalize=normalize,
        )
        return DeepSSFDataset(chips)

    def moveformer_dataset(
        self,
        *,
        split: str = "train",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        context_splits: list[str] | None = None,
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
    ):
        from movebench.datasets.moveformer import MoveFormerPointStoreDataset

        plan = self.layer_plan(layers_config)
        sampler = _load_sampler(sampler_config)
        return MoveFormerPointStoreDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            sampler_config=sampler,
            split=split,
            context_splits=context_splits,
            max_len=max_len,
            min_len=min_len,
            normalize_env=normalize_env,
            require_normalization_stats=require_normalization_stats,
            subset_band_ids=subset_band_ids,
        )

    def ssf_dataset(
        self,
        *,
        split: str = "train",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
    ):
        from movebench.datasets.ssf import SSFDataset

        plan = self.layer_plan(layers_config)
        sampler = _load_sampler(sampler_config)
        return SSFDataset(
            study_dir=self.study_dir,
            plan=plan,
            sampler_config=sampler,
            split=split,
            normalize=normalize,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
        )

    def issf_dataset(
        self,
        *,
        split: str = "train",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
        normalize_movement: bool = True,
        include_time: bool = False,
        include_individual_id: bool = False,
        movement_stats: dict[str, float] | None = None,
    ):
        from movebench.datasets.ssf import iSSFDataset

        plan = self.layer_plan(layers_config)
        sampler = _load_sampler(sampler_config)
        return iSSFDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            sampler_config=sampler,
            split=split,
            normalize=normalize,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
            normalize_movement=normalize_movement,
            include_time=include_time,
            include_individual_id=include_individual_id,
            movement_stats=movement_stats,
        )

    def compute_normalization_vals(
        self,
        *,
        from_: str = "points",
        layers_config: str | Path = "configs/layers/all.yaml",
        sampler_config: str | Path | SamplerConfig = "configs/samplers/issf_defaults.yaml",
        split: str = "train",
        skip_existing: bool = True,
    ) -> dict:
        from movebench.features.normalization import compute_points_normalization_stats

        if from_ != "points":
            raise NotImplementedError(
                f"Unsupported from_={from_!r}. Only 'points' is implemented."
            )
        plan = self.layer_plan(layers_config)
        sampler = _load_sampler(sampler_config)
        return compute_points_normalization_stats(
            study_dir=self.study_dir,
            plan=plan,
            sampler_config=sampler,
            split=split,
            skip_existing=skip_existing,
        )
