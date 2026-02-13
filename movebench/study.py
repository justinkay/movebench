from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from movebench import paths
from movebench.layers import REGISTRY
from movebench.layers.config import load_layers_config
from movebench.layers.plan import LayerPlan


@dataclass(frozen=True)
class Study:
    """A single movement study identified by `study_id` within the studies root."""

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

    def layer_plan(self, config: str | Path = "configs/layers_all.yaml") -> LayerPlan:
        cfg = load_layers_config(config)
        return REGISTRY.resolve(cfg)

    def layer_io(self, config: str | Path = "configs/layers_all.yaml"):
        """Construct a `LayerIO` for this study and a layers config.

        This is a convenience entrypoint that wires together:
        - `LayerPlan` (from the YAML config), and
        - `LayerIO` (to load rasters from disk/local datasets using the plan).
        """

        plan = self.layer_plan(config)

        # Imported lazily to keep `Study` lightweight and avoid import cycles.
        from movebench.layers.io import LayerIO

        return LayerIO(study_dir=self.study_dir, plan=plan)

    def download_rasters(
        self,
        *,
        config: str | Path = "configs/layers_all.yaml",
        splits: list[str] | None = None,
        n_workers: int = 16,
        skip_existing: bool = True,
        px_per_side: int = 500,
    ) -> dict:
        """Download all required GEE rasters for a layers config into this study directory.

        Writes rasters to `study_dir/rasters/<rasters_hash>/...` as expected by `LayerIO`.
        """
        plan = self.layer_plan(config)

        # Imported lazily: heavy deps (pandas, EE, rasterio, requests).
        from movebench.layers import download as mb2dl

        import pandas as pd

        df = pd.read_csv(self.trajectories_path)
        df[mb2dl.TIME_COL] = pd.to_datetime(df[mb2dl.TIME_COL], errors="coerce")
        df = df.dropna(subset=[mb2dl.TIME_COL])

        train_df = df[df[mb2dl.SET_COL] == "train"]
        if len(train_df) == 0:
            raise ValueError("No training data found (set=='train'); cannot compute buffer")
        buffer_km = mb2dl.get_buffer_m_for_study(train_df) / 1000.0

        if splits is None:
            splits = sorted(df[mb2dl.SET_COL].dropna().unique().tolist())

        chip_stats = mb2dl.download_chips(
            df,
            study_dir=self.study_dir,
            plan=plan,
            buffer_km=buffer_km,
            splits=splits,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

        tile_stats = mb2dl.download_tiles(
            df[df[mb2dl.SET_COL].isin(splits)].copy(),
            study_dir=self.study_dir,
            plan=plan,
            buffer_km=buffer_km,
            px_per_side=px_per_side,
            skip_existing=skip_existing,
        )

        return {
            "config_hash": plan.config_hash,
            "rasters_hash": plan.rasters_hash,
            "buffer_km": buffer_km,
            "chips": chip_stats,
            "tiles": tile_stats,
        }

    def extract_point_features(
        self,
        *,
        config: str | Path = "configs/layers_all.yaml",
        splits: list[str] | None = None,
        parametric: bool = False,
        num_candidates: int = 100,
        random_seed: int = 0,
        n_workers: int = 16,
        skip_existing: bool = True,
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
        bounds_buffer_deg: float = 0.01,
    ) -> dict:
        """Extract SSF / iSSF point feature stores for a layers config.

        Writes to:
          `study_dir/features/points/<split>/<config_hash>/<sampler_hash>/features.pt`
          `study_dir/features/points/<split>/<config_hash>/<sampler_hash>/manifest.json`
        """
        plan = self.layer_plan(config)

        # Imported lazily: heavy deps (numpy/pandas/torch/scipy).
        from movebench.features.points import extract_point_feature_stores

        return extract_point_feature_stores(
            study_id=self.study_id,
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            config_path=str(config),
            splits=splits,
            parametric=parametric,
            num_candidates=num_candidates,
            random_seed=random_seed,
            n_workers=n_workers,
            skip_existing=skip_existing,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

    def chip_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        mode: str = "points",
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
        normalization_sampler_hash: str | None = None,
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
    ):
        """Create a ChipDataset for this study.

        Loads full raster stacks around each point, aligned to a metric CRS.
        """
        plan = self.layer_plan(config)

        # Imported lazily to avoid torch/rasterio imports when unused.
        from movebench.datasets.chips import ChipDataset

        return ChipDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            split=split,
            plan=plan,
            mode=mode,
            output_hw=output_hw,
            normalize=normalize,
            normalization_sampler_hash=normalization_sampler_hash,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
        )

    def ssf_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ):
        """Create an SSFDataset (empirical sampling) from extracted point features."""
        plan = self.layer_plan(config)

        from movebench.datasets.ssf import SSFDataset

        return SSFDataset(
            study_dir=self.study_dir,
            plan=plan,
            split=split,
            normalize=normalize,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

    def issf_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        normalize: bool = True,
        subset_env_layers: list[str] | None = None,
        subset_band_ids: list[str] | None = None,
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
    ):
        """Create an iSSFDataset (parametric sampling) from extracted point features."""
        plan = self.layer_plan(config)

        from movebench.datasets.ssf import iSSFDataset

        return iSSFDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            split=split,
            normalize=normalize,
            subset_env_layers=subset_env_layers,
            subset_band_ids=subset_band_ids,
            normalize_movement=normalize_movement,
            include_time=include_time,
            include_individual_id=include_individual_id,
            movement_stats=movement_stats,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

    def compute_normalization_vals(
        self,
        *,
        from_: str = "points",
        config: str | Path = "configs/layers_all.yaml",
        sampler_hash: str | None = None,
        split: str = "train",
        skip_existing: bool = True,
    ) -> dict:
        """Compute and cache normalization statistics for environmental features."""
        if from_ != "points":
            raise NotImplementedError(f"Unsupported from_={from_!r}. Only 'points' is implemented.")

        plan = self.layer_plan(config)

        from movebench.features.normalization import compute_points_normalization_stats

        return compute_points_normalization_stats(
            study_dir=self.study_dir,
            plan=plan,
            sampler_hash=sampler_hash,
            split=split,
            skip_existing=skip_existing,
        )
