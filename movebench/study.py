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
    ) -> dict:
        """Download all required GEE rasters for a layers config into this study directory.

        Writes rasters to `study_dir/rasters/<rasters_hash>/...` as expected by `LayerIO`.
        """
        cfg = load_layers_config(config)
        plan = REGISTRY.resolve(cfg)

        # Imported lazily: heavy deps (pandas, EE, rasterio, requests).
        from movebench.layers import download as mb2dl

        import pandas as pd

        df = pd.read_csv(self.trajectories_path)
        df[mb2dl.TIME_COL] = pd.to_datetime(df[mb2dl.TIME_COL], errors="coerce")
        df = df.dropna(subset=[mb2dl.TIME_COL])

        train_df = df[df[mb2dl.SET_COL] == "train"]
        if len(train_df) == 0:
            raise ValueError("No training data found (set=='train'); cannot compute buffer")
        buf = cfg.raster_params["buffer"]
        buffer_km = (
            mb2dl.get_buffer_m_for_study(
                train_df,
                step_multiplier=float(buf["step_multiplier"]),
                step_q=float(buf["step_q"]),
                gap_q=float(buf["gap_q"]),
            )
            / 1000.0
        )

        if splits is None:
            splits = sorted(df[mb2dl.SET_COL].dropna().unique().tolist())

        rm = dict(cfg.raster_params["region_multiplier"])
        clamp = cfg.raster_params["scale_clamp"]
        min_pixels = int(clamp["min_pixels"])
        max_pixels = int(clamp["max_pixels"])
        px_per_side = int(cfg.raster_params["tiles"]["px_per_side"])

        chip_stats = mb2dl.download_chips(
            df,
            study_dir=self.study_dir,
            plan=plan,
            buffer_km=buffer_km,
            region_multiplier=rm,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            splits=splits,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

        tile_stats = mb2dl.download_tiles(
            df[df[mb2dl.SET_COL].isin(splits)].copy(),
            study_dir=self.study_dir,
            plan=plan,
            buffer_km=buffer_km,
            region_multiplier=rm,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            px_per_side=px_per_side,
            n_workers=n_workers,
            skip_existing=skip_existing,
        )

        return {
            "config_hash": plan.config_hash,
            "rasters_hash": plan.rasters_hash,
            "buffer_km": buffer_km,
            "chips": chip_stats,
            "tiles": tile_stats,
        }

    def download_point_values(
        self,
        *,
        config: str | Path = "configs/layers_all.yaml",
        splits: list[str] | None = None,
        n_workers: int = 32,
        batch_size: int = 500,
        skip_existing: bool = True,
    ) -> dict:
        """Sample GEE gee_chip layers at exact GPS coordinates for all points.

        Unlike ``download_rasters()``, this method:
        - Does not compute a buffer — no outlier sensitivity.
        - No step-length or time-gap filtering is applied to any GPS point.
        - Returns scalar covariate values stored in a flat Parquet file.

        Writes to:
            ``study_dir/point_values/<config_hash>/values.parquet``
            ``study_dir/point_values/<config_hash>/metadata.json``

        Args:
            config:        Path to a layers YAML config.
            splits:        Optional list of splits to process (e.g.
                           ``["train", "test"]``). If None, all points are processed.
            n_workers:     Number of multiprocessing workers.
            batch_size:    Max points per GEE sampleRegions call.
            skip_existing: If True, skip already-downloaded points.

        Returns:
            dict with keys: config_hash, parquet_path, n_points,
                            n_downloaded, n_existing, n_failed, failures.
        """
        cfg = load_layers_config(config)
        plan = REGISTRY.resolve(cfg)

        import pandas as pd
        from movebench.layers.sample_points import (
            TIME_COL,
            sample_point_values,
        )

        df = pd.read_csv(self.trajectories_path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

        return sample_point_values(
            df,
            study_dir=self.study_dir,
            plan=plan,
            registry=REGISTRY,
            splits=splits,
            n_workers=n_workers,
            batch_size=batch_size,
            skip_existing=skip_existing,
        )

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

    def deepssf_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        output_hw: tuple[int, int] = (101, 101),
        normalize: bool = True,
        normalization_sampler_hash: str | None = None,
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
    ):
        """Create a DeepSSFDataset for this study.

        This wraps `ChipDataset(mode="steps")` to provide deepSSF inputs:
        `(spatial, scalars_to_grid, bearing, target)`.
        """
        chips = self.chip_dataset(
            split=split,
            config=config,
            mode="steps",
            output_hw=output_hw,
            normalize=normalize,
            normalization_sampler_hash=normalization_sampler_hash,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
        )

        from movebench.datasets.deepssf import DeepSSFDataset

        return DeepSSFDataset(chips)

    def moveformer_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
        num_candidates: int | None = None,
        random_seed: int | None = None,
        step_multiplier: float | None = None,
        step_q: float | None = None,
        gap_q: float | None = None,
        bounds_buffer_deg: float | None = None,
    ):
        """Create a MoveFormer dataset backed by extracted iSSF point stores."""
        plan = self.layer_plan(config)

        from movebench.datasets.moveformer import MoveFormerPointStoreDataset

        return MoveFormerPointStoreDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            split=split,
            max_len=max_len,
            min_len=min_len,
            normalize_env=normalize_env,
            require_normalization_stats=require_normalization_stats,
            subset_band_ids=subset_band_ids,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
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
        include_speed: bool = False,
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
            include_speed=include_speed,
            movement_stats=movement_stats,
            num_candidates=num_candidates,
            random_seed=random_seed,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
            bounds_buffer_deg=bounds_buffer_deg,
        )

    def thumbnail_dataset(
        self,
        *,
        split: str | None = "train",
        img_size: int = 64,
        chunk_mode: str = "fixed",
        chunk_size: int = 50,
        stride: int | None = None,
        min_chunk_size: int = 10,
        burst_gap_seconds: float = 3600.0,
        line_width: int = 1,
        marker_radius: int = 1,
        padding: int = 2,
        scale_mode: str = "individual",
        scale_quantile: float = 0.5,
        basemap: str | None = None,
        transform=None,
    ):
        """Create a TrackThumbnailDataset for this study.

        Renders GPS trajectory chunks as MNIST-style B&W thumbnail images.
        Requires only ``trajectories.csv`` — no raster downloads needed.

        Each thumbnail shows a fixed-size chunk or, with ``chunk_mode="burst"``,
        one absolute time-gap burst drawn as a connected polyline with a marker
        at each fix.  By default a fixed per-individual spatial scale is used so
        that anomalously large chunks visibly overflow the image.

        Args:
            split:           "train", "test", or None for all splits.
            img_size:        Square image side in pixels (default 64).
            chunk_mode:      ``"fixed"`` (default) or ``"burst"``.
            chunk_size:      Fixes per thumbnail (default 50).
            stride:          Step between chunk starts; None = non-overlapping.
            min_chunk_size:  Drop chunks shorter than this (default 10).
            burst_gap_seconds:
                             In burst mode, start a new burst when the gap
                             between consecutive fixes is greater than this
                             many seconds (default 3600).
            line_width:      Polyline stroke width (default 1).
            marker_radius:   Radius of the filled circle drawn at each fix
                             (default 1). Set to 0 for polyline only.
            padding:         Blank border in pixels on each side (default 2).
            scale_mode:      ``"individual"`` (default) — fixed scale per
                             individual derived from its median chunk extent,
                             making large spatial anomalies visually obvious.
                             ``"chunk"`` — legacy per-chunk bounding-box scale.
            scale_quantile:  Quantile of per-chunk extents used when
                             ``scale_mode="individual"`` (default 0.5 = median).
            basemap:         Optional thumbnail basemap. Supported values are
                             ``"positron"`` (default), ``"osm"``,
                             ``"satellite"``, and ``"none"``.
            transform:       Optional torchvision-compatible transform applied
                             to the float32 tensor in __getitem__.
        """
        from movebench.datasets.thumbnail_dataset import TrackThumbnailDataset

        return TrackThumbnailDataset(
            trajectories_path=self.trajectories_path,
            split=split,
            img_size=img_size,
            chunk_mode=chunk_mode,
            chunk_size=chunk_size,
            stride=stride,
            min_chunk_size=min_chunk_size,
            burst_gap_seconds=burst_gap_seconds,
            line_width=line_width,
            marker_radius=marker_radius,
            padding=padding,
            scale_mode=scale_mode,
            scale_quantile=scale_quantile,
            basemap=basemap,
            transform=transform,
        )

    def anomaly_dataset(
        self,
        *,
        split: str = "train",
        config: str | Path = "configs/layers_all.yaml",
        include_env: bool = True,
        include_movement: bool = True,
        include_outgoing_movement: bool = False,
        normalize: bool = True,
        normalization_stats: dict | None = None,
        step_multiplier: float = 2.0,
        step_q: float = 0.95,
        gap_q: float = 0.999,
    ):
        """Create an AnomalyPointDataset backed by point_values parquet + trajectories.

        Step selection uses ``select_step_starts()`` from ``trajectories.py`` —
        identical to the iSSF pipeline — so ``step_multiplier``, ``step_q``, and
        ``gap_q`` have the same meaning as in ``extract_point_features()``.
        Each item corresponds to one valid step; the step-start GPS fix is
        identified by ``point_id_start`` (same naming as ``features/points.py``).

        No ``extract_point_features()`` step is needed; features are built
        on-the-fly from:
          - ``study_dir/point_values/<config_hash>/values.parquet`` (env covariates
            at ``point_id_start``), and
          - ``trajectories.csv`` (movement: step_length_m, step_dt_s, speed_m_s,
            turn_angle_rad — forward-looking, start → end).

        Requires ``Study.download_point_values()`` to have been run when
        ``include_env=True``.

        Args:
            split:               "train", "test", or None for all splits.
            config:              Path to a layers YAML config.
            include_env:         Include GEE environmental covariates.
            include_movement:    Include movement features.
            normalize:           Z-score normalize using this split's non-NaN stats.
            normalization_stats: Pre-computed stats dict (``train_ds.normalization_stats``)
                                 to apply to another split without recomputing.
            step_multiplier:     Same as in ``extract_point_features()``.
            step_q:              Same as in ``extract_point_features()``.
            gap_q:               Same as in ``extract_point_features()``.
        """
        plan = self.layer_plan(config)

        from movebench.datasets.anomaly import AnomalyPointDataset

        return AnomalyPointDataset(
            study_dir=self.study_dir,
            trajectories_path=self.trajectories_path,
            plan=plan,
            split=split,
            include_env=include_env,
            include_movement=include_movement,
            include_outgoing_movement=include_outgoing_movement,
            normalize=normalize,
            normalization_stats=normalization_stats,
            step_multiplier=step_multiplier,
            step_q=step_q,
            gap_q=gap_q,
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
