from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from movebench2 import paths
from movebench2.layers import REGISTRY
from movebench2.layers.config import load_layers_config
from movebench2.layers.plan import LayerPlan


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
        from movebench2.layers.io import LayerIO

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

        Writes rasters to `study_dir/rasters/<config_hash>/...` as expected by `LayerIO`.
        """
        plan = self.layer_plan(config)

        # Imported lazily: heavy deps (pandas, EE, rasterio, requests).
        from movebench2.layers import download as mb2dl

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

        return {"config_hash": plan.config_hash, "buffer_km": buffer_km, "chips": chip_stats, "tiles": tile_stats}
