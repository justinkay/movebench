from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from movebench2 import paths
from movebench2.download import download_chips_for_plan, download_tiles
from movebench2.layers import REGISTRY
from movebench2.layers.config import load_layers_config
from movebench2.layers.plan import LayerPlan
from movebench2.movement import SET_COL, TIME_COL, load_trajectories_csv, setup_df_for_ssf


def _required_layers_from_plan(plan: LayerPlan) -> set[str]:
    required: set[str] = set(plan.layers_in_order)
    required.update(plan.derived_dependencies.keys())
    changed = True
    while changed:
        changed = False
        for layer, parents in plan.derived_dependencies.items():
            if layer not in required:
                continue
            for p in parents:
                if p not in required:
                    required.add(p)
                    changed = True
    return required


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

    def layer_plan(self, config: str | Path = "configs/layers_all.yaml") -> LayerPlan:
        cfg = load_layers_config(config)
        return REGISTRY.resolve(cfg)

    def _trajectories_df(self):
        df = load_trajectories_csv(self.trajectories_path)
        # Parse timestamps once so downstream can rely on `.isoformat()` / `.year`.
        import pandas as pd

        if TIME_COL in df.columns:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
            df = df.dropna(subset=[TIME_COL])
        return df

    def download_rasters(
        self,
        *,
        config: str | Path = "configs/layers_all.yaml",
        n_workers: int = 16,
        skip_existing: bool = True,
        tile_scale_m: float = 30.0,
    ) -> dict[str, object]:
        cfg = load_layers_config(config)
        plan = REGISTRY.resolve(cfg)

        df = self._trajectories_df()

        required_layers = _required_layers_from_plan(plan)
        tile_layers = sorted([n for n in required_layers if REGISTRY.get(n).source == "gee_tile"])

        chips_stats = download_chips_for_plan(
            df=df,
            study_dir=self.study_dir,
            plan=plan,
            n_workers=int(n_workers),
            skip_existing=bool(skip_existing),
        )

        buffer_km = float(chips_stats.get("buffer_km", 0.0))
        if buffer_km <= 0.0:
            train_df = df[df[SET_COL] == "train"] if (SET_COL in df.columns) else df
            from movebench2.movement import get_buffer_m_for_study

            buffer_km = float(get_buffer_m_for_study(train_df) / 1000.0)
        tiles_stats = download_tiles(
            df=df,
            study_dir=self.study_dir,
            tile_layers=tile_layers,
            buffer_km=buffer_km,
            scale_m=float(tile_scale_m),
            skip_existing=bool(skip_existing),
        )

        return {
            "study_id": self.study_id,
            "config": str(config),
            "config_hash": plan.config_hash,
            "chip_groups": chips_stats,
            "tiles": tiles_stats,
        }

    def extract_point_features(
        self,
        *,
        config: str | Path = "configs/layers_all.yaml",
        parametric: bool = False,
        splits: tuple[str, ...] | None = None,
        num_candidates: int = 100,
        random_seed: int = 0,
        n_workers: int = 1,
        bounds_buffer_deg: float = 0.01,
    ) -> dict[str, object]:
        cfg = load_layers_config(config)
        plan = REGISTRY.resolve(cfg)

        df = self._trajectories_df()
        if splits is None:
            splits = ("train", "test") if (SET_COL in df.columns) else ("train",)

        # Fit / define sampler from training movement only, then reuse for all splits.
        import numpy as np

        train_df = df[df[SET_COL] == "train"] if (SET_COL in df.columns) else df
        df_train_ssf = setup_df_for_ssf(train_df)
        valid_lengths = df_train_ssf.loc[df_train_ssf["valid_step"], "step_length_m"].values
        valid_turns = df_train_ssf["turn_angle_rad"].dropna().values

        if len(valid_lengths) == 0 or len(valid_turns) == 0:
            raise ValueError("Not enough valid movement to build a sampler from training split")

        sampler: dict[str, Any] = {
            "type": "parametric" if parametric else "empirical",
            "num_candidates": int(num_candidates),
            "random_seed": int(random_seed),
            "fit_source_split": "train",
        }

        if parametric:
            try:
                from scipy import stats
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "SciPy is required for parametric sampling. Install with: pip install scipy"
                ) from e

            gamma_params = stats.gamma.fit(valid_lengths, floc=0)
            vonmises_params = stats.vonmises.fit(valid_turns)
            sampler["fit_params"] = {"gamma": list(map(float, gamma_params)), "vonmises": list(map(float, vonmises_params))}

        from movebench2.features.manifest import PointFeatureManifest, utc_now_iso, write_manifest
        from movebench2.features.points import extract_env_features_parallel, sampler_hash as _sampler_hash
        from movebench2.movement import LAT_COL, LON_COL, POINT_ID_COL, TARGET_CRS

        sampler_id = _sampler_hash(sampler)

        # Candidate generation helpers
        try:
            from pyproj import Transformer
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "pyproj is required for candidate point generation. Install with: pip install pyproj"
            ) from e

        transformer_to_m = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
        transformer_to_4326 = Transformer.from_crs(TARGET_CRS, "EPSG:4326", always_xy=True)

        def wrap_pi(x):
            return (x + np.pi) % (2.0 * np.pi) - np.pi

        report: dict[str, object] = {
            "study_id": self.study_id,
            "config": str(config),
            "config_hash": plan.config_hash,
            "sampler_hash": sampler_id,
            "sampler": sampler,
            "splits": {},
        }

        for split in splits:
            split_df = df[df[SET_COL] == split] if (SET_COL in df.columns) else df
            df_ssf = setup_df_for_ssf(split_df)
            sample_indices = df_ssf.index[df_ssf["can_be_ssf_start"]].to_numpy()

            # Stable per-split RNG so running one split doesn't change another.
            seed_material = f"{random_seed}|{split}".encode("utf-8")
            split_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
            rng = np.random.default_rng(split_seed)

            # Parametric sampler objects
            if parametric:
                from scipy import stats

                gamma_params = tuple(sampler["fit_params"]["gamma"])  # type: ignore[assignment]
                vonmises_params = tuple(sampler["fit_params"]["vonmises"])  # type: ignore[assignment]

            n_steps = int(len(sample_indices))
            P = int(num_candidates + 1)

            samples: list[dict[str, Any]] = []
            step_lengths = np.zeros((n_steps, P), dtype=np.float32)
            turn_angles = np.zeros((n_steps, P), dtype=np.float32)
            point_ids = np.zeros((n_steps,), dtype=np.int64)
            timestamps_ns = np.zeros((n_steps,), dtype=np.int64)

            for i, orig_idx in enumerate(sample_indices):
                row = df_ssf.loc[int(orig_idx)]

                start_lat = float(row[LAT_COL])
                start_lon = float(row[LON_COL])
                start_bearing = float(row["bearing"])

                next_idx = int(row["next_index"])
                next_row = df_ssf.loc[next_idx]
                end_lat = float(next_row[LAT_COL])
                end_lon = float(next_row[LON_COL])

                # Sample pseudo movement.
                if parametric:
                    pseudo_lens = stats.gamma.rvs(*gamma_params, size=num_candidates, random_state=rng).astype(np.float64)
                    pseudo_turns = stats.vonmises.rvs(*vonmises_params, size=num_candidates, random_state=rng).astype(np.float64)
                else:
                    pseudo_lens = rng.choice(valid_lengths, size=num_candidates, replace=True).astype(np.float64)
                    pseudo_turns = rng.choice(valid_turns, size=num_candidates, replace=True).astype(np.float64)

                # Work in metric CRS for step lengths.
                x0, y0 = transformer_to_m.transform(start_lon, start_lat)
                x1, y1 = transformer_to_m.transform(end_lon, end_lat)

                headings = wrap_pi(start_bearing + pseudo_turns)
                dx = pseudo_lens * np.sin(headings)
                dy = pseudo_lens * np.cos(headings)
                xs = x0 + dx
                ys = y0 + dy

                true_sl = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                true_heading = math.atan2(x1 - x0, y1 - y0)
                true_ta = float(wrap_pi(true_heading - start_bearing))

                xs = np.append(xs, x1)
                ys = np.append(ys, y1)
                sl = np.append(pseudo_lens, true_sl)
                ta = np.append(pseudo_turns, true_ta)

                lons, lats = transformer_to_4326.transform(xs, ys)
                lons = np.asarray(lons, dtype=np.float64)
                lats = np.asarray(lats, dtype=np.float64)

                bounds_4326 = (
                    max(-180.0, float(np.min(lons)) - bounds_buffer_deg),
                    max(-90.0, float(np.min(lats)) - bounds_buffer_deg),
                    min(180.0, float(np.max(lons)) + bounds_buffer_deg),
                    min(90.0, float(np.max(lats)) + bounds_buffer_deg),
                )

                samples.append(
                    {
                        "point_id": int(row[POINT_ID_COL]),
                        "timestamp": row[TIME_COL],
                        "lons": lons,
                        "lats": lats,
                        "bounds_4326": bounds_4326,
                    }
                )

                step_lengths[i] = sl.astype(np.float32)
                turn_angles[i] = ta.astype(np.float32)
                point_ids[i] = int(row[POINT_ID_COL])
                timestamps_ns[i] = int(row[TIME_COL].value)  # pandas Timestamp ns since epoch (UTC if tz-aware)

            env, valid = extract_env_features_parallel(
                study_dir=self.study_dir,
                plan=plan,
                samples=samples,
                n_workers=int(n_workers),
            )

            try:
                import torch
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "PyTorch is required to write feature stores. Install with: pip install torch"
                ) from e

            out_dir = self.study_dir / "features" / "points" / split / plan.config_hash / sampler_id
            out_dir.mkdir(parents=True, exist_ok=True)
            features_path = out_dir / "features.pt"
            manifest_path = out_dir / "manifest.json"

            tensors = {
                "env": {"shape": [n_steps, plan.num_channels(), P], "dtype": "float32"},
                "valid": {"shape": [n_steps, plan.num_channels(), P], "dtype": "bool"},
                "step_length_m": {"shape": [n_steps, P], "dtype": "float32"},
                "turn_angle_rad": {"shape": [n_steps, P], "dtype": "float32"},
                "point_id_start": {"shape": [n_steps], "dtype": "int64"},
                "timestamp_ns": {"shape": [n_steps], "dtype": "int64"},
            }

            data = {
                "env": torch.from_numpy(env),
                "valid": torch.from_numpy(valid),
                "step_length_m": torch.from_numpy(step_lengths),
                "turn_angle_rad": torch.from_numpy(turn_angles),
                "point_id_start": torch.from_numpy(point_ids),
                "timestamp_ns": torch.from_numpy(timestamps_ns),
            }
            torch.save(data, features_path)

            manifest = PointFeatureManifest(
                schema_version="v1",
                study_id=self.study_id,
                split=split,
                created_at=utc_now_iso(),
                config_path=str(config),
                config_hash=plan.config_hash,
                sampler_hash=sampler_id,
                layer_plan=plan,
                sampler=sampler,
                extraction={
                    "points_per_step": P,
                    "bounds_buffer_deg": float(bounds_buffer_deg),
                    "choice_index": int(num_candidates),  # true endpoint appended last
                    "target_crs": TARGET_CRS,
                },
                tensors=tensors,
            )
            write_manifest(manifest_path, manifest)

            report["splits"][split] = {
                "n_steps": n_steps,
                "points_per_step": P,
                "features_path": str(features_path),
                "manifest_path": str(manifest_path),
            }

        return report

    def compute_normalization_vals(
        self,
        *,
        source: str = "points",
        config: str | Path = "configs/layers_all.yaml",
        sampler_hash: str | None = None,
    ) -> dict[str, object]:
        if source != "points":
            raise ValueError(f"Unsupported normalization source: {source!r} (expected 'points')")

        plan = self.layer_plan(config)
        base = self.study_dir / "features" / "points" / "train" / plan.config_hash
        if not base.exists():
            raise FileNotFoundError(f"Train point features not found at: {base}")

        if sampler_hash is None:
            candidates = sorted([p for p in base.iterdir() if p.is_dir()])
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly 1 sampler under {base}, found {len(candidates)}. "
                    "Pass sampler_hash=... explicitly."
                )
            sampler_dir = candidates[0]
        else:
            sampler_dir = base / sampler_hash

        features_path = sampler_dir / "features.pt"
        if not features_path.exists():
            raise FileNotFoundError(f"features.pt not found: {features_path}")

        try:
            import torch
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "PyTorch is required to compute normalization from feature stores. Install with: pip install torch"
            ) from e

        data = torch.load(features_path, map_location="cpu")
        env = data["env"]  # (n, C, P)
        valid = data["valid"]  # (n, C, P)
        n, C, P = env.shape

        band_ids = plan.band_ids_in_order()
        if len(band_ids) != C:
            raise ValueError("features.pt env.shape[1] does not match LayerPlan channel count")

        means = []
        stds = []
        counts = []

        for c in range(C):
            ch = plan.channels[c]
            if not ch.normalize:
                means.append(0.0)
                stds.append(1.0)
                counts.append(0)
                continue

            vals = env[:, c, :]
            m = valid[:, c, :]
            valid_vals = vals[m]
            if valid_vals.numel() == 0:
                means.append(0.0)
                stds.append(1.0)
                counts.append(0)
                continue

            mean = float(valid_vals.mean().item())
            std = float(valid_vals.std().item())
            if std < 1e-8:
                std = 1.0
            means.append(mean)
            stds.append(std)
            counts.append(int(valid_vals.numel()))

        band_stats = {
            band_ids[i]: {"mean": means[i], "std": stds[i], "count": counts[i]}
            for i in range(C)
            if plan.channels[i].normalize
        }

        out_dir = self.study_dir / "normalization" / "points" / plan.config_hash
        out_dir.mkdir(parents=True, exist_ok=True)
        sampler_id = sampler_dir.name
        out_path = out_dir / f"{sampler_id}.json"

        from movebench2.features.manifest import utc_now_iso

        payload = {
            "schema_version": "v1",
            "study_id": self.study_id,
            "source": "points",
            "split": "train",
            "created_at": utc_now_iso(),
            "config": str(config),
            "config_hash": plan.config_hash,
            "sampler_hash": sampler_id,
            "band_stats": band_stats,
            "n_steps": int(n),
            "points_per_step": int(P),
        }

        import json

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

        return {"path": str(out_path), **payload}
