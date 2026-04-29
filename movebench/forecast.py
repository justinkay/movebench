from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import movebench.bench.config as bench_config
import movebench.paths as paths
from movebench.bench.run import load_saved_run_bundle
from movebench.layers.config import DEFAULT_RASTER_PARAMS
from movebench.layers.constants import (
    GEE_NODATA_SENTINEL,
    METERS_PER_DEG_LAT,
    METERS_PER_DEG_LON_EQUATOR,
    WGS84_MAX_LAT,
    WGS84_MAX_LON,
    WGS84_MIN_LAT,
    WGS84_MIN_LON,
)
from movebench.layers.registry import REGISTRY, LayerRegistry
from movebench.layers.plan import LayerPlan
from movebench.trajectories import (
    SET_COL,
    bearing_rad,
    canonicalize_split_key,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)


_SUPPORTED_METHODS = {"issf_linear", "issf_mlp", "deepssf", "moveformer"}


@dataclass(frozen=True)
class RecursiveForecastConfig:
    run_dir: Path | str
    split_key: str
    test_point_index: int
    num_trajectories: int
    num_steps: int
    burn_in_steps: int = 0
    seed: int = 0


@dataclass
class RecursiveForecastResult:
    run_key: str
    method: str
    split_key: str
    seed_point_id_start: int
    step_time_s: int
    pixel_size_m: float
    lon: torch.Tensor
    lat: torch.Tensor
    timestamp_ns: torch.Tensor
    sampled_row: torch.Tensor
    sampled_col: torch.Tensor


@dataclass(frozen=True)
class _SeedContext:
    row: Any
    step_time_s: int
    buffer_m: float
    pixel_size_m: float


@dataclass(frozen=True)
class _ForecastState:
    lon: torch.Tensor
    lat: torch.Tensor
    timestamp_ns: torch.Tensor
    start_bearing_rad: torch.Tensor
    individual_id: list[str]


@dataclass
class _GridBatch:
    env: torch.Tensor
    valid: torch.Tensor


@dataclass
class _ISSFForecastScorer:
    model: torch.nn.Module
    feature_adapter: Any
    sampler_state: dict[str, Any]
    plan: LayerPlan
    sampler_config: Any
    output_hw: tuple[int, int]
    pixel_size_m: float
    device: torch.device
    registry: LayerRegistry
    normalize_env: bool = True

    def score_grid(self, *, grid_batch: _GridBatch, state: _ForecastState, step_index: int) -> torch.Tensor:
        del step_index
        import movebench.bench.executors.ssf_grid as ssf_grid

        height, width = self.output_hw
        log_q_hw, _off_grid_mass = ssf_grid.movement_log_prior_hw(
            start_bearing_rad=state.start_bearing_rad,
            output_hw=self.output_hw,
            pixel_size_m=float(self.pixel_size_m),
            sampler_state=self.sampler_state,
        )
        n_steps = int(state.lon.shape[0])
        flat_env = grid_batch.env.view(n_steps, int(grid_batch.env.shape[1]), height * width)
        flat_valid = grid_batch.valid.view(n_steps, int(grid_batch.valid.shape[1]), height * width)
        step_length_m, turn_angle_rad = ssf_grid.movement_features_flat(
            start_bearing_rad=state.start_bearing_rad,
            output_hw=self.output_hw,
            pixel_size_m=float(self.pixel_size_m),
        )
        model_batch: dict[str, torch.Tensor | None] = {
            "env": flat_env.to(dtype=torch.float32),
            "valid": flat_valid.to(dtype=torch.bool),
            "move": self.feature_adapter._movement_tensor(
                step_length_m=step_length_m,
                turn_angle_rad=turn_angle_rad,
            ),
            "time": None,
            "individual_idx": None,
        }
        if bool(getattr(self.feature_adapter, "include_time", False)):
            model_batch["time"] = self.feature_adapter._time_tensor(state.timestamp_ns)
        if bool(getattr(self.feature_adapter, "include_individual_id", False)):
            model_batch["individual_idx"] = self.feature_adapter._individual_idx_tensor(
                np.asarray(state.individual_id, dtype=object)
            )
        self.model.eval()
        with torch.no_grad():
            batch_device = {
                key: None if value is None else value.to(self.device)
                for key, value in model_batch.items()
            }
            logits = self.model(batch_device)
        return log_q_hw + logits.detach().cpu().view(n_steps, height, width)

    def observe_sampled(self, **kwargs) -> None:
        del kwargs


@dataclass
class _DeepSSFForecastScorer:
    model: torch.nn.Module
    plan: LayerPlan
    sampler_config: Any
    output_hw: tuple[int, int]
    pixel_size_m: float
    device: torch.device
    registry: LayerRegistry
    normalize_env: bool = True

    def score_grid(self, *, grid_batch: _GridBatch, state: _ForecastState, step_index: int) -> torch.Tensor:
        del step_index
        spatial = grid_batch.env.to(device=self.device, dtype=torch.float32)
        scalars = _deepssf_time_features_from_timestamp_ns(state.timestamp_ns).to(
            device=self.device,
            dtype=torch.float32,
        )
        bearing = _deepssf_bearing_from_movebench(state.start_bearing_rad).to(
            device=self.device,
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            out = self.model((spatial, scalars, bearing))
        if out.ndim != 4 or int(out.shape[-1]) != 2:
            raise ValueError(
                f"Expected deepSSF output with shape (B,H,W,2); got {tuple(out.shape)}"
            )
        return (out[..., 0] + out[..., 1]).detach().cpu()

    def observe_sampled(self, **kwargs) -> None:
        del kwargs


@dataclass
class _MoveFormerForecastScorer:
    model: torch.nn.Module
    seed: Any
    histories: list[list[Any]]
    plan: LayerPlan
    sampler_config: Any
    output_hw: tuple[int, int]
    pixel_size_m: float
    device: torch.device
    registry: LayerRegistry
    split_key: str
    normalize_env: bool = True

    def score_grid(self, *, grid_batch: _GridBatch, state: _ForecastState, step_index: int) -> torch.Tensor:
        del step_index
        query_rows = torch.cat(
            [
                _moveformer_query_row(
                    model=self.model,
                    seed=self.seed,
                    records=history,
                    device=self.device,
                )
                for history in self.histories
            ],
            dim=0,
        )
        candidate_features = _moveformer_candidate_features(
            grid_batch=grid_batch,
            state=state,
            output_hw=self.output_hw,
            pixel_size_m=float(self.pixel_size_m),
            mov_vec_scale=float(self.seed.mov_vec_scale),
            device=self.device,
        )
        scores_flat = _moveformer_selection_scores(
            self.model,
            query_rows=query_rows.to(self.device, dtype=torch.float32),
            candidate_features=candidate_features,
        )
        return scores_flat.detach().cpu().reshape(-1, int(self.output_hw[0]), int(self.output_hw[1]))

    def observe_sampled(
        self,
        *,
        grid_batch: _GridBatch,
        state: _ForecastState,
        sampled_row: torch.Tensor,
        sampled_col: torch.Tensor,
        next_lon: torch.Tensor,
        next_lat: torch.Tensor,
        next_timestamp_ns: torch.Tensor,
        step_index: int,
    ) -> None:
        from movebench.datasets.moveformer import _StepRecord

        points_per_step = int(self.seed.num_candidates_total)
        for trajectory_idx, history in enumerate(self.histories):
            row = int(sampled_row[trajectory_idx].item())
            col = int(sampled_col[trajectory_idx].item())
            chosen_env = grid_batch.env[trajectory_idx, :, row, col].detach().cpu()
            chosen_valid = grid_batch.valid[trajectory_idx, :, row, col].detach().cpu()
            point_id_start = -1_000_000_000 - (trajectory_idx * 100_000) - int(step_index)
            history.append(
                _StepRecord(
                    split_key=str(self.split_key),
                    point_id_start=int(point_id_start),
                    point_id_end=int(point_id_start - 1),
                    timestamp_ns=int(state.timestamp_ns[trajectory_idx].item()),
                    timestamp_end_ns=int(next_timestamp_ns[trajectory_idx].item()),
                    start_lon=float(state.lon[trajectory_idx].item()),
                    start_lat=float(state.lat[trajectory_idx].item()),
                    env=chosen_env.view(-1, 1).expand(-1, points_per_step).contiguous(),
                    valid=chosen_valid.view(-1, 1).expand(-1, points_per_step).contiguous(),
                    candidate_lon=next_lon[trajectory_idx].detach().cpu().repeat(points_per_step),
                    candidate_lat=next_lat[trajectory_idx].detach().cpu().repeat(points_per_step),
                )
            )


def _timestamp_ns(value: Any) -> int:
    return int(pd.Timestamp(value).value)


def _deepssf_time_features_from_timestamp_ns(timestamp_ns: torch.Tensor) -> torch.Tensor:
    ts = pd.to_datetime(
        timestamp_ns.detach().cpu().numpy().astype(np.int64, copy=False),
        unit="ns",
        utc=True,
    )
    doy = ts.dayofyear.to_numpy(dtype=np.float64)
    hour = ts.hour.to_numpy(dtype=np.float64)
    minute = ts.minute.to_numpy(dtype=np.float64)
    second = ts.second.to_numpy(dtype=np.float64)
    hour_frac = hour + minute / 60.0 + second / 3600.0
    two_pi = 2.0 * math.pi
    return torch.from_numpy(
        np.stack(
            [
                np.sin(two_pi * hour_frac / 24.0),
                np.cos(two_pi * hour_frac / 24.0),
                np.sin(two_pi * doy / 365.25),
                np.cos(two_pi * doy / 365.25),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
    )


def _deepssf_bearing_from_movebench(start_bearing_rad: torch.Tensor) -> torch.Tensor:
    bearing = (math.pi / 2.0) - start_bearing_rad.to(dtype=torch.float32)
    bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
    return bearing.view(-1, 1)


def _moveformer_query_row(
    *,
    model: torch.nn.Module,
    seed: Any,
    records: list[Any],
    device: torch.device,
) -> torch.Tensor:
    from movebench.lib.moveformer.geo_transformers.training_utils import (
        collate_sequence_dicts,
    )

    max_len = int(getattr(model, "max_len", len(records)) or len(records))
    if max_len > 0 and len(records) > max_len:
        previous_record = records[-max_len - 1]
        records = records[-max_len:]
    else:
        previous_record = None
    encoded = _encode_moveformer_records(
        seed=seed,
        records=records,
        previous_record=previous_record,
    )
    batch = collate_sequence_dicts([encoded])
    features = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad():
        _embeddings, dec_out = model(features, run_prediction=False)
    return dec_out[:, -1, :].detach().to(dtype=torch.float32)


def _encode_moveformer_records(
    *,
    seed: Any,
    records: list[Any],
    previous_record: Any | None,
) -> dict[str, Any]:
    from movebench.datasets.moveformer import _coords_m, _to_datetime64

    window = list(records)
    seq_env = torch.stack([record.env for record in window], dim=0)
    seq_valid = torch.stack([record.valid for record in window], dim=0)
    candidate_lon = torch.stack([record.candidate_lon for record in window], dim=0)
    candidate_lat = torch.stack([record.candidate_lat for record in window], dim=0)
    cov_cand = seq_env.permute(0, 2, 1).numpy().astype(np.float32, copy=False)
    cov_valid = seq_valid.permute(0, 2, 1).numpy()
    cov_cand = np.where(cov_valid, cov_cand, np.float32(np.nan)).astype(
        np.float32,
        copy=False,
    )
    cov = np.full((len(window), int(seed.cov_dim)), np.float32(np.nan), dtype=np.float32)
    if previous_record is not None and len(window) > 0:
        prev_env = previous_record.env[:, 0].numpy().astype(np.float32, copy=False)
        prev_valid = previous_record.valid[:, 0].numpy()
        cov[0] = np.where(prev_valid, prev_env, np.float32(np.nan)).astype(
            np.float32,
            copy=False,
        )
    for idx in range(1, len(window)):
        prev_record = window[idx - 1]
        prev_env = prev_record.env[:, 0].numpy().astype(np.float32, copy=False)
        prev_valid = prev_record.valid[:, 0].numpy()
        cov[idx] = np.where(prev_valid, prev_env, np.float32(np.nan)).astype(
            np.float32,
            copy=False,
        )
    start_lon = np.asarray([record.start_lon for record in window], dtype=np.float64)
    start_lat = np.asarray([record.start_lat for record in window], dtype=np.float64)
    traj: dict[str, Any] = {
        "timestamp": _to_datetime64([record.timestamp_ns for record in window]),
        "timestamp_tgt": _to_datetime64([record.timestamp_end_ns for record in window]),
        "location_long": start_lon,
        "location_lat": start_lat,
        "location_long_tgt": candidate_lon[:, 0].numpy().astype(np.float64, copy=False),
        "location_lat_tgt": candidate_lat[:, 0].numpy().astype(np.float64, copy=False),
        "location_long_cand": candidate_lon.numpy().astype(np.float64, copy=False),
        "location_lat_cand": candidate_lat.numpy().astype(np.float64, copy=False),
        "taxon": np.zeros((len(window), 100), dtype=np.float32),
        "cov": cov,
        "cov_cand": cov_cand,
    }
    encoded = seed.processor.encode(traj)
    encoded["loss_mask"] = np.zeros((len(window),), dtype=np.bool_)
    encoded["prediction_mask"] = np.zeros((len(window),), dtype=np.bool_)
    encoded["prediction_mask"][-1] = True
    encoded["point_id_start"] = np.asarray(
        [record.point_id_start for record in window],
        dtype=np.int64,
    )
    encoded["coords_m"] = _coords_m(
        start_lon=start_lon,
        start_lat=start_lat,
        candidate_lon=candidate_lon,
        candidate_lat=candidate_lat,
    ).astype(np.float32, copy=False)
    return encoded


def _moveformer_candidate_features(
    *,
    grid_batch: _GridBatch,
    state: _ForecastState,
    output_hw: tuple[int, int],
    pixel_size_m: float,
    mov_vec_scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from pyproj import Transformer

    from movebench.lib.moveformer.geo_transformers.geo_utils import (
        gcs_to_bearing,
        gcs_to_distance,
        gcs_to_n_vector,
        to_rad,
    )

    height, width = (int(output_hw[0]), int(output_hw[1]))
    x_coords = np.asarray(
        [(col - (width // 2)) * float(pixel_size_m) for col in range(width)],
        dtype=np.float64,
    )
    y_coords = np.asarray(
        [((height // 2) - row) * float(pixel_size_m) for row in range(height)],
        dtype=np.float64,
    )
    x_hw, y_hw = np.meshgrid(x_coords, y_coords, indexing="xy")
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    from_3857 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    start_lon = state.lon.detach().cpu().numpy().astype(np.float64, copy=False)
    start_lat = state.lat.detach().cpu().numpy().astype(np.float64, copy=False)
    start_x, start_y = to_3857.transform(start_lon, start_lat)
    cand_x = np.asarray(start_x, dtype=np.float64).reshape(-1, 1, 1) + x_hw[None]
    cand_y = np.asarray(start_y, dtype=np.float64).reshape(-1, 1, 1) + y_hw[None]
    cand_lon, cand_lat = from_3857.transform(cand_x.reshape(-1), cand_y.reshape(-1))
    cand_lon = np.asarray(cand_lon, dtype=np.float64).reshape(-1, height * width)
    cand_lat = np.asarray(cand_lat, dtype=np.float64).reshape(-1, height * width)

    loc_x, loc_y, loc_z = gcs_to_n_vector(cand_lat, cand_lon)
    location_cand = torch.from_numpy(
        np.stack([loc_x, loc_y, loc_z], axis=-1).astype(np.float32, copy=False)
    ).to(device=device)
    bearing = to_rad(gcs_to_bearing(start_lat[:, None], start_lon[:, None], cand_lat, cand_lon))
    distance_m = gcs_to_distance(
        start_lat[:, None],
        start_lon[:, None],
        cand_lat,
        cand_lon,
    )[..., None]
    location_cand_mov = torch.from_numpy(
        (
            np.stack([np.sin(bearing), np.cos(bearing)], axis=-1)
            * distance_m
            * float(mov_vec_scale)
        ).astype(np.float32, copy=False)
    ).to(device=device)
    cov_cand = grid_batch.env.to(device=device, dtype=torch.float32).permute(0, 2, 3, 1).reshape(
        int(grid_batch.env.shape[0]),
        height * width,
        int(grid_batch.env.shape[1]),
    )
    cov_valid = grid_batch.valid.to(device=device, dtype=torch.bool).permute(0, 2, 3, 1).reshape_as(cov_cand)
    cov_cand = torch.where(cov_valid, cov_cand, torch.full_like(cov_cand, torch.nan))
    return {
        "location_cand": location_cand,
        "location_cand_mov": location_cand_mov,
        "cov_cand": cov_cand,
    }


def _moveformer_candidate_feature_embedding(
    model: torch.nn.Module,
    *,
    key: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if bool(getattr(model, "learnable_nan", False)):
        nan_fill = getattr(model, "nan_fill", None)
        if nan_fill is not None and key in nan_fill:
            value = torch.where(
                torch.isfinite(value),
                value,
                nan_fill[key].to(dtype=value.dtype),
            )
    return model.cont_proj[key](value)


def _moveformer_selection_scores(
    model: torch.nn.Module,
    *,
    query_rows: torch.Tensor,
    candidate_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    head = model.prediction_head
    embeddings = [
        _moveformer_candidate_feature_embedding(
            model,
            key=key,
            value=candidate_features[key],
        )
        for key in head.embedding_in_keys
    ]
    cand_emb = head.combine_emb(torch.stack(embeddings, dim=-1)).squeeze(-1)
    cand_emb = torch.cat(
        [
            cand_emb,
            query_rows.unsqueeze(-2).expand(
                int(query_rows.shape[0]),
                int(cand_emb.shape[1]),
                -1,
            ),
        ],
        dim=-1,
    )
    cand_emb = head.candidate_encoder(cand_emb)
    query = head.query_proj(query_rows)
    scale = float(query.shape[-1]) ** -0.5
    return torch.einsum("ncd,nd->nc", cand_emb, query) * scale


def _support_pixel_size_m(*, output_hw: tuple[int, int], buffer_m: float) -> float:
    if int(output_hw[0]) != int(output_hw[1]):
        raise ValueError(f"expected square output_hw, got {list(output_hw)}")
    if int(output_hw[1]) < 2:
        raise ValueError(f"expected output_hw width >= 2, got {list(output_hw)}")
    return (2.0 * float(buffer_m)) / float(int(output_hw[1]) - 1)


def _synthetic_tile_registry(
    plan,
    *,
    registry: LayerRegistry = REGISTRY,
) -> LayerRegistry:
    layers = {name: registry.get(name) for name in registry.names()}
    for layer in _gee_source_layers(plan, registry=registry):
        spec = registry.get(layer)
        if spec.source == "gee_chip":
            layers[layer] = replace(spec, source="gee_tile")
    return LayerRegistry(layers)


def _subset_layer_plan(plan: LayerPlan, band_ids: list[str]) -> LayerPlan:
    wanted = {str(band_id) for band_id in band_ids}
    missing = sorted(wanted - set(plan.band_ids_in_order()))
    if missing:
        raise ValueError(f"recursive forecast requested unknown band ids: {missing}")
    channels = [channel for channel in plan.channels if channel.band_id in wanted]
    layers_in_order: list[str] = []
    seen: set[str] = set()
    for channel in channels:
        if channel.layer in seen:
            continue
        seen.add(channel.layer)
        layers_in_order.append(channel.layer)
    return LayerPlan(
        rasters_hash=str(plan.rasters_hash),
        layers_in_order=layers_in_order,
        channels=channels,
        chip_groups={k: dict(v) for k, v in plan.chip_groups.items()},
        derived_dependencies={k: list(v) for k, v in plan.derived_dependencies.items()},
        step=plan.step,
    )


def _gee_source_layers(plan, *, registry: LayerRegistry = REGISTRY) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def visit(layer: str) -> None:
        spec = registry.get(layer)
        if spec.source == "derived":
            for parent in spec.derived_from or []:
                visit(str(parent))
            return
        if spec.source not in {"gee_chip", "gee_tile"} or layer in seen:
            return
        seen.add(layer)
        out.append(layer)

    for layer in plan.layers_in_order:
        visit(str(layer))
    return out


def _compute_valid_mask(values: torch.Tensor, nodata: Any) -> torch.Tensor:
    if nodata is None:
        return torch.isfinite(values)
    if isinstance(nodata, float) and np.isnan(nodata):
        return torch.isfinite(values)
    return torch.isfinite(values) & (values != float(nodata))


def _selected_splits_for_split(
    *,
    run_payload: dict[str, object],
    split_key: str,
) -> list[str] | None:
    manifest_path = (
        paths.study_dir(str(run_payload["study_id"]))
        / "features"
        / "points"
        / canonicalize_split_key(str(split_key))
        / str(run_payload["rasters_hash"])
        / str(run_payload["sampler_hash"])
        / "manifest.json"
    )
    if not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    extraction = payload.get("extraction", {})
    if not isinstance(extraction, dict):
        return None
    raw_selected_splits = extraction.get("selected_splits")
    if not isinstance(raw_selected_splits, list) or not raw_selected_splits:
        return None
    selected_splits: list[str] = []
    seen: set[str] = set()
    for raw_split in raw_selected_splits:
        split = canonicalize_split_key(raw_split)
        if not split or split in seen:
            continue
        selected_splits.append(split)
        seen.add(split)
    return selected_splits or None


def _resolve_seed_context(
    *,
    run_payload: dict[str, object],
    resolved_artifacts: dict[str, object],
    split_key: str,
    test_point_index: int,
) -> _SeedContext:
    split_key = str(split_key)
    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    step_time_s = int(timescale["step_time_s"])
    plan = timescale["plan"]
    output_hw = tuple(int(v) for v in timescale["output_hw"])

    study_dir = Path(
        resolved_artifacts.get("study_dir") or paths.study_dir(str(run_payload["study_id"]))
    )
    trajectories_path = Path(
        resolved_artifacts.get("trajectories_path") or study_dir / "trajectories.csv"
    )
    df = load_trajectories_csv(trajectories_path)
    train_df = df[df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Study has no train split for recursive forecast support")

    support = fit_interval_step_support(
        train_df,
        step_time_s=step_time_s,
        gap_q=float(plan.step_spec.gap_q),
        step_q=float(plan.step_spec.step_q),
        max_step_q=float(plan.step_spec.max_step_q),
        step_multiplier=float(plan.step_spec.step_multiplier),
    )
    selected_splits = _selected_splits_for_split(
        run_payload=run_payload,
        split_key=split_key,
    )
    if selected_splits:
        steps_source_df = df[df[SET_COL].isin(selected_splits)].copy()
    else:
        steps_source_df = df[df[SET_COL] == split_key].copy()
    steps = select_step_starts_with_support(
        steps_source_df,
        step_time_s=step_time_s,
        gap_q=float(plan.step_spec.gap_q),
        support=support,
    )
    steps = steps[steps[SET_COL] == split_key].reset_index(drop=True)

    index = int(test_point_index)
    if index < 0 or index >= int(len(steps)):
        raise ValueError(
            f"test_point_index={index} is invalid for split={split_key!r}; "
            f"available held-out steps={int(len(steps))}"
        )
    buffer_m = float(support.max_step_length_m)
    return _SeedContext(
        row=steps.iloc[index],
        step_time_s=step_time_s,
        buffer_m=buffer_m,
        pixel_size_m=_support_pixel_size_m(output_hw=output_hw, buffer_m=buffer_m),
    )


def _individual_id_from_row(row: Any) -> str:
    if hasattr(row, "individual_local_identifier"):
        return str(row.individual_local_identifier)
    try:
        return str(row["individual-local-identifier"])
    except Exception:
        return ""


def _step_union_bounds_latlon(
    *,
    lon: torch.Tensor,
    lat: torch.Tensor,
    buffer_m: float,
) -> tuple[float, float, float, float]:
    lon_np = lon.detach().cpu().numpy().astype(np.float64, copy=False)
    lat_np = lat.detach().cpu().numpy().astype(np.float64, copy=False)
    max_abs_lat = float(np.max(np.abs(lat_np))) if int(lat_np.size) else 0.0
    lon_deg_to_m = METERS_PER_DEG_LON_EQUATOR * math.cos(math.radians(max_abs_lat))
    buffer_deg_lon = (
        360.0 if abs(lon_deg_to_m) < 1e-6 else float(buffer_m) / float(lon_deg_to_m)
    )
    buffer_deg_lat = float(buffer_m) / METERS_PER_DEG_LAT
    return (
        max(WGS84_MIN_LON, float(np.min(lon_np)) - buffer_deg_lon),
        max(WGS84_MIN_LAT, float(np.min(lat_np)) - buffer_deg_lat),
        min(WGS84_MAX_LON, float(np.max(lon_np)) + buffer_deg_lon),
        min(WGS84_MAX_LAT, float(np.max(lat_np)) + buffer_deg_lat),
    )


def _sample_grid_rows_cols(
    scores_hw: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.as_tensor(scores_hw, dtype=torch.float32)
    if scores.ndim != 3:
        raise ValueError(f"scores_hw must have shape (N,H,W), got {tuple(scores.shape)}")
    n, height, width = (int(scores.shape[0]), int(scores.shape[1]), int(scores.shape[2]))
    flat = scores.reshape(n, height * width)
    probs = torch.softmax(flat, dim=1)
    if not bool(torch.isfinite(probs).all().item()):
        raise ValueError("recursive forecast scores produced non-finite probabilities")
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    return sampled // width, sampled % width


def _rows_cols_to_lon_lat(
    *,
    state: _ForecastState,
    sampled_row: torch.Tensor,
    sampled_col: torch.Tensor,
    output_hw: tuple[int, int],
    pixel_size_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from pyproj import Transformer

    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    from_3857 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    start_x, start_y = to_3857.transform(
        state.lon.detach().cpu().numpy(),
        state.lat.detach().cpu().numpy(),
    )
    x_offsets = (
        sampled_col.detach().cpu().numpy().astype(np.float64, copy=False)
        - (int(output_hw[1]) // 2)
    ) * float(pixel_size_m)
    y_offsets = (
        (int(output_hw[0]) // 2)
        - sampled_row.detach().cpu().numpy().astype(np.float64, copy=False)
    ) * float(pixel_size_m)
    next_lon, next_lat = from_3857.transform(
        np.asarray(start_x, dtype=np.float64) + x_offsets,
        np.asarray(start_y, dtype=np.float64) + y_offsets,
    )
    return (
        torch.as_tensor(next_lon, dtype=torch.float32),
        torch.as_tensor(next_lat, dtype=torch.float32),
    )


def _next_bearing_rad(
    *,
    start_lon: torch.Tensor,
    start_lat: torch.Tensor,
    end_lon: torch.Tensor,
    end_lat: torch.Tensor,
) -> torch.Tensor:
    values = bearing_rad(
        start_lat.detach().cpu().numpy().astype(np.float64, copy=False),
        start_lon.detach().cpu().numpy().astype(np.float64, copy=False),
        end_lat.detach().cpu().numpy().astype(np.float64, copy=False),
        end_lon.detach().cpu().numpy().astype(np.float64, copy=False),
    )
    return torch.nan_to_num(torch.as_tensor(values, dtype=torch.float32), nan=0.0)


def _load_forecast_scorer(
    *,
    run_payload: dict[str, object],
    resolved_artifacts: dict[str, object],
    run_dir: Path,
    seed_context: _SeedContext,
    split_key: str,
    num_trajectories: int,
) -> Any:
    method = str(run_payload["method"])
    if method in {"issf_linear", "issf_mlp"}:
        return _load_issf_forecast_scorer(
            run_payload=run_payload,
            resolved_artifacts=resolved_artifacts,
            run_dir=run_dir,
            seed_context=seed_context,
            model_kind="linear" if method == "issf_linear" else "mlp",
        )
    if method == "deepssf":
        return _load_deepssf_forecast_scorer(
            run_payload=run_payload,
            resolved_artifacts=resolved_artifacts,
            run_dir=run_dir,
            seed_context=seed_context,
        )
    if method == "moveformer":
        return _load_moveformer_forecast_scorer(
            run_payload=run_payload,
            resolved_artifacts=resolved_artifacts,
            run_dir=run_dir,
            seed_context=seed_context,
            split_key=split_key,
            num_trajectories=num_trajectories,
        )
    raise ValueError(f"Unsupported recursive forecast method: {method!r}")


def _checkpoint_state_dict(run_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint_path = Path(run_dir) / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint_best.pt not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return dict(checkpoint["state_dict"])


def _load_issf_forecast_scorer(
    *,
    run_payload: dict[str, object],
    resolved_artifacts: dict[str, object],
    run_dir: Path,
    seed_context: _SeedContext,
    model_kind: str,
) -> _ISSFForecastScorer:
    import movebench.bench.executors.ssf_grid as ssf_grid
    from movebench.datasets.ssf import iSSFDataset
    from movebench.models.ssf import LinearStepSelect, MLPStepSelect

    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    output_hw = tuple(int(v) for v in timescale["output_hw"])
    dataset = iSSFDataset(
        study_dir=resolved_artifacts["study_dir"],
        trajectories_path=resolved_artifacts["trajectories_path"],
        plan=resolved_artifacts["plan"],
        sampler_config=resolved_artifacts["sampler"],
        split="train",
        subset_band_ids=[str(band_id) for band_id in run_payload["band_ids"]],
    )
    if model_kind == "linear":
        model = LinearStepSelect(
            int(dataset.c_env),
            include_time=bool(dataset.include_time),
            n_individuals=int(dataset.n_individuals),
        )
    else:
        model_cfg = dict(run_payload.get("method_model", {}))
        model = MLPStepSelect(
            int(dataset.c_env),
            include_time=bool(dataset.include_time),
            n_individuals=int(dataset.n_individuals),
            hidden_dims=tuple(model_cfg.get("hidden_dims", [128])),
            dropout=float(model_cfg.get("dropout", 0.6)),
            interactions_before_mlp=bool(model_cfg.get("interactions_before_mlp", False)),
        )
    model.load_state_dict(_checkpoint_state_dict(run_dir))
    trajectories_df = load_trajectories_csv(resolved_artifacts["trajectories_path"])
    _support, train_steps = ssf_grid.fit_train_step_support(
        trajectories_df,
        step_spec=resolved_artifacts["plan"].step_spec,
    )
    train_manifest = resolved_artifacts["point_stores"]["train"]["manifest"]
    registry = _synthetic_tile_registry(dataset.plan)
    return _ISSFForecastScorer(
        model=model,
        feature_adapter=dataset,
        sampler_state=ssf_grid._replay_sampler_state(
            train_manifest=train_manifest,
            train_steps=train_steps,
        ),
        plan=dataset.plan,
        sampler_config=resolved_artifacts["sampler"],
        output_hw=output_hw,
        pixel_size_m=float(seed_context.pixel_size_m),
        device=torch.device("cpu"),
        registry=registry,
        normalize_env=True,
    )


def _load_deepssf_forecast_scorer(
    *,
    run_payload: dict[str, object],
    resolved_artifacts: dict[str, object],
    run_dir: Path,
    seed_context: _SeedContext,
) -> _DeepSSFForecastScorer:
    from movebench.models.deepssf import DeepSSF

    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    train_manifest = dict(resolved_artifacts["cache_manifests"]["train"]["manifest"])
    output_hw = tuple(int(v) for v in train_manifest.get("output_hw", timescale["output_hw"]))
    pixel_size_m = float(train_manifest.get("pixel_size_m", seed_context.pixel_size_m))
    if int(output_hw[0]) != int(output_hw[1]):
        raise ValueError(f"deepSSF recursive forecast requires square output_hw, got {output_hw}")
    plan = _subset_layer_plan(
        timescale["plan"],
        [str(band_id) for band_id in run_payload["band_ids"]],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSSF(
        int(plan.num_channels()),
        image_dim=int(output_hw[0]),
        pixel_size_m=float(pixel_size_m),
        device=device,
    )
    model.load_state_dict(_checkpoint_state_dict(run_dir))
    return _DeepSSFForecastScorer(
        model=model.to(device),
        plan=plan,
        sampler_config=timescale["sampler"],
        output_hw=output_hw,
        pixel_size_m=float(pixel_size_m),
        device=device,
        registry=_synthetic_tile_registry(plan),
        normalize_env=True,
    )


def _load_moveformer_forecast_scorer(
    *,
    run_payload: dict[str, object],
    resolved_artifacts: dict[str, object],
    run_dir: Path,
    seed_context: _SeedContext,
    split_key: str,
    num_trajectories: int,
) -> Any:
    from movebench.datasets.moveformer import build_moveformer_forecast_seed
    from movebench.models.moveformer import build_moveformer_selection_model

    timescale = bench_config.resolve_timescale(str(run_payload["timescale"]))
    output_hw = tuple(int(v) for v in timescale["output_hw"])
    band_ids = [str(band_id) for band_id in run_payload["band_ids"]]
    seed = build_moveformer_forecast_seed(
        study_dir=Path(resolved_artifacts["study_dir"]),
        plan=resolved_artifacts["plan"],
        sampler_config=resolved_artifacts["sampler"],
        split_key=str(split_key),
        point_id_start=int(seed_context.row.point_id_start),
        subset_band_ids=band_ids,
        normalize_env=bool(resolved_artifacts.get("normalization_stats_path") is not None),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_moveformer_selection_model(cov_dim=int(seed.cov_dim))
    model.load_state_dict(_checkpoint_state_dict(run_dir))
    model = model.to(device)
    plan = _subset_layer_plan(resolved_artifacts["plan"], band_ids)
    return _MoveFormerForecastScorer(
        model=model,
        seed=seed,
        histories=[list(seed.records) for _ in range(int(num_trajectories))],
        plan=plan,
        sampler_config=resolved_artifacts["sampler"],
        output_hw=output_hw,
        pixel_size_m=float(seed_context.pixel_size_m),
        device=device,
        registry=_synthetic_tile_registry(plan),
        split_key=str(split_key),
        normalize_env=bool(resolved_artifacts.get("normalization_stats_path") is not None),
    )


def _download_step_rasters(
    *,
    study_dir: Path,
    plan,
    registry,
    bounds_latlon: tuple[float, float, float, float],
    timestamp_ns: int,
    buffer_m: float,
    pixel_size_m: float,
) -> None:
    import movebench.layers.download as layer_download

    study_dir = Path(study_dir)
    min_pixels = int(DEFAULT_RASTER_PARAMS["scale_clamp"]["min_pixels"])
    max_pixels = int(DEFAULT_RASTER_PARAMS["scale_clamp"]["max_pixels"])
    timestamp = pd.Timestamp(int(timestamp_ns), unit="ns", tz="UTC")
    region = layer_download.ee.Geometry.Rectangle([float(x) for x in bounds_latlon])

    for layer in _gee_source_layers(plan, registry=registry):
        spec = registry.get(layer)
        tile_dir = study_dir / "rasters" / str(plan.rasters_hash) / "tiles" / str(layer)
        tile_dir.mkdir(parents=True, exist_ok=True)

        year = int(timestamp.year) if str(spec.temporal) == "annual" else None
        suffix = f"_{year}" if year is not None else ""
        tile_name = f"tile_000{suffix}.tif"
        tile_path = tile_dir / tile_name

        if spec.source == "gee_tile":
            img = layer_download._tile_image(layer, registry=registry, year=year)
            target_scale = float(spec.native_scale_m or pixel_size_m)
        elif spec.source == "gee_chip":
            img = layer_download._chip_layer_image(
                layer,
                spec=spec,
                timestamp=timestamp.isoformat(),
                region=region,
            )
            target_scale = _chip_target_scale_m(
                plan,
                layer=str(layer),
                registry=registry,
                fallback=float(spec.native_scale_m or pixel_size_m),
            )
        else:
            continue

        scale = layer_download.compute_group_scale(
            float(buffer_m),
            target_scale,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        layer_download.download_tiled_image(
            img.unmask(GEE_NODATA_SENTINEL).toFloat(),
            region,
            scale=float(scale),
            out_path=tile_path,
            num_bands=len(spec.bands),
            bounds_latlon=bounds_latlon,
        )
        _write_single_tile_index(
            tile_dir=tile_dir,
            layer=str(layer),
            tile_name=tile_name,
            bounds_latlon=bounds_latlon,
            scale_m=float(scale),
            num_bands=len(spec.bands),
            year=year,
        )


def _chip_target_scale_m(
    plan,
    *,
    layer: str,
    registry: LayerRegistry,
    fallback: float,
) -> float:
    for group in plan.chip_groups.values():
        layers = [str(x) for x in group.get("layers", [])]
        if str(layer) in layers:
            return float(group.get("target_scale_m", fallback))
    spec = registry.get(layer)
    return float(spec.native_scale_m or fallback)


def _write_single_tile_index(
    *,
    tile_dir: Path,
    layer: str,
    tile_name: str,
    bounds_latlon: tuple[float, float, float, float],
    scale_m: float,
    num_bands: int,
    year: int | None,
) -> None:
    min_lon, min_lat, max_lon, max_lat = (float(x) for x in bounds_latlon)
    payload: dict[str, Any] = {
        "schema_version": "1",
        "layer": str(layer),
        "scale_m": float(scale_m),
        "num_bands": int(num_bands),
        "n_tiles_reported": 1,
        "n_tiles_effective": 1,
        "tiles": [
            {
                "file": str(tile_name),
                "bounds": {
                    "min_lon": min_lon,
                    "min_lat": min_lat,
                    "max_lon": max_lon,
                    "max_lat": max_lat,
                },
            }
        ],
    }
    if year is not None:
        payload["year"] = int(year)
        (tile_dir / f"index_{int(year)}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    (tile_dir / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _extract_grid_batch(
    *,
    study_dir: Path,
    source_study_dir: Path,
    plan,
    sampler_config,
    registry,
    bounds_latlon: tuple[float, float, float, float],
    state: _ForecastState,
    output_hw: tuple[int, int],
    buffer_m: float,
    normalize_env: bool,
) -> _GridBatch:
    from pyproj import Transformer
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds

    from movebench.features.normalization import load_points_normalization_mean_std
    from movebench.layers.io import LayerIO
    from movebench.rasters import reproject_to_meters

    registry = registry or _synthetic_tile_registry(plan)
    io = LayerIO(study_dir=Path(study_dir), plan=plan, registry=registry)
    timestamp = pd.Timestamp(int(state.timestamp_ns[0].item()), unit="ns", tz="UTC")
    raster_cache = {}
    rasters = [
        io.load_layer_cached(
            str(layer),
            cache=raster_cache,
            bounds_latlon=bounds_latlon,
            timestamp=timestamp,
        )
        for layer in plan.layers_in_order
    ]
    categorical_layers = {
        str(layer)
        for layer in plan.layers_in_order
        if bool(registry.get(str(layer)).categorical)
    }

    mean = std = None
    if normalize_env:
        mean, std = load_points_normalization_mean_std(
            study_dir=Path(source_study_dir),
            plan=plan,
            sampler_config=sampler_config,
        )

    to_target = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    env_rows = []
    valid_rows = []
    height, width = (int(output_hw[0]), int(output_hw[1]))
    px = (2.0 * float(buffer_m)) / float(width - 1)

    for lon_i, lat_i in zip(state.lon.tolist(), state.lat.tolist()):
        x0, y0 = to_target.transform(float(lon_i), float(lat_i))
        fixed_dst_transform = from_bounds(
            float(x0) - float(buffer_m) - 0.5 * px,
            float(y0) - float(buffer_m) - 0.5 * px,
            float(x0) + float(buffer_m) + 0.5 * px,
            float(y0) + float(buffer_m) + 0.5 * px,
            width,
            height,
        )
        aligned_arrays, final_transform, _final_crs = reproject_to_meters(
            rasters,
            dst_crs="EPSG:3857",
            resampling=Resampling.bilinear,
            dst_height=height,
            dst_width=width,
            dst_transform=fixed_dst_transform,
        )
        if categorical_layers:
            for idx, layer in enumerate(plan.layers_in_order):
                if str(layer) not in categorical_layers:
                    continue
                nearest_arrays, _, _ = reproject_to_meters(
                    [rasters[idx]],
                    dst_crs="EPSG:3857",
                    resampling=Resampling.nearest,
                    dst_height=height,
                    dst_width=width,
                    dst_transform=final_transform,
                )
                aligned_arrays[idx] = nearest_arrays[0]

        env = torch.zeros((int(plan.num_channels()), height, width), dtype=torch.float32)
        valid = torch.zeros_like(env, dtype=torch.bool)
        for idx, layer in enumerate(plan.layers_in_order):
            layer = str(layer)
            start, end = plan.layer_slice(layer)
            spec = registry.get(layer)
            vals = torch.from_numpy(
                aligned_arrays[idx].astype(np.float32, copy=False)
            )
            vmask = _compute_valid_mask(vals, rasters[idx].nodata)
            if spec.categorical:
                class_vals = torch.clamp(vals[0].to(torch.int64), 0, int(spec.num_classes) - 1)
                one_hot = (
                    F.one_hot(class_vals, int(spec.num_classes))
                    .permute(2, 0, 1)
                    .to(torch.float32)
                )
                vexp = vmask[0].unsqueeze(0).expand(int(spec.num_classes), -1, -1)
                env[start:end] = one_hot * vexp.to(one_hot.dtype)
                valid[start:end] = vexp
                continue
            if spec.valid_range is not None:
                lo, hi = float(spec.valid_range[0]), float(spec.valid_range[1])
                vmask = vmask & (vals >= lo) & (vals <= hi)
            env[start:end] = torch.where(vmask, vals, torch.zeros_like(vals))
            valid[start:end] = vmask
        if normalize_env:
            assert mean is not None
            assert std is not None
            env = torch.where(valid, (env - mean.view(-1, 1, 1)) / std.view(-1, 1, 1), torch.zeros_like(env))
        env_rows.append(env)
        valid_rows.append(valid)

    return _GridBatch(
        env=torch.stack(env_rows, dim=0),
        valid=torch.stack(valid_rows, dim=0),
    )


def run_recursive_forecast(
    config: RecursiveForecastConfig,
) -> RecursiveForecastResult:
    if int(config.num_trajectories) <= 0:
        raise ValueError(
            f"num_trajectories must be > 0, got {config.num_trajectories}"
        )
    if int(config.num_steps) < 0:
        raise ValueError(f"num_steps must be >= 0, got {config.num_steps}")
    if int(config.burn_in_steps) < 0:
        raise ValueError(
            f"burn_in_steps must be >= 0, got {config.burn_in_steps}"
        )

    run_dir = Path(config.run_dir)
    run_payload, resolved_artifacts = load_saved_run_bundle(run_dir)
    method = str(run_payload["method"])
    if method not in _SUPPORTED_METHODS:
        raise ValueError(f"Unsupported recursive forecast method: {method!r}")

    seed = _resolve_seed_context(
        run_payload=run_payload,
        resolved_artifacts=resolved_artifacts,
        split_key=str(config.split_key),
        test_point_index=int(config.test_point_index),
    )
    row = seed.row
    n = int(config.num_trajectories)
    burn_in_steps = int(config.burn_in_steps)
    steps = int(config.num_steps)
    total_steps = burn_in_steps + steps
    lon = torch.empty((n, total_steps + 1), dtype=torch.float32)
    lat = torch.empty_like(lon)
    timestamp_ns = torch.empty((n, total_steps + 1), dtype=torch.int64)
    sampled_row = torch.empty((n, total_steps), dtype=torch.int64)
    sampled_col = torch.empty((n, total_steps), dtype=torch.int64)

    lon[:, 0] = float(row.start_lon)
    lat[:, 0] = float(row.start_lat)
    timestamp_ns[:, 0] = _timestamp_ns(row.timestamp)
    result_pixel_size_m = float(seed.pixel_size_m)

    if total_steps:
        scorer = _load_forecast_scorer(
            run_payload=run_payload,
            resolved_artifacts=resolved_artifacts,
            run_dir=run_dir,
            seed_context=seed,
            split_key=str(config.split_key),
            num_trajectories=n,
        )
        output_hw = tuple(int(v) for v in scorer.output_hw)
        result_pixel_size_m = float(getattr(scorer, "pixel_size_m", seed.pixel_size_m))
        rollout_buffer_m = (result_pixel_size_m * float(output_hw[1] - 1)) / 2.0
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed))
        state = _ForecastState(
            lon=lon[:, 0].clone(),
            lat=lat[:, 0].clone(),
            timestamp_ns=timestamp_ns[:, 0].clone(),
            start_bearing_rad=torch.full(
                (n,),
                float(getattr(row, "start_bearing", 0.0)),
                dtype=torch.float32,
            ),
            individual_id=[_individual_id_from_row(row)] * n,
        )
        source_study_dir = Path(
            resolved_artifacts.get("study_dir")
            or paths.study_dir(str(run_payload["study_id"]))
        )
        sampler_config = getattr(scorer, "sampler_config", None)
        if sampler_config is None:
            sampler_config = bench_config.resolve_timescale(str(run_payload["timescale"]))[
                "sampler"
            ]
        registry = getattr(scorer, "registry", REGISTRY)
        with tempfile.TemporaryDirectory(prefix="movebench_forecast_") as tmpdir:
            temp_root = Path(tmpdir)
            for step_index in tqdm(range(total_steps)):
                bounds_latlon = _step_union_bounds_latlon(
                    lon=state.lon,
                    lat=state.lat,
                    buffer_m=float(rollout_buffer_m) + float(result_pixel_size_m),
                )
                step_study_dir = temp_root / f"step_{step_index:04d}"
                _download_step_rasters(
                    study_dir=step_study_dir,
                    plan=scorer.plan,
                    registry=REGISTRY,
                    bounds_latlon=bounds_latlon,
                    timestamp_ns=int(state.timestamp_ns[0].item()),
                    buffer_m=float(rollout_buffer_m),
                    pixel_size_m=float(result_pixel_size_m),
                )
                grid_batch = _extract_grid_batch(
                    study_dir=step_study_dir,
                    source_study_dir=source_study_dir,
                    plan=scorer.plan,
                    sampler_config=sampler_config,
                    registry=registry,
                    bounds_latlon=bounds_latlon,
                    state=state,
                    output_hw=output_hw,
                    buffer_m=float(rollout_buffer_m),
                    normalize_env=bool(scorer.normalize_env),
                )
                scores_hw = scorer.score_grid(
                    grid_batch=grid_batch,
                    state=state,
                    step_index=step_index,
                )
                rows, cols = _sample_grid_rows_cols(scores_hw, generator=generator)
                next_lon, next_lat = _rows_cols_to_lon_lat(
                    state=state,
                    sampled_row=rows,
                    sampled_col=cols,
                    output_hw=output_hw,
                    pixel_size_m=float(result_pixel_size_m),
                )
                if step_index < burn_in_steps:
                    next_timestamp_ns = state.timestamp_ns.clone()
                else:
                    next_timestamp_ns = state.timestamp_ns + (
                        int(seed.step_time_s) * 1_000_000_000
                    )

                sampled_row[:, step_index] = rows.to(dtype=torch.int64)
                sampled_col[:, step_index] = cols.to(dtype=torch.int64)
                lon[:, step_index + 1] = next_lon.to(dtype=torch.float32)
                lat[:, step_index + 1] = next_lat.to(dtype=torch.float32)
                timestamp_ns[:, step_index + 1] = next_timestamp_ns.to(dtype=torch.int64)

                scorer.observe_sampled(
                    grid_batch=grid_batch,
                    state=state,
                    sampled_row=rows,
                    sampled_col=cols,
                    next_lon=next_lon,
                    next_lat=next_lat,
                    next_timestamp_ns=next_timestamp_ns,
                    step_index=step_index,
                )
                state = _ForecastState(
                    lon=next_lon.to(dtype=torch.float32),
                    lat=next_lat.to(dtype=torch.float32),
                    timestamp_ns=next_timestamp_ns.to(dtype=torch.int64),
                    start_bearing_rad=_next_bearing_rad(
                        start_lon=state.lon,
                        start_lat=state.lat,
                        end_lon=next_lon,
                        end_lat=next_lat,
                    ),
                    individual_id=state.individual_id,
                )

    run_key = str(run_payload.get("run_key") or run_dir.name)
    return RecursiveForecastResult(
        run_key=run_key,
        method=method,
        split_key=str(config.split_key),
        seed_point_id_start=int(row.point_id_start),
        step_time_s=int(seed.step_time_s),
        pixel_size_m=float(result_pixel_size_m),
        lon=lon,
        lat=lat,
        timestamp_ns=timestamp_ns,
        sampled_row=sampled_row,
        sampled_col=sampled_col,
    )
