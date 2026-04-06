from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import movebench.bench.evaluation.runner as evaluation_runner
from pyproj import Transformer
from torch.utils.data import DataLoader, Subset

from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.ssf import iSSFDataset
from movebench.features.points import _build_step_requests, _extract_env_block
from movebench.models.ssf import LinearStepSelect, MLPStepSelect
from movebench.trajectories import (
    SET_COL,
    canonicalize_split_key,
    fit_interval_step_support,
    load_trajectories_csv,
    select_step_starts_with_support,
)

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _log_progress(message: str) -> None:
    print(str(message), flush=True)


def _loss_step_select(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B,P), got {tuple(logits.shape)}")
    target = torch.full(
        (int(logits.shape[0]),),
        int(logits.shape[1] - 1),
        dtype=torch.long,
        device=logits.device,
    )
    return F.cross_entropy(logits, target)


def _base_dataset(ds):
    return ds.dataset if isinstance(ds, Subset) else ds


def _make_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    base = _base_dataset(dataset)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=base.collate,
        drop_last=False,
        pin_memory=device.type == "cuda",
        persistent_workers=bool(num_workers > 0),
    )


def _subset_loader(
    dataset: iSSFDataset,
    *,
    batch_size: int,
    seed: int,
    max_items: int | None,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    if max_items is None:
        return _make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        )
    k = min(int(max_items), int(len(dataset)))
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    subset = Subset(dataset, torch.randperm(len(dataset), generator=g)[:k].tolist())
    return _make_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
    )


def _move_batch_to_device(
    batch: dict[str, torch.Tensor | None], device: torch.device
) -> dict[str, torch.Tensor | None]:
    out: dict[str, torch.Tensor | None] = {}
    for key, value in batch.items():
        if value is None:
            out[key] = None
        else:
            out[key] = value.to(device, non_blocking=device.type == "cuda")
    return out


@torch.no_grad()
def _eval_step_select_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch)
        loss = _loss_step_select(logits)
        total_nll += float(loss.item()) * int(logits.shape[0])
        total += int(logits.shape[0])
    if total == 0:
        raise ValueError("Loader yielded zero items")
    return {"nll": float(total_nll / float(total))}


class _ISSFRuntime:
    early_stop_metric = "nll"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_loader: DataLoader,
        eval_train_loader: DataLoader,
        eval_loader: DataLoader,
        early_stop_split_key: str,
        device: torch.device,
        lr: float,
        weight_decay: float,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_train_loader = eval_train_loader
        self.eval_loader = eval_loader
        self.early_stop_split_key = str(early_stop_split_key)
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self._train_iter = iter(self.train_loader)

    def _next_batch(self) -> dict[str, torch.Tensor | None]:
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        return _move_batch_to_device(batch, self.device)

    def train_step(self, step: int) -> dict[str, float]:
        del step
        batch = self._next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(batch)
        loss = _loss_step_select(logits)
        loss.backward()
        self.optimizer.step()
        return {"loss/train_step": float(loss.item())}

    def evaluate(self, split_key: str) -> dict[str, float]:
        if split_key == "train":
            return _eval_step_select_loader(
                self.model, self.eval_train_loader, self.device
            )
        if split_key != self.early_stop_split_key:
            raise KeyError(f"Unknown split for iSSF runtime: {split_key!r}")
        return _eval_step_select_loader(self.model, self.eval_loader, self.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "model": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(dict(state["model"]))
        self.optimizer.load_state_dict(dict(state["optimizer"]))


def _selected_splits_from_manifest(manifest) -> list[str] | None:
    raw_selected_splits = manifest.extraction.get("selected_splits")
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


def _fit_prediction_support(
    trajectories_df,
    *,
    step_spec,
) -> tuple[object, object]:
    train_df = trajectories_df[trajectories_df[SET_COL] == "train"].copy()
    if len(train_df) == 0:
        raise ValueError("Study has no train split for iSSF prediction support")
    support = fit_interval_step_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        step_q=float(step_spec.step_q),
        max_step_q=float(step_spec.max_step_q),
        step_multiplier=float(step_spec.step_multiplier),
    )
    train_steps = select_step_starts_with_support(
        train_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        support=support,
    )
    return support, train_steps


def _prediction_steps_for_split(
    trajectories_df,
    *,
    split_key: str,
    selected_splits: list[str] | None,
    step_spec,
    support,
):
    if selected_splits:
        requested_df = trajectories_df[trajectories_df[SET_COL].isin(selected_splits)].copy()
    else:
        requested_df = trajectories_df[trajectories_df[SET_COL] == str(split_key)].copy()
    steps = select_step_starts_with_support(
        requested_df,
        step_time_s=int(step_spec.step_time_s),
        gap_q=float(step_spec.gap_q),
        support=support,
    )
    return steps[steps[SET_COL] == str(split_key)].reset_index(drop=True).copy()


def _replay_sampler_state(
    *,
    train_manifest,
    train_steps,
    num_candidates: int,
) -> dict[str, object]:
    valid_lengths = train_steps["step_length_m"].to_numpy(dtype=np.float64)
    valid_lengths = valid_lengths[np.isfinite(valid_lengths) & (valid_lengths > 0.0)]
    valid_turns = train_steps["turn_angle_rad"].to_numpy(dtype=np.float64)
    valid_turns = valid_turns[np.isfinite(valid_turns)]
    if len(valid_lengths) == 0:
        raise ValueError("No valid train step lengths available for iSSF prediction sampling")
    if len(valid_turns) == 0:
        raise ValueError("No valid train turn angles available for iSSF prediction sampling")

    sampler = dict(train_manifest.sampler)
    gamma = dict(sampler.get("gamma", {})) if isinstance(sampler.get("gamma"), dict) else {}
    gamma_params = gamma.get("params")
    if not (bool(gamma.get("used", False)) and isinstance(gamma_params, list) and gamma_params):
        gamma_params = None
    else:
        gamma_params = tuple(float(x) for x in gamma_params)

    vonmises = dict(sampler.get("vonmises", {})) if isinstance(sampler.get("vonmises"), dict) else {}
    vonmises_params = vonmises.get("params")
    if not (
        bool(vonmises.get("used", False))
        and isinstance(vonmises_params, list)
        and vonmises_params
    ):
        vonmises_params = None
    else:
        vonmises_params = tuple(float(x) for x in vonmises_params)

    return {
        "valid_lengths": valid_lengths,
        "valid_turns": valid_turns,
        "num_candidates": int(num_candidates),
        "random_seed": int(sampler["random_seed"]),
        "gamma_used": gamma_params is not None,
        "gamma_params": gamma_params,
        "vonmises_used": vonmises_params is not None,
        "vonmises_params": vonmises_params,
    }


def _prediction_batch(
    *,
    dataset: iSSFDataset,
    env: torch.Tensor,
    valid: torch.Tensor,
    shared: dict[str, object],
) -> dict[str, torch.Tensor | None]:
    env = env[:, :, :-1].to(dtype=torch.float32)
    valid = valid[:, :, :-1].to(dtype=torch.bool)
    if dataset.normalize:
        assert dataset._mean is not None
        assert dataset._std is not None
        env = (env - dataset._mean.view(1, -1, 1)) / dataset._std.view(1, -1, 1)
    env = torch.where(valid, env, torch.zeros_like(env))

    batch: dict[str, torch.Tensor | None] = {
        "env": env,
        "valid": valid,
        "move": dataset._movement_tensor(
            step_length_m=shared["step_length_m"][:, :-1],
            turn_angle_rad=shared["turn_angle_rad"][:, :-1],
        ),
        "time": None,
        "individual_idx": None,
    }
    if dataset.include_time:
        batch["time"] = dataset._time_tensor(shared["timestamp_ns"])
    if dataset.include_individual_id:
        batch["individual_idx"] = dataset._individual_idx_tensor(
            np.asarray(shared["individual_id"])
        )
    return batch


def _coords_from_shared(
    shared: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    point_ids = shared["point_id_start"].to(dtype=torch.int64).cpu()
    num_steps = int(point_ids.shape[0])
    candidate_lon = shared["candidate_lon"][:, :-1].to(dtype=torch.float64)
    candidate_lat = shared["candidate_lat"][:, :-1].to(dtype=torch.float64)
    num_candidates = int(candidate_lon.shape[1])
    if num_steps == 0:
        return point_ids, torch.zeros((0, num_candidates, 2), dtype=torch.float32)

    start_lon = shared["start_lon"].to(dtype=torch.float64)
    start_lat = shared["start_lat"].to(dtype=torch.float64)
    start_x_m, start_y_m = _TO_3857.transform(start_lon.tolist(), start_lat.tolist())
    cand_x_m, cand_y_m = _TO_3857.transform(
        candidate_lon.reshape(-1).tolist(),
        candidate_lat.reshape(-1).tolist(),
    )
    coords_m = torch.tensor(
        list(zip(cand_x_m, cand_y_m)),
        dtype=torch.float32,
    ).view(num_steps, num_candidates, 2)
    start_xy_m = torch.tensor(
        list(zip(start_x_m, start_y_m)),
        dtype=torch.float32,
    )
    coords_m[:, :, 0] -= start_xy_m[:, 0].view(-1, 1)
    coords_m[:, :, 1] -= start_xy_m[:, 1].view(-1, 1)
    return point_ids, coords_m


def _parametric_prediction_payload(
    *,
    model: torch.nn.Module,
    dataset: iSSFDataset,
    trajectories_df,
    train_manifest,
    train_steps,
    support,
    split_key: str,
    selected_splits: list[str] | None,
    study_id: str,
    study_dir: Path,
    num_candidates: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, torch.Tensor | str]:
    steps = _prediction_steps_for_split(
        trajectories_df,
        split_key=str(split_key),
        selected_splits=selected_splits,
        step_spec=dataset.plan.step_spec,
        support=support,
    )
    m = int(num_candidates)
    if len(steps) == 0:
        return {
            "prediction_kind": "discrete",
            "point_id_start": torch.zeros((0,), dtype=torch.int64),
            "coords_m": torch.zeros((0, m, 2), dtype=torch.float32),
            "probs": torch.zeros((0, m), dtype=torch.float32),
        }

    sampler_state = _replay_sampler_state(
        train_manifest=train_manifest,
        train_steps=train_steps,
        num_candidates=m,
    )
    reqs = _build_step_requests(steps)
    train_points_per_step = int(train_manifest.sampler["num_candidates"]) + 1
    prediction_points_per_step = int(m) + 1
    scaled_batch = int(
        int(batch_size) * train_points_per_step / prediction_points_per_step
    )
    step_batch_size = max(1, min(int(batch_size), scaled_batch))
    point_id_chunks: list[torch.Tensor] = []
    coords_chunks: list[torch.Tensor] = []
    probs_chunks: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, len(reqs), step_batch_size),
            desc=f"{study_id}:{split_key} prediction",
            leave=False,
        ):
            req_batch = reqs[start : start + step_batch_size]
            env, valid, shared = _extract_env_block(
                study_id=str(study_id),
                study_dir=Path(study_dir),
                plan=dataset.plan,
                reqs=req_batch,
                sampler_state=sampler_state,
                n_workers=int(num_workers),
            )
            batch = _prediction_batch(
                dataset=dataset,
                env=env,
                valid=valid,
                shared=shared,
            )
            logits = model(_move_batch_to_device(batch, device))
            point_ids, coords_m = _coords_from_shared(shared)
            probs = F.softmax(logits, dim=1).detach().cpu()
            if int(coords_m.shape[1]) != int(probs.shape[1]):
                raise ValueError(
                    "iSSF discrete prediction payload support mismatch: "
                    f"coords_m has {int(coords_m.shape[1])} candidates but probs has {int(probs.shape[1])}"
                )
            probs_chunks.append(probs)
            point_id_chunks.append(point_ids)
            coords_chunks.append(coords_m)

    return {
        "prediction_kind": "discrete",
        "point_id_start": torch.cat(point_id_chunks, dim=0),
        "coords_m": torch.cat(coords_chunks, dim=0),
        "probs": torch.cat(probs_chunks, dim=0),
    }


def _run(
    *,
    run_payload,
    resolved_artifacts,
    run_dir,
    tracker,
    model_kind: str,
) -> dict[str, object]:

    # TODO: specify in config?
    # issf currently not benefitting from GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    print("Using device", device)
    point_stores = resolved_artifacts["point_stores"]
    run_dir = Path(run_dir)
    run_key = str(run_payload.get("run_key", run_payload["method"]))

    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams.get("batch_size", 256))
    num_workers = int(method_hparams.get("num_workers", 0) or 0)
    energy_score_num_candidates = int(
        method_hparams.get("energy_score_num_candidates", 5000)
    )
    eval_max_items = (
        None
        if method_hparams.get("max_eval_items") is None
        else int(method_hparams["max_eval_items"])
    )
    if energy_score_num_candidates <= 0:
        raise ValueError(
            "energy_score_num_candidates must be > 0; "
            f"got {energy_score_num_candidates}"
        )

    split_datasets: dict[str, iSSFDataset] = {}

    def _dataset_for(split_key: str) -> iSSFDataset:
        if split_key not in split_datasets:
            split_datasets[split_key] = iSSFDataset(
                study_dir=resolved_artifacts["study_dir"],
                trajectories_path=resolved_artifacts["trajectories_path"],
                plan=resolved_artifacts["plan"],
                sampler_config=resolved_artifacts["sampler"],
                split=str(split_key),
                subset_band_ids=list(run_payload["band_ids"]),
            )
        return split_datasets[split_key]

    print("Creating train dataset")
    train_dataset = _dataset_for("train")
    eval_split_key = str(run_payload["early_stop_split_key"])
    print("Creating eval dataset")
    eval_dataset = _dataset_for(eval_split_key)

    print("Creating train loader")
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )
    print("Creating eval loaders")
    eval_train_loader = _subset_loader(
        train_dataset,
        batch_size=batch_size,
        seed=int(run_payload["seed"]) + 0,
        max_items=eval_max_items,
        num_workers=num_workers,
        device=device,
    )
    eval_loader = _subset_loader(
        eval_dataset,
        batch_size=batch_size,
        seed=int(run_payload["seed"]) + 1,
        max_items=eval_max_items,
        num_workers=num_workers,
        device=device,
    )

    print("Creating model")
    c_env = int(train_dataset.c_env)
    if model_kind == "linear":
        model = LinearStepSelect(
            c_env,
            include_time=bool(train_dataset.include_time),
            n_individuals=int(train_dataset.n_individuals),
        )
    elif model_kind == "mlp":
        model_cfg = dict(run_payload.get("method_model", {}))
        model = MLPStepSelect(
            c_env,
            include_time=bool(train_dataset.include_time),
            n_individuals=int(train_dataset.n_individuals),
            hidden_dims=tuple(model_cfg.get("hidden_dims", [128])),
            dropout=float(model_cfg.get("dropout", 0.6)),
            interactions_before_mlp=bool(
                model_cfg.get("interactions_before_mlp", False)
            ),
        )
    else:
        raise ValueError(f"Unsupported iSSF model kind: {model_kind!r}")

    print("Creating runtime")
    runtime = _ISSFRuntime(
        model=model,
        train_loader=train_loader,
        eval_train_loader=eval_train_loader,
        eval_loader=eval_loader,
        early_stop_split_key=eval_split_key,
        device=device,
        lr=float(method_hparams["lr"]),
        weight_decay=float(method_hparams["weight_decay"]),
    )
    checkpoint_path = run_dir / "checkpoint_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        runtime.model.load_state_dict(dict(checkpoint["state_dict"]))
        best_step = int(dict(checkpoint.get("meta", {})).get("best_step", 0))
        train_steps_completed = best_step
        history_path = run_dir / "history.json"
        if history_path.exists():
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))
            history_steps = [
                int(float(entry["step"]))
                for entry in history_payload
                if isinstance(entry, dict) and "step" in entry
            ]
            if history_steps:
                train_steps_completed = max(history_steps)
        fit = {
            "completed_steps": int(train_steps_completed),
            "best_step": int(best_step),
            "train_metrics": {},
            "eval_metrics": {},
        }
        _log_progress(
            f"{run_key}: reusing iSSF checkpoint best_step={fit['best_step']}"
        )
    else:
        print("entering training loop")
        fit = run_training_loop(
            runtime,
            TrainerConfig(
                max_steps=int(method_hparams["max_steps"]),
                eval_every_steps=int(method_hparams["eval_every_steps"]),
                patience_evals=int(method_hparams["patience_evals"]),
                seed=int(run_payload["seed"]),
            ),
            tracker,
        )
        _log_progress(
            f"{run_key}: finished iSSF training steps={fit['completed_steps']} best_step={fit['best_step']}"
        )
        torch.save(
            {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in runtime.model.state_dict().items()
                },
                "meta": {
                    "method": str(run_payload["method"]),
                    "best_step": int(fit["best_step"]),
                    "covariate_set": str(run_payload["covariate_set"]),
                },
            },
            checkpoint_path,
        )

    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    trajectories_df = None
    support = None
    train_steps = None
    train_manifest = point_stores["train"]["manifest"]
    for split_key in run_payload["evaluation_split_keys"]:
        pred_path = pred_dir / f"{split_key}.pt"
        if pred_path.exists():
            try:
                evaluation_runner._score_split(
                    split_key=str(split_key),
                    pred_artifact=dict(torch.load(pred_path, map_location="cpu")),
                    run_payload=run_payload,
                )
                _log_progress(f"{run_key}: reusing iSSF predictions split={split_key}")
                continue
            except Exception as exc:
                _log_progress(
                    f"{run_key}: regenerating iSSF predictions split={split_key}: {exc}"
                )
        if trajectories_df is None:
            trajectories_df = load_trajectories_csv(
                Path(resolved_artifacts["trajectories_path"])
            )
            support, train_steps = _fit_prediction_support(
                trajectories_df,
                step_spec=resolved_artifacts["plan"].step_spec,
            )
        dataset = _dataset_for(str(split_key))
        split_manifest = point_stores[split_key]["manifest"]
        _log_progress(
            f"{run_key}: starting iSSF prediction split={split_key} "
            f"candidates={energy_score_num_candidates} workers={num_workers}"
        )
        torch.save(
            _parametric_prediction_payload(
                model=runtime.model,
                dataset=dataset,
                trajectories_df=trajectories_df,
                train_manifest=train_manifest,
                train_steps=train_steps,
                support=support,
                split_key=str(split_key),
                selected_splits=_selected_splits_from_manifest(split_manifest),
                study_id=str(run_payload["study_id"]),
                study_dir=Path(resolved_artifacts["study_dir"]),
                num_candidates=energy_score_num_candidates,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
            ),
            pred_path,
        )
        _log_progress(f"{run_key}: wrote iSSF predictions split={split_key}")

    return {
        "status": "completed",
        "train_steps": int(fit["completed_steps"]),
        "best_step": int(fit["best_step"]),
        "early_stop_split_key": eval_split_key,
        "train_metrics": dict(fit["train_metrics"]),
        "eval_metrics": dict(fit["eval_metrics"]),
    }


def run_linear(*, run_payload, resolved_artifacts, run_dir, tracker):
    return _run(
        run_payload=run_payload,
        resolved_artifacts=resolved_artifacts,
        run_dir=run_dir,
        tracker=tracker,
        model_kind="linear",
    )


def run_mlp(*, run_payload, resolved_artifacts, run_dir, tracker):
    return _run(
        run_payload=run_payload,
        resolved_artifacts=resolved_artifacts,
        run_dir=run_dir,
        tracker=tracker,
        model_kind="mlp",
    )
