from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import movebench.bench.config as bench_config
import movebench.bench.evaluation.runner as evaluation_runner
import movebench.bench.executors.ssf_grid as ssf_grid
from torch.utils.data import DataLoader, Subset

from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.ssf import iSSFDataset
from movebench.models.ssf import LinearStepSelect, MLPStepSelect
from movebench.trajectories import load_trajectories_csv

torch.multiprocessing.set_sharing_strategy("file_system")


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
    return ssf_grid.selected_splits_from_manifest(manifest)


def _fit_prediction_support(
    trajectories_df,
    *,
    step_spec,
) -> tuple[object, object]:
    return ssf_grid.fit_train_step_support(trajectories_df, step_spec=step_spec)


def _grid_prediction_payload(
    *,
    model: torch.nn.Module,
    dataset: iSSFDataset,
    train_manifest,
    train_steps,
    split_key: str,
    selected_splits: list[str] | None,
    study_dir: Path,
    trajectories_path: Path,
    sampler_config,
    output_hw: tuple[int, int],
    batch_size: int,
    prediction_num_workers: int,
    device: torch.device,
) -> dict[str, object]:
    grid_dataset = ssf_grid.make_grid_step_dataset(
        study_dir=study_dir,
        trajectories_path=trajectories_path,
        split_key=str(split_key),
        plan=dataset.plan,
        sampler_config=sampler_config,
        output_hw=output_hw,
        selected_splits=selected_splits,
        include_env=True,
    )
    return ssf_grid.build_ssf_grid_artifact(
        grid_dataset=grid_dataset,
        train_manifest=train_manifest,
        train_steps=train_steps,
        batch_size=batch_size,
        num_workers=prediction_num_workers,
        device=device,
        model=model,
        feature_adapter=dataset,
    )


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
    train_num_workers = int(method_hparams.get("num_workers", 0) or 0)
    prediction_num_workers = int(method_hparams.get("prediction_num_workers", 4) or 0)
    eval_max_items = (
        None
        if method_hparams.get("max_eval_items") is None
        else int(method_hparams["max_eval_items"])
    )
    output_hw = tuple(
        int(v)
        for v in bench_config.resolve_timescale(
            str(run_payload["timescale"]),
            study_id=str(run_payload["study_id"]),
        )[
            "output_hw"
        ]
    )
    model_cfg = dict(run_payload.get("method_model", {}))
    include_time = bool(model_cfg.get("include_time", False))

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
                include_time=include_time,
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
        num_workers=train_num_workers,
        device=device,
    )
    print("Creating eval loaders")
    eval_train_loader = _subset_loader(
        train_dataset,
        batch_size=batch_size,
        seed=int(run_payload["seed"]) + 0,
        max_items=eval_max_items,
        num_workers=train_num_workers,
        device=device,
    )
    eval_loader = _subset_loader(
        eval_dataset,
        batch_size=batch_size,
        seed=int(run_payload["seed"]) + 1,
        max_items=eval_max_items,
        num_workers=train_num_workers,
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
        print("Entering training loop")
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
    train_steps = None
    train_manifest = point_stores["train"]["manifest"]
    for split_key in run_payload["evaluation_split_keys"]:
        pred_path = pred_dir / f"{split_key}.pt"
        if pred_path.exists():
            try:
                print("Preds exist. Scoring.")
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
        print("Preds don't exist. Computing.")
        if trajectories_df is None:
            trajectories_df = load_trajectories_csv(
                Path(resolved_artifacts["trajectories_path"])
            )
            _support, train_steps = _fit_prediction_support(
                trajectories_df,
                step_spec=resolved_artifacts["plan"].step_spec,
            )
        dataset = _dataset_for(str(split_key))
        split_manifest = point_stores[split_key]["manifest"]
        _log_progress(
            f"{run_key}: starting iSSF prediction split={split_key} "
            f"grid={output_hw[0]}x{output_hw[1]} workers={prediction_num_workers}"
        )
        torch.save(
            _grid_prediction_payload(
                model=runtime.model,
                dataset=dataset,
                train_manifest=train_manifest,
                train_steps=train_steps,
                split_key=str(split_key),
                selected_splits=_selected_splits_from_manifest(split_manifest),
                study_dir=Path(resolved_artifacts["study_dir"]),
                trajectories_path=Path(resolved_artifacts["trajectories_path"]),
                sampler_config=resolved_artifacts["sampler"],
                output_hw=output_hw,
                batch_size=batch_size,
                prediction_num_workers=prediction_num_workers,
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
