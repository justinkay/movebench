from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pyproj import Transformer
from torch.utils.data import DataLoader, Subset

from movebench.bench.train import TrainerConfig, run_training_loop
from movebench.datasets.ssf import iSSFDataset
from movebench.features.points import PointStoreReader
from movebench.models.ssf import LinearStepSelect, MLPStepSelect
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    TIME_COL,
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


def _start_xy_by_point_id(trajectories_path: Path) -> dict[int, tuple[float, float]]:
    coords: dict[int, tuple[float, float]] = {}
    rows: list[dict[str, object]] = []
    with open(trajectories_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = str(row.get(TIME_COL, "")).strip()
            if not timestamp:
                continue
            rows.append(
                {
                    "point_id": int(row[POINT_ID_COL]),
                    "timestamp": timestamp,
                    "lon": float(row[LON_COL]),
                    "lat": float(row[LAT_COL]),
                    "individual": str(row[INDIVIDUAL_COL]),
                }
            )

    rows.sort(key=lambda row: (str(row["individual"]), str(row["timestamp"])))
    for row in rows:
        x, y = _TO_3857.transform(float(row["lon"]), float(row["lat"]))
        coords[int(row["point_id"])] = (float(x), float(y))
    return coords


def _prediction_probs(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    probs_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)[:, :-1]
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs_chunks.append(probs.detach().cpu())
    return torch.cat(probs_chunks, dim=0)


def _discrete_prediction_payload(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    shared: dict[str, torch.Tensor],
    point_xy_m: dict[int, tuple[float, float]],
    device: torch.device,
) -> dict[str, torch.Tensor | str]:
    probs = _prediction_probs(model=model, loader=loader, device=device)
    point_ids = shared["point_id_start"].to(dtype=torch.int64).cpu()
    start_xy = torch.tensor(
        [point_xy_m[int(point_id)] for point_id in point_ids.tolist()],
        dtype=torch.float32,
    )
    lon = shared["candidate_lon"][:, :-1]
    lat = shared["candidate_lat"][:, :-1]
    x_cand, y_cand = _TO_3857.transform(
        lon.to(dtype=torch.float64).reshape(-1).tolist(),
        lat.to(dtype=torch.float64).reshape(-1).tolist(),
    )
    coords_m = torch.tensor(list(zip(x_cand, y_cand)), dtype=torch.float32).view(
        lon.shape[0], lon.shape[1], 2
    )
    coords_m[:, :, 0] -= start_xy[:, 0].view(-1, 1)
    coords_m[:, :, 1] -= start_xy[:, 1].view(-1, 1)
    return {
        "prediction_kind": "discrete",
        "point_id_start": point_ids,
        "coords_m": coords_m,
        "probs": probs,
    }


def _load_prediction_shared(*, store_dir: Path, manifest) -> dict[str, torch.Tensor]:
    reader = PointStoreReader(store_dir=store_dir, manifest=manifest)
    return reader.load_shared_rows(
        keys=["point_id_start", "candidate_lon", "candidate_lat"],
        row_idxs=np.arange(manifest.n_rows, dtype=np.int64),
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
    num_workers = int(method_hparams.get("num_workers", 0) or 0)
    eval_max_items = (
        None
        if method_hparams.get("max_eval_items") is None
        else int(method_hparams["max_eval_items"])
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

    checkpoint_path = run_dir / "checkpoint_best.pt"
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

    point_xy_m = _start_xy_by_point_id(Path(resolved_artifacts["trajectories_path"]))
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for split_key in run_payload["evaluation_split_keys"]:
        dataset = _dataset_for(str(split_key))
        loader = _make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        )
        shared = _load_prediction_shared(
            store_dir=Path(point_stores[split_key]["store_dir"]),
            manifest=point_stores[split_key]["manifest"],
        )
        torch.save(
            _discrete_prediction_payload(
                model=runtime.model,
                loader=loader,
                shared=shared,
                point_xy_m=point_xy_m,
                device=device,
            ),
            pred_dir / f"{split_key}.pt",
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
