from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from pyproj import Transformer

from movebench.bench_v2.train import TrainerConfig, run_training_loop
from movebench.datasets.ssf import iSSFDataset
from movebench.features.points import load_point_store_shared
from movebench.models.ssf import LinearStepSelect, MLPStepSelect
from movebench.trajectories import (
    INDIVIDUAL_COL,
    LAT_COL,
    LON_COL,
    POINT_ID_COL,
    TIME_COL,
)

_TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


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


@torch.no_grad()
def _eval_step_select_tensor(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total = 0
    for start in range(0, int(x.shape[0]), int(batch_size)):
        end = min(int(x.shape[0]), start + int(batch_size))
        xb = x[start:end]
        if xb.device != device:
            xb = xb.to(device)
        logits = model(xb)
        loss = _loss_step_select(logits)
        total_nll += float(loss.item()) * int(xb.shape[0])
        total += int(xb.shape[0])
    return {"nll": float(total_nll / float(total))}


class _ISSFRuntime:
    early_stop_metric = "nll"

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        x_train: torch.Tensor,
        x_eval: torch.Tensor,
        early_stop_split_key: str,
        device: torch.device,
        batch_size: int,
        lr: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        self.model = model.to(device)
        self.x_train = x_train
        self.x_eval = x_eval
        self.early_stop_split_key = str(early_stop_split_key)
        self.device = device
        self.batch_size = int(batch_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))
        self._perm = torch.randperm(int(self.x_train.shape[0]), generator=self._rng)
        self._pos = 0

    def _next_batch(self) -> torch.Tensor:
        n_train = int(self.x_train.shape[0])
        if self._pos >= n_train:
            self._perm = torch.randperm(n_train, generator=self._rng)
            self._pos = 0
        end = min(n_train, self._pos + int(self.batch_size))
        idx = self._perm[self._pos : end]
        self._pos = end
        xb = self.x_train.index_select(0, idx.to(device=self.x_train.device))
        if xb.device != self.device:
            xb = xb.to(self.device)
        return xb

    def train_step(self, step: int) -> dict[str, float]:
        xb = self._next_batch()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(xb)
        loss = _loss_step_select(logits)
        loss.backward()
        self.optimizer.step()
        return {"loss/train_step": float(loss.item())}

    def evaluate(self, split_key: str) -> dict[str, float]:
        if split_key == "train":
            metrics = _eval_step_select_tensor(
                self.model,
                self.x_train,
                self.device,
                batch_size=self.batch_size,
            )
            return {"nll": float(metrics["nll"])}
        if split_key != self.early_stop_split_key:
            raise KeyError(f"Unknown split for iSSF runtime: {split_key!r}")
        metrics = _eval_step_select_tensor(
            self.model,
            self.x_eval,
            self.device,
            batch_size=self.batch_size,
        )
        return {"nll": float(metrics["nll"])}

    def state_dict(self) -> dict[str, object]:
        return {
            "model": {
                key: value.detach().cpu().clone()
                for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "perm": self._perm.clone(),
            "pos": int(self._pos),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.model.load_state_dict(dict(state["model"]))
        self.optimizer.load_state_dict(dict(state["optimizer"]))
        self._perm = torch.as_tensor(state["perm"], dtype=torch.long, device="cpu")
        self._pos = int(state["pos"])


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


def _discrete_prediction_payload(
    *,
    model: torch.nn.Module,
    x: torch.Tensor,
    shared: dict[str, torch.Tensor],
    point_xy_m: dict[int, tuple[float, float]],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor | str]:
    if x.ndim != 3 or int(x.shape[2]) <= 1:
        raise ValueError("iSSF predictions require at least one pseudo-candidate")

    probs_chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), int(batch_size)):
            end = min(int(x.shape[0]), start + int(batch_size))
            xb = x[start:end]
            if xb.device != device:
                xb = xb.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)[:, :-1]
            probs = probs / probs.sum(dim=1, keepdim=True)
            probs_chunks.append(probs.detach().cpu())
    probs = torch.cat(probs_chunks, dim=0)

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
    coords_m = torch.tensor(
        list(zip(x_cand, y_cand)),
        dtype=torch.float32,
    ).view(lon.shape[0], lon.shape[1], 2)
    coords_m[:, :, 0] -= start_xy[:, 0].view(-1, 1)
    coords_m[:, :, 1] -= start_xy[:, 1].view(-1, 1)
    return {
        "prediction_kind": "discrete",
        "point_id_start": point_ids,
        "coords_m": coords_m,
        "probs": probs,
    }


def _run(
    *,
    run_payload,
    resolved_artifacts,
    run_dir,
    tracker,
    model_kind: str,
) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point_stores = resolved_artifacts["point_stores"]
    run_dir = Path(run_dir)

    method_hparams = dict(run_payload.get("method_hparams", {}))
    batch_size = int(method_hparams.get("batch_size", 256))
    train_dataset = iSSFDataset(
        study_dir=resolved_artifacts["study_dir"],
        trajectories_path=resolved_artifacts["trajectories_path"],
        plan=resolved_artifacts["plan"],
        sampler_config=resolved_artifacts["sampler"],
        split="train",
        subset_band_ids=list(run_payload["band_ids"]),
    )
    eval_dataset = iSSFDataset(
        study_dir=resolved_artifacts["study_dir"],
        trajectories_path=resolved_artifacts["trajectories_path"],
        plan=resolved_artifacts["plan"],
        sampler_config=resolved_artifacts["sampler"],
        split=str(run_payload["early_stop_split_key"]),
        subset_band_ids=list(run_payload["band_ids"]),
    )
    train_x = train_dataset.build_expanded_x(
        cache_device=torch.device("cpu"),
        dtype=torch.float32,
    )
    eval_x = eval_dataset.build_expanded_x(
        cache_device=torch.device("cpu"),
        dtype=torch.float32,
    )

    in_dim = int(train_x.shape[1])
    if model_kind == "linear":
        model = LinearStepSelect(in_dim)
    elif model_kind == "mlp":
        model_cfg = dict(run_payload.get("method_model", {}))
        model = MLPStepSelect(
            in_dim,
            hidden_dims=tuple(model_cfg.get("hidden_dims", [128])),
            dropout=float(model_cfg.get("dropout", 0.6)),
        )
    else:
        raise ValueError(f"Unsupported iSSF model kind: {model_kind!r}")

    runtime = _ISSFRuntime(
        model=model,
        x_train=train_x,
        x_eval=eval_x,
        early_stop_split_key=str(run_payload["early_stop_split_key"]),
        device=device,
        batch_size=batch_size,
        lr=float(method_hparams["lr"]),
        weight_decay=float(method_hparams["weight_decay"]),
        seed=int(run_payload["seed"]),
    )
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
        dataset = iSSFDataset(
            study_dir=resolved_artifacts["study_dir"],
            trajectories_path=resolved_artifacts["trajectories_path"],
            plan=resolved_artifacts["plan"],
            sampler_config=resolved_artifacts["sampler"],
            split=str(split_key),
            subset_band_ids=list(run_payload["band_ids"]),
        )
        x_split = dataset.build_expanded_x(
            cache_device=torch.device("cpu"),
            dtype=torch.float32,
        )
        shared = load_point_store_shared(Path(point_stores[split_key]["store_dir"]))
        torch.save(
            _discrete_prediction_payload(
                model=runtime.model,
                x=x_split,
                shared=shared,
                point_xy_m=point_xy_m,
                device=device,
                batch_size=batch_size,
            ),
            pred_dir / f"{split_key}.pt",
        )

    return {
        "status": "completed",
        "train_steps": int(fit["completed_steps"]),
        "best_step": int(fit["best_step"]),
        "early_stop_split_key": str(run_payload["early_stop_split_key"]),
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
