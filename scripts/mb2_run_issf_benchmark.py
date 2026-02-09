#!/usr/bin/env python
"""Run MoveBench2 iSSF benchmark (linear + MLP) over one or more studies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    # Allow running this script without installing the repo as a package.
    sys.path.insert(0, str(_REPO_ROOT))

from movebench2 import Study
from movebench2.models import LinearStepSelect, MLPStepSelect, random_baseline_metrics
from movebench2.train import collate_step_select, eval_step_select, fit_step_select


TOP_K_VALUES = (1, 5, 10, 25, 50)

LINEAR_HPARAMS = {
    "lr": 1e-2,
    "weight_decay": 1e-3,
    "max_steps": 1000,
    "eval_interval": 10,
    "patience": 50,
}

MLP_HPARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_steps": 2000,
    "eval_interval": 10,
    "patience": 50,
    "hidden_dims": (128,),
    "dropout": 0.6,
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run iSSF benchmark for MoveBench2.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--study", type=str, help="Study id (single).")
    g.add_argument("--studies", type=str, help="Comma-separated list of study ids.")
    p.add_argument("--config", type=str, default="configs/layers_all.yaml", help="Layer config YAML.")
    p.add_argument("--output-dir", type=str, default="results/mb2_issf", help="Output directory.")
    p.add_argument("--device", type=str, default="cpu", help="cpu or cuda (or cuda:0, etc).")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    return p.parse_args()


def _study_list(args: argparse.Namespace) -> list[str]:
    if args.study:
        return [str(args.study)]
    return [s.strip() for s in str(args.studies).split(",") if s.strip()]


def _make_loaders(ds_train, ds_test, *, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        ds_train,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_step_select,
        drop_last=False,
    )
    test_loader = DataLoader(
        ds_test,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_step_select,
        drop_last=False,
    )
    return train_loader, test_loader


def main() -> None:
    args = _parse_args()
    device = torch.device(str(args.device))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    for study_id in _study_list(args):
        study = Study(study_id)

        train_ds = study.issf_dataset(split="train", config=args.config, include_time=False, normalize=True)
        test_ds = study.issf_dataset(split="test", config=args.config, include_time=False, normalize=True)
        train_loader, test_loader = _make_loaders(
            train_ds,
            test_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Determine C,P for model construction and random baseline.
        x0, _valid0 = next(iter(train_loader))
        _, C, P = x0.shape
        random_metrics = random_baseline_metrics(int(P), TOP_K_VALUES)

        # Linear
        linear = LinearStepSelect(int(C))
        linear_fit = fit_step_select(
            linear,
            train_loader,
            test_loader,
            device,
            seed=int(args.seed),
            **LINEAR_HPARAMS,
        )
        linear_train = eval_step_select(linear, train_loader, device, topk=TOP_K_VALUES)
        linear_test = eval_step_select(linear, test_loader, device, topk=TOP_K_VALUES)

        # MLP
        mlp = MLPStepSelect(int(C), hidden_dims=MLP_HPARAMS["hidden_dims"], dropout=MLP_HPARAMS["dropout"])
        mlp_fit = fit_step_select(
            mlp,
            train_loader,
            test_loader,
            device,
            seed=int(args.seed),
            lr=float(MLP_HPARAMS["lr"]),
            weight_decay=float(MLP_HPARAMS["weight_decay"]),
            max_steps=int(MLP_HPARAMS["max_steps"]),
            eval_interval=int(MLP_HPARAMS["eval_interval"]),
            patience=int(MLP_HPARAMS["patience"]),
        )
        mlp_train = eval_step_select(mlp, train_loader, device, topk=TOP_K_VALUES)
        mlp_test = eval_step_select(mlp, test_loader, device, topk=TOP_K_VALUES)

        results[study_id] = {
            "config": str(args.config),
            "P": int(P),
            "random": random_metrics,
            "linear": {"fit": linear_fit, "train": linear_train, "test": linear_test},
            "mlp": {"fit": mlp_fit, "train": mlp_train, "test": mlp_test},
        }

        print(
            f"[{study_id}] "
            f"linear test nll={linear_test['nll']:.4f} "
            f"mlp test nll={mlp_test['nll']:.4f} "
            f"random nll={random_metrics['nll']:.4f}"
        )

    # Raw JSON
    (out_dir / "raw_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    # Summary CSVs
    def nll_table(split: str) -> pd.DataFrame:
        rows = []
        for study_id, r in results.items():
            rows.append(
                {
                    "study": study_id,
                    "Random": float(r["random"]["nll"]),
                    "Linear": float(r["linear"][split]["nll"]),
                    "MLP": float(r["mlp"][split]["nll"]),
                }
            )
        return pd.DataFrame(rows).set_index("study")

    nll_table("train").to_csv(out_dir / "nll_train.csv")
    nll_table("test").to_csv(out_dir / "nll_test.csv")

    for k in TOP_K_VALUES:
        def topk_table(split: str, k: int) -> pd.DataFrame:
            rows = []
            for study_id, r in results.items():
                rows.append(
                    {
                        "study": study_id,
                        "Random": float(r["random"]["topk"][int(k)]),
                        "Linear": float(r["linear"][split]["topk"][int(k)]),
                        "MLP": float(r["mlp"][split]["topk"][int(k)]),
                    }
                )
            return pd.DataFrame(rows).set_index("study")

        topk_table("train", int(k)).to_csv(out_dir / f"top{k}_train.csv")
        topk_table("test", int(k)).to_csv(out_dir / f"top{k}_test.csv")


if __name__ == "__main__":
    main()
