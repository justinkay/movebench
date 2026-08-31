"""MoveBench wrapper for the vendored MoveFormer selection model."""

from __future__ import annotations

from typing import Dict, List

import torch

from movebench.lib.moveformer.geo_transformers.models.any_horizon_forecast_transformer import (
    AnyHorizonForecastTransformer,
    SelectionHead,
)

from x_transformers.x_transformers import AttentionLayers


def build_moveformer_selection_model(
    *,
    cov_dim: int,
    depth: int = 6,
    heads: int = 8,
    var_len_training_v2: bool = True,
    max_len: int = 500,
    taxon_dim: int = 100,
) -> AnyHorizonForecastTransformer:
    """Instantiate MoveFormer VarCtx-style selection model over generic covariates.

    This matches `movebench/lib/moveformer/config/forecast_mbk+ung_12h_sel_nofut.yaml`
    as closely as possible, except that we treat MoveBench point-store env channels as a
    single continuous covariate vector (`cov` / `cov_cand`) so the baseline works with
    any layer plan.
    """
    if int(cov_dim) <= 0:
        raise ValueError(f"cov_dim must be > 0; got {cov_dim}")

    emb_dim = 128

    cont_feat_dims: Dict[str, int] = {
        "location": 3,
        "location_mov": 2,
        "location_cand": 3,
        "location_cand_mov": 2,
        "time_cont": 10,
        "time_cont_tgt": 10,
        "time_lmt": 2,
        "time_lmt_tgt": 2,
        "dtime_cont": 12,
        "taxon": int(taxon_dim),
        "cov": int(cov_dim),
        "cov_cand": int(cov_dim),
    }

    disc_feat_dims: Dict[str, int] = {
        "time_month": 12,
        "time_month_tgt": 12,
        "time_day": 31,
        "time_day_tgt": 31,
        "time_weekday": 7,
        "time_weekday_tgt": 7,
    }

    keys_in: List[str] = [
        "taxon",
        "location",
        "location_mov",
        "time_month",
        "time_day",
        "time_weekday",
        "time_cont",
        "time_lmt",
        "cov",
        "time_month_tgt",
        "time_day_tgt",
        "time_weekday_tgt",
        "time_cont_tgt",
        "time_lmt_tgt",
        "dtime_cont",
    ]

    encoder = AttentionLayers(
        dim=emb_dim,
        depth=int(depth),
        heads=int(heads),
        causal=True,
    )

    candidate_encoder_layers = [
        torch.nn.Linear(256, 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, 128),
    ]
    prediction_head = SelectionHead(
        candidate_encoder_layers=candidate_encoder_layers,
        embedding_in_keys=[
            "location_cand",
            "location_cand_mov",
            "cov_cand",
        ],
        in_features=128,
        hidden_features=128,
    )

    return AnyHorizonForecastTransformer(
        encoder=encoder,
        decoder=None,
        prediction_head=prediction_head,
        disc_feat_dims=disc_feat_dims,
        cont_feat_dims=cont_feat_dims,
        keys_in=keys_in,
        keys_future=None,
        emb_dim=emb_dim,
        learnable_nan=True,
        concat_enc_to_out=False,
        var_len_training=False,
        var_len_training_v2=bool(var_len_training_v2),
        max_len=int(max_len),
        optimizer_init=None,
        lr_warmup_init=None,
        lr_decay_init=None,
    )
