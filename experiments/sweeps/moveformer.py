METHOD_VARIANTS = [
    {"key": "current"},
    {
        "key": "publication_schedule",
        "hparams": {
            "batch_size": 24,
            "min_len": 2,
            "warmup_steps": 1000,
            "lr_gamma": 0.98,
        },
    },
    {"key": "lr1e-5", "hparams": {"lr": 1e-5}},
    {"key": "lr2e-4", "hparams": {"lr": 2e-4}},
    {"key": "d2_h4", "model": {"depth": 2, "heads": 4}},
    {"key": "d4_h8", "model": {"depth": 4, "heads": 8}},
    {"key": "ctx64", "hparams": {"max_len": 64}},
    {
        "key": "fullctx",
        "model": {"var_len_training_v2": False},
    },
]

EXPERIMENT = {
    "name": "moveformer_movebench_mini_sweep",
    "studies": "mini",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["alphaearth_only"],
    "methods": ["moveformer"],
    "method_variants": {"moveformer": METHOD_VARIANTS},
    "seed": 0,
}
