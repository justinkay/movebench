from itertools import product


LR_PAIRS = [
    ("m3e-6_h3e-5", 3e-6, 3e-5),
    ("m1e-5_h1e-4", 1e-5, 1e-4),
    ("m3e-5_h3e-4", 3e-5, 3e-4),
    ("m3e-6_h1e-4", 3e-6, 1e-4),
    ("m3e-5_h1e-4", 3e-5, 1e-4),
]
CAPACITIES = [
    ("c2_d64", 2, 64),
    ("c4_d128", 4, 128),
    ("c8_d256", 8, 256),
]

METHOD_VARIANTS = [
    {
        "key": f"{lr_key}__{capacity_key}",
        "model": {
            "output_channels": output_channels,
            "dense_dim_hidden": dense_dim_hidden,
            "dropout": 0.1,
        },
        "hparams": {
            "lr_movement": lr_movement,
            "lr_habitat": lr_habitat,
            "weight_decay": 0.0,
        },
    }
    for (lr_key, lr_movement, lr_habitat), (
        capacity_key,
        output_channels,
        dense_dim_hidden,
    ) in product(LR_PAIRS, CAPACITIES)
]

EXPERIMENT = {
    "name": "deepssf_movebench_mini_sweep",
    "studies": "mini",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["alphaearth_only"],
    "methods": ["deepssf"],
    "method_variants": {"deepssf": METHOD_VARIANTS},
    "deepssf_data": "gpu",
    "seed": 0,
}
