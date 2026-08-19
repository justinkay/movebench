from itertools import product


BATCH_SIZES = [256, 1048]
LEARNING_RATES = [
    ("3e-3", 3e-3),
    ("1e-2", 1e-2),
    ("3e-2", 3e-2),
]

METHOD_VARIANTS = [
    {
        "key": f"bs{batch_size}_lr{lr_key}",
        "hparams": {
            "batch_size": batch_size,
            "lr": learning_rate,
        },
    }
    for batch_size, (lr_key, learning_rate) in product(
        BATCH_SIZES,
        LEARNING_RATES,
    )
]

EXPERIMENT = {
    "name": "issf_linear_movebench_mini_sweep",
    "studies": "mini",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["alphaearth_only"],
    "methods": ["issf_linear"],
    "method_variants": {"issf_linear": METHOD_VARIANTS},
    "seed": 0,
}
