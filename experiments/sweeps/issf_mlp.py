from itertools import product


ARCHITECTURES = [
    ("h64", [64]),
    ("h128", [128]),
    ("h128x2", [128, 128]),
]
OPTIMIZER_PROFILES = [
    ("bs256_lr3e-4", 256, 3e-4),
    ("bs256_lr1e-3", 256, 1e-3),
    ("bs1048_lr1e-3", 1048, 1e-3),
    ("bs1048_lr3e-3", 1048, 3e-3),
]

METHOD_VARIANTS = [
    {
        "key": f"{architecture_key}__{optimizer_key}",
        "model": {"hidden_dims": list(hidden_dims)},
        "hparams": {
            "batch_size": batch_size,
            "lr": learning_rate,
        },
    }
    for (architecture_key, hidden_dims), (
        optimizer_key,
        batch_size,
        learning_rate,
    ) in product(ARCHITECTURES, OPTIMIZER_PROFILES)
]

EXPERIMENT = {
    "name": "issf_mlp_movebench_mini_sweep",
    "studies": "mini",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["alphaearth_only"],
    "methods": ["issf_mlp"],
    "method_variants": {"issf_mlp": METHOD_VARIANTS},
    "seed": 0,
}
