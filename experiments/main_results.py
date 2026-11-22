EXPERIMENT = {
    "name": "main_results",
    "studies": "all",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["alphaearth_only"],
    "methods": [
        "baseline_nomove",
        "baseline_constvel",
        "baseline_randomwalk",
        "issf_linear",
        "issf_mlp",
        "deepssf",
        "moveformer_full_predictonly",
    ],
    "deepssf_data": "gpu",
    "seed": 0,
}
