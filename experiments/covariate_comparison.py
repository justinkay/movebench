EXPERIMENT = {
    "name": "covariate_comparison",
    "studies": "mini",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": [
        # "all", # source these from main results
        "all_environmental",
        "alphaearth_only",
        "lite",
    ],
    "methods": ["issf_linear", "issf_mlp", "deepssf", "moveformer"],
    "seed": 0,
}
