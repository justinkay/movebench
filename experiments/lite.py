EXPERIMENT = {
    "name": "lite",
    "studies": "all",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["lite"],
    "methods": [
        "baseline_nomove",
        "baseline_constvel",
        "baseline_randomwalk",
        "issf_linear",
        "issf_mlp",
        "deepssf",
        "moveformer",
    ],
    "seed": 0,
}
