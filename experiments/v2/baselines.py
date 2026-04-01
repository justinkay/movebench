EXPERIMENT = {
    "name": "baselines",
    "studies": "all",
    "timescales": ["1h", "12h", "1d", "7d"],
    "covariate_sets": ["all"],
    "methods": [
        "baseline_nomove",
        "baseline_constvel",
        "baseline_randomwalk",
    ],
    "seed": 0,
}
