"""Local contribution demo; intentionally outside discovered experiments/."""

from movebench.bench.config import METHODS

METHODS["example_origin"] = {
    "implementation": "examples.origin_method:OriginMethod",
    "artifact_requirement": "points",
    "requires_normalization_stats": False,
    "model": {},
    "hparams": {},
}
METHODS["example_candidates"] = {
    "implementation": "examples.candidate_method:CandidateMethod",
    "artifact_requirement": "points",
    "requires_normalization_stats": True,
    "fit_reuse": "checkpoint",
    "model": {"include_time": True},
    "hparams": {"batch_size": 8, "max_steps": 2, "lr": 0.001},
}

EXPERIMENT = {
    "name": "contribution_examples",
    "studies": ["Accio_967b7813_raw_merged"],
    "timescales": ["12h"],
    "covariate_sets": ["alphaearth_only"],
    "methods": ["example_origin", "example_candidates"],
    "seed": 0,
}
