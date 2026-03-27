# Deduplicate Train-Fit Fixed-Interval Step Reconstruction

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, the repository will have one shared implementation for the train-fit fixed-interval step reconstruction flow that currently appears in multiple places. A contributor will be able to change the train-only gap cutoff or train-only spatial cutoff logic in one place and observe that `Study.download_rasters(...)`, point-feature extraction, chip dataset setup, and benchmark evaluation all still behave the same. The visible proof is that the targeted step-preparation, study-download, point-feature, and evaluation-runner tests all pass while the duplicated call-site logic is reduced to calls into `movebench/trajectories.py`.

## Progress

- [x] (2026-03-27 03:55Z) Re-read `PLANS.md`, the existing benchmark evaluation ExecPlan, and every duplicate call site in `movebench/study.py`, `movebench/features/points.py`, `movebench/datasets/chips.py`, `movebench/layers/download.py`, and `movebench/bench_v2/evaluation/runner.py`.
- [x] (2026-03-27 03:55Z) Chose the refactor seam: centralize train-fit gap cutoff, train-fit spatial cutoff, and requested-split step reconstruction in `movebench/trajectories.py`; do not refactor fixed-grid chip geometry in this change.
- [x] (2026-03-27 00:29Z) Added trajectory-helper tests in `tests/test_interval_step_preparation.py` for the new shared support object and train-fit selection helper.
- [x] (2026-03-27 00:29Z) Implemented `IntervalStepSupport`, `fit_interval_step_support(...)`, and `select_step_starts_with_support(...)` in `movebench/trajectories.py` while keeping the existing low-level primitives unchanged.
- [x] (2026-03-27 00:29Z) Migrated `movebench/study.py`, `movebench/features/points.py`, `movebench/datasets/chips.py`, `movebench/layers/download.py`, and `movebench/bench_v2/evaluation/runner.py` to the shared helper path.
- [x] (2026-03-27 00:29Z) Added direct wiring coverage for `ChipDataset(mode="steps")` in `tests/test_point_feature_store.py`.
- [x] (2026-03-27 00:29Z) Ran the targeted regression tests and reviewed passing outputs for the updated trajectory, study, point-feature, chip-dataset, and evaluation-runner paths.
- [x] (2026-03-27 00:48Z) Removed the now-unused `movebench/layers/download.py:get_buffer_m_for_study(...)` wrapper, switched the remaining diagnostic scripts to `fit_interval_step_support(...)`, and updated the interval-step tests to cover the underlying support helpers directly.

## Surprises & Discoveries

- Observation: the strongest duplication is not generic step math, but the same three-step orchestration repeated in several modules: train-only gap fitting, train-only spatial cutoff fitting, then requested-split `select_step_starts(...)`.
  Evidence: `movebench/study.py`, `movebench/features/points.py`, `movebench/datasets/chips.py`, and `movebench/bench_v2/evaluation/runner.py` each repeat this sequence with only small differences in which rows they pass to `select_step_starts(...)`.

- Observation: after the caller migrations, `get_buffer_m_for_study(...)` no longer served any runtime path and survived only in two diagnostic scripts plus tests.
  Evidence: repository-wide search after the refactor showed non-test references only in `scripts/list_studies_by_7d_extent.py` and `scripts/diagnose_download_sizes.py`.

- Observation: the fixed-grid chip geometry is duplicated between chips and evaluation, but that duplication is conceptually separate from train-fit step reconstruction.
  Evidence: `movebench/datasets/chips.py` duplicates the metric grid bounds math from `movebench/bench_v2/evaluation/runner.py`, but none of the other duplicate call sites use that logic.

- Observation: there was no pre-existing direct `ChipDataset(mode="steps")` test covering the train-fit support wiring.
  Evidence: repository-wide test search found no existing `ChipDataset` or `step_buffer_m` assertions, so the migration needed one focused caller-level test in an existing test module.

- Observation: once the helper was no longer used by runtime code, keeping a wrapper with special policy was worse than deleting it.
  Evidence: the remaining script callers already pass `train_df`, so `fit_interval_step_support(train_df, ...)` is the direct expression of what they actually need.

## Decision Log

- Decision: introduce the shared refactor in `movebench/trajectories.py`, not in a new module.
  Rationale: the existing trajectory-loading and interval-step primitives already live there, and the duplicated behavior is an orchestration layer over those primitives rather than a new subsystem.
  Date/Author: 2026-03-27 / assistant

- Decision: keep `fit_interval_gap_cutoffs(...)`, `fit_interval_step_length_cutoff_m(...)`, and `select_step_starts(...)` as public low-level building blocks.
  Rationale: existing tests and callers already rely on those functions, and replacing them would widen the refactor unnecessarily.
  Date/Author: 2026-03-27 / assistant

- Decision: do not refactor the fixed-grid chip transform in this change.
  Rationale: the user asked to de-duplicate the repo, but the biggest behavioral duplication is the train-fit step-selection path; moving the grid transform too would add a second independent refactor surface and more regression risk.
  Date/Author: 2026-03-27 / assistant

- Decision: use a small shared dataclass plus helper functions, not a broader orchestration object.
  Rationale: callers only need the fitted support object and a thin selection wrapper that reuses it. The train split itself is already explicit in each caller, so a dataclass plus one fit helper and one selection helper keeps the change explicit and low-risk.
  Date/Author: 2026-03-27 / assistant

- Decision: delete `get_buffer_m_for_study(...)` instead of keeping a wrapper with a stricter policy than the surviving helpers.
  Rationale: after the dedup refactor, the wrapper was no longer used in runtime code and the only remaining script callers already filter to `train_df` explicitly. Removing the wrapper keeps the API smaller and avoids carrying an extra contract surface.
  Date/Author: 2026-03-27 / assistant

## Outcomes & Retrospective

The repository now has one shared implementation for train-fit fixed-interval support in `movebench/trajectories.py`, and the duplicated call-site setup code has been replaced in study downloads, point-feature extraction, chip dataset setup, diagnostic buffer scripts, and benchmark evaluation. The refactor stayed focused on the repeated train-fit step-selection path and did not broaden into the separate fixed-grid transform duplication.

The targeted tests passed after the migration, including new helper coverage and a direct chip-dataset wiring test. No new functional gaps were found in the migrated callers. The remaining duplication candidate is the fixed-grid support transform shared by chips and evaluation, but that is intentionally deferred because it is a separate refactor surface.

## Context and Orientation

`movebench/trajectories.py` contains the repository’s fixed-interval trajectory primitives. The important existing functions are `fit_interval_gap_cutoffs(...)`, which learns per-individual burst gap cutoffs from training rows; `fit_interval_step_length_cutoff_m(...)`, which learns a train-only spatial support radius from valid fixed-interval steps; and `select_step_starts(...)`, which reconstructs valid fixed-interval steps for a dataframe.

Four modules currently duplicate the same orchestration around those primitives.

`movebench/study.py` loads `trajectories.csv`, fits train-only cutoffs and buffer, then selects valid requested-split starts so raster downloads only cover eligible fixed-horizon rows.

`movebench/features/points.py` repeats the same process to reconstruct requested steps and train-only steps before building the empirical sampler and writing point-feature stores.

`movebench/datasets/chips.py` repeats the same process in `ChipDataset(mode="steps")` so deepSSF chips use the same train-derived support as the rest of the benchmark.

`movebench/bench_v2/evaluation/runner.py` repeats the same process to rebuild held-out truth rows and the shared eligible subset for final evaluation.

The goal of this refactor is not to change semantics. “Train-fit” still means gap cutoffs and the spatial cutoff are always fit from `set == "train"` rows only. “Requested splits” still means the caller decides whether step starts are reconstructed from one split, several splits combined, or all rows. The shared helper must keep that distinction explicit.

## Plan of Work

First, extend `tests/test_interval_step_preparation.py` with tests that exercise the new shared orchestration directly. The tests should prove that a helper in `movebench/trajectories.py` can return a train-fit buffer together with reconstructed steps for requested rows, and that it preserves the current train-only gap and spatial cutoff behavior when training and requested rows differ.

Next, add one or two small helper functions in `movebench/trajectories.py` that wrap the existing primitives. The helper should accept the full trajectories dataframe, the requested rows to reconstruct, and the fixed-step parameters. It should fit the train-only gap cutoffs from `set == "train"` rows, fit the train-only spatial cutoff from the same rows, and return the reconstructed requested steps plus the fit artifacts needed by callers. Keep the return shape simple and explicit, either as a small dataclass or a tuple-like dataclass with named fields.

After the helper exists, replace the duplicated orchestration in `movebench/study.py`, `movebench/features/points.py`, `movebench/datasets/chips.py`, and `movebench/bench_v2/evaluation/runner.py` with calls to that helper. Preserve each caller’s current split-selection behavior. `study.py` still needs the subset of original rows whose `point_id` appears in the reconstructed starts. `features/points.py` still needs both combined requested steps and train-only steps for sampler fitting. `chips.py` still needs all step rows for the selected dataset split plus the train-derived buffer. `runner.py` still needs ordered held-out steps and the train-derived buffer for support checks.

Finally, run only the targeted tests that cover the refactor: the interval-step helper tests, the study download tests, the point-feature store tests, and the benchmark evaluation-runner tests. If any test reveals a behavior mismatch, update the helper or the caller to match the pre-refactor behavior rather than expanding scope.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the shared-helper tests and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_interval_step_preparation

Refactor the shared trajectory orchestration and its callers, then run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_study_download_buffers
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_feature_store
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_eval_runner

Observed result on 2026-03-27: all four targeted commands passed. `tests.test_interval_step_preparation` ran 12 tests, `tests.test_study_download_buffers` ran 6 tests, `tests.test_point_feature_store` ran 8 tests, and `tests.bench_v2.test_eval_runner` ran 2 tests.

Observed result on 2026-03-27 after removing the obsolete wrapper: `tests.test_interval_step_preparation` passed again with 12 tests, and `python -m py_compile scripts/list_studies_by_7d_extent.py scripts/diagnose_download_sizes.py` succeeded.

Expected result after completion: all targeted tests pass, and the new helper is the only place where the repository fits train-only gap and spatial cutoffs before reconstructing requested fixed-interval steps.

## Validation and Acceptance

Acceptance is reached when the following behaviors are observable.

`tests.test_interval_step_preparation` proves the new helper matches the existing train-only gap and spatial cutoff semantics.

`tests.test_study_download_buffers` proves `Study.download_rasters(...)` still downloads only the original rows whose fixed-horizon starts survive the train-fit filters.

`tests.test_point_feature_store` proves point-feature extraction still reconstructs combined requested steps, derives the sampler from train-only steps, and partitions results by split without changing the selected point ids.

`tests.bench_v2.test_eval_runner` proves the evaluation runner still reconstructs held-out truth rows correctly and writes the same scoring artifacts and MLflow metrics.

## Idempotence and Recovery

This refactor is code-only and idempotent. The helper can be introduced first and callers migrated one by one while keeping tests green. If one caller migration regresses behavior, revert just that caller to the previous inline orchestration, keep the helper tests, and retry without affecting artifacts on disk.

## Artifacts and Notes

The central helper introduced by this plan should encapsulate the repeated shape:

    train_df = df[df[SET_COL] == "train"].copy()
    gap_cutoffs = fit_interval_gap_cutoffs(train_df, ...)
    buffer_m = fit_interval_step_length_cutoff_m(train_df, ..., gap_cutoffs=gap_cutoffs, ...)
    steps = select_step_starts(requested_df, ..., gap_cutoffs=gap_cutoffs, max_step_length_m=buffer_m)

The implemented helper split this into:

    train_df = df[df[SET_COL] == "train"].copy()
    support = fit_interval_step_support(train_df, ...)
    steps = select_step_starts_with_support(requested_df, ..., support=support)

The refactor is complete only when the old three-step inline shape no longer appears in the migrated callers.

## Interfaces and Dependencies

Define the new shared helper in `movebench/trajectories.py`. It must build on the existing public functions `fit_interval_gap_cutoffs(...)`, `fit_interval_step_length_cutoff_m(...)`, and `select_step_starts(...)`; do not reimplement their internals.

The helper must expose enough information for callers to preserve existing behavior. At minimum, callers need the train-fit `gap_cutoffs`, the train-fit `buffer_m`, and the reconstructed `steps` for a caller-supplied dataframe.

Implemented interfaces:

    @dataclass(frozen=True)
    class IntervalStepSupport:
        gap_cutoffs: IntervalGapCutoffs
        max_step_length_m: float

    def fit_interval_step_support(...) -> IntervalStepSupport: ...
    def select_step_starts_with_support(...) -> pd.DataFrame: ...

Revision note (2026-03-27): created this plan after the user explicitly expanded the task from the benchmark evaluation scope to a repo-wide deduplication refactor.
Revision note (2026-03-27): updated the plan after implementation to record the shipped helper shape, the added chip-dataset test coverage, and the exact passing validation commands.
Revision note (2026-03-27): updated the plan after removing the obsolete `get_buffer_m_for_study(...)` wrapper and switching the remaining scripts to the shared support helper directly.
Revision note (2026-03-27): updated the plan after collapsing the support-fitting API from two methods to one and making the train split explicit at each caller.
