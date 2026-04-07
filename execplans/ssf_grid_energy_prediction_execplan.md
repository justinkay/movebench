# Replace Sampled SSF Energy-Score Prediction With Shared `101x101` Grid Support

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, every SSF-style benchmark method that currently relies on sampled pseudo-candidates for final energy-score prediction will instead predict on the same fixed square benchmark grid already used by deepSSF. In this repository that grid is `101x101`, not `100x100`: the active timescale config in [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py) sets `output_hw=(101, 101)` for every benchmark timescale, and the plan intentionally preserves that geometry so support, pixel size, and exact grid evaluation stay aligned across methods.

The user-visible result is that `baseline_randomwalk`, `issf_linear`, and `issf_mlp` will stop writing sampled discrete prediction artifacts with `coords_m` and `probs`. They will instead write `prediction_kind="grid"` artifacts with per-step `scores` over the full benchmark grid, just like deepSSF already does. For the SSF methods those scores will represent the model score plus the log of the discrete movement probability mass assigned to each grid cell. Final energy scores for those methods will therefore be exact on one shared support instead of Monte Carlo approximations over sampled pseudo-candidates. The current sampled final-prediction knob `energy_score_num_candidates` will disappear because it no longer has any meaning.

The visible proof is threefold. First, targeted runtime tests will fail before the refactor because iSSF and the random-walk baseline still emit discrete sampled supports, then pass after the change when those methods emit grid scores. Second, one real completed SSF run should regenerate `predictions/<split>.pt` whose top-level payload contains `prediction_kind == "grid"` and `scores.shape[-2:] == (101, 101)`. Third, the exact grid evaluator in [movebench/bench/evaluation/energy.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/energy.py) will become the only active final-evaluation path for SSF-style methods, eliminating the prediction-time sampled-support logic that currently bloats [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py).

## Progress

- [x] (2026-04-07 19:41Z) Re-read [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md), the active benchmark config, the iSSF, deepSSF, MoveFormer, and baseline executors, the exact energy evaluator, `ChipDataset(mode="steps")`, the deepSSF dataset/cache wrappers, and the earlier iSSF sampled-support ExecPlan.
- [x] (2026-04-07 19:41Z) Confirmed that the active benchmark grid is `101x101`, not `100x100`, and that [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) already defines the deterministic step grid used by deepSSF.
- [x] (2026-04-07 19:41Z) Confirmed that the active iSSF sampled-support path is isolated to final prediction generation in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py), while the evaluator already supports exact `prediction_kind="grid"` artifacts.
- [x] (2026-04-07 20:43Z) Audited the draft plan for duplication against [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py), [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py), [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py), and [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py), then revised Milestone 2 so it requires extraction and reuse of existing geometry, movement-feature, and grid-artifact logic instead of adding parallel implementations.
- [x] (2026-04-07 20:43Z) Audited the remaining high-risk gaps in the plan against [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py), [movebench/bench/executors/baselines.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/baselines.py), and [tests/bench/test_issf_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_issf_runtime.py), then revised the plan to specify exact `q(cell)` semantics, bounded grid batching, replacement coverage for sampled-path helper tests, a single source of truth for random-walk step context, and a Milestone 5 prerequisite check.
- [x] (2026-04-07 22:15Z) Added test-first coverage in [tests/bench/test_issf_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_issf_runtime.py), [tests/bench/test_ssf_grid_prediction.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_ssf_grid_prediction.py), and [tests/bench/test_experiment_config.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_experiment_config.py). The pre-change run failed exactly where expected: `movebench.bench.executors.ssf_grid` did not exist yet, [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py) still called `_parametric_prediction_payload(...)`, and `energy_score_num_candidates` was still present in active config.
- [x] (2026-04-07 22:15Z) Implemented one shared SSF grid-prediction helper path in [movebench/bench/executors/ssf_grid.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/ssf_grid.py), made [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) optionally honor `selected_splits` for `mode="steps"`, migrated `baseline_randomwalk`, `issf_linear`, and `issf_mlp` to grid artifacts, and removed `energy_score_num_candidates` from [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py).
- [x] (2026-04-07 22:15Z) Kept deepSSF behavior stable. [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py) was left unchanged; the refactor reused only the existing grid artifact contract, chip-grid geometry, and iSSF movement-feature semantics.
- [x] (2026-04-07 22:15Z) Ran the targeted benchmark suites with `LD_LIBRARY_PATH=/data/vision/beery/scratch/justin/micromamba/envs/movebench/lib:$LD_LIBRARY_PATH /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_issf_runtime tests.bench.test_ssf_grid_prediction tests.bench.test_runner tests.bench.test_energy tests.bench.test_experiment_config` and got `Ran 29 tests in 6.000s` / `OK`.
- [x] (2026-04-07 22:15Z) Ran the repository-level `python -m unittest -v` command. It did not go fully green for unrelated reasons: six `tests.bench.test_remove_dynamic_world_from_7d_high_res.*` cases failed because `/data/vision/beery/scratch/justin/movebench/configs/layers/all_ae_no_dw_7d_7d.yaml` is missing, [tests/test_local_tiff_window_cap.py](/data/vision/beery/scratch/justin/movebench/tests/test_local_tiff_window_cap.py) expected `max_window_hw == (1000, 1000)` but observed `(250, 250)`, and [tests/test_slurm_extract.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_extract.py) expected `--time 7-00:00:00` but observed `--time 2-00:00:00`. The same run still covered the new SSF grid tests successfully. Completed the real-run preflight (`True`, `True`) and regenerated a local [baseline_randomwalk](/data/vision/beery/scratch/justin/movebench/results/bench/runs/validation__study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__gridcheck/predictions/test_individual.pt) artifact whose inspection transcript was `grid`, `(367, 101, 101)`, `566.6628807268355`, `False False`.

## Surprises & Discoveries

- Observation: the user’s “100x100” request maps onto the repository’s existing `101x101` benchmark support, not a new grid size.
  Evidence: every active timescale in [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py) defines `output_hw = (101, 101)`, and [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) requires square step chips with that `output_hw` for deepSSF.

- Observation: deepSSF already supplies the exact support contract the user wants for the SSF methods.
  Evidence: [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py) writes `prediction_kind="grid"` with raw `scores` and `pixel_size_m`, and [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py) already evaluates those artifacts with the exact grid energy path.

- Observation: `ChipDataset(mode="steps")` already materializes the same deterministic per-step grid that deepSSF trains on, so iSSF does not need a new custom raster-grid extractor just to produce grid predictions.
  Evidence: [movebench/datasets/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf.py) is only a wrapper around `ChipDataset(mode="steps")`, and the cache builder in [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py) confirms that the cached deepSSF support comes directly from that dataset.

- Observation: the main code-size win is in prediction-time iSSF logic, not in training-time feature extraction.
  Evidence: the current sampled-support final prediction path is concentrated in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py) through `_parametric_prediction_payload(...)`, `_coords_from_shared(...)`, `_prediction_steps_for_split(...)`, and the `energy_score_num_candidates` config plumbing. Training still needs point-store candidate sampling and is outside this user request.

- Observation: the sampled-support investigation already showed enough Monte Carlo noise to justify replacing sampled final prediction with an exact shared-grid path.
  Evidence: [paired_vs_reference_summary.csv](/data/vision/beery/scratch/justin/movebench/results/investigations/energy_score_samples/garthe_1d_linear_test_individual_max64_seeds0_4_ref5000/paired_vs_reference_summary.csv) reports `mean_abs_delta_energy_score=2118.767177606352` for `1000` versus `5000` sampled iSSF supports on a real held-out slice.

- Observation: for iSSF, the learned grid score is not a pure habitat raster that can simply be multiplied by a movement raster after the fact.
  Evidence: [movebench/models/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/ssf.py) shows that `LinearStepSelect.forward(...)` adds environment, movement, and environment-by-movement interaction terms, while `MLPStepSelect.forward(...)` feeds concatenated environment and movement features into one network. The faithful grid contract is therefore `log q(cell) + model_score(cell)`, not a post hoc habitat-only factorization.

- Observation: much of the “new” Milestone 2 work already exists in reusable form, but it is spread across the evaluator, the chip dataset, the iSSF dataset, and the deepSSF executor.
  Evidence: [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py) already reconstructs ordered support rows and grid coordinates; [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) already defines deterministic step chips with `step_starts_df`, `step_buffer_m`, and `pixel_size_m`; [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py) already defines iSSF movement-feature normalization in `_movement_tensor(...)`; and [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py) already packs the standard grid prediction artifact.

- Observation: `ChipDataset(mode="steps")` does not automatically reuse the deepSSF WebDataset cache.
  Evidence: [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py) constructs `ChipDataset` directly for `chip_dataset(...)`, while the cache lives behind the separate explicit path in [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py) and is consumed explicitly by [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py). This plan therefore keeps cache reuse out of scope instead of implying automatic reuse.

- Observation: the plan previously referenced a deepSSF runtime test module that does not exist.
  Evidence: the active benchmark test files under [tests/bench](/data/vision/beery/scratch/justin/movebench/tests/bench) include [test_issf_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_issf_runtime.py), [test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py), [test_energy.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_energy.py), and [test_experiment_config.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_experiment_config.py), but no `test_deepssf_runtime.py`.

- Observation: the existing iSSF runtime tests pin more than the sampled-support artifact shape. They also pin bounded prediction batching, prediction worker propagation, and resume-time regeneration behavior that still matter after the grid migration.
  Evidence: [tests/bench/test_issf_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_issf_runtime.py) currently checks final-artifact generation, payload support trimming, scaled prediction batching plus worker propagation, and split regeneration behavior, so the grid refactor must replace those assertions rather than dropping them.

- Observation: the highest implementation risk is not “grid support” in general but the exact definition of `q(cell)` on a finite `101x101` window.
  Evidence: [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) currently samples continuous pseudo-steps by drawing independent step lengths and turn angles, then mapping them to Cartesian endpoints. Moving from that continuous proposal to deterministic per-cell mass requires explicit rules for off-grid mass, renormalization, zero-mass cells, and empirical fallback.

- Observation: `ChipDataset(mode="steps")` needed one extra hook to stay aligned with the evaluator when point-store manifests carry `selected_splits` such as `train + test_temporal`.
  Evidence: the new [tests/bench/test_ssf_grid_prediction.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_ssf_grid_prediction.py) runtime assertion requires `baseline_randomwalk` to recover `point_id_start == 3` from a held-out split whose previous-point context lives in the training split. Without the new `selected_splits` parameter in [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py), the reconstructed held-out steps drifted from the evaluator’s ordering.

- Observation: real iSSF grid regeneration on full local covariates is materially slower than the baseline grid path, even when checkpoint reuse removes training.
  Evidence: fresh validation directories for `Garthe` `1d` and `Lago` `1h` reused checkpoints immediately but then spent repeated polling intervals inside prediction after logging `starting iSSF prediction split=... grid=101x101 workers=0`, while the real [baseline_randomwalk](/data/vision/beery/scratch/justin/movebench/results/bench/runs/validation__study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__gridcheck/predictions/test_individual.pt) validation run completed quickly and still exercised the same shared movement-prior/grid-artifact path required by this plan.

- Observation: the repository-level unittest suite already had unrelated breakage outside this refactor.
  Evidence: the exact failing names were `tests.bench.test_remove_dynamic_world_from_7d_high_res.{test_apply_is_noop_for_already_patched_study,test_apply_resumes_mid_study_with_mixed_old_and_patched_files,test_apply_rewrites_tiffs_and_updates_metadata,test_dry_run_reports_patchable_without_mutating_files,test_dry_run_skips_incompatible_study,test_parallel_apply_rewrites_all_files}` with `FileNotFoundError: /data/vision/beery/scratch/justin/movebench/configs/layers/all_ae_no_dw_7d_7d.yaml`, plus [tests/test_local_tiff_window_cap.py](/data/vision/beery/scratch/justin/movebench/tests/test_local_tiff_window_cap.py) and [tests/test_slurm_extract.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_extract.py) expectation mismatches described above.

## Decision Log

- Decision: preserve the repository’s existing `101x101` benchmark support rather than introducing a new `100x100` geometry.
  Rationale: the whole active benchmark stack already assumes `output_hw=(101,101)` for deepSSF, exact grid evaluation, pixel size reconstruction, and target rasterization. Changing the grid size would widen scope without helping the user’s actual goal of replacing sampled final prediction.
  Date/Author: 2026-04-07 / Codex

- Decision: scope this plan to the SSF-style methods `baseline_randomwalk`, `issf_linear`, `issf_mlp`, and the shared grid contract already used by `deepssf`. Leave `baseline_nomove`, `baseline_constvel`, and `moveformer` unchanged.
  Rationale: `baseline_randomwalk`, `issf_linear`, and `issf_mlp` are the methods whose final prediction currently depends on sampled pseudo-candidate support. `deepssf` already uses the desired grid contract and should act as the reference path. The deterministic baselines are point-mass predictors, not movement-kernel grid models, and MoveFormer still depends on its own discrete candidate-set contract.
  Date/Author: 2026-04-07 / Codex

- Decision: do not add deepSSF cache requirements to iSSF or the random-walk baseline just to get grid support. Reuse `ChipDataset(mode="steps")` and the existing study rasters directly for final SSF grid prediction.
  Rationale: the user asked for simpler code and the removal of sampled final prediction, not a broader prerequisite expansion. `ChipDataset(mode="steps")` already defines the same support as deepSSF without changing benchmark readiness rules or requiring caches for methods that previously ran from point stores only.
  Date/Author: 2026-04-07 / Codex

- Decision: express SSF-style grid predictions as raw per-cell `scores`, not pre-normalized `probs_hw`.
  Rationale: this matches the existing deepSSF contract, lets the evaluator own the final `softmax`, and makes the iSSF/grid baseline contract identical to the already-supported grid path in [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py).
  Date/Author: 2026-04-07 / Codex

- Decision: for iSSF-style methods, the per-cell score must be `log movement_prior + model_score`, so the final grid probability is proportional to `movement_prior * exp(model_score)`.
  Rationale: this matches the semantics of the current sampled-support runtime, where pseudo-candidates are first drawn from the fitted movement sampler and then reweighted by the learned iSSF selection score. The grid refactor makes that base measure explicit instead of implicit in Monte Carlo support sampling.
  Date/Author: 2026-04-07 / Codex

- Decision: define the movement prior on the benchmark grid as discrete per-cell probability mass, not merely a continuous density evaluated at pixel centers.
  Rationale: the user explicitly wants “model predictions * discrete movement prob for each grid cell”, and the energy-score grid evaluator ultimately consumes normalized per-cell probabilities. On a uniform grid, density and mass differ only by a constant factor, but defining the contract in terms of cell mass avoids ambiguity and keeps the implementation aligned with the artifact semantics.
  Date/Author: 2026-04-07 / Codex

- Decision: `q(cell)` is the proposal mass conditional on the fixed benchmark window, not the unconditional continuous proposal over all space.
  Rationale: the user wants all SSF-style methods evaluated on the same `101x101` support as deepSSF. The prediction artifact therefore needs a normalized distribution over that support. The implementation must compute raw in-window cell masses, accumulate raw off-grid mass separately for diagnostics, then renormalize only the in-window cells before combining with model score.
  Date/Author: 2026-04-07 / Codex

- Decision: zero in-window proposal mass maps to `-inf` grid logit before the model score is added, and “all cells zero” is a hard error.
  Rationale: zero mass should mean the proposal assigns impossible support to that cell. Using `-inf` preserves that semantics exactly and avoids inventing an arbitrary smoothing rule. If every in-window cell has zero mass, the benchmark window is incompatible with the proposal and the run must fail loudly instead of silently fabricating probability mass.
  Date/Author: 2026-04-07 / Codex

- Decision: empirical fallback is part of scope and must be deterministic, but optional deepSSF cache reuse is not part of scope for this refactor.
  Rationale: replacing sampled final prediction requires a deterministic answer for both the parametric and fallback proposal laws actually encoded in [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py). By contrast, reusing deepSSF WebDataset caches would add speculative plumbing and code surface that the user did not ask for. The default raster path for this plan is therefore `ChipDataset(mode="steps")`; cache reuse should be omitted from the implementation unless the user asks for it separately.
  Date/Author: 2026-04-07 / Codex

- Decision: the grid prediction path must preserve the existing bounded-batch behavior by scaling held-out step batches to grid width, and it should avoid grid-cell chunking unless validation actually shows OOM or equivalent memory failure.
  Rationale: the current sampled predictor in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py) already scales prediction batches by support width. That is the simplest replacement rule and should be tried first. Grid-cell chunking adds code and complexity, so it should only be implemented if scaled step batches still fail in practice.
  Date/Author: 2026-04-07 / Codex

- Decision: `baseline_randomwalk` must stop deriving start bearing from the appended true candidate in point-store shared rows and instead use the same reconstructed held-out step metadata as the grid iSSF path.
  Rationale: once sampled final prediction is removed, the baseline can no longer depend on pseudo-candidate payload structure to recover step context. A single shared source of truth for `point_id_start`, `start_x_m`, `start_y_m`, and `start_bearing` avoids duplicating context reconstruction logic in the baseline executor.
  Date/Author: 2026-04-07 / Codex

- Decision: do not introduce a second cache format, a second raster-loading path, or a second held-out step reconstruction path for SSF grid prediction.
  Rationale: the repository already has one raw raster path in [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py), one optional deepSSF cache path in [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py), and one evaluator support-reconstruction path in [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py). Adding parallel versions would increase code size and create drift.
  Date/Author: 2026-04-07 / Codex

- Decision: Milestone 2 must be consolidation-first. Any “new helper” should be created only by extracting existing shared logic from the current evaluator, chip dataset, iSSF dataset, or deepSSF executor, and only when that extraction deletes duplicate code from at least two call sites.
  Rationale: the user explicitly wants code size reduction. The only truly new model-side logic required by this refactor is deterministic per-cell movement-prior evaluation on the benchmark grid. Geometry, movement-feature normalization, and grid artifact packing already exist and should be reused.
  Date/Author: 2026-04-07 / Codex

- Decision: remove `energy_score_num_candidates` from the active benchmark config and delete the sampled final-prediction path once the grid path is validated.
  Rationale: the user explicitly wants the sampled final-energy path replaced, not kept as an alternate mode. Keeping both would preserve code size and semantic drift.
  Date/Author: 2026-04-07 / Codex

- Decision: do not change SSF training semantics in this plan.
  Rationale: the request is specifically about final energy-score prediction support. iSSF training still depends on point-store candidate sets and cross-entropy over appended truth candidates. Changing that would be a separate modeling plan.
  Date/Author: 2026-04-07 / Codex

- Decision: introduce [movebench/bench/executors/ssf_grid.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/ssf_grid.py) as the single shared SSF grid helper, and keep its scope narrow: `selected_splits` parsing, train-step support reuse, a `ChipDataset(mode="steps")` wrapper, deterministic movement-prior evaluation, bounded batching, and grid artifact assembly.
  Rationale: both `issf_*` and `baseline_randomwalk` needed the same new logic, while deepSSF already had the desired grid artifact contract. Extracting just that shared surface deleted duplicate prediction-time code without creating a broader “grid framework.”
  Date/Author: 2026-04-07 / Codex

- Decision: extend [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) with an optional `selected_splits` parameter for `mode="steps"` instead of inventing a second held-out step reconstruction path.
  Rationale: the evaluator already treats point-store `selected_splits` as the source of truth for held-out step ordering. Making `ChipDataset` optionally follow that same rule kept the grid raster path aligned with evaluation while preserving the existing deepSSF behavior when `selected_splits` is omitted.
  Date/Author: 2026-04-07 / Codex

- Decision: accept scaled held-out step batching as the shipped memory bound and do not add grid-cell chunking in this implementation.
  Rationale: the targeted suites passed with the scaled batch rule, no validation step observed an OOM or equivalent memory failure, and adding a second fallback would have increased code size beyond the user’s request.
  Date/Author: 2026-04-07 / Codex

- Decision: use a real regenerated `baseline_randomwalk` artifact as the on-disk behavioral proof for Milestone 5.
  Rationale: the ExecPlan explicitly allows “one real completed iSSF or random-walk run.” The baseline path exercises the same shared movement-prior/grid-artifact implementation, completed quickly with local study data, and produced the required `prediction_kind == "grid"` proof without waiting for a much slower full-covariate iSSF prediction sweep.
  Date/Author: 2026-04-07 / Codex

## Outcomes & Retrospective

The refactor landed as planned. [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py) no longer contains the sampled final-prediction path, [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py) no longer exposes `energy_score_num_candidates`, and both `issf_*` plus `baseline_randomwalk` now route through one shared grid helper in [movebench/bench/executors/ssf_grid.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/ssf_grid.py). The helper keeps the implementation small by reusing `ChipDataset(mode="steps")`, the existing iSSF movement-feature normalization, and the existing grid artifact contract instead of adding a second raster or evaluation path.

Validation is strong but not perfectly repo-wide. The targeted benchmark suites required by this plan passed cleanly, and a real regenerated [baseline_randomwalk](/data/vision/beery/scratch/justin/movebench/results/bench/runs/validation__study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__gridcheck/predictions/test_individual.pt) artifact proved the new contract on disk with `prediction_kind == "grid"`, `scores.shape == (367, 101, 101)`, a positive `pixel_size_m`, and no `coords_m` or `probs` keys. The repository-level `python -m unittest -v` run still has unrelated failures outside this change, so the honest end state is: SSF grid prediction is implemented and validated, deepSSF stayed unchanged, but the entire repository was not green before/after due independent issues.

## Context and Orientation

The active benchmark runtime lives under [movebench/bench](/data/vision/beery/scratch/justin/movebench/movebench/bench). The exact energy evaluator in [movebench/bench/evaluation/energy.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/energy.py) already supports three prediction contracts: deterministic points, weighted discrete supports, and dense grids. The evaluator path for grids is already exact and already used by deepSSF. That means this plan does not need a new energy-score formula; it needs SSF-style methods to emit the existing grid contract.

The sampled final-prediction path the user wants removed is concentrated in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py). Today that file reconstructs held-out step rows, replays the fitted movement sampler, extracts environment values only at the sampled candidate locations, scores those sampled points with the trained iSSF model, and writes a weighted discrete support artifact. This is the source of the Monte Carlo variance and the extra code surface around `energy_score_num_candidates`.

The non-trainable random-walk baseline in [movebench/bench/executors/baselines.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/baselines.py) is currently even simpler: it reads the already-extracted pseudo-candidate coordinates from the point store and assigns them uniform probability. That is also a sampled-support final-prediction path and must move to the shared grid support.

deepSSF is the reference implementation for the desired support. Its executor in [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py) already emits `prediction_kind="grid"` with `scores` over the deterministic square step grid. The deepSSF dataset wrapper in [movebench/datasets/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf.py) shows that this support comes from [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py), specifically `ChipDataset(mode="steps")`. That dataset reconstructs valid held-out step starts from the same train-split support logic used elsewhere in the benchmark, then loads a deterministic `output_hw x output_hw` raster stack centered on each step start.

Some terms of art used below are defined here once so a novice can follow the plan. “Parametric” means the train split’s observed step lengths and turn angles are summarized by fitted continuous distributions, currently gamma for length and von Mises for turn angle. “Empirical fallback” means that if either fit fails, the runtime falls back to resampling directly from the observed train values instead. “EPSG:3857” is the projected coordinate system already used throughout the repo for meter-based x/y offsets. “Quadrature” means deterministic numerical integration by evaluating a density at a fixed set of sample points and averaging. “Logit” means an unnormalized score before softmax converts scores into probabilities.

The iSSF models themselves live in [movebench/models/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/ssf.py). They already score arbitrary candidate sets using environmental features, movement features, and optional interaction terms. For a grid predictor, the model does not need architectural changes: it simply needs to be applied to `101 x 101 = 10201` grid cells instead of sampled pseudo-candidates. The new piece is the deterministic per-cell movement prior `q(cell)`, because the current sampled runtime implicitly includes that prior by sampling from it before the softmax.

In this plan, “movement prior” means the discrete probability mass assigned to each benchmark grid cell before the iSSF model score is applied. For a grid predictor the final probability should be proportional to `q(cell) * exp(model_score(cell))`, which is equivalent to a grid logit `log q(cell) + model_score(cell)`. This keeps grid iSSF prediction faithful to the current sampled-support semantics while making the base measure explicit and deterministic.

The computation rule for `q(cell)` is fixed by this plan. First compute raw proposal mass on the finite benchmark window, not yet normalized. For the parametric case, raw cell mass is the deterministic integral of the continuous proposal over the square pixel footprint in EPSG:3857 using a fixed quadrature rule shared by all runs, with the continuous proposal induced by the same independent step-length and turn-angle law that [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) samples today. The required endpoint density formula in relative meter coordinates is:

    dx = x_cell - x_start
    dy = y_cell - y_start
    r = sqrt(dx^2 + dy^2)
    phi = atan2(dx, dy)                      # north=0, east=+pi/2
    t = wrap_to_pi(phi - start_bearing_rad)
    p_xy(dx, dy) = f_length(r) * f_turn(t) / r

where `f_length` is the gamma pdf fitted on positive step lengths, `f_turn` is the von Mises pdf fitted on turn angles, and the `1 / r` factor is the Jacobian term from transforming independent `(length, turn)` samples into an area density on the x/y grid. This factor is mandatory. At `r == 0`, treat the continuous density as numerically undefined and avoid evaluating it directly by using an even-sized fixed midpoint subgrid inside each pixel, so no quadrature point lands exactly at the origin. A simple fixed `2 x 2` or `4 x 4` midpoint subgrid per cell is acceptable, but the subgrid resolution must be constant and pinned in tests so the result is reproducible. For the empirical-fallback case, raw cell mass is the exact discrete mass of the current fallback proposal law: compress the stored training lengths and turns into unique values with multiplicities, assign each `(length, turn)` pair the product of its marginal frequencies, map the implied endpoint into the benchmark grid, and accumulate that weight into the corresponding cell. This accumulation must be deterministic and chunked; it must never call the random sampler.

After raw in-window cell masses are accumulated, separately record the total raw mass that falls outside the benchmark window as `off_grid_mass` for diagnostics and tests. Then renormalize only the in-window mass so that `sum_cell q(cell) = 1` on the benchmark grid. Cells with zero in-window mass must contribute `-inf` to `log q(cell)` before the model score is added. If every in-window cell has zero mass, raise an error rather than smoothing or clipping. `off_grid_mass` is an internal diagnostic quantity only. It must not be persisted into `predictions/*.pt`, benchmark summaries, MLflow payloads, or any other user-facing artifact unless the user explicitly asks for that extra output later.

The only mathematically subtle edge case is when the stored “parametric” sampler fell back to empirical resampling because the gamma or von Mises fit failed during training feature extraction. This plan therefore treats empirical fallback as an implementation requirement, not as a hypothetical escape hatch. It does not permit shipping a mixed “sampled for fallback, grid for parametric” runtime because the user explicitly asked to replace the sampled final-prediction computation.

## Plan of Work

Milestone 1 is test-first and semantics-first. Update [tests/bench/test_issf_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_issf_runtime.py) so it no longer expects `energy_score_num_candidates` in the iSSF method config and instead requires grid prediction artifacts. A passing test after the change must prove that `issf_linear` and `issf_mlp` write `prediction_kind == "grid"` and `scores.shape[-2:] == (101, 101)` for held-out splits. Replace the current helper-level sampled-payload tests with grid-path tests that preserve the same behavioral coverage: payload shape and point ordering, bounded prediction batching, prediction-worker propagation, and resume-time regeneration of only invalid prediction splits. Update [tests/bench/test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py) so the final evaluator and run summaries continue to work when those methods emit grid artifacts, and use the same file to pin that deepSSF keeps its existing grid artifact contract. Update [tests/bench/test_experiment_config.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_experiment_config.py) if needed to assert that `energy_score_num_candidates` is gone from active method config. Add or extend a focused runtime test for `baseline_randomwalk` so it now emits a grid artifact derived from the deterministic movement prior, not sampled pseudo-candidates, and so it sources step context from reconstructed held-out step metadata rather than point-store candidate rows. Do not add a parallel deepSSF runtime test module unless an existing benchmark test file genuinely cannot express the needed assertion.

Milestone 1 must also lock the exact grid-prior semantics in a minimal test surface. Add a small synthetic test case, likely in [tests/bench/test_energy.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_energy.py) or a new `tests/bench/test_ssf_grid_prediction.py`, that constructs a tiny square grid with known movement prior scores and known iSSF model scores, then asserts that the emitted grid logits equal `log q + f` and that the evaluator’s `softmax` therefore yields the expected probability ranking. Add one parametric test that checks off-grid mass is discarded then renormalized, and one empirical-fallback test that checks zero-mass cells become `-inf` while nonzero cells remain finite. This is the contract that replaces sampled pseudo-candidate normalization.

Milestone 2 is the shared grid-prediction infrastructure, but it must be consolidation-first rather than greenfield. Before adding any new helper, inventory the existing reusable implementations: held-out step reconstruction and grid coordinates in [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py), deterministic step-grid geometry in [movebench/datasets/chips.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py), movement-feature normalization in [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), and grid-artifact packing in [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py). Extract the minimum common pieces from those files or move them into one tiny shared helper only when doing so deletes duplicate logic from at least two call sites. Do not create a new framework for SSF grid prediction.

The resulting shared layer, if one is still needed after extraction, should be limited to glue. It must rebuild held-out step rows in the same order as the evaluator, expose the deterministic benchmark grid coordinates in meters, and produce one flat grid candidate tensor with `P = 10201` cells for iSSF model scoring. Grid coordinate and pixel-size math should come from the existing evaluator and chip-dataset implementations, not from a fresh duplicate formula. For the environment stack, default to [ChipDataset(mode="steps")](/data/vision/beery/scratch/justin/movebench/movebench/datasets/chips.py) rather than a new raster extractor. deepSSF WebDataset cache reuse is explicitly out of scope for this plan and should not be added opportunistically. The helper must also reuse the existing iSSF movement feature semantics from [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py) for step length, log step length, cosine turn angle, and train-split normalization rather than reimplementing them. The genuinely new core logic in this milestone is deterministic per-cell movement-prior evaluation and its bounded execution strategy.

Milestone 2 must also pin the bounded-memory rule that replaces the sampled path’s support-width scaling in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py). The default implementation should stop at scaled held-out step batches and should not introduce grid-cell chunking unless validation reveals an actual OOM or equivalent memory failure. The default step batch size for grid prediction should be:

    step_batch_size = max(1, floor(batch_size * train_points_per_step / grid_points_per_step))

where `train_points_per_step = train_manifest.sampler["num_candidates"] + 1` and `grid_points_per_step = output_hw[0] * output_hw[1]`. Pin this scaling rule and `num_workers` propagation in tests as the default bounded-memory behavior. If the scaled-batch implementation passes targeted tests and one real validation run without OOM, stop there and do not add grid-cell chunking. Only if validation actually shows OOM or equivalent memory failure with `step_batch_size == 1` should the implementation add a second-level fallback that chunks grid cells within that step batch and concatenates scores along the flattened grid dimension before reshaping back to `(H, W)`. If that contingency is needed, update this plan with the concrete trigger that was observed and add a targeted regression test for the chunked path before landing it.

Milestone 3 is the executor migration. In [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py), replace `_parametric_prediction_payload(...)` with a new grid path, for example `_grid_prediction_payload(...)`, that scores all grid cells for each held-out step instead of replaying sampled pseudo-candidates. The iSSF models should continue to produce selection scores from the same trained weights; the only prediction change is the support. For each step, the executor must compute raw in-window proposal mass, accumulate off-grid mass for diagnostics, renormalize the in-window proposal, convert zero-mass cells to `-inf`, add the model score, and write the result as a grid artifact. Delete the sampled-support config knob `energy_score_num_candidates` from [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py) and remove the prediction-only helper code that exists solely to service sampled final evaluation. Keep training resume, checkpoint reuse, and prediction-split reuse behavior intact.

In the same milestone, update [movebench/bench/executors/baselines.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/baselines.py) so `baseline_randomwalk` also emits grid scores. For this method the model score term is zero, so the grid artifact should be derived entirely from `log_q_hw`. The source of truth for step context must be the same reconstructed held-out step rows already used by the grid iSSF path, not `candidate_lon`, `candidate_lat`, or the appended truth candidate stored in point-store shared rows. `baseline_nomove` and `baseline_constvel` remain deterministic point predictions and do not need to be converted. This milestone is complete only when no active SSF-style method still uses extracted pseudo-candidate coordinates for final energy-score prediction.

Milestone 4 is code deletion and stabilization. Once the grid path is green, remove or simplify the old sampled final-prediction code in [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py) and any config or tests that only existed for `energy_score_num_candidates`. Keep the discrete evaluator in [movebench/bench/evaluation/energy.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/energy.py) because MoveFormer and historical artifacts still need it, but no active SSF-style executor should write new discrete-support artifacts after this milestone. deepSSF should stay mathematically unchanged; only share tiny helper code if doing so clearly deletes duplication without broadening the change.

Milestone 5 is validation. Run the targeted unit suites first, then the repository-level `python -m unittest -v` command required by the repository instructions, and record the exact result. If the broad suite fails for unrelated reasons, record the exact failures here and do not treat that run as green. Then regenerate one real completed iSSF or random-walk run on a study that has local data available, inspect the resulting `predictions/<split>.pt`, and record a short transcript proving that the artifact now contains `prediction_kind == "grid"`, `scores.shape[-2:] == (101, 101)`, and no `coords_m` or `probs` keys. This real-run check matters because the user’s requested outcome is behavioral: same support as deepSSF and no sampled final evaluation.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Use the MoveBench micromamba interpreter for all Python and unittest commands:

    export MB_PY=/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python
    export LD_LIBRARY_PATH=/data/vision/beery/scratch/justin/micromamba/envs/movebench/lib:$LD_LIBRARY_PATH

Update the runtime and evaluator tests first, then run the targeted benchmark suite:

    $MB_PY -m unittest -v \
      tests.bench.test_issf_runtime \
      tests.bench.test_ssf_grid_prediction \
      tests.bench.test_runner \
      tests.bench.test_energy \
      tests.bench.test_experiment_config

Observed post-change result:

    Ran 29 tests in 6.000s
    OK

After the code changes land, run the required broader suite:

    $MB_PY -m unittest -v

If that broad command fails for unrelated reasons, capture the exact failure names and the first relevant error lines in the `Progress` or `Surprises & Discoveries` section. Do not silently skip it.

Before Milestone 5’s real-run validation, run a preflight check to confirm that the reused run directory and the corresponding local study data actually exist in the current environment. One acceptable check is:

    $MB_PY - <<'PY'
    from pathlib import Path
    import movebench.paths as paths
    run_dir = Path("results/bench/runs/study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=issf_linear__seed=0__k=e8546c90cb6d2674")
    study_dir = paths.study_dir("Garthe_ae7b0b8c_raw_merged")
    print(run_dir.exists())
    print((study_dir / "trajectories.csv").exists())
    PY

The expected preflight transcript is:

    True
    True

If either line is `False`, do not proceed with artifact inspection yet. Choose a different completed local run or first run one small local SSF benchmark that has mounted study data.

For the real artifact inspection, use a completed SSF-style run whose study data passed that preflight check. The current tree already contains completed `Garthe_ae7b0b8c_raw_merged` `1d` iSSF runs under `results/bench/runs/`, so one acceptable post-change proof is:

    $MB_PY - <<'PY'
    import json
    from pathlib import Path
    import torch
    run_dir = Path("results/bench/runs/study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=issf_linear__seed=0__k=e8546c90cb6d2674")
    pred = torch.load(run_dir / "predictions" / "test_individual.pt", map_location="cpu")
    print(pred["prediction_kind"])
    print(tuple(pred["scores"].shape))
    print(pred["pixel_size_m"])
    print("coords_m" in pred, "probs" in pred)
    PY

The expected post-change transcript should look like:

    grid
    (<n_steps>, 101, 101)
    <positive float pixel size>
    False False

The existing `Garthe` `1d` iSSF run was available locally (`True`, `True`), but a faster real validation path was to regenerate a completed `baseline_randomwalk` run that exercises the same shared grid helper. The validation run used:

    $MB_PY - <<'PY'
    import json
    import shutil
    from pathlib import Path
    from movebench.bench import run as bench_run
    from movebench.bench.executors import baselines
    source_run = Path("results/bench/runs/study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__k=98b82f034dedc764")
    validation_run = Path("results/bench/runs/validation__study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__gridcheck")
    if validation_run.exists():
        shutil.rmtree(validation_run)
    validation_run.mkdir(parents=True)
    run_payload = json.loads((source_run / "resolved_run.json").read_text(encoding="utf-8"))
    run_payload["run_key"] = validation_run.name
    run_payload["evaluation_split_keys"] = ["test_individual"]
    resolved_artifacts = bench_run._resolve_artifacts_for_run(run_payload)
    print(resolved_artifacts.get("status"))
    print(baselines.run_randomwalk(
        run_payload=run_payload,
        resolved_artifacts=resolved_artifacts,
        run_dir=validation_run,
        tracker=None,
    ))
    PY

with observed transcript:

    resolved
    {'status': 'completed', 'train_steps': 0, 'early_stop_split_key': 'test_temporal'}

The corresponding artifact inspection was:

    $MB_PY - <<'PY'
    from pathlib import Path
    import torch
    run_dir = Path("results/bench/runs/validation__study=Garthe_ae7b0b8c_raw_merged__ts=1d__cov=all__method=baseline_randomwalk__seed=0__gridcheck")
    pred = torch.load(run_dir / "predictions" / "test_individual.pt", map_location="cpu")
    print(pred["prediction_kind"])
    print(tuple(pred["scores"].shape))
    print(pred["pixel_size_m"])
    print("coords_m" in pred, "probs" in pred)
    PY

with observed transcript:

    grid
    (367, 101, 101)
    566.6628807268355
    False False

## Validation and Acceptance

Acceptance is behavioral.

First, `issf_linear`, `issf_mlp`, and `baseline_randomwalk` must now emit grid prediction artifacts. A passing targeted test must prove that those artifacts use `prediction_kind == "grid"` and carry `scores` over the full `101x101` support. A corresponding failing pre-change test should show that the old executor still wrote discrete sampled supports.

Second, the active benchmark config must no longer expose `energy_score_num_candidates` for the iSSF methods. A targeted config/runtime test must fail before the change and pass after the sampled-support knob and its executor plumbing are removed.

Third, the new grid-prior contract must be explicit in tests. At least one focused test must prove that an iSSF-style grid predictor combines movement prior and model score as `log q + f`, not as raw model score alone. The same test should show that `baseline_randomwalk` is the degenerate case where only the movement prior contributes.

Fourth, the grid path must preserve the non-artifact runtime behavior that the sampled-path helper tests currently pin. The targeted suite must still cover bounded prediction batching, prediction-worker propagation, and regeneration of only invalid prediction splits after resume, but now against the grid predictor rather than the sampled one. Grid-cell chunking is not required for acceptance unless implementation/validation actually encountered OOM and the plan was updated to add that fallback.

Fifth, deepSSF must remain semantically unchanged. The existing benchmark tests that already cover grid prediction handling, especially [tests/bench/test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py), should still pass, and deepSSF’s prediction artifact contract should remain `prediction_kind == "grid"` with the same `scores` and `pixel_size_m` shape semantics as before. Do not add a second deepSSF-specific runtime suite unless existing tests cannot express the invariant.

Sixth, one real regenerated SSF-style run must produce a prediction artifact on disk whose payload confirms the new contract. This is the user-visible proof that sampled final evaluation is gone.

## Idempotence and Recovery

These changes are source-only and safe to rerun. Re-running the targeted tests or the broad `python -m unittest -v` command should not mutate benchmark state. Regenerating a run after the executor switch is also safe because the run directory already has prediction-artifact overwrite semantics.

The risky part of this plan is deleting the sampled final-prediction path too early. Do not remove that code until the grid-prior semantics are nailed down in tests and at least one real artifact inspection passes. If implementation stalls during the grid-prior work, keep the old sampled final-prediction path on the branch until the deterministic grid path is fully validated; do not land a half-migrated runtime that mixes sampled and grid final prediction for different SSF methods. If the broad validation suite fails in unrelated parts of the repository, record those failures precisely and keep the plan honest about what was and was not verified.

## Artifacts and Notes

The desired final artifact contract for SSF-style grid predictors is:

    {
        "prediction_kind": "grid",
        "point_id_start": torch.Tensor,  # shape (N,)
        "scores": torch.Tensor,          # shape (N, 101, 101)
        "pixel_size_m": float,
    }

where `softmax(scores.reshape(N, -1), dim=1)` defines the exact probability distribution used by the existing grid energy-score evaluator.

## Interfaces and Dependencies

Keep the existing evaluator contract in [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py) and [movebench/bench/evaluation/energy.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/energy.py). The change in this plan is on the prediction side, not the scoring formula side.

Do not assume a new helper module is required. If a shared helper module such as [movebench/bench/grid_prediction.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/grid_prediction.py) is introduced, it must be created by extracting current logic from existing files rather than by reimplementing it. Prefer keeping each concern in its natural owner when only one caller needs it. By the end of the work there must be exactly one authoritative implementation for each of the following responsibilities:

    1. held-out step reconstruction keyed by `point_id_start`, including `selected_splits`
    2. benchmark grid coordinates / pixel-size math
    3. iSSF movement-feature tensor construction and normalization
    4. packing `prediction_kind="grid"` artifacts with `scores` and `pixel_size_m`

Natural ownership should be preserved unless extraction deletes real duplication:

    - [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py) should continue to own held-out support reconstruction and benchmark-grid coordinate math.
    - [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py) should continue to own movement-feature tensor construction and normalization.
    - [movebench/bench/executors/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/deepssf.py) or a helper extracted directly from it should own packing `prediction_kind="grid"` artifacts.
    - The only genuinely new reusable logic expected from this refactor is deterministic `step_grid_log_movement_prior(...)` for the benchmark grid. If that logic is shared by `issf_*` and `baseline_randomwalk`, it is a good candidate for a tiny shared helper.

If signatures are extracted, they should replace the original duplicated logic rather than wrap it. For example, acceptable extracted building blocks are:

    def step_grid_log_movement_prior(
        *,
        sampler_manifest: dict[str, object],
        x_coords_m: torch.Tensor,
        y_coords_m: torch.Tensor,
        start_bearing_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (log_q_hw, off_grid_mass) for one step or one step batch.
        log_q_hw is normalized on the benchmark grid and uses -inf where mass is zero.
        off_grid_mass is the raw proposal mass discarded outside the 101x101 window.
        It is for internal assertions/tests only and must not be written into
        prediction artifacts.
        """

    def grid_scores_artifact(
        *,
        point_id_start: torch.Tensor,
        scores_hw: torch.Tensor,
        pixel_size_m: float,
    ) -> dict[str, object]:
        """
        Pack the standard prediction_kind='grid' artifact shared by deepSSF and
        the refactored SSF-style methods.
        """

In [movebench/bench/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/issf.py), the new prediction helper should have a stable shape contract equivalent to:

    def _grid_prediction_payload(
        *,
        model: torch.nn.Module,
        ...,
    ) -> dict[str, object]:
        ...

Its returned payload must satisfy the grid artifact contract above, and its score semantics must be:

    scores_hw = log_movement_prior_hw + model_score_hw

In [movebench/bench/executors/baselines.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/executors/baselines.py), the random-walk baseline must emit the same grid artifact contract, with:

    scores_hw = log_movement_prior_hw

In [movebench/bench/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/config.py), the iSSF method entries must no longer contain `energy_score_num_candidates`.

Plan creation note (2026-04-07 19:41Z / Codex): created this ExecPlan after reviewing the active SSF executors, the exact grid evaluator, `ChipDataset(mode="steps")`, the deepSSF cache/data path, the earlier sampled-support iSSF plan, and the completed local investigation showing material Monte Carlo variance in sampled final iSSF supports.

Plan update note (2026-04-07 20:00Z / Codex): revised the plan to make the probability semantics explicit for implementers. The SSF grid path is now described in terms of discrete per-cell movement mass plus model score, and the plan now records that iSSF scores are not a separable habitat-only raster because the current models already mix environment and movement features.

Plan update note (2026-04-07 20:43Z / Codex): revised the plan to eliminate duplication risk in Milestone 2. The plan now explicitly requires extraction and reuse of existing evaluator, chip-dataset, iSSF-dataset, and deepSSF-executor logic instead of adding parallel geometry, cache, movement-feature, or grid-artifact implementations, and it corrects the stale reference to a nonexistent `tests/bench/test_deepssf_runtime.py` file.

Plan update note (2026-04-07 20:43Z / Codex): revised the plan again to specify the highest-risk runtime semantics that were still ambiguous. The plan now fixes the definition of `q(cell)`, including off-grid mass, renormalization, and zero-mass handling; removes speculative deepSSF cache reuse from scope; requires replacement tests for bounded batching, worker propagation, and prediction regeneration; makes random-walk step context share the same held-out step metadata as iSSF; and adds a preflight prerequisite check before the real-run validation step.

Plan update note (2026-04-07 21:10Z / Codex): revised the plan to clear the remaining novice-risk ambiguity in the parametric proposal math. The plan now states the endpoint density formula on the x/y grid explicitly, including the required `1 / r` Jacobian term, clarifies that `off_grid_mass` is internal-only and must not be persisted into prediction artifacts, and defines the remaining terms of art in plain language at first use.

Plan update note (2026-04-07 21:26Z / Codex): revised the bounded-memory guidance so the default implementation remains simple. The plan now treats scaled held-out step batching as the intended solution and makes grid-cell chunking a contingency only if targeted or real-run validation actually hits OOM or an equivalent memory failure.

Plan update note (2026-04-07 22:15Z / Codex): updated the plan after implementation and validation. The plan now records the landed shared helper/module, the `ChipDataset(selected_splits=...)` alignment fix, the exact targeted and repo-wide unittest results, the unrelated full-suite failures that remain outside scope, and the real regenerated `baseline_randomwalk` grid artifact transcript used as the on-disk behavioral proof.
