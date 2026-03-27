# Close the Remaining Milestone 1-3 Acceptance Gaps in `bench_v2`

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_overview_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_overview_execplan.md), [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md), [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md), and [execplans/benchmark_rewrite_evaluation_reporting_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_evaluation_reporting_execplan.md). The `bench_v2` planner, runner, executors, evaluator, and report generator already exist. This follow-up closes the remaining gaps that still keep milestones 1-3 from being accepted at the overview level. Legacy cutover remains out of scope.

## Purpose / Big Picture

After this change, the first three benchmark-rewrite milestones will be complete at the user-visible level, not just “implemented somewhere in the tree.” `movebench bench plan ...` and `movebench bench status --format text` will stay on a lightweight import path that does not drag in raster IO, projection libraries, or plotting code. The default benchmark experiment surface will match the intended scope by including the baseline methods in `main_results` and by providing a small checked-in `smoke_one_study.py` experiment module. Most importantly, `movebench bench run ...` will no longer stop after writing predictions; it will also compute exact held-out energy metrics, write `final_metrics/` artifacts, and log those metrics to the same MLflow run before marking the run completed.

The visible proof is a short sequence of commands that now all work under one coherent runtime:

    movebench bench plan experiments/v2/main_results.py
    movebench bench run experiments/v2/smoke_one_study.py --studies <study_id> --max-runs 3
    movebench bench report

After the second command finishes, the run directory for each completed run should already contain `final_metrics/<split>/summary.json`, and MLflow should already contain `energy_score.*` metrics for the same run attempt before `bench report` is invoked.

## Progress

- [x] (2026-03-27 19:04Z) Reviewed the implemented `bench_v2` planner, runtime, evaluator, and report code against the overview acceptance criteria and isolated five remaining gaps: heavy control-plane imports, missing `existing` count in `bench plan`, missing `smoke_one_study.py`, missing baseline coverage in `main_results.py`, and final evaluation still being deferred from `bench run` to `bench report`.
- [x] (2026-03-27 19:04Z) Decided to keep `Study.list_available()` as the public discovery API and solve the lightweight-control-plane problem by making `movebench.study` and `movebench.features.points` import-safe at module import time rather than replacing the API.
- [x] (2026-03-27 19:04Z) Authored this follow-up ExecPlan to close the remaining milestone 1-3 acceptance gaps without reopening the milestone 4 legacy cutover.
- [x] (2026-03-27) Implemented milestone 1 acceptance fixes: `bench_v2.run` now imports report/evaluation at subsystem boundaries, `Study.list_available()` remains the public discovery API while `movebench.study`, `movebench.features.points`, and `movebench.datasets.__init__` no longer drag raster-backed code into light paths, and `bench plan` now prints planned, existing, blocked, and runnable counts.
- [x] (2026-03-27) Implemented milestone 2 acceptance fixes: `experiments/v2/main_results.py` now includes the three baseline methods and `experiments/v2/smoke_one_study.py` exists as the small one-study runtime surface.
- [x] (2026-03-27) Implemented milestone 3 acceptance fixes: `_execute_run(...)` now performs exact final evaluation inline, stale `final_metrics/` are cleared on retries, evaluator MLflow metrics can target the active run id directly, and report refresh remains the repair-only path.
- [x] (2026-03-27) Ran the targeted milestone suites, the wider 80-test `bench_v2` regression surface, `movebench bench plan experiments/v2/main_results.py`, and `movebench bench status --format text`. The real `bench run experiments/v2/smoke_one_study.py --studies <study_id> --max-runs 3` smoke command was not run in this environment because status reported `Fully runnable pairs: 0` and `deepSSF-cache-ready pairs: 0`.

## Surprises & Discoveries

- Observation: keeping `Study.list_available()` is compatible with a lightweight control plane, but only if `movebench.study` itself becomes import-safe.
  Evidence: [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py) currently imports `ChipDataset`, `LayerIO`, the download module, and point-feature extraction helpers at module import time even though `list_available()` only needs filesystem paths.

- Observation: `movebench.features.points` is part of the control-plane import problem, not just `movebench.study`.
  Evidence: [movebench/bench_v2/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/config.py) and [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) both import helper symbols from [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), and that file currently imports `LayerIO`, `Raster`, and `pyproj.Transformer` at module import time even when only path/hash helpers are used.

- Observation: the report layer already uses lazy `matplotlib` imports at the right boundary, so the remaining report-related import problem is the top-level import in `movebench/bench_v2/run.py`.
  Evidence: [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) imports `matplotlib` only inside `write_main_results_plots(...)` and `write_covariate_comparison_plots(...)`, while [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) imports the report module at file import time.

- Observation: inline final evaluation in `bench run` will make the current evaluation-runner dependency surface part of the normal execution path.
  Evidence: [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) currently calls `evaluate_run(...)` only as a backfill step, but [movebench/bench_v2/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/evaluation/runner.py) still imports `rasterio.transform` even though it only needs square-grid index math.

- Observation: the missing smoke experiment is a real product gap, not just a documentation bug.
  Evidence: the overview and method-runtime plans both reference `experiments/v2/smoke_one_study.py`, but [experiments/v2](/data/vision/beery/scratch/justin/movebench/experiments/v2) currently contains only `main_results.py` and `covariate_comparison.py`.

- Observation: import-safety needed one extra package boundary beyond the original audit.
  Evidence: the wider `bench_v2` regression run failed until [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) stopped importing `movebench.layers.derived` at module import time and [movebench/datasets/__init__.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/__init__.py) stopped eagerly importing `ChipDataset`, which otherwise reintroduced `rasterio` through `movebench.datasets.chips`.

## Decision Log

- Decision: preserve `Study.list_available()` and make its module import-safe instead of introducing a second study-discovery API for `bench_v2`.
  Rationale: the public API is already appropriate. The actual problem is that `movebench.study` currently imports raster-backed and extraction-backed modules too early.
  Date/Author: 2026-03-27 / assistant

- Decision: keep lazy imports narrowly scoped to subsystem boundaries only.
  Rationale: the user explicitly wants to avoid import clutter. The right fix is to localize `rasterio`, `pyproj`, and plotting imports in the few methods and modules that truly need them, not to scatter local imports throughout ordinary logic.
  Date/Author: 2026-03-27 / assistant

- Decision: define `bench plan`’s new `existing` count as “planned `run_key`s that already have at least one MLflow run attempt.”
  Rationale: this matches the overview wording, stays simple to explain, and does not depend on interpreting latest status as success or failure.
  Date/Author: 2026-03-27 / assistant

- Decision: `experiments/v2/smoke_one_study.py` should remain small and trainable, not comprehensive.
  Rationale: its job is to prove one iSSF, one deepSSF, and one MoveFormer run can go end to end through the unified runtime for one study. Baseline coverage belongs in `main_results.py`, not in the smoke module.
  Date/Author: 2026-03-27 / assistant

- Decision: exact final evaluation belongs in the `bench run` lifecycle, and report-time refresh should remain only as a repair path.
  Rationale: the overview acceptance criteria say a completed run should already have final energy metrics. The report generator can still backfill missing artifacts for historical or interrupted runs, but that should not be the normal completion path.
  Date/Author: 2026-03-27 / assistant

- Decision: remove `rasterio.transform` from the evaluation runner instead of lazily importing it.
  Rationale: the runner only needs square-grid support indexing, which can be implemented with direct arithmetic. This avoids reintroducing the same binary-fragile dependency into the normal `bench run` path once evaluation is made inline.
  Date/Author: 2026-03-27 / assistant

## Outcomes & Retrospective

This follow-up is now implemented. The remaining milestone 1-3 acceptance gaps are closed in code and validated by the bench-specific test surface: the planner/status path stays off `rasterio`, `pyproj`, and plotting imports; `main_results` includes the baselines; `smoke_one_study.py` exists; and completed `bench run` attempts now write `final_metrics/` and log exact held-out metrics before the run is marked completed.

Validation in this environment included the milestone-targeted test suites plus the broader 80-test `bench_v2` regression command listed below, all passing in the movebench micromamba environment. The live CLI checks also passed: `movebench bench plan experiments/v2/main_results.py` reported `Planned runs: 4144`, `Existing runs: 0`, `Blocked runs: 3924`, and `Runnable runs: 220`, while `movebench bench status --format text` completed successfully. The real smoke run and report regeneration commands were not exercised against a concrete study here because local status output reported `Fully runnable pairs: 0` and `deepSSF-cache-ready pairs: 0`, so there was no study in this environment that satisfied the one-study smoke prerequisites.

The plan intentionally still does not include the milestone 4 legacy cutover. That cutover remains a separate follow-up after the accepted `bench_v2` stack is used as the primary path.

## Context and Orientation

The relevant files for this follow-up are:

- [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py), which owns CLI registration for `bench`, planning, status snapshots, run dispatch, and MLflow status tagging.
- [movebench/bench_v2/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/config.py), which defines the canonical method catalog, timescales, covariate sets, and experiment expansion logic.
- [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py), which should remain the report/query layer and should not be needed for `bench plan` or `bench status --format text`.
- [movebench/bench_v2/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/evaluation/runner.py), which currently writes `final_metrics/` artifacts and logs energy metrics back to MLflow after the fact.
- [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py), which should keep `Study.list_available()` as the public discovery API but must stop importing raster-backed helpers at module import time.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), which currently mixes lightweight path/hash helpers with extraction-time geospatial imports.
- [experiments/v2/main_results.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/main_results.py) and [experiments/v2/covariate_comparison.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/covariate_comparison.py), which define the checked-in experiment surface today.

Three terms matter in this plan.

An “import-safe control plane” means that `movebench bench plan ...` and `movebench bench status --format text` can import their modules without pulling in `rasterio`, `pyproj`, or plotting libraries. Pandas, requests, tqdm, and other lightweight frequently used Python dependencies are not the concern here.

An “existing run” means a planned `run_key` that already has at least one MLflow run attempt in the benchmark experiment, regardless of whether the latest attempt completed, failed, or was blocked.

An “inline final evaluation” means that once an executor finishes writing predictions for a run, the same `bench run` command computes the exact held-out metrics, writes `final_metrics/<split>/energy_by_step.csv.gz` plus `summary.json`, logs `energy_score.*` metrics to the same MLflow run attempt, and only then marks the run completed.

## Plan of Work

The work should proceed in four milestones.

Milestone 1 makes the control plane lightweight and complete. Start with tests that guard the `bench plan` and text-status import path against `rasterio`, `pyproj`, and plotting imports. Then patch [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) so it no longer imports the report module at file import time. Import the report layer only inside `_command_status(...)` when `--format html` is requested and inside `_command_report(...)`. Do not add a new helper module for this; two local imports at the CLI boundary are enough.

Still in milestone 1, keep `Study.list_available()` but make [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py) import-safe. Move `ChipDataset`, `LayerIO`, the raster-download module, point-feature extraction helpers, and any other raster-backed imports into the specific instance methods that actually use them: `layer_io(...)`, `download_rasters(...)`, `extract_point_features(...)`, `chip_dataset(...)`, and any other method that would otherwise force raster-backed imports at module import time. Leave `list_available()` itself unchanged. The file may continue to import pandas at top level.

Milestone 1 also needs [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) to become import-safe, because the planner imports its path and hash helpers directly. Keep the public helper functions exactly where they are. Move only the extraction-time geospatial imports behind the extraction-specific code paths. The intended end state is that importing `point_store_dir(...)`, `point_store_manifest_path(...)`, `point_store_train_stats_path(...)`, and `resolve_sampler_hash(...)` no longer imports `LayerIO`, `Raster`, or `pyproj`. A small private import helper for the extraction path is acceptable here because the file truly mixes two different dependency surfaces.

After the control-plane imports are clean, extend `_command_plan(...)` in [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) so it prints counts for planned, existing, and blocked runs. Keep the implementation simple: after planning, query `search_latest_runs_by_key(...)` for the planned runnable `run_key`s and count how many keys already have at least one run record. Planned runs should mean total concrete runs for the experiment, including planner-blocked entries. Existing runs should count only runnable `run_key`s with MLflow history. Blocked runs should continue to come from `experiment["blocked"]`.

Milestone 2 completes the checked-in experiment surface. Update [experiments/v2/main_results.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/main_results.py) so it includes the baseline methods in addition to the existing trainable methods. Keep the covariate set as `all` only. Then create [experiments/v2/smoke_one_study.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/smoke_one_study.py) as a small fixed experiment module that exercises one iSSF method, one deepSSF method, and one MoveFormer method on one timescale and one covariate set. Use `studies: "all"` so callers can narrow to one actual study with `--studies <study_id>` at runtime. Keep the smoke config intentionally small: `timescales = ["1d"]`, `covariate_sets = ["all"]`, and `methods = ["issf_linear", "deepssf", "moveformer"]`.

Milestone 3 moves exact final evaluation into the run lifecycle. Start with tests that currently fail because `bench run` stops too early: a successful run should already have `final_metrics/` artifacts and MLflow `energy_score.*` metrics before `bench report` is called, and a run whose evaluation step fails should be marked failed instead of completed. Then patch [movebench/bench_v2/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/evaluation/runner.py). Remove the `rasterio.transform` dependency by replacing `_support_transform(...)` and `_is_truth_in_support(...)` with direct square-grid arithmetic based on `output_hw`, `buffer_m`, and the implied pixel size. Keep the current artifact formats and metric names unchanged.

Still in milestone 3, extend `evaluate_run_dir(...)` so it can log metrics to a specific MLflow run attempt when the caller already knows the run id. The simplest interface is:

    evaluate_run_dir(run_dir: str | Path, *, mlflow_run_id: str | None = None) -> dict[str, object]

When `mlflow_run_id` is provided, log metrics directly to that run id. When it is omitted, preserve the current fallback behavior of finding the latest run by `run_key` so the report layer can still repair historical gaps. Do not create a second evaluator for the inline path.

Then patch `_execute_run(...)` in [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py). Capture the MLflow run returned by `start_benchmark_run(...)`. If the executor returns `status=completed`, call `evaluate_run_dir(run_dir, mlflow_run_id=<current run id>)` before writing the final training summary and before tagging the run completed. If evaluation succeeds, keep the executor’s summary status as completed. If evaluation raises, write a failed summary with the evaluation error, tag the run failed, and end the MLflow run as failed. Do not treat “predictions written but final metrics missing” as a completed run anymore.

Milestone 3 should leave [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) mostly intact. Keep `_refresh_final_metrics(...)` as a repair path that fills missing metrics for historical or interrupted runs and honors `--refresh-metrics`. The main behavioral shift is that missing final metrics should become the exception instead of the normal steady state after `bench run`.

Milestone 4 updates the living plans after the code lands. When the implementation is complete, update [execplans/benchmark_rewrite_overview_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_overview_execplan.md) and any directly affected child plans so they reflect that milestones 1-3 now satisfy the accepted user-visible behavior. Do not perform the milestone 4 legacy cutover in this follow-up.

## Concrete Steps

Work from the repository root and use the movebench micromamba environment:

    cd /data/vision/beery/scratch/justin/movebench

Milestone 1 starts with import-safety and plan-output tests. Add or update targeted tests in:

    tests/bench_v2/test_status.py
    tests/bench_v2/test_plan.py

The new or updated tests should prove two things:

    1. Importing `movebench.bench_v2.run` for planning or text status does not import `rasterio`, `pyproj`, or plotting modules.
    2. `bench plan` prints `Planned runs`, `Existing runs`, and `Blocked runs` with stable counts.

Run the milestone-1 tests first:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan

Milestone 2 then adds experiment-surface tests in:

    tests/bench_v2/test_plan.py

These tests should assert that:

    1. `experiments/v2/main_results.py` includes the baseline methods.
    2. `experiments/v2/smoke_one_study.py` exists, loads through `load_experiment(...)`, and expands to the expected method/timescale/covariate surface for one study when `--studies` is applied.

Run the milestone-2 tests:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan

Milestone 3 starts with failing runner and report-layer tests in:

    tests/bench_v2/test_runner.py
    tests/bench_v2/test_eval_runner.py
    tests/bench_v2/test_report_html.py

The new tests should prove:

    1. A successful `bench run` attempt writes `final_metrics/<split>/summary.json` before the run is marked completed.
    2. The same MLflow run attempt created by `_execute_run(...)` ends up with `energy_score.*`, `energy_score.accuracy.*`, `energy_score.calibration.*`, `n_steps_scored.*`, and `n_steps_excluded_out_of_support.*` metrics.
    3. If evaluation fails after predictions are written, the run is failed, not completed.
    4. `bench report` still refreshes only missing metrics by default and still respects `--refresh-metrics`.

Run the milestone-3 tests:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner tests.bench_v2.test_eval_runner tests.bench_v2.test_report_html

After each milestone, rerun the affected combined surface before moving on. When milestones 1-3 are all implemented, run the widest relevant `bench_v2` command set in this environment:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner tests.bench_v2.test_energy_score tests.bench_v2.test_eval_runner tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html

Then run the user-visible smoke commands:

    movebench bench plan experiments/v2/main_results.py
    movebench bench run experiments/v2/smoke_one_study.py --studies <study_id> --max-runs 3
    movebench bench report

Record the observed outputs and any environment limitations in the living sections before marking the plan complete.

## Validation and Acceptance

Acceptance for this follow-up is reached when all of the following behaviors are observable.

First, the control plane is import-safe in the intended sense. Importing the planner and running `movebench bench plan ...` or `movebench bench status --format text` must not require `rasterio`, `pyproj`, or plotting imports. Keeping `Study.list_available()` is acceptable, but it must no longer drag those dependencies into the control-plane path.

Second, `movebench bench plan experiments/v2/main_results.py` must print counts for planned, existing, and blocked runs. Re-running the same plan against the same MLflow state must report the same planned count and the same existing count unless new run attempts were actually created.

Third, the checked-in experiment surface is complete for milestones 1-2. `experiments/v2/main_results.py` must include the baseline methods, and `experiments/v2/smoke_one_study.py` must exist and be runnable with `--studies <study_id> --max-runs 3`.

Fourth, `movebench bench run experiments/v2/smoke_one_study.py --studies <study_id> --max-runs 3` must create three completed run attempts for one iSSF method, one deepSSF method, and one MoveFormer method, assuming the chosen study has the required existing artifacts on disk.

Fifth, each completed run from `bench run` must already contain final held-out metrics. That means `results/bench_v2/runs/<run_key>/final_metrics/<split>/summary.json` exists for every available held-out split, and the same MLflow run attempt already has `energy_score.*` plus the related component and count metrics before `bench report` runs.

Sixth, report-time metric refresh remains a repair path, not the normal completion path. `movebench bench report` should skip recomputation for runs whose final metric artifacts and MLflow metrics already exist, refresh only missing ones by default, and recompute all when `--refresh-metrics` is passed.

Seventh, overview-level milestone 1-3 acceptance can be updated truthfully after these changes land. The overview should no longer need to list any remaining gaps for lightweight control-plane imports, baseline coverage in `main_results`, missing `smoke_one_study.py`, missing `existing` counts in `bench plan`, or final evaluation still being deferred from `bench run` to `bench report`.

## Idempotence and Recovery

This follow-up must preserve the current planner and runner idempotence rules. Planning the same experiment twice must still write the same canonical run identities without duplicating manifests or MLflow state. Adding the `existing` count to `bench plan` must remain read-only with respect to MLflow; it should query for existing runs, not create placeholder runs.

The import-safety cleanup must not change behavior for the heavy paths themselves. Moving `rasterio`, `pyproj`, and extraction imports into the methods that need them is meant only to narrow when those dependencies are loaded, not to redesign extraction, dataset loading, or reporting behavior.

Inline evaluation must also be safe to retry. If execution fails before final evaluation, rerunning the same `run_key` should still behave like any other failed run attempt. If evaluation itself fails after predictions are written, the run must remain failed so a later rerun or report refresh can repair it. The report refresh path should remain able to backfill missing `final_metrics/` artifacts for historical runs created before this plan landed.

Correcting `main_results.py` and adding `smoke_one_study.py` must not change any existing concrete run payload semantics beyond intentionally adding the baseline runs to the main-results experiment surface. Shared `run_key`s between `main_results` and `covariate_comparison` must continue to match exactly when the canonical payload matches.

## Artifacts and Notes

The most important evidence for this plan is now the test and CLI transcript below:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner tests.bench_v2.test_eval_runner tests.bench_v2.test_report_html
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner tests.bench_v2.test_energy_score tests.bench_v2.test_eval_runner tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html
    Ran 80 tests in 5.556s
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench plan experiments/v2/main_results.py
    Planned runs: 4144
    Existing runs: 0
    Blocked runs: 3924
    Runnable runs: 220
    Resolved manifest: /data/vision/beery/scratch/justin/movebench/results/bench_v2/resolved_experiments/main_results.json

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench status --format text
    Studies: 148
    Blocked studies: 148
    SSF-ready pairs: 44
    deepSSF-cache-ready pairs: 0
    Fully runnable pairs: 0

The real one-study smoke execution and `bench report` regeneration commands were not run against a concrete study in this environment because the status snapshot above showed no fully runnable study-timescale pairs with deepSSF cache readiness.

For artifact shape, a completed run directory now looks like:

    movebench bench plan experiments/v2/main_results.py
    Planned runs: 280
    Existing runs: 112
    Blocked runs: 16
    Runnable runs: 264
    Resolved manifest: results/bench_v2/resolved_experiments/main_results.json

and a completed run directory excerpt like:

    results/bench_v2/runs/<run_key>/
      resolved_run.json
      training_summary.json
      predictions/
      final_metrics/
        test-temporal/
          energy_by_step.csv.gz
          summary.json
        test-individual/
          energy_by_step.csv.gz
          summary.json

The MLflow run for the same attempt should show metrics such as:

    energy_score.test_temporal
    energy_score.accuracy.test_temporal
    energy_score.calibration.test_temporal
    n_steps_scored.test_temporal
    n_steps_excluded_out_of_support.test_temporal

The exact numeric values are not important in this plan. The important proof is that those metrics now belong to the normal `bench run` completion path instead of appearing only after a later report refresh.

## Interfaces and Dependencies

Keep the existing CLI surface:

    movebench bench plan <experiment_module.py>
    movebench bench run <experiment_module.py>
    movebench bench report
    movebench bench status

The intended code-level interfaces after this follow-up are:

In [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py):

    _command_plan(args: argparse.Namespace) -> int

should print `Planned runs`, `Existing runs`, and `Blocked runs`, and may also continue to print `Runnable runs` plus the resolved manifest path.

    _command_status(args: argparse.Namespace) -> int
    _command_report(args: argparse.Namespace) -> int

should import the report module only inside the HTML/report code path, not at module import time.

    _execute_run(run_payload: dict[str, object]) -> dict[str, object]

should call the evaluation runner after a completed executor result and before tagging the MLflow run completed.

In [movebench/bench_v2/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/evaluation/runner.py):

    def evaluate_run_dir(
        run_dir: str | Path,
        *,
        mlflow_run_id: str | None = None,
    ) -> dict[str, object]: ...

should write the existing final-metric artifacts and log metrics either to the supplied MLflow run id or, when omitted, to the latest run for the run key as a repair path.

In [experiments/v2/smoke_one_study.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/smoke_one_study.py), define exactly one plain `EXPERIMENT` dictionary with:

    name = "smoke_one_study"
    kind = "smoke_one_study"
    studies = "all"
    timescales = ["1d"]
    covariate_sets = ["all"]
    methods = ["issf_linear", "deepssf", "moveformer"]
    seed = 0

In [experiments/v2/main_results.py](/data/vision/beery/scratch/justin/movebench/experiments/v2/main_results.py), the `methods` list must include:

    baseline_nomove
    baseline_constvel
    baseline_randomwalk
    issf_linear
    issf_mlp
    deepssf
    moveformer

Plan creation note (2026-03-27 / assistant): created this follow-up ExecPlan after the milestone 1-3 audit found that the implementation tree already contained the core planner/runtime/evaluator/report code, but the overview acceptance still had five remaining user-visible gaps around control-plane imports, plan output counts, missing experiment modules, baseline coverage, and inline final evaluation.
