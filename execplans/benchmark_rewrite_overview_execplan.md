# Rebuild the MoveBench Training and Benchmarking Stack

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.MD](/data/vision/beery/scratch/justin/movebench/PLANS.MD).

This plan coordinates three implementation plans that should be executed in order:

- [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md)
- [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md)
- [execplans/benchmark_rewrite_evaluation_reporting_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_evaluation_reporting_execplan.md)

## Purpose / Big Picture

After this rewrite, a user will be able to define the benchmark suite in plain Python, validate that the required study artifacts already exist on disk, train every requested method with one clean runtime, compute exact final energy scores on the full held-out sets, and open one static HTML report that shows completion status and plots for every finished study. The visible proof is a sequence of CLI commands that plan runs, execute runs against existing point stores and deepSSF caches, and regenerate `results/bench_v2/report/index.html` without touching the old YAML benchmark path.

Today the benchmark flow is split across [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py), [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py), and several `scripts/slurm/*.py` utilities. Those files mix configuration parsing, data preparation, model fitting, evaluation, and reporting. This plan replaces that path with a new package under `movebench/bench_v2/`, keeps the old code only as a temporary reference, and removes the legacy path after the new one is proven.

## Progress

- [x] (2026-03-24 17:10Z) Reviewed the current benchmark runner, training loops, dataset wrappers, Slurm evaluation script, and artifact code to locate the existing responsibilities and failure points.
- [x] (2026-03-24 17:30Z) Reviewed the current point-store and deepSSF WebDataset cache path contracts and confirmed that the new runtime can treat them as prerequisites instead of rebuilding them.
- [x] (2026-03-24 17:40Z) Verified that some studies expose only one held-out split, so planning, execution, and reporting must treat the missing split as not applicable rather than as an error.
- [x] (2026-03-24 18:20Z) Chose the replacement architecture: plain Python experiment modules, a compact `movebench/bench_v2/` runtime package, MLflow-only tracking, exact final energy-score evaluation, and a generated static HTML report.
- [x] (2026-03-24 18:45Z) Authored the coordinated ExecPlans for the core runtime, method runtime, and evaluation/reporting layers.
- [ ] Implement [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md).
- [ ] Implement [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md).
- [ ] Implement [execplans/benchmark_rewrite_evaluation_reporting_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_evaluation_reporting_execplan.md).
- [ ] Delete the legacy YAML benchmark path, legacy benchmark-specific training helpers, and obsolete Slurm wrappers after the new runtime reaches parity and passes validation.

## Surprises & Discoveries

- Observation: the existing point-store layout is already a stable enough runtime contract.
  Evidence: [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) resolves stores under `features/points/<split>/<rasters_hash>/<sampler_hash>/` and stores both `manifest.json` and `shared.pt`, which is enough for the new runtime to validate and load them directly.

- Observation: deepSSF’s current WebDataset cache key includes `band_ids`, so naive cache lookup would duplicate subset-covariate caches unnecessarily.
  Evidence: `deepssf_wds_cache_hash(...)` in [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py) hashes `band_ids` directly. The new runtime should therefore resolve the existing `all` cache and slice channels at read time for subset covariate runs.

- Observation: the exact energy score already exists in mathematical form inside the repo, but only in a sampling-heavy Slurm script.
  Evidence: [scripts/slurm/compare_outputs.py](/data/vision/beery/scratch/justin/movebench/scripts/slurm/compare_outputs.py) computes `mean ||X - y|| - 0.5 * mean ||X - X'||` from sampled predictive surfaces. The new runtime should preserve the formula while removing the approximation where exact evaluation is feasible.

- Observation: MLflow is not currently a project dependency.
  Evidence: [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) does not mention `mlflow`, so the rewrite must add and validate that dependency explicitly.

## Decision Log

- Decision: the new benchmark runtime lives under `movebench/bench_v2/` rather than in a separate top-level package.
  Rationale: it keeps the new code inside the existing installable package while making the cutover and later deletion of legacy modules straightforward.
  Date/Author: 2026-03-24 / assistant

- Decision: experiment configuration will be plain Python modules, not YAML and not Hydra.
  Rationale: the experiment surface needs composition and deduplication across methods, covariate sets, studies, and timescales, but a full spec hierarchy was unnecessary.
  Date/Author: 2026-03-24 / assistant

- Decision: one concrete run is identified by study, timescale, method, covariate set, and fixed training settings, and it produces metrics for every available held-out split.
  Rationale: `main_results` and `covariate_comparison` must be able to share the `all` runs; evaluation split is an output of a run, not part of the run identity.
  Date/Author: 2026-03-24 / assistant

- Decision: MLflow alone will be the tracking system in v1.
  Rationale: MLflow already stores params, tags, per-step metrics, and artifacts in its backend database. Adding custom tables would make the design heavier before there is evidence that MLflow API queries are a real bottleneck.
  Date/Author: 2026-03-24 / assistant

- Decision: final energy scores must be exact with respect to the benchmark support used by each method family.
  Rationale: the user explicitly asked for correctness on the full test sets. Sampling is acceptable only as an internal implementation detail for baseline support construction, not for the final energy-score estimator itself.
  Date/Author: 2026-03-24 / assistant

- Decision: the runtime should be additive until parity is proven, then the old benchmark path should be removed.
  Rationale: the old code is still the only working reference for some model details. Deleting it before the new runtime is verified would make regression diagnosis unnecessarily hard.
  Date/Author: 2026-03-24 / assistant

## Outcomes & Retrospective

This plan is a blueprint, not the implementation. Its intended result is a benchmark stack where artifact validation, training, evaluation, status, and reporting read like one coherent system instead of several stitched scripts. If the sibling plans are followed, a later reviewer should be able to judge correctness by inspecting the new `bench_v2` package, running the new CLI, and ignoring the old YAML benchmark path entirely.

## Context and Orientation

The current repository already has reusable model definitions and much of the low-level data plumbing. The rewrite should preserve working pieces of math and IO, while replacing the orchestration layer.

The most important existing files are:

- [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py), the current monolithic benchmark runner.
- [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py), which contains repeated method-specific fit loops.
- [movebench/models/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/ssf.py), [movebench/models/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/deepssf.py), and [movebench/models/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/models/moveformer.py), which contain the model implementations that should be reused.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), [movebench/features/normalization.py](/data/vision/beery/scratch/justin/movebench/movebench/features/normalization.py), and [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), which already define the point-store IO contract used by SSF-style methods.
- [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py), which defines the existing deepSSF cache layout under `cache/deepssf_wds/<split>/<cache_hash>/`.
- [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py), which shows what can be reused and what must be replaced for MoveFormer sequence construction.

A “timescale” in this plan means the movement prediction horizon and corresponding raster buffer configuration. For the initial implementation the timescales are exactly `1h`, `12h`, `1d`, and `7d`.

A “covariate set” means a named subset of the full extracted feature channels. The required sets are `all`, `all_environmental`, `alphaearth_only`, and `lite`.

A “concrete run” means one trainable or non-trainable method evaluated for one study, one timescale, and one covariate set. `test-temporal` and `test-individual` are evaluation outputs of that run.

A “run attempt” means one execution of a concrete run. There may be several attempts if a job crashes and resumes.

A “benchmark-ready point store” means an existing point-store layout under `features/points/<split>/<rasters_hash>/<sampler_hash>/` with a valid `manifest.json`, `shared.pt`, and any referenced band tensors already on disk.

A “benchmark-ready deepSSF cache” means an existing cache under `cache/deepssf_wds/<split>/<cache_hash>/` with a valid `manifest.json` and shard set already on disk.

## Plan of Work

The work should proceed in four milestones.

Milestone 1 builds the new control plane described in [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md). That means plain Python experiment configs, a concrete run planner, MLflow bootstrapping, and a study readiness scanner over existing point stores and deepSSF caches. At the end of this milestone the repository should be able to answer “what runs should exist” and “which studies and timescales are runnable from existing artifacts” without training anything.

Milestone 2 builds the reusable method runtime described in [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md). That means one shared step-based training loop for trainable models, artifact-consuming executors for every method family, superset artifact reuse for covariate subsets, and the new `bench run` execution path. At the end of this milestone the repository should be able to train every requested method and produce prediction artifacts for later evaluation using only the artifacts that already exist on disk.

Milestone 3 implements the exact energy-score evaluator and reporting layer described in [execplans/benchmark_rewrite_evaluation_reporting_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_evaluation_reporting_execplan.md). At the end of this milestone the repository should be able to compute final metrics on all available held-out splits, log them to MLflow, regenerate plots, and write a browsable HTML report, all without invoking extraction code.

Milestone 4 performs the cutover. Only after the first three milestones pass their validations should the implementer remove the old YAML benchmark configs, old benchmark-specific training loops, and obsolete Slurm wrappers. The cutover should preserve the old low-level reusable modules that are still imported by `bench_v2`, but it should remove entrypoints that encourage future contributors to use the legacy pipeline.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Install the dependencies that the new stack needs:

    python -m pip install -e .[bench,moveformer]

Implement the plans in order and keep the progress sections inside each plan current:

    1. execplans/benchmark_rewrite_core_runtime_execplan.md
    2. execplans/benchmark_rewrite_method_runtime_execplan.md
    3. execplans/benchmark_rewrite_evaluation_reporting_execplan.md

After Milestone 1, the following commands should succeed and should not start any training:

    movebench bench plan experiments/v2/main_results.py
    movebench bench status --format text

After Milestone 2, the following command should execute at least one small smoke run end to end, assuming the referenced study already has the required point stores and deepSSF caches on disk:

    movebench bench run experiments/v2/smoke_one_study.py --max-runs 2

After Milestone 3, the following commands should regenerate metrics, plots, and the HTML report:

    movebench bench report
    movebench bench status --format html

When the cutover is complete, the legacy benchmark path should be gone or turned into thin wrappers that immediately direct the user to `movebench bench ...`.

## Validation and Acceptance

The rewrite is complete when all of the following are true.

First, `movebench bench plan experiments/v2/main_results.py` imports the experiment config, expands it into concrete runs, writes a resolved manifest, and prints counts for planned, existing, and blocked runs. Re-running the same command must report the same run count without creating duplicates.

Second, `movebench bench run experiments/v2/main_results.py --studies <study_id>` must train or execute every requested method for one study, write MLflow runs and local artifacts, log step-indexed training and validation metrics to MLflow with `step=global_step`, and record final energy scores for every available held-out split as MLflow metrics. If a required point store, normalization artifact, or deepSSF cache is missing, the affected run must be skipped with a recorded blocked or missing-prerequisite status rather than crashing the whole command.

Third, `movebench bench report` must create or refresh `results/bench_v2/report/index.html`, and that HTML file must include global completion counts, artifact-readiness status, main-results plots, and covariate-comparison plots for all finished runs.

Fourth, `movebench bench status --format text` must summarize what is runnable from existing artifacts and what is still blocked on missing prerequisites.

Fifth, the old path must no longer be the preferred way to run benchmarks. The repository should have one obvious CLI path, one obvious results root, and one obvious experiment-config format.

## Idempotence and Recovery

Every planner action must be idempotent. Planning the same experiment twice must not duplicate rows. Running the same concrete run twice must either skip the completed run or create a new run attempt linked to the same concrete run.

If a study is missing required point stores, normalization stats, or deepSSF caches, it must be surfaced as blocked in command output and in the HTML report, and later reruns should continue past it. If a training attempt crashes, the MLflow run should be marked failed and later retries should simply create a newer MLflow run with the same `run_key` tag.

The cutover itself should be done last. Until then, the old benchmark code should remain readable as a reference and should not be modified in ways that risk breaking unrelated user work.

## Artifacts and Notes

The intended new package shape is intentionally small:

    movebench/bench_v2/
      __init__.py
      config.py
      run.py
      report.py
      train.py
      mlflow.py
      methods/
      evaluation/

The intended new experiment surface is:

    experiments/v2/
      main_results.py
      covariate_comparison.py
      smoke_one_study.py

The intended top-level results layout is:

    results/bench_v2/
      resolved_experiments/
      mlartifacts/
      runs/
      report/

The concrete run key should be human-readable. A recommended format is:

    study=<study_id>__ts=<timescale_key>__cov=<covariate_set_key>__method=<method_key>__seed=<seed>

Do not put evaluation split into that key.

## Interfaces and Dependencies

The rewrite should add a `bench` optional dependency group in [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) that includes at least `mlflow` and `matplotlib`. `moveformer` remains a separate optional dependency because only one method family requires it.

The top-level CLI should be extended in [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) to register the `bench` subcommands directly from the small `bench_v2` modules. There is no need for a second CLI module.

The implementation plans must preserve these stable user commands:

    movebench bench plan <experiment_module.py>
    movebench bench run <experiment_module.py>
    movebench bench report
    movebench bench status

Plan creation note (2026-03-24 / assistant): initial umbrella plan authored after repository inspection to sequence the rewrite into core runtime, method runtime, and evaluation/reporting milestones with an explicit final cutover step.
