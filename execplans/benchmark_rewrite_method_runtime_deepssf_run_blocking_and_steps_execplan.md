# Fix deepSSF Bench Run Blocking and Step-Based Scheduling

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md) and [execplans/benchmark_rewrite_method_runtime_deepssf_followup_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_deepssf_followup_execplan.md). It covers only three remaining deepSSF runtime issues from the scoped review: resolver-owned cache metadata validation, `bench run` blocked-run accounting, and removal of epoch-based scheduling from the `bench_v2` deepSSF path. The all-invalid-batch optimizer-step semantics issue is intentionally out of scope for this plan.

## Purpose / Big Picture

After this change, the `bench_v2` deepSSF path will behave like the ExecPlan documents describe instead of a partially migrated prototype. Cache metadata mismatches such as inconsistent chip size or pixel scale will be rejected during artifact resolution before executor startup. `movebench bench run ...` will report planner-blocked deepSSF runs instead of silently showing zero planned and zero blocked runs when prerequisites are missing. The deepSSF runtime and method catalog will use step-based schedules only, matching the shared trainer and the original runtime rewrite design.

The visible proof is threefold. First, a deepSSF experiment whose cache manifests disagree on `output_hw` or `pixel_size_m` is marked blocked before the executor loads. Second, a `bench run` invocation against a missing-cache deepSSF experiment reports the blocked run in its summary instead of dropping it from the counts. Third, the deepSSF runtime tests and default method configuration use `max_steps`, `eval_every_steps`, and `patience_evals`, with no epoch translation layer left in `bench_v2`.

## Progress

- [x] (2026-03-25 21:13Z) Reviewed the follow-up scope and decided to create a new ExecPlan for issues 2, 3, and 4 rather than extending the completed deepSSF follow-up plan.
- [x] (2026-03-26 14:41Z) Added failing tests for resolver-owned deepSSF cache metadata validation, including `output_hw` and `pixel_size_m` mismatches.
- [x] (2026-03-26 14:41Z) Added failing tests for `bench run` blocked-run accounting when deepSSF runs are blocked during planning rather than executor dispatch.
- [x] (2026-03-26 14:41Z) Converted deepSSF runtime tests and default `bench_v2` method configuration to step-based schedules only.
- [x] (2026-03-26 14:41Z) Moved deepSSF cache metadata validation fully into `movebench/bench_v2/run.py` and removed unneeded epoch and duplicate validation code from `movebench/bench_v2/executors/deepssf.py`.
- [x] (2026-03-26 14:41Z) Ran the targeted and affected `bench_v2` test suites in the movebench micromamba environment and recorded the observed results below.

## Surprises & Discoveries

- Observation: `run_experiment(...)` currently reports `blocked_runs` only for runs that reached `_execute_run(...)`, so planner-blocked deepSSF runs are omitted from the user-visible summary.
  Evidence: `movebench/bench_v2/run.py` records planner-blocked runs in `experiment["blocked"]`, but `run_experiment(...)` increments `blocked_runs` only from executor summaries.

- Observation: deepSSF cache compatibility is split across the resolver and the executor, which means some malformed cache sets start an MLflow run and only fail after executor startup.
  Evidence: `movebench/bench_v2/run.py::_resolve_deepssf_cache_artifacts(...)` validates shard visibility and `env_channels`, while `movebench/bench_v2/executors/deepssf.py::run(...)` separately validates `output_hw` and `pixel_size_m`.

- Observation: the `bench_v2` deepSSF runtime is already step-based under the hood, and the vendored deepSSF model does not require epoch scheduling.
  Evidence: `movebench/bench_v2/train.py` takes `TrainerConfig(max_steps, eval_every_steps, patience_evals, seed)`, while the epoch vocabulary survives only in `movebench/bench_v2/config.py` and the executor’s `_resolve_schedule(...)` adapter.

- Observation: the blocked-run accounting fix did not require any change to planning semantics or MLflow behavior.
  Evidence: keeping `expand_experiment(...)` unchanged and folding `len(experiment["blocked"])` into `run_experiment(...)` was enough to make the missing-cache runner test pass while `start_benchmark_run(...)` remained uncalled for planner-blocked runs.

## Decision Log

- Decision: make the resolver the only owner of deepSSF cache metadata compatibility checks for `output_hw`, `pixel_size_m`, and `env_channels`.
  Rationale: the resolver already reads every required cache manifest, so this adds no meaningful computational cost and removes duplicated validation logic from the executor.
  Date/Author: 2026-03-25 / assistant

- Decision: fix the blocked-run regression at the `run_experiment(...)` summary layer instead of synthesizing new MLflow runs or blocked run directories for planner-blocked entries in this pass.
  Rationale: the user-visible bug is that `bench run` hides planner-blocked deepSSF runs from its counts. Counting and surfacing those blocked entries fixes the regression without reopening concrete-run identity, run-key, or MLflow semantics.
  Date/Author: 2026-03-25 / assistant

- Decision: remove epoch compatibility from the `bench_v2` deepSSF path instead of preserving it behind a translation helper.
  Rationale: the user explicitly wants the new refactor to replace the old deepSSF config style. Keeping epoch shims would preserve an interface the ExecPlan documents already said to retire.
  Date/Author: 2026-03-25 / assistant

## Outcomes & Retrospective

This plan is complete. The deepSSF cache resolver in `movebench/bench_v2/run.py` now owns all split-metadata compatibility checks that depend only on cache manifests: `env_channels`, `output_hw`, and `pixel_size_m`. The executor no longer duplicates the split-consistency checks and now reads only step-based schedule keys. The `bench_v2` deepSSF defaults in `movebench/bench_v2/config.py` also use `max_steps`, `eval_every_steps`, and `patience_evals`, which aligns the method catalog with the shared step-based trainer.

The user-visible runner regression is also fixed. `run_experiment(...)` now reports planner-blocked deepSSF runs in its summary without changing planning semantics, run-directory behavior, or MLflow behavior for those cases. A missing-cache deepSSF experiment now reports one planned run and one blocked run instead of zero and zero.

Observed validation runs:

- `2026-03-26 14:38Z`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner tests.bench_v2.test_plan` -> `Ran 27 tests ... OK`
- `2026-03-26 14:41Z`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner` -> `Ran 37 tests ... OK`

## Context and Orientation

The relevant runtime code lives in three places. `movebench/bench_v2/run.py` plans concrete benchmark runs, records planner-blocked entries, resolves on-disk artifacts, and executes supported methods. `movebench/bench_v2/executors/deepssf.py` adapts the deepSSF model to the shared training loop, creates WebDataset loaders, writes checkpoints, and writes prediction artifacts. `movebench/bench_v2/config.py` defines the default deepSSF method configuration used by `expand_experiment(...)`.

In this repository, a deepSSF cache is an existing on-disk artifact under `studies/<study_id>/cache/deepssf_wds/<split>/<cache_hash>/`. Each required split has a `manifest.json` that records grid metadata including `output_hw`, `pixel_size_m`, `num_samples`, and `env_channels`. The runtime is supposed to validate those manifests before execution starts and block the run if the cache is unusable. Right now, only part of that validation happens during resolution. The rest happens after the executor has already been chosen, which is too late for a read-only prerequisite check.

The blocked-run summary bug is narrower. `expand_experiment(...)` currently returns only runnable runs and stores planner-blocked cases in `experiment["blocked"]`. `run_experiment(...)` writes the resolved manifest and executes runnable runs, but it reports `blocked_runs` using only executor return values. That means a deepSSF experiment with missing caches can produce a summary with `planned_runs=0` and `blocked_runs=0` even though the resolved manifest contains one blocked entry. The fix in this plan is to make `bench run` report those planner-blocked entries instead of hiding them.

The step-schedule issue is entirely in the new refactor, not in the vendored deepSSF model. The shared trainer in `movebench/bench_v2/train.py` is step-based already. The remaining epoch vocabulary is in the default deepSSF method configuration in `movebench/bench_v2/config.py`, the epoch-to-step adapter in `movebench/bench_v2/executors/deepssf.py`, and the deepSSF runtime tests that still construct run payloads with `max_epochs`, `eval_every_epochs`, and `patience_epochs`. Those pieces should be rewritten to use step keys directly.

The main tests for this work are `tests/bench_v2/test_artifact_resolution.py`, `tests/bench_v2/test_deepssf_runtime.py`, `tests/bench_v2/test_runner.py`, and `tests/bench_v2/test_plan.py`. The runtime regression surface also includes `tests/bench_v2/test_status.py`, because the readiness and planning helpers share the same blocked-run accounting inputs as `bench run`.

## Plan of Work

Start with failing tests so each requested change has an observable before-and-after behavior. In `tests/bench_v2/test_artifact_resolution.py`, add one case where the train and held-out deepSSF manifests disagree on `output_hw` and a second case where they disagree on `pixel_size_m`. Both tests should call `_resolve_artifacts_for_run(...)` and assert `{"status": "blocked", ...}` with a concise reason that names the incompatible metadata field. Add a companion runner-level test in `tests/bench_v2/test_runner.py` that constructs a deepSSF experiment with one of those mismatches and asserts that `_load_executor(...)` is never called. This proves the validation moved fully into the resolver instead of staying in the executor.

Then add the blocked-run accounting tests. In `tests/bench_v2/test_runner.py`, create a minimal deepSSF experiment whose prerequisite caches are missing entirely. The test should assert that `run_experiment(...)` returns a summary where `planned_runs` counts the experiment’s blocked run, `executed_runs` remains zero, and `blocked_runs` includes the planner-blocked entry. Add a narrow planning test update in `tests/bench_v2/test_plan.py` if needed so the expected blocked-entry bookkeeping remains explicit when `run_experiment(...)` consumes the same plan.

Next, convert the deepSSF runtime tests to the step vocabulary. Update the deepSSF payload fixtures in `tests/bench_v2/test_deepssf_runtime.py` to use `max_steps`, `eval_every_steps`, and `patience_evals`. If a small explicit schedule assertion improves clarity, add one that checks the summary’s `schedule` payload matches the step keys passed in. The goal is for the tests to fail if the executor still expects epoch keys. Update any runner fixtures or default method assumptions in `tests/bench_v2/test_runner.py` that still mention epoch fields.

Only after the tests are failing, patch `movebench/bench_v2/run.py`. Extend `_resolve_deepssf_cache_artifacts(...)` so it treats the train manifest as the reference for grid shape and pixel scale, then blocks any required split whose `output_hw` or `pixel_size_m` differs. Keep the existing `env_channels` check in the same resolver function so all deepSSF cache compatibility logic lives in one place. In `run_experiment(...)`, fold `len(experiment["blocked"])` into the summary counts before applying `max_runs`, and make `planned_runs` report the full number of deepSSF runs considered, not just the runnable subset. This keeps execution semantics unchanged while making the user-visible summary honest.

Finally, patch `movebench/bench_v2/config.py` and `movebench/bench_v2/executors/deepssf.py`. Replace the deepSSF default method hyperparameters with step keys, delete the epoch translation helper, and have `run(...)` read the step-based schedule directly from `method_hparams`. Keep the executor’s model-configuration check that `image_dim` matches the cached chip size, because that is not artifact validation; it is a runtime/model consistency check. Remove only the duplicate cache metadata checks that now belong to the resolver.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the new failing deepSSF artifact-resolution tests first and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution

Add the blocked-run accounting tests and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner tests.bench_v2.test_plan

Update the deepSSF runtime tests to use step-based schedules and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_deepssf_runtime

After the code changes are complete, rerun the targeted deepSSF surface:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner tests.bench_v2.test_plan

Then rerun the affected `bench_v2` regression suite:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner

Observed command transcripts:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution
    Ran 8 tests in 0.423s
    FAILED (failures=2)

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner tests.bench_v2.test_plan
    Ran 17 tests in 1.221s
    FAILED (failures=3)

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner tests.bench_v2.test_plan
    Ran 27 tests in 2.986s
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner
    Ran 37 tests in 3.455s
    OK

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are all observable.

First, a deepSSF run whose required cache manifests disagree on `output_hw` or `pixel_size_m` is blocked by `_resolve_artifacts_for_run(...)`, and a runner-level test proves `_load_executor(...)` is never called for that experiment.

Second, `run_experiment(...)` reports planner-blocked deepSSF runs in its returned summary. A missing-cache deepSSF experiment that previously returned `planned_runs=0` and `blocked_runs=0` must now report one blocked run and zero executed runs.

Third, the `bench_v2` deepSSF path uses step schedules only. The default deepSSF method configuration and the runtime tests must use `max_steps`, `eval_every_steps`, and `patience_evals`, and the deepSSF executor must no longer contain an epoch-to-step adapter.

Fourth, the valid deepSSF path must still work. The existing deepSSF runtime test must continue to prove channel slicing, checkpoint writing, history writing, and prediction writing, and the runner test must continue to prove that a valid deepSSF experiment executes end to end.

## Idempotence and Recovery

This plan keeps artifact resolution read-only. Re-running the same missing-cache or metadata-mismatched deepSSF experiment should continue to block until the cache on disk is repaired by the external extraction pipeline. Moving metadata validation into the resolver does not change any files under `studies/<study_id>/cache/deepssf_wds/`; it only changes when the runtime refuses to proceed.

The blocked-run accounting fix is also safe to repeat. It changes only the summary numbers returned by `run_experiment(...)` and the expectations in tests. It does not introduce new destructive behavior, new MLflow retry rules, or new cache writes.

The step-schedule cleanup has no migration step in `bench_v2`. It is safe because the user explicitly wants the new refactor to replace the old deepSSF epoch-style configuration. Legacy deepSSF benchmark YAML files outside `bench_v2` are not part of this plan and should not be used as compatibility targets while implementing it.

## Artifacts and Notes

The key evidence for this plan should be concise test outputs and one blocked-run summary example. The expected blocked summary shape after the fix is:

    {
      "planned_runs": 1,
      "executed_runs": 0,
      "blocked_runs": 1,
      "failed_runs": 0
    }

The most useful negative test examples are:

    invalid deepSSF cache metadata for test-temporal: output_hw train=[101, 101] test-temporal=[99, 99]

and:

    invalid deepSSF cache metadata for test-temporal: pixel_size_m train=30.0 test-temporal=60.0

These messages do not need to match this wording exactly, but they must identify the split and the mismatched field so the blocked reason is actionable.

## Interfaces and Dependencies

In `movebench/bench_v2/run.py`, keep the existing public function signatures, but make the following end-state behavior true.

`_resolve_deepssf_cache_artifacts(run_payload, timescale=...) -> dict[str, object]` must validate `env_channels`, `output_hw`, and `pixel_size_m` for every required split and return `{"status": "blocked", "reason": ...}` when any split is incompatible with the train cache metadata.

`run_experiment(experiment_module, studies_override=None, methods=None, max_runs=None) -> dict[str, object]` must report `planned_runs` as the full number of runs considered by the plan and `blocked_runs` as the sum of planner-blocked and executor-blocked entries. This plan does not require new MLflow runs or blocked run directories for planner-blocked cases.

In `movebench/bench_v2/config.py`, `METHODS["deepssf"]["hparams"]` must expose only:

    {
      "lr_movement": ...,
      "lr_habitat": ...,
      "weight_decay": ...,
      "max_steps": ...,
      "eval_every_steps": ...,
      "patience_evals": ...
    }

In `movebench/bench_v2/executors/deepssf.py`, remove the epoch conversion helper and have `run(...)` read the step-based schedule directly from `run_payload["method_hparams"]`. The executor should keep only runtime-owned checks that cannot be done during artifact resolution, such as verifying that `model.image_dim` matches the cached chip size.

Plan change note (2026-03-25 21:13Z): created this ExecPlan after the scoped review follow-up narrowed the remaining deepSSF work to three issues: resolver-owned cache metadata validation, truthful `bench run` blocked-run accounting, and removal of epoch-style scheduling from the `bench_v2` deepSSF path.

Plan revision note (2026-03-26 14:41Z): updated this ExecPlan after implementation to mark progress complete, record the final behavior, and capture the exact failing-then-passing test runs that proved the change.
