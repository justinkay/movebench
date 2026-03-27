# Fix Planner-Owned Blocking and Canonical deepSSF Run Identity

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md), [execplans/benchmark_rewrite_method_runtime_scoped_followup_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_scoped_followup_execplan.md), and [execplans/benchmark_rewrite_method_runtime_deepssf_run_blocking_and_steps_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_deepssf_run_blocking_and_steps_execplan.md). It covers only two remaining gaps from the scoped deepSSF review: planner versus executor ownership of blocked artifact checks, and the fact that deepSSF still has behavior-shaping defaults outside the canonical concrete-run payload. MoveFormer, evaluation/reporting, extraction, and cache building remain out of scope.

## Purpose / Big Picture

After this change, a run that is missing artifacts or has incompatible artifacts will usually be blocked during planning, not after execution has already started. That keeps the runtime simpler to follow: planning decides whether a run is runnable, and execution only rechecks the same contract right before dispatch as a safety net. If the missing or incompatible artifact is later repaired on disk, a plain `movebench bench run ...` should execute the run automatically with no `--rerun` flag.

After the same change, deepSSF runs will be identified by the full set of knobs that already affect execution. The deepSSF batch size and repeated-evaluation budget will live in the method catalog, so they become part of `run_key`, `resolved_run.json`, and MLflow params automatically. The visible proof is a small set of tests that show planning now blocks per-run artifact problems up front, the runner skips only `completed` and `running` latest runs by default, and changing deepSSF `batch_size` or `max_eval_items` changes the run key.

## Progress

- [x] (2026-03-26 16:20Z) Reviewed the current rerun policy, deepSSF executor defaults, and existing follow-up ExecPlans to isolate the remaining scope for this pass.
- [x] (2026-03-26 16:39Z) Revised this ExecPlan after user feedback to move artifact blocking into planning wherever possible and to keep the codepath as simple as possible.
- [x] (2026-03-26 17:05Z) Added failing planning tests that prove per-run artifact problems are blocked during `expand_experiment(...)`, not only during `_execute_run(...)`.
- [x] (2026-03-26 17:05Z) Added failing runner tests for the simpler skip policy and the explicit rerun override.
- [x] (2026-03-26 17:05Z) Added failing planning and runner assertions proving deepSSF `batch_size` and `max_eval_items` are part of the canonical run payload and run identity.
- [x] (2026-03-26 17:16Z) Updated `movebench/bench_v2/run.py` to reuse `_resolve_artifacts_for_run(...)` during planning, kept execution-time revalidation, and skipped only `completed` and `running` latest runs by default.
- [x] (2026-03-26 17:16Z) Moved deepSSF `batch_size` and `max_eval_items` defaults into `movebench/bench_v2/config.py`, removed executor-owned fallbacks, and updated the targeted runtime tests.
- [x] (2026-03-26 17:31Z) Ran the affected `bench_v2` test suites in the movebench micromamba environment and recorded the observed results below.

## Surprises & Discoveries

- Observation: the current split between readiness checks and per-run resolution is the main reason “planner-passed but executor-blocked” still happens.
  Evidence: `expand_experiment(...)` currently uses `_study_timescale_readiness(...)` to decide runnable versus blocked, but it does not call `_resolve_artifacts_for_run(...)` before appending a run to the runnable list.

- Observation: reusing the existing per-run resolver in planning is simpler than introducing a second planner-only validation layer.
  Evidence: `_resolve_artifacts_for_run(...)` already contains the concrete checks for point-store band coverage, normalization stats, deepSSF shard presence, deepSSF channel compatibility, and deepSSF split metadata consistency.

- Observation: deepSSF still keeps execution-shaping defaults inside the executor even though the planner already expands method defaults into each run payload.
  Evidence: `movebench/bench_v2/config.py` defines deepSSF optimizer and step-schedule defaults but does not define `batch_size` or `max_eval_items`, while `movebench/bench_v2/executors/deepssf.py::run(...)` still falls back to `32` and `DEFAULT_EVAL_MAX_ITEMS`.

- Observation: fixing deepSSF run identity will intentionally change default deepSSF `run_key` values even when the user does not edit the experiment file.
  Evidence: adding `batch_size` and `max_eval_items` to default `method_hparams` changes the canonical payload hashed by `_run_key(...)`.

- Observation: once planning started calling the concrete resolver, some existing tests were no longer modeling valid planner inputs.
  Evidence: `tests/bench_v2/test_plan.py` needed train normalization stats in fixtures that expected iSSF runs to be planned, and `tests/bench_v2/test_artifact_resolution.py` had to switch invalid-artifact cases to “plan valid payload first, then mutate artifacts before revalidation.”

## Decision Log

- Decision: planner should call the same `_resolve_artifacts_for_run(...)` function that execution uses, instead of maintaining a separate planner-only artifact validation path.
  Rationale: this is the smallest change that makes blocked behavior straightforward. One resolver defines the artifact contract, planning uses it to decide runnable versus blocked, and execution reuses it as a last-second safety check against filesystem drift.
  Date/Author: 2026-03-26 / assistant

- Decision: keep `_study_timescale_readiness(...)` for coarse split discovery and status reporting, but stop relying on it as the final word on whether a concrete run is runnable.
  Rationale: readiness is still useful for choosing held-out splits and for `bench status`, but it is too coarse to validate band subsets, normalization coverage, and deepSSF per-run metadata constraints.
  Date/Author: 2026-03-26 / assistant

- Decision: the default runner policy will skip only latest `completed` and `running` runs.
  Rationale: that policy is simple to explain and matches the user’s desired recovery behavior. If an artifact problem is fixed, the run becomes runnable again at planning time and will execute on the next plain `bench run`. Failed and blocked latest runs remain retryable by default.
  Date/Author: 2026-03-26 / assistant

- Decision: keep one explicit boolean rerun control, `--rerun`, to force reruns of otherwise-skipped runnable runs.
  Rationale: this stays simple in the CLI and is enough to cover “run again even though it completed already” or “run again despite a stale running record.”
  Date/Author: 2026-03-26 / assistant

- Decision: deepSSF `batch_size` and repeated-evaluation budget belong in `METHODS["deepssf"]["hparams"]`.
  Rationale: those values change execution behavior and therefore belong in the same canonical payload as the rest of the deepSSF schedule and optimizer settings. Keeping them in `method_hparams` automatically makes them part of run identity, resolved manifests, and MLflow parameter logging.
  Date/Author: 2026-03-26 / assistant

- Decision: do not migrate or delete older deepSSF run directories or MLflow runs whose `run_key` was computed before the identity fix.
  Rationale: the old run key was incomplete. The safest correction is to let future planning produce a new, more precise `run_key` and let old artifacts coexist as historical outputs rather than trying to rewrite them in place.
  Date/Author: 2026-03-26 / assistant

## Outcomes & Retrospective

Implemented.

The runtime now follows the intended simple flow. `expand_experiment(...)` still uses `_study_timescale_readiness(...)` to choose available splits, but it now calls `_resolve_artifacts_for_run(...)` before a run is admitted to the runnable list. Concrete point-store and deepSSF artifact defects are therefore planner-owned by default, while `_execute_run(...)` still re-runs the same resolver as a safety check against filesystem drift. The runner policy is now equally direct: skip latest `completed` and `running` runs by default, retry latest `failed`, `blocked`, and unknown runs by default, and let `--rerun` force execution for otherwise-skipped runnable runs.

The deepSSF identity gap is also closed. `METHODS["deepssf"]["hparams"]` now includes `batch_size` and `max_eval_items`, and the executor reads both values directly from the canonical payload instead of inventing defaults internally. That means `run_key`, `resolved_run.json`, and MLflow params all reflect those knobs. As expected, this changes default deepSSF `run_key` values relative to pre-fix historical outputs; old outputs were left in place.

Observed validation in the movebench micromamba environment:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_runner
    initial result after code changes: 22 tests, 2 failures in planner fixtures

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_runner
    final result after fixture updates: 22 tests, OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution
    result after converting invalid-artifact cases to drift-style revalidation tests: 8 tests, OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner
    final regression result: 42 tests, OK

## Context and Orientation

The relevant runtime surface lives in four files. `movebench/bench_v2/run.py` expands experiments into concrete runs, computes each `run_key`, decides which planned runs are pending, and dispatches execution. `movebench/bench_v2/config.py` defines the default method catalog that `expand_experiment(...)` copies into each `run_payload`. `movebench/bench_v2/executors/deepssf.py` adapts the deepSSF model to the shared trainer and still owns two hidden defaults that affect behavior. `movebench/bench_v2/mlflow.py` starts benchmark runs and logs `method_hparams` into MLflow.

In this repository, a “concrete run payload” means the fully expanded Python dictionary for one benchmark run after planning has chosen the study, timescale, covariate set, method, schedule, and artifact requirement. The runner writes that payload to `results/bench_v2/runs/<run_key>/resolved_run.json`, except for a few experiment-membership fields that are intentionally excluded. The same canonical payload is also the source of truth for `run_key`, because `movebench/bench_v2/run.py::_run_key(...)` hashes it.

The current planner has two layers. `_study_timescale_readiness(...)` is a coarse readiness helper that answers questions like “does this study have a train deepSSF cache and at least one held-out deepSSF cache?” It is useful for status reporting and for choosing available evaluation splits. `_resolve_artifacts_for_run(...)` is the concrete per-run resolver that checks the exact artifact contract for one run payload. That second layer is where the real band-subset, normalization, shard, and deepSSF metadata checks already live. The simplest fix is to reuse it during planning instead of treating it as execution-only.

The current rerun-policy bug is mostly a consequence of that split. A run can be admitted by planning, later block during `_execute_run(...)`, and then be retried repeatedly because the runner treats blocked latest runs as retryable. If planning owns the concrete artifact checks, those blocked runs usually never reach execution. The remaining retry logic becomes much easier to reason about.

The deepSSF identity bug is mechanical. The planner already expands deepSSF defaults into `run_payload["method_hparams"]`, and MLflow already logs every key in that mapping. The only reason `batch_size` and the repeated-evaluation budget are missing from the canonical payload is that `movebench/bench_v2/config.py` does not define them there and `movebench/bench_v2/executors/deepssf.py` silently invents them at runtime. That means two deepSSF runs can behave differently without their resolved payloads, run keys, or MLflow params reflecting the difference.

The main tests for this work are `tests/bench_v2/test_plan.py`, `tests/bench_v2/test_runner.py`, `tests/bench_v2/test_mlflow.py`, and `tests/bench_v2/test_deepssf_runtime.py`. `test_plan.py` already has helpers for creating synthetic point stores and deepSSF caches. `test_runner.py` already has helpers for synthetic latest MLflow statuses. `test_mlflow.py` already proves that `method_hparams` are logged as MLflow params. Those tests are the shortest path to proving the change without building a larger smoke experiment first.

## Plan of Work

Begin with planning tests so the ownership change is visible. In `tests/bench_v2/test_plan.py`, add one case for a point-store-backed method and one case for deepSSF showing that a run with a per-run artifact defect is blocked during `expand_experiment(...)`. For the point-store side, use a missing or incomplete train normalization stats file, because that is a concrete per-run check that currently lives only in `_resolve_point_artifacts(...)`. For deepSSF, use a covariate subset whose requested channel view exceeds the cache manifest’s `env_channels`, or a split metadata mismatch such as inconsistent `output_hw`. The important assertion is that these runs land in `experiment["blocked"]` and do not appear in the runnable `runs` list.

Next, update the runner tests to match the simpler policy. In `tests/bench_v2/test_runner.py`, replace the current “blocked and running are retried” expectation with two clearer tests. One should prove that latest `completed` and latest `running` runs are skipped by default. A second should prove that latest `failed` and latest `blocked` runs are retried by default. Add a third test that passes `rerun=True` into `run_experiment(...)` and proves a latest `completed` or `running` run is executed again when forced. Keep the existing stale-output cleanup test unchanged, because the same cleanup should still happen whenever a new execution attempt actually starts.

Then add the canonical-identity tests for deepSSF. In `tests/bench_v2/test_plan.py`, extend the existing deepSSF planning assertions so every planned deepSSF run carries `batch_size` and `max_eval_items` in `method_hparams`, not just the step-schedule keys. Add an identity-focused test that builds two otherwise identical deepSSF payloads differing only in `batch_size` or only in `max_eval_items` and asserts that `run_key` changes. In `tests/bench_v2/test_runner.py`, extend the successful deepSSF runner test to assert that the `start_benchmark_run(...)` payload or the written `resolved_run.json` contains those same keys. In `tests/bench_v2/test_mlflow.py`, add or extend one helper test so a deepSSF-like payload with those keys proves they are logged as MLflow params.

After the tests fail, patch `movebench/bench_v2/run.py` as simply as possible. Keep `_study_timescale_readiness(...)` for split discovery, blocked reasons in `bench status`, and choosing early-stop and evaluation splits. After `expand_experiment(...)` builds a tentative `run_payload`, immediately call `_resolve_artifacts_for_run(run_payload)`. If the resolver returns blocked, append that run to `experiment["blocked"]` with the resolver’s reason and do not add it to `runs`. If it resolves, add the payload to `runs` exactly as before. Do not add a second resolver, do not store resolved artifact objects in the manifest, and do not create planner-only MLflow runs or run directories. Then keep `_execute_run(...)` revalidation exactly because files can still change between planning and execution.

Still in `movebench/bench_v2/run.py`, simplify the pending-run policy. `run_experiment(...)` should gain `rerun: bool = False`. When `rerun` is false, skip a runnable run only if the latest MLflow status is `completed` or `running`. When `rerun` is true, do not skip on status at all. Leave failed and blocked runs retryable by default. Then update `_command_run(...)` and `register(...)` in the same file so `movebench bench run ... --rerun` threads that control into `run_experiment(...)`.

Finally, patch deepSSF default expansion. In `movebench/bench_v2/config.py`, add `batch_size` and `max_eval_items` to `METHODS["deepssf"]["hparams"]` beside the existing optimizer and schedule defaults. In `movebench/bench_v2/executors/deepssf.py`, remove the silent fallbacks for those values and read them directly from `run_payload["method_hparams"]`. Keep the rest of the deepSSF runtime unchanged. The goal is not a deepSSF redesign; it is simply to stop inventing behavior-shaping defaults inside the executor.

Update the small test fixtures that currently rely on the old deepSSF fallback behavior. `tests/bench_v2/test_deepssf_runtime.py` should build `method_hparams` with both `batch_size` and `max_eval_items`, and the bounded-eval test should set `max_eval_items` directly in the payload instead of patching an executor default constant. If helper functions such as `tests/bench_v2/test_plan.py::_base_run_payload(...)` still create incomplete deepSSF `method_hparams`, update them so the fixtures reflect the new canonical defaults and stay easy to read.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the failing planning tests first and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan

Add the retry-policy tests and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner

Add the deepSSF run-identity tests and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow

Update the deepSSF runtime fixtures and rerun the targeted runtime surface:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner tests.bench_v2.test_plan tests.bench_v2.test_mlflow

After the code changes are complete, rerun the affected `bench_v2` regression suite:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner

For a small manual CLI check after implementation, run the same experiment three ways:

    movebench bench run <experiment_module.py> --max-runs 1
    movebench bench run <experiment_module.py> --max-runs 1
    movebench bench run <experiment_module.py> --max-runs 1 --rerun

If the latest run completed on the first command, the second should report a skipped run and zero executions. The third should report one executed run for the same planned run key. If the experiment was previously blocked only because an artifact was missing and that artifact is repaired before the first command, the first command should execute without needing `--rerun`.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are all observable.

First, planning owns concrete artifact blocking wherever possible. A run with a missing normalization stats file, incomplete normalization stats, missing deepSSF shards, incompatible deepSSF channel view, or incompatible deepSSF split metadata must be recorded in `experiment["blocked"]` during `expand_experiment(...)` and must not appear in the runnable `runs` list. This proves the planner is using the concrete resolver instead of only coarse readiness checks.

Second, execution still revalidates the same artifact contract. `_execute_run(...)` must continue to call the same resolver before executor dispatch, so a file that disappears or changes after planning can still block safely. This blocked-at-execution path should remain possible, but it should become the exception rather than the normal artifact-failure path.

Third, the default retry policy is simple. If the latest MLflow run for a runnable `run_key` has status `completed` or `running`, `run_experiment(...)` skips it by default. If there is no latest run, or the latest status is `failed`, `blocked`, or unknown, the run executes. Passing `rerun=True`, or invoking `movebench bench run ... --rerun`, forces execution regardless of latest status.

Fourth, a run that becomes unblocked after an artifact repair should execute on the next plain `bench run`. This is observable because the repaired run will now be admitted by planning, and the default skip logic does not require `--rerun` for blocked or missing prior runs.

Fifth, deepSSF default runs are canonicalized correctly. A planned deepSSF run now carries `batch_size` and `max_eval_items` in `method_hparams`, those fields appear in `resolved_run.json`, and MLflow logs them as params for the run. Two deepSSF run payloads that differ only in `batch_size` or `max_eval_items` must produce different `run_key` values.

Sixth, the real deepSSF runtime path still works after the cleanup. The targeted runtime tests must still prove channel slicing, bounded repeated evaluation, history writing, checkpoint writing, and prediction writing with the now-explicit deepSSF hyperparameter payload.

## Idempotence and Recovery

This plan keeps reruns idempotent by reusing the existing run-directory cleanup in `_execute_run(...)`. Repeating `movebench bench run ... --rerun` for the same `run_key` is safe: each new execution attempt clears runtime-owned outputs, writes a fresh `resolved_run.json`, and creates a new MLflow run while leaving earlier MLflow history intact.

The planner-owned blocking change is also safe to repeat. `expand_experiment(...)` will read the current artifact state each time it is called. If an artifact is still missing or incompatible, the run stays in `experiment["blocked"]`. If the artifact has been repaired, the run becomes runnable again with no extra state transition or manual unblocking step. This is exactly why the same resolver should be used in planning and execution.

Correcting deepSSF run identity is not backwards-compatible at the `run_key` level. Default deepSSF plans created after this change will get different run keys than equivalent plans created before it, because the canonical payload is more complete. Do not attempt to rewrite old MLflow runs, old run directories, or old resolved manifests. Let old and new outputs coexist, and treat the post-fix identity as authoritative going forward.

## Artifacts and Notes

The most useful evidence for this plan will be short planning and runner summaries plus one concrete deepSSF `resolved_run.json` excerpt. The key examples to capture after implementation are:

    planning after repaired artifact:
      blocked_runs = 0
      planned_runs = 1

    default runner policy with an already-completed run:
      planned_runs = 1
      executed_runs = 0
      skipped_runs = 1

    forced rerun:
      planned_runs = 1
      executed_runs = 1
      skipped_runs = 0

and a deepSSF resolved payload excerpt like:

    "method_hparams": {
      "batch_size": 32,
      "eval_every_steps": 1000,
      "lr_habitat": 0.0001,
      "lr_movement": 0.00001,
      "max_eval_items": 2500,
      "max_steps": 100000,
      "patience_evals": 5,
      "weight_decay": 0.0
    }

The exact numbers may differ if the defaults are revised during implementation, but the important requirement is that the behavior-shaping keys now live in the canonical payload instead of being invented inside the executor.

## Interfaces and Dependencies

In `movebench/bench_v2/run.py`, keep the main public API compact. The end-state signature should be:

    run_experiment(
        experiment_module: str,
        *,
        studies_override: list[str] | None = None,
        methods: list[str] | None = None,
        max_runs: int | None = None,
        rerun: bool = False,
    ) -> dict[str, object]

The key end-state behavior must be:

    expand_experiment(...)
      uses _resolve_artifacts_for_run(run_payload) before appending a run to the runnable list

    rerun = False
      skip only latest status in {"completed", "running"}

    rerun = True
      skip nothing based on latest status

In the same file, extend the `bench run` parser registration so the CLI accepts:

    --rerun

This flag must be wired through `_command_run(...)` into `run_experiment(...)`.

In `movebench/bench_v2/config.py`, `METHODS["deepssf"]["hparams"]` must expose all of the following keys directly in the planned payload:

    {
      "lr_movement": ...,
      "lr_habitat": ...,
      "weight_decay": ...,
      "batch_size": ...,
      "max_steps": ...,
      "eval_every_steps": ...,
      "patience_evals": ...,
      "max_eval_items": ...
    }

In `movebench/bench_v2/executors/deepssf.py`, `run(...)` must read `batch_size` and `max_eval_items` directly from `run_payload["method_hparams"]`. After this plan, there must be no executor-owned default for either value that can change behavior without changing the canonical run payload.

Plan change note (2026-03-26 16:39Z): revised this ExecPlan after user feedback to keep the codepath simple. Artifact blocking is now specified as planner-owned wherever possible by reusing the existing concrete resolver during `expand_experiment(...)`, execution keeps only a safety-net revalidation, and the default retry policy is simplified to “skip completed and running, retry everything else.”
