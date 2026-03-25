# Fix Scoped Runtime Recovery and Validation for Baselines + iSSF

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md). It intentionally covers only the scoped baseline + iSSF runtime from that pass. deepSSF, MoveFormer, evaluation/reporting, legacy cutover, and a new explicit rerun CLI flag are out of scope here.

## Purpose / Big Picture

After this change, the existing scoped `movebench bench run ...` path for baselines and iSSF will behave like a recoverable runtime instead of a brittle prototype. A malformed point-store manifest, malformed deepSSF cache manifest, or incomplete iSSF normalization stats file will mark only the affected run blocked instead of crashing planning or execution. Re-running after fixing prerequisites will actually retry the run unless the latest run already completed, and retrying a run with the same `run_key` will no longer leave stale checkpoints or prediction artifacts from an older attempt on disk.

The visible proof is a small runner scenario that first records a blocked or interrupted baseline/iSSF run, then succeeds after the underlying artifact or status problem is repaired, while the run directory contains only the newest attempt's summary, history, checkpoint, and predictions.

## Progress

- [x] (2026-03-25 15:22Z) Reviewed the scoped review findings and narrowed the follow-up scope to four issues: invalid artifact blocking, retry/skip recovery, stale run-directory cleanup, and upfront normalization validation for iSSF.
- [x] (2026-03-25 15:25Z) Added failing tests for invalid point manifests during planning/status, incomplete normalization stats during artifact resolution, blocked/running reruns, and stale run-directory cleanup on retries.
- [x] (2026-03-25 15:27Z) Hardened readiness and artifact-resolution paths in `movebench/bench_v2/run.py` so malformed point-store and deepSSF cache manifests become blocked reasons instead of uncaught exceptions.
- [x] (2026-03-25 15:28Z) Updated retry behavior so `run_experiment(...)` skips only completed runs and retries blocked, failed, and running latest statuses by default.
- [x] (2026-03-25 15:28Z) Added cleanup of runtime-owned outputs before each execution attempt so stale checkpoints, history, and prediction artifacts do not survive blocked or failed retries.
- [x] (2026-03-25 15:29Z) Ran the milestone test commands plus the full scoped `bench_v2` suite and confirmed all targeted baseline + iSSF tests pass.

## Surprises & Discoveries

- Observation: the current readiness helper validates manifests earlier than the per-run resolver does, so a malformed on-disk artifact can break `bench plan`, `bench status`, and `bench run` before blocked-run handling has a chance to run.
  Evidence: `_study_timescale_readiness(...)` in `movebench/bench_v2/run.py` calls `read_manifest(...)` and `read_deepssf_wds_manifest(...)` directly.

- Observation: the current runner treats `blocked` and `running` the same way as `completed` for skip purposes, even though the runtime reuses the same `run_key` for retries.
  Evidence: `run_experiment(...)` skips latest statuses in `{"completed", "blocked", "running"}`.

- Observation: normalization validation for iSSF currently happens indirectly inside dataset loading, after the MLflow run has already started.
  Evidence: `_resolve_point_artifacts(...)` checks only that `train_stats.json` exists, while band-level validation happens later in `load_points_normalization_mean_std(...)`.

- Observation: the `tests/bench_v2/test_status.py` fake `movebench.features.points` module had drifted from the production import surface, so the status tests could not import `movebench/bench_v2/run.py` reliably.
  Evidence: before updating the fixture, the status module failed at import time because the fake module did not expose `point_store_dir`, `point_store_manifest_path`, or `point_store_train_stats_path`.

## Decision Log

- Decision: do not add a new `--force` or rerun flag in this plan.
  Rationale: the scoped prompt asked for fixes to the existing baseline + iSSF runtime, and the review concluded that the critical correctness issue is the default skip policy, not missing extra CLI surface area.
  Date/Author: 2026-03-25 / assistant

- Decision: fix invalid artifact handling in the shared readiness helpers, not only in `_resolve_artifacts_for_run(...)`.
  Rationale: the same helper path is used by `bench plan`, `bench status`, and `bench run`, so patching only the later resolver would leave the runtime inconsistent and still crash on malformed external artifacts.
  Date/Author: 2026-03-25 / assistant

- Decision: keep stale-output cleanup narrowly scoped to files that this runtime itself owns.
  Rationale: AGENTS.md asks for minimal changes. The runner owns `training_summary.json`, `history.json`, `checkpoint_best.pt`, and `predictions/`; clearing only those files is enough to make retries authoritative without introducing broader directory deletion behavior.
  Date/Author: 2026-03-25 / assistant

- Decision: validate normalization coverage by reusing `load_points_normalization_mean_std(...)` with a subset `LayerPlan` instead of parsing `train_stats.json` in a second code path.
  Rationale: this keeps one source of truth for the band-level normalization contract while still enforcing the scoped requirement that only the requested covariate view must validate before execution starts.
  Date/Author: 2026-03-25 / assistant

## Outcomes & Retrospective

This follow-up fixed all four scoped baseline + iSSF runtime issues. Planning and status no longer crash on invalid point-store manifests, artifact resolution now blocks iSSF runs whose normalization stats are incomplete for the requested band set, the runner retries blocked/running latest statuses instead of skipping them forever, and stale runtime-owned files are cleared before each new attempt for the same `run_key`.

The targeted proof for this plan is the passing test sequence:

    python -m unittest -v tests.bench_v2.test_artifact_resolution
    python -m unittest -v tests.bench_v2.test_runner
    python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan
    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_runner

Residual scope remains exactly where intended: deepSSF and MoveFormer executors, evaluation/reporting, and any future explicit rerun CLI control still belong to later plans.

## Context and Orientation

The scoped baseline + iSSF runtime lives primarily in `movebench/bench_v2/run.py`, `movebench/bench_v2/methods/baselines.py`, `movebench/bench_v2/methods/issf.py`, and `movebench/bench_v2/train.py`. The runner resolves "external artifacts" from the study directory before dispatching an executor. In this repository, an external artifact means an on-disk file created by a different pipeline step, such as a point-store `manifest.json`, point-store `shared.pt`, train normalization stats JSON, or a deepSSF WebDataset cache manifest plus shard directory.

The relevant external validation logic already exists in repository helpers. `movebench/features/manifest.py` validates point-store manifests and confirms referenced files exist. `movebench/datasets/deepssf_wds_cache.py` validates deepSSF cache manifests. `movebench/features/normalization.py` validates that a normalization stats JSON contains every requested band. The current bug is not that these validators are missing; it is that the runner uses them inconsistently, so some malformed artifacts raise uncaught exceptions during readiness checks while others fail later inside execution.

The main tests for this area are in `tests/bench_v2/test_artifact_resolution.py`, `tests/bench_v2/test_runner.py`, and `tests/bench_v2/test_status.py`. `tests/bench_v2/test_plan.py` also matters because planning uses the same readiness helpers as status and run. The follow-up should keep the existing scope: baselines, iSSF, and the shared runner only. Do not implement deepSSF or MoveFormer executors while doing this work.

## Plan of Work

Begin with tests. In `tests/bench_v2/test_artifact_resolution.py`, add failing cases that build a syntactically present but semantically invalid point-store manifest, a syntactically present but semantically invalid deepSSF cache manifest, and a present normalization stats file that is missing one of the requested bands. These tests should assert that the resolver returns `{"status": "blocked", ...}` with a concise reason, not that it raises. Keep the fixtures small and local to the existing fake-study helpers already used in the file.

Next, extend `tests/bench_v2/test_runner.py` to cover recovery semantics. Add one scenario where the latest MLflow status is `blocked` and the underlying artifact problem has been repaired before the rerun; the test should prove that `run_experiment(...)` executes the run instead of skipping it. Add another scenario where the latest status is `running` but the runner is invoked again for the same `run_key`; this follow-up should treat that stale status as retryable and execute the run. Keep the existing "completed reruns are skipped" assertion intact so the skip policy becomes: skip only completed by default. In the same test file, add a stale-output scenario where the run directory already contains an old checkpoint or prediction artifact before a retry, and assert that those files do not survive if the new attempt is blocked or failed.

Because the readiness helpers are shared, also add a narrow test in `tests/bench_v2/test_status.py` or `tests/bench_v2/test_plan.py` proving that a malformed manifest no longer crashes status or planning. The test does not need to duplicate runner behavior; it only needs to show that the snapshot or plan marks the study-timescale pair blocked and keeps producing output.

After the failing tests exist, patch `movebench/bench_v2/run.py`. Start with `_point_store_state(...)` and `_deepssf_cache_state(...)`. Catch validator exceptions from `read_manifest(...)` and `read_deepssf_wds_manifest(...)` and convert them into unavailable states with concise blocked reasons such as `invalid point store manifest for train: ...` or `invalid deepSSF cache manifest for test-temporal: ...`. Do not add broad fallback behavior or rebuild logic. This is a system boundary that reads external disk state, so validation belongs here.

Still in `movebench/bench_v2/run.py`, tighten normalization handling inside `_resolve_point_artifacts(...)`. The resolver should continue to be read-only, but it must validate that the stats file covers the requested covariate view before execution starts. Reuse the existing normalization reader rather than duplicating its band-checking logic in a second format parser. If the stats file is missing or incomplete for the requested band set, return a blocked result immediately and do not let the run start.

Then fix retry semantics in `run_experiment(...)`. The scoped behavior should be simple: skip only when the latest run for a `run_key` has `status=completed`. Treat `blocked`, `failed`, and `running` as retryable so a repaired prerequisite or interrupted process can be retried by just invoking `movebench bench run ...` again. Do not add a new CLI option in this plan. Update the tests to make this policy explicit.

Finally, make each new execution attempt authoritative on disk. Before dispatching an executor or writing a new blocked summary, remove stale `training_summary.json`, `history.json`, `checkpoint_best.pt`, and the `predictions/` directory from the run directory if they exist. Leave `resolved_run.json` in place to be overwritten with the current canonical payload. This cleanup must happen for both retryable training runs and newly blocked runs so later tooling never mistakes old artifacts for the newest attempt's outputs.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the new failing tests first. Run the smallest relevant command after each group of tests is added:

    python -m unittest -v tests.bench_v2.test_artifact_resolution

Then add and run the recovery and stale-output runner tests:

    python -m unittest -v tests.bench_v2.test_runner

Add the malformed-manifest planning or status test and run the narrow command that covers it:

    python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan

Only after those tests exist, patch `movebench/bench_v2/run.py` and rerun each milestone command. When the file changes are complete, run the full scoped suite used by the current baseline + iSSF runtime:

    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_runner

If one of the existing tests fails because its fake `movebench.features.points` module no longer exposes every symbol imported by `movebench/bench_v2/run.py`, update the test fixture so it models the production import surface. That fixture maintenance is in scope because it is required to verify the follow-up behavior.

## Validation and Acceptance

Acceptance for this follow-up is reached when the following behaviors are all observable in tests.

First, a malformed point-store manifest, malformed deepSSF cache manifest, or incomplete normalization stats JSON does not raise out of `expand_experiment(...)`, `status_snapshot()`, or `_resolve_artifacts_for_run(...)`. Instead, the affected study-timescale pair or run is marked blocked with a concise reason that names the invalid artifact type.

Second, re-running the scoped baseline + iSSF runtime skips a `run_key` only when the latest MLflow run for that key is completed. If the latest status is blocked, failed, or running, the runner executes a new attempt for that same `run_key`.

Third, when a run is retried, the run directory reflects only the newest attempt. After a blocked or failed retry, there must be no leftover `checkpoint_best.pt`, stale `history.json`, or stale prediction files from an older attempt.

Fourth, the existing baseline + iSSF smoke behavior still works after these changes. The current runner test should still prove that baselines write zero-step prediction artifacts, iSSF writes a checkpoint and history, and rerunning after a completed attempt still skips execution.

## Idempotence and Recovery

This follow-up must remain read-only with respect to external artifacts. If a manifest or normalization stats file is bad, rerunning the command must continue to produce a blocked result until the artifact on disk is repaired by the external extraction pipeline. The runtime must never regenerate a point store, a deepSSF cache, or normalization stats as part of this plan.

The runner cleanup step must be safe to repeat. Deleting only runtime-owned outputs from `results/bench_v2/runs/<run_key>/` is idempotent: if the file or directory is already absent, the cleanup is a no-op; if it exists, removing it prevents stale-output confusion on the next attempt. Do not delete unrelated files under `results/bench_v2/`, and do not change the run-key scheme.

## Artifacts and Notes

The most important evidence for this plan will be short test transcripts showing the new recovery behaviors. Capture concise examples here as implementation proceeds, such as a runner test that first records a blocked run, then executes successfully after the artifact issue is repaired, and a stale-output test proving that an old checkpoint file is removed before a retry writes a new summary.

The expected runtime-owned files remain:

    results/bench_v2/runs/<run_key>/
      resolved_run.json
      training_summary.json
      history.json            # trainable runs only
      checkpoint_best.pt      # trainable runs only
      predictions/

For blocked runs in this scoped runtime, the authoritative artifact should remain `training_summary.json` plus the MLflow status tags for that attempt. Old checkpoints or predictions from a different attempt must not remain beside that blocked summary.

## Interfaces and Dependencies

Keep the existing public interfaces unchanged unless the change is strictly required for the scoped fixes. `run_experiment(...)`, `status_snapshot()`, `expand_experiment(...)`, and `_resolve_artifacts_for_run(...)` should keep their current call signatures. The behavior change is in their return values and retry policy, not in new parameters.

In `movebench/bench_v2/run.py`, the following end-state behavior must hold:

    _point_store_state(study_id, timescale_key, split) -> dict[str, object]

returns `{"available": False, "reason": ...}` for missing or invalid point artifacts rather than raising.

    _deepssf_cache_state(study_id, timescale_key, split) -> dict[str, object]

returns `{"available": False, "reason": ...}` for missing or invalid deepSSF cache artifacts rather than raising.

    _resolve_point_artifacts(run_payload, method=..., timescale=...) -> dict[str, object]

returns `{"status": "blocked", "reason": ...}` when normalization stats are missing or incomplete for the requested `band_ids`, before any executor runs.

    run_experiment(experiment_module, studies_override=None, methods=None, max_runs=None) -> dict[str, object]

skips only completed runs by default and retries blocked, failed, and running ones.

Plan change note (2026-03-25 15:22Z): created this follow-up ExecPlan after the scoped review found four baseline + iSSF runtime recovery defects that are small enough to fix without reopening the broader deepSSF and MoveFormer work.
