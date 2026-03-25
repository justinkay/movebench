# Fix deepSSF Runtime Validation and Early-Stopping Coverage

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md) and complements, but does not replace, [execplans/benchmark_rewrite_method_runtime_scoped_followup_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_scoped_followup_execplan.md). It intentionally covers only the deepSSF runtime path from the scoped method-runtime pass. Baselines, iSSF, MoveFormer, evaluation/reporting, extraction, and cache building are out of scope here.

## Purpose / Big Picture

After this change, the scoped `movebench bench run ...` path for `deepssf` will reject malformed or incompatible cache artifacts before execution starts, instead of failing later inside the executor after an MLflow run has already begun. The deepSSF training loop will also use a deterministic held-out evaluation subset for repeated early stopping, which was required by the original runtime ExecPlan but never implemented.

The visible proof is a small set of targeted tests that show three behaviors. First, a deepSSF cache directory that has no real shard files is marked blocked during readiness and artifact resolution. Second, a covariate subset whose requested channel indices exceed the cache manifest’s `env_channels` is marked blocked before the executor loads. Third, the deepSSF executor repeatedly evaluates a fixed prefix of the held-out cache for early stopping while still writing full prediction artifacts for every requested evaluation split.

## Progress

- [x] (2026-03-25 18:53Z) Reviewed the deepSSF follow-up findings and narrowed scope to four items: shard-file validation, cache-channel compatibility validation, deterministic early-stopping subset behavior, and missing targeted tests.
- [x] (2026-03-25 20:26Z) Added failing planner, artifact-resolution, and runner tests for broken deepSSF shard directories, incompatible `env_channels`, and blocked-before-executor deepSSF runs.
- [x] (2026-03-25 20:26Z) Centralized deepSSF shard discovery so readiness checks and the real loader enforce the same on-disk shard contract.
- [x] (2026-03-25 20:26Z) Hardened deepSSF cache resolution to block subset runs whose requested channel indices are not representable by the cache manifest.
- [x] (2026-03-25 20:48Z) Implemented bounded deterministic repeated evaluation in the deepSSF executor while leaving prediction artifact generation on full loaders.
- [x] (2026-03-25 20:52Z) Passed the deepSSF runtime test milestone, the combined deepSSF-targeted suite, and the affected `bench_v2` regression surface in the `movebench` micromamba environment.

## Surprises & Discoveries

- Observation: the current readiness helper and per-run resolver only check whether `shards/` is non-empty, but the real WebDataset loader only reads files matching `shard-*.tar*`.
  Evidence: `movebench/bench_v2/run.py` currently uses `any(shards_dir.iterdir())`, while `movebench/datasets/deepssf_wds_cache.py` builds its URL list from `shards_dir.glob("shard-*.tar*")`.

- Observation: the deepSSF cache manifest already carries `env_channels`, but the scoped runtime never compares that value to the requested covariate-subset view.
  Evidence: `build_deepssf_wds_cache(...)` writes `env_channels` into `manifest.json`, and the current resolver computes `channel_indices` from the timescale layer plan alone.

- Observation: the existing deepSSF runtime test proves channel slicing and output writing, but it does not prove that early stopping uses a deterministic subset instead of the full held-out cache.
  Evidence: `tests/bench_v2/test_deepssf_runtime.py` currently asserts output files and `n_valid` logging only.

- Observation: the current artifact-resolution tests import `movebench/bench_v2/run.py` through a fake `movebench.datasets.deepssf_wds_cache` module, so they cannot catch divergences from the real shard-file matching logic.
  Evidence: `tests/bench_v2/test_plan.py::_load_module(...)` injects a fake `movebench.datasets.deepssf_wds_cache` module before importing `movebench.bench_v2.run`.

- Observation: importing the production deepSSF cache module directly in tests also requires bypassing `movebench/__init__.py`, because that package imports `Study`, which in turn imports raster-backed dataset code that is not needed for these runtime tests.
  Evidence: the first attempt to use the real cache module in tests failed on a `rasterio`/`libgdal` ABI import error before `movebench.bench_v2.run` even loaded.

## Decision Log

- Decision: create a separate deepSSF follow-up ExecPlan instead of extending the baseline+iSSF follow-up file.
  Rationale: the existing follow-up plan explicitly excludes deepSSF scope, and mixing the two would make the plan misleading for the next contributor.
  Date/Author: 2026-03-25 / assistant

- Decision: reuse the cache manifest’s `env_channels` value as the compatibility check for covariate-subset channel slicing.
  Rationale: this adds essentially no runtime cost, because the resolver already reads every manifest. It is enough to catch stale or incompatible superset caches before execution without opening shard payloads.
  Date/Author: 2026-03-25 / assistant

- Decision: fix shard validation by sharing one shard-discovery helper between `movebench/bench_v2/run.py` and `movebench/datasets/deepssf_wds_cache.py`.
  Rationale: two different filename rules caused the current bug. Reusing one helper is the smallest way to keep readiness and loading behavior aligned.
  Date/Author: 2026-03-25 / assistant

- Decision: implement the deterministic early-stopping subset as a fixed prefix of the non-shuffled cache loaders, not as an in-memory materialized tensor cache.
  Rationale: deepSSF samples are large enough that materializing thousands of cached examples would create unnecessary memory pressure. Re-reading only a bounded deterministic prefix at each evaluation keeps memory flat and preserves the intended early-stopping behavior.
  Date/Author: 2026-03-25 / assistant

- Decision: keep the follow-up free of new CLI flags or experiment-schema refactors.
  Rationale: this is a correctness and completeness pass over the existing deepSSF runtime, not a redesign of `bench_v2` configuration. The executor can carry a small internal evaluation cap that mirrors the old benchmark default until a broader runtime-level eval-budget design exists.
  Date/Author: 2026-03-25 / assistant

- Decision: in tests, load the real `movebench/datasets/deepssf_wds_cache.py` file under a lightweight fake `movebench` package shell instead of importing the whole production package tree.
  Rationale: this exercises the real shard-discovery and manifest-validation logic without pulling in unrelated raster-backed dependencies that are not needed for `bench_v2` runtime tests.
  Date/Author: 2026-03-25 / assistant

## Outcomes & Retrospective

deepSSF cache validation now blocks the two resolver failures that motivated this follow-up. `movebench/bench_v2/run.py` and `movebench/datasets/deepssf_wds_cache.py` share the same shard discovery helper, so caches whose `shards/` directory lacks `shard-*.tar*` files are blocked consistently by planning, status, and per-run artifact resolution. The resolver also compares the requested covariate-subset `channel_indices` against each split manifest’s `env_channels`, so incompatible subset views are blocked before executor dispatch.

The deepSSF executor now bounds repeated evaluation through an internal `DEFAULT_EVAL_MAX_ITEMS` budget. Early-stopping evaluation uses a deterministic prefix of the non-shuffled loaders, while post-fit prediction writing still iterates over the full requested evaluation splits and writes complete prediction tensors.

The targeted proof points are:

- `tests.bench_v2.test_artifact_resolution` now covers missing loader-visible shard files and `env_channels` incompatibility using the real deepSSF cache module logic.
- `tests.bench_v2.test_runner` now proves a deepSSF run with incompatible cache channels is marked blocked before the executor is loaded.
- `tests.bench_v2.test_deepssf_runtime` now proves bounded deterministic eval counts while full prediction artifacts are still written.
- `tests.bench_v2.test_plan` and `tests.bench_v2.test_status` exercise the shared shard-validation behavior through the planning/status surfaces.

Observed validation runs:

- `2026-03-25 20:48Z`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_deepssf_runtime` -> `Ran 2 tests ... OK`
- `2026-03-25 20:49Z`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner` -> `Ran 13 tests ... OK`
- `2026-03-25 20:52Z`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner` -> `Ran 33 tests ... OK`

## Context and Orientation

The scoped deepSSF runtime lives in three main places. `movebench/bench_v2/run.py` performs study readiness checks, expands concrete run payloads, and resolves on-disk artifacts before dispatching an executor. `movebench/bench_v2/executors/deepssf.py` owns the deepSSF-specific training loop adapter, including channel slicing from cached spatial tensors, early stopping, checkpoint writing, and prediction artifact writing. `movebench/datasets/deepssf_wds_cache.py` defines the on-disk deepSSF cache layout and the real WebDataset loader behavior.

In this repository, a deepSSF cache is an external artifact under `studies/<study_id>/cache/deepssf_wds/<split>/<cache_hash>/`. Each cache has a `manifest.json` and a `shards/` directory. The manifest records grid metadata such as `output_hw`, `pixel_size_m`, `num_samples`, and `env_channels`. The shard directory is only valid for loading if it contains actual tar shards that match the real loader’s filename pattern. The runtime must treat these caches as read-only prerequisites. If the manifest is malformed, if the shard directory does not match the loader contract, or if a covariate subset requests channels outside the cache’s `env_channels`, the run must be marked blocked. It must not try to rebuild the cache.

The missing early-stopping behavior is narrower. The original method-runtime ExecPlan required a deterministic early-stopping subset for deepSSF, meaning that repeated evaluations during training should score the same bounded slice of the held-out split every time. The current executor evaluates the full non-shuffled held-out loader each time instead. That is deterministic, but it is not the required bounded subset, and it makes the training loop do more repeated work than the scoped design intended.

The main tests for this area are `tests/bench_v2/test_artifact_resolution.py`, `tests/bench_v2/test_deepssf_runtime.py`, `tests/bench_v2/test_runner.py`, and one of `tests/bench_v2/test_status.py` or `tests/bench_v2/test_plan.py`. The current artifact tests use fake modules to reduce dependencies. That is fine for general planning tests, but at least one new deepSSF artifact-resolution test in this follow-up must exercise the real shard-file discovery logic rather than the fake cache module.

## Plan of Work

Begin with tests so the failures are concrete. In `tests/bench_v2/test_artifact_resolution.py`, add one case where the cache directory exists and `manifest.json` is valid, but `shards/` contains only a non-shard file such as `README.txt`. The test should assert that `_resolve_artifacts_for_run(...)` returns `{"status": "blocked", ...}` with a reason that names missing deepSSF shards, not that the run resolves and fails later. Add a second case where the manifest reports `env_channels=1` but the requested covariate set resolves to many more channel indices. That test should assert that the run is blocked before the executor is invoked.

Because the existing `_load_module(...)` helper in `tests/bench_v2/test_plan.py` injects a fake deepSSF cache module, add a small import helper for this follow-up that keeps fake `mlflow` and fake `Study` state if needed but leaves `movebench.datasets.deepssf_wds_cache` real. Use that helper for the new shard-pattern and `env_channels` tests so the test actually exercises the production shard-discovery rules.

Next, extend `tests/bench_v2/test_deepssf_runtime.py` with a failing runtime test for the deterministic early-stopping subset. Build a fake train loader and a fake held-out loader that each contain more examples than the bounded subset should use. Keep the prediction split full. The test should show that repeated training-loop evaluations log `test-temporal/n_valid` for only the bounded deterministic prefix, while the final prediction artifact still contains every held-out example. Make the evaluation cap patchable in the module so the test can use a tiny bound such as two or three examples instead of the production default.

Add one runner-level test in `tests/bench_v2/test_runner.py` that proves invalid deepSSF caches are blocked before executor dispatch. Patch `_load_executor(...)` to raise if called. Then construct a runnable-looking `deepssf` experiment whose cache manifest is incompatible or whose shard directory is missing real shard files. The acceptance condition is that `run_experiment(...)` returns a blocked run, writes a blocked summary, and never attempts to import or call the executor.

After the failing tests exist, patch `movebench/datasets/deepssf_wds_cache.py`. Introduce one small helper such as `list_deepssf_wds_shards(cache_dir: str | Path) -> list[Path]` that returns the sorted shard files matching the real loader contract `shard-*.tar*`. Update `make_deepssf_wds_loader(...)` to use that helper instead of duplicating the glob inline. While in this file, tighten `read_deepssf_wds_manifest(...)` to require a positive integer `env_channels`, because the follow-up resolver will rely on that field for compatibility checks and the cache builder already writes it.

Then patch `movebench/bench_v2/run.py`. In `_deepssf_cache_state(...)`, use the shared shard-list helper instead of `any(shards_dir.iterdir())`, so readiness, planning, and status all agree with the loader about what a usable shard set looks like. In `_resolve_deepssf_cache_artifacts(...)`, after reading each manifest, compare the requested `channel_indices` to `manifest["env_channels"]`. If any index is greater than or equal to `env_channels`, return `{"status": "blocked", "reason": ...}` with a concise incompatibility message that names the split and the channel counts. Keep this check read-only and per-run: `_deepssf_cache_state(...)` cannot know the covariate subset, so it should not try to perform the band-view compatibility check there.

Finally, patch `movebench/bench_v2/executors/deepssf.py`. Add a small module-level constant for the default repeated-evaluation budget, using the old benchmark runtime’s `2500`-example default as the reference behavior. Thread an optional `max_items` or similar argument into the internal held-out evaluation helper so early-stopping evaluations score only the first fixed prefix of each non-shuffled evaluation loader. The prefix must be deterministic and must not require materializing the whole subset in memory. Keep full, uncapped loaders for prediction artifact writing after the best checkpoint is restored. The shared training loop does not need API changes; the deepSSF runtime object can keep the cap and apply it inside `evaluate(...)`.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the new failing deepSSF artifact-resolution tests first. Run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution

Then add the deterministic early-stopping subset runtime test and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_deepssf_runtime

Add the blocked-before-executor runner test and run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner

If the shared readiness helper changes affect `bench status` or planning, add the narrow follow-up test in `tests/bench_v2/test_status.py` or `tests/bench_v2/test_plan.py` and run the relevant command:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan

After the code changes are complete, rerun the combined deepSSF-targeted suite:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_artifact_resolution tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner

Then run the affected `bench_v2` regression surface:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_runner

As implementation progresses, replace these commands with short observed transcripts and note any deviations here.

## Validation and Acceptance

Acceptance for this follow-up is reached when the following behaviors are all observable in tests.

First, a deepSSF cache directory that has a valid-looking manifest but no files matching `shard-*.tar*` is treated as blocked by readiness and artifact resolution. `bench status`, planning, and per-run resolution must agree with the real loader’s shard contract instead of disagreeing on what counts as a usable cache.

Second, a covariate-subset deepSSF run whose requested channel view does not fit inside the cache manifest’s `env_channels` is blocked before executor dispatch. The run must not resolve successfully and then fail inside `index_select(...)`.

Third, repeated early-stopping evaluations inside `movebench/bench_v2/executors/deepssf.py` use a fixed deterministic prefix of the non-shuffled held-out split and log `n_valid` for that bounded subset. The prediction artifact written after best-checkpoint restore still contains the full requested evaluation split in cache iteration order.

Fourth, a runner-level deepSSF experiment with an invalid cache is marked blocked, writes a blocked `training_summary.json`, and does not attempt to load or call the deepSSF executor.

Fifth, the existing deepSSF smoke behavior still works after these changes. The current runtime test must still prove channel slicing, checkpoint writing, history writing, and prediction writing, and the runner test must still prove that `bench run` can execute a valid deepSSF run end to end.

## Idempotence and Recovery

This follow-up must remain read-only with respect to deepSSF caches. If a cache manifest is malformed, if the shard directory is missing real shard files, or if a covariate subset requests more channels than the cache contains, rerunning the command must continue to block until the cache on disk is repaired by the existing extraction pipeline. This plan must not add any cache rebuilding, cache migration, or shard renaming behavior.

The deterministic early-stopping subset must also be idempotent. Running the same deepSSF experiment twice against the same cache with the same seed and batch size should evaluate the same held-out prefix during training-loop early stopping. The prefix may be a simple first-`N` view over the non-shuffled loader; it does not need random sampling as long as it is stable across evaluations and reruns.

## Artifacts and Notes

The most important evidence for this plan will be concise test transcripts and a blocked-run summary example. Capture examples such as a resolver test that reports `missing deepSSF shards` for a cache containing only `README.txt`, or a runner test that proves `_load_executor(...)` is never called when `env_channels` is too small for the requested subset.

The relevant runtime-owned files remain:

    results/bench_v2/runs/<run_key>/
      resolved_run.json
      training_summary.json
      history.json
      checkpoint_best.pt
      predictions/

For this follow-up, the important distinction is that a blocked deepSSF run must stop before executor dispatch and therefore should only write the blocked summary. A completed deepSSF run should continue to write checkpoint, history, and prediction artifacts exactly as before.

## Interfaces and Dependencies

Keep public CLI and runner signatures unchanged. `movebench bench run`, `run_experiment(...)`, `expand_experiment(...)`, and `_resolve_artifacts_for_run(...)` should keep their current call signatures. The changes in this plan are to validation behavior, one small shared loader helper, and deepSSF executor internals.

In `movebench/datasets/deepssf_wds_cache.py`, define:

    list_deepssf_wds_shards(cache_dir: str | Path) -> list[Path]

This helper must return the sorted shard files matching the production loader contract and must be used by `make_deepssf_wds_loader(...)`. `read_deepssf_wds_manifest(...)` must also validate that `env_channels` is present and greater than zero.

In `movebench/bench_v2/run.py`, the following end-state behavior must hold:

    _deepssf_cache_state(study_id, timescale_key, split) -> dict[str, object]

returns `{"available": False, "reason": ...}` when the cache has no loader-visible shard files, not just when `shards/` is empty.

    _resolve_deepssf_cache_artifacts(run_payload, timescale=...) -> dict[str, object]

returns `{"status": "blocked", "reason": ...}` when any requested channel index is outside the cache manifest’s `env_channels` for any required split.

In `movebench/bench_v2/executors/deepssf.py`, the internal evaluation helper should accept a bounded repeated-evaluation budget and stop after a deterministic prefix of each non-shuffled evaluation loader. This budget should be kept internal to the executor in this follow-up. Prediction writing must continue to consume the full loaders.

Plan change note (2026-03-25 18:53Z): created this deepSSF-specific follow-up ExecPlan after review found two resolver correctness defects, one incomplete early-stopping acceptance criterion, and a gap in the targeted test surface that the existing baseline+iSSF follow-up plan did not cover.
