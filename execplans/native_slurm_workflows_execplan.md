# Add First-Class `pack` Commands and Native Slurm Launchers for Extract, Pack, and `bench`

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, a user will be able to run the deepSSF cache-packaging workflow directly from the main CLI as `movebench pack ...`, and submit the three batch-oriented workflows from the main CLI instead of jumping into `scripts/slurm/`: `movebench slurm extract ...`, `movebench slurm pack ...`, and `movebench slurm bench run ...`. The visible proof is that the Slurm commands support `--dry-run`, print the exact `srun` commands they would launch, and use the same underlying runtime code as the non-Slurm workflows instead of a second, drifting implementation.

Extraction is no longer an open design question in this plan. The prerequisite follow-up [execplans/extract_workflow_alignment_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/extract_workflow_alignment_execplan.md) already promoted the authoritative extract behavior into `movebench/extract.py`, and `movebench extract`, `movebench slurm extract`, and `scripts/slurm/extract.py` now delegate to that shared runtime. This broader plan therefore treats extract as a completed dependency and focuses the remaining implementation on the benchmark Slurm path while keeping the aligned extract and pack behavior green.

For benchmarking, Slurm support applies only to execution. Planning, status inspection, and report generation stay local as `movebench bench plan`, `movebench bench status`, and `movebench bench report`. The new Slurm benchmark path must therefore extend `movebench/bench/` instead of reviving the legacy YAML benchmark launcher in `scripts/slurm/bench.py`.

## Progress

- [x] (2026-03-27 18:01Z) Reviewed the current CLI entrypoints in `movebench/cli/main.py`, `movebench/cli/extract.py`, and `movebench/bench/run.py`.
- [x] (2026-03-27 18:01Z) Reviewed the legacy Slurm launchers in `scripts/slurm/extract.py`, `scripts/slurm/wds.py`, and `scripts/slurm/bench.py` to identify reusable behavior and drift from the current runtime.
- [x] (2026-03-27 18:01Z) Chose the initial Slurm CLI shape: `movebench slurm extract`, `movebench slurm wds`, and `movebench slurm bench run`, with no Slurm versions of `bench plan`, `bench status`, or `bench report`.
- [x] (2026-03-27 18:01Z) Chose the benchmark job unit: one concrete `bench` `run_key` per Slurm job, not one study per job.
- [x] (2026-03-27 18:01Z) Chose the cache-packaging scope: drive deepSSF cache building from `bench` timescale metadata and study split discovery, not from the legacy YAML benchmark config path.
- [x] (2026-03-28 18:28Z) Reviewed the split-tracking changes from `f58b5b2` and `92c41fe` and updated this plan to use the current `Study.split_inventory()` contract and canonical underscore-form split keys.
- [x] (2026-03-28 18:41Z) Revised the command surface so cache building is a first-class `movebench pack` workflow with `movebench slurm pack` as its batch wrapper, and reserved `wds` for internal artifact names only.
- [x] (2026-04-01 22:15Z) Updated this plan after the benchmark namespace cutover so active runtime references now point at `movebench/bench/`, `experiments/`, `tests/bench`, and `results/bench`.
- [x] (2026-04-01 22:38Z) Implemented the shared package-native Slurm launcher/runtime helpers and top-level CLI registration in `movebench/cli/main.py`, `movebench/cli/slurm.py`, and `movebench/slurm/`.
- [x] (2026-04-01 22:38Z) Implemented the first-class local `movebench pack` runtime and CLI in `movebench/cli/pack.py` and `movebench/pack.py`, including shared pack argument parsing, study/timescale queue resolution, cache skip detection, prerequisite checks, and per-job execution helpers.
- [x] (2026-04-02 18:45Z) Implemented an initial native `movebench slurm extract` and `movebench slurm pack` in `movebench/slurm/extract.py` and `movebench/slurm/pack.py`, including hidden worker mode, shared `srun` command construction, `--dry-run`, extract skip/block planning, and Slurm pack reuse of `movebench/pack.py`.
- [x] (2026-04-02 19:39Z) Reviewed the initial Milestone 3 extract implementation, identified parity drift, and split the corrective work into the prerequisite [execplans/extract_workflow_alignment_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/extract_workflow_alignment_execplan.md) so the broader plan could stop evolving against the wrong extract contract.
- [x] (2026-04-04 17:03Z) Revalidated the completed extract prerequisite: `movebench extract`, `movebench slurm extract`, and `scripts/slurm/extract.py` now share `movebench/extract.py`, and `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_cli_extract tests.test_slurm_extract` still passed (`Ran 9 tests in 31.488s`, `OK`).
- [x] (2026-04-04 17:03Z) Updated this broader plan to treat extract as complete background context and narrowed the remaining work to the shared benchmark helpers plus native `movebench slurm bench run`.
- [x] (2026-04-04 17:18Z) Added `tests/test_slurm_bench.py` and extended `tests/bench/test_runner.py`, then ran `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner` and captured the expected failing baseline: the Slurm bench parser does not accept `--studies` or worker flags yet, `movebench/bench/run.py` has no `resolve_pending_runs(...)` or `load_resolved_run(...)`, and `run_experiment(...)` still bypasses the planned shared helper boundary.
- [x] (2026-04-04 17:23Z) Refactored `movebench/bench/run.py` so local execution and Slurm execution now share `resolve_pending_runs(...)`, `load_resolved_run(...)`, and `execute_run_payload(...)`, with manifest path/payload resolution centralized alongside pending-run selection.
- [x] (2026-04-04 17:23Z) Implemented native `movebench slurm bench run` in `movebench/slurm/bench.py`, including launcher mode, worker mode, hidden `--manifest-path` / `--run-key` flags, method-specific GPU defaults, `--dry-run`, and manifest writing only for real launches.
- [x] (2026-04-01 22:38Z) Added `tests/test_pack.py` and ran `python -m unittest -v tests.test_pack` to verify the Milestone 1 parser/launcher skeleton and the Milestone 2 local pack planner/executor.
- [x] (2026-04-04 17:23Z) Reran the benchmark-focused targeted suite after implementation: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner` passed (`Ran 8 tests`, `OK`).
- [x] (2026-04-04 17:23Z) Reran the broader targeted suites after implementation: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_pack tests.test_cli_extract tests.test_slurm_extract tests.test_slurm_pack tests.test_slurm_bench` passed (`Ran 25 tests`, `OK`), and `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_experiment_config tests.bench.test_runner` passed (`Ran 7 tests`, `OK`).
- [x] (2026-04-04 17:23Z) Recorded the repo-wide verification result in `/tmp/full_unittest_movebench_20260404_native_slurm_bench.log`: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v` still failed in unrelated pre-existing areas (`Ran 186 tests in 1.929s`, `FAILED (failures=5, errors=95)`).
- [x] (2026-04-04 17:23Z) Ran a real CLI dry-run with local data: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main slurm bench run experiments/smoke_one_study.py --studies Accio_967b7813_raw_merged --dry-run` returned `planned=3 pending=0 skipped=0 blocked=3`, which matches the current environment’s lack of fully runnable benchmark pairs.
- [x] (2026-04-02 18:45Z) Added `tests/test_slurm_extract.py` and `tests/test_slurm_pack.py`, confirmed they failed against the placeholder implementations, then reran `python -m unittest -v tests.test_slurm_extract tests.test_slurm_pack` with the MoveBench micromamba environment after implementation and reviewed the passing output.

## Surprises & Discoveries

- Observation: the legacy Slurm scripts are three copies of the same launch-and-poll loop, not a shared runtime.
  Evidence: `scripts/slurm/extract.py`, `scripts/slurm/wds.py`, and `scripts/slurm/bench.py` each define their own `subprocess.Popen` queue, polling interval loop, progress printouts, and interruption handling.

- Observation: the legacy benchmark and cache-related Slurm launchers still depend on the old YAML benchmark path.
  Evidence: `scripts/slurm/bench.py` and `scripts/slurm/wds.py` both import `movebench.benchmark.load_config` and `resolve_studies`, while the active benchmark runtime is now under `movebench/bench/`.

- Observation: `bench` already has the correct unit of work for Slurm execution.
  Evidence: `movebench/bench/run.py` expands experiments into concrete run payloads, assigns stable `run_key` values, and executes runs one payload at a time through `_execute_run(...)`.

- Observation: split tracking is now centralized in `Study.split_inventory()`, and split keys are canonicalized to underscore-form names such as `test_temporal` and `test_individual`.
  Evidence: `movebench/study.py` reads `manifest.json` `trajectories.split_keys` and `trajectories.split_counts` before falling back to `trajectories.csv`, and `movebench/trajectories.py` normalizes split keys with `canonicalize_split_key(...)`.

- Observation: the legacy extract and cache-related Slurm scripts have already been partially updated for the newer split model, so they are no longer evidence of `train`/`test`-only behavior.
  Evidence: `scripts/slurm/extract.py` now derives default splits from `Study.split_inventory()`, and `scripts/slurm/wds.py` defaults `--splits` to the study split inventory.

- Observation: there is already a local, non-Slurm deepSSF cache builder, but it lives in `scripts/` and mixes cache building with ad hoc throughput benchmarking.
  Evidence: `scripts/deepssf_cache_wds.py` can build caches for studies and splits, but it is not registered in `movebench/cli/main.py` and also contains benchmarking helpers such as `_bench_loader(...)` and `_bench_train(...)`.

- Observation: the legacy local cache script is stale in one pack-prerequisite detail: it imports a helper that no longer exists in the active points module.
  Evidence: importing `select_point_store_dir` from `movebench.features.points` now raises `ImportError`, while the current exported path helpers are `point_store_dir(...)` and `point_store_train_stats_path(...)`.

- Observation: `movebench/pack.py`’s shared argument registration makes `--studies` and `--timescales` required, which conflicts with Slurm worker mode even though the worker should reuse the same parsing and execution helpers.
  Evidence: the first pre-implementation run of `python -m unittest -v tests.test_slurm_extract tests.test_slurm_pack` failed in `tests.test_slurm_pack.SlurmPackTests.test_worker_executes_shared_pack_job` because the parser exited before hidden worker flags could be handled.

- Observation: extract-specific design guidance in this broader plan became stale once the prerequisite alignment work landed.
  Evidence: `movebench/extract.py` now owns `build_extraction_runs(...)`, `resolve_requested_studies(...)`, `default_splits(...)`, `is_extract_complete(...)`, `uses_gee_layers(...)`, and `run_extract_job(...)`, while `movebench/cli/extract.py`, `movebench/slurm/extract.py`, and `scripts/slurm/extract.py` are thin wrappers over that shared runtime.

- Observation: GEE raster prerequisite detection for extract planning must consider derived layer dependencies, not only `plan.layers_in_order`.
  Evidence: `movebench.layers.plan.LayerPlan` records `derived_dependencies`, and other Slurm planning code already walks those parents before deciding whether a plan depends on GEE-backed rasters.

- Observation: the benchmark-focused tests were specific enough to catch the missing parser and helper boundaries immediately, and they passed without any follow-up redesign once those boundaries were implemented.
  Evidence: the pre-change `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner` run failed with parser and missing-helper errors, while the post-change rerun passed (`Ran 8 tests`, `OK`).

- Observation: method-specific GPU defaults required overriding the shared Slurm parser default only for the benchmark command.
  Evidence: `register_common_slurm_arguments(...)` defaults `gpus_per_node` to `0`, which would suppress the desired per-method defaults; `movebench/slurm/bench.py` now sets the benchmark parser default to `None` and fills in `0` or `1` GPUs per run based on the method family unless the user passes `--gpus-per-node`.

- Observation: the current local data root contains zero fully runnable benchmark pairs, so a real `movebench slurm bench run ... --dry-run` can legitimately report no pending jobs even though the command is implemented correctly.
  Evidence: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench status --format text` reported `Fully runnable pairs: 0`, and the real dry-run for `Accio_967b7813_raw_merged` printed `[bench] planned=3 pending=0 skipped=0 blocked=3` followed by `[bench] No pending runs.`.

## Decision Log

- Decision: add a top-level `slurm` CLI group rather than adding `--executor slurm` flags to the existing commands.
  Rationale: the user explicitly wants to place `slurm` before `extract` and `bench`, and a dedicated group cleanly separates local workflows from batch submission workflows.
  Date/Author: 2026-03-27 / assistant

- Decision: keep the Slurm orchestration in package code under `movebench/slurm/` and treat `scripts/slurm/*.py` as temporary reference material.
  Rationale: package-native modules can be imported, tested, and registered in the CLI directly; continuing to shell out to repo scripts would preserve the current duplication and drift.
  Date/Author: 2026-03-27 / assistant

- Decision: benchmark Slurm jobs will launch one concrete `run_key` per job.
  Rationale: `bench` is already organized around run payloads and per-run artifacts. Re-aggregating by study would reintroduce legacy behavior that fights the current refactor and makes resource assignment less precise.
  Date/Author: 2026-03-27 / assistant

- Decision: `movebench slurm bench` will expose only `run`.
  Rationale: the user only wants Slurm for experiment execution. Planning, status inspection, and report generation are metadata queries and should stay local.
  Date/Author: 2026-03-27 / assistant

- Decision: the first-class cache-building workflow will be named `pack`, with command surfaces `movebench pack` and `movebench slurm pack`.
  Rationale: `wds` is implementation jargon for the internal artifact format. `pack` communicates the user-visible job more clearly while still allowing the code and on-disk layout to keep the existing `deepssf_wds` names.
  Date/Author: 2026-03-28 / assistant

- Decision: the native pack commands will accept `--timescales` and `--studies` directly, and they will resolve cache hashes from `movebench/bench/config.py`.
  Rationale: deepSSF caches are timescale-specific study artifacts, not experiment-specific artifacts. Requiring an experiment module would add coupling without adding capability.
  Date/Author: 2026-03-27 / assistant

- Decision: default benchmark Slurm resources will be chosen per method family, with CLI flags allowed to override them.
  Rationale: `deepssf` and `moveformer` should default to GPU-backed jobs, while baselines and iSSF jobs should not consume GPUs by default. A single global default for mixed experiments would waste cluster resources.
  Date/Author: 2026-03-27 / assistant

- Decision: reuse `Study.split_inventory()` as the shared split-discovery contract instead of introducing new `split_keys()` or `held_out_split_keys()` helpers.
  Rationale: the recent ingest and study changes already record canonical split metadata in `manifest.json` and normalize split names centrally; a second API would duplicate that contract without adding capability.
  Date/Author: 2026-03-28 / assistant

- Decision: implement native pack prerequisite detection against `point_store_dir(...)` and `point_store_train_stats_path(...)` instead of trying to preserve the legacy `select_point_store_dir(...)` path.
  Rationale: the current `movebench.features.points` module no longer exports `select_point_store_dir(...)`, so the native runtime should validate prerequisites through the active path helpers that correspond to the current point-store and normalization artifacts.
  Date/Author: 2026-04-01 / assistant

- Decision: keep Slurm worker identity in hidden flags (`--worker-study`, `--worker-timescale`, `--worker-layers-config`, `--worker-sampler-config`) while reusing the visible `--splits` and pack build-option flags for both launcher and worker parsing.
  Rationale: this keeps the worker commands short, lets explicit split/build overrides round-trip through the same canonicalization code as the user-facing CLI, and avoids a second set of worker-only option names for settings that already exist.
  Date/Author: 2026-04-02 / assistant

- Decision: relax `--studies` and `--timescales` from “argparse-required” to “launcher-required” only inside `movebench/slurm/pack.py`, without changing `movebench/pack.py`.
  Rationale: local `movebench pack` should keep its strict user-facing parser contract, but `movebench slurm pack` still needs to accept hidden worker mode that identifies the job with `--worker-study` and `--worker-timescale` instead of the visible selection flags.
  Date/Author: 2026-04-02 / assistant

- Decision: this broader plan will no longer reopen extract semantics. The authoritative extract contract now lives in `movebench/extract.py` and is maintained by the prerequisite [execplans/extract_workflow_alignment_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/extract_workflow_alignment_execplan.md).
  Rationale: keeping two plans independently redefining extract behavior caused drift. The shared extract runtime is now in place, so this plan should treat extract as completed context and only guard it with regression tests while the benchmark Slurm work proceeds.
  Date/Author: 2026-04-04 / assistant

- Decision: native Slurm benchmark execution will reuse `movebench/bench/run.py` by exposing public “resolve pending runs”, “load one resolved run”, and “execute one run payload” helpers rather than adding a second planner in `movebench/slurm/bench.py`.
  Rationale: the current benchmark runtime already knows how to expand experiments, derive stable `run_key` values, and execute one run payload safely. Rebuilding that logic in the Slurm module would repeat the same mistake that the first extract implementation made.
  Date/Author: 2026-04-04 / assistant

- Decision: `movebench slurm bench run --dry-run` must compute and print the resolved manifest path without writing the manifest file.
  Rationale: the plan’s global `--dry-run` contract is “no mutation”. The launcher still needs a stable path to show in the printed worker command, so the manifest path must be derivable from the resolved payloads even when the file is not written.
  Date/Author: 2026-04-04 / assistant

## Outcomes & Retrospective

This plan started as the design and implementation blueprint for the native Slurm and pack work. It resolved the main architecture questions that mattered before writing code: where the CLI should live, what the benchmark job boundary should be, how cache packaging should relate to `bench`, and why the legacy `scripts/slurm/` files should not become the permanent runtime. On 2026-03-28 it was revised first to reflect the current split contract, and then to make cache packaging a first-class `pack` workflow with a local CLI and a Slurm wrapper. On 2026-04-01, Milestones 1 and 2 landed and were validated with `tests.test_pack`, which means the first user-visible slice now exists: `movebench pack ...` is registered, plans work from timescale metadata, skips complete caches, and executes pending study/timescale jobs locally, while the top-level `slurm` parser tree and shared launcher helper are in place. On 2026-04-02 through 2026-04-04, the extract milestone was pulled into the prerequisite [execplans/extract_workflow_alignment_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/extract_workflow_alignment_execplan.md) and completed there; extract is now shared through `movebench/extract.py` and revalidated in the current worktree. Later on 2026-04-04, Milestones 5 and 6 landed: `movebench/bench/run.py` now exposes shared pending-run and worker helpers, `movebench/slurm/bench.py` is a real launcher/worker instead of a stub, the targeted benchmark and CLI suites pass, and the full unittest record still fails only in unrelated pre-existing areas.

## Context and Orientation

The current top-level CLI lives in `movebench/cli/main.py`. It already registers `download`, `extract`, `ingest`, `pack`, `slurm`, and `bench`. `movebench/extract.py` owns the shared extraction workflow, while `movebench/cli/extract.py` and `movebench/slurm/extract.py` are thin local and Slurm wrappers over that runtime. `movebench/pack.py` owns the shared local and Slurm pack planner/executor helpers. `movebench/bench/run.py` owns the benchmark runtime and now also exposes the shared pending-run and worker helpers used by both local `bench run` and native Slurm execution. `movebench/slurm/bench.py` is the native benchmark launcher and worker. At this point the command surface described by this plan exists; the remaining value of the plan is as a self-contained record of the design, validation steps, and residual environment limitations.

In this plan, “pack” means building the deepSSF WebDataset cache shards and manifest files that live under each study’s `cache/deepssf_wds/` directory. The internal code and on-disk paths can keep the existing `deepssf_wds` name because that is the artifact format, but the user-facing CLI should call this workflow `pack`.

In this repository, a “Slurm launcher” means the process that runs on the login node, decides what work is still needed, and spawns one or more `srun` commands while throttling the number of concurrent jobs. A “worker” means the code that runs inside the `srun` job itself and performs exactly one unit of work.

For extraction, the natural work unit is one study plus one layer-config/sampler-config combination. For cache packaging, the natural work unit is one study plus one timescale, because the cache hash is determined by the timescale’s raster hash, chip size, normalization settings, and split key. For `bench`, the natural work unit is one concrete benchmark run payload, identified by `run_key`.

The legacy scripts are useful references for CLI flags and for the queue/poll output style, but they are not aligned with the current codebase:

- `scripts/deepssf_cache_wds.py` is the closest existing local reference for `pack`, but it lives outside the main CLI and mixes cache building with benchmarking code that should not become part of the first-class command.
- `scripts/slurm/extract.py` never became part of the main CLI, even though it now derives default splits from `Study.split_inventory()`.
- `scripts/slurm/wds.py` now uses the study split inventory too, but it is still tied to legacy YAML benchmark configs rather than `movebench/bench/config.py`, and its user-facing name no longer matches the desired CLI.
- `scripts/slurm/bench.py` is entirely built around the deleted `movebench.benchmark` path, not `movebench/bench/`.

The important current files for the finished implementation are:

- `movebench/cli/main.py`, which already exposes the supported top-level command tree and must keep doing so after the benchmark refactor.
- `movebench/extract.py`, which is now the authoritative shared extract runtime and should only need regression coverage from this plan.
- `movebench/cli/pack.py` and `movebench/pack.py`, which already provide the first-class local `movebench pack` command plus the shared planner/executor used by `movebench slurm pack`.
- `movebench/bench/run.py`, which now exposes the shared helpers for “plan runs, choose pending runs, load one resolved run payload, execute one resolved run payload”.
- `movebench/slurm/bench.py`, which implements the native Slurm benchmark launcher and worker.
- `tests/bench/test_runner.py`, which covers the shared helper boundaries inside `movebench/bench/run.py`.
- `tests/test_slurm_bench.py`, which covers the launcher dry-run, worker reload path, and per-method resource defaults.
- `movebench/bench/config.py`, which is the authoritative source for timescale-specific layer configs, sampler configs, and deepSSF cache hashes.
- `movebench/study.py`, which already owns high-level study operations and now centralizes split discovery through `Study.split_inventory()`.
- `movebench/trajectories.py`, which canonicalizes split keys so user input like `test-temporal` and stored metadata like `test_temporal` resolve to the same on-disk paths.

## Plan of Work

Milestone 1 creates the shared Slurm runtime and the CLI skeleton. Add `movebench/cli/slurm.py` and register it from `movebench/cli/main.py`. Add `movebench/cli/pack.py` and register it from `movebench/cli/main.py`. The `slurm` group must contain `extract`, `pack`, and `bench`, with `bench` containing only `run`. Create package modules under `movebench/slurm/` for the shared launcher runtime and the three workflow-specific commands. Keep file count low: one shared launcher module plus one module per workflow, plus one shared pack runtime module, is enough.

The shared launcher module should define a small dataclass for Slurm resources and two reusable helpers. The first helper registers the common Slurm arguments on an `argparse` parser: `--partition`, `--qos`, `--account`, `--time`, `--mem`, `--cpus-per-task`, `--gpus-per-node`, `--max-concurrent-jobs`, `--polling-interval`, `--log-dir`, and `--dry-run`. The second helper runs the common queue/poll loop: launch `srun` commands up to the concurrency limit, poll them, print progress, collect failures, and return `0`, `2`, or `130` with the same meaning the legacy scripts use. The worker command should always be invoked as `sys.executable -m movebench.cli.main ...`, not by executing repo scripts directly, so the launched job uses the same installed package and Python interpreter as the launcher.

Milestone 2 implements the first-class local `pack` workflow. Create `movebench/cli/pack.py` for the user-facing command and `movebench/pack.py` for the shared runtime helpers that local and Slurm execution both call. Reuse the useful cache-build behavior from `scripts/deepssf_cache_wds.py`, but do not carry over its throughput benchmarking helpers into the main CLI. The local command should accept `--studies`, `--timescales`, and optional `--splits`, along with the cache build knobs already supported by `ensure_deepssf_wds_cache_split(...)`. If `--splits` is omitted, it should use `Study.split_inventory()["split_keys"]`. It should resolve each timescale through `movebench/bench/config.py`, compute the expected cache hash with `deepssf_cache_hash_for_split(...)`, skip complete study/timescale pairs by checking both manifest and shards, and then call `ensure_deepssf_wds_cache_split(...)` for the required splits of one study/timescale pair at a time.

Milestone 3 is complete under the prerequisite [execplans/extract_workflow_alignment_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/extract_workflow_alignment_execplan.md). Do not reopen extract design work here unless the benchmark refactor exposes a regression. The current extract contract is the shared package runtime in `movebench/extract.py`: omitted `--studies` resolves all available studies, omitted `--splits` resolves `Study.split_inventory()["split_keys"]` with a `["train"]` fallback, GEE detection inspects both direct and derived layer dependencies, known failed download requests block unless overridden, completion checks include train normalization stats, and both the local CLI and the Slurm worker compute train normalization after extraction.

Milestone 4 implements native Slurm pack submission in a way that matches the local `pack` command, not the legacy benchmark path. Create `movebench/slurm/pack.py`. Its user-facing arguments should be the same work-selection and cache-build arguments as local `movebench pack`, plus the shared Slurm resource flags. If `--splits` is omitted, the launcher should use `Study.split_inventory()["split_keys"]`, which yields `train` plus every held-out split recorded in the study manifest and falls back to `trajectories.csv` only when that metadata is absent. Reuse `movebench/pack.py` directly so the local and Slurm pack commands read the same split contract, plan the same study/timescale queue, and execute the same per-job worker helper. Each Slurm job should still correspond to one study/timescale pair, and the worker should call the shared pack execution helper so local and Slurm behavior stay aligned.

Milestone 5 refactors the benchmark runtime just enough to support native Slurm execution cleanly. `movebench/bench/run.py` currently bundles together “expand the experiment”, “write the resolved manifest”, “look up latest MLflow runs”, “choose which runs are pending”, and “execute each run immediately”. Extract that into stable public helpers that both the existing local `bench run` command and the new Slurm launcher can call. Define the interfaces in `movebench/bench/run.py` as:

    def resolve_pending_runs(
        experiment_module: str,
        *,
        studies_override: list[str] | None = None,
        methods: list[str] | None = None,
        max_runs: int | None = None,
        rerun: bool = False,
    ) -> dict[str, object]: ...

    def load_resolved_run(manifest_path: str | Path, run_key: str) -> dict[str, object]: ...

    def execute_run_payload(run_payload: dict[str, object]) -> dict[str, object]: ...

`resolve_pending_runs(...)` should return at least the normalized experiment metadata, the manifest path, the full planned run list, the pending run list, and the counts that the local CLI currently prints. `execute_run_payload(...)` should be the public wrapper around the current private run execution path.

Milestone 6 implements `movebench/slurm/bench.py`. The launcher should call `resolve_pending_runs(...)`, then launch one Slurm job per pending `run_key`. The worker should accept a hidden `--manifest-path` and `--run-key`, reload the exact run payload with `load_resolved_run(...)`, and then call `execute_run_payload(...)`. This keeps the login-node planner and the cluster worker in sync with the same resolved manifest instead of rebuilding payloads twice. The launcher should also choose default Slurm resources per method family. Keep that mapping local to the Slurm benchmark module rather than embedding cluster-specific defaults into `bench` itself. The initial mapping should be minimal and explicit: baselines and iSSF methods default to zero GPUs, `deepssf` defaults to one GPU, and `moveformer` defaults to one GPU. Every resource field remains overridable from the CLI.

Milestone 7 verifies the native CLI and only then decides what to do with `scripts/slurm/`. Do not delete the old scripts as part of the first code change. Leave them in place while the new path is being validated. After the new commands pass dry-run and unit-test validation, either leave the old scripts marked as legacy reference or convert them into thin wrappers in a follow-up task. The important part of this task is that the main CLI becomes the obvious supported path.

## Concrete Steps

Work from the repository root in the MoveBench micromamba environment:

    cd /data/vision/beery/scratch/justin/movebench

The extract prerequisite is already complete, so the next code change starts with the missing benchmark tests. Before editing `movebench/bench/run.py` or `movebench/slurm/bench.py`, add `tests/test_slurm_bench.py` and extend `tests/bench/test_runner.py` to cover the public pending-run helpers. Cover at least: pending-run filtering with and without `--rerun`, manifest-path computation, loading one resolved run by `run_key`, launcher dry-run command generation, per-method GPU defaults, and worker execution through the shared run executor. Run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner

Observed pre-implementation result:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner

Result:

    FAILED (failures=1, errors=5)

The failure reasons were the expected contract gaps: the Slurm bench parser did not accept `--studies` or worker flags, `movebench.bench.run` did not expose `resolve_pending_runs(...)` or `load_resolved_run(...)`, and `run_experiment(...)` still bypassed the planned shared helper boundary.

After the benchmark helper slice lands, keep using the broader targeted commands for the remaining milestones:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_pack tests.test_cli_extract tests.test_slurm_extract tests.test_slurm_pack tests.test_slurm_bench
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_experiment_config tests.bench.test_runner

After each milestone, rerun the targeted suites before moving on. When the implementation is complete, run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v

Observed post-implementation results:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_slurm_bench tests.bench.test_runner

    Ran 8 tests in 0.155s
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_pack tests.test_cli_extract tests.test_slurm_extract tests.test_slurm_pack tests.test_slurm_bench

    Ran 25 tests in 4.529s
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_experiment_config tests.bench.test_runner

    Ran 7 tests in 0.080s
    OK

Repo-wide verification record for this environment:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v

Observed result:

    Ran 186 tests in 1.929s
    FAILED (failures=5, errors=95)

See `/tmp/full_unittest_movebench_20260404_native_slurm_bench.log`. The remaining failures are unrelated to this task and continue to come from `old_tests_dont_use` plus environment-level optional import and native-library issues.

The expected commands after implementation are:

    movebench pack --studies <study_id> --timescales 1d
    movebench slurm extract --studies <study_id> --layers-config configs/layers/all_ae_1d_1d.yaml --sampler-config configs/samplers/issf_defaults.yaml --dry-run
    movebench slurm pack --studies <study_id> --timescales 1d --dry-run
    movebench slurm bench run experiments/smoke_one_study.py --studies <study_id> --dry-run

The benchmark status commands remain local and should still be run without `slurm`:

    movebench bench plan experiments/smoke_one_study.py --studies <study_id>
    movebench bench status --format text
    movebench bench report

Expected output should look like this, with actual paths and run keys filled in:

    [pack] Processing 1 study/timescale jobs locally.
    [pack] study=<study_id> timescale=1d splits=train,test_temporal ...

    [extract] Dry-run: would launch 1 jobs (max_concurrent=32).
    [extract] srun --export=ALL ... python -m movebench.cli.main slurm extract --worker-study <study_id> ...

    [pack] Dry-run: would launch 1 jobs (max_concurrent=32).
    [pack] srun --export=ALL ... python -m movebench.cli.main slurm pack --worker-study <study_id> --worker-timescale 1d ...

    [bench] Dry-run: would launch 2 jobs (max_concurrent=32).
    [bench] srun --export=ALL ... python -m movebench.cli.main slurm bench run --manifest-path ... --run-key study=<study_id>__ts=1d__cov=all__method=deepssf__seed=0__k=...

## Validation and Acceptance

Acceptance for this task is behavioral.

First, `movebench pack ...` must resolve study/timescale work from `bench` metadata, detect missing point-store and normalization prerequisites, detect complete caches by manifest plus shards, and process one study/timescale pair at a time locally. If `--splits` is omitted, it must default to `Study.split_inventory()["split_keys"]`, and cache paths must use canonical underscore-form split keys.

Second, the previously completed extract milestone must stay correct. `movebench slurm extract ... --dry-run` and its worker mode must continue to preserve the shared `movebench/extract.py` semantics for study selection, split resolution, `skip_existing`, known download failure handling, GEE derived-dependency detection, completion checks, and post-extract train normalization. The benchmark refactor in this plan must not reintroduce extract drift.

Third, `movebench slurm pack ... --dry-run` must use the same queue resolution and per-job execution helper as local `movebench pack`, but instead of executing jobs inline it must print one `srun` command per pending study/timescale pair that still needs cache work.

Fourth, `movebench slurm bench run ... --dry-run` must use the same resolved manifest and pending-run selection as local `movebench bench run`, but instead of executing runs inline it must print one `srun` command per pending `run_key`. The CLI must not expose `movebench slurm bench status`, `movebench slurm bench plan`, or `movebench slurm bench report`.

Fifth, the local benchmark commands must remain correct after the refactor. `movebench bench run` must still execute pending runs inline, and `movebench bench status` must still work without touching Slurm.

Sixth, the full unit-test suite must pass. At minimum the new targeted pack and Slurm tests and the affected `bench` tests must pass before the task can be considered complete.

## Idempotence and Recovery

The local `pack` command and all three launcher commands must remain safe to rerun. `--dry-run` performs no mutation. Extract reruns should skip already-written point stores when `skip_existing` is enabled; when `--no-skip-existing` is set, local extract and Slurm extract should both rerun the work instead of pruning it in the launcher. Pack reruns, whether local or through Slurm, should skip study/timescale pairs whose expected manifests and shards already exist. Benchmark reruns should reuse `bench`'s existing “skip completed or running runs unless `--rerun` is set” behavior.

If the launcher process is interrupted, already-submitted `srun` jobs may continue running. That is acceptable. The recovery path is to rerun the same launcher command; the existing skip logic should prevent duplicate finished work. For local pack interruptions, rerun `movebench pack ...` and let the same skip logic avoid rebuilding complete caches. `movebench bench status` remains the way to inspect benchmark completion.

If Slurm is unavailable in the local development environment, unit tests and `--dry-run` output are still sufficient to validate the command planning logic. Record that limitation explicitly when updating the plan during implementation.

## Artifacts and Notes

The intended package shape is:

    movebench/cli/pack.py
    movebench/cli/slurm.py
    movebench/pack.py
    movebench/slurm/__init__.py
    movebench/slurm/launcher.py
    movebench/slurm/extract.py
    movebench/slurm/pack.py
    movebench/slurm/bench.py

The intended command surface is:

    movebench pack ...
    movebench slurm extract ...
    movebench slurm pack ...
    movebench slurm bench run ...

For extraction, `movebench/extract.py` is now the source of truth. `movebench/cli/extract.py`, `movebench/slurm/extract.py`, and `scripts/slurm/extract.py` are wrappers over that shared runtime and should remain unchanged unless the benchmark work exposes a regression.

The benchmark worker path should always load the already-written resolved manifest instead of reconstructing run payloads from ad hoc CLI flags. That keeps local and Slurm execution paths identical at the run-payload level.

The results layout should stay under the existing repository roots. The new launchers may continue writing logs under `results/slurm/...`, but benchmark artifacts and MLflow state must remain under `results/bench/...`.

## Interfaces and Dependencies

Define in `movebench/cli/pack.py`:

    def register(subparsers: argparse._SubParsersAction) -> None: ...

Define in `movebench/cli/slurm.py`:

    def register(subparsers: argparse._SubParsersAction) -> None: ...

Reuse the existing shared extract interfaces in `movebench/extract.py`:

    def build_extraction_runs(...) -> list[ExtractionRun]: ...
    def resolve_requested_studies(studies_arg: str | None) -> list[str]: ...
    def default_splits(study: Study) -> list[str]: ...
    def is_extract_complete(...) -> tuple[bool, str]: ...
    def uses_gee_layers(plan: Any) -> bool: ...
    def run_extract_job(...) -> int: ...

These interfaces already own the extract semantics for local execution, Slurm worker execution, completion checks, and known-download preflight behavior. This plan should only preserve them with regression tests.

Define in `movebench/pack.py`:

    @dataclass(frozen=True)
    class PackJob:
        study_id: str
        timescale_key: str
        split_keys: tuple[str, ...]

    def register_pack_arguments(parser: argparse.ArgumentParser) -> None: ...
    def resolve_pack_jobs(...) -> list[PackJob]: ...
    def execute_pack_job(...) -> dict[str, object]: ...

Define in `movebench/slurm/launcher.py`:

    @dataclass(frozen=True)
    class SlurmResources:
        partition: str | None
        qos: str | None
        account: str | None
        time: str | None
        mem: str | None
        cpus_per_task: int
        gpus_per_node: int

    def register_common_slurm_arguments(parser: argparse.ArgumentParser) -> None: ...
    def build_srun_command(...) -> list[str]: ...
    def run_srun_queue(...) -> int: ...

Define in `movebench/slurm/extract.py`, `movebench/slurm/pack.py`, and `movebench/slurm/bench.py`:

    def register(subparsers: argparse._SubParsersAction) -> None: ...

Define in `movebench/bench/run.py`:

    def resolve_pending_runs(...) -> dict[str, object]: ...
    def load_resolved_run(manifest_path: str | Path, run_key: str) -> dict[str, object]: ...
    def execute_run_payload(run_payload: dict[str, object]) -> dict[str, object]: ...

Reuse the existing split-discovery API in `movebench/study.py`:

    def split_inventory(self) -> dict[str, object]: ...

The returned mapping must remain the shared contract for local pack, Slurm extract, Slurm pack, and `bench`: it includes `has_train`, ordered `split_keys`, and ordered `held_out_splits`, with split names already canonicalized to underscore form.

Plan creation note (2026-03-27 / assistant): initial plan drafted after reviewing the current CLI, the active benchmark runtime, and the legacy `scripts/slurm/` launchers, with the goal of making Slurm submission a first-class CLI path without reintroducing the legacy benchmark architecture.
Revision note (2026-03-28 / assistant): updated the plan after the split-tracking changes in `f58b5b2` and `92c41fe` so it now reflects `Study.split_inventory()`, manifest-backed split metadata, and underscore-canonical split keys instead of the older `train`/`test` assumptions.
Revision note (2026-03-28 / assistant): revised the plan so deepSSF cache building becomes a first-class `movebench pack` workflow, with `movebench slurm pack` as a wrapper over the same runtime and `wds` retained only as an internal artifact name.
Revision note (2026-04-01 / assistant): updated the plan after the benchmark namespace cutover so active references point at the permanent `bench` worktree and paths: `movebench/bench/`, `experiments/`, `tests/bench`, and `results/bench`.
Revision note (2026-04-01 / assistant): implemented Milestones 1 and 2, recorded the active point-store prerequisite helpers that replaced the stale legacy import, and updated the plan state to reflect the passing `tests.test_pack` validation for the first `movebench pack` slice.
Revision note (2026-04-02 / assistant): implemented Milestones 3 and 4, added `tests.test_slurm_extract` and `tests.test_slurm_pack`, recorded the parser/normalization discoveries that affected the worker shape, and updated the plan state after the passing targeted slice validation in the MoveBench micromamba environment.
Revision note (2026-04-02 / assistant): split the extract corrective work into the prerequisite `execplans/extract_workflow_alignment_execplan.md` after confirming the broader plan was evolving against the wrong extract contract.
Revision note (2026-04-04 / assistant): updated the broader plan after the extract prerequisite landed. The plan now treats extract as completed shared runtime work in `movebench/extract.py`, records the current revalidation results, and narrows the remaining implementation to the benchmark helper refactor plus native `movebench slurm bench run`.
Revision note (2026-04-04 / assistant): implemented the remaining benchmark Slurm work, recorded the failing and passing benchmark test slices, captured the real CLI dry-run result for the current environment, and updated the repo-wide unittest record and interface notes to match the final code.
