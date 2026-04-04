# Add First-Class `pack` Commands and Native Slurm Launchers for Extract, Pack, and `bench`

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, a user will be able to run the deepSSF cache-packaging workflow directly from the main CLI as `movebench pack ...`, and submit the three batch-oriented workflows from the main CLI instead of jumping into `scripts/slurm/`: `movebench slurm extract ...`, `movebench slurm pack ...`, and `movebench slurm bench run ...`. The visible proof is that the Slurm commands support `--dry-run`, print the exact `srun` commands they would launch, and use the same underlying runtime code as the non-Slurm workflows instead of a second, drifting implementation.

For extraction in particular, the supported source of truth is the current local CLI behavior in `movebench/cli/extract.py`. `movebench slurm extract` is only allowed to add Slurm queue planning, repeated config expansion, and worker submission; it must preserve the current local extract semantics for study selection, split resolution, skip-existing behavior, environment overrides, and known download failure handling.

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
- [x] (2026-04-02 19:39Z) Reviewed the Milestone 3 extract implementation against the supported local `movebench extract` behavior and identified parity drift in study resolution, split resolution, `skip_existing`, environment overrides, known download failure handling, and the unexpected normalization side effect.
- [ ] Correct Milestone 3 so `movebench slurm extract` becomes a strict Slurm wrapper over the current local extract runtime, with launcher-only planning limited to queue construction, completion pruning when `skip_existing` is enabled, and GEE raster-root blocking that also considers derived dependencies.
- [ ] Refactor `movebench/bench/run.py` so local execution and Slurm execution share the same pending-run selection and run execution helpers.
- [ ] Implement native `movebench slurm bench run`.
- [x] (2026-04-01 22:38Z) Added `tests/test_pack.py` and ran `python -m unittest -v tests.test_pack` to verify the Milestone 1 parser/launcher skeleton and the Milestone 2 local pack planner/executor.
- [ ] Add or update unit tests for parser shape, local pack execution, launcher logic, extract, pack, and benchmark run selection; then run the targeted suites and `python -m unittest -v` (completed so far: `tests.test_pack`, `tests.test_slurm_extract`, and `tests.test_slurm_pack`; remaining: Slurm bench, the affected `bench` suites, and the full test run).
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

- Observation: the current local extract CLI is the only supported source of truth for extract behavior, and the first native Slurm extract implementation drifted because it reimplemented that behavior instead of sharing it.
  Evidence: `movebench/cli/extract.py` supports omitted or `all` studies, `--no-skip-existing`, `--allow-known-download-failures`, `--data-root`, `--paths-config`, and `_default_splits(...)` with a `["train"]` fallback, while `movebench/slurm/extract.py` currently omits several of those flags, requires `--studies`, passes `splits=None`, and computes normalization stats that local extract does not compute.

- Observation: the earlier Milestone 3 wording about train-normalization work in the Slurm extract path is stale relative to the current local CLI.
  Evidence: `movebench/cli/extract.py` now stops after `Study.extract_point_features(...)`; the explicit `Study.compute_normalization_vals(...)` call only exists in `movebench/slurm/extract.py`, so preserving that side effect would make Slurm extract differ from the supported local command.

- Observation: GEE raster prerequisite detection for extract planning must consider derived layer dependencies, not only `plan.layers_in_order`.
  Evidence: `movebench.layers.plan.LayerPlan` records `derived_dependencies`, and other Slurm planning code already walks those parents before deciding whether a plan depends on GEE-backed rasters.

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

- Decision: treat `movebench/cli/extract.py` as the authoritative extract runtime and make `movebench/slurm/extract.py` import its extract semantics instead of maintaining a second implementation.
  Rationale: the user wants current `movebench extract` behavior preserved exactly. The initial native Slurm extract implementation drifted because it duplicated study selection, split handling, and preflight logic instead of sharing the current runtime.
  Date/Author: 2026-04-02 / assistant

- Decision: remove the Slurm-only train-normalization side effect from `movebench slurm extract` unless local `movebench extract` is changed to compute normalization too.
  Rationale: preserving parity with the supported local CLI matters more than preserving an older side effect from `scripts/slurm/extract.py` or stale milestone wording.
  Date/Author: 2026-04-02 / assistant

- Decision: launcher-side extract completion pruning must honor `skip_existing`; when `--no-skip-existing` is set, the launcher must still queue work even if point-store manifests already exist.
  Rationale: otherwise the Slurm launcher would block a rerun that the local extract CLI explicitly allows.
  Date/Author: 2026-04-02 / assistant

## Outcomes & Retrospective

This plan started as the design and implementation blueprint for the native Slurm and pack work. It resolved the main architecture questions that mattered before writing code: where the CLI should live, what the benchmark job boundary should be, how cache packaging should relate to `bench`, and why the legacy `scripts/slurm/` files should not become the permanent runtime. On 2026-03-28 it was revised first to reflect the current split contract, and then to make cache packaging a first-class `pack` workflow with a local CLI and a Slurm wrapper. On 2026-04-01 it was updated again to follow the permanent benchmark namespace and paths. Later on 2026-04-01, Milestones 1 and 2 were implemented and validated with `python -m unittest -v tests.test_pack`, which means the first user-visible slice now exists: `movebench pack ...` is registered, plans work from timescale metadata, skips complete caches, and executes pending study/timescale jobs locally, while the top-level `slurm` parser tree and shared launcher helper are in place for the remaining milestones. On 2026-04-02, Milestones 3 and 4 were implemented and validated with `python -m unittest -v tests.test_slurm_extract tests.test_slurm_pack`, but the extract slice is now known to drift from the supported local `movebench extract` CLI. The next required step is therefore a corrective extract-parity slice before any benchmark Slurm work continues; `movebench slurm pack` remains aligned with the shared `movebench/pack.py` runtime.

## Context and Orientation

The current top-level CLI lives in `movebench/cli/main.py`. It registers `download`, `extract`, `ingest`, and `bench`. `movebench/cli/extract.py` is the supported source of truth for extraction behavior: it resolves studies, resolves default splits, applies environment overrides, checks known download failures, and calls `Study.extract_point_features(...)`. `movebench/bench/run.py` owns the benchmark runtime. It already defines the `bench` subcommands `plan`, `status`, `report`, and `run`, and its `run` path is based on concrete run payloads and stable `run_key` values. This plan adds two new user-facing surfaces: a local `pack` command and a top-level `slurm` group.

In this plan, “pack” means building the deepSSF WebDataset cache shards and manifest files that live under each study’s `cache/deepssf_wds/` directory. The internal code and on-disk paths can keep the existing `deepssf_wds` name because that is the artifact format, but the user-facing CLI should call this workflow `pack`.

In this repository, a “Slurm launcher” means the process that runs on the login node, decides what work is still needed, and spawns one or more `srun` commands while throttling the number of concurrent jobs. A “worker” means the code that runs inside the `srun` job itself and performs exactly one unit of work.

For extraction, the natural work unit is one study plus one layer-config/sampler-config combination. For cache packaging, the natural work unit is one study plus one timescale, because the cache hash is determined by the timescale’s raster hash, chip size, normalization settings, and split key. For `bench`, the natural work unit is one concrete benchmark run payload, identified by `run_key`.

The legacy scripts are useful references for CLI flags and for the queue/poll output style, but they are not aligned with the current codebase:

- `scripts/deepssf_cache_wds.py` is the closest existing local reference for `pack`, but it lives outside the main CLI and mixes cache building with benchmarking code that should not become part of the first-class command.
- `scripts/slurm/extract.py` never became part of the main CLI, even though it now derives default splits from `Study.split_inventory()`.
- `scripts/slurm/wds.py` now uses the study split inventory too, but it is still tied to legacy YAML benchmark configs rather than `movebench/bench/config.py`, and its user-facing name no longer matches the desired CLI.
- `scripts/slurm/bench.py` is entirely built around the deleted `movebench.benchmark` path, not `movebench/bench/`.

The important current files for this work are:

- `movebench/cli/main.py`, which must gain the new top-level `pack` command and the new top-level `slurm` group.
- `movebench/cli/pack.py`, which will provide the first-class local `movebench pack` command.
- `movebench/cli/extract.py`, which is the extract source of truth and must expose the minimal helper functions needed so `movebench slurm extract` can preserve the current local semantics instead of reimplementing them.
- `movebench/pack.py`, which should hold the shared pack argument registration, queue resolution, and worker execution helpers used by both `movebench pack` and `movebench slurm pack`.
- `movebench/download_reports.py`, which provides the current extract preflight check for known failed chip-download requests.
- `movebench/bench/run.py`, which must expose shared helpers for “plan runs, choose pending runs, execute one resolved run payload”.
- `movebench/bench/config.py`, which is the authoritative source for timescale-specific layer configs, sampler configs, and deepSSF cache hashes.
- `movebench/datasets/deepssf_wds_cache.py`, which already contains the low-level cache build primitives that `pack` should orchestrate instead of reimplementing.
- `movebench/study.py`, which already owns high-level study operations and now centralizes split discovery through `Study.split_inventory()`.
- `movebench/trajectories.py`, which canonicalizes split keys so user input like `test-temporal` and stored metadata like `test_temporal` resolve to the same on-disk paths.

## Plan of Work

Milestone 1 creates the shared Slurm runtime and the CLI skeleton. Add `movebench/cli/slurm.py` and register it from `movebench/cli/main.py`. Add `movebench/cli/pack.py` and register it from `movebench/cli/main.py`. The `slurm` group must contain `extract`, `pack`, and `bench`, with `bench` containing only `run`. Create package modules under `movebench/slurm/` for the shared launcher runtime and the three workflow-specific commands. Keep file count low: one shared launcher module plus one module per workflow, plus one shared pack runtime module, is enough.

The shared launcher module should define a small dataclass for Slurm resources and two reusable helpers. The first helper registers the common Slurm arguments on an `argparse` parser: `--partition`, `--qos`, `--account`, `--time`, `--mem`, `--cpus-per-task`, `--gpus-per-node`, `--max-concurrent-jobs`, `--polling-interval`, `--log-dir`, and `--dry-run`. The second helper runs the common queue/poll loop: launch `srun` commands up to the concurrency limit, poll them, print progress, collect failures, and return `0`, `2`, or `130` with the same meaning the legacy scripts use. The worker command should always be invoked as `sys.executable -m movebench.cli.main ...`, not by executing repo scripts directly, so the launched job uses the same installed package and Python interpreter as the launcher.

Milestone 2 implements the first-class local `pack` workflow. Create `movebench/cli/pack.py` for the user-facing command and `movebench/pack.py` for the shared runtime helpers that local and Slurm execution both call. Reuse the useful cache-build behavior from `scripts/deepssf_cache_wds.py`, but do not carry over its throughput benchmarking helpers into the main CLI. The local command should accept `--studies`, `--timescales`, and optional `--splits`, along with the cache build knobs already supported by `ensure_deepssf_wds_cache_split(...)`. If `--splits` is omitted, it should use `Study.split_inventory()["split_keys"]`. It should resolve each timescale through `movebench/bench/config.py`, compute the expected cache hash with `deepssf_cache_hash_for_split(...)`, skip complete study/timescale pairs by checking both manifest and shards, and then call `ensure_deepssf_wds_cache_split(...)` for the required splits of one study/timescale pair at a time.

Milestone 3 corrects native extraction submission so `movebench slurm extract` is a strict Slurm wrapper around the current local extract workflow. Keep file count low. Do not add a new parallel extract runtime. Instead, refactor only the minimum needed inside `movebench/cli/extract.py` so its current behavior can be imported and reused by `movebench/slurm/extract.py`. The shared helper path must own the current local semantics for applying `--data-root` and `--paths-config`, resolving omitted or `all` studies, resolving default splits with the existing `["train"]` fallback, honoring `--no-skip-existing`, and blocking known failed chip-download requests unless `--allow-known-download-failures` is set. Then update `movebench/slurm/extract.py` so the worker path calls those shared helpers for exactly one study plus one layer-config/sampler-config combination, while the launcher remains responsible only for repeated or comma-separated `--layers-config`, repeated or comma-separated `--sampler-config`, per-combination log directories, completion-based queue pruning when `skip_existing` is enabled, and dry-run output. The worker must accept and forward the same extract behavior flags as local `movebench extract`: `--no-skip-existing`, `--allow-known-download-failures`, `--data-root`, and `--paths-config`. Do not compute train normalization stats in the Slurm path unless local `movebench extract` is changed to do so. The launcher must block a job when the required raster cache root is missing for a GEE-backed layer plan, and that GEE check must inspect both direct layers and derived dependencies.

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

Before correcting Milestone 3, add or update the extract parity tests in `tests/test_slurm_extract.py` before making code changes. Cover omitted `--studies`, `--studies all`, `--no-skip-existing`, `--allow-known-download-failures`, `--data-root`, `--paths-config`, the current local default-split behavior, and the derived-dependency GEE raster block. Run:

    python -m unittest -v tests.test_slurm_extract

After the corrective extract slice passes, keep using the broader targeted commands for the remaining milestones:

    python -m unittest -v tests.test_pack tests.test_slurm_extract tests.test_slurm_pack tests.test_slurm_bench
    python -m unittest -v tests.bench.test_experiment_config tests.bench.test_runner

After each milestone, rerun the targeted suites before moving on. When the implementation is complete, run:

    python -m unittest -v

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

Second, `movebench slurm extract ... --dry-run` and its worker mode must preserve the current `movebench extract` semantics exactly for study selection, split resolution, `skip_existing`, known download failure handling, and environment overrides. Slurm adds only queue planning, repeated config expansion, log placement, and `srun` submission. If `--splits` is omitted, the worker must resolve the same split list that local extract uses today, including the existing `["train"]` fallback when the study split inventory is empty. The launcher may skip already-complete extraction outputs only when `skip_existing` is enabled. It must block missing raster prerequisites for GEE-backed plans, including plans whose GEE dependency appears only through `derived_dependencies`. The worker must not compute train normalization stats unless local extract is changed to do so.

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

For extraction, `movebench/cli/extract.py` is the source of truth. `movebench/slurm/extract.py` should reuse the local extract semantics directly and keep only the extra parsing and queue helpers that exist solely because the Slurm launcher expands one logical extract request into many `srun` jobs.

The benchmark worker path should always load the already-written resolved manifest instead of reconstructing run payloads from ad hoc CLI flags. That keeps local and Slurm execution paths identical at the run-payload level.

The results layout should stay under the existing repository roots. The new launchers may continue writing logs under `results/slurm/...`, but benchmark artifacts and MLflow state must remain under `results/bench/...`.

## Interfaces and Dependencies

Define in `movebench/cli/pack.py`:

    def register(subparsers: argparse._SubParsersAction) -> None: ...

Define in `movebench/cli/slurm.py`:

    def register(subparsers: argparse._SubParsersAction) -> None: ...

Define the minimal shared extract helpers in `movebench/cli/extract.py` so the local CLI and Slurm worker can share the same current runtime behavior:

    def apply_extract_environment(*, data_root: str | None, paths_config: str | None) -> None: ...
    def resolve_requested_studies(studies: list[str] | None) -> list[str]: ...
    def resolve_splits_for_study(study: Study, splits: list[str] | None) -> list[str]: ...
    def run_extract_for_study(...) -> int: ...

`run_extract_for_study(...)` must own the current local extract semantics for `skip_existing`, known download failure blocking, and the call to `Study.extract_point_features(...)`. `movebench/slurm/extract.py` may keep launcher-only helpers for multi-config argument parsing, completion checks, GEE prerequisite checks, worker argument construction, and log-path layout, but it should not duplicate the local extract rules.

Define in `movebench/pack.py`:

    @dataclass(frozen=True)
    class PackJob:
        study_id: str
        timescale_key: str
        splits: tuple[str, ...]

    def add_pack_args(parser: argparse.ArgumentParser) -> None: ...
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
        cpus_per_task: int | None
        gpus_per_node: int | None

    def add_slurm_args(parser: argparse.ArgumentParser, *, defaults: SlurmResources) -> None: ...
    def launch_queue(...) -> int: ...

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
Revision note (2026-04-02 / assistant): revised the plan after reviewing the native extract slice against the current supported `movebench extract` CLI. The plan now treats `movebench/cli/extract.py` as the extract source of truth, removes the stale Slurm-only normalization requirement, and adds a corrective extract-parity slice before benchmark Slurm work continues.
