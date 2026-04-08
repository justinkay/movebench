# Transition All `movebench slurm` Workflows To `sbatch` Arrays With Requeue

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, every public Slurm submission path in MoveBench will submit `sbatch` job arrays instead of launching one `srun` process per task from the login node. That includes `movebench slurm extract`, `movebench slurm pack`, and `movebench slurm bench run`. The practical gain is that Slurm, not the local launcher process, owns queued and running work, so preempted jobs can be requeued on this cluster.

The visible proof is that `--dry-run` prints `sbatch --requeue --array=...` commands instead of `srun ...` commands, and real launches print Slurm job ids for one or more arrays and then wait for those arrays to finish. `bench` will submit separate arrays when methods require different resources, with CPU-only methods and GPU methods split into different arrays by default. `extract` and `pack` remain one array each because their tasks already share one resource profile per command invocation.

This plan deliberately keeps the code change minimal. It does not redesign the underlying extract, pack, or bench worker logic. It reuses the existing hidden worker command shapes and adds one shared array-manifest launcher path so all three workflows can transition together without three separate implementations.

## Progress

- [x] (2026-04-07 03:03Z) Read `PLANS.md`, the current shared Slurm launcher in `movebench/slurm/launcher.py`, the three workflow launchers in `movebench/slurm/extract.py`, `movebench/slurm/pack.py`, and `movebench/slurm/bench.py`, and the existing regression tests in `tests/test_pack.py`, `tests/test_slurm_extract.py`, `tests/test_slurm_pack.py`, and `tests/test_slurm_bench.py`.
- [x] (2026-04-07 03:03Z) Verified the cluster-specific scheduler settings relevant to this migration: `PreemptMode=REQUEUE`, `PreemptType=preempt/qos`, `JobRequeue=0`, and `MaxArraySize=20000`.
- [x] (2026-04-07 03:03Z) Decided the implementation scope after user clarification: all public Slurm workflows must transition, not only `bench`.
- [x] (2026-04-07 03:03Z) Chose the minimal shared design: store literal worker argument vectors in per-array manifest files, execute them through one shared internal launcher entrypoint in `movebench/slurm/launcher.py`, and keep resource bucketing logic local to `movebench/slurm/bench.py`.
- [x] (2026-04-07 15:05Z) Revised the plan after review to clarify the `sbatch --wait` launch model, move shared launcher tests into a dedicated file, keep dry-run side-effect free, remove README work, and narrow required acceptance to deterministic tests plus dry-run behavior.
- [x] (2026-04-07 22:04Z) Revised the plan after review to make the `sbatch --wrap` contract explicit: the outer `sbatch` invocation stays a `subprocess.Popen(...)` argv list, the inner launcher command is converted to one shell-safe string with `shlex.join(...)`, and v1 will not write temporary batch scripts.
- [x] (2026-04-08 01:52Z) Added `tests/test_slurm_launcher.py`, updated the `extract`, `pack`, and `bench` dry-run expectations for array submissions, and ran the targeted unittest suite to confirm the new assertions fail against the current `srun` implementation.
- [x] (2026-04-08 02:06Z) Implemented the shared `sbatch --wait --requeue` array backend in `movebench/slurm/launcher.py`, including hashed array manifests, `build_sbatch_command(...)`, `run_sbatch_queue(...)`, and the hidden `python -m movebench.slurm.launcher --array-manifest ...` runtime.
- [x] (2026-04-08 02:06Z) Migrated `movebench/slurm/extract.py`, `movebench/slurm/pack.py`, and `movebench/slurm/bench.py` to the shared `sbatch` backend, keeping `extract` and `pack` as one-array launchers and adding local resource bucketing plus cap allocation in `bench`.
- [x] (2026-04-08 02:06Z) Ran the targeted Slurm-facing unittest suite: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_pack tests.test_slurm_launcher tests.test_slurm_extract tests.test_slurm_pack tests.test_slurm_bench tests.bench.test_runner` and observed `Ran 41 tests in 1.372s` / `OK`.
- [ ] The broader Slurm-facing and full-repository unittest runs from the original plan were not rerun after the user explicitly restricted validation to the targeted Slurm-facing suite and requested no out-of-scope work.

## Surprises & Discoveries

- Observation: the current Slurm implementation is already centralized enough that one backend swap can move all three workflows.
  Evidence: `movebench/slurm/extract.py`, `movebench/slurm/pack.py`, and `movebench/slurm/bench.py` all depend on `register_common_slurm_arguments(...)`, `apply_slurm_config(...)`, `build_srun_command(...)`, and `run_srun_queue(...)` from `movebench/slurm/launcher.py`.

- Observation: this cluster supports requeue-based preemption, but jobs are not requeueable by default.
  Evidence: `scontrol show config` reported `PreemptMode = REQUEUE`, `PreemptType = preempt/qos`, and `JobRequeue = 0`.

- Observation: Slurm job arrays are available at a scale that is large enough for this repo’s current launcher model.
  Evidence: `scontrol show config` reported `MaxArraySize = 20000`.

- Observation: the user-facing “max jobs” behavior can map directly to one array for `extract` and `pack`, but not to a single `bench` array because `bench` has mixed resource defaults.
  Evidence: `movebench/slurm/bench.py` currently defaults `issf_*` and baseline methods to `gpus_per_node = 0`, but defaults `deepssf` and `moveformer` to `gpus_per_node = 1`.

- Observation: a fully asynchronous `sbatch` migration would create an easy duplicate-submission failure mode while jobs are still pending in Slurm.
  Evidence: `movebench/bench/run.resolve_pending_runs(...)` filters by MLflow run status, not by queued Slurm state, so resubmitting after an async launch but before worker startup could enqueue the same `run_key` again. `sbatch --help` includes `-W, --wait`, so the simpler and safer migration is to keep the current blocking launcher behavior.

- Observation: the current public CLI tree should not gain a visible internal subcommand.
  Evidence: `tests/test_pack.py` asserts that the public `movebench slurm` parser exposes only `bench`, `extract`, and `pack`. Using `python -m movebench.slurm.launcher` for the hidden array-task runtime avoids changing that public contract.

- Observation: the current repo already has an established pattern for hashed launch manifests and workflow-specific Slurm log roots.
  Evidence: `movebench/bench/run.py` writes resolved manifests under `results/bench/manifests/<slug>__<hash>.json`, while `configs/slurm/beery.yaml` and `configs/slurm/shared.yaml` already pin per-workflow Slurm log directories under `results/slurm/...`.

- Observation: the refreshed targeted suite still includes one pre-existing local config mismatch unrelated to the launcher backend.
  Evidence: after the test updates, `tests.test_slurm_extract.test_checked_in_beery_config_sets_extract_time` still observed `--time 2-00:00:00` from the local untracked `configs/slurm/beery.yaml` while the tracked test expects `7-00:00:00`; the new array-launcher failures remained the expected missing `build_sbatch_command(...)`, missing `run_sbatch_queue(...)`, missing `movebench.slurm.launcher.main(...)`, and dry-runs still printing `srun`.

## Decision Log

- Decision: transition all three public Slurm workflows in one plan.
  Rationale: the user explicitly clarified that `extract` and `pack` must also move to `sbatch`, and the existing shared launcher makes one coordinated backend migration smaller than three separate ones.
  Date/Author: 2026-04-07 / assistant

- Decision: keep the launcher blocking until completion by using `sbatch --wait` instead of turning the commands into fire-and-forget submitters.
  Rationale: this preserves the current command shape, avoids a new queued-job deduplication problem, and still gives the user Slurm-managed requeue behavior.
  Date/Author: 2026-04-07 / assistant

- Decision: always pass `--requeue` on real `sbatch` submissions in v1.
  Rationale: the whole purpose of this migration is to make preempted jobs requeueable, and this cluster reports `JobRequeue=0`, so omitting the flag would silently defeat the goal.
  Date/Author: 2026-04-07 / assistant

- Decision: reuse the existing worker command interfaces by storing literal worker argument vectors in array manifests and replaying them from one shared internal launcher runtime.
  Rationale: this is the smallest DRY change. `extract`, `pack`, and `bench` already know how to build the right hidden worker arguments, so the new shared code only needs to choose which task index to run.
  Date/Author: 2026-04-07 / assistant

- Decision: keep `--polling-interval` and the related config field in v1.
  Rationale: with `sbatch --wait`, the launcher still polls local waiting processes for progress, so the existing knob remains meaningful and keeping it avoids unnecessary config churn.
  Date/Author: 2026-04-07 / assistant

- Decision: keep bench resource bucketing logic local to `movebench/slurm/bench.py`.
  Rationale: only `bench` currently needs multiple arrays from one command, so a repo-wide generic “multi-bucket scheduler” would be premature abstraction. The shared launcher should accept precomputed array requests and know nothing about method families.
  Date/Author: 2026-04-07 / assistant

- Decision: remove the old per-task launcher pool, but still keep a small local tracker of already-submitted `sbatch --wait` processes.
  Rationale: this is the smallest way to preserve current blocking semantics while letting Slurm, not the login node, own task concurrency through per-array `%K` throttles. The launcher should submit each array once, print its job id immediately, and then only poll those already-started submitters.
  Date/Author: 2026-04-07 / assistant + user

- Decision: keep dry-run side-effect free.
  Rationale: `--dry-run` should not write manifests, create directories, or produce any other launch artifacts.
  Date/Author: 2026-04-07 / assistant + user

- Decision: use `sbatch --wrap` in v1 without temporary batch-script files, and build its value from the internal launcher argv via `shlex.join(...)`.
  Rationale: this is the smallest implementation that keeps the launcher backend self-contained. `sbatch --wrap` expects one command string, while the outer `sbatch` submission should still be a normal `subprocess.Popen(...)` argv list. Converting only the inner launcher argv to one shell-safe string avoids `shell=True`, avoids ad hoc quoting helpers, and avoids extra file lifecycle logic for generated shell scripts.
  Date/Author: 2026-04-07 / assistant + user

- Decision: write array manifests under the existing workflow log roots using hashed filenames.
  Rationale: `Path(args.log_dir) / "manifests" / <label>__<hash>.json` reuses existing Slurm result directories and the repo’s existing hashed-manifest pattern without inventing a new top-level convention.
  Date/Author: 2026-04-07 / assistant + user

- Decision: keep the manifest schema minimal and move shared launcher tests into a dedicated test module.
  Rationale: the manifest only needs `worker_args`, and a new `tests/test_slurm_launcher.py` keeps shared launcher coverage from further bloating `tests/test_pack.py`.
  Date/Author: 2026-04-07 / assistant + user

- Decision: make unit tests and dry-run behavior the required acceptance surface for this plan; real cluster smoke checks are optional follow-up only.
  Rationale: there is no single known-local smoke command that can be prescribed self-containedly for a novice, so required acceptance should stay within deterministic repo-local validation.
  Date/Author: 2026-04-07 / assistant + user

- Decision: explicitly treat `bench` bucket counts greater than `--max-concurrent-jobs` as out of scope for v1.
  Rationale: current defaults do not require extra scheduler logic here, and adding serialization or fallback behavior would create avoidable complexity.
  Date/Author: 2026-04-07 / assistant + user

- Decision: stop validation at the targeted Slurm-facing unittest suite for the final implementation pass.
  Rationale: after the implementation was complete, the user explicitly instructed that broader or full unittest runs were out of scope and that no further out-of-scope fixes should be made.
  Date/Author: 2026-04-08 / assistant + user

## Outcomes & Retrospective

The public `movebench slurm` workflows now launch `sbatch` job arrays with `--wait`, `--parsable`, and `--requeue` instead of per-task `srun` commands. The shared backend lives in `movebench/slurm/launcher.py`, replays literal worker argument vectors from hashed JSON array manifests, and keeps the public CLI tree unchanged by using the hidden `python -m movebench.slurm.launcher --array-manifest ...` runtime.

`extract` and `pack` each submit one array per launcher invocation, while `bench` groups pending runs by resolved `SlurmResources` and allocates per-bucket `%K` throttles deterministically before submitting one array per bucket. The targeted Slurm-facing suite passed after the migration with `Ran 41 tests in 1.372s` / `OK`, covering the shared launcher backend, all three public Slurm launchers, the public parser contract, and the existing bench resolved-manifest worker path.

The remaining gap relative to the original plan is only validation breadth, not implementation scope. Broader or full-repository unittest runs were intentionally not rerun after the user restricted the task to the targeted Slurm-facing suite and asked not to touch out-of-scope surfaces.

## Context and Orientation

The current Slurm implementation is a local queue of `srun` commands. `movebench/slurm/launcher.py` registers shared flags such as `--partition`, `--qos`, `--account`, `--time`, `--mem`, `--cpus-per-task`, `--gpus-per-node`, `--max-concurrent-jobs`, `--polling-interval`, `--log-dir`, and `--dry-run`. It then builds one `srun` command per worker and keeps at most `--max-concurrent-jobs` `subprocess.Popen(...)` workers running at once.

The three public workflow launchers are thin wrappers over that shared queue. `movebench/slurm/extract.py` plans one job per `(study_id, ExtractionRun)` pair after skip and prerequisite checks. `movebench/slurm/pack.py` plans one job per `PackJob`, which is one study/timescale pair plus one or more split keys. `movebench/slurm/bench.py` plans one job per pending benchmark `run_key`, where each task reloads the exact run payload from the resolved manifest in `movebench/bench/run.py`.

Three existing details make this migration smaller than it first appears. First, each workflow already has a hidden worker mode. `extract` already accepts `--worker-study`, `--worker-layers-config`, and `--worker-sampler-config`. `pack` already accepts `--worker-study` and `--worker-timescale`. `bench` already accepts `--manifest-path` and `--run-key`. Second, the public CLI registration in `movebench/cli/slurm.py` is already thin, so the migration can stay inside `movebench/slurm/`. Third, the shared config files in `configs/slurm/beery.yaml` and `configs/slurm/shared.yaml` already express one set of Slurm defaults per workflow, so only the meaning of the launcher backend changes, not the overall user-facing configuration shape.

This plan uses three terms repeatedly. A “job array” is one `sbatch --array=...` submission whose tasks differ only by array index. A “resource bucket” is a group of planned tasks that share the exact same resolved `SlurmResources` tuple and therefore can safely live in the same array. An “array manifest” is a small JSON file written by the launcher that stores the literal worker argument vector for each array index. The shared internal launcher runtime reads that file, looks up `SLURM_ARRAY_TASK_ID`, and runs only the one worker command for that index.

The cluster-specific assumptions for this plan were verified on 2026-04-07. `sbatch` supports `--array`, `--requeue`, `--parsable`, and `--wait`. The scheduler reports `PreemptMode=REQUEUE` and `JobRequeue=0`, which means requeue behavior is available but must be explicitly requested on submit. The scheduler also reports `MaxArraySize=20000`, which is comfortably above the sizes implied by the current MoveBench Slurm launchers. This plan treats that array-size headroom as a v1 assumption and does not add fallback chunking logic for larger launches. The association and QoS queries available to this user did not expose explicit `MaxJobs` or `MaxSubmitJobs` limits, so the repo should continue to own its own concurrency limit instead of assuming the cluster will do it.

The key files for implementation are:

- [movebench/slurm/launcher.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/launcher.py), which currently owns the shared `srun` helpers and will become the shared `sbatch` array backend plus the hidden internal array-task entrypoint.
- [movebench/slurm/extract.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/extract.py), [movebench/slurm/pack.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/pack.py), and [movebench/slurm/bench.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/bench.py), which will keep their planning logic but stop constructing one `srun` command per task.
- [movebench/bench/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run.py), which already provides the resolved-manifest worker contract that `bench` should keep using.
- [tests/test_pack.py](/data/vision/beery/scratch/justin/movebench/tests/test_pack.py), which should keep the public CLI parser coverage but should no longer absorb more shared launcher tests.
- [tests/test_slurm_launcher.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_launcher.py), which should be added for the shared array backend and hidden internal runtime.
- [tests/test_slurm_extract.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_extract.py), [tests/test_slurm_pack.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_pack.py), and [tests/test_slurm_bench.py](/data/vision/beery/scratch/justin/movebench/tests/test_slurm_bench.py), which must flip from `srun` expectations to `sbatch` array expectations.

## Plan of Work

Milestone 1 updates the tests first. Keep the existing public parser assertion in `tests/test_pack.py` unchanged so the public `movebench slurm` tree remains `bench`, `extract`, and `pack`, but move the shared launcher coverage into a new `tests/test_slurm_launcher.py`. In that new file, replace the `run_srun_queue(...)` expectations with tests for the new shared array backend. Add one test that writes a temporary array manifest, sets `SLURM_ARRAY_TASK_ID`, and verifies that `movebench.slurm.launcher.main(["--array-manifest", <path>])` selects the correct worker argument vector and forwards it into `movebench.cli.main.main(...)`. Add another shared-launcher test that inspects `build_sbatch_command(...)` directly and proves the returned outer argv contains exactly one `--wrap` value whose content is the internal launcher command rendered as one shell-safe string, rather than splitting the wrapped command into multiple outer argv entries or writing a temporary batch script. Add another shared-launcher test for dry-run output that asserts `sbatch --wait --parsable --requeue --array=...` appears and that the underlying worker commands are still printed for readability. Keep these as direct function-level tests rather than subprocess tests so the new coverage stays small and deterministic.

In `tests/test_slurm_extract.py` and `tests/test_slurm_pack.py`, keep the existing planning fixtures and worker tests, but change the launcher dry-run expectations. Each should now expect one array submission rather than one printed `srun` command per task. The dry-run output should still include the worker-mode arguments so the user can inspect what each task would run. In `tests/test_slurm_bench.py`, change the mixed-method dry-run test so it expects two array submissions by default, one for CPU methods and one for GPU methods, both using `--requeue`. Also add a bench-specific test for the resource-bucket cap allocator so the split of `--max-concurrent-jobs` across buckets is deterministic and documented by the test.

When these new tests are first added, they should fail against the current code for straightforward reasons: `movebench/slurm/launcher.py` still emits `srun`, there is no array-manifest runtime, and `bench` still prints one launch command per pending run instead of one array command per resource bucket.

Milestone 2 replaces the shared launcher backend in `movebench/slurm/launcher.py`. Keep `SlurmResources`, `register_common_slurm_arguments(...)`, and `apply_slurm_config(...)` as the stable entrypoints for argument and config handling. Remove the `srun`-specific helpers and replace them with one small shared array backend. The shared backend should define one small request object, for example `ArrayRequest`, that contains the resolved `SlurmResources`, the array concurrency for this request, the list of worker argument vectors for this array, and the Slurm log path pattern for this array.

The launcher backend should then implement four concrete responsibilities and nothing more. First, render Slurm resource flags from `SlurmResources` into a list that can be reused by every array submission. Second, write the array manifest JSON for one request. The manifest should be a stable, simple structure with a top-level `tasks` list where each entry contains only `worker_args`. Manifest paths should be absolute paths under the existing workflow log root, specifically `Path(args.log_dir) / "manifests" / <label>__<hash>.json`, so the batch wrapper does not depend on the submitter’s working directory. Dry-run must skip all writes, including manifest creation and directory creation. Third, build the actual `sbatch` command:

    sbatch --wait --parsable --requeue --export=ALL --array=0-(N-1)%K \
      --output <log_pattern> --error <log_pattern> \
      <resource flags> \
      --wrap "<sys.executable> -m movebench.slurm.launcher --array-manifest <absolute_manifest_path>"

Fourth, run the common queue/poll loop for those array requests. This loop should be much smaller than today’s `run_srun_queue(...)`: it should not maintain a per-task pool or a second concurrency cap. Instead, it should submit every `ArrayRequest` immediately, because each request already carries its own `%K` throttle and `bench` will ensure that the sum of those throttles does not exceed `--max-concurrent-jobs`. Each local process is therefore only one waiting `sbatch --wait` submitter. On dry-run, print the array command and the underlying worker commands for each task. On real execution, start each `sbatch --wait --parsable` process with `subprocess.Popen(...)`, read the first stdout line immediately to capture the job id, print that job id, then poll the waiting processes on `--polling-interval` and return `0` on success, `2` if any array submission exits nonzero, and `130` on keyboard interrupt. If one array fails, continue waiting for the others and report failure at the end; do not add cancellation logic.

Implement that third responsibility in one specific way. Build the outer `sbatch` submission as a normal `list[str]` for `subprocess.Popen(...)`; do not use `shell=True`. Build the inner launcher command as its own argv list, for example `[sys.executable, "-m", "movebench.slurm.launcher", "--array-manifest", str(manifest_path)]`, then convert only that inner argv list into one shell-safe string with `shlex.join(...)` and pass that single string as the value of `--wrap`. Do not introduce generated temporary batch-script files in v1.

The same module should also gain a tiny internal runtime entrypoint that is not part of the public CLI tree. The simplest shape is `python -m movebench.slurm.launcher --array-manifest <path>`. Its only job is to read `SLURM_ARRAY_TASK_ID`, select the corresponding `worker_args` entry from the manifest, and invoke `movebench.cli.main.main(["movebench", *worker_args])`. Do not introduce a new public `movebench slurm` subcommand for this. Keeping the internal runtime in `movebench/slurm/launcher.py` avoids changing the public parser contract or adding a second internal dispatch surface.

Milestone 3 migrates `extract` and `pack`. In [movebench/slurm/extract.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/extract.py), keep the existing study/run planning, skip logic, GEE raster checks, known-download-failure checks, and existing worker mode untouched. The only launcher change is that `_run_launcher(...)` should collect worker argument vectors instead of fully built `srun` commands. After the existing filters run, create one `ArrayRequest` with those worker arguments, the current shared `SlurmResources`, and array concurrency equal to `--max-concurrent-jobs`. Reuse the current workflow log root under `--log-dir`, for example `results/slurm/extract_points/<run-group>/%A_%a.log`, and place the array manifest for that request under `results/slurm/extract_points/manifests/`. Then call the shared `sbatch` queue helper.

In [movebench/slurm/pack.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/pack.py), do the same. Keep `resolve_pack_jobs(...)`, `_make_pack_kwargs(...)`, and the existing worker mode. Replace the per-job `build_srun_command(...)` construction with one list of worker argument vectors and one `ArrayRequest`. The job unit does not change: one study/timescale pair per task. The launcher still uses `resolve_pack_jobs(...)` exactly once and still prints “No pending study/timescale jobs.” when there is nothing to submit. Reuse the current workflow log root under `results/slurm/pack/` and place the array manifest under `results/slurm/pack/manifests/`.

Milestone 4 migrates `bench`. Keep the existing `movebench/bench/run.resolve_pending_runs(...)`, resolved-manifest writing, `load_resolved_run(...)`, and worker mode untouched. `bench` already computes task-specific resources in `_resources_for_run(...)`; reuse that function and group pending runs by the full resolved `SlurmResources` tuple, not by method name. This matters because explicit CLI overrides such as `--gpus-per-node 0` or `--mem 128GB` should naturally collapse runs into fewer buckets without new special cases. With the current defaults, this grouping will usually yield one CPU bucket for baselines and `issf_*` methods and one GPU bucket for `deepssf` and `moveformer`.

Keep the shared launcher generic by doing the bucketing and cap split inside `movebench/slurm/bench.py`. The cap allocator should be intentionally simple and deterministic. If only one bucket exists, give it `min(len(bucket_tasks), --max-concurrent-jobs)`. If multiple non-empty buckets exist, give each bucket one slot to start, then distribute the remaining slots one at a time to the buckets with the most remaining unallocated tasks, never giving a bucket more slots than it has tasks. This preserves a strict upper bound on total concurrent tasks across all submitted arrays without adding a second scheduler inside the repo. The allocator should be local to `bench.py` because `extract` and `pack` do not need it. This plan explicitly treats the case where the number of non-empty buckets exceeds `--max-concurrent-jobs` as out of scope for v1; current defaults are expected to stay below that threshold.

Once each bucket has a concurrency cap, create one `ArrayRequest` per bucket. Each task entry should still be the existing hidden worker invocation:

    slurm bench run --manifest-path <resolved_manifest> --run-key <run_key> [--rerun]

Dry-run output should print one array command per bucket and should still print every underlying worker invocation so a user can see which runs landed in which bucket. For bench, keep the workflow log root at `results/slurm/bench/` and place the array manifests under `results/slurm/bench/manifests/`.

Milestone 5 is intentionally small. Update [configs/slurm/beery.yaml](/data/vision/beery/scratch/justin/movebench/configs/slurm/beery.yaml) and [configs/slurm/shared.yaml](/data/vision/beery/scratch/justin/movebench/configs/slurm/shared.yaml) only if implementation work proves they are still out of sync with the chosen `results/slurm/bench` convention. Do not add README work to this plan and do not change the config schema.

## Concrete Steps

Work from the repository root and use the MoveBench micromamba environment:

    cd /data/vision/beery/scratch/justin/movebench

Start by updating the targeted tests for the shared launcher and all three workflow launchers, then run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v \
      tests.test_pack \
      tests.test_slurm_launcher \
      tests.test_slurm_extract \
      tests.test_slurm_pack \
      tests.test_slurm_bench \
      tests.bench.test_runner

Expected pre-implementation result after the tests are updated:

    The suite fails because the current launcher still prints `srun`, there is no
    `python -m movebench.slurm.launcher --array-manifest ...` runtime yet, and
    `bench` still emits one launch command per pending run instead of one array
    command per resource bucket.

After the shared launcher backend lands, rerun the same targeted suite before touching the next workflow. After `extract` and `pack` are migrated, rerun the same suite again. After `bench` is migrated, rerun the same suite a third time and record the actual passing output in this plan.

When the code changes are complete, run the broader Slurm-facing regression surface:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v \
      tests.test_pack \
      tests.test_slurm_launcher \
      tests.test_cli_extract \
      tests.test_slurm_extract \
      tests.test_slurm_pack \
      tests.test_slurm_bench \
      tests.bench.test_experiment_config \
      tests.bench.test_runner

## Validation and Acceptance

Acceptance is reached when the following behaviors are all observable.

First, the public CLI surface is unchanged: `movebench slurm` still exposes only `bench`, `extract`, and `pack`, and the hidden array-task runtime lives only in `python -m movebench.slurm.launcher`.

Second, `tests.test_slurm_launcher`, `tests.test_slurm_extract`, `tests.test_slurm_pack`, and `tests.test_slurm_bench` all pass with dry-run expectations that show `sbatch --wait --parsable --requeue --array=...` instead of `srun ...`. `tests.test_pack` also still passes with the public CLI parser coverage intact, and `tests.test_slurm_launcher` proves both that the array manifest runtime picks the right task from `SLURM_ARRAY_TASK_ID` and that `build_sbatch_command(...)` returns one outer argv entry for `--wrap` whose value is the shell-safe internal launcher command string.

Third, `movebench slurm extract ... --dry-run` and `movebench slurm pack ... --dry-run` each print one array submission with array concurrency equal to `--max-concurrent-jobs`, and they still print the underlying worker arguments for each task so the user can inspect them.

Fourth, `movebench slurm bench run ... --dry-run` prints one array when every run resolves to one resource profile, and prints multiple arrays when runs resolve to different resource profiles. With the current default method mapping, a mixed `issf` plus `deepssf` dry-run should print one CPU array and one GPU array. The sum of the printed per-array `%K` throttles must never exceed the requested `--max-concurrent-jobs`.

Fifth, the shared launcher unit tests demonstrate the required non-dry-run control flow: each array submission prints its Slurm job id immediately, the launcher keeps waiting for all submitted arrays, and the overall command returns a nonzero exit code if any array submission finishes nonzero. A real scheduler-backed smoke test is useful follow-up validation when a known-local workload is available, but it is not required acceptance for this plan.

## Idempotence and Recovery

The migration should remain additive and safe. Array manifests are plain JSON files and should live under each workflow’s existing `--log-dir` in a `manifests/` subdirectory. Their filenames should include a stable hash so a second launch does not have to invent cleanup logic for a previous interrupted launch. Rewriting them is safe because they are launch artifacts, not canonical experiment state.

Because the new launcher remains blocking via `sbatch --wait`, the duplicate-submission risk stays close to the current `srun` behavior. If the launcher is interrupted, some Slurm arrays may still continue running. The implementation should preserve the current interruption warning pattern so the user is told that jobs may still be running. Recovery is then the same as today: either wait for the running jobs, inspect them with Slurm tools, or cancel them explicitly before resubmitting.

This plan must not change the internal worker identities or output locations beyond the Slurm wrapper logs. `bench` must keep using the existing resolved manifest and `run_key` contract. `extract` and `pack` must keep their current skip and resume behavior, because those are the safety nets that make repeated submission attempts tolerable.

## Artifacts and Notes

The shared backend should produce dry-run output that is human-readable and still exposes the real worker commands. A concise target shape is:

    [bench] Dry-run: would launch 2 array jobs (max_concurrent=32).
    [bench] ARRAY bucket=0 tasks=24 concurrency=31
    [bench] sbatch --wait --parsable --requeue --export=ALL --array=0-23%31 ...
    [bench] task[0] python -m movebench.cli.main slurm bench run --manifest-path ... --run-key ...
    [bench] ARRAY bucket=1 tasks=3 concurrency=1
    [bench] sbatch --wait --parsable --requeue --export=ALL --array=0-2%1 ...
    [bench] task[0] python -m movebench.cli.main slurm bench run --manifest-path ... --run-key ...

For `extract` and `pack`, the same shape applies but there should normally be only one printed array request. Dry-run should print every task line and should not create manifests or directories.

## Interfaces and Dependencies

In [movebench/slurm/launcher.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/launcher.py), keep these existing interfaces stable:

    @dataclass(frozen=True)
    class SlurmResources:
        partition: str | None = None
        qos: str | None = None
        account: str | None = None
        time: str | None = None
        mem: str | None = None
        cpus_per_task: int = 1
        gpus_per_node: int = 0

    def register_common_slurm_arguments(parser: argparse.ArgumentParser) -> None: ...
    def apply_slurm_config(
        args: argparse.Namespace,
        *,
        command_name: str,
        defaults: dict[str, object] | None = None,
    ) -> None: ...

Add one small shared request type and the new `sbatch` helpers, for example:

    @dataclass(frozen=True)
    class ArrayRequest:
        resources: SlurmResources
        array_concurrency: int
        log_path: Path
        worker_args_list: tuple[tuple[str, ...], ...]

    def build_sbatch_command(*, request: ArrayRequest, manifest_path: Path) -> list[str]: ...
    def run_sbatch_queue(
        *,
        label: str,
        requests: Sequence[ArrayRequest],
        polling_interval: float,
        dry_run: bool,
    ) -> int: ...
    def main(argv: list[str] | None = None) -> int: ...

`build_sbatch_command(...)` must return the outer `sbatch` argv list that will be passed directly to `subprocess.Popen(...)`. Its `--wrap` payload must be exactly one string argument built from the internal launcher argv via `shlex.join(...)`; do not use `shell=True` for the outer submission and do not generate temporary batch-script files.

`main(...)` is the internal array-task runtime, not a public CLI command. It must read `SLURM_ARRAY_TASK_ID`, load the manifest entry for that index, and call `movebench.cli.main.main(...)` with the stored worker arguments.

In [movebench/slurm/bench.py](/data/vision/beery/scratch/justin/movebench/movebench/slurm/bench.py), keep `_resources_for_run(...)` and the existing worker args contract, and add only local helpers for bucket grouping and bucket-cap allocation, for example:

    def _bucket_pending_runs(...) -> list[tuple[SlurmResources, list[dict[str, object]]]]: ...
    def _allocate_bucket_caps(task_counts: list[int], requested_cap: int) -> list[int]: ...

Do not move that logic into the shared launcher unless `extract` or `pack` later need the same behavior.

No new third-party Python dependencies are needed. The implementation should continue using the system Slurm commands already available on this cluster, especially `sbatch`.

Revision note (2026-04-07 / assistant): created this plan after confirming the cluster supports `sbatch --array`, `--requeue`, and `--wait`, and after the user clarified that `extract` and `pack` must transition alongside `bench`.
Revision note (2026-04-07 / assistant): revised the plan after review to clarify that v1 preserves blocking semantics with `sbatch --wait` but removes the old per-task pool, writes hashed array manifests under each workflow’s existing Slurm log root, moves shared launcher tests into `tests/test_slurm_launcher.py`, drops README work, and narrows required acceptance to deterministic unit tests plus dry-run behavior.
Revision note (2026-04-07 / assistant): revised the plan after review to make the `sbatch --wrap` contract explicit: keep the outer submission as a `subprocess.Popen(...)` argv list, build the inner runtime command separately, convert it to one shell-safe string with `shlex.join(...)`, and do not add temporary batch-script files in v1.
Revision note (2026-04-08 / assistant): updated the plan after implementation to mark the shared launcher migration complete, record the passing targeted Slurm-facing unittest run, and note the later user instruction that broader/full unittest runs and out-of-scope follow-up fixes should not be performed in this task.
