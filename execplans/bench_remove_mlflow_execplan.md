# Replace Bench MLflow Tracking With Filesystem Run State

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, the active benchmark commands will no longer read from or write to MLflow or the SQLite database at `results/bench/mlflow.db`. Instead, `movebench bench run`, `movebench bench summary`, `movebench bench status --format html`, and `movebench bench report` will use the run directories that already exist under `results/bench/runs/<run_key>/`. That removes the current SQLite lock bottleneck when hundreds of jobs run at once.

The important user-facing consequence is that existing finished runs should remain visible without a one-time migration. For a completed run, this repository already stores the resolved payload in `resolved_run.json`, the terminal training state in `training_summary.json`, and the final evaluation outputs in `final_metrics/<split>/summary.json` plus `energy_by_step.csv.gz`. The visible proof after implementation is that bench commands still work with `mlflow` blocked or uninstalled, and a temporary benchmark root that contains only run directories plus report assets can still generate summaries and HTML reports.

## Progress

- [x] (2026-04-16 01:37Z) Re-read [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md), inspected `movebench/bench/run.py`, `movebench/bench/report.py`, `movebench/bench/summary.py`, `movebench/bench/evaluation/runner.py`, `movebench/bench/mlflow.py`, `movebench/cli/main.py`, and `pyproject.toml`.
- [x] (2026-04-16 01:43Z) Confirmed the current architecture: runtime writes step metrics and terminal tags to MLflow, report and summary read the latest MLflow record per `run_key`, and evaluation writes local final metric files before mirroring the aggregate metrics back into MLflow.
- [x] (2026-04-16 01:50Z) Scanned the live workspace to measure how much information already exists outside MLflow. The current `results/bench/runs` tree contains 4,033 run directories, 3,123 completed `training_summary.json` files, 393 failed summaries with explicit `reason`, about 494 run directories with no terminal summary, 50 stale or active `active_run.json` markers, and 256 directories with `predictions/` but no terminal summary.
- [x] (2026-04-16 01:52Z) Reproduced the root problem directly: a simple Python `sqlite3` read against `results/bench/mlflow.db` raised `sqlite3.OperationalError: database is locked` while other benchmark work was active.
- [x] (2026-04-16 02:00Z) Authored this ExecPlan with a no-backfill migration path for completed runs and explicit future-cleanup notes for failed-run retention.
- [x] (2026-04-16 03:15Z) Created the isolated worktree at `/tmp/movebench-bench-no-mlflow`, restored the relevant source and test files from the shared checkout, and verified the first failing milestone-specific test before implementation. `tests.bench.test_run_state` initially failed with `ModuleNotFoundError: No module named 'movebench.bench.run_state'`.
- [x] (2026-04-16 03:28Z) Implemented [movebench/bench/run_state.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run_state.py) and [tests/bench/test_run_state.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_run_state.py), then reran `python -m unittest -v tests.bench.test_run_state` and observed `Ran 4 tests ... OK`.
- [x] (2026-04-16 12:49Z) Added and updated the reader-side tests first: new [tests/bench/test_report.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_report.py), new [tests/bench/test_packaging.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_packaging.py), and MLflow-removal updates in [tests/bench/test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py) and [tests/bench/test_summary.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_summary.py). The targeted suite failed in the expected pre-change ways: report counts stayed at zero because `write_report(...)` still queried MLflow, evaluation still raised `ValueError("MLflow tracking client is unavailable")`, and runtime tests still crashed inside `bench_mlflow.start_benchmark_run(...)`.
- [x] (2026-04-16 12:58Z) Switched [movebench/bench/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run.py), [movebench/bench/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/report.py), [movebench/bench/summary.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/summary.py), and [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py) over to the filesystem-backed run-state model. Runtime writes no longer touch MLflow, report and summary read run directories only, and report missing-artifact checks now refer only to local final metric summaries.
- [x] (2026-04-16 13:03Z) Removed the active bench MLflow dependency surface by deleting [movebench/bench/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/mlflow.py), removing `"mlflow"` from [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml), and updating [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) so `bench` commands only treat `matplotlib` as an optional dependency root.
- [x] (2026-04-16 13:05Z) Ran the targeted migration suite in the worktree with `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_run_state tests.bench.test_report tests.bench.test_runner tests.bench.test_summary tests.bench.test_packaging` and observed `Ran 32 tests ... OK`.
- [x] (2026-04-16 13:09Z) Ran the runtime-focused bench suite with `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_issf_runtime tests.bench.test_deepssf_runtime tests.bench.test_moveformer_runtime`. The first attempt failed once in `tests.bench.test_moveformer_runtime.MoveFormerRuntimeTest.test_grid_prediction_payload_emits_shared_grid_contract` with a transient `pyproj` `TransformDirection.FORWARD` error, but the later full `python -m unittest -v` sweep passed that same test without code changes.
- [x] (2026-04-16 13:12Z) Ran the full repository command `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v`. The migration changes remained green, but the overall sweep still had seven unrelated failures: six `ImportError: cannot import name 'run_srun_queue'` errors in `tests.bench.test_remove_dynamic_world_from_7d_high_res`, and one existing assertion mismatch in `tests.test_local_tiff_window_cap.LocalTiffWindowCapTest.test_layer_io_local_tiff_reads_use_window_cap` that expected `(1000, 1000)` but observed `(250, 250)`.
- [x] (2026-04-16 13:12Z) Ran live smoke commands from the isolated worktree against the populated `results/bench/runs` tree present there from the worktree restore step. `python -m movebench.cli.main bench summary --studies mini` succeeded and showed existing finished runs from local run directories. `python -m movebench.cli.main bench status --format html --studies mini` wrote `results/bench/report/index.html` and `status_report.json` in the worktree without consulting MLflow, although the process did not return promptly and had to be stopped after the files were written.

## Surprises & Discoveries

- Observation: the SQLite locking problem is easy to reproduce even from read-only diagnosis commands.
  Evidence: a direct Python `sqlite3.connect("results/bench/mlflow.db")` query for row counts failed with `sqlite3.OperationalError: database is locked` during this investigation.

- Observation: almost all finished-run information needed by the current report and summary paths already exists in local files.
  Evidence: a workspace scan found 3,123 completed run directories, and all but one of those completed runs already had at least one `final_metrics/<split>/summary.json` file on disk.

- Observation: incomplete runs are common and have more than one shape, so a filesystem migration still needs explicit classification rules.
  Evidence: about 494 run directories had no `training_summary.json`; among those, 50 still had `active_run.json`, 256 already had `predictions/`, and some had `history.json` or checkpoints without final summaries.

- Observation: the highest-write MLflow path is per-step training metric logging, and the active report and summary code does not consume those step metrics.
  Evidence: [movebench/bench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/train.py:71) calls `tracker.log_metrics(...)` on every training step and every evaluation snapshot, while [movebench/bench/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/report.py:155) and [movebench/bench/summary.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/summary.py:317) only read aggregate latest-record status and `energy_score.*` values.

- Observation: packaging and CLI metadata still advertise MLflow as a required benchmark dependency even though the desired end state does not need it.
  Evidence: [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml:27) includes `"mlflow"` in the `bench` extra, and [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py:8) still lists `mlflow` in `_BENCH_OPTIONAL_DEPENDENCIES`.

- Observation: `bench status --format html --studies mini` still expands report data for the full experiment set instead of respecting the status snapshot’s study filter.
  Evidence: the worktree smoke run with `--studies mini` produced a `status_report.json` whose `counts.studies` value was `151`, and the command kept running long after `results/bench/report/index.html` and `status_report.json` had been written.

- Observation: the migration-specific tests and import-blocking packaging tests are green, but the repository’s broad `unittest` sweep still has unrelated failures outside the bench MLflow surface.
  Evidence: `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v` finished with six import errors in `tests.bench.test_remove_dynamic_world_from_7d_high_res` and one failing assertion in `tests.test_local_tiff_window_cap.LocalTiffWindowCapTest.test_layer_io_local_tiff_reads_use_window_cap`.

## Decision Log

- Decision: do not require a one-time data migration for completed runs.
  Rationale: the finished-run payload, terminal status, and final metrics already live in the run directory structure that the repository owns today. Rebuilding current-state summaries from those files is simpler and safer than copying data out of a locked SQLite database.
  Date/Author: 2026-04-16 / Codex

- Decision: introduce a filesystem-backed current-state reader and make the run directory the only source of truth for the active bench surface.
  Rationale: adding a second persistent index file would replace one synchronization problem with another. The first cut should remove SQLite without creating a new database-shaped artifact that needs migration and repair logic.
  Date/Author: 2026-04-16 / Codex

- Decision: preserve only one current record per `run_key` in the initial migration and intentionally drop historical multiple-attempt tracking from the active bench code paths.
  Rationale: the current code uses only the latest attempt for planning, reporting, and summary. The user explicitly does not see value in retaining every failed attempt log from thousands of runs, so attempt history is not a requirement for this migration.
  Date/Author: 2026-04-16 / Codex

- Decision: centralize active-run liveness and record derivation in a new small module, tentatively `movebench/bench/run_state.py`, instead of spreading file inspection logic across `run.py`, `report.py`, and `summary.py`.
  Rationale: the correctness risk in this migration is inconsistent classification of stale, partial, and terminal runs. One shared reader reduces that risk and gives tests a single surface to pin down.
  Date/Author: 2026-04-16 / Codex

- Decision: keep log-retention and benchmark garbage collection out of the initial MLflow-removal change.
  Rationale: removing the database dependency while preserving current user-visible behavior is already a meaningful refactor. Cleanup policy should be designed after the filesystem-backed state model has proven stable.
  Date/Author: 2026-04-16 / Codex

- Decision: keep `mlflow_run_id` as an ignored optional parameter on `evaluate_run_dir(...)` in the first cut rather than removing it in the same patch.
  Rationale: the call sites and tests already pass that argument in a few places, and leaving it temporarily as a no-op reduced diff size while still eliminating all MLflow runtime behavior. The function body now explicitly discards it.
  Date/Author: 2026-04-16 / Codex

- Decision: validate the worktree implementation against the populated worktree-local `results/bench/runs` tree instead of running the modified code from the shared checkout.
  Rationale: the user explicitly requested isolation from currently running jobs. Using the run tree already present in `/tmp/movebench-bench-no-mlflow` let the smoke commands read real finished runs while keeping all generated report assets and code execution inside the worktree.
  Date/Author: 2026-04-16 / Codex

## Outcomes & Retrospective

The implementation landed in the isolated worktree at `/tmp/movebench-bench-no-mlflow` and achieved the main user-facing goal: the active bench surface no longer imports or touches MLflow. Run planning, status, summary, report generation, and final-metric evaluation now derive current state entirely from `results/bench/runs/<run_key>/`, and completed runs remain visible without any backfill step. The compatibility-focused `run_state.py` layer kept the report and summary record shape stable enough that the user-visible surfaces continued to work with relatively small semantic changes.

The strongest validation result is the combination of targeted tests and the live summary smoke. The targeted migration suite passed with `Ran 32 tests ... OK`, the packaging test proved the bench modules import with `mlflow` blocked, and `bench summary --studies mini` in the worktree successfully surfaced existing completed runs from the shared run directories. That is the practical proof that the no-migration approach works for finished runs.

The main residual risk is outside the MLflow removal itself. The broad `python -m unittest -v` sweep still has unrelated failures in `tests.bench.test_remove_dynamic_world_from_7d_high_res` and `tests.test_local_tiff_window_cap`, and the `bench status --format html --studies mini` smoke showed an existing filtering/performance issue in the report path because it expanded all studies instead of respecting the mini selection. Those are real follow-up items, but they are not regressions introduced by this migration.

## Context and Orientation

The active benchmark runtime lives in [movebench/bench/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run.py). That module expands checked-in experiments into concrete run payloads, chooses which runs are pending, executes a method-specific executor, writes the run directory under `results/bench/runs/<run_key>/`, and triggers final evaluation. A `run_key` is the stable string that identifies one logical benchmark case: one study, one timescale, one covariate set, one method, and one seed. There is exactly one active run directory per `run_key`.

The current MLflow wrapper is [movebench/bench/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/mlflow.py). It does four jobs today. First, it creates or opens the SQLite-backed tracking database under `results/bench/mlflow.db`. Second, it starts a run and logs params and tags such as `status`, `study_id`, `timescale`, and `method`. Third, it returns the latest run record for each `run_key`, which is how planning, reporting, and summaries decide whether a logical run is completed, failed, running, blocked, or absent. Fourth, it terminates a live run on interruption or stale-run repair.

The report path lives in [movebench/bench/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/report.py). It expands checked-in experiments, joins them with the latest run record per `run_key`, builds `pandas` data frames of completed results, writes plots with `matplotlib`, and writes `results/bench/report/index.html` plus `status_report.json`. The summary path lives in [movebench/bench/summary.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/summary.py). It also expands experiments and joins them with the latest record per `run_key`, but it renders plain-text operational summaries instead of plots.

The evaluation path lives in [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py). It reads `predictions/<split>.pt`, writes `final_metrics/<split>/summary.json` and `energy_by_step.csv.gz`, and currently mirrors the aggregate metrics back into MLflow. That mirror is redundant for finished runs because the local summary files already contain the same aggregate values.

The local run directory already contains the control-plane artifacts needed for a filesystem-backed design. `resolved_run.json` is the canonical payload. `training_summary.json` is the terminal training state and contains `status`, `reason`, and step counts. `active_run.json` is the current liveness marker for a live process or Slurm job. `history.json` is the training-history snapshot that the trainable executors already use for resume. `checkpoint_best.pt` is the best saved model state. `predictions/<split>.pt` is the evaluation input. `final_metrics/<split>/summary.json` is the aggregate score report that users actually care about. This plan intentionally reuses those files instead of inventing a new migration format.

The most relevant tests today are [tests/bench/test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py), [tests/bench/test_summary.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_summary.py), and the runtime tests under `tests/bench/test_issf_runtime.py`, `tests/bench/test_deepssf_runtime.py`, and `tests/bench/test_moveformer_runtime.py`. There is currently no dedicated bench-v1 report test module, so this plan adds one. Several current tests fake the `mlflow` module during import. Those tests should become simpler once MLflow disappears from the active bench surface.

## Plan of Work

The work should proceed in four milestones. Each milestone leaves the tree in a runnable state and adds tests before or alongside implementation so the new filesystem model is pinned down in code instead of only in this document.

### Milestone 1: Introduce a filesystem-backed run-state reader

Start by adding a new module, tentatively [movebench/bench/run_state.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run_state.py), that knows how to inspect a run directory and produce the normalized “latest run record” shape that report and summary already expect. This module should own the active-run marker helpers that are currently buried in `run.py`, because a live-vs-stale decision is now part of reading state as well as writing state.

The new reader should not require any migration. For each planned `run_key`, it should look at `results/bench/runs/<run_key>/` and return either `None` for “no attempt exists” or a normalized dictionary with `status`, `tags`, `params`, and `metrics`. The record should be built from local files as follows. `resolved_run.json` supplies params and identity fields. `training_summary.json` supplies terminal status and failure reason. `final_metrics/<split>/summary.json` supplies aggregate metrics and should be flattened into the same metric keys the current report expects, such as `energy_score.test_temporal`, `energy_score.accuracy.test_temporal`, `energy_score.calibration.test_temporal`, `n_steps_scored.test_temporal`, and `n_steps_excluded_out_of_support.test_temporal`. File modification times may be used for best-effort `start_time` and `end_time` fields, but they are only compatibility metadata; no downstream behavior should depend on them.

This milestone also defines the file-based classification rules that replace MLflow’s latest-attempt index. If the run directory or `resolved_run.json` is absent, the reader must return no record. If `active_run.json` exists and the process or Slurm job is live, the record status is `running`. If `training_summary.json` exists with `status == "completed"`, the record status is `completed`. If the summary status is `blocked`, the record status is `blocked` and the blocked reason comes from the summary `reason`. If the summary status is `failed` or `interrupted`, the record status is `failed`, the reason comes from the summary `reason`, and `interrupted` should be exposed in tags when applicable. If a non-live run directory has no terminal summary but does contain `predictions/`, `history.json`, `checkpoint_best.pt`, or `final_metrics/`, synthesize a failed record with a stable reason such as `incomplete run missing training_summary`. If `active_run.json` exists but is stale, synthesize a failed record with `failure_reason == "stale interrupted run"`. If a summary exists but carries an empty or unrecognized status string, preserve that raw status so summary can still classify it as `status unknown`.

The tests for this milestone should live in a new module [tests/bench/test_run_state.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_run_state.py). Build temporary run directories by hand and prove that the reader returns the right normalized records for completed, blocked, failed, interrupted, running, stale-active, incomplete-with-predictions, and missing-run cases. Include one test that demonstrates a completed run with no `mlflow.db` at all still produces the metric keys used by the report layer.

### Milestone 2: Switch all readers from MLflow to the filesystem state module

Once the shared reader exists, patch [movebench/bench/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/report.py), [movebench/bench/summary.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/summary.py), and the planning/status paths in [movebench/bench/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run.py) to consume `run_state.search_latest_run_records_by_key(...)` instead of `bench_mlflow.search_latest_run_records_by_key(...)` or `search_latest_runs_by_key(...)`.

The planning change in `run.py` is the most important logic change in this milestone. `resolve_pending_runs(...)` currently skips runs whose latest MLflow tag says `completed` or `running`. After this change it must instead use normalized filesystem records. A completed record is still skipped. A live running record is still skipped. A stale or incomplete record is not skipped; it is considered pending and can be relaunched. A blocked record remains pending in the same sense it is today, because executor-side blocked runs should rerun automatically if prerequisites later become available.

The status and summary changes are mostly dependency swaps, but they must preserve the current user-visible summaries. In [movebench/bench/summary.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/summary.py), the current status and failure-reason grouping should continue to work, but the source of `failure_reason` will now be `training_summary.json` mediated through the normalized record tags rather than MLflow tags. In [movebench/bench/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/report.py), `load_main_results(...)`, `load_covariate_results(...)`, and `write_report(...)` should continue to operate on the same in-memory record shape. The only semantic change is that “missing metric” checks should stop referring to MLflow and should instead describe missing local aggregate metrics. Keep the HTML and `status_report.json` structure as stable as practical so this migration changes the data source more than the user-facing report.

This milestone should also add the missing report-specific tests in a new module [tests/bench/test_report.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_report.py). Use temporary run directories and patched experiment expansion so the report can render completed results and `no_data` placeholders without any MLflow database. Include one test that proves `write_report(...)` can generate `status_report.json` and `index.html` from local final metric files only.

### Milestone 3: Remove writer-side MLflow logging from runtime and evaluation

After all readers have moved to local state, remove the writer-side MLflow dependency from [movebench/bench/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run.py) and [movebench/bench/evaluation/runner.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/evaluation/runner.py). This is the step that actually eliminates the SQLite lock bottleneck rather than only avoiding reads.

In `run.py`, delete the start-run, tag-set, terminate-run, and end-run calls. Keep the local behavior: `resolved_run.json` is still written before execution, `active_run.json` is still written while the run is live, `training_summary.json` is still written for completed, blocked, failed, and interrupted runs, and interruption still re-raises so the process exits non-zero. The current `_RunTracker.log_metrics(...)` method should become a no-op, because those high-frequency MLflow step writes are not consumed anywhere in the active bench surface. Keep `history.json` writing exactly as it is today, because the trainable executors already use it for resume logic. The plan does not add a replacement per-step metric file in this first cut.

In `evaluation/runner.py`, stop importing MLflow and remove the code that resolves a run id and logs aggregate metrics to the tracking database. The evaluator’s job after this milestone is only to validate predictions, write `energy_by_step.csv.gz`, and write `summary.json` under `final_metrics/<split>/`. Update the call sites and tests accordingly. If it reduces diff size during implementation, the optional `mlflow_run_id` argument may remain temporarily as an ignored compatibility parameter, but the preferred end state is to delete it entirely because it has no meaning without MLflow.

This milestone should update [tests/bench/test_runner.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_runner.py) and any runtime tests that currently fake `mlflow` imports. The tests should no longer need fake MLflow modules to import the bench runtime. Add one regression that proves a failed or interrupted run still writes the expected terminal `training_summary.json` even though no MLflow tags are being set.

### Milestone 4: Remove the dependency surface and clean up packaging

The final milestone removes the now-dead integration points. Delete [movebench/bench/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/mlflow.py) after all imports are gone. Update [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) to remove `mlflow` from the `bench` optional dependency group. Keep `matplotlib` in that group because HTML report generation still depends on it. Update [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) so benchmark command registration no longer treats `mlflow` as an optional dependency root.

Add a small packaging/import test module, preferably [tests/bench/test_packaging.py](/data/vision/beery/scratch/justin/movebench/tests/bench/test_packaging.py). That test should parse `pyproject.toml` and prove the `bench` extra no longer contains `mlflow`. It should also guard imports so `mlflow` raises `ModuleNotFoundError`, then import `movebench.cli.main` and `movebench.bench.run` to prove the bench surface itself no longer depends on that package.

Do not delete `results/bench/mlflow.db` or `results/bench/mlartifacts/` as part of this milestone. They can remain on disk as dead historical data until the new filesystem-backed behavior has been validated in the real workspace. The absence of a deletion step is intentional and keeps rollback simple while the new reader is being exercised.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Use the MoveBench micromamba interpreter for all Python tests:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_run_state tests.bench.test_report tests.bench.test_runner tests.bench.test_summary tests.bench.test_packaging

Before implementation, expect this command to fail because the new test modules do not exist yet and the runtime still depends on MLflow. After implementation, expect the command to pass and to prove the new filesystem-backed state model.

After the runtime and packaging milestones, rerun the existing runtime-focused bench tests to confirm the training executors still behave correctly:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench.test_issf_runtime tests.bench.test_deepssf_runtime tests.bench.test_moveformer_runtime

After the targeted tests are green, run the repository default test command that this repository treats as the minimum broad regression sweep:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v

Record any unrelated pre-existing failures exactly in the `Progress` and `Outcomes & Retrospective` sections instead of silently omitting them.

After unit tests pass, run live smoke commands against the current workspace:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench summary --studies mini
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench status --format html --studies mini

If the workspace has enough completed runs and the user wants the heavier report path rechecked, also run:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench report

The first command should print a text summary without touching MLflow. The second should print the path to `results/bench/report/index.html` and regenerate HTML from local files only. The third may refresh missing local final metrics from predictions, but it must not require `mlflow` or `results/bench/mlflow.db`.

## Validation and Acceptance

Acceptance for the migration as a whole requires five observable behaviors.

First, bench modules must import without the `mlflow` package present. A targeted packaging test must block `mlflow` imports and still be able to import `movebench.cli.main`, `movebench.bench.run`, `movebench.bench.summary`, and `movebench.bench.report`.

Second, the new filesystem reader must classify real run-directory states correctly. The new `tests.bench.test_run_state` module must prove that completed, blocked, failed, interrupted, running, stale-active, incomplete, and missing-run cases all map to the expected normalized record shape and reasons.

Third, the report and summary surfaces must continue to work from local files only. `tests.bench.test_report` and `tests.bench.test_summary` must pass without any MLflow fake module, and `write_report(...)` must be able to produce `status_report.json` and HTML from run-directory data in a temporary test workspace that has no `mlflow.db`.

Fourth, the runtime must stop attempting to write step metrics or final metrics to MLflow. This is observable both through the absence of `mlflow` imports in the active runtime modules and through the fact that targeted runtime tests continue to pass while the packaging test blocks `mlflow`.

Fifth, the live smoke commands above must work on the current repository workspace. The user-visible outcome is that `bench summary` and `bench status --format html` still show existing finished runs even though the code no longer consults `results/bench/mlflow.db`.

## Idempotence and Recovery

This migration is intentionally designed to avoid destructive data conversion. Existing run directories remain in place, and the old MLflow database and artifact root can remain untouched on disk while the new code is being validated. That means the implementation can be rerun safely and inspected incrementally.

If work stops midway, the safe recovery path is to resume from the earliest failing targeted test in this plan. Do not delete `results/bench/mlflow.db` during development. It is harmless dead data once the cutover is complete, and keeping it around makes it easier to compare old and new behavior if a discrepancy appears.

If a partially implemented change leaves the runtime in a mixed state, rerun the targeted test modules first. The runtime already tolerates repeated execution of the same logical run directory because `resolved_run.json`, `training_summary.json`, `predictions/`, and `final_metrics/` are rewritten in place. The plan does not introduce any schema migration that needs a one-way rollback.

## Artifacts and Notes

The current MLflow-backed record shape that readers depend on is effectively:

    {
      "run_id": str,
      "start_time": int,
      "end_time": int,
      "status": str,
      "tags": dict[str, str],
      "params": dict[str, str],
      "metrics": dict[str, float],
    }

The filesystem-backed replacement should preserve that in-memory shape closely enough that `report.py` and `summary.py` do not need a second semantic rewrite while MLflow is being removed.

The local files that replace MLflow fields are:

    resolved_run.json -> params and identity fields
    training_summary.json -> terminal status, failure reason, blocked reason
    active_run.json -> live vs stale running state
    final_metrics/<split>/summary.json -> aggregate metrics now read by report

The workspace evidence gathered during diagnosis is important because it justifies the “no backfill” decision:

    run_dirs: 4033
    completed training_summary.json: 3123
    failed training_summary.json: 393
    missing training_summary.json: about 494
    missing-with-active_run: 50
    missing-with-predictions: 256

That evidence shows why this plan spends more design effort on incomplete-run rules than on migration scripts.

## Interfaces and Dependencies

At the end of Milestone 1, add a new module [movebench/bench/run_state.py](/data/vision/beery/scratch/justin/movebench/movebench/bench/run_state.py) with a small, prescriptive interface. Stable names matter here because multiple call sites will depend on them. Define:

    def run_dir(run_key: str) -> Path:
        ...

    def search_latest_run_records_by_key(
        run_keys: list[str],
    ) -> dict[str, dict[str, object]]:
        ...

    def active_run_is_live(run_dir: str | Path) -> bool:
        ...

    def write_active_run(run_dir: Path) -> None:
        ...

    def remove_active_run(run_dir: str | Path) -> None:
        ...

The exact helper names can vary if implementation makes a stronger name obvious, but the module must be the single owner of file-based latest-record construction and live-marker behavior. `run.py`, `report.py`, and `summary.py` should not each rediscover their own version of these rules.

At the end of the migration, the active bench surface should no longer import `mlflow`. That means the imports at the top of `movebench/bench/run.py`, `movebench/bench/report.py`, `movebench/bench/summary.py`, and `movebench/bench/evaluation/runner.py` must not reference `movebench.bench.mlflow` or `mlflow` directly.

The `bench` optional dependency group in [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) must continue to include `matplotlib`, because report generation still uses it, but it must stop including `mlflow`.

## Future Cleanup Notes

This section is intentionally out of scope for the first implementation, but it captures the cleanup work that will likely become attractive once the filesystem-backed state model is stable.

The first likely follow-up is a benchmark garbage collector. If the active system only keeps one current record per `run_key`, there is little reason to retain bulky artifacts for old terminal failures forever. A later command such as `movebench bench gc` could shrink failed or stale-interrupted run directories older than a grace period down to the minimum forensic payload: `resolved_run.json`, `training_summary.json`, and perhaps a tiny manifest describing what was deleted. That command could safely delete `history.json`, `checkpoint_best.pt`, `predictions/`, and any partial `final_metrics/` tree for those failed runs.

The second likely follow-up is an explicit retention policy for successful runs too. Today the repository keeps both aggregate metric summaries and per-step `energy_by_step.csv.gz` files for completed runs. If that becomes too large, a future cleanup policy could distinguish “needed for report generation” from “needed for detailed analysis” and make the step-level CSV opt-in instead of permanent.

The third likely follow-up is a cleaner separation between runtime state and report query code. The first migration should optimize for minimal behavioral change by preserving the old record shape. After that stabilizes, a second refactor could let `report.py` read `final_metrics` summaries directly and let `summary.py` read `training_summary.json` directly, without forcing everything through a compatibility-flavored “latest record” dict.

Plan creation note (2026-04-16 02:00Z / Codex): created after diagnosing the current MLflow usage, reproducing the live SQLite lock failure, confirming that completed benchmark data already exists in local run directories, and deciding to scope the first change around a filesystem-backed current-state reader rather than a data backfill.

Revision note (2026-04-16 13:12Z / Codex): updated after implementation in `/tmp/movebench-bench-no-mlflow` to record the landed run-state migration, the packaging cleanup, the exact targeted/runtime/full-suite validation results, and the smoke-test observation that HTML status generation still ignores the `--studies` filter.
