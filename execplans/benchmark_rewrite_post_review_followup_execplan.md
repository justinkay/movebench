# Close the Post-Review `bench_v2` Optional-Dependency, Read-Only Status, and MLflow Metadata Gaps

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan follows the accepted milestone 1-3 work in [execplans/benchmark_rewrite_overview_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_overview_execplan.md) and [execplans/benchmark_rewrite_milestones_1_3_acceptance_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_milestones_1_3_acceptance_execplan.md). The planner, runner, evaluator, and report generator already exist. This follow-up addresses four remaining gaps found during review: the top-level CLI now requires the optional bench dependencies even for non-bench commands, the `bench` extra does not install everything the report path needs, `bench status --format html` is mutating benchmark state even though status is supposed to be read-only, and MLflow runs still omit part of the model configuration required to reproduce non-default runs.

## Purpose / Big Picture

After this change, a user who installs the base package without `.[bench]` will still be able to run non-benchmark commands such as `movebench download`, `movebench extract`, and `movebench ingest` without `mlflow` installed. A user who does install `.[bench]` will get a complete benchmark surface: `movebench bench report` and `movebench bench status --format html` will work without needing the unrelated `apps` extra, `movebench bench status --format html` will remain read-only with respect to run artifacts and MLflow metrics, and each benchmark MLflow run will contain both runtime hyperparameters and the model-shape parameters needed to reproduce the run.

The visible proof is a short set of behaviors. First, importing [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) with `mlflow` intentionally blocked no longer fails. Second, parsing [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) shows that the `bench` extra includes both `mlflow` and `matplotlib`. Third, `movebench bench status --format html` writes the HTML status/report files but does not trigger metric backfill when final metrics are missing. Fourth, `start_benchmark_run(...)` records model-definition parameters such as `issf_mlp` hidden dimensions in MLflow params in addition to the existing runtime hyperparameters.

## Progress

- [x] (2026-03-27) Reviewed the accepted milestone 1-3 tree and isolated four remaining correctness and completeness gaps: eager benchmark imports in the top-level CLI, incomplete `bench` extra packaging, HTML status mutating run state, and incomplete MLflow reproduction params.
- [x] (2026-03-27) Reproduced the optional-dependency regression by blocking `mlflow` and confirming that importing [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) currently raises `ModuleNotFoundError`.
- [x] (2026-03-27) Confirmed that [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) currently places `mlflow` in the `bench` extra but leaves `matplotlib` only in the `apps` extra even though [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) imports it for report generation.
- [x] (2026-03-27) Confirmed that [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) routes `bench status --format html` through `write_report(...)`, and that [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) currently backfills missing metrics by default.
- [x] (2026-03-27) Confirmed that [movebench/bench_v2/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/mlflow.py) logs only `step_time_s`, `seed`, and `method_hparams`, while `method_model` values such as `hidden_dims`, `dropout`, and `image_dim` are currently absent from MLflow params.
- [x] (2026-03-27) Authored this ExecPlan to address the four review findings without reopening the broader benchmark rewrite or the legacy cutover work.
- [x] (2026-03-27 21:15Z) Implemented milestone 1. Added [tests/bench_v2/test_packaging.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_packaging.py), changed [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) to register `bench` lazily with a missing-dependency stub, and added `matplotlib` to the `bench` extra in [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml). Verified with the new packaging tests plus the existing `test_status` and `test_plan` suites.
- [x] (2026-03-27 21:15Z) Implemented milestone 2. Added HTML-status read-only coverage in [tests/bench_v2/test_report_html.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_report_html.py), added an explicit `allow_metric_repair` policy switch to [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py), kept `bench report` on the repair-capable path, and made `bench status --format html` read-only in [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py).
- [x] (2026-03-27 21:15Z) Implemented milestone 3. Extended [tests/bench_v2/test_mlflow.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_mlflow.py) to require `model.*` params, flattened `method_model` into stable MLflow params in [movebench/bench_v2/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/mlflow.py), and reran the reporting slice to confirm the added params did not disturb report queries.
- [x] (2026-03-27 21:15Z) Verified the implemented follow-up with the full 83-test `bench_v2` regression surface, `python -m movebench.cli.main --help`, a sequential `bench status --format html`, and a sequential `bench report`. During validation, running `bench status --format html` and `bench report` concurrently exposed a shared report-directory race that is recorded below as an out-of-scope discovery for this follow-up.

## Surprises & Discoveries

- Observation: the optional-dependency regression is caused by the import chain, not by command execution.
  Evidence: guarding `__import__` so that `mlflow` raises `ModuleNotFoundError` currently makes `import movebench.cli.main` fail before any CLI command is parsed.

- Observation: the current milestone 1 import-safety tests do not cover the top-level CLI module.
  Evidence: the existing import-safety assertion in [tests/bench_v2/test_status.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_status.py) only checks [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py), while [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) still imports the benchmark runtime eagerly.

- Observation: the packaging gap is real user-facing behavior, not only a metadata mismatch.
  Evidence: [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) imports `matplotlib` inside the plotting helpers, so a user following the documented `.[bench]` install path can reach `ModuleNotFoundError: No module named 'matplotlib'` when running `bench report`.

- Observation: the HTML status path currently violates the original read-only status design even when `refresh_metrics=False`.
  Evidence: [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) calls `write_report(refresh_metrics=False)` for `status --format html`, but [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) still evaluates runs when final metric files or MLflow metrics are missing.

- Observation: the missing MLflow params only affect some methods, so the gap is easy to miss in smoke coverage.
  Evidence: `issf_linear`, the baselines, and the current default `moveformer` model shape use empty `method_model` dictionaries, but `issf_mlp` and `deepssf` store meaningful shape-defining settings under `method_model` that never reach MLflow today.

- Observation: the report writer is not concurrency-safe when two commands regenerate the report directory at the same time.
  Evidence: a validation run that started `movebench bench status --format html` and `movebench bench report` in parallel produced `FileNotFoundError: [Errno 2] No such file or directory: 'covariate_comparison'` from `shutil.rmtree(assets_root)` in [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py). Sequential runs of the same commands completed successfully. This race is outside the scope of the four findings addressed by this plan.

## Decision Log

- Decision: fix the optional-dependency regression at the CLI registration boundary instead of making benchmark dependencies hard requirements for the whole package.
  Rationale: the user-facing regression is that non-benchmark commands now require bench-only packages. The clean fix is to let [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) register the `bench` surface lazily and degrade to an informative stub only when bench-only imports are missing.
  Date/Author: 2026-03-27 / assistant

- Decision: keep [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) as the single HTML writer and separate “repair missing metrics” from “render status” with an explicit flag.
  Rationale: duplicating the report writer for status would create drift. The missing behavior is a policy distinction, not a rendering distinction.
  Date/Author: 2026-03-27 / assistant

- Decision: add `matplotlib` to the `bench` extra and allow it to remain duplicated in the `apps` extra for now.
  Rationale: the bug is that `.[bench]` is incomplete. Duplicating a dependency in two extras is acceptable and lower risk than using this follow-up to refactor unrelated app packaging.
  Date/Author: 2026-03-27 / assistant

- Decision: log model-definition settings to MLflow as flat prefixed params such as `model.hidden_dims` and `model.image_dim`, not only as a JSON blob.
  Rationale: flat params remain visible in the MLflow UI and are easy to assert in tests. A single JSON blob would be harder to scan and query.
  Date/Author: 2026-03-27 / assistant

- Decision: only swallow missing optional benchmark dependencies when the missing root module is bench-specific.
  Rationale: a blanket `except ModuleNotFoundError` around benchmark registration would hide genuine code bugs. The stub path should activate only for known bench-only modules such as `mlflow` and `matplotlib`.
  Date/Author: 2026-03-27 / assistant

## Outcomes & Retrospective

This follow-up is now implemented. The top-level CLI no longer imports the benchmark runtime eagerly, so non-benchmark commands remain usable when `mlflow` is absent and the guarded-import path now presents an informative `bench` stub instead of crashing. The `bench` extra now includes `matplotlib`, which closes the mismatch between the documented benchmark install path and the plotting/report code.

The HTML status path now honors the original read-only contract: it still writes `results/bench_v2/report/index.html` and `status_report.json`, but it no longer repairs missing run metrics. `bench report` remains the repair path, including `--refresh-metrics`. MLflow run metadata is also more complete: `method_model` values now appear as visible `model.*` params so non-default `issf_mlp` and `deepssf` configurations are reproducible from MLflow alone.

Validation in this environment included the milestone-targeted suites, the broader 83-test `bench_v2` regression command, `python -m movebench.cli.main --help`, a sequential `bench status --format html`, and a sequential `bench report`, all passing. The only newly discovered gap during implementation was a concurrent report-directory race when two report-writing commands run at the same time. That race was observed only during parallel validation and remains outside the scope of this four-finding follow-up.

## Context and Orientation

The relevant repository areas are small but easy to confuse because they sit on different boundaries.

[movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) is the package entrypoint. It builds the top-level `movebench` parser and currently imports the `bench_v2` runtime at module import time. That eager import is the reason the optional `bench` dependencies leak into every CLI command.

[movebench/bench_v2/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/mlflow.py) owns the benchmark MLflow experiment, run creation, and latest-run queries. This module is the source of the current `mlflow` import requirement and also the place where run params are flattened before being sent to MLflow.

[movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) owns the `bench` subcommands. It delegates HTML generation to the report layer and execution to the runner/evaluator, so this is where the current status-vs-report policy mismatch is surfaced.

[movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py) builds the PNG plots, `status_report.json`, and `index.html`. Today it also repairs missing final metrics, which is correct for `bench report` but not for `bench status --format html`.

[pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) defines the optional dependency groups. In this repository, `bench` is supposed to be the benchmark runtime extra and `apps` is a separate extra for the interactive application surface. The report path belongs to `bench`, not to `apps`.

The most relevant tests today are [tests/bench_v2/test_status.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_status.py), [tests/bench_v2/test_report_html.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_report_html.py), and [tests/bench_v2/test_mlflow.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_mlflow.py). They already cover import-safety for `bench_v2.run`, report refresh behavior, and the existing MLflow helper surface, but they do not yet cover the four review gaps directly.

Three terms matter in this follow-up.

An “optional bench dependency” means a package that should only be required for `movebench bench ...` commands. In the current tree that includes `mlflow` and `matplotlib`.

A “read-only status command” means a command that may inspect manifests, MLflow, and existing report artifacts and may rewrite the generated report files themselves, but must not create benchmark runs, compute final metrics, or mutate MLflow metrics and tags for benchmark runs.

A “reproduction param” means a run parameter stored in MLflow that a human can read from the MLflow UI and use to reconstruct how the model was configured. In this follow-up, `hidden_dims`, `dropout`, and `image_dim` are the concrete missing examples.

## Plan of Work

Milestone 1 restores the intended optional-dependency boundary and fixes the incomplete `bench` extra.

Start by adding a new focused packaging-and-CLI test module at [tests/bench_v2/test_packaging.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_packaging.py). This test file should do two things. First, it should parse [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) with `tomllib` and assert that the `bench` extra contains both `mlflow` and `matplotlib`. Second, it should guard Python imports so `mlflow` and `matplotlib` raise `ModuleNotFoundError`, then import [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py), build the parser, and prove that non-bench parsing still works and that the `bench` command resolves to an informative stub rather than crashing the import.

Then patch [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py). Remove the module-level benchmark import. Add one small helper such as `_register_bench_commands(subparsers)` that attempts to import `movebench.bench_v2.run` only when building the parser. If the import succeeds, register the real benchmark commands exactly as today. If the import fails because the missing root module is `mlflow` or `matplotlib`, register a placeholder `bench` parser that accepts the remainder of the command line and prints a concise installation message such as `Benchmark commands require the bench extra: python -m pip install -e .[bench]`. If the missing root module is anything else, re-raise the error so real code bugs remain visible.

Finally for milestone 1, patch [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) so the `bench` optional dependency group includes `matplotlib` in addition to `mlflow`. Leave the `apps` extra untouched unless a targeted test proves it is necessary to change. The purpose of this milestone is to restore the documented promise that `python -m pip install -e .[bench]` provides a complete benchmark surface.

Milestone 2 restores the read-only contract for HTML status.

Start with targeted failing tests in [tests/bench_v2/test_report_html.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_report_html.py) and, if needed, [tests/bench_v2/test_status.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_status.py). Add one test that calls the HTML status path with a completed run whose final metrics are missing and asserts that the evaluator is not called. Add a second test that keeps the current `bench report` behavior: report generation should still repair missing metrics by default and should still force recomputation when `--refresh-metrics` is passed. These tests should use the existing fake MLflow/report harness rather than adding a second fake stack.

Then patch [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py). Introduce an explicit policy flag on `write_report(...)`, for example `allow_metric_repair: bool = True`. Thread that flag into `_refresh_final_metrics(...)`. When repair is disabled, `_refresh_final_metrics(...)` must become a no-op that returns the existing MLflow run records unchanged and records no refresh attempts. The report should still surface missing final metrics in `status_report.json` and in the HTML table, but it must not compute them.

Patch [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py) last in this milestone. Change `_command_status(...)` so the HTML path calls `write_report(..., refresh_metrics=False, allow_metric_repair=False)`. Leave `_command_report(...)` on the repair-capable path by calling `write_report(..., refresh_metrics=args.refresh_metrics, allow_metric_repair=True)`. This keeps one renderer and two policies: HTML status is read-only, while `bench report` remains the repair-and-regenerate command.

Milestone 3 makes MLflow run metadata complete enough for non-default model configurations.

Start with failing targeted tests in [tests/bench_v2/test_mlflow.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_mlflow.py). Extend the current `start_benchmark_run(...)` test so it uses a payload with non-empty `method_model`, for example `{"hidden_dims": [128], "dropout": 0.6}` for `issf_mlp`, and assert that those fields are logged as MLflow params with stable prefixed keys. Keep the current assertions for tags and runtime params intact.

Then patch [movebench/bench_v2/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/mlflow.py). Add one small helper that converts method-model fields into flat MLflow params. Scalar values should be converted to their obvious string or numeric param representation. Lists and other structured values should be JSON-encoded with stable ordering and stored under explicit keys such as `model.hidden_dims`. Merge these model params into the existing params dictionary before calling `mlflow.log_params(...)`. Do not rename the existing `step_time_s`, `seed`, or runtime hyperparameter keys in this follow-up because the review finding is about missing information, not about re-keying the whole tracking schema.

Finish milestone 3 by re-running the benchmark report/status tests to confirm the metadata change does not disturb report queries, then update the living sections in this plan and the parent benchmark rewrite plans if the implementation changes any accepted milestone statement.

## Concrete Steps

Work from the repository root and use the movebench micromamba environment:

    cd /data/vision/beery/scratch/justin/movebench

Milestone 1 starts with the packaging and CLI tests:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_packaging

The expected outcome after implementation is a short passing transcript:

    test_bench_extra_includes_mlflow_and_matplotlib ... ok
    test_cli_main_imports_without_bench_deps_and_registers_stub ... ok

    ----------------------------------------------------------------------
    Ran 2 tests in <time>

    OK

After the code changes for milestone 1, rerun the benchmark import surface that was already guarded previously:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_status tests.bench_v2.test_plan

Milestone 2 adds the read-only HTML status checks:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_report_html tests.bench_v2.test_status

The key expected behaviors are that `bench report` tests still show refresh calls when metrics are missing, while the new HTML status test proves that no evaluator call occurs on the status path.

Milestone 3 adds the MLflow param completeness test and rechecks the reporting slice:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_mlflow tests.bench_v2.test_report_html tests.bench_v2.test_report_queries

When all milestones are complete, rerun the widest relevant benchmark surface:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_packaging tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner tests.bench_v2.test_energy_score tests.bench_v2.test_eval_runner tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html

Then run the user-visible command checks in the same environment:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main --help
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench status --format html
    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench report

If you need to prove the no-bench-extra import path in this already-provisioned environment, use the guarded-import test rather than trying to uninstall `mlflow` from the shared environment.

## Validation and Acceptance

Acceptance for this follow-up is reached when all of the following behaviors are observable.

First, the base CLI no longer depends on the optional bench extra. A guarded-import test that forces `mlflow` and `matplotlib` to be absent must still be able to import [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py), build the parser, and resolve non-benchmark commands. The same test must show that `movebench bench ...` routes to an informative installation message instead of raising import-time `ModuleNotFoundError`.

Second, the benchmark packaging metadata is complete. Parsing [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) must show `mlflow` and `matplotlib` inside the `bench` optional dependency group. After installing `.[bench]`, the report path must not rely on the `apps` extra for plotting.

Third, `movebench bench status --format html` is read-only with respect to benchmark execution state. When a completed run is missing final metric files or MLflow metrics, the command may still generate `results/bench_v2/report/index.html` and `status_report.json`, but it must not call the evaluator, write `final_metrics/`, or log new MLflow metrics. The HTML and JSON should instead surface the missing metrics as missing.

Fourth, `movebench bench report` remains the repair path. With missing metrics, `bench report` should still backfill them by default, and `bench report --refresh-metrics` should still recompute them even when they already exist.

Fifth, MLflow params for a benchmark run must include both runtime hyperparameters and model-definition fields. For an `issf_mlp` payload with non-default `hidden_dims` and `dropout`, `start_benchmark_run(...)` must log visible params such as `model.hidden_dims` and `model.dropout` in addition to the current runtime params.

## Idempotence and Recovery

This follow-up must preserve the current read-only behavior of planning and status. The new CLI stub path must not create benchmark state. The HTML status path may rewrite report files under `results/bench_v2/report/`, but it must not touch run directories or MLflow metrics.

The packaging change is idempotent. Adding `matplotlib` to the `bench` extra changes installation metadata only; repeating the edit or re-installing `.[bench]` should not create benchmark artifacts or alter run state.

The MLflow param change is additive. Existing historical runs without `model.*` params should remain readable. The report and status paths must not assume those params are present for old runs. Only newly created run attempts after this fix need to contain the additional params.

If the CLI stub registration is implemented incorrectly and begins swallowing unrelated import errors, that is a correctness bug, not an acceptable fallback. The recovery path is to tighten the missing-module filter so only known bench-only optional dependencies trigger the stub path.

## Artifacts and Notes

The current reproductions that motivated this plan are short and should remain useful as implementation checks.

Blocking `mlflow` today still breaks top-level CLI import:

    ModuleNotFoundError
    No module named 'mlflow'

The current `bench` extra in [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) is incomplete for the report path:

    [project.optional-dependencies]
    bench = [
      "mlflow",
    ]

The current `start_benchmark_run(...)` param payload also shows the metadata gap. For a payload with both `method_hparams` and `method_model`, the logged params currently look like:

    {'step_time_s': 86400, 'seed': 0, 'lr': 0.1, 'batch_size': 32}

After milestone 3, the same proof should include model params as well, for example:

    {
      'step_time_s': 86400,
      'seed': 0,
      'lr': 0.1,
      'batch_size': 32,
      'model.hidden_dims': '[128]',
      'model.dropout': 0.6,
    }

The completed validation surface for this implementation was:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_packaging tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status tests.bench_v2.test_trainer tests.bench_v2.test_artifact_resolution tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner tests.bench_v2.test_energy_score tests.bench_v2.test_eval_runner tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html
    Ran 83 tests in 5.748s
    OK

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main --help
    usage: movebench [-h] {download,extract,ingest,bench} ...

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench status --format html
    /data/vision/beery/scratch/justin/movebench/results/bench_v2/report/index.html

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench report
    /data/vision/beery/scratch/justin/movebench/results/bench_v2/report/index.html

## Interfaces and Dependencies

In [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py), define one helper with a stable name such as:

    def _register_bench_commands(subparsers) -> None: ...

This helper must attempt to import `movebench.bench_v2.run` locally, register the real benchmark commands when available, and register an informative stub parser only when the missing dependency root is bench-specific.

In [movebench/bench_v2/report.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/report.py), change the report writer interface to distinguish rendering from repair, for example:

    def write_report(
        *,
        status_snapshot: dict[str, object],
        refresh_metrics: bool = False,
        allow_metric_repair: bool = True,
    ) -> dict[str, object]: ...

`allow_metric_repair=False` must guarantee that no evaluation backfill occurs.

In [movebench/bench_v2/mlflow.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/mlflow.py), keep the public entrypoint:

    def start_benchmark_run(run_payload: dict[str, object]) -> object: ...

but extend its implementation so the logged params include flattened `method_model` fields under stable `model.*` keys in addition to the existing runtime params.

In [tests/bench_v2/test_packaging.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_packaging.py), define focused tests for:

    1. `bench` extra completeness in `pyproject.toml`.
    2. Top-level CLI import and stub registration when bench-only dependencies are absent.

Revision note (2026-03-27 / assistant): created this follow-up ExecPlan after code review found four remaining gaps that were outside the accepted milestone 1-3 test surface: top-level CLI optional-dependency leakage, incomplete `bench` packaging metadata, HTML status mutating benchmark state, and missing model-definition MLflow params.

Revision note (2026-03-27 / assistant): updated the living sections after implementing the plan, recording the passing milestone-targeted and broad regression suites, the successful sequential CLI checks, and the newly discovered but out-of-scope concurrent report-directory race.
