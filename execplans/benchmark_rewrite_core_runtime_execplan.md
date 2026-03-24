# Build the Core Runtime, Planning, and Tracking Layer

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.MD](/data/vision/beery/scratch/justin/movebench/PLANS.MD).

This plan is the first implementation stage of [execplans/benchmark_rewrite_overview_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_overview_execplan.md). Finish this plan before implementing the method runtime or reporting plans.

## Purpose / Big Picture

After this change, a user will be able to express the benchmark suite in plain Python, expand it into stable concrete runs, write resolved run manifests, and inspect which studies and timescales are runnable from existing artifacts before any model training begins. The visible proof is that `movebench bench plan ...` and `movebench bench status` work against the new `bench_v2` runtime and no longer depend on legacy YAML benchmark files.

This plan does not train any models yet. It builds the control plane: plain Python experiment configs, concrete run planning, MLflow bootstrapping, and readiness scanning over pre-existing point stores and deepSSF caches. Extraction, split correction, and cache building are outside the scope of this plan.

## Progress

- [x] (2026-03-24 17:10Z) Reviewed the current benchmark runner, `Study` API, point-store loaders, and deepSSF cache helpers to identify the control-plane responsibilities that must be replaced.
- [x] (2026-03-24 17:30Z) Verified that the existing point-store layout already exposes stable manifests and tensor files under split-specific directories, which is sufficient for runtime validation and loading.
- [x] (2026-03-24 17:40Z) Verified that most studies have both `test-temporal` and `test-individual`, but some studies have only one held-out split and must be supported without special casing in experiment configs.
- [x] (2026-03-24 17:50Z) Verified that the current deepSSF WebDataset cache helpers already define a stable on-disk manifest and shard layout that the new runtime can treat as a prerequisite.
- [x] (2026-03-24 21:10Z) Created the `movebench/bench_v2/` package skeleton, added the `bench` extra with `mlflow`, added `experiments/v2/` experiment modules, and registered the `bench` CLI entrypoint in [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py).
- [x] (2026-03-24 21:10Z) Implemented plain Python experiment config normalization, timescale and covariate expansion, blocked-run detection, stable `run_key` generation, and resolved manifest writing under `results/bench_v2/resolved_experiments/`.
- [x] (2026-03-24 21:10Z) Implemented MLflow bootstrap and query helpers for experiment creation, run startup, full run search, and latest-run lookup by `run_key`.
- [x] (2026-03-24 21:10Z) Implemented study-timescale readiness scanning plus `movebench bench status --format text`, then verified the scoped runtime with `python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status`.

## Surprises & Discoveries

- Observation: the existing point-store layout already encodes split membership in the directory structure.
  Evidence: [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) resolves stores under `features/points/<split>/<rasters_hash>/<sampler_hash>/` and loads them by manifest plus `shared.pt`.

- Observation: the current benchmark code assumes only `train` and `test`.
  Evidence: [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py) passes `splits=["train", "test"]` into preparation functions.

- Observation: deepSSF caches already have a readably structured manifest and directory contract.
  Evidence: [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py) stores caches under `cache/deepssf_wds/<split>/<cache_hash>/` with a manifest plus shard directory, which is enough for readiness checks.

- Observation: the project does not currently have an experiment-planning abstraction at all.
  Evidence: [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) registers only `download`, `extract`, and `ingest`.

- Observation: the current local environment cannot import the repository’s raster stack because the installed `rasterio` binary expects a newer `libstdc++` ABI than the host provides.
  Evidence: the first scoped test run failed while importing `movebench` with `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found ... libgdal.so.38`.

## Decision Log

- Decision: the new benchmark runtime starts after extraction and cache building.
  Rationale: the user explicitly wants this rewrite to consume existing point stores and WDS caches on disk, not redesign or orchestrate extraction.
  Date/Author: 2026-03-24 / assistant

- Decision: experiments will be ordinary Python modules under `experiments/v2/`, each exporting one plain Python `EXPERIMENT` dictionary.
  Rationale: this is simpler than a full spec hierarchy and is sufficient for the fixed experiment families the user described.
  Date/Author: 2026-03-24 / assistant

- Decision: there should be no benchmark-owned SQL tables in v1.
  Rationale: MLflow already stores the data the benchmark needs. Status and report code should query MLflow through its API and group by `run_key` instead of introducing a second source of truth.
  Date/Author: 2026-03-24 / assistant

- Decision: the planner should raise an error if an existing `run_key` resolves to a different materialized payload than before.
  Rationale: this prevents accidentally mutating the meaning of a finished run while reusing its key.
  Date/Author: 2026-03-24 / assistant

- Decision: the study readiness scanner will operate per study and per timescale, not only per study.
  Rationale: point stores and deepSSF caches can exist for some timescales but not others, and the user wants operational visibility that is detailed enough for reporting and batch experiment management.
  Date/Author: 2026-03-24 / assistant

## Outcomes & Retrospective

This plan now leaves the repository with a working benchmark control plane under `movebench/bench_v2/`: plain Python experiment modules, concrete-run planning, resolved manifest writing, MLflow setup helpers, a study-timescale readiness scanner, and CLI registration for `movebench bench plan` and `movebench bench status`.

The scoped verification passed with:

    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status

The later implementation plans can now focus on method execution and reporting instead of rebuilding split logic, run identity, or readiness accounting. The only validation gap in this environment is the final smoke CLI command from this plan, which was not run because the existing repository import path still trips an unrelated `rasterio` / `libgdal` ABI mismatch before the CLI can fully import.

## Context and Orientation

The current repository has no standalone planning layer. The closest thing to experiment definition is the legacy YAML under `configs/benchmarks/`, which mixes layer configs, runtime settings, and method lists. The new runtime must replace that with simple Python configs and lightweight planning code.

Relevant existing files are:

- [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py), which currently validates YAML and dispatches per-method runs.
- [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py), which exposes the current high-level study API.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), whose point-store layout is the main artifact contract the new runtime should build around.
- [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py), whose cache-manifest layout is the raster artifact contract the new runtime should validate.

One design constraint is crucial here.

The planning/runtime layer should assume that benchmark-ready artifacts already exist at their current on-disk locations. This plan must not add split-overlay logic, point-store extraction, or WDS-cache building. It should only validate and consume those artifacts.

## Plan of Work

Start by creating the new package structure under `movebench/bench_v2/` with at least `__init__.py`, `config.py`, `run.py`, `report.py`, and `mlflow.py`. Update [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) to add a `bench` extra containing `mlflow` and any small helper dependency that is truly required. Do not add Hydra, OmegaConf, Alembic, any parquet-only dependency, or any web server framework.

Keep file count low. `config.py` should hold the benchmark definitions and run-expansion helpers. `run.py` should hold planning-oriented commands such as `plan` and `status`, plus any small helpers that do not justify their own module. `report.py` is reserved for the later reporting plan. `mlflow.py` should contain only MLflow setup and search helpers. There is no need for a second CLI module; [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) can register the new commands directly from these modules.

Build the lightweight config layer in `movebench/bench_v2/config.py`. Keep this file simple and explicit. It should define dictionaries for the current timescales, covariate sets, and methods, plus one helper that expands an experiment dictionary into concrete run dictionaries. A timescale entry should carry the user-facing key such as `1d`, the numeric `step_time_s`, and the artifact identifiers needed later to locate existing point stores and deepSSF caches. A covariate-set entry should describe how to pick band ids from the full extracted feature list. A method entry should state whether the method trains, whether it requires point-store artifacts only or raster artifacts as well, and where its executor lives.

Create `experiments/v2/main_results.py` and `experiments/v2/covariate_comparison.py` once `config.py` exists. Each file should export one plain `EXPERIMENT` dictionary. `main_results.py` should include all methods and only the `all` covariate set. `covariate_comparison.py` should include `issf_linear`, `issf_mlp`, `deepssf`, and `moveformer`, with covariate sets `all`, `all_environmental`, `alphaearth_only`, and `lite`.

Implement planning inside `movebench/bench_v2/run.py` next. This code should validate an `EXPERIMENT` dictionary, resolve the target studies, inspect existing point-store and deepSSF-cache manifests to detect blocked or missing studies early, and expand the experiment into concrete run dictionaries. Every concrete run must include the resolved `early_stop_split_key`, which should be `test-temporal` when the required held-out artifacts exist there and otherwise the sole available held-out split. The planner should compute a stable `run_key` from the concrete run payload and write `results/bench_v2/resolved_experiments/<experiment_name>.json`.

Implement the MLflow helper layer in one compact module, `movebench/bench_v2/mlflow.py`. It should configure MLflow against a backend URI, create or fetch the global MLflow experiment `movebench_bench_v2`, and expose helper functions for:

    get_or_create_experiment() -> str
    start_benchmark_run(run_payload: dict[str, object]) -> object
    search_latest_runs_by_key(run_keys: list[str]) -> dict[str, object]
    search_all_benchmark_runs(filter_string: str | None = None) -> list[object]

Every concrete run should be represented in MLflow by params and tags, not by a second SQL table. At minimum, every MLflow run must record:

    tags:
      run_key
      study_id
      timescale
      covariate_set
      method
      experiment_kind
      early_stop_split
      status

    params:
      step_time_s
      seed
      any method hparams needed to reproduce the run

The planner itself should not write anything to MLflow. Planning writes only the resolved manifest. Training and evaluation create MLflow runs.

Implement status reporting inside `movebench/bench_v2/run.py` as well. It should scan every available study and every configured timescale, check for existing point-store manifests and deepSSF-cache manifests, and summarize which study-timescale pairs are SSF-ready, raster-ready, or blocked by missing prerequisites. It should also compare planned run keys from the resolved experiment manifests against MLflow runs. The simplest rule is: search MLflow for all runs in the benchmark experiment, group them by `tags.run_key`, keep the newest run for each `run_key`, and treat `tags.status == completed` as finished.

Extend [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) only after the `bench_v2` modules are in place. Register a `bench` subcommand group there directly. `plan` must support `--studies` overrides. `status` must support a text output mode before the later HTML report exists.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the new dependency group and package skeleton first, then create a new test package:

    tests/bench_v2/
      __init__.py
      test_plan.py
      test_mlflow.py
      test_status.py

Start with config and planning tests using tiny fixture experiment dictionaries and fake artifact roots. Run:

    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow

When the status scanner exists, add tests for the readiness categories using temporary directory trees. Run:

    python -m unittest -v tests.bench_v2.test_status

After the scoped implementation is in place, rerun the targeted suite together. The passing command in this environment was:

    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow tests.bench_v2.test_status

The final smoke command for this plan is:

    movebench bench plan experiments/v2/main_results.py --studies all

That command should write a resolved run manifest and print planned counts without trying to import training code.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are observable.

First, `movebench bench plan experiments/v2/main_results.py --studies all` creates one concrete run per valid combination of study, timescale, method, and covariate set. Running it a second time must not change the resolved run keys unless the experiment definition itself changed.

Second, `movebench bench status --format text` must print counts for blocked studies, SSF-ready studies, raster-ready studies, fully runnable study-timescale pairs, and planned versus completed benchmark runs.

## Idempotence and Recovery

Planning writes only manifests, so rerunning the planner is safe. MLflow writes happen later during execution, keyed operationally by the `run_key` tag.

The status scanner must never mutate training state. It may write a status cache later, but the first version should be read-only.

## Artifacts and Notes

The resolved experiment manifest should include both the experiment metadata and the fully materialized concrete runs. That manifest is the easiest artifact for a later reviewer to inspect when checking whether planning is correct.

The status output for one study should be detailed enough to answer questions such as “the 1d point store exists but the 7d deepSSF cache is missing.”

## Interfaces and Dependencies

Define in `movebench/bench_v2/config.py`:

    EXPERIMENT_DEFAULTS: dict[str, object]
    TIMESCALES: dict[str, dict[str, object]]
    COVARIATE_SETS: dict[str, dict[str, object]]
    METHODS: dict[str, dict[str, object]]

Define in `movebench/bench_v2/run.py`:

    def load_experiment(module_path: str) -> dict[str, object]: ...
    def expand_experiment(experiment: dict[str, object], studies_override: list[str] | None = None) -> list[dict[str, object]]: ...
    def write_resolved_manifest(experiment: dict[str, object], runs: list[dict[str, object]]) -> Path: ...
    def status_snapshot() -> dict[str, object]: ...

Plan creation note (2026-03-24 / assistant): initial draft written after validating the current artifact contracts and the absence of an existing planning layer in the repository.

Revision note (2026-03-24 / assistant): updated the living sections after implementing the scoped `bench_v2` runtime, recording the passing targeted test command, and documenting the unrelated local `rasterio` ABI import failure that prevented running the final smoke CLI command in this environment.
