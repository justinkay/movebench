# Build the Core Runtime, Planning, and Tracking Layer

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.MD](/data/vision/beery/scratch/justin/movebench/PLANS.MD).

This plan is the first implementation stage of [execplans/benchmark_rewrite_overview_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_overview_execplan.md). Finish this plan before implementing the method runtime or reporting plans.

## Purpose / Big Picture

After this change, a user will be able to express the benchmark suite in plain Python, expand it into stable concrete runs, write resolved run manifests, and inspect which studies are blocked or ready before any model training begins. The visible proof is that `movebench bench plan ...` and `movebench bench status` work against the new `bench_v2` runtime and no longer depend on legacy YAML benchmark files.

This plan does not train any models yet. It builds the control plane: a temporary split override shim for `cleaned-v2-splits/`, split-correct step selection for preparation, plain Python experiment configs, concrete run planning, MLflow bootstrapping, and readiness scanning.

## Progress

- [x] (2026-03-24 17:10Z) Reviewed the current benchmark runner, `Study` API, point-store extraction code, dataset wrappers, and the cleaned split files to identify the existing control-plane responsibilities.
- [x] (2026-03-24 17:30Z) Verified that the cleaned split files use the `event-id` column present in `trajectories.csv`, making an exact overlay join possible.
- [x] (2026-03-24 17:40Z) Verified that most studies have both `test-temporal` and `test-individual`, but some studies have only one held-out split and must be supported without special casing in experiment configs.
- [x] (2026-03-24 17:50Z) Verified that the current point extractor can consume an explicit step dataframe, which means split-correct in-memory step selection can drive extraction without reusing the old split-selection wrapper.
- [ ] Create the `movebench/bench_v2/` package skeleton, dependency updates, and CLI entrypoint.
- [ ] Implement the temporary split override loader against `cleaned-v2-splits/`.
- [ ] Implement split-correct in-memory step selection for preparation and evaluation.
- [ ] Implement plain Python experiment config validation and concrete-run planning.
- [ ] Implement MLflow bootstrap plus lightweight query helpers.
- [ ] Implement study readiness scanning and `movebench bench status`.

## Surprises & Discoveries

- Observation: the cleaned split files already carry a `set` column with the exact labels needed by the new runtime.
  Evidence: sample files in `cleaned-v2-splits/` contain `train`, `test-temporal`, `test-individual`, and `unused`, matching the user’s requested terminology.

- Observation: the existing `trajectories.csv` files usually contain the same `event-id` column as the cleaned split files.
  Evidence: sample study files under `/archive/vision/beery/movebench/studies/<study_id>/trajectories.csv` include `event-id`, making a direct overlay safer than fuzzy matching on timestamp and coordinates.

- Observation: the current benchmark code assumes only `train` and `test`.
  Evidence: [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py) passes `splits=["train", "test"]` into preparation functions.

- Observation: studies with invalid cleaned split overrides are rare enough that they should be blocked explicitly rather than tolerated by lossy joins.
  Evidence: 146 of 148 inspected studies were full bijections by `event-id`. That makes “fail closed” the safer default.

- Observation: the project does not currently have an experiment-planning abstraction at all.
  Evidence: [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) registers only `download`, `extract`, and `ingest`.

## Decision Log

- Decision: the cleaned split handling should be a small temporary adapter, not a persisted feature with its own artifact format.
  Rationale: the corrected split labels are expected to move into the normal study directory later, so the implementation should be easy to delete when that happens.
  Date/Author: 2026-03-24 / assistant

- Decision: there should be no standalone step-table artifact.
  Rationale: the step-level data already effectively lives in the point stores and related caches. The runtime only needs a split-correct helper used during extraction and dataset construction.
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
  Rationale: downloads, point stores, and deepSSF caches can finish at different times across timescales, and the user wants operational visibility that is detailed enough for daily automation.
  Date/Author: 2026-03-24 / assistant

## Outcomes & Retrospective

This plan should leave the repository with a new benchmark control plane and no ambiguity about what work exists. If successful, the later implementation plans can focus on method execution and reporting instead of rebuilding split logic, run identity, or database shape from scratch.

## Context and Orientation

The current repository has no standalone planning layer. The closest thing to experiment definition is the legacy YAML under `configs/benchmarks/`, which mixes layer configs, runtime settings, and method lists. The new runtime must replace that with simple Python configs and lightweight planning code.

Relevant existing files are:

- [movebench/benchmark.py](/data/vision/beery/scratch/justin/movebench/movebench/benchmark.py), which currently validates YAML and dispatches per-method runs.
- [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py), which exposes the current high-level study API.
- [movebench/trajectories.py](/data/vision/beery/scratch/justin/movebench/movebench/trajectories.py), which provides the step-selection logic that the new split-correct helper must reuse carefully.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), whose `extract_split_feature_store(...)` helper can later be reused once the split-correct step rows are available.
- [cleaned-v2-splits/](/data/vision/beery/scratch/justin/movebench/cleaned-v2-splits), which is the authoritative source for train and held-out membership.

Two rules are crucial here.

First, benchmark code must load trajectories through one tiny helper that replaces the `set` column from `cleaned-v2-splits/<study_id>.csv` when that file exists. This helper should validate unique `event-id` values on both sides and fail loudly on mismatches. It should not write a new on-disk overlay format.

Second, benchmark step selection must be done on the full split-corrected trajectory dataframe, but a step only belongs to split `X` when both the step start point and the step endpoint are in `X`. The previous point may be outside `X` because `start_bearing` depends on it. This logic should live in a helper used during extraction and dataset construction, not in a separate persisted artifact.

## Plan of Work

Start by creating the new package structure under `movebench/bench_v2/` with at least `__init__.py`, `cli.py`, `config.py`, `data.py`, `plan.py`, `mlflow.py`, and `status.py`. Update [pyproject.toml](/data/vision/beery/scratch/justin/movebench/pyproject.toml) to add a `bench` extra containing `mlflow`, `pyarrow`, and any small helper dependency that is truly required. Do not add Hydra, OmegaConf, Alembic, or any web server framework.

Implement `movebench/bench_v2/data.py` first. Keep this module small. It should expose `load_benchmark_trajectories(study_id)` and one helper for split-correct step selection. `load_benchmark_trajectories(study_id)` should load `trajectories.csv`, look for `cleaned-v2-splits/<study_id>.csv`, and if that file exists replace the `set` column by joining `event-id -> set`. Validate uniqueness on both sides and require a full row-count match. On mismatch, raise a clear error such as “cleaned split file does not match trajectories.csv for study X.” Do not write an overlay artifact to disk. The point of this helper is to be temporary and easy to delete later.

The split-correct step-selection helper should take the full split-corrected dataframe and a requested split name, call the existing interval step-selection logic on the full dataframe, and then keep only rows whose start point and endpoint both belong to the requested split. This is the only subtle split rule the rest of the runtime needs. Keep it as a normal function and do not introduce a standalone step-table artifact.

Build the lightweight config layer in `movebench/bench_v2/config.py`. Keep this file simple and explicit. It should define dictionaries for the current timescales, covariate sets, and methods. A timescale entry should carry the user-facing key such as `1d`, the numeric `step_time_s`, and the layer-config paths needed later by extraction. A covariate-set entry should describe how to pick band ids from the full extracted feature list. A method entry should state whether the method trains and where its executor lives.

Create `experiments/v2/main_results.py` and `experiments/v2/covariate_comparison.py` once `config.py` exists. Each file should export one plain `EXPERIMENT` dictionary. `main_results.py` should include all methods and only the `all` covariate set. `covariate_comparison.py` should include `issf_linear`, `issf_mlp`, `deepssf`, and `moveformer`, with covariate sets `all`, `all_environmental`, `alphaearth_only`, and `lite`.

Implement `movebench/bench_v2/plan.py` next. This module should validate an `EXPERIMENT` dictionary, resolve the target studies, call `load_benchmark_trajectories(study_id)` to detect blocked studies early, and expand the experiment into concrete run dictionaries. Every concrete run must include the resolved `early_stop_split_key`, which should be `test-temporal` when present and otherwise the sole available held-out split. The planner should compute a stable `run_key` from the concrete run payload and write `results/bench_v2/resolved_experiments/<experiment_name>.json`.

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

Implement `movebench/bench_v2/status.py` last in this plan. This module should scan every available study and every configured timescale, use `load_benchmark_trajectories(study_id)` to identify blocked studies, check whether the requested raster downloads appear complete, and check for the later point-store and deepSSF-cache markers. Return a structured object that later report code can render directly. The first version may compute readiness live from the filesystem without persisting anything.

`status.py` should also be able to compare planned run keys from the resolved experiment manifests against MLflow runs. The simplest rule is: search MLflow for all runs in the benchmark experiment, group them by `tags.run_key`, keep the newest run for each `run_key`, and treat `tags.status == completed` as finished. This is enough to answer “how many planned runs are complete?” without custom tables.

Extend `movebench/cli/main.py` only after the `bench_v2` modules are in place. Register a `bench` subcommand group implemented in `movebench/bench_v2/cli.py` with at least `plan` and `status`. `plan` must support `--studies` overrides. `status` must support a text output mode before the later HTML report exists.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Add the new dependency group and package skeleton first, then create a new test package:

    tests/bench_v2/
      __init__.py
      test_data.py
      test_plan.py
      test_mlflow.py
      test_status.py

Start with the temporary split-override tests. Use tiny synthetic CSV fixtures created under `tests/fixtures/bench_v2/` or in temporary directories. Run:

    python -m unittest -v tests.bench_v2.test_data

Add tests that prove the split-correct step-selection rule and previous-point context rule. The smallest useful fixture has at least one step whose previous point lies outside the target partition and one step whose endpoint crosses into a different partition. Run:

    python -m unittest -v tests.bench_v2.test_data

When the planner and MLflow helper layer exist, add tests that prove deduplication, simple run planning, and grouping latest MLflow runs by `run_key`. Run:

    python -m unittest -v tests.bench_v2.test_plan tests.bench_v2.test_mlflow

When the status scanner exists, add tests for the readiness categories using temporary directory trees. Run:

    python -m unittest -v tests.bench_v2.test_status

The final smoke command for this plan is:

    movebench bench plan experiments/v2/main_results.py --studies all

That command should write a resolved run manifest and print planned counts without trying to import training code.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are observable.

First, a study with a perfectly matching cleaned split file loads successfully through `load_benchmark_trajectories(study_id)`, while a deliberately mismatched fixture raises a clear error that the status command reports as blocked.

Second, a step-selection fixture shows the intended eligibility behavior. A step whose start point is `train` and endpoint is `test-temporal` must not be included in the `train` extraction set or the `test-temporal` extraction set. A step whose previous point is outside `train` but whose start and endpoint are both `train` must remain eligible for `train`.

Third, `movebench bench plan experiments/v2/main_results.py --studies all` creates one concrete run per valid combination of study, timescale, method, and covariate set. Running it a second time must not change the resolved run keys unless the experiment definition itself changed.

Fourth, `movebench bench status --format text` must print counts for blocked studies, ready-for-extraction studies, ready-for-SSF studies, ready-for-raster studies, and planned benchmark runs.

## Idempotence and Recovery

The split override helper must be safe to call repeatedly because it writes nothing. If the cleaned split file changes, the next call should simply reflect the new labels or error out if the file no longer matches.

Planning writes only manifests, so rerunning the planner is safe. MLflow writes happen later during execution, keyed operationally by the `run_key` tag.

The status scanner must never mutate training state. It may write a status cache later, but the first version should be read-only.

## Artifacts and Notes

The resolved experiment manifest should include both the experiment metadata and the fully materialized concrete runs. That manifest is the easiest artifact for a later reviewer to inspect when checking whether planning is correct.

The status output for one study should be detailed enough to answer questions such as “the 1d point store exists but the 7d deepSSF cache is missing.”

## Interfaces and Dependencies

Define in `movebench/bench_v2/data.py`:

    def load_benchmark_trajectories(study_id: str) -> pd.DataFrame: ...
    def select_steps_for_split(df: pd.DataFrame, *, split: str, step_time_s: int, gap_q: float, step_q: float, max_step_q: float, step_multiplier: float) -> pd.DataFrame: ...

Define in `movebench/bench_v2/plan.py`:

    def load_experiment(module_path: str) -> dict[str, object]: ...
    def expand_experiment(experiment: dict[str, object], studies_override: list[str] | None = None) -> list[dict[str, object]]: ...
    def write_resolved_manifest(experiment: dict[str, object], runs: list[dict[str, object]]) -> Path: ...

Plan creation note (2026-03-24 / assistant): initial draft written after validating the cleaned split files, the step-selection semantics, and the absence of an existing planning layer in the repository.
