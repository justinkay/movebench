# Build Exact Energy-Score Evaluation, Reporting, and Daily Automation

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.MD](/data/vision/beery/scratch/justin/movebench/PLANS.MD).

This plan depends on [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md) and [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md). Do not begin this plan until concrete runs can be executed and prediction artifacts exist.

## Purpose / Big Picture

After this change, every finished run will have exact final energy scores on every available held-out split, those metrics will be queryable from MLflow, and `movebench bench report` will regenerate a static HTML dashboard showing progress and plots for main results and covariate comparisons. The visible proof is that a user can run the daily automation commands and then open `results/bench_v2/report/index.html` to inspect both status and completed benchmark plots.

This plan is the user-facing finish of the rewrite. It turns run artifacts into trustworthy metrics and an operational dashboard.

## Progress

- [x] (2026-03-24 17:10Z) Located the existing energy-score implementation in `scripts/slurm/compare_outputs.py` and confirmed the exact formula used there.
- [x] (2026-03-24 17:20Z) Verified that candidate-based methods must remove the true endpoint candidate and renormalize probabilities before final held-out energy evaluation.
- [x] (2026-03-24 17:25Z) Verified that deepSSF’s final metric should not discard out-of-bounds targets even though training and early stopping losses do.
- [x] (2026-03-24 17:35Z) Confirmed that the current app is unrelated to benchmark reporting, making a generated static HTML report the simplest and safest presentation layer.
- [ ] Implement exact energy-score evaluation for deterministic, candidate-based, and grid-based methods.
- [ ] Implement final-evaluation orchestration and MLflow logging.
- [ ] Implement plot generation for main results and covariate comparisons.
- [ ] Implement static HTML report generation and status snapshots.
- [ ] Implement daily automation commands and wrapper scripts.
- [ ] Perform the final legacy cutover after the new report layer is validated.

## Surprises & Discoveries

- Observation: the current energy-score script is already carefully excluding the true endpoint from candidate-based evaluation.
  Evidence: [scripts/slurm/compare_outputs.py](/data/vision/beery/scratch/justin/movebench/scripts/slurm/compare_outputs.py) zeroes the true candidate probability for MoveFormer and samples only pseudo-candidates for iSSF-style surfaces.

- Observation: deterministic methods are a special case of energy score and can be evaluated without any sampling.
  Evidence: for a point mass prediction `X = x`, the energy score reduces to `||x - y||`, so `baseline_nomove` and `baseline_constvel` can be exact with almost no compute.

- Observation: deepSSF can be evaluated exactly over the whole grid without sampling if the pairwise term is computed by autocorrelation.
  Evidence: the energy score term `sum_{i,j} p_i p_j ||x_i - x_j||` over a regular grid is a convolution of the probability map with itself against a precomputed distance kernel. FFT makes this exact and tractable.

- Observation: the user’s requested report is better served by a generated static site than by extending the existing app.
  Evidence: the current `app/` directory is a trajectory viewer, not an experiment dashboard, and the user wants an always-updated HTML file that can be checked quickly.

- Observation: some studies have only one held-out split, so the report layer must treat missing plots as “not applicable,” not as failures.
  Evidence: the cleaned split inventory includes studies with only `test-temporal` or only `test-individual`.

## Decision Log

- Decision: final energy scores must be exact for the benchmark support that each method runtime emits.
  Rationale: this is the strongest correctness guarantee compatible with the user’s requirement and the existing discrete-support method designs.
  Date/Author: 2026-03-24 / assistant

- Decision: the random-walk baseline’s final energy score will be exact with respect to the discrete pseudo-candidate support stored for the benchmark step, not with respect to an unbounded continuous distribution.
  Rationale: the benchmark support for candidate-based methods is already discrete and shared with the point-store machinery. This keeps the baseline consistent with the rest of the benchmark and makes exact evaluation tractable.
  Date/Author: 2026-03-24 / assistant

- Decision: main-results plots will show one line per method over ordered timescales, and covariate-comparison plots will be faceted by method with one line per covariate set.
  Rationale: that layout matches the user’s requested comparisons while staying readable when four covariate sets are present.
  Date/Author: 2026-03-24 / assistant

- Decision: the report will be generated as static HTML with embedded links to PNG assets and JSON status summaries.
  Rationale: this is simpler to regenerate daily, easy to browse from shared storage, and avoids coupling benchmark reporting to the unrelated interactive app.
  Date/Author: 2026-03-24 / assistant

- Decision: daily automation will call the same library-backed CLI commands as manual usage, with only thin shell or Python wrappers for scheduling convenience.
  Rationale: the daily path should not become a second implementation of the runtime.
  Date/Author: 2026-03-24 / assistant

## Outcomes & Retrospective

This plan should leave the repository with benchmark outputs that are both trustworthy and easy to inspect. If it succeeds, a reviewer can assess correctness from exact metric tests and assess completeness from the generated report without spelunking through run directories.

## Context and Orientation

The evaluation layer starts from the standardized prediction artifacts created by the method runtime. The reporting layer starts from MLflow runs, resolved experiment manifests, and those final metric artifacts.

Relevant existing files are:

- [scripts/slurm/compare_outputs.py](/data/vision/beery/scratch/justin/movebench/scripts/slurm/compare_outputs.py), which contains the current energy-score math and baseline behavior.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), which stores candidate coordinates for candidate-based methods.
- [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py), whose deepSSF evaluation path documents the current in-bounds loss behavior that must not be confused with final energy evaluation.
- [results/](/data/vision/beery/scratch/justin/movebench/results), which is the natural home for the new static report and run outputs.

An “energy score” in this plan is the scalar

    E ||X - y|| - 0.5 E ||X - X'||

where `X` and `X'` are independent draws from the predictive distribution and `y` is the observed held-out endpoint. Lower is better.

## Plan of Work

Create a new evaluation package under `movebench/bench_v2/evaluation/` with at least `types.py`, `energy.py`, `runner.py`, and `artifacts.py`.

Implement `types.py` first. Define a small number of standardized prediction-artifact shapes that method executors must emit. At minimum there should be one deterministic point prediction type, one discrete-support prediction type with coordinates and probabilities, and one dense-grid prediction type with probability map, pixel size, and origin or grid-coordinate tensors. These types should be plain dataclasses or typed dictionaries that can be serialized to disk and loaded again without importing model code.

Implement `energy.py` next. Add one exact function per prediction family. For deterministic point predictions, return the Euclidean distance to the observed endpoint. For discrete-support predictions, compute

    sum_i p_i ||x_i - y|| - 0.5 sum_{i,j} p_i p_j ||x_i - x_j||

exactly with vectorized tensor code. This path covers `baseline_randomwalk`, `issf_linear`, `issf_mlp`, and `moveformer`. Require that the discrete-support artifact already excludes the true endpoint candidate from its support; if the artifact still includes it, fail loudly. For dense-grid predictions, compute the first term directly over the full grid, then compute the pairwise term by convolving the probability map with itself against a precomputed distance kernel using FFT. The implementation must be numerically tested against a brute-force small-grid reference.

Implement `runner.py` after the metric core exists. This module should load one completed concrete run, detect which held-out splits are available from the run metadata and prepared artifacts, materialize the standardized prediction artifact for each split, compute per-step energy scores, write one artifact such as `energy_by_step.parquet` or `energy_by_step.csv.gz`, and then write an aggregated summary JSON. Log final aggregate metrics back to MLflow, for example `energy_score.test_temporal` and `energy_score.test_individual`, plus optional counts such as `n_steps.test_temporal` and `n_steps.test_individual`. The per-step file belongs in the run artifact directory.

The evaluation runner must treat out-of-bounds deepSSF targets correctly. The model distribution is still defined on the chip grid, and the observed endpoint can lie outside that grid. The energy score remains valid because it only needs the true endpoint coordinate in meters relative to the start point, not a one-hot target pixel.

Implement `movebench/bench_v2/reporting/` next with `queries.py`, `plots.py`, `html.py`, and `status.py`. `queries.py` should expose simple MLflow-backed functions such as `load_main_results(split_key: str) -> pd.DataFrame` and `load_covariate_results(split_key: str) -> pd.DataFrame`. These functions should:

1. load the resolved experiment manifests,
2. gather the planned `run_key` values for the requested experiment kind,
3. query MLflow for runs in the benchmark experiment,
4. group runs by `tags.run_key`,
5. keep the newest run per `run_key`,
6. read params, tags, and final metrics into plot-ready dataframes.

Keep these queries explicit and easy to read. Do not query MLflow’s raw SQL tables directly.

In `plots.py`, generate the required figures. For main results, create one plot per study and per held-out split, with ordered x-axis `[1h, 12h, 1d, 7d]`, y-axis equal to mean energy score, and one line color per method. For covariate comparisons, create one figure per study and per held-out split that is faceted by method, with one line color per covariate set and the same ordered x-axis. Only draw data points that exist. If a study lacks one held-out split, omit that split’s figure and mark it “not applicable” in the HTML.

In `html.py`, generate `results/bench_v2/report/index.html` plus an `assets/` directory of PNGs and a `status_report.json` summary. The HTML should contain, in order, a global summary section, a study readiness section, a completed-runs section, a main-results section, and a covariate-comparison section. The page may use simple inline CSS and no JavaScript. The only requirement is that it is easy to scan quickly from disk or over shared storage.

Implement `movebench/bench_v2/status.py` or a similarly named module that emits a machine-readable snapshot of study readiness, planned runs, completed runs, blocked studies, and missing artifacts. `movebench bench status --format text` from the core plan should reuse this module, and `movebench bench status --format html` can later delegate to the report generator.

Implement the automation layer last in `movebench/bench_v2/automation/`. Add library functions for three actions: preparing newly download-complete studies, running newly ready experiments, and regenerating the report. Then add thin CLI or shell wrappers. A reasonable interface is:

    movebench bench prepare --selection ready-for-extraction
    movebench bench run experiments/v2/main_results.py --selection ready-for-experiment
    movebench bench run experiments/v2/covariate_comparison.py --selection ready-for-experiment
    movebench bench report

Also add one convenience script such as `scripts/daily/run_bench_v2_daily.sh` that calls those commands in order and writes a log file under `results/bench_v2/logs/`.

Finish with the cutover described in the overview plan. Once the report is correct and the new CLI covers daily operations, remove or convert the old benchmark-only scripts that would otherwise confuse future contributors.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Create a new test package for evaluation and reporting:

    tests/bench_v2/
      test_energy_score.py
      test_eval_runner.py
      test_report_queries.py
      test_report_plots.py
      test_report_html.py
      test_daily_automation.py

Start with energy-score tests. Use tiny handcrafted supports where the exact answer can be computed manually and small grids where the FFT path can be compared to a brute-force pairwise sum. Run:

    python -m unittest -v tests.bench_v2.test_energy_score

Add evaluation-runner tests that read fake run artifacts and write final metrics. Run:

    python -m unittest -v tests.bench_v2.test_eval_runner

Add report-query and plot tests using a temporary MLflow backend, for example a temporary SQLite-backed tracking URI, populated with a few fake completed runs. Run:

    python -m unittest -v tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html

Add automation tests that verify the wrapper commands call the underlying library functions and respect ready-versus-blocked status. Run:

    python -m unittest -v tests.bench_v2.test_daily_automation

The final smoke commands for this plan are:

    movebench bench report
    movebench bench status --format html

After those commands run, open `results/bench_v2/report/index.html` and verify that it contains plots and status summaries.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are observable.

First, the energy-score unit tests prove exactness. The deterministic path must match Euclidean distance exactly. The discrete-support path must match a direct brute-force calculation exactly or within a tiny floating-point tolerance. The dense-grid FFT path must match a brute-force small-grid calculation within a tiny floating-point tolerance.

Second, candidate-based final evaluation must fail if the prediction artifact still contains the true endpoint in its support. This is an intentional guardrail against leakage.

Third, `movebench bench report` must generate plots for every finished run and must not crash when some studies are blocked or have only one held-out split. The HTML should include visible counts for completed, pending, failed, blocked, and not-applicable cases.

Fourth, the daily wrapper should be able to run twice in a row without duplicating completed work. The second run should report mostly skips and a freshly regenerated report.

Fifth, after the cutover, the legacy benchmark entrypoints must no longer be the default path. A user following the repository should naturally discover `movebench bench ...` and the static report instead of the old YAML benchmark flow.

## Idempotence and Recovery

Final evaluation must be safe to rerun. If the latest MLflow run for a `run_key` already contains final metrics, rerunning should overwrite them only when the user explicitly requests refresh or when the evaluation artifact is missing or stale by a clear rule. The simplest acceptable initial rule is “always recompute when `movebench bench report --refresh-metrics` is passed; otherwise skip existing final metrics.”

Report generation must be fully regenerable from MLflow and run artifacts. The `report/` directory may be deleted and rebuilt from scratch without retraining models.

The daily wrapper must continue past individual study failures and end by producing a report that records those failures rather than hiding them.

## Artifacts and Notes

A finished run should eventually contain evaluation artifacts like:

    results/bench_v2/runs/<run_key>/final_metrics/
      test-temporal/
        energy_by_step.parquet
        summary.json
      test-individual/
        energy_by_step.parquet
        summary.json

The report root should look like:

    results/bench_v2/report/
      index.html
      status_report.json
      assets/
        main_results/
        covariate_comparison/

The HTML does not need client-side interactivity. Clarity and robustness matter more than animation.

## Interfaces and Dependencies

Define in `movebench/bench_v2/evaluation/types.py`:

    @dataclass(frozen=True)
    class DeterministicPrediction:
        dx_m: float
        dy_m: float

    @dataclass(frozen=True)
    class DiscreteSupportPrediction:
        coords_m: np.ndarray
        probs: np.ndarray

    @dataclass(frozen=True)
    class GridPrediction:
        probs_hw: np.ndarray
        x_coords_m: np.ndarray
        y_coords_m: np.ndarray

Define in `movebench/bench_v2/evaluation/energy.py`:

    def energy_score_deterministic(pred: DeterministicPrediction, truth_xy_m: np.ndarray) -> float: ...
    def energy_score_discrete(pred: DiscreteSupportPrediction, truth_xy_m: np.ndarray) -> float: ...
    def energy_score_grid(pred: GridPrediction, truth_xy_m: np.ndarray) -> float: ...

Define in `movebench/bench_v2/reporting/queries.py` a small set of explicit data loaders rather than one generic query builder.

Keep plotting dependencies minimal. Use `matplotlib` and standard library HTML generation unless a truly necessary requirement appears later.

Plan creation note (2026-03-24 / assistant): initial draft written after tracing the current energy-score math, identifying the exactness requirements for each method family, and choosing a static HTML report over app integration.
