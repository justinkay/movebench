# Build Exact Energy-Score Evaluation and Reporting

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan depends on [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md) and [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md). Do not begin this plan until concrete runs can be executed and prediction artifacts exist.

## Purpose / Big Picture

After this change, every finished run will have exact final energy scores on every available held-out split, those metrics will be queryable from MLflow, and `movebench bench report` will regenerate a static HTML dashboard showing progress and plots for main results and covariate comparisons. The visible proof is that a user can run `movebench bench report` and then open `results/bench_v2/report/index.html` to inspect both status and completed benchmark plots.

This plan is the user-facing finish of the rewrite. It turns run artifacts into trustworthy metrics and an operational dashboard.

## Progress

- [x] (2026-03-24 17:10Z) Located the existing energy-score implementation in `scripts/slurm/compare_outputs.py` and confirmed the exact formula used there.
- [x] (2026-03-24 17:20Z) Verified that candidate-based methods must remove the true endpoint candidate and renormalize probabilities before final held-out energy evaluation.
- [x] (2026-03-24 17:25Z) Verified that benchmark support is shared across method families, so endpoints outside that support should be excluded from final evaluation consistently for every method.
- [x] (2026-03-24 17:35Z) Confirmed that the current app is unrelated to benchmark reporting, making a generated static HTML report the simplest and safest presentation layer.
- [ ] Implement exact energy-score evaluation for deterministic, candidate-based, and grid-based methods.
- [ ] Implement final-evaluation orchestration and MLflow logging.
- [ ] Implement plot generation for main results and covariate comparisons.
- [ ] Implement static HTML report generation and status snapshots.
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
  Evidence: the existing study inventories and artifact layouts include studies with only `test-temporal` or only `test-individual`.

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

- Decision: final evaluation excludes out-of-support steps for every method family.
  Rationale: all methods are scored on the same benchmark support defined by the existing extraction buffers and prepared artifacts, so endpoints outside that support should be excluded uniformly rather than treated as a deepSSF-only corner case.
  Date/Author: 2026-03-24 / assistant

- Decision: per-step evaluation artifacts should default to gzipped CSV, not parquet.
  Rationale: this avoids adding a parquet-only dependency surface for a single reporting artifact. Parquet can be introduced later only if profiling shows CSV is an actual bottleneck.
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

Create a new evaluation package under `movebench/bench_v2/evaluation/` with `energy.py` and `runner.py`.

Do not add a separate `types.py`. Keep the evaluation boundary simpler than that. Instead, define one small documented artifact contract in `runner.py`: each prediction artifact must include a `prediction_kind` field with value `deterministic`, `discrete`, or `grid`, plus the minimal fields needed by that scorer. Methods may keep whatever internal output format they want during training; they only need to convert their final predictions into one of those three artifact schemas at the evaluation boundary.

Implement `energy.py` next. Add one exact function per prediction family. For deterministic point predictions, return the Euclidean distance to the observed endpoint. For discrete-support predictions, compute

    sum_i p_i ||x_i - y|| - 0.5 sum_{i,j} p_i p_j ||x_i - x_j||

exactly with vectorized tensor code. This path covers `baseline_randomwalk`, `issf_linear`, `issf_mlp`, and `moveformer`. Require that the discrete-support artifact already excludes the true endpoint candidate from its support; if the artifact still includes it, fail loudly. For dense-grid predictions, compute the first term directly over the full grid, then compute the pairwise term by convolving the probability map with itself against a precomputed distance kernel using FFT. The implementation must be numerically tested against a brute-force small-grid reference.

Implement `runner.py` after the metric core exists. This module should load one completed concrete run, detect which held-out splits are available from the run metadata and existing artifacts, load the method’s prediction artifact, dispatch on `prediction_kind`, compute per-step energy scores, write `energy_by_step.csv.gz`, and then write an aggregated summary JSON. Log final aggregate metrics back to MLflow, including the total energy score and its two named components for each split, where `accuracy` means the `E ||X - y||` term and `calibration` means the `E ||X - X'||` term, for example:

    energy_score.test_temporal
    energy_score.accuracy.test_temporal
    energy_score.calibration.test_temporal
    n_steps_scored.test_temporal
    n_steps_excluded_out_of_support.test_temporal

and the analogous `test_individual` metrics when that split exists. The per-step file belongs in the run artifact directory.

The evaluation runner must exclude out-of-support endpoints before scoring any method. This support should be defined once from the existing benchmark artifact metadata, using the same buffer and support rules that governed earlier downloads and feature preparation. The important rule is not “special-case deepSSF,” but rather “score all methods on the same eligible subset.” Report the number of excluded steps in the summary JSON and in MLflow.

Implement reporting inside `movebench/bench_v2/report.py`. Keep it simple: one file may internally define `load_main_results(...)`, `load_covariate_results(...)`, plotting helpers, HTML generation, and status rendering. These functions should:

1. load the resolved experiment manifests,
2. gather the planned `run_key` values for the requested experiment kind,
3. query MLflow for runs in the benchmark experiment,
4. group runs by `tags.run_key`,
5. keep the newest run per `run_key`,
6. read params, tags, and final metrics into plot-ready dataframes.

Keep these queries explicit and easy to read. Do not query MLflow’s raw SQL tables directly.

Experiment association for reporting should come from the resolved experiment manifests and their planned `run_key` sets, not from an MLflow experiment-membership tag on each run attempt.

Generate the required figures in `report.py`. For main results, create one plot per study and per held-out split, with ordered x-axis `[1h, 12h, 1d, 7d]`, y-axis equal to mean energy score, and one line color per method. For covariate comparisons, create one figure per study and per held-out split that is faceted by method, with one line color per covariate set and the same ordered x-axis. Only draw data points that exist. If a study lacks one held-out split, omit that split’s figure and mark it “not applicable” in the HTML.

Generate `results/bench_v2/report/index.html` plus an `assets/` directory of PNGs and a `status_report.json` summary from the same `report.py` module. The HTML should contain, in order, a global summary section, a study readiness section, a completed-runs section, a main-results section, and a covariate-comparison section. The page may use simple inline CSS and no JavaScript. The only requirement is that it is easy to scan quickly from disk or over shared storage.

Have `report.py` also emit a machine-readable snapshot of study readiness, planned runs, completed runs, blocked studies, and missing artifacts. `movebench bench status --format text` from the core plan can call a small helper in `run.py`, and `movebench bench status --format html` can delegate to `report.py`.

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

Start with energy-score tests. Use tiny handcrafted supports where the exact answer can be computed manually and small grids where the FFT path can be compared to a brute-force pairwise sum. Run:

    python -m unittest -v tests.bench_v2.test_energy_score

Add evaluation-runner tests that read fake run artifacts and write final metrics. Run:

    python -m unittest -v tests.bench_v2.test_eval_runner

Add report-query and plot tests using a temporary MLflow backend, for example a temporary SQLite-backed tracking URI, populated with a few fake completed runs. Run:

    python -m unittest -v tests.bench_v2.test_report_queries tests.bench_v2.test_report_plots tests.bench_v2.test_report_html

The final smoke commands for this plan are:

    movebench bench report
    movebench bench status --format html

After those commands run, open `results/bench_v2/report/index.html` and verify that it contains plots and status summaries.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are observable.

First, the energy-score unit tests prove exactness. The deterministic path must match Euclidean distance exactly. The discrete-support path must match a direct brute-force calculation exactly or within a tiny floating-point tolerance. The dense-grid FFT path must match a brute-force small-grid calculation within a tiny floating-point tolerance.

Second, candidate-based final evaluation must fail if the prediction artifact still contains the true endpoint in its support. This is an intentional guardrail against leakage.

Third, the evaluation runner must exclude out-of-support endpoints consistently across method families, write the excluded counts into the run summary, and log those counts plus the two raw energy-score components to MLflow.

Fourth, `movebench bench report` must generate plots for every finished run and must not crash when some studies are blocked or have only one held-out split. The HTML should include visible counts for completed, pending, failed, blocked, and not-applicable cases.

Fifth, after the cutover, the legacy benchmark entrypoints must no longer be the default path. A user following the repository should naturally discover `movebench bench ...` and the static report instead of the old YAML benchmark flow.

## Idempotence and Recovery

Final evaluation must be safe to rerun. If the latest MLflow run for a `run_key` already contains final metrics, rerunning should overwrite them only when the user explicitly requests refresh or when the evaluation artifact is missing or stale by a clear rule. The simplest acceptable initial rule is “always recompute when `movebench bench report --refresh-metrics` is passed; otherwise skip existing final metrics.”

Report generation must be fully regenerable from MLflow and run artifacts. The `report/` directory may be deleted and rebuilt from scratch without retraining models.

## Artifacts and Notes

A finished run should eventually contain evaluation artifacts like:

    results/bench_v2/runs/<run_key>/final_metrics/
      test-temporal/
        energy_by_step.csv.gz
        summary.json
      test-individual/
        energy_by_step.csv.gz
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

Define in `movebench/bench_v2/evaluation/energy.py`:

    def energy_score_deterministic(pred: dict[str, object], truth_xy_m: np.ndarray) -> float: ...
    def energy_score_discrete(pred: dict[str, object], truth_xy_m: np.ndarray) -> float: ...
    def energy_score_grid(pred: dict[str, object], truth_xy_m: np.ndarray) -> float: ...

Define in `movebench/bench_v2/evaluation/runner.py` a small documented dispatcher contract such as:

    prediction_kind == "deterministic":
        {"prediction_kind": "deterministic", "dx_m": ..., "dy_m": ...}

    prediction_kind == "discrete":
        {"prediction_kind": "discrete", "coords_m": ..., "probs": ...}

    prediction_kind == "grid":
        {"prediction_kind": "grid", "probs_hw": ..., "x_coords_m": ..., "y_coords_m": ...}

Keep this as documentation plus validation logic in `runner.py`, not as a separate abstraction layer.

Keep plotting dependencies minimal. Use `matplotlib` and standard library HTML generation unless a truly necessary requirement appears later.

Plan creation note (2026-03-24 / assistant): initial draft written after tracing the current energy-score math, identifying the exactness requirements for each method family, and choosing a static HTML report over app integration.
