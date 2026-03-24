# Build the Method Runtime, Shared Trainer, and Artifact-Consuming Executors

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.MD](/data/vision/beery/scratch/justin/movebench/PLANS.MD).

This plan depends on [execplans/benchmark_rewrite_core_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_core_runtime_execplan.md). Do not begin this plan until the planner and MLflow helper layer exist.

## Purpose / Big Picture

After this change, a user will be able to run `movebench bench run ...` through one clean runtime that validates and reuses shared artifacts already on disk, trains or executes every requested method, logs per-epoch statistics to MLflow, and produces prediction artifacts for exact evaluation. The visible proof is a smoke experiment that trains at least one iSSF model, one deepSSF model, and one MoveFormer model through the same CLI and writes run directories plus MLflow runs without using `movebench/benchmark.py`.

This plan is about execution, not reporting. It standardizes how runs are resolved, trained, resumed, and logged.

## Progress

- [x] (2026-03-24 17:10Z) Reviewed the existing iSSF, deepSSF, and MoveFormer model modules to confirm that the model definitions themselves can be reused.
- [x] (2026-03-24 17:20Z) Reviewed `movebench/train.py` and confirmed that the lifecycle logic is duplicated across method families even though early stopping and best-checkpoint restore are conceptually shared.
- [x] (2026-03-24 17:50Z) Verified that the point-store loading helpers and dataset wrappers already provide a usable read path for SSF-style methods, so the new runtime can start from existing artifacts instead of rebuilding them.
- [x] (2026-03-24 18:00Z) Verified that the current deepSSF cache design is keyed too narrowly to reuse superset artifacts for subset covariate experiments.
- [x] (2026-03-24 18:10Z) Verified that MoveFormer evaluation already uses a loss mask and therefore fits naturally with split-aware token masking in the new runtime.
- [ ] Implement the shared trainer lifecycle and stable runtime interfaces under `movebench/bench_v2/`.
- [ ] Implement read-only artifact resolution for point stores, normalization stats, and deepSSF caches.
- [ ] Implement the three simple baselines inside the new runtime.
- [ ] Implement iSSF linear and iSSF MLP executors against the new artifact layer.
- [ ] Implement deepSSF and MoveFormer executors against the new artifact layer.
- [ ] Implement `movebench bench run`.

## Surprises & Discoveries

- Observation: the current iSSF path already avoids dataloader overhead by training on cached tensors.
  Evidence: `fit_step_select_tensor(...)` in [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py) trains on a pre-materialized `(N,C,P)` tensor, which is a good optimization worth preserving inside the new runtime.

- Observation: the existing SSF/iSSF dataset wrappers already load point stores directly and support covariate subsetting.
  Evidence: [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py) reads `manifest.json`, `shared.pt`, and band tensors from the point-store layout and supports `subset_band_ids`.

- Observation: deepSSF’s current validation loss ignores out-of-bounds targets.
  Evidence: `eval_deepssf(...)` skips targets whose one-hot pixel is absent. That is acceptable for early stopping loss, but the final energy score later must not discard those steps.

- Observation: MoveFormer’s dataset wrapper already supports a `loss_mask`.
  Evidence: `eval_moveformer(...)` in [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py) reads `loss_mask` from the batch when present.

- Observation: the current MoveFormer dataset wrapper still recomputes step sequences from `trajectories.csv`, which makes it a partial fit rather than a drop-in runtime component.
  Evidence: [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py) calls `select_step_starts(...)` on trajectory rows instead of relying only on split-resolved point-store artifacts.

- Observation: the current point-store format stores the true endpoint as one candidate among the candidates.
  Evidence: point-store shared tensors include `num_candidates + 1` points per step, and the legacy selection losses assume the true endpoint is the final or first candidate depending on model family. The new runtime must keep this for training but remove the true endpoint from final candidate-based energy evaluation.

## Decision Log

- Decision: standardize new training schedules on epochs, not on the legacy mixture of steps and epochs.
  Rationale: the user asked for per-epoch statistics in MLflow, and epoch-based schedules are easier to explain and compare across methods.
  Date/Author: 2026-03-24 / assistant

- Decision: use one shared `EpochTrainer` lifecycle and method-specific executor objects.
  Rationale: optimizer stepping, early stopping, checkpoint restore, and metric logging are shared behaviors even though batch structure differs by method family.
  Date/Author: 2026-03-24 / assistant

- Decision: the new runtime will reuse superset covariate artifacts for `all` and derive subset views at read time wherever possible.
  Rationale: the user explicitly wants `all_environmental`, `alphaearth_only`, and `lite` runs to reuse artifacts from the `all` runs.
  Date/Author: 2026-03-24 / assistant

- Decision: this runtime will never create point stores or deepSSF WDS caches.
  Rationale: the user explicitly restricted this rewrite to the steps after extraction and cache building. Missing prerequisites should block runs, not trigger new extraction work here.
  Date/Author: 2026-03-24 / assistant

- Decision: early stopping will use a deterministic subset of the preferred held-out split, choosing `test-temporal` when present and otherwise the only available held-out split.
  Rationale: this matches the user’s request while remaining workable for studies that lack `test-temporal`.
  Date/Author: 2026-03-24 / assistant

- Decision: baselines will execute through the same concrete-run layer as trainable methods, but they will record zero epochs.
  Rationale: they still need run identity, MLflow artifacts, final metrics, and report visibility.
  Date/Author: 2026-03-24 / assistant

- Decision: MoveFormer evaluation sequences may include earlier same-individual context from outside the target partition, but loss and metrics must apply only to target-partition tokens.
  Rationale: autoregressive forecasting legitimately conditions on observed history, but benchmark metrics must only score the intended held-out steps.
  Date/Author: 2026-03-24 / assistant

## Outcomes & Retrospective

This plan should leave the repository with one execution path for all methods and one place to inspect per-run artifacts. If successful, the later evaluation and report plan can treat model execution as a solved problem and read standardized prediction artifacts.

## Context and Orientation

The new runtime should reuse the current model definitions but should not extend the legacy benchmark orchestration. The old `movebench/train.py` file is a reference for loss math and a warning about what not to keep duplicating.

Relevant existing files are:

- [movebench/models/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/ssf.py), which defines the linear and MLP iSSF models.
- [movebench/models/deepssf.py](/data/vision/beery/scratch/justin/movebench/movebench/models/deepssf.py), which defines the deepSSF network.
- [movebench/models/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/models/moveformer.py), which wraps the vendored MoveFormer implementation.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), which contains the reusable point-store loading helpers.
- [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), which already provides the right point-store-backed shape for SSF and iSSF data loading.
- [movebench/datasets/deepssf_wds_cache.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/deepssf_wds_cache.py), which defines the existing deepSSF cache manifest and loader behavior.
- [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py), which shows what needs to be replaced for split-resolved sequence construction.
- [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py), which contains the existing fit and eval loops that the new trainer will replace.

A “runnable run” in this plan means the resolved concrete run plus the already-existing study-local artifacts and dataset objects it needs.

A “superset artifact” means an existing artifact built once using the full `all` covariate set so that later covariate subsets can reuse it by selecting channels instead of regenerating pixels.

## Plan of Work

Keep the runtime compact. Put orchestration in `movebench/bench_v2/run.py`, the shared epoch loop in `movebench/bench_v2/train.py`, and family-specific code in `movebench/bench_v2/methods/`. Do not add `prepare.py` in this scope; artifact discovery and validation can stay in `run.py` or the method modules unless duplication proves otherwise.

Implement the shared trainer first in `movebench/bench_v2/train.py`. Define one `EpochTrainer` that is independent of any specific batch shape. It should own seed setup, epoch loops, best-checkpoint tracking, early-stopping decisions, per-epoch metric logging, and final restore of the best checkpoint. It may use a very small method-facing interface, but do not over-abstract it. The trainer must log per-epoch metrics to MLflow with `step=epoch` for trainable methods and none for baselines. It should also write a small `epoch_history.json` artifact so later debugging does not depend on MLflow API calls alone.

Use epoch schedules everywhere in the new runtime. The method catalog should expose defaults like `max_epochs`, `eval_every_epochs`, and `patience_epochs` instead of the legacy `max_steps` vocabulary. Preserve method-specific optimizer and scheduler details inside the method executor, not inside the trainer.

Implement read-only artifact resolution next. This logic should locate point-store manifests via `load_point_store_manifest(...)`, `shared.pt`, and train normalization stats under the existing `features/points/<split>/<rasters_hash>/<sampler_hash>/` layout. It should also locate deepSSF cache manifests under `cache/deepssf_wds/<split>/<cache_hash>/`. If a required manifest, shard set, or normalization file is missing, the resolver should return a concise blocked reason and the runner should skip that concrete run instead of trying to build missing artifacts.

The point artifact layer must support view selection. A covariate subset must be implemented as a list of band indices into the `all` point store. `all_environmental` means the `all` band order minus every alphaearth band. `alphaearth_only` means only the alphaearth bands. `lite` means the subset of bands whose layer ids correspond to elevation, NDVI, and Dynamic World. The subset view should not create a new store on disk.

Keep normalization handling read-only in this plan. It should load normalization stats from the existing `all` train split once per study and timescale. Subset covariate sets should read the same stats file and index into the needed bands. If the stats file is missing, mark the run blocked instead of computing new stats here.

Implement deepSSF cache consumption in the same compact style. The runtime should resolve the existing `all` cache manifests for the needed split, output grid shape, and normalization policy, and then slice channel indices out of the cached tensors before they reach the model. This is the mechanism that lets `all_environmental`, `alphaearth_only`, and `lite` reuse the `all` deepSSF cache without building duplicates. If the needed superset cache is missing, mark the run blocked.

Implement the three baseline executors in `methods/baselines.py`. The no-move baseline predicts the start point exactly. The constant-velocity baseline uses the previous observed step length and current bearing exactly as in the old comparison script. The random-walk baseline uses the same discrete pseudo-candidate support built for the point store and treats those pseudo-candidates as a uniform predictive distribution. The baseline executor does not train, but it must still create an MLflow run, write a prediction artifact placeholder, and mark the run completed.

Implement the iSSF executor in `methods/issf.py`. It should reuse `LinearStepSelect` and `MLPStepSelect` from the existing model module. Build a pre-materialized `(N,C,P)` tensor cache from the point store and the selected covariate view, keyed by the point-store identity plus the covariate-set key. The tensor cache is a runtime optimization, not a replacement for the extracted point store, and may live under `results/bench_v2/cache/issf_tensor/<run_key>/` or a similarly isolated runtime cache root. Training should use the shared `EpochTrainer`, an Adam optimizer, and the same candidate-cross-entropy objective as the existing code. The early-stopping subset must be a deterministic subset of the preferred held-out split, not a random sample per epoch.

Implement the deepSSF executor in `methods/deepssf.py`. Reuse the existing model definition and the existing WebDataset loader path where possible, but add a thin adapter that slices cached spatial channels to the requested covariate set and exposes a deterministic early-stopping subset. Keep the current cross-entropy-on-grid loss for training and early stopping. Log the number of valid in-bounds held-out targets used by early stopping because that count can differ across studies and timescales.

Implement the MoveFormer executor in `methods/moveformer.py`. Reuse `build_moveformer_selection_model(...)`, but replace only the old sequence-construction path that still depends on `trajectories.csv` split labels. The new sequence builder should start from split-resolved point-store artifacts and shared tensors directly. Training sequences should contain only train tokens in the loss mask. Evaluation sequences for `test-temporal` or `test-individual` may include earlier same-individual context from the same split-resolved representation, but the `loss_mask` must be true only for the target tokens being scored. Preserve the candidate permutation that makes the true endpoint the target index expected by the model.

Implement `movebench/bench_v2/run.py` last in this plan. This module should load the concrete runs from the planner, create MLflow runs, resolve and validate existing artifacts, dispatch to the method executors, and write a run directory such as `results/bench_v2/runs/<run_key>/`. The directory should contain the resolved payload, checkpoints for trainable methods, per-epoch history, a concise training summary JSON, and standardized prediction artifacts for later exact evaluation. Each MLflow run must be tagged with `run_key`, `study_id`, `timescale`, `covariate_set`, `method`, `experiment_kind`, and `status`.

Extend [movebench/cli/main.py](/data/vision/beery/scratch/justin/movebench/movebench/cli/main.py) with `bench run` after the runtime exists. `run` should accept an experiment module, optional study filters, optional method filters, and a `--max-runs` limit for smoke testing.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

Create a new test package for runtime and method behavior:

    tests/bench_v2/
      test_trainer.py
      test_artifact_resolution.py
      test_issf_runtime.py
      test_deepssf_runtime.py
      test_moveformer_runtime.py
      test_runner.py

Start with the shared trainer tests using a fake `EpochJob` that has a predictable loss curve. The tests must prove best-checkpoint restore and early stopping. Run:

    python -m unittest -v tests.bench_v2.test_trainer

Next, add artifact-resolution tests for point-store reuse and deepSSF-cache reuse across covariate subsets. A key assertion is that `all_environmental` after `all` resolves to the same underlying deepSSF cache manifest and slices channels at read time instead of expecting a second cache on disk. Run:

    python -m unittest -v tests.bench_v2.test_artifact_resolution

Add one small smoke test per method family that uses synthetic or tiny fixture data to exercise the new executor interface without depending on the report layer. Run:

    python -m unittest -v tests.bench_v2.test_issf_runtime tests.bench_v2.test_deepssf_runtime tests.bench_v2.test_moveformer_runtime

Add a runner test that creates concrete runs, executes a fake or tiny real experiment, and then verifies run directories plus MLflow run tags and metrics. Run:

    python -m unittest -v tests.bench_v2.test_runner

The final smoke command for this plan is:

    movebench bench run experiments/v2/smoke_one_study.py --max-runs 3

That command should produce at least one trainable run and one baseline run under `results/bench_v2/runs/`.

## Validation and Acceptance

Acceptance for this plan is reached when the following behaviors are observable.

First, `movebench bench run experiments/v2/smoke_one_study.py --max-runs 3` must resolve the required point stores, normalization stats, and deepSSF caches from their existing on-disk locations before execution begins. If any required artifact is missing, the affected run must be marked blocked with a concise reason and skipped.

Second, trainable methods log per-epoch metrics to MLflow with `step=epoch`, restore the best checkpoint before writing their final training summary, and store the preferred held-out split used for early stopping in their run payload and MLflow tags.

Third, baseline methods create completed runs with zero epochs, no checkpoint files, and standardized prediction artifacts that later evaluation code can read.

Fourth, covariate-subset runs reuse the existing `all` point-store and deepSSF-cache artifacts at read time instead of requiring new extracted artifacts on disk.

Fifth, `movebench bench run experiments/v2/smoke_one_study.py --max-runs 3` skips already completed concrete runs when rerun, or creates a new MLflow run only when the previous run failed or the user explicitly requests re-execution.

## Idempotence and Recovery

Artifact resolution must be read-only and idempotent. If the point store exists but normalization stats are missing, rerunning should continue to mark the run blocked; this plan must not compute new stats. If the deepSSF cache manifest or shards are missing, rerunning should continue to mark the run blocked until the existing extraction pipeline has repaired them.

Training runs must not be considered complete until the method executor writes its final summary and the runner marks the MLflow run with `status=completed`. Partially written checkpoint files are not completion markers. Re-running a failed run may reuse the same `run_key`; status and reporting code should then treat the newest MLflow run for that `run_key` as authoritative.

## Artifacts and Notes

The run directory for a trainable method should look like:

    results/bench_v2/runs/<run_key>/
      resolved_run.json
      training_summary.json
      history.json
      checkpoint_best.pt
      predictions/

The point-store manifest should record the full band order so that covariate subset views can validate their indices against it.

The deepSSF cache manifest should record the full channel order and enough metadata to prove that two subset covariate runs are reading slices of the same cached tensors.

## Interfaces and Dependencies

Define in `movebench/bench_v2/train.py`:

    class EpochJob(Protocol):
        def train_one_epoch(self, epoch: int) -> dict[str, float]: ...
        def evaluate_subset(self, split_key: str) -> dict[str, float]: ...
        def state_dict(self) -> dict[str, object]: ...
        def load_state_dict(self, state: dict[str, object]) -> None: ...
        def close(self) -> None: ...

    @dataclass
    class TrainerConfig:
        max_epochs: int
        eval_every_epochs: int
        patience_epochs: int
        seed: int

    class EpochTrainer:
        def fit(self, job: EpochJob, config: TrainerConfig, tracker: object) -> dict[str, object]: ...

If a formal protocol is useful, keep it tiny. A simple pair of callables such as `resolve_inputs(run)` and `execute(resolved_run)` is enough.

Define stable method keys exactly as:

    baseline_nomove
    baseline_constvel
    baseline_randomwalk
    issf_linear
    issf_mlp
    deepssf
    moveformer

Do not invent alternate user-facing names.

Plan creation note (2026-03-24 / assistant): initial draft written after confirming which existing model definitions can be reused, which artifact-loading paths are salvageable, and which legacy dataset wrappers are too tied to the old split semantics to keep.
