# Move MoveFormer Sequence Construction Back Into the Dataset Module and Point Store

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This document must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

This plan is a follow-up to [execplans/benchmark_rewrite_method_runtime_execplan.md](/data/vision/beery/scratch/justin/movebench/execplans/benchmark_rewrite_method_runtime_execplan.md). It intentionally covers one bounded cleanup plus one bounded artifact change: MoveFormer sequence-construction and feature-packaging logic should live in [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py), while the `bench_v2` executor should shrink back to runtime concerns such as dataloaders, optimizer state, scheduler behavior, checkpointing, MLflow-facing metrics, and prediction artifact writing. To make that dataset fully artifact-driven, this follow-up now also widens scope slightly into [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) so `shared.pt` includes `start_lon`, `start_lat`, `individual_id`, and `burst_id`. Evaluation/reporting and broad refactors to other runtimes remain out of scope.

## Purpose / Big Picture

After this change, the MoveFormer path will be easier to maintain because the code that interprets point-store artifacts as model-ready sequences will live with the other MoveFormer dataset logic instead of being split between the dataset module and the `bench_v2` executor. A user should still be able to run `movebench bench run ...` for MoveFormer exactly as before, but the executor file should become substantially shorter and the dataset module should become the single place that defines candidate permutation, normalization, movement-vector scaling, chain grouping, overlap behavior, and split-aware loss masking.

The visible proof is that the existing targeted MoveFormer runtime tests still pass, at least one point-store test proves the new metadata lands in `shared.pt`, and at least one new dataset-focused test shows that `context_splits=["train", "test-temporal"]` yields held-out sequences with earlier train context and a `loss_mask` that is true only on the held-out tokens without loading `trajectories.csv` at runtime.

## Progress

- [x] (2026-03-27 13:40Z) Reviewed the current MoveFormer executor and dataset implementations and identified the precise duplication boundary: sequence construction, normalization, candidate permutation, movement-scale derivation, and token metadata currently live in the executor even though they are dataset responsibilities.
- [x] (2026-03-26 16:20Z) Widened scope slightly after review: the point-store artifact must now carry `start_lon`, `start_lat`, `individual_id`, and `burst_id` so MoveFormer no longer needs `trajectories.csv` at runtime.
- [x] (2026-03-26 17:05Z) Added failing tests that pinned the new point-store metadata contract, the new dataset boundary, and the remaining executor dependency on `trajectories.csv`.
- [x] (2026-03-26 17:25Z) Refactored `movebench/features/points.py` so extracted `shared.pt` files now include `start_lon`, `start_lat`, `individual_id`, and `burst_id`, and added an extraction metadata version to invalidate stale stores.
- [x] (2026-03-26 17:55Z) Rewrote `movebench/datasets/moveformer.py` so `MoveFormerPointStoreDataset` builds windows from split-resolved point stores only, groups by `(individual_id, burst_id)`, sorts by `timestamp_ns`, and returns `point_id_start` plus `coords_m`.
- [x] (2026-03-26 18:05Z) Slimmed `movebench/bench_v2/executors/moveformer.py` so it now instantiates datasets, reuses `collate_moveformer(...)` and `eval_moveformer(...)`, and keeps only training-loop, checkpoint, and prediction responsibilities.
- [x] (2026-03-26 18:15Z) Ran the targeted point-store, MoveFormer runtime, and runner tests and reviewed the outputs.

## Surprises & Discoveries

- Observation: the current executor duplicates several responsibilities that already conceptually belong to the dataset or shared training helpers.
  Evidence: [movebench/bench_v2/executors/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/executors/moveformer.py) currently contains custom collation, normalization loading, movement-scale calculation, sequence chaining, windowing, metric evaluation, and prediction-token metadata assembly.

- Observation: the existing dataset module already contains most of the feature encoding logic needed by MoveFormer, but it still re-runs step selection from `trajectories.csv`.
  Evidence: [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py) already handles candidate permutation, normalization, `SimpleTrajectoryProcessor`, and `loss_mask`, but `_select_step_starts_with_targets(...)` still calls `select_step_starts(...)`.

- Observation: the point-store artifacts already contain enough linkage to rebuild sequence order without recomputing step starts from raw trajectories.
  Evidence: [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) stores `point_id_start`, `point_id_end`, `timestamp_ns`, `timestamp_end_ns`, `candidate_lon`, and `candidate_lat` in `shared.pt`.

- Observation: the current point-store artifact is still missing the start-point coordinates and explicit grouping metadata that would let MoveFormer stay fully artifact-driven at runtime.
  Evidence: [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) does not currently write `start_lon`, `start_lat`, `individual_id`, or `burst_id` into `shared.pt`, so the executor compensates by reading `trajectories.csv` and rebuilding step chains itself.

- Observation: a manifest-level extraction version was enough to invalidate stale point stores; a full point-manifest schema bump was not required for this scoped change.
  Evidence: adding `shared_metadata_version = 1` to `manifest.extraction` and checking it in `_manifest_matches_request(...)` forced rebuilds while keeping unrelated point-manifest consumers unchanged.

- Observation: once `individual_id`, `burst_id`, and `timestamp_ns` were in `shared.pt`, the executor’s point-id link-walking logic became unnecessary.
  Evidence: the final dataset implementation groups by `(individual_id, burst_id)` and sorts by `timestamp_ns`, and the runtime test now passes without any `trajectories.csv` lookup.

## Decision Log

- Decision: refactor the existing `MoveFormerPointStoreDataset` in place instead of keeping a second benchmark-only dataset class.
  Rationale: the user wants dataset logic to stay in the dataset class, and keeping one point-store-backed MoveFormer dataset avoids creating a second source of truth for candidate permutation, normalization, and sequence window semantics.
  Date/Author: 2026-03-27 / assistant

- Decision: preserve the existing `split` parameter as the target split, and add an optional `context_splits` parameter that defaults to `[split]`.
  Rationale: this keeps the current call sites simple while allowing held-out datasets to include earlier same-individual context from `train` without redesigning the surrounding API.
  Date/Author: 2026-03-27 / assistant

- Decision: extend the point-store `shared.pt` payload with `start_lon`, `start_lat`, `individual_id`, and `burst_id`.
  Rationale: these are the missing fields that let `MoveFormerPointStoreDataset` group and order steps directly from extracted artifacts and remove the runtime dependency on `trajectories.csv`.
  Date/Author: 2026-03-26 / assistant

- Decision: after the metadata lands in `shared.pt`, the MoveFormer dataset should stop loading `trajectories.csv` entirely.
  Rationale: the widened scope is specifically intended to make the point store, not the raw CSV, the runtime source of truth for MoveFormer sequence construction and token geometry.
  Date/Author: 2026-03-26 / assistant

- Decision: keep prediction artifact writing in the executor, but make the dataset return `point_id_start` and `coords_m` so the executor does not reconstruct them.
  Rationale: packaging and saving run outputs is executor responsibility, but the per-token geometric metadata belongs to the dataset representation.
  Date/Author: 2026-03-27 / assistant

## Outcomes & Retrospective

This follow-up is complete. The point-store extractor now writes `start_lon`, `start_lat`, `individual_id`, and `burst_id` into `shared.pt`, the MoveFormer dataset is fully artifact-driven at runtime, and the `bench_v2` executor has been reduced to dataset instantiation, training-loop state, checkpointing, and prediction writing. [movebench/study.py](/data/vision/beery/scratch/justin/movebench/movebench/study.py) also now passes `context_splits` through to the dataset.

The exact passing verification command was:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_feature_store tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner

Reviewed output: all 20 targeted tests passed. The only warning observed was a CUDA availability warning from `torch.cuda` during the MoveFormer runtime test in this environment; the executor correctly fell back to CPU and the test still passed.

## Context and Orientation

The relevant current files are:

- [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py), which now owns point-store-backed sequence construction, context-split windowing, candidate permutation, normalization, and per-token geometry metadata for MoveFormer.
- [movebench/bench_v2/executors/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/executors/moveformer.py), which now limits itself to dataset instantiation, runtime state, shared-loop training, checkpointing, and prediction writing.
- [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py), which already provides `collate_moveformer(...)` and `eval_moveformer(...)`.
- [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), which now writes the MoveFormer start and grouping metadata directly into `shared.pt`.

In this repository, a “point-store-backed MoveFormer dataset” means a dataset whose items are built entirely from the already-extracted point-store files (`manifest.json`, band tensors, and `shared.pt`) plus read-only train normalization stats. It must not recompute the extracted step set from raw trajectories, and after the widened schema change it should not need to load `trajectories.csv` at runtime at all.

The current executor is too large because it is doing both “dataset work” and “runtime work.” Dataset work means interpreting artifacts into model-ready examples. Runtime work means creating dataloaders, running optimization, evaluating checkpoints, and writing run outputs. This follow-up should move only the former category.

## Plan of Work

Start with tests. Update [tests/test_point_feature_store.py](/data/vision/beery/scratch/justin/movebench/tests/test_point_feature_store.py) so one focused extraction test proves that `shared.pt` now contains `start_lon`, `start_lat`, `individual_id`, and `burst_id`. Then update [tests/bench_v2/test_moveformer_runtime.py](/data/vision/beery/scratch/justin/movebench/tests/bench_v2/test_moveformer_runtime.py) so one test directly exercises the dataset boundary rather than the executor’s private helpers. The test should construct tiny train and `test-temporal` point stores with the new shared metadata, instantiate `MoveFormerPointStoreDataset` with `split="test-temporal"` and `context_splits=["train", "test-temporal"]`, and assert that the resulting item includes train context tokens while `loss_mask` is true only on the held-out tokens.

If that test becomes awkward to express inside `tests/bench_v2/test_moveformer_runtime.py`, add one small dedicated dataset test file such as `tests/test_moveformer_dataset.py`. Prefer not to add a new file unless the existing runtime test becomes hard to read. The priority is to make the new dataset boundary explicit in tests before refactoring code.

Next, refactor [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) so `shared.pt` includes `start_lon`, `start_lat`, `individual_id`, and `burst_id` for every extracted step. Keep this change local to point-store extraction; do not touch cache creation or unrelated feature stores.

Then refactor [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py). Keep the public class name `MoveFormerPointStoreDataset`, but change its internal sequence-construction path so it no longer calls `select_step_starts(...)` or loads `trajectories.csv`. The class should load one or more split-resolved point stores, reorder candidates so the true endpoint is index `0`, compute or load normalization stats exactly once, derive the movement-vector scale from the train point store, group steps by `(individual_id, burst_id)`, sort them by `timestamp_ns`, split those chains into `max_len` windows with one-token overlap, and set `loss_mask` to score only tokens whose target split equals the dataset `split` and that have a previous context token.

Use an optional `context_splits: list[str] | None = None` parameter on the dataset constructor. When it is `None`, default to `[split]` so current single-split behavior remains straightforward. For training, the executor should instantiate the dataset with `split="train"` and leave `context_splits=None`. For held-out evaluation, the executor should pass `split=<heldout>` and `context_splits=["train", <heldout>]`. The dataset should also continue to expose `cov_dim` and `num_candidates_total`, and each returned item should include `point_id_start` and `coords_m` so the executor can write prediction artifacts without rebuilding geometry.

After the dataset refactor, slim [movebench/bench_v2/executors/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/executors/moveformer.py). Remove the executor-private sequence builder, normalization loader, movement-scale logic, custom collate function, and custom evaluation helper. Replace them with dataset instantiation plus imports from the existing shared helpers where practical. At minimum, the executor should reuse `MoveFormerPointStoreDataset`, `collate_moveformer(...)`, and `eval_moveformer(...)`. Keep a small runtime object only for the optimizer, the warmup/decay behavior, the variable-length attention-mask behavior, and the `state_dict()` / `load_state_dict()` contract required by `run_training_loop(...)`.

Prediction writing should stay in the executor because it is part of the benchmark run output contract, but the executor should treat the dataset batch as the source of truth for `point_id_start` and `coords_m`. The executor may still translate logits into pseudo-candidate probabilities and save them to `predictions/<split>.pt`, but it should not reconstruct token geometry or point ids from raw stores.

Keep scope tight. Only touch [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py) enough to add the new shared metadata fields. Do not redesign the shared training loop. Do not refactor other executors while cleaning up MoveFormer.

## Concrete Steps

Work from the repository root:

    cd /data/vision/beery/scratch/justin/movebench

First, add or update the failing point-store and MoveFormer tests and run only the narrow command:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_feature_store tests.bench_v2.test_moveformer_runtime

Then refactor `movebench/features/points.py` and `movebench/datasets/moveformer.py` and rerun the same narrow command. Do not touch the executor yet until the metadata and dataset tests pass.

After the dataset is the source of truth for sequence construction, simplify `movebench/bench_v2/executors/moveformer.py` and rerun:

    /data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_moveformer_runtime tests.bench_v2.test_runner

If everything passes, record the exact passing command and any warnings observed in the `Outcomes & Retrospective` section before considering the cleanup complete.

## Validation and Acceptance

Acceptance for this follow-up is reached when the following behaviors are all observable.

First, extracted point stores write `start_lon`, `start_lat`, `individual_id`, and `burst_id` into `shared.pt` with stable per-step ordering.

Second, `MoveFormerPointStoreDataset` can be instantiated for a held-out split with explicit `context_splits`, and its items include earlier same-individual context from `train` while `loss_mask` is true only for the held-out tokens. No code path in the dataset should call `select_step_starts(...)` or load `trajectories.csv` anymore.

Third, the `bench_v2` MoveFormer executor still trains through the shared `run_training_loop(...)`, restores the best checkpoint, writes `history.json`, writes `checkpoint_best.pt`, and saves standardized discrete prediction artifacts.

Fourth, the executor becomes materially smaller because it no longer owns normalization loading, movement-scale derivation, point-store chaining, sequence windowing, or per-token geometry reconstruction.

Fifth, the targeted tests pass in this environment using only the point-store artifacts and normalization stats required by the scoped code paths. No extraction or cache-building code should run outside the dedicated point-store schema test.

## Idempotence and Recovery

This follow-up must remain read-only with respect to extracted artifacts. If a required point store or normalization stats file is missing, the behavior should remain blocked exactly as it is now. The dataset refactor must not try to regenerate artifacts or “heal” missing files.

The cleanup should also remain safe for reruns. Re-instantiating the dataset with the same resolved artifacts and `context_splits` should produce the same sequence membership and `loss_mask` layout deterministically. This is especially important because the shared training loop relies on stable held-out evaluation semantics for early stopping.

## Artifacts and Notes

The most important evidence for this follow-up will be short test transcripts showing that the point store now carries the needed metadata and that the dataset now owns the context-split behavior. Capture a concise example here when implementation begins, such as a test asserting:

    shared["start_lon"]      == [...]
    shared["individual_id"]  == [...]
    point_id_start           == [train_step_1, train_step_2, heldout_step_1, heldout_step_2]
    loss_mask                == [False, False, True, True]

The executor’s prediction artifact format should remain unchanged:

    predictions/<split>.pt
      prediction_kind = "discrete"
      point_id_start
      coords_m
      probs

The difference after this cleanup is where those tensors come from: `point_id_start` and `coords_m` should come from dataset items, not from executor-private reconstruction.

## Interfaces and Dependencies

Keep the public dataset class name:

    class MoveFormerPointStoreDataset(Dataset): ...

Update its constructor in [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py) to support:

    def __init__(
        self,
        *,
        study_dir: str | Path,
        trajectories_path: str | Path,
        plan: LayerPlan,
        sampler_config: SamplerConfig,
        split: str,
        context_splits: list[str] | None = None,
        max_len: int = 500,
        min_len: int = 2,
        normalize_env: bool = True,
        require_normalization_stats: bool = False,
        subset_band_ids: list[str] | None = None,
    ) -> None: ...

The meaning of these parameters should be:

- `split`: the target split whose tokens should be scored.
- `context_splits`: the ordered set of point-store splits whose earlier tokens may appear as context. When omitted, use `[split]`.

Each dataset item should remain a dictionary of tensors compatible with `collate_moveformer(...)` in [movebench/train.py](/data/vision/beery/scratch/justin/movebench/movebench/train.py). In addition to the current model features, each item must expose:

    point_id_start: int64 [L]
    coords_m: float32 [L, P-1, 2]

In the executor, prefer these existing shared helpers:

    from movebench.train import collate_moveformer, eval_moveformer

The executor should still define a small runtime object for optimizer and scheduler behavior, but it should not define its own point-store sequence builder.

Plan creation note (2026-03-27 / assistant): created this follow-up ExecPlan after the first MoveFormer `bench_v2` implementation landed with working behavior but too much dataset logic inside the executor. The goal of this follow-up is structural cleanup without reopening extraction or evaluation scope.

Update note (2026-03-26 / assistant): after reviewing how the executor still depended on `trajectories.csv` for start coordinates and chain grouping, widened scope slightly so the point-store artifact itself carries `start_lon`, `start_lat`, `individual_id`, and `burst_id`. This keeps the cleanup local while allowing the runtime MoveFormer path to become fully artifact-driven.
