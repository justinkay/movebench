# Replace `.pt` Point Stores With Mmap-Readable Point Store V2 And Batch-Oriented iSSF Loading

This ExecPlan is a living document. The sections `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` must be kept up to date as work proceeds.

This plan must be maintained in accordance with [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md).

## Purpose / Big Picture

After this change, `Study.extract_point_features(...)` will write point stores in a contiguous array format that can be memory-mapped (that is, opened by the operating system so code can read only the requested rows without deserializing the entire file). The on-disk point store will no longer depend on `torch.save(...)` files for per-band arrays, so large studies will stop paying the current full-file deserialization cost and the current oversized-storage failure mode.

The user-visible behavior is that point extraction writes only the new schema-4 point-store layout, every point-store consumer moves to row-oriented or block-oriented reads instead of loading full splits into RAM, and old schema-3 stores are treated as unsupported inputs rather than something to migrate in code. A concrete success case is that the Morato `1h` `issf_linear` benchmark can be rerun inside a 64 GiB Slurm allocation without dying during `Creating eval dataset`, because the dataset now loads only batch rows in DataLoader workers.

## Progress

- [x] (2026-03-30 16:05Z) Re-read [PLANS.md](/data/vision/beery/scratch/justin/movebench/PLANS.md), inspected the current point-store manifest and read/write code in `movebench/features/manifest.py` and `movebench/features/points.py`, and traced every repository consumer of `load_point_store_manifest(...)`, `load_point_store_shared(...)`, and `load_point_store_bands(...)`.
- [x] (2026-03-30 16:10Z) Confirmed that the current point store is a shared repository contract used by iSSF, SSF, MoveFormer, baselines, normalization, and `bench_v2` readiness, so this work must produce a general point-store v2 rather than an iSSF-only cache.
- [x] (2026-03-30 16:15Z) Quantified the current Morato failure surface: the `train` `1h` point store occupies about `25G` on disk and the `test_temporal` `1h` point store occupies about `41G`, while the logical `env+valid` payload is only about `0.27 GiB` and `0.46 GiB` respectively, confirming that the present `.pt` layout is the immediate storage problem.
- [x] (2026-03-30 16:20Z) Locked the implementation direction for this plan: keep the current layer-oriented logical grouping from the raster plan, replace the `.pt` payload format with mmap-readable contiguous arrays, preserve the high-level extraction entrypoint, and migrate all point-store consumers to the new row/block reader surface without forcing a broader row-major rewrite.
- [x] (2026-03-30 17:18Z) Added a new active `tests/` package with schema-4 coverage for point-store writing/reading, `bench_v2` readiness, iSSF batch loading, MoveFormer compatibility, and benchmark default-split behavior without using `old_tests_dont_use`.
- [x] (2026-03-30 17:24Z) Implemented schema-4 point-store manifests and fresh extraction output with `shared/shared.npy` plus per-layer `layers/<layer>.env.npy` and `layers/<layer>.valid.npy`, and removed schema-3 write/read expectations from supported code paths.
- [x] (2026-03-30 17:29Z) Implemented `PointStoreReader` as the worker-safe mmap row/block reader and ported point-store consumers in SSF, iSSF, MoveFormer, normalization, baselines, and `bench_v2` prediction/readiness code to the new interface.
- [x] (2026-03-30 17:33Z) Refactored `iSSFDataset` to metadata-only construction with batch-time row reads and updated the iSSF executor/config to use DataLoader workers and collate-driven batch assembly.
- [x] (2026-03-30 17:38Z) Ran `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -B -m unittest -v tests.test_point_store_v2 tests.bench_v2.test_runner tests.bench_v2.test_issf_runtime tests.bench_v2.test_moveformer_runtime tests.test_benchmark_default_splits`; result: `Ran 14 tests in 117.864s`, `OK`.
- [ ] The real Morato `1h` point-store regeneration and `bench run experiments/v2/issf.py --studies Morato_8fed4a87_raw_merged --methods issf_linear --max-runs 1 --rerun` validation have not been executed in this turn, so the exact 64 GiB acceptance case is still unverified here.

## Surprises & Discoveries

- Observation: the point store is not an internal iSSF detail; it is the shared extracted-feature contract for several method families.
  Evidence: `movebench/study.py` exposes `extract_point_features(...)`, `issf_dataset(...)`, `moveformer_dataset(...)`, and `compute_normalization_vals(from_='points', ...)` over the same storage root, and `movebench/bench_v2/run.py` marks point-backed methods runnable from the same manifest checks.

- Observation: the current point-store manifest shape locks readers to the old serialization format.
  Evidence: `PointFeatureManifest.validate_basic(...)` in `movebench/features/manifest.py` currently requires `shared.pt`, and `_band_meta(...)` in `movebench/features/points.py` stores only a single `bands/{band_id}.pt` path for each channel.

- Observation: the immediate Morato failure is host-memory pressure during CPU dataset construction, not GPU memory exhaustion.
  Evidence: `iSSFDataset` loads all band files with `torch.load(..., map_location='cpu')` in `movebench/features/points.py`, the executor constructs and keeps both train and eval datasets in `movebench/bench_v2/executors/issf.py`, and the interactive Slurm allocation reports `SLURM_MEM_PER_NODE=65536`.

- Observation: the present `.pt` layout is pathologically larger than the logical tensors for this workload.
  Evidence: for Morato `1h`, one representative train band file is about `280M` on disk even though its logical `(3693, 101)` `env+valid` payload should be only a few megabytes, and the full train store occupies about `25G` even though the logical full-split `env+valid` payload is only about `0.27 GiB`.

- Observation: there is currently no active top-level `tests/` package in this working tree.
  Evidence: repository inspection found no `tests/` directory, so this plan must create the active test package as part of the implementation instead of extending a checked-in one.

- Observation: the repository already has a successful precedent for “derived training cache on top of an extracted base artifact”.
  Evidence: `movebench/datasets/deepssf_wds_cache.py` writes a WebDataset cache from the chip dataset, and `movebench/bench_v2/executors/deepssf.py` trains entirely from that cache without changing the higher-level benchmark entrypoints.

- Observation: importing `movebench` eagerly imported `Study`, which pulled heavy pipeline dependencies into point-store-only tests and made cold startup look hung.
  Evidence: `movebench/__init__.py` previously imported `movebench.study.Study` at module import time; changing that file to lazy-load `Study` via `__getattr__` decoupled point-store module imports from the full study stack.

- Observation: the active MoveBench environment has a broken compiled `pyproj` dependency, so runtime tests cannot rely on the real module.
  Evidence: importing `pyproj` in the current env raised `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found ... libproj.so.25`, so the targeted runtime tests stub `pyproj` and other optional MoveFormer runtime modules to keep the tests focused on schema-4 behavior.

## Decision Log

- Decision: implement a general point-store v2, not an iSSF-only cache.
  Rationale: the current point-store extraction is the shared source of truth for multiple consumers, and replacing it with an iSSF-specific format would either duplicate storage or break non-iSSF code paths.
  Date/Author: 2026-03-30 / Codex

- Decision: group feature arrays by logical raster layer, not by individual band, and replace only the payload format instead of switching immediately to one row-major `env[N,C,P]` store.
  Rationale: layer-grouped mmap is still a relatively small refactor, but it reduces file count and per-batch open/slice overhead substantially compared with one file per band. It removes the `.pt` deserialization and oversized-storage failure modes while preserving the current layer semantics and deferring a larger row-major layout rewrite.
  Date/Author: 2026-03-30 / Codex

- Decision: point-store v2 will write one `env` array and one `valid` array per logical layer under `layers/<layer>.env.npy` and `layers/<layer>.valid.npy`, and it will write all shared row-aligned fields into one structured `shared/shared.npy` file.
  Rationale: a plain `.npy` file can store one dense layer tensor efficiently, and a structured NumPy array can represent mixed shared field shapes and dtypes in one mmap-readable file. Grouping by layer keeps file count manageable, aligns with the current `LayerPlan` grouping, and still allows selective channel reads within a layer.
  Date/Author: 2026-03-30 / Codex

- Decision: do not preserve the eager point-store helper surface for backward compatibility; remove the load-whole-split helpers and port every consumer to the new reader interface.
  Rationale: the user explicitly does not want any load-everything-into-RAM path kept alive. Carrying the old helpers would keep the wrong architecture in the codebase and blur which path is supported.
  Date/Author: 2026-03-30 / Codex

- Decision: use DataLoader batching and worker processes, but keep the dataset in charge of translating a batch of row indices into tensors.
  Rationale: batching and shuffling belong in PyTorch’s `DataLoader` and `BatchSampler`, while storage-specific row gathering belongs in the dataset. In this repository that means `iSSFDataset.collate(indices)` remains the DataLoader hook and delegates to an internal batch-loading method.
  Date/Author: 2026-03-30 / Codex

- Decision: do not implement any schema-3 migration or backward-compatibility path. Old point stores will be deleted and regenerated in schema 4.
  Rationale: the user explicitly does not want code for legacy asset migration. Supporting only schema 4 keeps the implementation smaller and avoids carrying one-off conversion logic in the main extraction path.
  Date/Author: 2026-03-30 / Codex

- Decision: make `movebench.__init__` lazy-load `Study` instead of importing it eagerly.
  Rationale: point-store readers, tests, and smaller utilities should be able to import `movebench` submodules without paying the import cost of the full study pipeline and its heavier optional dependencies.
  Date/Author: 2026-03-30 / Codex

- Decision: keep the new runtime tests hermetic by stubbing broken or optional modules such as `pyproj`, `numba`, `pytorch_lightning`, and `jsonargparse` where the test only needs the point-store integration surface.
  Rationale: these tests are meant to verify schema-4 storage and dataset behavior, not the availability of unrelated compiled dependencies in the local environment.
  Date/Author: 2026-03-30 / Codex

## Outcomes & Retrospective

The repository implementation described by this plan is now in place. Point stores under `study_dir/features/points/<split>/<rasters_hash>/<sampler_hash>` are written and validated only as schema 4 with mmap-readable `.npy` payloads, `PointStoreReader` provides row/block access for shared and per-layer arrays, `bench_v2` readiness recognizes only the new layout, `iSSFDataset` no longer materializes full-split `env`/`valid` arrays at construction time, and MoveFormer/normalization/baseline consumers now read through the same schema-4 reader surface.

Targeted verification passed in the current environment. The consolidated command `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -B -m unittest -v tests.test_point_store_v2 tests.bench_v2.test_runner tests.bench_v2.test_issf_runtime tests.bench_v2.test_moveformer_runtime tests.test_benchmark_default_splits` reported `Ran 14 tests in 117.864s` and `OK`. Those tests cover schema-4 extraction, row/block reads, SSF subset reads, readiness validation, iSSF worker/collate runtime, MoveFormer compatibility, and default-split runner behavior. The real Morato `1h` extraction + benchmark rerun from the acceptance section has not been executed in this turn, so the exact 64 GiB workload acceptance case remains pending rather than verified.

The largest remaining risk is throughput on the full real workload. Layer-grouped mmap solves the current correctness and memory-fragility problem and avoids the `.pt` failure mode, but each batch still touches every requested layer file. If the real benchmark rerun shows that this remains too slow, the follow-up plan should revisit a row-major point-store v3 design. That is intentionally out of scope here; this plan implements the smaller refactor first.

## Context and Orientation

In this repository, a “point store” is the extracted per-split feature cache written under `study_dir/features/points/<split>/<rasters_hash>/<sampler_hash>`. The current format is defined by [movebench/features/manifest.py](/data/vision/beery/scratch/justin/movebench/movebench/features/manifest.py) and [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py). Today each store contains a `manifest.json`, a `shared.pt`, and one `bands/{band_id}.pt` file per channel. Those files are loaded eagerly with `torch.load(...)`, which is what makes current large studies slow and memory-fragile.

The current point-store consumer surface lives primarily in [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), [movebench/datasets/moveformer.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/moveformer.py), and [movebench/features/normalization.py](/data/vision/beery/scratch/justin/movebench/movebench/features/normalization.py). Today `SSFDataset` and `iSSFDataset` load whole splits at dataset construction time, `MoveFormerPointStoreDataset` also loads whole env/valid tensors for its context splits, and normalization reads one whole band at a time. This plan removes that eager helper pattern rather than preserving it.

The supported iSSF benchmark runtime is [movebench/bench_v2/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/executors/issf.py). Its `DataLoader` currently uses `num_workers=0`, and the dataset it wraps already has the entire split resident in `self.env`, `self.valid`, `self.move`, and optional full-split time/individual tensors before the first training step. That eager design is what this plan changes.

A “worker” in this plan means a PyTorch DataLoader subprocess. A worker-safe point-store reader is one that can lazily open its mmap-backed array files once inside a worker process and reuse those file-backed arrays across many batches without reopening them every time. A “batch-oriented dataset” in this plan means a dataset whose `__getitem__` is cheap or trivial and whose `collate(indices)` method is responsible for turning a list of requested row indices into the actual tensors needed by the model.

The benchmark readiness logic lives in [movebench/bench_v2/run.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/run.py). Today it directly requires `shared.pt` to exist and uses `read_manifest(...)` to validate schema 3 manifests, so this plan must update those checks as part of the format change.

## Plan of Work

The first milestone is a test-and-prototype milestone. Create a new active `tests/` package with `tests/__init__.py` and `tests/bench_v2/__init__.py`. Add `tests/test_point_store_v2.py` so it describes schema-4 writing and row-block equivalence for the new layout. Add `tests/bench_v2/test_runner.py` so readiness accepts schema-4 manifests and no longer hard-codes `shared.pt`. Add `tests/bench_v2/test_issf_runtime.py` so it fails if `iSSFDataset` still loads whole splits eagerly at construction time. Add `tests/bench_v2/test_moveformer_runtime.py` or a comparably scoped targeted regression so MoveFormer remains functional on the new point-store layout without any full-split load helper. Add `tests/test_benchmark_default_splits.py` if needed for benchmark payload defaults that are affected by the readiness changes. These tests are the contract for the new format.

The second milestone introduces point-store manifest schema 4 and the new on-disk layout. In `movebench/features/manifest.py`, revise `PointFeatureManifest` so schema 4 `shared` metadata is keyed by shared tensor name rather than by the old single `shared.pt` pseudo-file, and so each band metadata entry contains separate `env_path` and `valid_path`. The schema must be explicit about shapes and dtypes. In `movebench/features/points.py`, change `extract_split_feature_store(...)` so fresh stores write:

    manifest.json
    layers/<layer>.env.npy
    layers/<layer>.valid.npy
    shared/shared.npy

Write the numeric arrays with NumPy so they are contiguous and mmap-readable. Keep the existing directory naming rooted at `point_store_dir(...)`; only the contents change. The manifest write must be the final atomic step so a partially written store does not strand the directory in an unreadable state. There is no schema-3 rewrite path in code; extraction writes schema 4, and old stores are expected to be deleted and regenerated.

The third milestone introduces the reader layer that makes schema 4 useful. In `movebench/features/points.py`, add a point-store row/block reader that lazily opens the `.npy` files with `np.load(..., mmap_mode='r')`, caches those handles per process, and can gather rows in the requested order or iterate contiguous row blocks. The reader should expose one method that loads `env` and `valid` blocks for a list of band ids and row indices by grouping those band ids by layer, one method that loads shared rows for one or more shared keys, and one iterator for contiguous row blocks used by normalization or other sequential consumers. Remove the old load-whole-split helper functions instead of reimplementing them.

The fourth milestone migrates iSSF to the new batch reader. In `movebench/datasets/ssf.py`, keep `SSFDataset` behaviorally unchanged apart from reading from schema 4. Refactor `iSSFDataset` so dataset construction loads only manifest metadata, normalization stats, and any small mappings that are genuinely needed in memory. Do not build or keep full `self.env`, `self.valid`, `self.move`, or full-split time tensors. `iSSFDataset.__len__` must come from manifest shape metadata. `iSSFDataset.collate(indices)` should allocate `env[B, C, P]` and `valid[B, C, P]`, fill them by reading the requested rows from the per-layer mmaps and slicing the requested channels within each layer, normalize `env` using the cached mean/std, and compute movement, optional time, and optional individual index tensors for that same batch only. `collate(...)` may delegate to a private `load_batch(indices)` helper if that makes the code easier to test, but the DataLoader hook remains `dataset.collate`.

The fifth milestone updates the iSSF executor and benchmark readiness. In `movebench/bench_v2/executors/issf.py`, add an explicit `num_workers` method hyperparameter for `issf_linear` and `issf_mlp` in `movebench/bench_v2/config.py`, defaulting to `4`. Build `DataLoader`s with `num_workers=int(method_hparams["num_workers"])`, `persistent_workers=True` when that count is greater than zero, and `pin_memory=True` on CUDA. Keep batching and shuffling in DataLoader. The runtime should continue to call `dataset.collate`, but dataset construction must be fast because it no longer loads full splits. In `movebench/bench_v2/run.py`, replace every `shared.pt` existence check with schema-4 manifest validation against the `shared/shared.npy` entry and its declared field metadata. Readiness should treat anything other than a valid schema-4 store as not ready.

The sixth milestone updates the remaining schema-4 readers. In `movebench/features/normalization.py`, compute band statistics by iterating row chunks from the block reader rather than loading whole bands with `torch.load(...)`. In `movebench/datasets/moveformer.py`, replace the current eager env/valid loads with row/block reads from the new reader while assembling context chains. Baselines and prediction code that currently use `shared.pt` through `load_point_store_shared(...)` must be ported to the new shared-row reader directly. At the end of this milestone there should be no supported code path in the repository that loads an entire point-store split into RAM through an old helper API.

The final milestone is validation on the real workload. Delete the existing Morato `1h` point stores, rerun point extraction so it regenerates them directly in schema 4, and then rerun the single-study `bench_v2` iSSF benchmark that currently dies during eval dataset creation. Acceptance is that dataset setup no longer shows the full-split `loading env bands` progress bars at construction time, the run enters the training loop, and the process is not killed by Slurm during train/eval dataset creation.

## Concrete Steps

All commands below run from `/data/vision/beery/scratch/justin/movebench` and use the MoveBench micromamba environment at `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python`.

1. Add or update the point-store schema-4 tests, then run:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_store_v2`

   Expected pre-implementation behavior: the new tests fail because the manifest still requires `shared.pt` and no `.npy` files are written.

2. Add or update the benchmark readiness and iSSF runtime tests, then run:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.bench_v2.test_runner tests.bench_v2.test_issf_runtime tests.bench_v2.test_moveformer_runtime`

   Expected pre-implementation behavior: the new assertions fail because readiness still hard-codes `shared.pt`, `iSSFDataset` still eagerly loads whole splits, and MoveFormer has not been revalidated on schema 4.

3. Implement schema 4 manifest validation and point-store writing, then rerun:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_store_v2 tests.bench_v2.test_runner`

   Expected post-implementation behavior: the point-store tests and readiness tests pass, and the test output demonstrates that fresh extraction writes the new schema-4 layout.

4. Implement the batch reader, iSSF dataset migration, normalization changes, and executor worker changes, then rerun:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m unittest -v tests.test_point_store_v2 tests.bench_v2.test_runner tests.bench_v2.test_issf_runtime tests.bench_v2.test_moveformer_runtime tests.test_benchmark_default_splits`

   Expected post-implementation behavior: all targeted tests pass, and the iSSF runtime tests show that batches are assembled through `dataset.collate` without full-split eager loading.

5. Delete the real Morato `1h` point stores, then rerun extraction so they are regenerated in schema 4. Use the same layers and sampler config that `bench_v2` resolves for `1h`:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python - <<'PY'`
   `from movebench.study import Study`
   `Study("Morato_8fed4a87_raw_merged").extract_point_features(`
   `    layers_config="configs/layers/all_ae_1h_1h.yaml",`
   `    sampler_config="configs/samplers/issf_defaults.yaml",`
   `    splits=["train", "test_temporal", "test_individual"],`
   `    skip_existing=True,`
   `)`
   `PY`

   Expected behavior after implementation: the command writes fresh schema-4 point stores. Record the actual output in this plan.

6. Rerun the failing iSSF benchmark on the regenerated schema-4 store:

   `/data/vision/beery/scratch/justin/micromamba/envs/movebench/bin/python -m movebench.cli.main bench run experiments/v2/issf.py --studies Morato_8fed4a87_raw_merged --methods issf_linear --max-runs 1 --rerun`

   Expected behavior after implementation: the run no longer dies during `Creating eval dataset`; it enters training and writes the usual `checkpoint_best.pt`, `history.json`, and prediction artifacts under `results/bench_v2/runs/<run_key>`.

## Validation and Acceptance

Acceptance for the storage format change is that rerunning `Study.extract_point_features(...)` produces a point store whose manifest is schema 4, whose layer payloads are `layers/<layer>.env.npy` and `layers/<layer>.valid.npy`, and whose shared payload is one structured `shared/shared.npy` file. Schema-3 stores are out of scope and unsupported by the new code path.

Acceptance for the reader layer is value equivalence and indexing correctness. For the same schema-4 point store and band selection, the new row/block reader must preserve row order exactly, return channels in the requested global `band_ids` order, and agree with direct array slicing on the same schema-4 fixture data. Normalization computed from schema 4 through block iteration must match direct full-array computations to float tolerance.

Acceptance for the iSSF runtime is that `iSSFDataset` construction no longer reads all band arrays eagerly. The benchmark logs should no longer spend minutes in `loading env bands` during `Creating train dataset` and `Creating eval dataset`, because those phases now build metadata-only dataset objects. The benchmark must proceed into the training loop on the Morato `1h` study inside the existing 64 GiB allocation.

Acceptance for repository compatibility is that the targeted MoveFormer and baseline surfaces still run against the new point-store format through the new reader interface, no supported code path depends on the removed eager helper functions, and `bench_v2` readiness reports only valid schema-4 stores as available.

## Idempotence and Recovery

The new extraction path must be safe to rerun. Schema-4 writing is naturally idempotent because rerunning extraction with unchanged raster and sampler settings should detect that the current manifest already matches the requested store and return `status="existing"`.

Write safety still matters. The implementation must write the new `.npy` files first and replace `manifest.json` last so a process death cannot leave the store with a valid-looking manifest that points at incomplete files.

If benchmark validation fails after regenerating the storage, keep the schema-4 point store and debug from there. The intended recovery path is to fix the row/block reader or dataset code and rerun the targeted tests, then rerun the benchmark command.

## Artifacts and Notes

The present Morato failure surface that motivates this plan is:

    Slurm allocation:
    SLURM_MEM_PER_NODE=65536

    Morato 1h point-store sizes:
    train store on disk: about 25G
    test_temporal store on disk: about 41G

    Morato 1h logical env+valid payload:
    train: about 0.27 GiB
    test_temporal: about 0.46 GiB

These numbers show why the current `.pt` layout is the immediate problem: the store is vastly larger than the logical data.

The intended schema-4 directory layout is:

    features/points/train/<rasters_hash>/<sampler_hash>/
      manifest.json
      layers/
        alphaearth_embeddings.env.npy
        alphaearth_embeddings.valid.npy
        modis_lc.env.npy
        modis_lc.valid.npy
        ...
      shared/
        shared.npy

The key runtime design for iSSF is:

    __getitem__(idx) -> int(idx)
    DataLoader batches row indices
    dataset.collate(indices) -> dataset.load_batch(indices)
    load_batch allocates env[B,C,P] and valid[B,C,P]
    load_batch groups requested band_ids by layer, reads only the touched layer files, and fills env/valid in the requested band order
    movement/time/individual tensors are computed for the same batch only

That design keeps batching in PyTorch while keeping storage-specific logic in the dataset. The same reader family must also support sequential block iteration so non-iSSF consumers no longer need any full-load helper.

## Interfaces and Dependencies

In [movebench/features/manifest.py](/data/vision/beery/scratch/justin/movebench/movebench/features/manifest.py), revise `PointFeatureManifest` so schema 4 has this effective structure:

    shared: {
        "path": "shared/shared.npy",
        "shape": [N],
        "dtype": "structured",
        "fields": {
            "candidate_lon": {"dtype": "float32", "shape": [P]},
            "candidate_lat": {"dtype": "float32", "shape": [P]},
            "step_length_m": {"dtype": "float32", "shape": [P]},
            "turn_angle_rad": {"dtype": "float32", "shape": [P]},
            "start_lon": {"dtype": "float32", "shape": []},
            "start_lat": {"dtype": "float32", "shape": []},
            "burst_id": {"dtype": "int64", "shape": []},
            "individual_id": {"dtype": "unicode", "shape": [], "max_chars": "<manifest value>"},
            "point_id_start": {"dtype": "int64", "shape": []},
            "point_id_end": {"dtype": "int64", "shape": []},
            "timestamp_ns": {"dtype": "int64", "shape": []},
            "timestamp_end_ns": {"dtype": "int64", "shape": []},
            "actual_step_time_s": {"dtype": "float32", "shape": []},
            "step_time_error_s": {"dtype": "float32", "shape": []},
        },
    }

    layers: {
        "<layer>": {
            "env_path": "layers/<layer>.env.npy",
            "valid_path": "layers/<layer>.valid.npy",
            "shape": [N, K_layer, P],
            "env_dtype": "float32",
            "valid_dtype": "bool",
            "band_ids": ["<band_id_0>", "...", "<band_id_k>"],
        }
    }

    bands: {
        "<band_id>": {
            "layer": "<layer>",
            "channel_idx": <index within that layer file>,
            "categorical": false,
            "normalize": true,
        }
    }

In [movebench/features/points.py](/data/vision/beery/scratch/justin/movebench/movebench/features/points.py), define a worker-safe row/block reader with this effective interface:

    class PointStoreReader:
        def __init__(self, *, store_dir: Path, manifest: PointFeatureManifest) -> None: ...
        def load_env_valid_rows(
            self,
            *,
            band_ids: list[str],
            row_idxs: np.ndarray | torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]: ...
        def load_shared_rows(
            self,
            *,
            keys: list[str],
            row_idxs: np.ndarray | torch.Tensor,
        ) -> dict[str, torch.Tensor | np.ndarray]: ...
        def iter_row_blocks(
            self,
            *,
            band_ids: list[str] | None,
            shared_keys: list[str],
            block_size: int,
        ) -> Iterator[dict[str, torch.Tensor | np.ndarray]]: ...

`load_env_valid_rows(...)` must preserve the caller’s row order exactly and return `env` and `valid` with shape `(B, C, P)`. It should group the requested `band_ids` by layer, read one `(B, K_layer, P)` slab per touched layer, slice the requested channel indices within that slab, and fill the output tensors without ever reading unrelated layers.
This is what preserves support for covariate sets that select arbitrary subsets of bands: `bench_v2` and dataset callers continue to express the subset as ordered `band_ids`, while schema 4 resolves each `band_id` through `manifest.bands[band_id] -> (layer, channel_idx)`. If a covariate set requests every band in a layer, the reader can copy that whole layer slab directly. If it requests only some bands from a layer, the reader gathers only those channel indices from that layer slab and still returns output in the exact requested global band order.
`load_shared_rows(...)` must read fields from the structured `shared.npy` memmap without materializing unrelated fields. `iter_row_blocks(...)` must yield contiguous row spans in split order and support sequential consumers such as normalization and MoveFormer preprocessing without ever requiring a full-split materialization helper.

In [movebench/datasets/ssf.py](/data/vision/beery/scratch/justin/movebench/movebench/datasets/ssf.py), `iSSFDataset` must end with this effective interface:

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> int: ...
    def collate(self, indices: list[int] | torch.Tensor) -> dict[str, torch.Tensor | None]: ...

The returned batch dict must contain:

    env: float32 (B, C_env, P)
    valid: bool (B, C_env, P)
    move: float32 (B, 3, P)
    time: float32 (B, 4) or None
    individual_idx: int64 (B,) or None

In [movebench/bench_v2/config.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/config.py), add:

    "num_workers": 4

to the default `hparams` for both `issf_linear` and `issf_mlp`.

In [movebench/bench_v2/executors/issf.py](/data/vision/beery/scratch/justin/movebench/movebench/bench_v2/executors/issf.py), the loader construction must end with this effective behavior:

    DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=...,
        num_workers=num_workers,
        collate_fn=dataset.collate,
        pin_memory=device.type == "cuda",
        persistent_workers=(num_workers > 0),
    )

`num_workers=0` must still work for debugging, but the default benchmark path should use worker-side batching.

Revision note: created on 2026-03-30 after repository inspection, discussion of row-major versus band-based point-store designs, confirmation that the current `.pt` files are the immediate scalability bottleneck, and agreement to pursue a smaller band-based mmap point-store v2 first because it solves the OOM risk with a narrower refactor.

Revision note: updated on 2026-03-30 after the user clarified that no load-whole-split helper functionality should survive anywhere in the supported codebase. The plan now removes the old eager helper APIs instead of preserving them behind the new storage format, and it explicitly ports MoveFormer, baselines, normalization, and iSSF to the new row/block reader surface.

Revision note: updated again on 2026-03-30 after the user questioned the need for one file per shared field. The plan now uses a single structured `shared/shared.npy` file for all shared row-aligned fields.

Revision note: updated again on 2026-03-30 after the user pointed out that the same grouping logic should apply to feature arrays. The plan now stores `env` and `valid` arrays per logical layer instead of per band, with manifest entries that map `band_id` to `(layer, channel_idx)` for selective reads.
