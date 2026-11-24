# MoveBench

## Installation

1. Install the geospatial dependencies with conda-forge.

```bash
conda create -n movebench -c conda-forge python=3.11 gdal rasterio pyproj
conda activate movebench
```

2. Then, install the `movebench` Python package with pip from the repo root:

```bash
pip install -e .
```

## Quickstart

As a quickstart, let's prepare data and train models for some white storks! 𓅮  We will use a sample from [this study](https://link.springer.com/article/10.1186/s40462-022-00328-3).

### CLI Version

**1. Download:** We will use just the `lite` set of covariates (NDVI and DEM) to quickly get up and running. Use the MoveBench CLI to download:

```bash
movebench download --studies Accio_967b7813_raw_merged --layers-config configs/layers/lite_12h_12h.yaml --n-workers 4 
```

**2. Extract point features:** To quickly train SSFs, we will precompute all observed and candidate point features.

```bash
movebench extract --layers-config configs/layers/lite_12h_12h.yaml --sampler-config configs/samplers/issf_defaults.yaml --studies Accio_967b7813_raw_merged
```

**3. Pack deepSSF shards (Optional):** If you would like to train a deepSSF model as part of this quickstart, pack WebDataset shards:

```bash
movebench pack --studies Accio_967b7813_raw_merged --timescales 12h --layers-config configs/layers/lite_12h_12h.yaml
```

**4. Install MoveFormer dependencies (Optional):** If you would like to train a MoveFormer model as part of this quickstart, install its dependencies:

```bash
pip install -e '.[moveformer]'
```

**5. Train models!**

```bash
movebench bench run experiments/lite.py --studies Accio_967b7813_raw_merged
```

Note that the deep learning models (deepSSF and MoveFormer) will take much longer to train unless you have a GPU.

**6. Visualize results (Optional):**

If you would like to visualize prediction results from your trained models, first install visualization dependencies:

```bash
pip install -e '.[plotting]'
```

Use [quickstart.ipynb](notebooks/quickstart.ipynb) to visualize outputs.


### Notebook version

A walkthrough of the same steps using the MoveBench Python API instead of the CLI can be found in [quickstart.ipynb](notebooks/quickstart.ipynb). 

In order for the visualizations in the notebook to work, first install visualization dependencies:

```bash
pip install -e '.[plotting]'
```

There are known issues with OpenMP during deepSSF training in the notebook workflow on some platforms; if you run into issues, the easiest workaround is to use the CLI workflow for training.




## Full downloads

Use `movebench download` to download published MoveBench studies.

**Download all covariates for all studies.** This is massive (7 TB) and will take a very long time to transfer. Consider downloading a smaller subset (detailed below):

```bash
movebench download --studies all
```

Alternatively, you can use `--studies mini` to download the MoveBench-Mini set (10 studies, ~500GB for all covariates), or a comma separated list of named studies (e.g., in the quickstart, we used `--studies Accio_967b7813_raw_merged`). You can find the full list of named studies and how large they are with the full covariate set [here](https://data.csail.mit.edu/movebench/index.html).


**To download a subset of covaraites, use layer configs:** This is useful for smaller workflows. You can find some examples in [configs/layers](configs/layers/). Any subset of the full published set in MoveBench can be downloaded directly from the archive. For example, to fetch only the 12h `lite` covariates for the quickstart study:

```bash
movebench download \
  --studies Accio_967b7813_raw_merged \
  --layers-config configs/layers/lite_12h_12h.yaml \
  --n-workers 4
```

## Starting from Movebank CSVs

You can also start from the full MoveBench CSV bundle [here](https://drive.google.com/file/d/1kh-6EcgCx_ds_RBK9tJfKw5cbTaD6m8Y/view?usp=sharing) or from your own Movebank-formatted CSVs.

**Ingest one CSV:**

```bash
movebench ingest /path/to/study.csv --study-id my_study
```

**Ingest a directory of CSVs:**

```bash
movebench ingest /path/to/movebench_csvs
```

## Downloading with GEE

If you want to download rasters for your own study, or download different rasters for the MoveBench studies, or for some other reason want to use Google Earth Engine to perform downloads, use `movebench download gee`. You will first need to the optional GEE dependencies and authenticate with your Google Earth Engine account:

```bash
pip install -e '.[gee]'
```

```bash
earthengine authenticate --auth_mode=notebook
```

`movebench download gee` works on local studies that already exist under `data/rasters/studies`, whether they came from `movebench ingest` or from the public ZIP workflow.


To define a custom layer set, start from one of the YAMLs in `configs/layers/`, copy it, and edit the `raster_params`, `layers`, and `chip_groups` sections. `configs/layers/all.yaml` is a good general starting point, and `configs/layers/lite_12h_12h.yaml` is a minimal example. Then pass the new file to `movebench download gee`:

```bash
movebench download gee \
  --studies my_study \
  --layers-config configs/layers/my_layers.yaml
```

## Running experiments

**Extract point features:** Run `movebench extract` with the layer config(s) you want to use for modeling, and a sampler config:

```bash
movebench extract \
  --layers-config configs/layers/my_layers.yaml \
  --sampler-config configs/samplers/issf_defaults.yaml \
  --studies my_study
```

You can repeat `--layers-config` to prepare multiple layer sets.

**Prepare deepSSF data (Optional):** The default `wds` backend uses packed
WebDataset shards:

```bash
movebench pack --studies my_study --timescales 12h --layers-config configs/layers/my_layers.yaml
```

For the built-in benchmark timescales, you can omit `--layers-config` and pack directly from the defaults:

```bash
movebench pack --studies mini --timescales 1h 12h 1d 7d
```

**Run experiments:** Experiment definitions live in [experiments](experiments/). For example, to run the full benchmark on the `mini` set:

```bash
movebench bench run experiments/main_results.py --studies mini
```

Other provided experiment definitions include `experiments/baselines.py` and `experiments/covariate_comparison.py`.

Shared MoveFormer uses one model trained on several studies. Train that model first, then use the second command to make predictions and calculate benchmark scores:

```bash
movebench bench run experiments/moveformer_full_trainonly.py --studies mini
movebench bench run experiments/main_results.py --studies mini --methods moveformer_full_predictonly
```

Wait for training to finish before running prediction. The prediction command only loads an already trained model; it cannot start training for you. The shared MoveFormer results in `main_results.py` also need this training step.

Use `--studies mini` to train on MoveBench-Mini, or `--studies all` to train on all available studies.

If you want results for only some studies, use `trained_on_studies` in the prediction experiment to say which studies the saved model was trained on. For example, `studies="mini"` with `trained_on_studies="all"` generates results for MoveBench-Mini using the model trained on all studies. This is also useful when rerunning predictions for a few studies that failed.

If you leave out `trained_on_studies`, MoveBench looks for a model trained on the studies selected for prediction. Keep this setting fixed when adding more prediction studies so MoveBench can use the same model and skip finished results. Each study you predict for must be in the model's training study list.

To repeat work, add `--rerun` to the command you want to repeat:

- On the prediction command, it replaces results for the studies you requested.
- On the training command, it trains a new shared model and deletes the study results made with the old model. Run prediction again afterward. MoveBench will refuse to replace a model while a prediction job is using it.

If training stopped after saving the model but before writing its summary, repeat the training command without `--rerun`. It will load the saved model and finish the summary, keeping existing predictions. If the saved model is missing, it must train again and clear the old predictions.

**Inspect benchmark state (Optional):**

```bash
movebench bench plan experiments/main_results.py --studies mini
movebench bench summary --studies mini
movebench bench status --studies mini
movebench bench report
```


## Slurm

`movebench ingest` and the download commands run locally. The heavier downstream steps have Slurm launchers that accept shared defaults via `--slurm-config`.

You will need to create your own slurm config that matches your configuration. You can find examples in [configs/slurm/](configs/slurm/).

```bash
movebench slurm extract \
  --layers-config configs/layers/all_ae_1h_1h.yaml \
  --layers-config configs/layers/all_ae_12h_12h.yaml \
  --layers-config configs/layers/all_ae_1d_1d.yaml \
  --layers-config configs/layers/all_ae_7d_7d.yaml \
  --sampler-config configs/samplers/issf_defaults.yaml \
  --studies mini \
  --slurm-config path/to/slurm.yaml

movebench slurm pack \
  --studies mini \
  --timescales 1h 12h 1d 7d \
  --slurm-config path/to/slurm.yaml

movebench slurm bench run \
  experiments/main_results.py \
  --studies mini \
  --max-cpu-jobs 32 \
  --max-gpu-jobs 32 \
  --slurm-config path/to/slurm.yaml
```

Shared MoveFormer also needs two separate commands on Slurm. Submit training first:

```bash
movebench slurm bench run experiments/moveformer_full_trainonly.py \
  --studies mini \
  --slurm-config path/to/slurm.yaml
```

Wait for the training job to finish, then submit prediction:

```bash
movebench slurm bench run experiments/main_results.py --methods moveformer_full_predictonly \
  --studies mini \
  --slurm-config path/to/slurm.yaml
```

Submitting both methods together will not make prediction wait for training. Once the model is ready, predictions for different studies can run in parallel.

## Adding a benchmark method

To add a method, write a Python class that can fit a model and make predictions, then list it in `METHODS` in [bench/config.py](movebench/bench/config.py). MoveBench will save its predictions, calculate Energy Scores, and include the results in reports.

### Start with an example

There are two small examples you can copy:

- [origin_method.py](examples/origin_method.py) predicts that the animal stays where it is. It needs no training and shows the simplest prediction format.
- [candidate_method.py](examples/candidate_method.py) trains a linear model to score possible next locations. It shows how to load training data, save and load a trained model, and predict over a grid of locations. Training is limited to two batches to keep the example short.

If you have prepared the 12-hour point features, normalization statistics, and AlphaEarth rasters for Accio, run both examples from the repository directory:

```bash
python -m movebench.cli.main bench run examples/methods_experiment.py \
  --studies Accio_967b7813_raw_merged
```

These examples live outside `experiments/`, so they are not included in the regular benchmark. The example experiment adds its methods when you load it. For your own method, put the class in `movebench/models/` and add its entry to `METHODS` in `bench/config.py`. That also makes the method available to separately launched Slurm jobs.

### Write the fitting and prediction code

Your class needs three methods:

```python
class MyMethod:
    def __init__(self, *, model_config, hparams, seed):
        self.model_config = dict(model_config)
        self.hparams = dict(hparams)
        self.seed = int(seed)

    def fit(self, train, validation, *, work_dir, restore_from=None):
        # Build the model and train it, or load the saved model in restore_from.
        # Keep it on this object, for example as self.model_.
        return self

    def predict(self, data):
        # Use the fitted model to predict for every point in data.
        # Return a dictionary in one of the formats described below.
        ...
```

Use `__init__` to remember settings. Build the model and load its data in `fit`. MoveBench sets the Python, NumPy, and PyTorch random seeds before creating your object.

For each run that needs work, MoveBench calls `fit` once, then calls `predict` for each test dataset that needs predictions. `fit` should return `self`. Even a method that needs no training has a `fit` method, as the origin example shows. If your method only trains a model for later use, set `evaluates=False` in its registration and omit `predict`.

If you use PyTorch, keep the network on your method object, such as `self.model_`. PyTorch's `model.train()` switches the network into training mode; your method's `fit` still needs to perform the training itself.

### Load the data your method needs

For a method trained on one study, `train`, `validation`, and `data` are [DataSplit](movebench/datasets/types.py) objects. A split is one part of a study's data, such as its training observations or its temporal test set. `validation` can be `None` if the method does not need it.

Each object tells you which split it describes (`split_key`), the study and environmental variables it uses (`metadata`), and where to find the prepared data (`inputs`). It does not load all the observations into memory. Use the helpers in `movebench.datasets` to read the data in the form your model needs:

| What you need | Helper |
| --- | --- |
| Observation IDs and the animal's previous movement | `points.point_context(data)` |
| Movement steps from the training data | `points.training_steps(train)` |
| Features at possible next locations, including the observed location | `ssf.candidate_dataset(data)` and `candidate_loader(...)` |
| Environmental features on a grid around each observation | `chips.grid_dataset(data, ...)` |
| Prepared images for deepSSF | `rasters.PreparedRasters(train, ...)` |
| Sequences of observations for MoveFormer | `moveformer.sequence_dataset(data, ...)` and `sequence_loader(...)` |

The candidate example shows how to use these helpers together. During training, the candidate loader reads and normalizes environmental features, then calls `make_model_batch` to assemble the model's inputs. For each observation, the observed next location is the last candidate. During prediction, `build_ssf_grid_artifact` uses the same `make_model_batch` function for the grid cells. This keeps the environmental, movement, and optional time features consistent between training and prediction.

If you call `make_model_batch` directly, pass environmental features that have already been normalized. For a batch with `N` observations, `C` environmental variables, and `P` possible locations, `env` and its validity mask have shape `[N, C, P]`; distances and angles have shape `[N, P]`. The function assembles the features without normalizing them again. See [ssf.py](movebench/datasets/ssf.py) for the `timestamp_ns` argument used when `include_time=True`.

### Return predictions

Return a dictionary containing CPU tensors. Include `point_id_start`, with one unique 64-bit integer ID for every requested observation, and copy `geometry` from `data.metadata["geometry"]`. MoveBench uses the IDs to match each prediction to the observed movement. Return all requested predictions; MoveBench decides afterward which observations can be scored.

Choose one of these formats. Here, `N` is the number of observations, `P` is the number of possible locations, and `H` and `W` are the height and width of a grid:

| Prediction | `prediction_kind` | Fields to return |
| --- | --- | --- |
| One displacement per observation | `deterministic` | `dx_m[N]` and `dy_m[N]`, in meters |
| Probabilities for a list of locations | `discrete` | `coords_m[N, P, 2]` in meters and `probs[N, P]` |
| A score for each grid cell | `grid` | `scores[N, H, W]` and the cell width in meters, `pixel_size_m` |
| A probability for each grid cell | `grid` | `probs_hw[N, H, W]` and the cell coordinates, `x_coords_m[W]` and `y_coords_m[H]`, in meters |

For the list of locations, probabilities must be nonnegative and have a positive sum for each observation. For a probability grid, each grid must sum to one. The origin and candidate examples return the first and third formats, respectively.

### Tell MoveBench how to run your method

Add an entry to `METHODS` in [bench/config.py](movebench/bench/config.py). [methods_experiment.py](examples/methods_experiment.py) shows entries for both examples. The main settings are:

- `implementation`: the Python module and class to import, such as `movebench.models.my_method:MyMethod`.
- `model` and `hparams`: the default model and training settings passed to your constructor.
- `artifact_requirement`: the kind of prepared data your method needs. Use `points` for the point-based examples.
- `requires_normalization_stats`: whether the prepared data must include training means and standard deviations.

You can also set `gpus` to the number of GPUs a Slurm job should request. Your method must still choose its own PyTorch device. Reports will include a newly registered method without changes to the plotting code.

### Save and load trained models

A checkpoint is a file containing a trained model and its training results. When `restore_from` is `None`, `fit` should train a new model. Otherwise, it must load the model specified by `restore_from`. If loading fails, let the error stop the run. Starting training instead would produce a different model from the one MoveBench expects to reuse.

Use `save_checkpoint` and `read_checkpoint` from [training.py](movebench/training.py), as the candidate example does. They save the model weights, training step counts, and training and validation scores. The `restore_from` dictionary contains the file path as `checkpoint_path`; its `training_run_key` and `training_signature` fields identify a shared training run and are `None` for an ordinary model. Keep the saved training results when loading a model rather than calculating them again.

Once your method can load a saved model this way, add `fit_reuse="checkpoint"` to its registration. This lets MoveBench keep predictions already made with that model when retrying a run. Methods that can recreate exactly the same fitted state from unchanged data and settings can use `fit_reuse="rebuild"` instead. Without either setting, MoveBench removes old predictions before calling `fit` again. Rerun affected experiments when you change a method or its input data.

To include training results in the run summary, set `self.training_summary_` in `fit`. The candidate example shows the fields for step counts and training and validation scores. MoveBench saves this information before prediction, so a later prediction failure does not erase the training record.

You can write your own training loop, use another library, or use MoveBench's `fit_or_restore` helper. For the helper, supply an object that holds the model and defines how to perform one training step and evaluate it. `TrainerConfig` sets the training limit and when to evaluate; the helper saves the training history in `work_dir/history.json`. See `ISSFMethod` in [models/ssf.py](movebench/models/ssf.py) for a complete example.

### Train one model on several studies

A shared method receives a `SharedTrainingData` object as `train` and `None` as `validation`. Each item in `train.studies` contains one study's training data and its validation splits. Keep these together when building datasets. For example, a shared MoveFormer method can create sequences like this:

```python
from movebench.bench.moveformer_taxa import moveformer_taxon_for_study
from movebench.datasets.moveformer import sequence_dataset

for study in train.studies:
    taxon, _ = moveformer_taxon_for_study(study.train.metadata["study_id"])
    train_sequences = sequence_dataset(
        study.train, max_len=16, min_len=1, candidate_subset_size=16,
        taxon_embedding=taxon, lazy_rows=True,
    )
    validation_sequences = [
        sequence_dataset(
            split, max_len=16, min_len=1,
            context_splits=["train", split.split_key],
            taxon_embedding=taxon, lazy_rows=True,
        )
        for split in study.validation
    ]
```

Here, `moveformer_taxon_for_study` supplies the study's species information. `context_splits` lets each validation sequence include earlier observations from the training split, and `lazy_rows=True` reads features as they are needed. Some studies have no validation data, or too few observations to form a usable sequence; your method needs to account for that while keeping their training data.

When predicting for a split, include its earlier observations as well. `allow_prediction_only_segments=True` includes sequences needed for prediction even when they cannot be used to calculate a training loss:

```python
prediction_sequences = sequence_dataset(
    data, max_len=16, min_len=1,
    context_splits=["train", data.split_key],
    taxon_embedding=taxon, allow_prediction_only_segments=True, lazy_rows=False,
)
```

The shared input also records the full requested study list and any studies with no usable training steps. These records let MoveBench identify which shared model a prediction should use. See [the input and method reference](astra_refactor_bench_plan_v2/target_architecture.md) for the complete fields.

### Current limits

Saved models are for making predictions after training has finished. They cannot resume an interrupted training job. A method returns all predictions for one split at a time, and predictions describe a single movement horizon rather than a chain of future movements. Shared training and prediction use separate commands, as shown above. There is no automatic conversion of checkpoints or results from older versions.
