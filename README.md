# MoveBench

## Installation

1. Make sure to **build GDAL with HDF4 drivers**. It's recommended to do this with conda-forge.

```bash
conda create -n movebench -c conda-forge python=3.11 gdal hdf4 rasterio pyproj libgdal-hdf4
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

Other provided experiment definitions include `experiments/baselines.py`, `experiments/covariate_comparison.py`, and `experiments/moveformer.py`.

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
  --slurm-config path/to/slurm.yaml
```
