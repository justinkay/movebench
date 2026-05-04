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


## Downloading with GEE


If you will be downloading rasters with `movebench download gee`, also install the optional GEE dependencies and authenticate with your Google Earth Engine account:

```bash
pip install -e '.[gee]'
```

```bash
earthengine authenticate --auth_mode=notebook
```

## Slurm

The Slurm launchers mirror the local `extract`, `pack`, and `bench run` commands and accept shared defaults via `--slurm-config`.

```bash
movebench slurm pack --studies all --timescales 1h --slurm-config configs/slurm/beery.yaml
movebench slurm bench run experiments/main_results.py --studies mini --slurm-config configs/slurm/beery.yaml
```
