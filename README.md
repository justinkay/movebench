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

If you will run benchmark commands, also install the optional bench dependencies:

```bash
pip install -e .[bench]
```

3. If you will be downloading rasters, authenticate with your Google Earth Engine account

```bash
earthengine authenticate --auth_mode=notebook
```

## Quickstart: Data setup

Download all trajectories in MoveBench v1.0 here (165 MB): https://drive.google.com/file/d/1LAcoi-XzvmtXj4VYQ1pGNFXeZfiua4WP/view?usp=sharing

As a quickstart, let's prepare data and train models for some white storks 𓅮 We will use a sample from MoveBench that includes data from [this study](https://link.springer.com/article/10.1186/s40462-022-00328-3).

**1. Ingest:** Ingest MoveBank-formatted CSVs into the repository.

```
movebench ingest path/to/movebench-csvs-v1.0/
```

**2. Download:** Download a set of raster files for the study at some forecasting timescale. To get up and running quickly, lets use the `lite` set of covariates and a 12h forecast horizon.

Make sure you have followed the instructions above to authenticate Google Earth Engine. Then,

```bash
movebench download --layers-config configs/layers/lite_12h_12h.yaml --studies Accio_967b7813_raw_merged
```

**3. Extract:** Extract point features for observed and candidate points from downloaded rasters, to be used by SSF-style methods.

```bash
movebench extract --layers-config configs/layers/lite_12h_12h.yaml --sampler-config configs/samplers/issf_defaults.yaml --studies Accio_967b7813_raw_merged
```

**4. Pack (Optional: only required for deepSSF):** Create WebDataset shards, which bundle full raster stacks for efficient training/inference of raster-based methods.
```bash
movebench pack --studies Accio_967b7813_raw_merged --timescales 12h
```

**5. Bench:** Benchmark forecasting methods. If you skipped the previous step, the deepSSF runs will stay blocked.
```bash
movebench bench run experiments/main_results.py --studies Accio_967b7813_raw_merged
```

You can visualize predictions for your newly trained models with [prediction_examples.ipynb](notebooks/prediction_examples.ipynb) or [recursive_forecasting.ipynb](notebooks/recursive_forecasting.ipynb).

## Full data download

Download all covariates for all studies at all benchmark timescales (warning: this will download roughly 7TB of data):

```bash
movebench download --layers-config configs/layers/all_ae_1h_1h.yaml --studies all
movebench download --layers-config configs/layers/all_ae_12h_12h.yaml --studies all
movebench download --layers-config configs/layers/all_ae_1d_1d.yaml --studies all
movebench download --layers-config configs/layers/all_ae_7d_7d.yaml --studies all
```

## Full documentation

- [Ingest](#ingest)
- [Download](#download)
- [Extract](#extract)
- [Pack](#pack)
- [Bench](#bench)
- [Slurm](#slurm)

### Ingest

Ingest one MoveBank-formatted CSV or a directory of CSVs into the configured MoveBench data root:

```bash
movebench ingest path/to/movebench-csvs-v1.0/
```

### Download

```bash
movebench download --layers-config configs/layers/all_ae_1h_1h.yaml --studies all
```

For "MoveBench-Mini", specify `--studies mini`. 

### Extract

Extract point features for training SSF-style methods. To extract the default layers for all studies at a 1 hour timescale:

```bash
movebench extract --layers-config configs/layers/all_ae_1h_1h.yaml --sampler-config configs/samplers/issf_defaults.yaml --studies all
```

To launch this on many slurm jobs,

```bash
movebench slurm extract --layers-config configs/layers/all_ae_1h_1h.yaml --sampler-config configs/samplers/issf_defaults.yaml --studies all --slurm-config {path/to/slurm/config.yaml}
```

### Pack

Create deepSSF cache shards after extraction:

```bash
movebench pack --studies all --timescales 1h
```

### Bench

Run the checked-in main benchmark experiment locally:

```bash
movebench bench run experiments/main_results.py --studies mini
```

### Slurm

The Slurm launchers mirror the local `extract`, `pack`, and `bench run` commands and accept shared defaults via `--slurm-config`.

```bash
movebench slurm pack --studies all --timescales 1h --slurm-config configs/slurm/beery.yaml
movebench slurm bench run experiments/main_results.py --studies mini --slurm-config configs/slurm/beery.yaml
```
