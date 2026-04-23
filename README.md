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

```
movebench download --layers-config configs/layers/lite_12h_12h.yaml --studies Accio_967b7813_raw_merged
```

**3. Extract:** Extract point features for observed and candidate points from downloaded rasters, to be used by SSF-style methods.

```
```

**4. Pack (Optional: only required for deepSSF):** Create WebDataset shards, which bundle full raster stacks for efficient training/inference of raster-based methods.
```
```

**5. Bench:** Benchmark all forecasting methods.
```
```

You can visualize predictions for your newly trained models with [TODO: Notebook links]

## Full data download

Download all covariates for all studies at at all timescales (warning: this will download roughly 7TB of data):

TODO
```bash
movebench download --layers-config configs/layers/all_ae_1h_1h.yaml --studies all
```

## Full documentation

Ingest: [Doc]
Download: [Doc]
Extract: [Doc]
Pack: [Doc]
Bench: [Doc]
Slurm: [Doc]

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
