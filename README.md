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

## CLI functionality

### Download

Download all GEE rasters for all studies at at a 1 hour timescale:

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

Create WebDataset shards for training raster-based methods like DeepSSF. 

```bash

```

To launch on slurm,

### Bench

```bash
```

To launch on slurm,

```bash
```
