# MoveBench

## Installation

1. Make sure to **build GDAL with HDF4 drivers**. It's recommended to do this with **conda-forge**.

```bash
conda create -n movebench -c conda-forge python=3.11 \
  gdal hdf4 rasterio pyproj
conda activate movebench
```

2. Then, install the `movebench` Python package with pip from the repo root:

```bash
pip install -e .
```

## Optional: Weights & Biases logging

MoveBench's YAML benchmark runner supports optional Weights & Biases (W&B) logging for train/test loss (NLL).

Install the extra:

```bash
pip install -e ".[wandb]"
```

Enable in a benchmark config:

```yaml
wandb:
  enabled: true
  project: movebench
  # entity: your-team
  mode: online  # or offline
```

For quick local runs without uploading, set `WANDB_MODE=offline` (or use `mode: offline`).

3. If you will be downloading rasters, authenticate with your Google Earth Engine account

```bash
earthengine authenticate --auth_mode=notebook
```

## Download rasters

Download GEE rasters for a single study:

```bash
python scripts/download_study.py <study_id> --config configs/layers_all.yaml
```

Download GEE rasters for all studies present under the studies root:

```bash
movebench download --config configs/layers_all.yaml
```

## Example notebook

See `notebooks/example.ipynb` to get started.
