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

3. If you will be downloading rasters, authenticate with your Google Earth Engine account

```bash
earthengine authenticate --auth_mode=notebook
```

## Example notebook

See `notebooks/example.ipynb` to get started.
