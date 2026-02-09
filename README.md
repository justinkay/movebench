# MoveBench

## Prereqs

```bash
pip install -e .
```

For downloads you also need the Earth Engine Python API and to be authenticated.
```bash
earthengine authenticate --auth_mode=notebook
```

### Path config(s)

Some layers load from local datasets (MODIS HDFs, bioclim tiffs, etc). Set these
paths in `configs/paths.yaml` or point to an alternate file via:

- `MOVEBENCH_PATHS_CONFIG=/abs/path/to/paths.yaml`

### Data root

By default, studies live under `data/movebench/studies/`. Override via:

- `MOVEBENCH_DATA_ROOT=/abs/path/to/movebench_data_root`

## 1) Ingestion

Ingest a Movebank-formatted CSV into a new study directory:

```bash
python scripts/ingest.py /path/to/movebank_export.csv
python scripts/ingest.py /path/to/movebank_export.csv --dry-run
```

Outputs:

- `data/movebench/studies/<study_id>/trajectories.csv`
- `data/movebench/studies/<study_id>/manifest.json`

The ingester ensures required columns exist and will compute `set=train/test`
splits if they are not already present.

## 2) Raster Download

Download all rasters required by a layer config:

```bash
python scripts/download_study.py tsavo_lions --config configs/layers_all.yaml
```

Outputs are written under:

- `data/movebench/studies/<study_id>/rasters/<rasters_hash>/...`

## 3) Feature Extraction (SSF / iSSF point stores)

Extract point feature stores for a config. SSF uses empirical sampling
(`parametric=False`), iSSF uses parametric sampling (`parametric=True`).

```bash
conda run -n movement python -c "from movebench import Study; Study('tsavo_lions').extract_point_features(config='configs/layers_local.yaml', parametric=False, splits=['train','test'], n_workers=1)"
conda run -n movement python -c "from movebench import Study; Study('tsavo_lions').extract_point_features(config='configs/layers_local.yaml', parametric=True,  splits=['train','test'], n_workers=1)"
```

Outputs:

- `.../features/points/<split>/<config_hash>/<sampler_hash>/features.pt`
- `.../features/points/<split>/<config_hash>/<sampler_hash>/manifest.json`

## 4) Normalization

Compute and cache per-channel normalization stats from the extracted point store:

```bash
conda run -n movement python -c "from movebench import Study; Study('tsavo_lions').compute_normalization_vals(from_='points', config='configs/layers_local.yaml')"
```

Outputs:

- `.../normalization/points/<config_hash>/<sampler_hash>/stats.json`

## 5) Dataset Creation

```bash
conda run -n movement python - <<'PY'
from movebench import Study
study = Study("tsavo_lions")
chip_ds = study.chip_dataset(split="train", config="configs/layers_local.yaml")
ssf_ds  = study.ssf_dataset(split="train", config="configs/layers_local.yaml")
issf_ds = study.issf_dataset(split="train", config="configs/layers_local.yaml")
print(len(chip_ds), len(ssf_ds), len(issf_ds))
PY
```

## 6) Training / Benchmark

Run the iSSF benchmark (linear + MLP). This script will automatically download
rasters, extract parametric point features, and compute normalization if they
are missing for the requested study/config.

```bash
conda run -n movement python scripts/run_issf_benchmark.py \
  --study tsavo_lions \
  --config configs/layers_local.yaml \
  --output-dir results/issf \
  --device cpu
```

Outputs:

- `results/issf/raw_results.json`
- `results/issf/nll_{train,test}.csv`
- `results/issf/top{k}_{train,test}.csv`

