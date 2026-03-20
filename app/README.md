# MoveBench trajectory app

This is a standalone FastAPI launcher for the trajectory viewer.

Run it from the repo root:

```bash
python -m app.server
```

It reads the default studies directory from the same repo-level path config used by MoveBench, which resolves from:

1. `MOVEBENCH_DATA_ROOT`, if set
2. otherwise `configs/paths.yaml:data_root`

The UI lists studies that contain `trajectories.csv` and visualizes that file directly.
