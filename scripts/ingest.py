#!/usr/bin/env python
"""Ingest a Movebank-formatted CSV into movebench2 as a new study.

This script creates:
  data/movebench2/studies/<study_id>/trajectories.csv
  data/movebench2/studies/<study_id>/manifest.json

Unless overridden via:
  - `--data-root`, or
  - `MOVEBENCH2_DATA_ROOT`
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _insert_repo_root_on_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Movebank often uses hyphenated column names; allow underscore variants.
    return df.rename(columns=lambda c: c.replace("_", "-"))


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    from movebench2.trajectories import (
        INDIVIDUAL_COL,
        LAT_COL,
        LON_COL,
        TIME_COL,
    )

    df = df.copy()

    # Fill individual-local-identifier from individual-id if needed.
    if INDIVIDUAL_COL not in df.columns or df[INDIVIDUAL_COL].isna().all():
        if "individual-id" in df.columns:
            df[INDIVIDUAL_COL] = df["individual-id"]
        else:
            raise ValueError(
                "Missing required individual column. Expected either "
                f"{INDIVIDUAL_COL!r} or 'individual-id'."
            )

    required = [TIME_COL, LAT_COL, LON_COL, INDIVIDUAL_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required column(s): {missing}")

    return df


def _parse_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    from movebench2.trajectories import LAT_COL, LON_COL, TIME_COL

    df = df.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, LAT_COL, LON_COL])
    return df


def _ensure_set_split(df: pd.DataFrame) -> pd.DataFrame:
    from movebench2.trajectories import INDIVIDUAL_COL, SET_COL, TIME_COL

    df = df.copy()

    if SET_COL in df.columns:
        vals = df[SET_COL].dropna().astype(str).unique().tolist()
        if set(vals).issubset({"train", "test"}):
            return df

    # Compute per-individual midpoint split.
    spans = df.groupby(INDIVIDUAL_COL)[TIME_COL].agg(start="min", end="max")
    split_times = spans["start"] + (spans["end"] - spans["start"]) / 2
    df["split_time"] = df[INDIVIDUAL_COL].map(split_times)
    df[SET_COL] = (df[TIME_COL] <= df["split_time"]).map({True: "train", False: "test"})
    return df


def _sort_and_assign_point_id(df: pd.DataFrame) -> pd.DataFrame:
    from movebench2.trajectories import INDIVIDUAL_COL, POINT_ID_COL, TIME_COL

    df = df.sort_values([INDIVIDUAL_COL, TIME_COL]).reset_index(drop=True)
    df[POINT_ID_COL] = df.index.astype(int)
    return df


def _write_manifest(*, study_dir: Path, study_id: str, source_csv: str, df: pd.DataFrame) -> None:
    from movebench2.features.manifest import utc_now_iso
    from movebench2.trajectories import INDIVIDUAL_COL, SET_COL

    n_points = int(len(df))
    n_individuals = int(df[INDIVIDUAL_COL].nunique())
    n_train = int((df[SET_COL] == "train").sum()) if SET_COL in df.columns else 0
    n_test = int((df[SET_COL] == "test").sum()) if SET_COL in df.columns else 0

    manifest = {
        "schema_version": "1",
        "created_at": utc_now_iso(),
        "study_id": str(study_id),
        "source_csv": str(source_csv),
        "trajectories": {
            "file": "trajectories.csv",
            "n_points": n_points,
            "n_individuals": n_individuals,
            "n_train": n_train,
            "n_test": n_test,
        },
    }
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ingest a Movebank CSV into movebench2.")
    parser.add_argument("csv_path", type=Path, help="Path to a Movebank-formatted CSV.")
    parser.add_argument("--study-id", type=str, default=None, help="Study id (default: CSV stem).")
    parser.add_argument("--data-root", type=Path, default=None, help="Override MOVEBENCH2_DATA_ROOT for this run.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing study directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without writing.")
    args = parser.parse_args(argv[1:])

    _insert_repo_root_on_path()

    if args.data_root is not None:
        os.environ["MOVEBENCH2_DATA_ROOT"] = str(args.data_root)

    from movebench2 import paths
    from movebench2.trajectories import INDIVIDUAL_COL, SET_COL

    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        parser.error(f"CSV not found: {csv_path}")

    study_id = str(args.study_id) if args.study_id else csv_path.stem
    study_dir = paths.study_dir(study_id)

    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_required_columns(df)
    df = _parse_and_filter(df)

    if len(df) == 0:
        raise ValueError("No valid rows remaining after filtering timestamps/lat/lon.")

    df = _ensure_set_split(df)
    df = _sort_and_assign_point_id(df)

    n_train = int((df[SET_COL] == "train").sum())
    n_test = int((df[SET_COL] == "test").sum())
    if n_train == 0:
        raise ValueError("No train rows found after split assignment; cannot proceed.")
    if n_test == 0:
        print("WARNING: no test rows found after split assignment (all points are train).")

    n_points = int(len(df))
    n_individuals = int(df[INDIVIDUAL_COL].nunique())

    print(f"csv_path:   {csv_path}")
    print(f"study_id:   {study_id}")
    print(f"study_dir:  {study_dir}")
    print(f"n_points:   {n_points}")
    print(f"individuals:{n_individuals}")
    print(f"split:      train={n_train} test={n_test}")

    if args.dry_run:
        return 0

    if study_dir.exists():
        if not args.force:
            raise FileExistsError(f"Study directory already exists: {study_dir} (use --force to overwrite)")
        shutil.rmtree(study_dir)

    study_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(study_dir / "trajectories.csv", index=False)
    _write_manifest(study_dir=study_dir, study_id=study_id, source_csv=str(csv_path), df=df)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

