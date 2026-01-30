"""Ingest CSVs from a source directory into studies/ as trajectories.csv + manifest.json."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["timestamp", "location-lat", "location-long", "individual-local-identifier"]


def read_ok_files(src_dir: Path) -> list[str]:
    ok_path = src_dir / "ok_files.txt"
    if not ok_path.exists():
        raise FileNotFoundError(f"No ok_files.txt in {src_dir}")
    lines = ok_path.read_text().strip().splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def read_bad_individuals(src_dir: Path) -> set[tuple[str, str]]:
    path = src_dir / "bad_individuals.txt"
    if not path.exists():
        return set()
    bad = set()
    for ln in path.read_text().strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",", 1)
        if len(parts) == 2:
            bad.add((parts[0].strip(), parts[1].strip()))
    return bad


def read_split_overrides(src_dir: Path) -> dict[tuple[str, str], pd.Timestamp]:
    path = src_dir / "split_overrides.txt"
    if not path.exists():
        return {}
    overrides = {}
    for ln in path.read_text().strip().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",", 2)
        if len(parts) != 3:
            continue
        fname, indiv, date_str = parts
        try:
            ts = pd.to_datetime(date_str.strip())
        except Exception:
            continue
        overrides[(fname.strip(), indiv.strip())] = ts
    return overrides


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace underscores with hyphens in column names (Movebank convention)."""
    return df.rename(columns=lambda c: c.replace("_", "-"))


def compute_split(
    df: pd.DataFrame,
    csv_basename: str,
    split_overrides: dict[tuple[str, str], pd.Timestamp],
) -> pd.DataFrame:
    indiv_col = "individual-local-identifier"

    # Per-individual time spans
    spans = df.groupby(indiv_col)["timestamp"].agg(start="min", end="max")
    spans["default_mid"] = spans["start"] + (spans["end"] - spans["start"]) / 2

    split_times = spans["default_mid"].copy()

    # Apply overrides
    for indiv in split_times.index:
        key = (csv_basename, str(indiv))
        if key in split_overrides:
            ts = split_overrides[key]
            start = spans.loc[indiv, "start"]
            end = spans.loc[indiv, "end"]
            split_times.loc[indiv] = min(max(ts, start), end)

    df = df.copy()
    df["split_time"] = df[indiv_col].map(split_times)
    df["set"] = np.where(df["timestamp"] <= df["split_time"], "train", "test")
    return df


def ingest_csv(
    csv_path: Path,
    studies_dir: Path,
    bad_individuals: set[tuple[str, str]],
    split_overrides: dict[tuple[str, str], pd.Timestamp],
    force: bool = False,
    dry_run: bool = False,
) -> bool:
    csv_basename = csv_path.name
    study_name = csv_path.stem
    study_dir = studies_dir / study_name

    if study_dir.exists() and not force:
        print(f"  SKIP {csv_basename} (already exists, use --force to overwrite)")
        return False

    # Read and parse
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_columns(df)

    # Fill individual-local-identifier from individual-id if all missing
    indiv_col = "individual-local-identifier"
    if indiv_col in df.columns and df[indiv_col].isna().all() and "individual-id" in df.columns:
        df[indiv_col] = df["individual-id"]
        print(f"  Filled {indiv_col} from individual-id")
    elif indiv_col not in df.columns and "individual-id" in df.columns:
        df[indiv_col] = df["individual-id"]
        print(f"  Using individual-id as {indiv_col}")

    # Validate required columns
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  ERROR {csv_basename}: missing columns {missing}")
        return False

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["timestamp"])
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with unparseable timestamps")

    # Remove bad individuals
    indiv_col = "individual-local-identifier"
    bad_for_csv = {indiv for (fname, indiv) in bad_individuals if fname == csv_basename}
    if bad_for_csv:
        before = len(df)
        df = df[~df[indiv_col].astype(str).isin(bad_for_csv)]
        print(f"  Removed {before - len(df)} rows from {len(bad_for_csv)} bad individuals")

    if len(df) == 0:
        print(f"  ERROR {csv_basename}: no rows remaining after filtering")
        return False

    # Compute split
    df = compute_split(df, csv_basename, split_overrides)

    # Sort and assign point_id
    df = df.sort_values([indiv_col, "timestamp"]).reset_index(drop=True)
    df["point_id"] = df.index

    n_individuals = df[indiv_col].nunique()
    train_count = (df["set"] == "train").sum()
    test_count = (df["set"] == "test").sum()

    if dry_run:
        print(f"  DRY RUN {csv_basename}: {len(df)} points, {n_individuals} individuals, "
              f"train={train_count}, test={test_count} → {study_dir}")
        return True

    # Write outputs
    study_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(study_dir / "trajectories.csv", index=False)

    manifest = {
        "study_id": study_name,
        "version": "1.0.0",
        "trajectories": {
            "file": "trajectories.csv",
            "n_points": len(df),
            "n_individuals": n_individuals,
        },
        "splits": {
            "train": {"n_points": int(train_count)},
            "test": {"n_points": int(test_count)},
        },
    }
    with open(study_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"  OK {csv_basename}: {len(df)} points, {n_individuals} individuals, "
          f"train={train_count}, test={test_count}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest CSVs into studies/")
    parser.add_argument("src_dir", type=Path, help="Source directory containing CSVs and ok_files.txt")
    parser.add_argument("--force", action="store_true", help="Overwrite existing study directories")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--studies-dir", type=Path, default=None,
                        help="Output studies directory (default: studies/ in repo root)")
    args = parser.parse_args()

    src_dir = args.src_dir.resolve()
    if not src_dir.is_dir():
        parser.error(f"Not a directory: {src_dir}")

    studies_dir = args.studies_dir or Path(__file__).resolve().parent.parent / "studies"

    ok_files = read_ok_files(src_dir)
    bad_individuals = read_bad_individuals(src_dir)
    split_overrides = read_split_overrides(src_dir)

    print(f"Source: {src_dir}")
    print(f"Output: {studies_dir}")
    print(f"Files to process: {len(ok_files)}")
    if bad_individuals:
        print(f"Bad individuals: {len(bad_individuals)} entries")
    if split_overrides:
        print(f"Split overrides: {len(split_overrides)} entries")
    print()

    success = 0
    for filename in ok_files:
        csv_path = src_dir / filename
        if not csv_path.exists():
            print(f"  MISSING {filename}")
            continue
        if ingest_csv(csv_path, studies_dir, bad_individuals, split_overrides,
                      force=args.force, dry_run=args.dry_run):
            success += 1

    print(f"\nDone: {success}/{len(ok_files)} studies ingested")


if __name__ == "__main__":
    main()
